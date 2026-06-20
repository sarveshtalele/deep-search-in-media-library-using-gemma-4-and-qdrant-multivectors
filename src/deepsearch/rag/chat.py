"""Retrieval-grounded conversational interface (library-wide RAG).

This is the primary way to use the system: the user asks a natural-language
question and gets a grounded answer, with the supporting media moments returned
as sources (timestamps are secondary, surfaced on demand by the UI).

Design points that fix real failures:
  * **Answers, not just hits** — retrieve relevant fragments, then have Gemma 4
    answer the question from them.
  * **Counting / "how many times X"** — we feed the *entire* time-ordered
    timeline of the most relevant asset (not just top-k snippets) into a large
    context window, so the model can actually count every occurrence.
  * **No-audio awareness** — the library overview tells the model which assets
    have an audio track, and the prompt forces it to say so when asked about
    spoken content that doesn't exist.
  * **Anti-hallucination** — strict "answer only from context" system prompt,
    low temperature, and an explicit "say you cannot find it" rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from deepsearch.logging_utils import get_logger
from deepsearch.models import check_query, get_model_client
from deepsearch.vectorstore.store import MediaVectorStore, get_store

log = get_logger(__name__)

_SYSTEM = (
    "You are a retrieval-grounded assistant for a user's LOCAL media library. "
    "You answer questions about their indexed media using ONLY the CONTEXT provided "
    "(timestamped descriptions and transcripts of their video, audio and images).\n"
    "RULES:\n"
    "1. Use ONLY facts present in the CONTEXT. Never invent objects, names, numbers, "
    "brands or events that are not there.\n"
    "2. Treat the words 'image', 'images', 'screen', 'screens', 'footage', 'clip' and "
    "'frame' as the visual content of the user's media — VIDEO frames and screenshots "
    "all count. When asked to DESCRIBE, SUMMARISE, or list WHAT/WHICH media they "
    "uploaded, describe what IS in the CONTEXT (state the real media types).\n"
    "3. Only reply that you can't find something when a SPECIFIC requested detail is "
    "genuinely absent from the CONTEXT — never refuse just because the wording differs.\n"
    "4. When asked HOW MANY / to COUNT, count only explicit mentions in the CONTEXT, "
    "state the number, and list the timestamps.\n"
    "5. If asked about spoken words / audio and the asset is marked (no audio), say it "
    "has no audio track to analyse.\n"
    "6. COUNTING PEOPLE/OBJECTS IN A VIDEO: the SAME people/objects usually reappear "
    "across many timestamps — do NOT add up per-frame counts. Report the number of "
    "DISTINCT entities, using the LARGEST single-frame count as your basis, and say it "
    "is approximate (e.g. 'about 3 people'). Never sum the frames.\n"
    "7. Format answers in clean Markdown: a short lead sentence, then bullet points with "
    "**bold** labels and [HH:MM:SS] timestamps. Be concise and do not pad."
)

_AUDIO_HINT = re.compile(
    r"\b(said|say|says|speak|spoke|spoken|mention|mentioned|hear|heard|audio|"
    r"voice|talk|told|count|how many times)\b",
    re.IGNORECASE,
)

_COUNT_HINT = re.compile(r"\bhow many times\b|\bcount\b|\bnumber of times\b", re.IGNORECASE)

# Library/meta questions ("how many files", "what types", "list my media") are
# answered deterministically from the asset list — never the LLM, and never with
# per-fragment sources (no single fragment "answers" a library-wide question).
_LIBRARY_HINT = re.compile(
    r"\bhow many (files|assets|videos?|audios?|images?|media|types|kinds)\b"
    r"|\bwhat (types?|kinds?) of (file|media)"
    r"|\blist (my|the|all)?\s*(files|media|videos|assets|library)"
    r"|\bwhat('?s| is| do i have)?\b.*\bin my library\b"
    r"|\bwhich (assets|files|videos|media)\b",
    re.IGNORECASE,
)


def _extract_count_term(q: str) -> str | None:
    """Pull the term to count out of a 'how many times … X … mentioned' question."""
    m = re.search(r"[\"'“‘]([^\"'”’]+)[\"'”’]", q)
    if m:
        return m.group(1).strip()
    m = re.search(
        r"times\s+(?:is|are|does|do|did|was|were|the|that|word)?\s*([A-Za-z][\w\-]+)"
        r"\s+(?:mentioned|said|appears?|appeared|spoken|used|come|comes|shown)",
        q, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(r"count\s+(?:the\s+)?(?:word\s+|number of\s+)?([A-Za-z][\w\-]+)", q, re.IGNORECASE)
    return m.group(1).strip() if m else None


@dataclass
class Source:
    asset_id: str
    asset_name: str
    file_path: str
    modality: str
    start_s: float
    end_s: float

    @property
    def timestamp_label(self) -> str:
        m, s = divmod(int(self.start_s), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


def _ts(start_s: float) -> str:
    m, s = divmod(int(start_s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class LibraryRAG:
    def __init__(self, store: MediaVectorStore | None = None) -> None:
        self.store = store or get_store()
        self.model = get_model_client()

    # ------------------------------------------------------------------
    def _overview(self, assets: list[dict]) -> str:
        if not assets:
            return "INDEXED MEDIA: (none yet)"
        lines = ["INDEXED MEDIA:"]
        for a in assets:
            audio = "has audio" if a.get("has_audio") else "no audio"
            mods = ", ".join(m.split("_")[0] for m in a.get("modalities", []))
            lines.append(f"- {a['asset_name']} ({mods}; {audio})")
        return "\n".join(lines)

    def _timeline(self, asset_id: str, max_chars: int = 16000) -> str:
        frags = self.store.asset_fragments(asset_id)
        frags = sorted(frags, key=lambda p: float(p.get("start_s", 0.0)))
        rows = []
        for p in frags:
            text = (p.get("text") or "").strip()
            if not text:
                continue
            tag = {"video_frames": "VIDEO", "audio_chunks": "AUDIO",
                   "text_descriptions": "TEXT"}.get(p.get("modality", ""), "?")
            rows.append(f"[{_ts(float(p.get('start_s', 0.0)))}] ({tag}) {text}")
        return "\n".join(rows)[:max_chars]

    # ------------------------------------------------------------------
    def _count(self, term: str) -> dict:
        """Deterministic, exact occurrence count across all indexed text — no LLM,
        so 'how many times X is mentioned' is always accurate."""
        needle = term.lower()
        frags = self.store.all_fragments()
        total = 0
        sources: list[Source] = []
        per_asset: dict[str, int] = {}
        for p in frags:
            text = (p.get("text") or "").lower()
            n = text.count(needle)
            if n:
                total += n
                name = p.get("asset_name", "?")
                per_asset[name] = per_asset.get(name, 0) + n
                sources.append(
                    Source(p.get("asset_id", ""), name, p.get("file_path", ""),
                           p.get("modality", ""), float(p.get("start_s", 0.0)),
                           float(p.get("end_s", 0.0)))
                )
        sources.sort(key=lambda s: (s.asset_name, s.start_s))
        if total == 0:
            answer = f"“{term}” is not mentioned anywhere in your indexed media."
        else:
            breakdown = ", ".join(f"{k}: {v}" for k, v in per_asset.items())
            spots = ", ".join(f"{s.asset_name} @ {s.timestamp_label}" for s in sources[:12])
            answer = (
                f"“{term}” appears {total} time{'s' if total != 1 else ''} across your "
                f"indexed media ({breakdown}).\nMoments: {spots}."
            )
        return {"answer": answer, "sources": sources[:12]}

    # ------------------------------------------------------------------
    def _prepare(self, question: str, history: list[dict] | None) -> dict:
        """Resolve a question to either a deterministic final text, or the LLM
        messages to generate. Returns {kind, text?, messages?, sources}."""
        verdict = check_query(question)
        if verdict.blocked:
            return {"kind": "final", "text": f"Query blocked: {verdict.reason}", "sources": []}

        # Counting is answered deterministically (exact match), never by the LLM.
        if _COUNT_HINT.search(question):
            term = _extract_count_term(question)
            if term:
                res = self._count(term)
                return {"kind": "final", "text": res["answer"], "sources": res["sources"]}

        from deepsearch.query import get_engine  # local import avoids cycle

        assets = self.store.list_assets()
        if not assets:
            return {
                "kind": "final",
                "text": "Your library is empty — attach a file in chat to index it first.",
                "sources": [],
            }

        # Library composition questions -> exact answer from the asset list, no sources.
        if _LIBRARY_HINT.search(question):
            return {"kind": "final", "text": self._library_summary(assets), "sources": []}

        hits = get_engine().search(question).hits
        dominant = hits[0].asset_id if hits else assets[0]["asset_id"]
        dominant_name = next(
            (a["asset_name"] for a in assets if a["asset_id"] == dominant), dominant
        )

        parts = [self._overview(assets)]
        timeline = self._timeline(dominant)
        if timeline:
            parts.append(f"FULL TIMELINE of '{dominant_name}':\n{timeline}")
        others = [
            f"[{h.timestamp_label}] ({h.modality.split('_')[0]}) {h.asset_name}: {h.text[:300]}"
            for h in hits if h.asset_id != dominant
        ][:6]
        if others:
            parts.append("OTHER RELEVANT MOMENTS:\n" + "\n".join(others))
        if _AUDIO_HINT.search(question):
            da = next((a for a in assets if a["asset_id"] == dominant), {})
            if not da.get("has_audio"):
                parts.append(f"NOTE: '{dominant_name}' has NO audio track.")

        context = "\n\n".join(parts)[:28000]
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "system", "content": f"CONTEXT:\n{context}"},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        return {
            "kind": "generate", "messages": messages,
            "sources": self._select_sources(hits, dominant),
        }

    def _library_summary(self, assets: list[dict]) -> str:
        """Exact library composition by media type (each asset counted once)."""
        order = ["video (with audio)", "video", "audio", "image", "text", "other"]
        groups: dict[str, list[str]] = {}
        for a in assets:
            mods = set(a.get("modalities", []))
            if "video_frames" in mods and "audio_chunks" in mods:
                kind = "video (with audio)"
            elif "video_frames" in mods:
                kind = "video"
            elif "audio_chunks" in mods:
                kind = "audio"
            elif "text_descriptions" in mods:
                kind = "text"
            else:
                kind = "other"
            groups.setdefault(kind, []).append(a.get("asset_name", "?"))
        present = [k for k in order if k in groups]
        lines = [
            f"You have **{len(assets)}** asset{'s' if len(assets) != 1 else ''} "
            f"across **{len(present)}** media type{'s' if len(present) != 1 else ''}:"
        ]
        for k in present:
            names = groups[k]
            lines.append(f"- **{k}**: {len(names)} ({', '.join(names)})")
        return "\n".join(lines)

    def _select_sources(self, hits, dominant: str, limit: int = 5) -> list[Source]:
        """Surface only the moments the answer is grounded in. The answer is built
        from the dominant asset's timeline, so sources come from THAT asset — never
        unrelated low-score fragments from other assets the user didn't ask about."""
        out: list[Source] = []
        for h in hits:
            if h.asset_id != dominant:
                continue
            out.append(Source(h.asset_id, h.asset_name, h.file_path, h.modality, h.start_s, h.end_s))
            if len(out) >= limit:
                break
        return out

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        prep = self._prepare(question, history)
        if prep["kind"] == "final":
            return {"answer": prep["text"], "sources": prep["sources"]}
        return {"answer": self.model.chat(prep["messages"]), "sources": prep["sources"]}

    def stream(self, question: str, history: list[dict] | None = None):
        """Yield ('token', str) chunks, then a final ('sources', list[Source])."""
        prep = self._prepare(question, history)
        if prep["kind"] == "final":
            yield ("token", prep["text"])
        else:
            for chunk in self.model.chat_stream(prep["messages"]):
                yield ("token", chunk)
        yield ("sources", prep["sources"])


_RAG: LibraryRAG | None = None


def get_rag() -> LibraryRAG:
    global _RAG
    if _RAG is None:
        _RAG = LibraryRAG()
    return _RAG
