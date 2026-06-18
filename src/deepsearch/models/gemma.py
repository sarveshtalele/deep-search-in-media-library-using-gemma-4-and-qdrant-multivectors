"""Gemma 4 model client (via Ollama) with a deterministic offline stub.

Design notes
------------
Gemma 4 is a *generative* multimodal decoder, not a contrastive dual-encoder.
That means its image/audio token activations do **not** live in the same metric
space as text, so a text query vector cannot be compared directly against a raw
"frame vector" with any reliability. We therefore default to a
**describe-then-embed** strategy:

    keyframe / audio chunk  --(Gemma 4 vision/audio)-->  text description
    text description / caption / query  --(embedding model)-->  one unified space

All modalities collapse into a single text-embedding space, so a natural-language
query genuinely retrieves across video, audio and text. The optional ``native``
strategy keeps Gemma's own embeddings per modality for same-modal experiments.

If Ollama is unreachable (or ``runtime.use_stub`` is set) the client transparently
falls back to a hash-based deterministic stub so the full pipeline — ingestion,
indexing, search, RAG — runs offline and in CI without any model weights.
"""

from __future__ import annotations

import base64
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from deepsearch.config import Settings, get_settings
from deepsearch.logging_utils import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------------
# Prompt templates (kept here so they are easy to audit / tune)
# ----------------------------------------------------------------------------
_FRAME_PROMPT = (
    "You are indexing a video keyframe for search. Describe, in 2-3 dense "
    "sentences, the visible objects, people, actions, on-screen text, charts "
    "and setting. Be concrete and literal. Do not speculate about audio."
)
_AUDIO_PROMPT = (
    "You are indexing an audio segment for search. Transcribe any speech "
    "verbatim, then add one sentence describing tone, speaker sentiment and "
    "notable non-speech sounds. If silent, say 'silence'."
)
_INTENT_PROMPT = (
    "Classify the user's media-search intent and propose retrieval weights.\n"
    "Return STRICT JSON: {{\"intent\": one of [visual, spoken, conceptual, mixed], "
    "\"weights\": {{\"text_descriptions\": float, \"video_frames\": float, "
    "\"audio_chunks\": float}}}} with weights summing to 1.0.\n"
    "Query: {query}"
)
_RERANK_PROMPT = (
    "Rank the candidate media fragments by how well they answer the query. "
    "Return STRICT JSON: {{\"order\": [list of candidate ids, best first]}}.\n"
    "Query: {query}\n\nCandidates:\n{candidates}"
)


@dataclass
class _Backend:
    name: str  # "ollama" | "stub"


class GemmaModelClient:
    """Single entry point for every Gemma 4 / embedding interaction."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dim = self.settings.models.embed_dim
        self.strategy = self.settings.models.embed_strategy
        # In `native` mode Gemma 4 itself produces the (text) embeddings, so the
        # query and every fragment share one Gemma-defined space. In the default
        # `describe_then_embed` mode a dedicated embedder owns the space.
        self.embed_model_name = (
            self.settings.models.gemma_model
            if self.strategy == "native"
            else self.settings.models.embed_model
        )
        self.backend = self._select_backend()
        if not self.is_stub:
            self._detect_dim()
        log.info(
            f"Gemma client backend={self.backend.name} strategy={self.strategy} "
            f"embed_model={self.embed_model_name} dim={self.dim}"
        )

    def _detect_dim(self) -> None:
        """Probe the real embedding dimensionality once at startup."""
        try:
            resp = self._client.embeddings(model=self.embed_model_name, prompt="dimension probe")
            self.dim = len(resp["embedding"])
        except Exception as exc:  # pragma: no cover - env dependent
            log.warning(
                f"Could not probe embed dim for {self.embed_model_name} ({exc!s}); "
                f"using configured {self.dim}."
            )

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------
    def _select_backend(self) -> _Backend:
        self.unavailable_reason = ""
        if self.settings.runtime.use_stub:
            return _Backend("stub")
        try:
            import ollama  # noqa: F401

            self._client = ollama.Client(
                host=self.settings.models.ollama_host,
                timeout=self.settings.models.request_timeout_s,
            )
            self._client.list()  # connectivity probe
            return _Backend("ollama")
        except Exception as exc:  # pragma: no cover - depends on local env
            self.unavailable_reason = (
                f"Ollama not reachable at {self.settings.models.ollama_host} ({exc!s})."
            )
            log.warning(f"{self.unavailable_reason} Falling back to deterministic stub.")
            return _Backend("stub")

    @property
    def is_stub(self) -> bool:
        return self.backend.name == "stub"

    # ==================================================================
    # Health
    # ==================================================================
    def health(self) -> dict:
        """Live check of the Gemma 4 runtime. Always pings fresh so it reflects the
        current state (not the startup snapshot). Returns a structured report the API
        surfaces to the UI — never silently hides an unavailable backend."""
        cfg = self.settings.models
        out: dict = {
            "ok": False,
            "backend": self.backend.name,
            "stub_mode": self.settings.runtime.use_stub,
            "ollama_host": cfg.ollama_host,
            "gemma_model": cfg.gemma_model,
            "embed_model": cfg.embed_model,
            "ollama_reachable": False,
            "gemma_present": False,
            "embed_present": False,
            "models": [],
            "detail": "",
        }
        if self.settings.runtime.use_stub:
            out["ok"] = True
            out["detail"] = "Stub mode (DEEPSEARCH_USE_STUB=true) — deterministic, no real model."
            return out
        try:
            import ollama

            client = ollama.Client(host=cfg.ollama_host, timeout=5)
            models = client.list().get("models", [])
            names = [m.get("model") or m.get("name") or "" for m in models]
            out["ollama_reachable"] = True
            out["models"] = names
            out["gemma_present"] = _model_present(names, cfg.gemma_model)
            out["embed_present"] = _model_present(names, cfg.embed_model)
            out["ok"] = out["gemma_present"] and out["embed_present"]
            if not out["ok"]:
                missing = [
                    m for m, present in (
                        (cfg.gemma_model, out["gemma_present"]),
                        (cfg.embed_model, out["embed_present"]),
                    ) if not present
                ]
                out["detail"] = (
                    "Ollama is running but missing model(s): "
                    + ", ".join(missing)
                    + ". Pull with: " + " && ".join(f"ollama pull {m}" for m in missing)
                )
        except Exception as exc:
            out["detail"] = (
                f"Ollama not reachable at {cfg.ollama_host} ({exc!s}). "
                "Start it with `ollama serve` and pull the models, then retry."
            )
        return out

    # ==================================================================
    # Embeddings
    # ==================================================================
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into the unified retrieval space."""
        if not texts:
            return []
        if self.is_stub:
            return [self._stub_vector(t) for t in texts]
        out: list[list[float]] = []
        ka = self.settings.models.keep_alive
        for text in texts:
            # Never send an empty prompt — some embedders return a 0-length vector.
            prompt = text if text and text.strip() else " "
            resp = self._client.embeddings(model=self.embed_model_name, prompt=prompt, keep_alive=ka)
            vec = list(resp.get("embedding") or [])
            if len(vec) != self.dim:
                raise ValueError(
                    f"Embedding model returned {len(vec)} dims (expected {self.dim}). "
                    f"Check DEEPSEARCH_EMBED_MODEL / embed_dim."
                )
            out.append(_normalize(vec))
        return out

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text([query])[0]

    # ==================================================================
    # Multimodal description (the describe-then-embed bridge)
    # ==================================================================
    def _desc_opts(self) -> dict:
        return {"temperature": 0.2, "num_predict": self.settings.models.num_predict}

    def _gen_opts(self, temperature: float = 0.3) -> dict:
        return {"temperature": temperature, "num_predict": self.settings.models.gen_num_predict}

    def describe_image(self, image_path: str | Path) -> str:
        if self.is_stub:
            return f"[stub frame description for {Path(image_path).name}]"
        b64 = _b64_file(image_path)
        resp = self._client.generate(
            model=self.settings.models.gemma_model,
            prompt=_FRAME_PROMPT,
            images=[b64],
            options=self._desc_opts(),
            keep_alive=self.settings.models.keep_alive,
            think=False,
        )
        return resp["response"].strip()

    def describe_audio(self, audio_path: str | Path) -> str:
        """Transcribe + characterise an audio chunk using Gemma 4's audio input."""
        if self.is_stub:
            return f"[stub audio transcript for {Path(audio_path).name}]"
        b64 = _b64_file(audio_path)
        # Ollama passes non-image media through the same `images` channel for
        # Gemma 4's unified multimodal encoder.
        resp = self._client.generate(
            model=self.settings.models.gemma_model,
            prompt=_AUDIO_PROMPT,
            images=[b64],
            options=self._desc_opts(),
            keep_alive=self.settings.models.keep_alive,
            think=False,
        )
        return resp["response"].strip()

    # ==================================================================
    # Generation / reasoning
    # ==================================================================
    def generate(self, prompt: str, images: list[str] | None = None) -> str:
        if self.is_stub:
            return f"[stub generation] {prompt[:120]}"
        resp = self._client.generate(
            model=self.settings.models.gemma_model,
            prompt=prompt,
            images=images or None,
            options=self._gen_opts(),
            keep_alive=self.settings.models.keep_alive,
            think=False,
        )
        return resp["response"].strip()

    def chat(self, messages: list[dict]) -> str:
        if self.is_stub:
            last = messages[-1]["content"] if messages else ""
            return f"[stub chat reply grounded in context] re: {last[:120]}"
        m = self.settings.models
        resp = self._client.chat(
            model=m.gemma_model,
            messages=messages,
            # Larger context + answer budget + low temperature for grounded RAG.
            options={"temperature": 0.1, "num_predict": m.chat_num_predict, "num_ctx": m.num_ctx},
            keep_alive=m.keep_alive,
            think=False,
        )
        return resp["message"]["content"].strip()

    def chat_stream(self, messages: list[dict]):
        """Yield the assistant reply token-by-token (for a streaming UI)."""
        if self.is_stub:
            yield "[stub streamed reply grounded in context]"
            return
        m = self.settings.models
        for part in self._client.chat(
            model=m.gemma_model,
            messages=messages,
            options={"temperature": 0.1, "num_predict": m.chat_num_predict, "num_ctx": m.num_ctx},
            keep_alive=m.keep_alive,
            think=False,
            stream=True,
        ):
            chunk = part.get("message", {}).get("content", "")
            if chunk:
                yield chunk

    # ==================================================================
    # Stub helpers — deterministic, normalized pseudo-embeddings
    # ==================================================================
    def _stub_vector(self, text: str) -> list[float]:
        vec: list[float] = []
        counter = 0
        seed = text.encode("utf-8")
        while len(vec) < self.dim:
            h = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for i in range(0, len(h), 4):
                if len(vec) >= self.dim:
                    break
                val = int.from_bytes(h[i : i + 4], "big") / 2**32
                vec.append(val - 0.5)
            counter += 1
        return _normalize(vec)


# ----------------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------------
def _model_present(names: list[str], model: str) -> bool:
    """Match an Ollama model id loosely (`nomic-embed-text` ≈ `nomic-embed-text:latest`)."""
    base = model.split(":")[0]
    return any(n == model or n.split(":")[0] == base for n in names)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _b64_file(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


_SINGLETON: GemmaModelClient | None = None


def get_model_client() -> GemmaModelClient:
    """Return a process-wide singleton model client."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = GemmaModelClient()
    return _SINGLETON
