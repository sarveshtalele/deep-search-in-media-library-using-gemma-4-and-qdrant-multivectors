"""End-to-end offline ingestion orchestration.

For each media file:
    video  -> keyframes (described) + audio chunks (transcribed) + captions
    audio  -> audio chunks (transcribed)
    text   -> timed/untimed segments

Every fragment's text is embedded into the unified retrieval space and upserted
into Qdrant with second-accurate offsets. Embedding happens in batches to keep
GPU/unified-memory pressure bounded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from deepsearch.config import PROJECT_ROOT, Settings, get_settings
from deepsearch.ingestion import audio as audio_mod
from deepsearch.ingestion import media_probe
from deepsearch.ingestion import text as text_mod
from deepsearch.ingestion import video as video_mod
from deepsearch.logging_utils import get_logger
from deepsearch.models import get_model_client, sanitize_generated
from deepsearch.vectorstore import Fragment, Modality
from deepsearch.vectorstore.store import MediaVectorStore, get_store

log = get_logger(__name__)

# A progress event is a small dict: {stage, message, current?, total?}.
ProgressFn = Callable[[dict], None]


def _noop(_event: dict) -> None:
    pass


@dataclass
class IngestResult:
    asset_id: str
    asset_name: str
    fragments: int = 0
    by_modality: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    error: str | None = None


def _asset_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]


def _batched(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        store: MediaVectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store()
        self.model = get_model_client()
        self.cache_root = (PROJECT_ROOT / "data" / "cache").resolve()

    # ------------------------------------------------------------------
    def ingest_file(
        self,
        media_path: str | Path,
        category: str = "uncategorized",
        on_progress: ProgressFn | None = None,
    ) -> IngestResult:
        progress = on_progress or _noop
        path = Path(media_path).resolve()
        if not path.exists():
            return IngestResult("", str(media_path), error="File not found")

        self.store.ensure_collection()
        asset_id = _asset_id(path)
        # Re-ingesting the same file replaces its points instead of duplicating them.
        self.store.delete_asset(asset_id)
        asset_name = path.stem
        cache_dir = self.cache_root / asset_id
        kind = media_probe.media_kind(path)
        result = IngestResult(asset_id=asset_id, asset_name=asset_name)

        if kind == "unknown":
            return IngestResult(asset_id, asset_name, error=f"Unsupported file type: {path.suffix}")

        duration = media_probe.probe_duration(path) if kind in {"video", "audio"} else 0.0
        progress(
            {
                "stage": "probe",
                "message": f"Detected {kind}"
                + (f" · {duration:.1f}s" if duration else "")
                + (" · has audio" if kind == "video" and media_probe.has_audio_stream(path) else ""),
            }
        )
        ctx = dict(
            asset_id=asset_id,
            asset_name=asset_name,
            category=category,
            cache_dir=cache_dir,
            duration=duration,
            progress=progress,
        )
        fragments: list[Fragment] = []
        try:
            if kind == "video":
                fragments += self._video_fragments(path, **ctx)
                fragments += self._audio_fragments(path, **ctx)
                fragments += self._caption_fragments(path, **ctx)
            elif kind == "audio":
                fragments += self._audio_fragments(path, **ctx)
            elif kind == "image":
                fragments += self._image_fragments(path, **ctx)
            elif kind == "text":
                fragments += self._text_fragments(path, **ctx)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Ingestion failed")
            progress({"stage": "error", "message": str(exc)})
            return IngestResult(asset_id, asset_name, error=str(exc))

        # Drop fragments whose description/transcript came back empty — embedding an
        # empty string yields a zero-length vector that breaks the Qdrant upsert.
        fragments = [f for f in fragments if (f.text or "").strip()]

        if not fragments:
            result.error = "No fragments produced (check FFmpeg / codecs)."
            progress({"stage": "error", "message": result.error})
            return result

        self._embed_and_upsert(fragments, progress)
        result.fragments = len(fragments)
        for f in fragments:
            result.by_modality[f.modality.value] = result.by_modality.get(f.modality.value, 0) + 1
        log.info(f"Ingested '{asset_name}': {result.fragments} fragments {result.by_modality}")
        progress(
            {
                "stage": "done",
                "message": f"Indexed {result.fragments} fragments {result.by_modality}",
            }
        )
        return result

    def ingest_directory(
        self,
        directory: str | Path,
        category: str = "uncategorized",
        on_progress: ProgressFn | None = None,
    ) -> list[IngestResult]:
        directory = Path(directory)
        results = []
        for p in sorted(directory.rglob("*")):
            if p.is_file() and media_probe.media_kind(p) in {"video", "audio", "image", "text"}:
                # Skip caption sidecars; they are folded into their parent video.
                if p.suffix.lower() in {".srt", ".vtt"} and _has_media_sibling(p):
                    continue
                results.append(self.ingest_file(p, category, on_progress))
        return results

    # ------------------------------------------------------------------
    # Fragment builders
    # ------------------------------------------------------------------
    def _video_fragments(
        self, path, asset_id, asset_name, category, cache_dir, duration, progress
    ) -> list[Fragment]:
        progress({"stage": "keyframes", "message": "Detecting scenes & extracting keyframes…"})
        frames = video_mod.extract_keyframes(path, cache_dir, duration)
        total = len(frames)
        progress({"stage": "keyframes", "message": f"{total} keyframes extracted"})
        out = []
        for i, kf in enumerate(frames, 1):
            progress(
                {
                    "stage": "describe_frame",
                    "current": i,
                    "total": total,
                    "message": f"Gemma 4 describing keyframe {i}/{total} @ {kf.timestamp_s:.1f}s",
                }
            )
            desc = sanitize_generated(self.model.describe_image(kf.image_path))
            out.append(
                Fragment(
                    asset_id=asset_id,
                    asset_name=asset_name,
                    file_path=str(path),
                    modality=Modality.VIDEO,
                    text=desc,
                    start_s=kf.timestamp_s,
                    end_s=kf.end_s,
                    duration_s=duration,
                    category=category,
                    thumbnail_path=kf.image_path,
                )
            )
        return out

    def _image_fragments(
        self, path, asset_id, asset_name, category, cache_dir, duration, progress
    ) -> list[Fragment]:
        progress(
            {"stage": "describe_frame", "current": 1, "total": 1, "message": "Gemma 4 describing image…"}
        )
        desc = sanitize_generated(self.model.describe_image(path))
        return [
            Fragment(
                asset_id=asset_id,
                asset_name=asset_name,
                file_path=str(path),
                modality=Modality.VIDEO,
                text=desc,
                start_s=0.0,
                end_s=0.0,
                duration_s=0.0,
                category=category,
                thumbnail_path=str(path),
            )
        ]

    def _audio_fragments(
        self, path, asset_id, asset_name, category, cache_dir, duration, progress
    ) -> list[Fragment]:
        if not media_probe.has_audio_stream(path):
            return []
        progress({"stage": "audio", "message": "Extracting & slicing audio (30s/5s overlap)…"})
        chunks = audio_mod.extract_audio_chunks(path, cache_dir, duration)
        total = len(chunks)
        progress({"stage": "audio", "message": f"{total} audio chunks"})
        out = []
        for i, ch in enumerate(chunks, 1):
            progress(
                {
                    "stage": "transcribe",
                    "current": i,
                    "total": total,
                    "message": f"Gemma 4 transcribing audio chunk {i}/{total} @ {ch.start_s:.0f}s",
                }
            )
            transcript = sanitize_generated(self.model.describe_audio(ch.audio_path))
            out.append(
                Fragment(
                    asset_id=asset_id,
                    asset_name=asset_name,
                    file_path=str(path),
                    modality=Modality.AUDIO,
                    text=transcript,
                    start_s=ch.start_s,
                    end_s=ch.end_s,
                    duration_s=duration,
                    category=category,
                )
            )
        return out

    def _caption_fragments(
        self, path, asset_id, asset_name, category, cache_dir, duration, progress
    ) -> list[Fragment]:
        out = []
        caps = text_mod.find_sidecar_captions(path)
        if caps:
            progress({"stage": "captions", "message": "Parsing sidecar captions…"})
        for cap in caps:
            for seg in text_mod.parse_text_file(cap):
                out.append(
                    Fragment(
                        asset_id=asset_id,
                        asset_name=asset_name,
                        file_path=str(path),
                        modality=Modality.TEXT,
                        text=seg.text,
                        start_s=seg.start_s,
                        end_s=seg.end_s,
                        duration_s=duration,
                        category=category,
                    )
                )
        return out

    def _text_fragments(
        self, path, asset_id, asset_name, category, cache_dir, duration, progress
    ) -> list[Fragment]:
        progress({"stage": "captions", "message": "Parsing text segments…"})
        out = []
        for seg in text_mod.parse_text_file(path):
            out.append(
                Fragment(
                    asset_id=asset_id,
                    asset_name=asset_name,
                    file_path=str(path),
                    modality=Modality.TEXT,
                    text=seg.text,
                    start_s=seg.start_s,
                    end_s=seg.end_s,
                    duration_s=duration,
                    category=category,
                )
            )
        return out

    # ------------------------------------------------------------------
    def _embed_and_upsert(self, fragments: list[Fragment], progress: ProgressFn = _noop) -> None:
        """Embed fragments, upsert fragment points, then build the asset-level
        MaxSim multivector point from the per-modality vector arrays."""
        frame_vecs: list[list[float]] = []
        audio_vecs: list[list[float]] = []
        asset_payload: dict | None = None

        batch = self.settings.ingestion.batch_size
        groups = list(_batched(fragments, batch))
        done = 0
        for gi, group in enumerate(groups, 1):
            progress(
                {
                    "stage": "embed",
                    "current": gi,
                    "total": len(groups),
                    "message": f"Embedding & indexing batch {gi}/{len(groups)} "
                    f"({done}/{len(fragments)} fragments)",
                }
            )
            vectors = self.model.embed_text([f.text for f in group])
            self.store.upsert_fragments(group, vectors)
            done += len(group)
            for frag, vec in zip(group, vectors, strict=True):
                if asset_payload is None:
                    asset_payload = {
                        "asset_id": frag.asset_id,
                        "asset_name": frag.asset_name,
                        "file_path": frag.file_path,
                        "category": frag.category,
                        "duration_s": frag.duration_s,
                        "kind": "asset",
                    }
                if frag.modality is Modality.VIDEO:
                    frame_vecs.append(vec)
                elif frag.modality is Modality.AUDIO:
                    audio_vecs.append(vec)

        if asset_payload and (frame_vecs or audio_vecs):
            progress(
                {
                    "stage": "multivector",
                    "message": f"Building MaxSim asset point ({len(frame_vecs)} frame + "
                    f"{len(audio_vecs)} audio vectors)",
                }
            )
            self.store.upsert_asset_multivectors(
                asset_payload["asset_id"], asset_payload, frame_vecs, audio_vecs
            )


def _has_media_sibling(caption_path: Path) -> bool:
    for ext in media_probe.VIDEO_EXTS | media_probe.AUDIO_EXTS:
        if caption_path.with_suffix(ext).exists():
            return True
    return False
