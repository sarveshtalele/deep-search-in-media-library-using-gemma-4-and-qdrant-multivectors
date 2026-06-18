"""Fast audio transcription via faster-whisper (CTranslate2).

Replaces the per-30s-chunk Gemma audio calls (the dominant cost on long media)
with a single Whisper pass. Whisper segments already carry accurate start/end
timestamps, which we merge into ~window-sized fragments for retrieval/RAG.

Falls back gracefully: callers check ``available()`` and use the Gemma audio
path if faster-whisper is not installed.
"""

from __future__ import annotations

from deepsearch.ingestion.text import TextSegment
from deepsearch.logging_utils import get_logger

log = get_logger(__name__)

_MODEL = None
_MODEL_NAME = None


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        return False


def _get_model(name: str):
    global _MODEL, _MODEL_NAME
    if _MODEL is None or _MODEL_NAME != name:
        from faster_whisper import WhisperModel

        log.info(f"Loading faster-whisper model '{name}' (cpu/int8)…")
        _MODEL = WhisperModel(name, device="cpu", compute_type="int8")
        _MODEL_NAME = name
    return _MODEL


def transcribe(
    wav_path: str,
    model_name: str = "base",
    window_s: int = 30,
    on_progress=None,
) -> list[TextSegment]:
    """Transcribe a WAV file into ~``window_s`` second timestamped segments.

    ``on_progress(done_s, total_s)`` is called as Whisper streams segments.
    """
    model = _get_model(model_name)
    segments, info = model.transcribe(wav_path, beam_size=1)
    total = float(getattr(info, "duration", 0.0)) or 0.0

    out: list[TextSegment] = []
    buf: list[str] = []
    start: float | None = None
    last = 0.0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        if start is None:
            start = float(seg.start)
        buf.append(text)
        last = float(seg.end)
        if on_progress:
            on_progress(last, total)
        if last - (start or 0.0) >= window_s:
            out.append(TextSegment(start or 0.0, last, " ".join(buf)))
            buf, start = [], None
    if buf:
        out.append(TextSegment(start or 0.0, last, " ".join(buf)))
    log.info(f"Whisper produced {len(out)} transcript segments from {total:.0f}s audio")
    return out
