"""Audio extraction and overlapping chunk slicing via FFmpeg.

Continuous audio is sliced into fixed windows (default 30 s) with an overlap
(default 5 s) so a phrase spanning a boundary is never lost. Audio is
normalised to 16 kHz mono WAV — the canonical input for Gemma 4's audio encoder.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from deepsearch.config import get_settings
from deepsearch.ingestion.media_probe import ffmpeg_available, probe_duration
from deepsearch.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class AudioChunk:
    start_s: float
    end_s: float
    audio_path: str


def _extract_wav(src: str, dst: Path, denoise: bool) -> bool:
    # Normalise to 16 kHz mono (stereo -> mono downmix, resample) for Gemma 4's
    # audio encoder; optionally apply afftdn adaptive noise reduction.
    cmd = ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000"]
    if denoise:
        cmd += ["-af", "afftdn=nf=-25"]
    cmd += ["-f", "wav", str(dst)]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as exc:  # pragma: no cover
        log.error(f"FFmpeg audio extraction failed for {src}: {exc!s}")
        return False


def _slice(src_wav: str, start: float, dur: float, dst: Path) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                "-i", src_wav, "-ac", "1", "-ar", "16000", str(dst),
            ],
            capture_output=True, check=True,
        )
        return True
    except Exception:  # pragma: no cover
        return False


def extract_full_wav(media_path: str | Path, cache_dir: str | Path) -> str | None:
    """Extract the full track to a 16 kHz mono WAV (for Whisper). Returns path or None."""
    if not ffmpeg_available():
        return None
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{Path(media_path).stem}_audio.wav"
    return str(out) if _extract_wav(str(media_path), out, get_settings().ingestion.audio_denoise) else None


def extract_audio_chunks(
    media_path: str | Path,
    cache_dir: str | Path,
    duration_s: float = 0.0,
) -> list[AudioChunk]:
    if not ffmpeg_available():
        log.warning("FFmpeg unavailable; skipping audio ingestion.")
        return []

    cfg = get_settings().ingestion
    media_path = str(media_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(media_path).stem

    full_wav = cache_dir / f"{stem}_audio.wav"
    if not _extract_wav(media_path, full_wav, cfg.audio_denoise):
        return []

    duration = duration_s or probe_duration(str(full_wav))
    if duration <= 0:
        return []

    window = cfg.audio_chunk_seconds
    overlap = cfg.audio_overlap_seconds
    hop = max(1, window - overlap)
    # On very long media, grow the window so the chunk count (each = one Gemma
    # transcription call) stays bounded instead of scaling linearly with length.
    est = math.ceil(duration / hop)
    if est > cfg.max_audio_chunks:
        hop = math.ceil(duration / cfg.max_audio_chunks)
        window = hop + overlap
        log.info(f"Long media ({duration:.0f}s): widening audio window to {window}s "
                 f"to cap at ~{cfg.max_audio_chunks} chunks.")
    chunks: list[AudioChunk] = []
    start = 0.0
    idx = 0
    while start < duration:
        dur = min(window, duration - start)
        out = cache_dir / f"{stem}_chunk{idx:03d}.wav"
        if _slice(str(full_wav), start, dur, out):
            chunks.append(AudioChunk(start_s=start, end_s=start + dur, audio_path=str(out)))
        idx += 1
        start += hop
    log.info(f"Sliced {len(chunks)} audio chunks from {Path(media_path).name}")
    return chunks
