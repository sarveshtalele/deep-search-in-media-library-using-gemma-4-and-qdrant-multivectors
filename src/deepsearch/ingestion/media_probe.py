"""ffprobe-based media inspection and type detection."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
TEXT_EXTS = {".srt", ".vtt", ".txt", ".md", ".json"}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def media_kind(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in TEXT_EXTS:
        return "text"
    return "unknown"


def probe_duration(path: str | Path) -> float:
    """Return media duration in seconds (0.0 if unknown)."""
    if not ffmpeg_available():
        return 0.0
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_format", "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def has_audio_stream(path: str | Path) -> bool:
    if not ffmpeg_available():
        return False
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        return bool(json.loads(out.stdout).get("streams"))
    except Exception:
        return False
