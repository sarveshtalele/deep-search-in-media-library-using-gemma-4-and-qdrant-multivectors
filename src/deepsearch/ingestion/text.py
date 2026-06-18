"""Parsing of sidecar captions and metadata.

Supports SRT and VTT subtitle files (timed) plus plain text / JSON metadata
(untimed, treated as a single asset-level description fragment).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from deepsearch.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class TextSegment:
    start_s: float
    end_s: float
    text: str


def parse_srt(path: str | Path) -> list[TextSegment]:
    import srt

    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [
        TextSegment(s.start.total_seconds(), s.end.total_seconds(), s.content.strip())
        for s in srt.parse(raw)
        if s.content.strip()
    ]


def parse_vtt(path: str | Path) -> list[TextSegment]:
    import webvtt

    segs: list[TextSegment] = []
    for cap in webvtt.read(str(path)):
        segs.append(
            TextSegment(_hms(cap.start), _hms(cap.end), cap.text.strip().replace("\n", " "))
        )
    return [s for s in segs if s.text]


def parse_plain(path: str | Path) -> list[TextSegment]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            text = json.dumps(data, ensure_ascii=False)
        except Exception:
            pass
    return [TextSegment(0.0, 0.0, text)] if text else []


def parse_text_file(path: str | Path) -> list[TextSegment]:
    ext = Path(path).suffix.lower()
    try:
        if ext == ".srt":
            return parse_srt(path)
        if ext == ".vtt":
            return parse_vtt(path)
        return parse_plain(path)
    except Exception as exc:  # pragma: no cover
        log.warning(f"Failed to parse text file {path}: {exc!s}")
        return []


def find_sidecar_captions(media_path: str | Path) -> list[Path]:
    """Locate caption files next to a media file (same stem, .srt/.vtt)."""
    media_path = Path(media_path)
    found = []
    for ext in (".srt", ".vtt"):
        cand = media_path.with_suffix(ext)
        if cand.exists():
            found.append(cand)
    return found


def _hms(ts: str) -> float:
    """Convert an HH:MM:SS.mmm WebVTT timestamp to seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)
