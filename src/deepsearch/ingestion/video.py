"""Keyframe extraction via scene-boundary detection.

We avoid rigid fixed-interval sampling. PySceneDetect's content detector finds
true scene cuts; we grab the representative middle frame of each scene. If
PySceneDetect is unavailable we fall back to uniform sampling so ingestion never
hard-fails. Each keyframe is written as a JPEG thumbnail (used both as the
Gemma 4 vision input and as the UI preview) and tagged with its timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from deepsearch.config import get_settings
from deepsearch.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Keyframe:
    timestamp_s: float
    end_s: float
    image_path: str


def _scene_boundaries(path: str, threshold: float) -> list[tuple[float, float]]:
    """Return [(start_s, end_s), ...] scene spans, or [] if detection unavailable."""
    try:
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(video)
        scenes = manager.get_scene_list()
        return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    except Exception as exc:  # pragma: no cover - optional dependency / codec
        log.warning(f"Scene detection failed ({exc!s}); falling back to uniform sampling.")
        return []


def _uniform_spans(duration: float, n: int) -> list[tuple[float, float]]:
    if duration <= 0:
        return [(0.0, 0.0)]
    step = duration / n
    return [(i * step, (i + 1) * step) for i in range(n)]


def extract_keyframes(
    video_path: str | Path,
    cache_dir: str | Path,
    duration_s: float = 0.0,
) -> list[Keyframe]:
    cfg = get_settings().ingestion
    video_path = str(video_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Scene detection decodes the whole video — far too slow for long media, where
    # we only keep `max_keyframes` frames anyway. Skip it past a duration limit.
    if duration_s and duration_s > cfg.scene_detect_max_duration:
        log.info(
            f"Long video ({duration_s:.0f}s) > {cfg.scene_detect_max_duration}s: "
            f"sampling {cfg.max_keyframes} keyframes uniformly (skipping scene detection)."
        )
        spans = _uniform_spans(duration_s, cfg.max_keyframes)
    else:
        spans = _scene_boundaries(video_path, cfg.scene_threshold)
        if not spans:
            # ~1 keyframe per 5s of footage, capped at max_keyframes.
            n = max(1, min(cfg.max_keyframes, round((duration_s or 5) / 5)))
            spans = _uniform_spans(duration_s, n)

    # Cap to max_keyframes, evenly subsampling if a clip has very many scenes.
    if len(spans) > cfg.max_keyframes:
        stride = len(spans) / cfg.max_keyframes
        spans = [spans[int(i * stride)] for i in range(cfg.max_keyframes)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"OpenCV could not open {video_path}")
        return []

    keyframes: list[Keyframe] = []
    stem = Path(video_path).stem
    for idx, (start, end) in enumerate(spans):
        mid = (start + end) / 2.0 if end > start else start
        cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = _resize_longest(frame, cfg.frame_resize)
        out = cache_dir / f"{stem}_kf{idx:03d}.jpg"
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        keyframes.append(Keyframe(timestamp_s=start, end_s=end, image_path=str(out)))
    cap.release()
    log.info(f"Extracted {len(keyframes)} keyframes from {Path(video_path).name}")
    return keyframes


def _resize_longest(frame, longest: int):
    h, w = frame.shape[:2]
    scale = longest / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
