"""Score fusion across the three named vector spaces.

Qdrant has no single-call "weighted sum across named vectors" knob, so fusion is
performed client-side over per-space candidate lists. Two strategies:

* **RRF** (Reciprocal Rank Fusion) — rank-based, scale-free, robust default.
* **Weighted** — normalised-score weighted sum using the (intent-derived) per
  space weights, for when the caller wants explicit control.

Both fuse at the *fragment* level (a point id), preserving temporal payloads.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from qdrant_client import models


@dataclass
class FusedHit:
    point_id: str
    score: float
    payload: dict
    contributions: dict[str, float]  # which spaces contributed and how much


def _min_max(scores: list[float]) -> tuple[float, float]:
    if not scores:
        return 0.0, 1.0
    lo, hi = min(scores), max(scores)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def rrf_fuse(
    per_space: dict[str, list[models.ScoredPoint]],
    k: int = 60,
) -> list[FusedHit]:
    acc: dict[str, float] = defaultdict(float)
    payloads: dict[str, dict] = {}
    contrib: dict[str, dict[str, float]] = defaultdict(dict)
    for space, points in per_space.items():
        for rank, p in enumerate(points):
            pid = str(p.id)
            add = 1.0 / (k + rank + 1)
            acc[pid] += add
            contrib[pid][space] = round(add, 5)
            payloads[pid] = p.payload
    fused = [FusedHit(pid, acc[pid], payloads[pid], contrib[pid]) for pid in acc]
    return sorted(fused, key=lambda h: h.score, reverse=True)


def weighted_fuse(
    per_space: dict[str, list[models.ScoredPoint]],
    weights: dict[str, float],
) -> list[FusedHit]:
    acc: dict[str, float] = defaultdict(float)
    payloads: dict[str, dict] = {}
    contrib: dict[str, dict[str, float]] = defaultdict(dict)
    for space, points in per_space.items():
        w = weights.get(space, 0.0)
        if w <= 0 or not points:
            continue
        lo, hi = _min_max([p.score for p in points])
        for p in points:
            pid = str(p.id)
            norm = (p.score - lo) / (hi - lo)
            add = w * norm
            acc[pid] += add
            contrib[pid][space] = round(add, 5)
            payloads[pid] = p.payload
    fused = [FusedHit(pid, acc[pid], payloads[pid], contrib[pid]) for pid in acc]
    return sorted(fused, key=lambda h: h.score, reverse=True)


def fuse(
    per_space: dict[str, list[models.ScoredPoint]],
    strategy: str,
    weights: dict[str, float] | None = None,
) -> list[FusedHit]:
    if strategy == "weighted" and weights:
        return weighted_fuse(per_space, weights)
    # "dbsf" (distribution-based) is approximated here by weighted-equal; RRF is default.
    return rrf_fuse(per_space)
