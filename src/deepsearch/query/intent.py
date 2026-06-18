"""Query intent analysis -> dynamic retrieval weights.

Gemma 4 classifies whether the user wants a visual occurrence, a spoken phrase
or a conceptual topic, and proposes per-space weights. A deterministic keyword
heuristic backs it up when the model is stubbed or returns malformed JSON, so
the engine always gets usable weights.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from deepsearch.logging_utils import get_logger
from deepsearch.models import get_model_client
from deepsearch.models.gemma import _INTENT_PROMPT

log = get_logger(__name__)

_VISUAL = {"show", "see", "look", "color", "colour", "chart", "scene", "wearing",
           "appear", "screen", "visible", "graph", "image", "frame"}
_SPOKEN = {"say", "said", "speak", "spoke", "mention", "talk", "voice", "told",
           "quote", "phrase", "word"}


@dataclass
class Intent:
    intent: str  # visual | spoken | conceptual | mixed
    weights: dict[str, float]


def _normalize_weights(w: dict[str, float]) -> dict[str, float]:
    keys = ("text_descriptions", "video_frames", "audio_chunks")
    clean = {k: max(0.0, float(w.get(k, 0.0))) for k in keys}
    total = sum(clean.values()) or 1.0
    return {k: round(v / total, 4) for k, v in clean.items()}


def _heuristic(query: str) -> Intent:
    tokens = set(re.findall(r"[a-z']+", query.lower()))
    visual = len(tokens & _VISUAL)
    spoken = len(tokens & _SPOKEN)
    if visual and not spoken:
        return Intent("visual", _normalize_weights(
            {"video_frames": 0.6, "text_descriptions": 0.3, "audio_chunks": 0.1}))
    if spoken and not visual:
        return Intent("spoken", _normalize_weights(
            {"audio_chunks": 0.6, "text_descriptions": 0.3, "video_frames": 0.1}))
    return Intent("mixed", _normalize_weights(
        {"text_descriptions": 0.5, "video_frames": 0.3, "audio_chunks": 0.2}))


def analyze_intent(query: str) -> Intent:
    from deepsearch.config import get_settings

    model = get_model_client()
    fallback = _heuristic(query)
    # Heuristic is instant; the LLM call costs 10-100s on local e4b. Off by default.
    if model.is_stub or not get_settings().retrieval.intent_llm:
        return fallback
    try:
        raw = model.generate(_INTENT_PROMPT.format(query=query))
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
        weights = _normalize_weights(data.get("weights", {}))
        intent = str(data.get("intent", fallback.intent))
        if sum(weights.values()) == 0:
            return fallback
        return Intent(intent, weights)
    except Exception as exc:
        log.warning(f"Intent analysis fell back to heuristic: {exc!s}")
        return fallback
