"""LLM-as-reranker.

Gemma 4 is a generative decoder, not a trained cross-encoder, so reranking is a
*listwise* prompt: the model is shown the query and the candidate fragment texts
and asked to return their ids in best-first order. Falls back to the fused order
if the model is stubbed or returns an unusable response.
"""

from __future__ import annotations

import json
import re

from deepsearch.logging_utils import get_logger
from deepsearch.models import get_model_client
from deepsearch.models.gemma import _RERANK_PROMPT
from deepsearch.query.fusion import FusedHit

log = get_logger(__name__)


def _format_candidates(hits: list[FusedHit]) -> str:
    lines = []
    for i, h in enumerate(hits):
        snippet = (h.payload.get("text") or "").replace("\n", " ")[:240]
        mod = h.payload.get("modality", "?")
        lines.append(f"[{i}] ({mod}) {snippet}")
    return "\n".join(lines)


def rerank(query: str, hits: list[FusedHit], top_n: int) -> list[FusedHit]:
    if len(hits) <= 1:
        return hits
    model = get_model_client()
    candidates = hits[:top_n]
    if model.is_stub:
        return hits

    try:
        raw = model.generate(
            _RERANK_PROMPT.format(query=query, candidates=_format_candidates(candidates))
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        order = json.loads(match.group(0)).get("order", []) if match else []
        order = [int(i) for i in order if isinstance(i, (int, str)) and str(i).isdigit()]
        seen, reranked = set(), []
        for idx in order:
            if 0 <= idx < len(candidates) and idx not in seen:
                reranked.append(candidates[idx])
                seen.add(idx)
        # Append any candidates the model omitted, then the untouched tail.
        for i, h in enumerate(candidates):
            if i not in seen:
                reranked.append(h)
        return reranked + hits[top_n:]
    except Exception as exc:
        log.warning(f"Rerank fell back to fused order: {exc!s}")
        return hits
