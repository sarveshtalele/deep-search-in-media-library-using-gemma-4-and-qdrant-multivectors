"""The unified multi-modal query execution engine.

Pipeline:
    safety pre-filter
      -> intent analysis (dynamic weights)
      -> embed query into the unified space
      -> per-named-vector prefetch (with optional payload pre-filter)
      -> score fusion (RRF / weighted)
      -> optional Gemma 4 listwise rerank
      -> SearchHit list with second-accurate offsets for playback
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import models

from deepsearch.config import Settings, get_settings
from deepsearch.logging_utils import get_logger
from deepsearch.models import check_query, get_model_client
from deepsearch.query import fusion as fusion_mod
from deepsearch.query.intent import Intent, analyze_intent
from deepsearch.query.rerank import rerank
from deepsearch.vectorstore import NAMED_VECTORS
from deepsearch.vectorstore.schema import MULTIVECTOR_VECTORS
from deepsearch.vectorstore.store import MediaVectorStore, build_filter, get_store

log = get_logger(__name__)


@dataclass
class SearchHit:
    point_id: str
    asset_id: str
    asset_name: str
    file_path: str
    modality: str
    text: str
    start_s: float
    end_s: float
    score: float
    thumbnail_path: str | None = None
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def timestamp_label(self) -> str:
        m, s = divmod(int(self.start_s), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class SearchResponse:
    query: str
    intent: Intent
    hits: list[SearchHit]
    blocked: bool = False
    message: str = ""


class SearchEngine:
    def __init__(
        self, settings: Settings | None = None, store: MediaVectorStore | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store()
        self.model = get_model_client()

    def search(
        self,
        query: str,
        categories: list[str] | None = None,
        modalities: list[str] | None = None,
        ingested_after: int | None = None,
        weights_override: dict[str, float] | None = None,
        asset_recall: bool = False,
    ) -> SearchResponse:
        verdict = check_query(query)
        if verdict.blocked:
            return SearchResponse(query, Intent("blocked", {}), [], blocked=True,
                                  message=verdict.reason)

        intent = analyze_intent(query)
        weights = weights_override or intent.weights
        cfg = self.settings.retrieval

        qvec = self.model.embed_query(query)
        flt = build_filter(categories, modalities, ingested_after)

        # Optional two-stage retrieval: asset-level MaxSim multivector recall first,
        # then restrict fragment search to the winning assets for the exact second.
        if asset_recall:
            flt = self._with_asset_recall(qvec, flt, cfg.prefetch_limit)

        # Only query spaces with non-trivial weight (saves work for focused intents).
        spaces = [s for s in NAMED_VECTORS if weights.get(s, 0.0) > 0.01] or list(NAMED_VECTORS)
        per_space = {
            space: self.store.search_named(space, qvec, cfg.prefetch_limit, flt)
            for space in spaces
        }

        # Manual weight overrides force weighted fusion so the UI sliders actually
        # bite; auto/intent mode uses the configured default (RRF). Without this
        # the weights were silently ignored under the default rrf strategy.
        strategy = "weighted" if weights_override is not None else cfg.fusion
        fused = fusion_mod.fuse(per_space, strategy, weights)

        top = fused[: cfg.rerank_top_n] if cfg.rerank else fused[: cfg.top_k]
        if cfg.rerank and len(top) > 1:
            top = rerank(query, top, cfg.rerank_top_n)
        hits = [self._to_hit(h) for h in top][: cfg.top_k]

        return SearchResponse(
            query, intent, hits, message=f"intent={intent.intent} fusion={strategy}"
        )

    # ------------------------------------------------------------------
    def _with_asset_recall(
        self, qvec: list[float], flt: models.Filter | None, limit: int
    ) -> models.Filter | None:
        """Run MaxSim over the asset-level multivector spaces and fold the top
        asset ids into the fragment-search filter."""
        asset_ids: set[str] = set()
        for name in MULTIVECTOR_VECTORS:
            try:
                for p in self.store.search_maxsim(name, qvec, limit):
                    aid = p.payload.get("asset_id")
                    if aid:
                        asset_ids.add(aid)
            except Exception as exc:  # pragma: no cover - env dependent
                log.warning(f"MaxSim recall on {name} failed: {exc!s}")
        if not asset_ids:
            return flt
        cond = models.FieldCondition(key="asset_id", match=models.MatchAny(any=list(asset_ids)))
        if flt is None:
            return models.Filter(must=[cond])
        flt.must = list(flt.must or []) + [cond]
        return flt

    # ------------------------------------------------------------------
    def _to_hit(self, fused: fusion_mod.FusedHit) -> SearchHit:
        p = fused.payload
        return SearchHit(
            point_id=fused.point_id,
            asset_id=p.get("asset_id", ""),
            asset_name=p.get("asset_name", ""),
            file_path=p.get("file_path", ""),
            modality=p.get("modality", ""),
            text=p.get("text", ""),
            start_s=float(p.get("start_s", 0.0)),
            end_s=float(p.get("end_s", 0.0)),
            score=round(fused.score, 5),
            thumbnail_path=p.get("thumbnail_path"),
            contributions=fused.contributions,
        )


_ENGINE: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SearchEngine()
    return _ENGINE
