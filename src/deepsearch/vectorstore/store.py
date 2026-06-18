"""Typed wrapper around the Qdrant client.

Runs in **embedded local mode** by default (on-disk, no server/Docker). Point a
``DEEPSEARCH_QDRANT_URL`` at a running Qdrant instance to switch to client/server
mode with zero code changes.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

from qdrant_client import QdrantClient, models

from deepsearch.config import Settings, get_settings
from deepsearch.logging_utils import get_logger
from deepsearch.vectorstore.schema import (
    INTEGER_INDEXES,
    KEYWORD_INDEXES,
    MULTIVECTOR_VECTORS,
    NAMED_VECTORS,
    Fragment,
)

log = get_logger(__name__)


class MediaVectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.collection = self.settings.qdrant.collection
        self.client = self._connect()

    # ------------------------------------------------------------------
    def _connect(self) -> QdrantClient:
        url = self.settings.qdrant.url.strip()
        if url:
            log.info(f"Connecting to Qdrant server at {url}")
            return QdrantClient(url=url)
        path = self.settings.qdrant_storage_path
        path.mkdir(parents=True, exist_ok=True)
        log.info(f"Using embedded Qdrant (local mode) at {path}")
        return QdrantClient(path=str(path))

    # ------------------------------------------------------------------
    def _embedding_dim(self) -> int:
        """Dimensionality of the live embedding model (falls back to config)."""
        try:
            from deepsearch.models import get_model_client

            return get_model_client().dim
        except Exception:
            return self.settings.models.embed_dim

    # ------------------------------------------------------------------
    def _distance(self) -> models.Distance:
        return {
            "Cosine": models.Distance.COSINE,
            "Dot": models.Distance.DOT,
            "Euclid": models.Distance.EUCLID,
        }[self.settings.qdrant.distance]

    def _quantization(self) -> models.QuantizationConfig | None:
        q = self.settings.qdrant.quantization
        if not q.enabled or q.type == "none":
            return None
        if q.type == "scalar":
            return models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, always_ram=q.always_ram
                )
            )
        return models.ProductQuantization(
            product=models.ProductQuantizationConfig(
                compression=models.CompressionRatio.X16, always_ram=q.always_ram
            )
        )

    # ------------------------------------------------------------------
    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists and recreate:
            self.client.delete_collection(self.collection)

        dim = self._embedding_dim()
        distance = self._distance()
        vectors = {
            name: models.VectorParams(size=dim, distance=distance)
            for name in NAMED_VECTORS
        }
        # Asset-level MaxSim multivector spaces (arrays of frame/audio vectors).
        for name in MULTIVECTOR_VECTORS:
            vectors[name] = models.VectorParams(
                size=dim,
                distance=distance,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            )
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=vectors,
            quantization_config=self._quantization(),
        )
        for field in KEYWORD_INDEXES:
            self.client.create_payload_index(
                self.collection, field, models.PayloadSchemaType.KEYWORD
            )
        for field in INTEGER_INDEXES:
            self.client.create_payload_index(
                self.collection, field, models.PayloadSchemaType.INTEGER
            )
        log.info(f"Created collection '{self.collection}' with {len(vectors)} named vectors")

    # ------------------------------------------------------------------
    def upsert_fragments(
        self, fragments: list[Fragment], vectors: list[list[float]]
    ) -> int:
        """Upsert fragments. ``vectors[i]`` is the embedding of ``fragments[i]``."""
        points = []
        for frag, vec in zip(fragments, vectors, strict=True):
            points.append(
                models.PointStruct(
                    id=frag.point_id,
                    vector={frag.modality.value: vec},  # populate one named space
                    payload=frag.payload(),
                )
            )
        if points:
            self.client.upsert(self.collection, points=points, wait=True)
        return len(points)

    # ------------------------------------------------------------------
    def upsert_asset_multivectors(
        self,
        asset_id: str,
        payload: dict,
        frame_vectors: list[list[float]],
        audio_vectors: list[list[float]],
    ) -> None:
        """Upsert ONE asset-level point holding MaxSim arrays of frame/audio vectors.

        Implements the requirement of a single asset holding N keyframe vectors
        (e.g. 20 vectors for 20 distinct keyframes) scored with MaxSim/ColBERT
        semantics. Point id is namespaced so it never collides with fragments.
        """
        vec: dict[str, list[list[float]]] = {}
        if frame_vectors:
            vec["frames_maxsim"] = frame_vectors
        if audio_vectors:
            vec["audio_maxsim"] = audio_vectors
        if not vec:
            return
        point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"asset:{asset_id}"))
        self.client.upsert(
            self.collection,
            points=[models.PointStruct(id=point_id, vector=vec, payload=payload)],
            wait=True,
        )

    def search_maxsim(
        self,
        vector_name: str,
        query_vector: list[float],
        limit: int,
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        """Asset-level MaxSim search over a multivector space.

        MaxSim is computed client-side: score(asset) = max over the asset's stored
        sub-vectors of <query, sub-vector>. Vectors are L2-normalized so the dot
        product is cosine similarity. This is portable across embedded *and* server
        Qdrant (the embedded client's native multivector query path is unreliable)
        and preserves true ColBERT/MaxSim semantics.
        """
        records, _ = self.client.scroll(
            self.collection,
            scroll_filter=query_filter,
            limit=10_000,
            with_payload=True,
            with_vectors=[vector_name],
        )
        scored: list[models.ScoredPoint] = []
        for r in records:
            vecs = (r.vector or {}).get(vector_name) if isinstance(r.vector, dict) else None
            if not vecs:
                continue
            best = max(_dot(query_vector, sub) for sub in vecs)
            scored.append(
                models.ScoredPoint(id=r.id, version=0, score=best, payload=r.payload)
            )
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    def delete_asset(self, asset_id: str) -> None:
        """Remove all points (fragments + asset-level MaxSim) for an asset, so a
        re-ingest replaces rather than duplicates them."""
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            self.collection,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="asset_id", match=models.MatchValue(value=asset_id))]
            ),
            wait=True,
        )

    # ------------------------------------------------------------------
    def search_named(
        self,
        vector_name: str,
        query_vector: list[float],
        limit: int,
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        """Search a single named vector space."""
        return self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points

    # ------------------------------------------------------------------
    def asset_fragments(self, asset_id: str, limit: int = 256) -> list[dict]:
        """All fragments for an asset, ordered by start time (for temporal RAG)."""
        flt = models.Filter(
            must=[models.FieldCondition(key="asset_id", match=models.MatchValue(value=asset_id))]
        )
        records, _ = self.client.scroll(
            self.collection, scroll_filter=flt, limit=limit, with_payload=True
        )
        payloads = [r.payload for r in records]
        return sorted(payloads, key=lambda p: (p.get("modality", ""), p.get("start_s", 0.0)))

    # ------------------------------------------------------------------
    def all_fragments(self) -> list[dict]:
        """Every fragment payload (descriptions + transcripts), for exact counting."""
        if not self.client.collection_exists(self.collection):
            return []
        out: list[dict] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                self.collection, limit=512, offset=offset, with_payload=True
            )
            out.extend(r.payload for r in records if r.payload.get("modality"))
            if offset is None:
                break
        return out

    # ------------------------------------------------------------------
    def list_assets(self) -> list[dict]:
        """Distinct assets with their modalities (for the RAG library overview)."""
        if not self.client.collection_exists(self.collection):
            return []
        assets: dict[str, dict] = {}
        offset = None
        while True:
            records, offset = self.client.scroll(
                self.collection, limit=512, offset=offset, with_payload=True
            )
            for r in records:
                p = r.payload
                mod = p.get("modality")
                aid = p.get("asset_id")
                if not aid or not mod:  # skip asset-level MaxSim points (no modality)
                    continue
                a = assets.setdefault(
                    aid,
                    {
                        "asset_id": aid, "asset_name": p.get("asset_name", aid),
                        "file_path": p.get("file_path", ""),
                        "duration_s": p.get("duration_s", 0.0), "modalities": set(),
                    },
                )
                a["modalities"].add(mod)
            if offset is None:
                break
        for a in assets.values():
            a["has_audio"] = "audio_chunks" in a["modalities"]
            a["modalities"] = sorted(a["modalities"])
        return list(assets.values())

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        if not self.client.collection_exists(self.collection):
            return {"assets": 0, "fragments": 0, "by_modality": {}}
        by_mod: dict[str, int] = defaultdict(int)
        assets: set[str] = set()
        fragments = 0
        offset = None
        while True:
            records, offset = self.client.scroll(
                self.collection, limit=512, offset=offset, with_payload=True
            )
            for r in records:
                assets.add(r.payload.get("asset_id", ""))
                mod = r.payload.get("modality")
                if mod:  # fragment point (asset-level MaxSim points have no modality)
                    by_mod[mod] += 1
                    fragments += 1
            if offset is None:
                break
        return {"assets": len(assets - {""}), "fragments": fragments, "by_modality": dict(by_mod)}


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def build_filter(
    categories: list[str] | None = None,
    modalities: list[str] | None = None,
    ingested_after: int | None = None,
) -> models.Filter | None:
    """Construct a Qdrant pre-filter from common UI facets."""
    must: list[models.Condition] = []
    if categories:
        must.append(models.FieldCondition(key="category", match=models.MatchAny(any=categories)))
    if modalities:
        must.append(models.FieldCondition(key="modality", match=models.MatchAny(any=modalities)))
    if ingested_after is not None:
        must.append(
            models.FieldCondition(key="ingested_at", range=models.Range(gte=ingested_after))
        )
    return models.Filter(must=must) if must else None


_STORE: MediaVectorStore | None = None


def get_store() -> MediaVectorStore:
    global _STORE
    if _STORE is None:
        _STORE = MediaVectorStore()
    return _STORE
