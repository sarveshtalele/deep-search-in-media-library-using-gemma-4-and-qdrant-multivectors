"""Offline ingestion pipeline: media -> fragments -> embeddings -> Qdrant."""

from deepsearch.ingestion.pipeline import IngestionPipeline, IngestResult

__all__ = ["IngestionPipeline", "IngestResult"]
