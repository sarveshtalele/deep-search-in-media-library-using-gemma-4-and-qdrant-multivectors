"""Qdrant vector store: schema definition and a thin typed client."""

from deepsearch.vectorstore.schema import (
    MODALITIES,
    NAMED_VECTORS,
    Fragment,
    Modality,
)
from deepsearch.vectorstore.store import MediaVectorStore, get_store

__all__ = [
    "MODALITIES",
    "NAMED_VECTORS",
    "Fragment",
    "Modality",
    "MediaVectorStore",
    "get_store",
]
