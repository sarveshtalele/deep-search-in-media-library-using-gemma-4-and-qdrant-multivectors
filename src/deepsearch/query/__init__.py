"""Real-time query pipeline: intent -> multi-space search -> fusion -> rerank."""

from deepsearch.query.search import SearchEngine, SearchHit, get_engine

__all__ = ["SearchEngine", "SearchHit", "get_engine"]
