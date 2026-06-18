"""Shared fixtures. All tests run against the deterministic stub backend and an
isolated on-disk Qdrant collection, so no model weights or servers are needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def stub_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DEEPSEARCH_USE_STUB", "true")
    monkeypatch.setenv("DEEPSEARCH_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("DEEPSEARCH_COLLECTION", "test_collection")
    monkeypatch.setenv("DEEPSEARCH_LOG_LEVEL", "WARNING")

    # Reset cached singletons / settings so the new env takes effect.
    import deepsearch.config as config

    config.get_settings.cache_clear()
    import deepsearch.logging_utils as lg

    lg._configure.cache_clear()

    from deepsearch.models import gemma
    from deepsearch.query import search
    from deepsearch.rag import chat as rag_chat
    from deepsearch.vectorstore import store

    gemma._SINGLETON = None
    store._STORE = None
    search._ENGINE = None
    rag_chat._RAG = None
    yield
