"""Library RAG: deterministic counting + no-audio awareness (stub backend)."""

from pathlib import Path


def _ingest(text: str, tmp_path: Path, name: str):
    from deepsearch.ingestion import IngestionPipeline

    p = tmp_path / f"{name}.txt"
    p.write_text(text, encoding="utf-8")
    IngestionPipeline().ingest_file(p, category="x")


def test_rag_counts_exact(stub_env, tmp_path: Path):
    from deepsearch.rag import get_rag

    _ingest("finance and finance again", tmp_path, "a")  # 2
    _ingest("nothing relevant here", tmp_path, "b")       # 0
    _ingest("a single finance mention", tmp_path, "c")    # 1

    res = get_rag().answer("how many times is finance mentioned")
    assert "3 times" in res["answer"], res["answer"]
    assert res["sources"], "counting should return source moments"


def test_rag_count_zero(stub_env, tmp_path: Path):
    from deepsearch.rag import get_rag

    _ingest("totally unrelated content", tmp_path, "a")
    res = get_rag().answer("count the word elephant")
    assert "not mentioned" in res["answer"].lower()
