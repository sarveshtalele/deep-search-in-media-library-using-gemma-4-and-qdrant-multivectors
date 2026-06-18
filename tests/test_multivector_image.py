"""Image ingestion + asset-level MaxSim multivector recall (stub backend)."""

from pathlib import Path


def test_image_ingest_and_maxsim_recall(stub_env, tmp_path: Path):
    from deepsearch.ingestion import IngestionPipeline
    from deepsearch.query import get_engine

    # The stub describe_image path does not read pixels, so an empty file is fine.
    img = tmp_path / "frame.png"
    img.write_bytes(b"")

    result = IngestionPipeline().ingest_file(img, category="photos")
    assert result.error is None
    assert result.by_modality.get("video_frames", 0) == 1

    # Asset-level MaxSim recall should not error and should still return the asset.
    resp = get_engine().search("a picture", asset_recall=True)
    assert not resp.blocked
    assert resp.hits, "expected the ingested image asset to be retrievable"
    assert resp.hits[0].asset_name == "frame"


def test_multivector_named_spaces_created(stub_env):
    from deepsearch.vectorstore.schema import MULTIVECTOR_VECTORS
    from deepsearch.vectorstore.store import get_store

    store = get_store()
    store.ensure_collection(recreate=True)
    info = store.client.get_collection(store.collection)
    vectors = info.config.params.vectors
    for name in MULTIVECTOR_VECTORS:
        assert name in vectors, f"{name} multivector space missing"
        assert vectors[name].multivector_config is not None
