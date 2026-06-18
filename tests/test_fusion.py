from dataclasses import dataclass

from deepsearch.query.fusion import fuse, rrf_fuse, weighted_fuse


@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict


def _space(ids_scores):
    return [FakePoint(i, s, {"text": i}) for i, s in ids_scores]


def test_rrf_rewards_agreement_across_spaces():
    per_space = {
        "video_frames": _space([("a", 0.9), ("b", 0.8)]),
        "audio_chunks": _space([("a", 0.7), ("c", 0.6)]),
    }
    fused = rrf_fuse(per_space)
    # 'a' is top-ranked in both spaces -> should win.
    assert fused[0].point_id == "a"
    assert "video_frames" in fused[0].contributions


def test_weighted_respects_weights():
    per_space = {
        "video_frames": _space([("a", 1.0), ("b", 0.0)]),
        "audio_chunks": _space([("b", 1.0), ("a", 0.0)]),
    }
    fused = weighted_fuse(per_space, {"video_frames": 0.9, "audio_chunks": 0.1})
    assert fused[0].point_id == "a"


def test_fuse_dispatch_defaults_to_rrf():
    per_space = {"video_frames": _space([("a", 0.5)])}
    assert fuse(per_space, "rrf")[0].point_id == "a"
