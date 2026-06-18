from deepsearch.query.intent import analyze_intent


def test_visual_intent(stub_env):
    intent = analyze_intent("show me the scene where the chart appears on screen")
    assert intent.weights["video_frames"] >= intent.weights["audio_chunks"]


def test_spoken_intent(stub_env):
    intent = analyze_intent("what did the speaker say about the budget")
    assert intent.weights["audio_chunks"] >= intent.weights["video_frames"]


def test_weights_sum_to_one(stub_env):
    intent = analyze_intent("a generic conceptual query about climate")
    assert abs(sum(intent.weights.values()) - 1.0) < 1e-6
