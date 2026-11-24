import dspy
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
from test_dsp8587_f01 import collect, test_DSP_80412ce96d70_N004

__all__ = ["test_DSP_80412ce96d70_N004"]


def test_DSP_80412ce96d70_N005():
    now = [1.0]
    listener = dspy.streaming.StreamListener("answer", allow_reuse=True, idle_timeout_s=1.0, clock=lambda: now[0])
    with dspy.context(adapter=dspy.ChatAdapter()):
        for content in ("[[", " ## answer ## ]]\n", "OLD_PARTIAL"):
            listener.receive(ModelResponseStream(model="test", choices=[StreamingChoices(delta=Delta(content=content))]))
        now[0] = 2.0
        timeout = listener.tick()
        assert timeout is not None and timeout.chunk == ""
        assert listener.tick() is None

    # The old incomplete buffer must not leak into a subsequent complete stream.
    # This observes receive/tick, not the representation of either internal queue.
    now[0] = 3.0
    pieces = ["[[", " ##", " answer", " ##", " ]]\n", *list("fresh stream"), "[[ ##", " completed", " ##", " ]]"]
    for _ in range(2):
        responses = collect(listener, dspy.ChatAdapter(), pieces)
        assert "".join(response.chunk for response in responses) == "fresh stream"
        assert all(response.signature_field_name == "answer" for response in responses)
        assert listener.tick() is None
