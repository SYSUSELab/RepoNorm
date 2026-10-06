import dspy
import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


def model_chunk(content):
    return ModelResponseStream(model="test", choices=[StreamingChoices(delta=Delta(content=content))])


def collect(listener, adapter, pieces):
    responses = []
    with dspy.context(adapter=adapter):
        for piece in pieces:
            response = listener.receive(model_chunk(piece))
            if response is not None:
                responses.append(response)
    return responses


@pytest.mark.parametrize(
    ("adapter", "pieces", "expected"),
    [
        (
            dspy.ChatAdapter(),
            ["[[", " ##", " answer", " ##", " ]]\n", *list("hello world"), "[[ ##", " completed", " ##", " ]]"],
            "hello world",
        ),
        (
            dspy.JSONAdapter(),
            ['{"', "answer", '":"', *list("hello world"), '"', "}"],
            "hello world",
        ),
        (
            dspy.XMLAdapter(),
            ["<", "answer", ">", *list("hello world"), "<", "/answer", ">"],
            "hello world",
        ),
    ],
)
def test_DSP_80412ce96d70_N004(adapter, pieces, expected):
    listener = dspy.streaming.StreamListener("answer", predict_name="predict")
    responses = collect(listener, adapter, pieces)

    assert "".join(response.chunk for response in responses) == expected
    assert all(response.signature_field_name == "answer" for response in responses)


def test_DSP_80412ce96d70_N005():
    pieces = ["[[", " ##", " answer", " ##", " ]]\n", *list("hello world"), "[[ ##", " completed", " ##", " ]]"]
    listener = dspy.streaming.StreamListener("answer", allow_reuse=True)

    first = collect(listener, dspy.ChatAdapter(), pieces)
    second = collect(listener, dspy.ChatAdapter(), pieces)

    assert "".join(response.chunk for response in first) == "hello world"
    assert "".join(response.chunk for response in second) == "hello world"
