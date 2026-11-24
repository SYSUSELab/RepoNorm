import inspect
import typing

import pytest

import tiktoken


@pytest.fixture(scope="module")
def encoding():
    return tiktoken.get_encoding("cl100k_base")


def test_TIK_N01_position_path_preserves_special_policy(encoding):
    special = "<|endoftext|>"
    with pytest.raises(ValueError, match="disallowed special token"):
        encoding.encode(special, return_positions=True)
    tokens, positions = encoding.encode(
        special, allowed_special="all", return_positions=True
    )
    assert tokens == encoding.encode(special, allowed_special="all")
    assert positions == [0]


def test_TIK_N02_default_path_preserves_surrogate_recovery(encoding):
    text = "before \ud800 after"
    assert encoding.encode(text) == encoding.encode_batch([text])[0]


def test_TIK_N03_position_batch_preserves_order_and_encode_results(encoding):
    texts = ["z", "🙂", "alpha"]
    result = encoding.encode_batch(texts, return_positions=True)
    assert [tokens for tokens, positions in result] == [
        encoding.encode(text) for text in texts
    ]


def test_TIK_N06_position_parameters_and_results_are_typed():
    encode_hints = typing.get_type_hints(tiktoken.Encoding.encode)
    batch_hints = typing.get_type_hints(tiktoken.Encoding.encode_batch)
    assert "return_positions" in encode_hints
    assert "return_positions" in batch_hints
    assert inspect.signature(tiktoken.Encoding.encode).return_annotation != inspect.Signature.empty
    assert inspect.signature(tiktoken.Encoding.encode_batch).return_annotation != inspect.Signature.empty
