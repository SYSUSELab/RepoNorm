import inspect
import typing

import pytest

import tiktoken


@pytest.fixture(scope="module")
def encoding():
    return tiktoken.get_encoding("cl100k_base")


def test_TIK_N01_validation_preserves_special_policy(encoding):
    special = "<|endoftext|>"
    with pytest.raises(ValueError, match="disallowed special token"):
        encoding.encode(special, validators=[lambda tokens: True])


def test_TIK_N02_validation_preserves_surrogate_recovery(encoding):
    text = "before \ud800 after"
    assert encoding.encode(text, validators=[lambda tokens: True]) == encoding.encode(text)


def test_TIK_N03_validation_batch_preserves_order_and_encode_results(encoding):
    texts = ["z", "🙂", "alpha", "z"]
    assert encoding.encode_batch(
        texts, validators=[lambda tokens: True], num_threads=3
    ) == [encoding.encode(text) for text in texts]


def test_TIK_N06_validator_parameters_and_results_are_typed():
    encode_hints = typing.get_type_hints(tiktoken.Encoding.encode)
    batch_hints = typing.get_type_hints(tiktoken.Encoding.encode_batch)
    assert "validators" in encode_hints
    assert "validators" in batch_hints
    assert (
        inspect.signature(tiktoken.Encoding.encode).return_annotation
        != inspect.Signature.empty
    )
    assert (
        inspect.signature(tiktoken.Encoding.encode_batch).return_annotation
        != inspect.Signature.empty
    )
