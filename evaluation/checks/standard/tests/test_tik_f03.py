import inspect
import typing

import pytest

import tiktoken


@pytest.fixture(scope="module")
def encoding():
    return tiktoken.get_encoding("cl100k_base")


def test_TIK_N01_frequency_path_preserves_special_policy(encoding):
    special = "<|endoftext|>"
    with pytest.raises(ValueError, match="disallowed special token"):
        encoding.encode(special, analyze_frequency=True)
    tokens, frequency = encoding.encode(
        special, allowed_special="all", analyze_frequency=True
    )
    assert tokens == encoding.encode(special, allowed_special="all")
    assert frequency == {tokens[0]: 1}


def test_TIK_N02_frequency_path_preserves_surrogate_recovery(encoding):
    text = "before \ud800 after"
    tokens, frequency = encoding.encode(text, analyze_frequency=True)
    assert tokens == encoding.encode(text)
    assert sum(frequency.values()) == len(tokens)


def test_TIK_N03_frequency_batch_preserves_order_and_encode_results(encoding):
    texts = ["z", "🙂", "alpha", "z"]
    rows, frequency = encoding.encode_batch(
        texts, analyze_frequency=True, num_threads=3
    )
    assert rows == [encoding.encode(text) for text in texts]
    assert sum(frequency.values()) == sum(map(len, rows))


def test_TIK_N06_frequency_parameters_and_results_are_typed():
    encode_hints = typing.get_type_hints(tiktoken.Encoding.encode)
    batch_hints = typing.get_type_hints(tiktoken.Encoding.encode_batch)
    assert "analyze_frequency" in encode_hints
    assert "analyze_frequency" in batch_hints
    assert (
        inspect.signature(tiktoken.Encoding.encode).return_annotation
        != inspect.Signature.empty
    )
    assert (
        inspect.signature(tiktoken.Encoding.encode_batch).return_annotation
        != inspect.Signature.empty
    )
