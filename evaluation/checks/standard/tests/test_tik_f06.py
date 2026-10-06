import inspect
import typing

import pytest

import tiktoken


@pytest.fixture(scope="module")
def encoding():
    return tiktoken.get_encoding("cl100k_base")


def test_TIK_N01_transformed_text_preserves_special_policy(encoding):
    special = "<|endoftext|>"
    with pytest.raises(ValueError, match="disallowed special token"):
        encoding.encode("x", transformers=lambda text: special)
    assert encoding.encode(
        "x", transformers=lambda text: special, allowed_special="all"
    ) == encoding.encode(special, allowed_special="all")


def test_TIK_N02_transformed_text_preserves_surrogate_recovery(encoding):
    transformed = "before \ud800 after"
    assert encoding.encode("x", transformers=lambda text: transformed) == encoding.encode(
        transformed
    )


def test_TIK_N03_transform_batch_preserves_order_and_encode_results(encoding):
    texts = ["z", "🙂", "alpha", "z"]
    transform = lambda text: text.upper()
    assert encoding.encode_batch(texts, transformers=transform, num_threads=3) == [
        encoding.encode(text, transformers=transform) for text in texts
    ]


def test_TIK_N06_transformer_parameters_and_results_are_typed():
    encode_hints = typing.get_type_hints(tiktoken.Encoding.encode)
    batch_hints = typing.get_type_hints(tiktoken.Encoding.encode_batch)
    assert "transformers" in encode_hints
    assert "transformers" in batch_hints
    assert (
        inspect.signature(tiktoken.Encoding.encode).return_annotation
        != inspect.Signature.empty
    )
    assert (
        inspect.signature(tiktoken.Encoding.encode_batch).return_annotation
        != inspect.Signature.empty
    )
