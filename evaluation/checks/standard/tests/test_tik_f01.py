import inspect
import typing

import pytest

import tiktoken


@pytest.fixture(scope="module")
def encoding():
    return tiktoken.get_encoding("cl100k_base")


def test_TIK_N01_budget_path_preserves_special_token_policy(encoding):
    special = "<|endoftext|>"
    with pytest.raises(ValueError, match="disallowed special token"):
        encoding.encode(special, max_tokens=100)
    expected = encoding.encode(special, allowed_special="all")
    assert encoding.encode(special, allowed_special="all", max_tokens=1) == expected


def test_TIK_N02_budget_path_preserves_surrogate_recovery(encoding):
    text = "before \ud800 after"
    expected = encoding.encode(text)
    assert encoding.encode(text, max_tokens=len(expected)) == expected


def test_TIK_N03_budgeted_batch_preserves_order_and_document_encoding(encoding):
    texts = ["z", "alpha", "\ud800", "omega"]
    expected = [encoding.encode(item) for item in texts]
    assert encoding.encode_batch(texts, max_tokens=max(map(len, expected))) == expected


def test_TIK_N06_new_budget_parameters_are_publicly_typed():
    encode_hints = typing.get_type_hints(tiktoken.Encoding.encode)
    batch_hints = typing.get_type_hints(tiktoken.Encoding.encode_batch)
    assert "max_tokens" in encode_hints
    assert {"max_tokens", "max_total_tokens"} <= batch_hints.keys()
    assert inspect.signature(tiktoken.Encoding.encode).return_annotation != inspect.Signature.empty
    assert inspect.signature(tiktoken.Encoding.encode_batch).return_annotation != inspect.Signature.empty
