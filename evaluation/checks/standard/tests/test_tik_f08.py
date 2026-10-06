import inspect
import typing

import pytest

import tiktoken


@pytest.fixture(scope="module")
def encoding():
    return tiktoken.get_encoding("cl100k_base")


def test_TIK_N01_pattern_path_preserves_special_policy(encoding):
    special = "<|endoftext|>"
    with pytest.raises(ValueError, match="disallowed special token"):
        encoding.encode(special, return_repeated_pattern=True)
    tokens, patterns = encoding.encode(
        special, allowed_special="all", return_repeated_pattern=True
    )
    assert tokens == encoding.encode(special, allowed_special="all")
    assert patterns == {}


def test_TIK_N02_pattern_path_preserves_surrogate_recovery(encoding):
    text = "before \ud800 after"
    tokens, patterns = encoding.encode(text, return_repeated_pattern=True)
    assert tokens == encoding.encode(text)


def test_TIK_N06_pattern_parameter_and_result_are_typed():
    hints = typing.get_type_hints(tiktoken.Encoding.encode)
    assert "return_repeated_pattern" in hints
    assert (
        inspect.signature(tiktoken.Encoding.encode).return_annotation
        != inspect.Signature.empty
    )
