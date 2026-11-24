import inspect
import typing

import pytest

import tiktoken


@pytest.fixture(scope="module")
def encoding():
    return tiktoken.get_encoding("cl100k_base")


def test_TIK_N01_compression_preserves_special_policy(encoding):
    special = "<|endoftext|>"
    with pytest.raises(ValueError, match="disallowed special token"):
        encoding.encode(special, compress=True)
    assert encoding.encode(special, allowed_special="all", compress=True)


def test_TIK_N02_compression_preserves_surrogate_recovery(encoding):
    text = "before \ud800 after"
    ordinary = encoding.encode(text)
    expected = [ordinary[0]]
    for token in ordinary[1:]:
        if token != expected[-1]:
            expected.append(token)
    assert encoding.encode(text, compress=True) == expected


def test_TIK_N06_compress_parameter_and_result_are_typed():
    hints = typing.get_type_hints(tiktoken.Encoding.encode)
    assert "compress" in hints
    assert (
        inspect.signature(tiktoken.Encoding.encode).return_annotation
        != inspect.Signature.empty
    )
