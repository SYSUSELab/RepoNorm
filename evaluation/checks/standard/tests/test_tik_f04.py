import inspect
import typing

import pytest

import tiktoken


@pytest.fixture(scope="module")
def encoding():
    return tiktoken.get_encoding("cl100k_base")


def test_TIK_N01_chunk_path_preserves_special_policy(encoding):
    special = "<|endoftext|>"
    with pytest.raises(ValueError, match="disallowed special token"):
        encoding.encode(special, chunk_size=3, chunk_overlap=1)
    chunks = encoding.encode(
        special, allowed_special="all", chunk_size=3, chunk_overlap=1
    )
    assert chunks


def test_TIK_N02_chunk_change_preserves_surrogate_recovery(encoding):
    text = "before \ud800 after"
    assert encoding.encode(text) == encoding.encode_batch([text])[0]
    assert encoding.encode(text, chunk_size=4)


def test_TIK_N06_chunk_parameters_and_results_are_typed():
    hints = typing.get_type_hints(tiktoken.Encoding.encode)
    assert "chunk_size" in hints
    assert "chunk_overlap" in hints
    assert (
        inspect.signature(tiktoken.Encoding.encode).return_annotation
        != inspect.Signature.empty
    )
