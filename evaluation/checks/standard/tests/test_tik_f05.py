import inspect
import typing

import pytest

import tiktoken


@pytest.fixture(scope="module")
def encoding():
    return tiktoken.get_encoding("cl100k_base")


def test_TIK_N01_filter_paths_preserve_special_policy(encoding):
    special = "<|endoftext|>"
    for method in (encoding.encode, encoding.encode_to_numpy):
        with pytest.raises(ValueError, match="disallowed special token"):
            method(special, filter_tokens=[encoding.eot_token])


def test_TIK_N02_list_filtering_preserves_surrogate_recovery(encoding):
    text = "before \ud800 after"
    assert encoding.encode(text, filter_tokens=()) == encoding.encode_batch([text])[0]


def test_TIK_N06_filter_parameters_and_results_are_typed():
    encode_hints = typing.get_type_hints(tiktoken.Encoding.encode)
    numpy_signature = inspect.signature(tiktoken.Encoding.encode_to_numpy)
    assert "filter_tokens" in encode_hints
    assert numpy_signature.parameters["filter_tokens"].annotation != inspect.Signature.empty
    assert (
        inspect.signature(tiktoken.Encoding.encode).return_annotation
        != inspect.Signature.empty
    )
    assert (
        numpy_signature.return_annotation != inspect.Signature.empty
    )
