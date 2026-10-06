"""Token counting treats repository contents as literal source text."""

from types import SimpleNamespace

import pytest

from codewiki.src.be.utils import count_tokens, enc
from codewiki.src.be.cluster_modules import get_clustering_input_token_count


@pytest.mark.parametrize("text", [
    "normal Python text", "", "<|endoftext|>", "<|fim_prefix|>",
    'def encode():\n    """Example: <|endoftext|> and <|fim_suffix|>."""\n',
])
def test_literal_source_matches_ordinary_encoding(text):
    assert count_tokens(text) == len(enc.encode_ordinary(text))


def test_clustering_counts_tokenizer_documentation_without_special_token_error():
    components = {"example.py::encode": SimpleNamespace(
        relative_path="example.py", source_code='def encode():\n    """<|endoftext|>"""',
    )}
    assert get_clustering_input_token_count(list(components), components) > 0
