import re

from pydantic import BaseModel

from outlines import types
from outlines.types.dsl import Regex, to_regex


def test_OUT_4dd7802a8a50_N001_hash_sha256_keeps_public_term_contract():
    assert isinstance(types.hash_sha256, Regex)

    class Model(BaseModel):
        value: types.hash_sha256

    pattern = Model.model_json_schema()["properties"]["value"]["pattern"]
    assert re.fullmatch(pattern, "A" * 64)
    assert re.fullmatch(to_regex("<" + types.hash_sha256 + ">"), "<" + "0" * 64 + ">")
