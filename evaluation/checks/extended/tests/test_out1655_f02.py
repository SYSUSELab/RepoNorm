import re

from pydantic import BaseModel

from outlines import types
from outlines.types.dsl import Regex, to_regex


def test_OUT_4dd7802a8a50_N001_binary_str_keeps_public_term_contract():
    assert isinstance(types.binary_str, Regex)

    class Model(BaseModel):
        value: types.binary_str

    pattern = Model.model_json_schema()["properties"]["value"]["pattern"]
    assert re.fullmatch(pattern, "0b101")
    combined = "value=" + types.binary_str
    assert re.fullmatch(to_regex(combined), "value=0b101")
