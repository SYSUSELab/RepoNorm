import re

from pydantic import BaseModel

from outlines import types
from outlines.types.dsl import Regex, to_regex


def test_OUT_4dd7802a8a50_N001_octal_str_keeps_public_term_contract():
    assert isinstance(types.octal_str, Regex)

    class Model(BaseModel):
        value: types.octal_str

    pattern = Model.model_json_schema()["properties"]["value"]["pattern"]
    assert re.fullmatch(pattern, "0o755")
    assert re.fullmatch(to_regex("mode=" + types.octal_str), "mode=0o755")
