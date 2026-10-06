import re

from pydantic import BaseModel

from outlines import types
from outlines.types.dsl import Regex, to_regex


def test_OUT_4dd7802a8a50_N001_ssn_keeps_public_term_contract():
    assert isinstance(types.ssn, Regex)

    class Model(BaseModel):
        value: types.ssn

    pattern = Model.model_json_schema()["properties"]["value"]["pattern"]
    assert re.fullmatch(pattern, "123-45-6789")
    assert re.fullmatch(to_regex("<" + types.ssn + ">"), "<000-00-0000>")
