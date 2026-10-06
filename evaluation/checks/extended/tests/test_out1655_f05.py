import re

from pydantic import BaseModel

from outlines import types
from outlines.types.dsl import Regex, to_regex


def test_OUT_4dd7802a8a50_N001_ipv6_keeps_public_term_contract():
    assert isinstance(types.ipv6, Regex)

    class Model(BaseModel):
        value: types.ipv6

    pattern = Model.model_json_schema()["properties"]["value"]["pattern"]
    assert re.fullmatch(pattern, "2001:db8:85a3:0:0:8a2e:370:7334")
    assert re.fullmatch(to_regex("<" + types.ipv6 + ">"), "<1:2:3:4:5:6:7:8>")
