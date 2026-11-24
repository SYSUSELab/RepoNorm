import re

from pydantic import BaseModel

from outlines import types
from outlines.types.dsl import Regex, to_regex


def test_OUT_4dd7802a8a50_N001_mac_address_keeps_public_term_contract():
    assert isinstance(types.mac_address, Regex)

    class Model(BaseModel):
        value: types.mac_address

    pattern = Model.model_json_schema()["properties"]["value"]["pattern"]
    assert re.fullmatch(pattern, "00:1B:44:11:3A:B7")
    assert re.fullmatch(to_regex("<" + types.mac_address + ">"), "<AA-BB-CC-DD-EE-FF>")
