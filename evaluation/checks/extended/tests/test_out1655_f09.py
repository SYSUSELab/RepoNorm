import re

from pydantic import BaseModel

from outlines import types
from outlines.types.dsl import Regex, to_regex


def test_OUT_4dd7802a8a50_N001_guid_keeps_public_term_contract():
    assert isinstance(types.guid, Regex)

    class Model(BaseModel):
        value: types.guid

    pattern = Model.model_json_schema()["properties"]["value"]["pattern"]
    assert re.fullmatch(pattern, "{ABCDEF12-3456-7890-ABCD-EF1234567890}")
    framed = "<{00000000-0000-0000-0000-000000000000}>"
    assert re.fullmatch(to_regex("<" + types.guid + ">"), framed)
