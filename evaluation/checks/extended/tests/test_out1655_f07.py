import re

from pydantic import BaseModel

from outlines import types
from outlines.types.dsl import Regex, to_regex


def test_OUT_4dd7802a8a50_N001_credit_card_keeps_public_term_contract():
    assert isinstance(types.credit_card, Regex)

    class Model(BaseModel):
        value: types.credit_card

    pattern = Model.model_json_schema()["properties"]["value"]["pattern"]
    assert re.fullmatch(pattern, "1234-5678-9012-3456")
    assert re.fullmatch(to_regex("<" + types.credit_card + ">"), "<1234567890123456>")
