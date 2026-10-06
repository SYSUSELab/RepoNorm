import re

from pydantic import BaseModel

from outlines import types
from outlines.types.dsl import Regex, to_regex


def test_OUT_4dd7802a8a50_N001_url_keeps_public_term_contract():
    assert isinstance(types.url, Regex)

    class Model(BaseModel):
        value: types.url

    pattern = Model.model_json_schema()["properties"]["value"]["pattern"]
    assert re.fullmatch(pattern, "https://api.example.com/v1?q=x#result")
    assert re.fullmatch(to_regex("<" + types.url + ">"), "<http://localhost>")
