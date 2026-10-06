import re

from pydantic import BaseModel

from outlines import types
from outlines.types.dsl import Regex, to_regex


def test_OUT_4dd7802a8a50_N001_new_public_regex_types_keep_term_contract():
    for name, value in (
        ("hex_str", "0xCAFE"),
        ("uuid4", "123e4567-e89b-42d3-a456-426614174000"),
        ("ipv4", "001.002.003.004"),
    ):
        custom_type = getattr(types, name)
        assert isinstance(custom_type, Regex)

        class Model(BaseModel):
            field: custom_type

        schema_pattern = Model.model_json_schema()["properties"]["field"]["pattern"]
        assert re.fullmatch(schema_pattern, value)
        assert re.fullmatch(to_regex(custom_type), value)

    combined = types.hex_str + ":" + types.ipv4
    assert re.fullmatch(to_regex(combined), "0xCAFE:001.002.003.004")
