import asyncio

import pytest

from jinja2 import Environment
from jinja2 import StrictUndefined
from jinja2 import UndefinedError


def test_JIN_N01_hierarchy_uses_environment_lookup_at_each_level():
    class PrefixEnvironment(Environment):
        def getitem(self, obj, argument):
            return f"env:{super().getitem(obj, argument)}"

    groups = PrefixEnvironment().call_filter(
        "groupby", [{"a": "x", "b": "y"}], [["a", "b"]], {"levels": 2}
    )
    assert groups[0].grouper == "env:x"
    assert groups[0].list[0].grouper == "env:y"


def test_JIN_N11_hierarchy_preserves_group_objects_at_each_level():
    rows = [{"a": "x", "b": "y"}]
    groups = Environment().call_filter(
        "groupby", rows, [["a", "b"]], {"levels": 2}
    )
    outer = groups[0]
    inner = outer.list[0]
    assert tuple(outer) == ("x", [inner])
    assert tuple(inner) == ("y", rows)
    assert repr(inner) == repr(("y", rows))


def test_JIN_N12_hierarchy_materializes_async_input():
    async def rows():
        for value in [{"a": "x", "b": "y"}, {"a": "x", "b": "y"}]:
            yield value

    async def run():
        return await Environment(enable_async=True).call_filter(
            "groupby", rows(), [["a", "b"]], {"levels": 2}
        )

    groups = asyncio.run(run())
    assert len(groups[0].list[0].list) == 2


def test_JIN_N13_hierarchy_applies_default_at_each_level():
    groups = Environment().call_filter(
        "groupby", [{}, {}], [["a", "b"]], {"levels": 2, "default": "missing"}
    )
    assert groups[0].grouper == "missing"
    assert groups[0].list[0].grouper == "missing"


def test_JIN_N14_hierarchy_propagates_strict_undefined():
    env = Environment(undefined=StrictUndefined)
    with pytest.raises(UndefinedError):
        env.call_filter("groupby", [{}, {}], [["a", "b"]], {"levels": 2})
