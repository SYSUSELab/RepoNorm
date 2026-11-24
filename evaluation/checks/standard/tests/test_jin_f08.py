import asyncio

import pytest

from jinja2 import Environment
from jinja2 import StrictUndefined
from jinja2 import UndefinedError


def test_JIN_N01_counted_path_uses_environment_lookup():
    class PrefixEnvironment(Environment):
        def getitem(self, obj, argument):
            if argument == "kind":
                return f"env:{obj[argument]}"

            return super().getitem(obj, argument)

    groups = PrefixEnvironment().call_filter(
        "groupby", [{"kind": "x"}], ["kind"], {"with_counts": True}
    )
    assert [(g.grouper, g.count) for g in groups] == [("env:x", 1)]


def test_JIN_N11_default_and_extended_group_interfaces():
    env = Environment()
    rows = [{"kind": "x"}, {"kind": "x"}]
    legacy = env.call_filter("groupby", rows, ["kind"])[0]
    counted = env.call_filter("groupby", rows, ["kind"], {"with_counts": True})[0]
    assert tuple(legacy) == ("x", rows)
    assert callable(legacy.count)
    assert tuple(counted) == ("x", rows, 2)
    assert repr(counted) == repr(("x", rows, 2))


def test_JIN_N12_counted_async_path_materializes_async_input():
    async def rows():
        for value in [{"kind": "x"}, {"kind": "x"}]:
            yield value

    async def run():
        env = Environment(enable_async=True)
        return await env.call_filter(
            "groupby", rows(), ["kind"], {"with_counts": True}
        )

    groups = asyncio.run(run())
    assert [(g.grouper, g.count, len(g.list)) for g in groups] == [("x", 2, 2)]


def test_JIN_N13_counted_path_preserves_default():
    groups = Environment().call_filter(
        "groupby", [{}, {}], ["kind"], {"default": "missing", "with_counts": True}
    )
    assert [(g.grouper, g.count) for g in groups] == [("missing", 2)]


def test_JIN_N14_counted_path_propagates_strict_undefined():
    env = Environment(undefined=StrictUndefined)
    with pytest.raises(UndefinedError):
        env.call_filter("groupby", [{}, {}], ["kind"], {"with_counts": True})
