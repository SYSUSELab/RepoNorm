import asyncio

import pytest

from jinja2 import Environment
from jinja2 import StrictUndefined
from jinja2 import UndefinedError


def test_JIN_N01_filtered_path_uses_environment_lookup():
    class PrefixEnvironment(Environment):
        def getitem(self, obj, argument):
            return f"env:{super().getitem(obj, argument)}"

    groups = PrefixEnvironment().call_filter(
        "groupby",
        [{"kind": "x"}],
        ["kind"],
        {"filter_groups": lambda items: True},
    )
    assert groups[0].grouper == "env:x"


def test_JIN_N11_filtered_path_preserves_group_interface():
    rows = [{"kind": "x"}, {"kind": "x"}]
    group = Environment().call_filter(
        "groupby", rows, ["kind"], {"filter_groups": lambda items: True}
    )[0]
    assert tuple(group) == ("x", rows)
    assert group.grouper == "x"
    assert group.list == rows
    assert repr(group) == repr(("x", rows))


def test_JIN_N12_filtered_path_materializes_async_input():
    async def rows():
        for value in [{"kind": "x"}, {"kind": "x"}]:
            yield value

    async def run():
        return await Environment(enable_async=True).call_filter(
            "groupby", rows(), ["kind"], {"filter_groups": lambda items: True}
        )

    assert len(asyncio.run(run())[0].list) == 2


def test_JIN_N13_filtered_path_preserves_default():
    groups = Environment().call_filter(
        "groupby",
        [{}, {}],
        ["kind"],
        {"default": "missing", "filter_groups": lambda items: True},
    )
    assert groups[0].grouper == "missing"


def test_JIN_N14_filtered_path_propagates_strict_undefined():
    env = Environment(undefined=StrictUndefined)
    with pytest.raises(UndefinedError):
        env.call_filter(
            "groupby", [{}, {}], ["kind"], {"filter_groups": lambda items: True}
        )
