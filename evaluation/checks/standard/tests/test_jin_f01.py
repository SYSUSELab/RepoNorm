import asyncio

import pytest

from jinja2 import Environment
from jinja2 import StrictUndefined
from jinja2.exceptions import UndefinedError


class AliasEnvironment(Environment):
    def getitem(self, obj, argument):
        if argument == "alias":
            argument = "key"
        return super().getitem(obj, argument)


def test_JIN_N01_environment_lookup_strategy():
    env = AliasEnvironment()
    rows = [{"key": "same"}, {"key": "same"}]
    groups = env.call_filter("groupby", rows, ["alias"])
    assert len(groups) == 1
    assert groups[0].grouper == "same"


def test_JIN_N11_group_result_interface():
    env = Environment()
    rows = [{"key": "same"}, {"key": "same"}]
    group = env.call_filter("groupby", rows, ["key"])[0]
    assert tuple(group) == ("same", rows)
    assert group.grouper == "same"
    assert group.list == rows
    assert repr(group) == repr(tuple(group))


def test_JIN_N12_async_iterable_materialized_groups():
    env = Environment(enable_async=True)

    async def rows():
        for key in ["same", "same"]:
            yield {"key": key}

    groups = asyncio.run(env.call_filter("groupby", rows(), ["key"]))
    assert [group.grouper for group in groups] == ["same"]
    assert [len(group.list) for group in groups] == [2]


def test_JIN_N13_default_for_missing_attribute():
    env = Environment()
    groups = env.call_filter("groupby", [{}, {}], ["key"], {"default": "missing"})
    assert len(groups) == 1
    assert groups[0].grouper == "missing"


def test_JIN_N14_strict_undefined_error_preserved():
    env = Environment(undefined=StrictUndefined)
    with pytest.raises(UndefinedError):
        env.call_filter("groupby", [{}, {}], ["key"])
