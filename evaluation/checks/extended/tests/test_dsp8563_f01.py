import copy

from dspy.adapters.types.tool import ToolCalls


def test_DSP_80412ce96d70_N001_format_contract():  # noqa: N802
    calls = ToolCalls.from_dict_list([{"name": "search", "args": {"q": "x"}}])
    assert calls.format() == [
        {
            "type": "tool_calls",
            "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"q": "x"}}}],
        }
    ]


def test_DSP_80412ce96d70_N002_construction_preserves_order_and_values():  # noqa: N802
    source = [{"name": "a", "args": {"nested": [1]}}, {"name": "b", "args": {}}]
    before = copy.deepcopy(source)
    calls = ToolCalls.from_dict_list(source)
    assert source == before
    assert [(item.name, item.args) for item in calls.tool_calls] == [("a", {"nested": [1]}), ("b", {})]
