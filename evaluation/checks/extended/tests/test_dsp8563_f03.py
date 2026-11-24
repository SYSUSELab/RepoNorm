import asyncio

import pytest
from dspy.adapters.types.tool import Tool


def test_DSP_80412ce96d70_N003_sync_async_validation_contract():  # noqa: N802
    def add(value: int):
        return value + 1

    tool = Tool(add)
    assert tool(value=2) == 3
    assert asyncio.run(tool.acall(value=2)) == 3
    with pytest.raises(ValueError):
        tool(value="not-an-integer")
