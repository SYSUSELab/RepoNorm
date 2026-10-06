import asyncio
from dataclasses import dataclass

import dspy
import pytest
from dspy.clients.cache import Cache, request_cache
from dspy.utils.usage_tracker import track_usage


@dataclass
class Response:
    value: str
    usage: dict


def make_cache(tmp_path, *, disk=False):
    return Cache(disk, True, str(tmp_path), memory_max_entries=20)


def test_DSP_3ded7a626651_N001_stats_keep_hit_lifecycle(tmp_path):  # noqa: N802
    cache = make_cache(tmp_path, disk=True)
    request = {"value": 1}
    original = Response("stored", {"tokens": 2})
    cache.put(request, original)
    hit = cache.get(request)
    assert hit == Response("stored", {}) and hit is not original
    cache.reset_memory_cache()
    assert cache.get(request) == Response("stored", {})
    assert cache.cache_key(request) in cache.memory_cache


@pytest.mark.asyncio
async def test_DSP_3ded7a626651_N002_stats_keep_request_cache(tmp_path):  # noqa: N802
    cache = make_cache(tmp_path)
    calls = []

    @request_cache()
    async def compute(value):
        calls.append(value)
        await asyncio.sleep(0)
        return len(calls)

    original = dspy.cache
    dspy.cache = cache
    try:
        assert await compute("same") == await compute("same") == 1
    finally:
        dspy.cache = original
    assert calls == ["same"]


@pytest.mark.asyncio
async def test_DSP_3ded7a626651_N003_call_tracking_is_isolated(tmp_path):  # noqa: N802
    cache = make_cache(tmp_path)
    original = dspy.cache
    dspy.cache = cache

    class Read(dspy.Module):
        async def aforward(self, count):
            await asyncio.sleep(0)
            for index in range(count):
                cache.get({"missing": index})
            return dspy.Prediction(count=count)

    try:
        with dspy.settings.context(track_usage=True):
            one, two = await asyncio.gather(Read().acall(1), Read().acall(2))
    finally:
        dspy.cache = original
    assert one.get_cache_usage()["misses"] == 1
    assert two.get_cache_usage()["misses"] == 2


def test_DSP_3ded7a626651_N010_lm_usage_semantics_are_preserved():  # noqa: N802
    class Plain(dspy.Module):
        def forward(self):
            return dspy.Prediction(answer="ok")

    with dspy.settings.context(track_usage=True):
        direct = Plain()()
        with track_usage() as outer:
            nested = Plain()()
    assert direct.get_lm_usage() == {}
    assert nested.get_lm_usage() is None
    assert outer.get_total_tokens() == {}
