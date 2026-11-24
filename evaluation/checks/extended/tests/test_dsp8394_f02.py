import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest

import dspy
from dspy.clients import configure_cache
from dspy.clients.cache import Cache, request_cache


@dataclass
class Response:
    value: str
    usage: dict


def make_cache(tmp_path, *, disk=False):
    return Cache(disk, True, str(tmp_path), memory_max_entries=20)


def test_DSP_3ded7a626651_N001_namespace_keeps_normal_hits(tmp_path):  # noqa: N802
    cache = make_cache(tmp_path, disk=True)
    request = {"value": 1}
    original = Response("stored", {"tokens": 2})
    cache.put(request, original)
    hit = cache.get(request)
    assert hit == Response("stored", {})
    assert hit is not original
    cache.reset_memory_cache()
    assert cache.get(request) == Response("stored", {})
    assert cache.cache_key(request) in cache.memory_cache


@pytest.mark.asyncio
async def test_DSP_3ded7a626651_N002_namespace_keeps_wrappers(tmp_path):  # noqa: N802
    cache = make_cache(tmp_path)
    calls = []

    @request_cache()
    async def compute(value):
        calls.append(value)
        await asyncio.sleep(0)
        return len(calls)

    with patch("dspy.cache", cache):
        assert await compute("same") == 1
        assert await compute("same") == 1
    assert calls == ["same"]


@pytest.mark.asyncio
async def test_DSP_3ded7a626651_N003_namespace_does_not_leak(tmp_path):  # noqa: N802
    cache = make_cache(tmp_path)
    ready = asyncio.Event()
    release = asyncio.Event()

    async def sibling():
        ready.set()
        await release.wait()
        return cache.namespace.value

    task = asyncio.create_task(sibling())
    await ready.wait()
    with cache.namespace("parent"):
        release.set()
        assert await task is None
        assert cache.namespace == "parent"
    assert cache.namespace.value is None


def test_DSP_3ded7a626651_N008_configure_contract(tmp_path):  # noqa: N802
    with pytest.raises(ValueError):
        configure_cache(enable_disk_cache=True, enable_litellm_cache=True)
    with patch.object(dspy, "cache", None):
        configure_cache(enable_disk_cache=False, enable_memory_cache=True, disk_cache_dir=str(tmp_path))
        assert dspy.cache.enable_disk_cache is False
        assert dspy.cache.enable_memory_cache is True
