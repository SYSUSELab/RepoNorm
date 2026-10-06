import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from dspy.clients.cache import Cache, request_cache


@dataclass
class Response:
    value: str
    usage: dict


def make_cache(tmp_path, name="cache", *, disk=True):
    return Cache(disk, True, str(tmp_path / name), memory_max_entries=20)


def test_DSP_3ded7a626651_N001_preserves_normal_hit_lifecycle(tmp_path):  # noqa: N802
    cache = make_cache(tmp_path)
    request = {"value": 1}
    original = Response("stored", {"tokens": 2})
    cache.put(request, original)

    hit = cache.get(request)
    assert hit == Response("stored", {})
    assert hit is not original
    assert original.usage == {"tokens": 2}

    cache.reset_memory_cache()
    disk_hit = cache.get(request)
    assert disk_hit == Response("stored", {})
    assert cache.cache_key(request) in cache.memory_cache


@pytest.mark.asyncio
async def test_DSP_3ded7a626651_N002_preserves_sync_and_async_decorators(tmp_path):  # noqa: N802
    cache = make_cache(tmp_path, disk=False)
    sync_calls = []
    async_calls = []

    @request_cache()
    def sync_fn(*, value):
        sync_calls.append(value)
        return len(sync_calls)

    @request_cache()
    async def async_fn(*, value):
        async_calls.append(value)
        return len(async_calls)

    with patch("dspy.cache", cache):
        assert sync_fn(value="sync") == sync_fn(value="sync") == 1
        assert await async_fn(value="async") == 1
        assert await async_fn(value="async") == 1


@pytest.mark.asyncio
async def test_DSP_3ded7a626651_N003_read_control_does_not_leak(tmp_path):  # noqa: N802
    cache = make_cache(tmp_path, disk=False)
    request = {"value": 1}
    cache.put(request, "stored")
    sibling_ready = asyncio.Event()
    sibling_go = asyncio.Event()

    async def sibling():
        sibling_ready.set()
        await sibling_go.wait()
        return cache.get(request)

    task = asyncio.create_task(sibling())
    await sibling_ready.wait()
    with cache.bypass():
        assert cache.get(request) is None
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(cache.get, request).result() == "stored"
        sibling_go.set()
        assert await task == "stored"
    assert cache.get(request) == "stored"
