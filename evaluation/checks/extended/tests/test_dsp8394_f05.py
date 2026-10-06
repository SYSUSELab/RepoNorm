from dataclasses import dataclass

import dspy
import pytest
from dspy.clients import configure_cache
from dspy.clients.cache import Cache, request_cache


@dataclass
class Response:
    value: str
    usage: dict


def test_DSP_3ded7a626651_N001_compression_keeps_hit_lifecycle(tmp_path):  # noqa: N802
    cache = Cache(True, True, str(tmp_path), memory_max_entries=10, compress="gzip")
    request = {"value": 1}
    original = Response("stored" * 50, {"tokens": 2})
    cache.put(request, original)
    hit = cache.get(request)
    assert hit == Response("stored" * 50, {}) and hit is not original
    cache.reset_memory_cache()
    assert cache.get(request) == Response("stored" * 50, {})
    assert cache.cache_key(request) in cache.memory_cache


def test_DSP_3ded7a626651_N002_compression_keeps_request_cache(tmp_path):  # noqa: N802
    cache = Cache(True, False, str(tmp_path), compress="zlib")
    calls = []

    @request_cache()
    def compute(value):
        calls.append(value)
        return {"call": len(calls)}

    original = dspy.cache
    dspy.cache = cache
    try:
        assert compute("same") == compute("same") == {"call": 1}
    finally:
        dspy.cache = original
    assert calls == ["same"]


def test_DSP_3ded7a626651_N008_compression_keeps_configure_contract(tmp_path):  # noqa: N802
    with pytest.raises(ValueError):
        configure_cache(enable_disk_cache=True, enable_litellm_cache=True)
    original = dspy.cache
    try:
        configure_cache(enable_disk_cache=False, disk_cache_dir=str(tmp_path), compress="gzip")
        assert dspy.cache.enable_disk_cache is False
    finally:
        dspy.cache = original


def test_DSP_3ded7a626651_N009_membership_tracks_compressed_disk_layer(tmp_path):  # noqa: N802
    cache = Cache(True, True, str(tmp_path), memory_max_entries=10, compress="gzip")
    request = {"value": 1}
    cache.put(request, "stored")
    key = cache.cache_key(request)
    cache.reset_memory_cache()
    assert key in cache
    assert cache.get(request) == "stored"
