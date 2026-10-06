from dataclasses import dataclass
from unittest.mock import patch

import dspy
import pytest
from dspy.clients import configure_cache
from dspy.clients.cache import Cache, request_cache


@dataclass
class Response:
    value: str
    usage: dict


def test_DSP_3ded7a626651_N001_ttl_keeps_hit_lifecycle(tmp_path):  # noqa: N802
    cache = Cache(True, True, str(tmp_path), memory_max_entries=10)
    request = {"value": 1}
    original = Response("stored", {"tokens": 2})
    cache.put(request, original)
    hit = cache.get(request)
    assert hit == Response("stored", {}) and hit is not original
    cache.reset_memory_cache()
    assert cache.get(request) == Response("stored", {})
    assert cache.cache_key(request) in cache.memory_cache


def test_DSP_3ded7a626651_N002_ttl_keeps_request_cache(tmp_path):  # noqa: N802
    cache = Cache(False, True, str(tmp_path), memory_max_entries=10)
    calls = []

    @request_cache()
    def compute(value):
        calls.append(value)
        return len(calls)

    with patch("dspy.cache", cache):
        assert compute("same") == compute("same") == 1


def test_DSP_3ded7a626651_N008_ttl_keeps_configure_contract(tmp_path):  # noqa: N802
    with pytest.raises(ValueError):
        configure_cache(enable_disk_cache=True, enable_litellm_cache=True)
    with patch.object(dspy, "cache", None):
        configure_cache(enable_disk_cache=False, disk_cache_dir=str(tmp_path), ttl=None)
        assert dspy.cache.enable_disk_cache is False


def test_DSP_3ded7a626651_N009_membership_tracks_enabled_layers(tmp_path):  # noqa: N802
    request = {"value": 1}
    cache = Cache(True, True, str(tmp_path), memory_max_entries=10, ttl=None)
    cache.put(request, "stored")
    assert cache.cache_key(request) in cache
    cache.reset_memory_cache()
    assert cache.cache_key(request) in cache
