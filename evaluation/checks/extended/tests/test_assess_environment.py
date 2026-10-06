"""Offline infrastructure tests: complete child results cannot hide failure."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "environment_assess", Path(__file__).parents[1] / "assess.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example():
    data = {
        "tasks": [{"id": "example", "norms": [{"norm_id": "N"}]}],
        "norms": [{"id": "N", "oracle_kind": "runtime", "strength": "current_source_contract"}],
    }
    record = {
        "task": "example", "state": "gold", "patch_sha256": "digest",
        "container_exit_code": 0,
        "dynamic": {
            "norms": {"N": {"status": "pass"}},
            "functional": {"status": "pass"},
            "regression": {"status": "pass"},
        },
    }
    return data, record


@pytest.mark.parametrize("profile", ["strict", "coarse_auto"])
@pytest.mark.parametrize("exit_code", [None, 1, -15, False, "0"])
def test_invalid_container_overrides_complete_child_results(profile, exit_code):
    data, record = example()
    record["container_exit_code"] = exit_code
    result = MODULE.assess(record, profile, data, {})
    assert result["overall"] == result["functional"] == result["regression"] == "error"
    assert result["norms"]["N"]["status"] == "error"
    assert result["norms"]["N"]["reason"] == "invalid_container_execution"


@pytest.mark.parametrize("profile", ["strict", "coarse_auto"])
def test_missing_attestation_does_not_silently_accept_legacy_receipt(profile):
    data, record = example()
    del record["container_exit_code"]
    del record["dynamic"]
    result = MODULE.assess(record, profile, data, {})
    assert result["overall"] == "error"
    assert result["norms"]["N"]["reason"] == "missing_container_exit_attestation"


@pytest.mark.parametrize("profile", ["strict", "coarse_auto"])
def test_successful_container_keeps_functional_and_norm_failures_distinct(profile):
    data, record = example()
    assert MODULE.assess(record, profile, data, {})["overall"] == "pass"
    record["dynamic"]["functional"]["status"] = "fail"
    assert MODULE.assess(record, profile, data, {})["overall"] == "pass"
    record["dynamic"]["norms"]["N"]["status"] = "fail"
    result = MODULE.assess(record, profile, data, {})
    assert result["functional"] == result["overall"] == "fail"
