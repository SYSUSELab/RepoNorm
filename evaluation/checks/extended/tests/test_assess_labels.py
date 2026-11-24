"""Offline reporting tests; no Docker, source clone or API required."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("task_norm_assess", Path(__file__).parents[1] / "assess.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compatibility_reason_tracks_task_version_and_profile():
    for version in ("3.7", "3.9"):
        data = {
            "tasks": [{"id": "example", "minimum_python": version, "norms": [{"norm_id": "N"}]}],
            "norms": [{"id": "N", "oracle_kind": "compatibility", "strength": "active_configuration"}],
        }
        record = {
            "task": "example", "state": "gold", "patch_sha256": "digest",
            "container_exit_code": 0,
            "dynamic": {"syntax": {"status": "pass"}, "functional": {"status": "pass"}, "regression": {"status": "pass"}},
        }
        cards = {"digest": {"compatibility": {"status": "pass"}}}
        for profile in ("strict", "coarse_auto"):
            value = MODULE.assess(record, profile, data, cards)["norms"]["N"]
            assert value["status"] == "pass"
            expected = "python_{}_syntax".format(version.replace(".", "_"))
            if profile == "strict":
                expected += "_plus_patch_bound_API_review"
            assert value["reason"] == expected
