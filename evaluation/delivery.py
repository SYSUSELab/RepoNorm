"""Total coarse decisions for observable contribution-delivery signals.

The coarse authored-test oracle deliberately does not try to prove that a test
is semantically relevant or that it fails on the base snapshot.  Those are
strict-oracle concerns.  It asks only whether the submitted patch contains a
recognised author-owned test change and whether the selected tests execute and
pass on that patch.
"""
from __future__ import annotations

from typing import Any, Mapping


def assess_authored_test_delivery(dynamic: Mapping[str, Any]) -> dict[str, Any]:
    """Return a total pass/fail coarse verdict for authored-test delivery."""
    def result(status: str, reason: str) -> dict[str, Any]:
        return {
            "status": status,
            "reason": reason,
            "protocol": "authored-test-delivery-1.0",
        }

    authored = dynamic.get("authored_paths")
    if not authored:
        return result("fail", "no_author_owned_test_change")

    selection = dynamic.get("authored_selection", {})
    if selection.get("status") != "pass":
        return result("fail", "authored_test_selection_not_completed")
    nodeids = selection.get("nodeids") or []
    changed_paths = selection.get("changed_test_paths") or selection.get("paths") or []
    if not nodeids or not changed_paths:
        return result("fail", "no_recognized_changed_test_nodes")
    if not set(changed_paths).issubset(set(authored)):
        return result("fail", "selected_tests_not_bound_to_authored_paths")

    green = dynamic.get("author_green", {})
    if green.get("status") != "pass":
        return result("fail", "authored_tests_do_not_pass_on_patch")
    if green.get("counts", {}).get("tests", 0) <= 0:
        return result("fail", "authored_test_execution_has_no_tests")
    return result("pass", "recognized_authored_tests_pass_on_patch")
