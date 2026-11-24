import json

from pathlib import Path

def tool_status(checks):
    if not checks or any(check.get("status") == "error" for check in checks):
        return "error"
    if any(check.get("exit_code") is None for check in checks):
        return "error"
    return "pass" if all(check["exit_code"] == 0 for check in checks) else "fail"

def review_status(cards, digest, key):
    value = cards.get(digest, {}).get(key, {}).get("status", "review_required")
    if value not in {"pass", "fail", "error", "review_required"}:
        return "error"
    return value

def authored_status(record, profile, cards):
    dynamic = record["dynamic"]
    if not dynamic.get("authored_paths"):
        return "fail", "no_authored_tests"
    green = dynamic.get("author_green", {}).get("status")
    red = dynamic.get("author_red", {}).get("status")
    if green == "error" or red == "error":
        return "error", "invalid_red_green_execution"
    if green != "pass" or red != "fail":
        return "fail", "authored_tests_do_not_form_red_green"
    if profile == "coarse_auto":
        return "pass", "automatic_red_green_signal"
    return (
        review_status(cards, record["patch_sha256"], "authored_tests"),
        "red_green_plus_patch_bound_relevance_review",
    )

def overall(statuses):
    for status in ("error", "fail", "review_required"):
        if status in statuses:
            return status
    return "pass"

def assess(record, profile, data, cards):
    if profile not in {"strict", "coarse_auto"}:
        raise ValueError(profile)
    task = next(task for task in data["tasks"] if task["id"] == record["task"])
    selected = {link["norm_id"] for link in task["norms"]}
    container_exit = record.get("container_exit_code")
    if type(container_exit) is not int or container_exit != 0:
        reason = (
            "missing_container_exit_attestation"
            if "container_exit_code" not in record
            else "invalid_container_execution"
        )
        return {
            "task": record["task"],
            "state": record["state"],
            "profile": profile,
            "patch_sha256": record["patch_sha256"],
            "functional": "error",
            "regression": "error",
            "norms": {
                norm["id"]: {
                    "status": "error",
                    "kind": norm["oracle_kind"],
                    "strength": norm["strength"],
                    "reason": reason,
                }
                for norm in data["norms"] if norm["id"] in selected
            },
            "overall": "error",
        }
    result = {}
    for norm in data["norms"]:
        if norm["id"] not in selected:
            continue
        kind = norm["oracle_kind"]
        reason = None
        if kind in {"runtime", "public_interface"}:
            status = record["dynamic"].get("norms", {}).get(norm["id"], {}).get("status", "error")
        elif kind == "authored_tests":
            status, reason = authored_status(record, profile, cards)
        elif kind == "docs":
            signal = record["dynamic"].get("docs_signal", {}).get("status", "error")
            if signal != "pass":
                status, reason = signal, "no_recognized_documentation_change"
            elif profile == "coarse_auto":
                status, reason = "pass", "documentation_change_signal_only"
            else:
                status = review_status(cards, record["patch_sha256"], "docs")
                reason = "patch_bound_content_review"
        elif kind in {"changelog", "inline_change"}:
            signal = record["dynamic"].get(f"{kind}_signal", {}).get("status", "error")
            if signal != "pass":
                status, reason = signal, f"no_{kind}_change_signal"
            elif profile == "coarse_auto":
                status, reason = "pass", f"{kind}_change_signal_only"
            else:
                status = review_status(cards, record["patch_sha256"], kind)
                reason = "patch_bound_content_review"
        elif kind == "style":
            status = tool_status(record["static"]["style"])
        elif kind == "typing":
            status = tool_status(record["static"]["typing"])
        elif kind == "compatibility":
            status = record["dynamic"].get("syntax", {}).get("status", "error")
            if status == "pass" and profile == "strict":
                status = review_status(cards, record["patch_sha256"], "compatibility")
            reason = "python_{}_syntax".format(task["minimum_python"].replace(".", "_"))
            if profile == "strict":
                reason += "_plus_patch_bound_API_review"
        else:
            status, reason = "error", "unsupported_oracle_kind"
        result[norm["id"]] = {
            "status": status,
            "kind": kind,
            "strength": norm["strength"],
            "reason": reason,
        }
    return {
        "task": record["task"],
        "state": record["state"],
        "profile": profile,
        "patch_sha256": record["patch_sha256"],
        "functional": record["dynamic"].get("functional", {}).get("status", "error"),
        "regression": record["dynamic"].get("regression", {}).get("status", "error"),
        "norms": result,
        "overall": overall({item["status"] for item in result.values()}),
    }
