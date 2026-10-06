from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3.unified.git_evidence import (
    BoundedGitRunner,
    RunLocalGitRunner,
    GitAnchor,
    GitChangeSlice,
    GitLimits,
    GitProcessOutcome,
    GitQueryManifest,
    _make_bundle,
    _make_slice,
    _make_topology,
    _manifest,
    _slice_selection,
    _anchor_hits_changed_line,
    _change_kind,
    _changed_source_lines,
    _RawChange,
    collect_git_evidence,
    plan_git_queries,
    preflight_git_repository,
    run_git_queries,
)


_SNAPSHOT = "sha256:" + "1" * 64


@dataclass(frozen=True)
class GitDag:
    root: Path
    commits: dict[str, str]


def _git(
    root: Path,
    *argv: str,
    date: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = {
        "HOME": root.as_posix(),
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    if date is not None:
        environment["GIT_AUTHOR_DATE"] = date
        environment["GIT_COMMITTER_DATE"] = date
    return subprocess.run(
        ["git", "-C", root.as_posix(), *argv],
        check=check,
        capture_output=True,
        text=True,
        env=environment,
    )


def _commit(root: Path, label: str, date: str) -> str:
    _git(root, "add", "--all")
    _git(
        root,
        "-c",
        "user.name=Norm4Code Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--no-gpg-sign",
        "-m",
        label,
        date=date,
    )
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def _write_rule(
    root: Path,
    *,
    rule: str | None,
    condition: str,
    guard: bool,
    path: str = "src/rule.py",
) -> None:
    lines: list[str] = []
    if guard:
        lines.append("LEGACY_GUARD = True")
    if rule is not None:
        lines.append(f'RULE = "{rule}"')
    lines.extend(
        [
            "",
            "def apply(enabled: bool, safe: bool) -> str:",
            f"    if {condition}:",
            '        return "enforced"',
            '    return "fallback"',
            "",
        ]
    )
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def _build_git_dag(tmp_path: Path) -> GitDag:
    root = tmp_path / "history-repository"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Norm4Code Fixture")
    _git(root, "config", "user.email", "fixture@example.invalid")
    commits: dict[str, str] = {}

    _write_rule(root, rule=None, condition="enabled", guard=True)
    commits["root"] = _commit(root, "root", "2001-01-01T00:00:00+0000")

    _write_rule(root, rule="strict", condition="enabled", guard=True)
    commits["introduction"] = _commit(
        root, "introduce strict rule", "2001-01-02T00:00:00+0000"
    )

    _write_rule(root, rule="lenient", condition="enabled", guard=True)
    commits["contradiction"] = _commit(
        root, "contradict strict rule", "2001-01-03T00:00:00+0000"
    )

    _write_rule(root, rule="lenient", condition="enabled and safe", guard=True)
    commits["condition"] = _commit(
        root, "narrow condition", "2001-01-04T00:00:00+0000"
    )

    _write_rule(root, rule="lenient", condition="enabled and safe", guard=False)
    commits["removal"] = _commit(
        root, "remove legacy guard", "2001-01-05T00:00:00+0000"
    )
    _git(
        root,
        "-c",
        "user.name=Norm4Code Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "revert",
        "--no-edit",
        commits["removal"],
        date="2001-01-06T00:00:00+0000",
    )
    commits["revert"] = _git(root, "rev-parse", "HEAD").stdout.strip()

    _git(root, "mv", "src/rule.py", "src/policy.py")
    commits["rename"] = _commit(
        root, "rename rule module", "2001-01-07T00:00:00+0000"
    )
    _git(root, "branch", "feature")

    _write_rule(
        root,
        rule="main",
        condition="enabled and safe",
        guard=True,
        path="src/policy.py",
    )
    # Deliberately later timestamp than the incomparable feature commit.
    commits["main_branch"] = _commit(
        root, "main branch rule", "2001-01-10T00:00:00+0000"
    )

    _git(root, "checkout", "feature")
    _write_rule(
        root,
        rule="feature",
        condition="enabled and safe",
        guard=True,
        path="src/policy.py",
    )
    commits["feature_branch"] = _commit(
        root, "feature branch rule", "2001-01-08T00:00:00+0000"
    )

    _git(root, "checkout", "main")
    merge = _git(root, "merge", "--no-commit", "feature", check=False)
    assert merge.returncode != 0
    _write_rule(
        root,
        rule="resolved",
        condition="enabled and safe",
        guard=True,
        path="src/policy.py",
    )
    commits["merge_resolution"] = _commit(
        root, "resolve rule merge", "2001-01-11T00:00:00+0000"
    )
    return GitDag(root=root, commits=commits)


def _build_mixed_merge_dag(tmp_path: Path) -> GitDag:
    """Create a merge with one resolution and changes inherited per parent."""

    root = tmp_path / "mixed-merge-repository"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Norm4Code Fixture")
    _git(root, "config", "user.email", "fixture@example.invalid")
    commits: dict[str, str] = {}

    _write_rule(
        root,
        rule="base",
        condition="enabled",
        guard=True,
        path="src/policy.py",
    )
    (root / "src/main-only.txt").write_text("base\n", encoding="utf-8")
    (root / "src/feature-only.txt").write_text("base\n", encoding="utf-8")
    (root / "src/main-delete.txt").write_text("delete on main\n", encoding="utf-8")
    (root / "src/feature-delete.txt").write_text(
        "delete on feature\n",
        encoding="utf-8",
    )
    commits["root"] = _commit(root, "mixed merge root", "2002-01-01T00:00:00+0000")
    _git(root, "branch", "feature")

    _write_rule(
        root,
        rule="main",
        condition="enabled",
        guard=True,
        path="src/policy.py",
    )
    (root / "src/main-only.txt").write_text("main\n", encoding="utf-8")
    (root / "src/main-delete.txt").unlink()
    commits["main"] = _commit(root, "main changes", "2002-01-02T00:00:00+0000")

    _git(root, "checkout", "feature")
    _write_rule(
        root,
        rule="feature",
        condition="enabled",
        guard=True,
        path="src/policy.py",
    )
    (root / "src/feature-only.txt").write_text("feature\n", encoding="utf-8")
    (root / "src/feature-delete.txt").unlink()
    commits["feature"] = _commit(
        root,
        "feature changes",
        "2002-01-03T00:00:00+0000",
    )

    _git(root, "checkout", "main")
    merge = _git(root, "merge", "--no-commit", "feature", check=False)
    assert merge.returncode != 0
    _write_rule(
        root,
        rule="resolved",
        condition="enabled",
        guard=True,
        path="src/policy.py",
    )
    commits["merge"] = _commit(
        root,
        "resolve mixed merge",
        "2002-01-04T00:00:00+0000",
    )
    return GitDag(root=root, commits=commits)


def _full_limits(**updates: int) -> GitLimits:
    values = {
        "max_results_per_query": 64,
        "max_results_per_family": 128,
        "max_total_candidate_commits": 128,
        "max_change_slices": 256,
        "max_total_patch_bytes": 20_000_000,
        "max_patch_bytes": 2_000_000,
    }
    values.update(updates)
    return GitLimits(**values)


def _anchors() -> list[GitAnchor]:
    return [
        GitAnchor(
            anchor_id="ANCHOR-rule",
            kind="symbol",
            path="src/policy.py",
            value="RULE",
            origin_ids=["ORIGIN-rule"],
            precise_origin_ids=["ORIGIN-rule"],
        ),
        GitAnchor(
            anchor_id="ANCHOR-current-line",
            kind="line",
            path="src/policy.py",
            start_line=1,
            end_line=8,
            origin_ids=["ORIGIN-rule"],
        ),
    ]


def test_run_local_raw_reuse_preserves_each_case_query_plan_and_slices(tmp_path: Path) -> None:
    dag = _build_git_dag(tmp_path)
    cutoff = dag.commits["merge_resolution"]
    limits = _full_limits()
    runner = RunLocalGitRunner(BoundedGitRunner(), snapshot_digest=_SNAPSHOT)
    preflight = preflight_git_repository(dag.root, cutoff, limits=limits, runner=runner)
    bundles = [
        collect_git_evidence(
            dag.root, case_id=case_id, snapshot_digest=_SNAPSHOT,
            cutoff_commit=cutoff, purpose="augmentative", anchors=_anchors(),
            limits=limits, runner=runner, preflight=preflight,
        )
        for case_id in ("CASE-first", "CASE-second")
    ]
    first, second = bundles
    assert first.raw_execution.physical_calls > 0
    assert second.raw_execution.physical_calls == 0
    assert second.raw_execution.logical_calls == first.raw_execution.logical_calls
    assert second.raw_execution.cache_hits == second.raw_execution.logical_calls
    assert first.manifest.commands == second.manifest.commands
    assert first.manifest.commit_ids == second.manifest.commit_ids
    assert first.status == second.status == "complete"
    assert first.case_id != second.case_id
    assert first.manifest.manifest_id != second.manifest.manifest_id
    assert [(item.commit_id, item.parent_id, item.hunk) for item in first.slices] == [
        (item.commit_id, item.parent_id, item.hunk) for item in second.slices
    ]


def test_full_dag_yields_exact_slices_dedup_and_topology(tmp_path: Path) -> None:
    dag = _build_git_dag(tmp_path)
    cutoff = dag.commits["merge_resolution"]

    bundle = collect_git_evidence(
        dag.root,
        case_id="CASE-history",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=cutoff,
        purpose="augmentative",
        anchors=_anchors(),
        limits=_full_limits(),
    )

    assert bundle.preflight.status == "complete"
    assert bundle.manifest.status == "complete"
    assert bundle.status == "complete"
    assert {item.family for item in bundle.manifest.family_results} == {
        "path_history",
        "blame",
        "pickaxe_literal",
        "pickaxe_regex",
        "rename_tracking",
    }
    assert len({item.item_id for item in bundle.slices}) == len(bundle.slices)

    by_commit: dict[str, list] = {}
    for item in bundle.slices:
        by_commit.setdefault(item.commit_id, []).append(item)
    introduction_items = by_commit[dag.commits["introduction"]]
    assert {
        item.commit_message_context_status for item in introduction_items
    } == {"complete"}
    assert {
        item.commit_message_context for item in introduction_items
    } == {"introduce strict rule"}
    assert {
        item.commit_message_context_sha256 for item in introduction_items
    } == {
        hashlib.sha256(b"introduce strict rule").hexdigest()
    }
    assert dag.commits["root"] not in by_commit
    assert any(
        'RULE = "strict"' in (item.hunk or "")
        for item in by_commit[dag.commits["introduction"]]
    )
    assert any(
        '-RULE = "strict"' in (item.hunk or "")
        and '+RULE = "lenient"' in (item.hunk or "")
        for item in by_commit[dag.commits["contradiction"]]
    )
    # Context lines and historical blame line numbers cannot admit a hunk:
    # these commits do not change the exact RULE symbol itself.
    assert dag.commits["condition"] not in by_commit
    assert dag.commits["removal"] not in by_commit
    assert dag.commits["revert"] not in by_commit
    rename_items = by_commit[dag.commits["rename"]]
    assert any(item.change_kind == "rename" for item in rename_items)
    assert all(item.selection_reason == "direct_pure_rename" for item in rename_items)
    assert all(item.hunk is None for item in rename_items)
    assert all("rename_tracking" in item.query_families for item in rename_items)
    rename_item = rename_items[0]
    copy_change = _RawChange(
        parent_id=rename_item.parent_id,
        old_blob_id=rename_item.old_blob_id,
        new_blob_id=rename_item.new_blob_id,
        status="C",
        before_path=rename_item.before_path,
        after_path=rename_item.after_path,
    )
    local_results = [
        result
        for result in bundle.manifest.command_results
        if rename_item.commit_id in result.commit_ids
        and result.family == "rename_tracking"
    ]
    retained, _, _, filtered = _slice_selection(
        change=copy_change,
        hunk=None,
        local_results=local_results,
        anchor_by_id={
            item.anchor_id: item for item in bundle.manifest.anchors
        },
    )
    assert retained is None
    assert filtered == "weak_path_only_without_changed_line_anchor_match"
    assert any(
        item.comparison_kind == "merge_resolution"
        and 'RULE = "resolved"' in (item.hunk or "")
        for item in by_commit[dag.commits["merge_resolution"]]
    )
    assert any(
        len(item.query_families) > 1
        for item in bundle.slices
    )

    assert bundle.topology.status == "complete"
    ranks = {
        commit_id: group.ordinal
        for group in bundle.topology.groups
        for commit_id in group.commit_ids
    }
    assert ranks[dag.commits["introduction"]] < ranks[dag.commits["rename"]]
    assert ranks[dag.commits["rename"]] < ranks[dag.commits["main_branch"]]
    assert ranks[dag.commits["rename"]] < ranks[dag.commits["feature_branch"]]
    assert ranks[dag.commits["main_branch"]] < ranks[dag.commits["merge_resolution"]]
    assert ranks[dag.commits["feature_branch"]] < ranks[dag.commits["merge_resolution"]]
    incomparable = {
        frozenset((item.left_commit_id, item.right_commit_id))
        for item in bundle.topology.incomparable_pairs
    }
    assert frozenset(
        (dag.commits["main_branch"], dag.commits["feature_branch"])
    ) in incomparable
    slice_ranks = [ranks[item.commit_id] for item in bundle.slices]
    assert slice_ranks == sorted(slice_ranks)


def test_mixed_merge_resolution_replays_per_path_conservatively(
    tmp_path: Path,
) -> None:
    from repo_constraint_memory_v3.unified.validation import (
        _git_object_replay_errors,
    )

    dag = _build_mixed_merge_dag(tmp_path)
    cutoff = dag.commits["merge"]
    bundle = collect_git_evidence(
        dag.root,
        case_id="CASE-mixed-merge",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=cutoff,
        purpose="augmentative",
        anchors=[
            GitAnchor(
                anchor_id="ANCHOR-policy",
                kind="symbol",
                path="src/policy.py",
                value="RULE",
                origin_ids=["ORIGIN-policy"],
                precise_origin_ids=["ORIGIN-policy"],
            ),
            GitAnchor(
                anchor_id="ANCHOR-path-specific-deletion",
                kind="path",
                path="src/feature-delete.txt",
                origin_ids=["ORIGIN-deletion"],
            ),
        ],
        limits=_full_limits(),
    )

    merge_slices = [item for item in bundle.slices if item.commit_id == cutoff]
    assert merge_slices
    assert {item.comparison_kind for item in merge_slices} == {"merge_resolution"}
    assert {item.after_path for item in merge_slices} == {"src/policy.py"}
    assert all(item.change_kind != "removal" for item in merge_slices)
    assert bundle.selection_audit.filter_reason_counts[
        "merge_inherited_or_ambiguous"
    ] > 0
    assert "merge_inherited_or_ambiguous" not in bundle.skipped_change_reasons
    assert _git_object_replay_errors(
        dag.root,
        cutoff,
        {bundle.bundle_id: bundle},
    ) == []

    # Coordinated resealing cannot turn a path inherited from one parent into
    # an independent merge-resolution event.
    original = merge_slices[0]
    first_parent = original.commit_parents[0]
    inherited_payload = original.model_dump(
        mode="python",
        exclude={"item_id", "artifact_digest"},
    )
    inherited_payload.update(
        {
            "change_kind": "modification",
            "before_path": "src/feature-only.txt",
            "after_path": "src/feature-only.txt",
            "old_blob_id": _git(
                dag.root,
                "rev-parse",
                f"{first_parent}:src/feature-only.txt",
            ).stdout.strip(),
            "new_blob_id": _git(
                dag.root,
                "rev-parse",
                f"{cutoff}:src/feature-only.txt",
            ).stdout.strip(),
        }
    )
    resealed_slice = _make_slice(inherited_payload)
    bundle_payload = bundle.model_dump(
        mode="python",
        exclude={"bundle_id", "artifact_digest"},
    )
    bundle_payload["slices"] = [
        (
            resealed_slice.model_dump(mode="python")
            if item["item_id"] == original.item_id
            else item
        )
        for item in bundle_payload["slices"]
    ]
    resealed_bundle = _make_bundle(bundle_payload)
    errors = _git_object_replay_errors(
        dag.root,
        cutoff,
        {resealed_bundle.bundle_id: resealed_bundle},
    )
    assert any("strong-slice selection/provenance differs" in item for item in errors)


def test_no_match_and_result_plus_one_truncation_are_typed(tmp_path: Path) -> None:
    dag = _build_git_dag(tmp_path)
    cutoff = dag.commits["merge_resolution"]
    preflight = preflight_git_repository(dag.root, cutoff, limits=_full_limits())
    no_match = run_git_queries(
        dag.root,
        case_id="CASE-no-match",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=cutoff,
        purpose="augmentative",
        anchors=[
            GitAnchor(
                anchor_id="ANCHOR-missing",
                kind="literal",
                path="src/does-not-exist.py",
                value="NEVER_PRESENT_NORM4CODE_LITERAL",
                origin_ids=["ORIGIN-missing"],
                precise_origin_ids=["ORIGIN-missing"],
            )
        ],
        preflight=preflight,
        limits=_full_limits(),
    )
    assert no_match.status == "bounded_no_match"
    assert not no_match.commit_ids
    terminal_bundle = collect_git_evidence(
        dag.root,
        case_id="CASE-no-match-terminal",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=cutoff,
        purpose="augmentative",
        anchors=[
            GitAnchor(
                anchor_id="ANCHOR-missing-terminal",
                kind="literal",
                path="src/does-not-exist.py",
                value="NEVER_PRESENT_NORM4CODE_LITERAL",
                origin_ids=["ORIGIN-missing-terminal"],
                precise_origin_ids=["ORIGIN-missing-terminal"],
            )
        ],
        preflight=preflight,
        limits=_full_limits(),
    )
    assert terminal_bundle.manifest.status == "bounded_no_match"
    topology_payload = terminal_bundle.topology.model_dump(
        mode="python", exclude={"artifact_digest"}
    )
    topology_payload["stdout_sha256"] = "f" * 64
    bundle_payload = terminal_bundle.model_dump(
        mode="python", exclude={"bundle_id", "artifact_digest"}
    )
    bundle_payload["topology"] = _make_topology(topology_payload).model_dump(
        mode="python"
    )
    with pytest.raises(ValidationError, match="no-match Git bundle topology"):
        _make_bundle(bundle_payload)

    truncated = collect_git_evidence(
        dag.root,
        case_id="CASE-truncated",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=cutoff,
        purpose="augmentative",
        anchors=_anchors(),
        limits=_full_limits(
            max_results_per_query=1,
            max_results_per_family=2,
            max_total_candidate_commits=3,
        ),
    )
    assert truncated.manifest.status == "partial"
    assert truncated.status == "partial"
    assert "git_query_result_limit" in truncated.manifest.reason_codes
    assert any(item.truncated_count > 0 for item in truncated.manifest.command_results)


def test_change_slice_cap_is_explicit_partial_not_silent_truncation(
    tmp_path: Path,
) -> None:
    dag = _build_git_dag(tmp_path)
    capped = collect_git_evidence(
        dag.root,
        case_id="CASE-slice-cap",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=dag.commits["merge_resolution"],
        purpose="ambiguity_resolving",
        anchors=_anchors(),
        limits=_full_limits(max_change_slices=1),
    )

    assert len(capped.slices) == 1
    assert capped.status == "partial"
    assert "change_slice_limit" in capped.reason_codes
    assert "change_slice_limit" in capped.skipped_change_reasons
    assert capped.status != "complete"


def test_preflight_rejects_shallow_missing_objects_and_stale_cutoff(
    tmp_path: Path,
) -> None:
    dag = _build_git_dag(tmp_path)
    cutoff = dag.commits["merge_resolution"]
    shallow = tmp_path / "shallow"
    subprocess.run(
        [
            "git",
            "clone",
            "--no-local",
            "--depth=1",
            dag.root.as_posix(),
            shallow.as_posix(),
        ],
        check=True,
        capture_output=True,
    )
    shallow_result = preflight_git_repository(shallow, cutoff)
    assert shallow_result.status == "unavailable"
    assert "shallow_repository" in shallow_result.reason_codes

    damaged = tmp_path / "damaged"
    shutil.copytree(dag.root, damaged)
    missing_oid = dag.commits["introduction"]
    object_path = damaged / ".git" / "objects" / missing_oid[:2] / missing_oid[2:]
    assert object_path.is_file()
    object_path.unlink()
    damaged_result = preflight_git_repository(damaged, cutoff)
    assert damaged_result.status == "unavailable"
    assert set(damaged_result.reason_codes) & {
        "reachable_object_scan_failed",
        "reachable_objects_missing",
    }

    (dag.root / "post_cutoff.txt").write_text("later\n", encoding="utf-8")
    _commit(dag.root, "post cutoff", "2001-01-12T00:00:00+0000")
    stale = preflight_git_repository(dag.root, cutoff)
    assert stale.status == "unavailable"
    assert stale.reason_codes == ["cutoff_not_bound_to_head"]


def test_fixed_argv_isolated_environment_timeout_and_output_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "fake-git"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
import time
if "SLEEP" in sys.argv:
    time.sleep(5)
elif "FLOOD" in sys.argv:
    os.write(1, b"x" * 100000)
else:
    print(json.dumps({
        "argv": sys.argv[1:],
        "secret": os.environ.get("NORM4CODE_TEST_SECRET"),
        "no_lazy_fetch": os.environ.get("GIT_NO_LAZY_FETCH"),
        "optional_locks": os.environ.get("GIT_OPTIONAL_LOCKS"),
    }))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("NORM4CODE_TEST_SECRET", "must-not-cross-boundary")
    runner = BoundedGitRunner(executable)
    injection = "--all; touch SHOULD_NOT_EXIST"
    probe = runner.run(
        tmp_path,
        ["PROBE", "--", injection],
        timeout_seconds=2,
        max_output_bytes=4096,
    )
    payload = json.loads(probe.stdout)
    assert payload["secret"] is None
    assert payload["no_lazy_fetch"] == "1"
    assert payload["optional_locks"] == "0"
    assert injection in payload["argv"]
    assert payload["argv"].count(injection) == 1
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()

    timed = runner.run(
        tmp_path, ["SLEEP"], timeout_seconds=1, max_output_bytes=1024
    )
    assert timed.timed_out
    flooded = runner.run(
        tmp_path, ["FLOOD"], timeout_seconds=2, max_output_bytes=1024
    )
    assert flooded.stdout_truncated
    assert len(flooded.stdout) == 1024


class _TimeoutQueryRunner:
    def __init__(self) -> None:
        self.delegate = BoundedGitRunner()
        self.executable = self.delegate.executable

    def run(
        self,
        repository_root: Path,
        argv,
        *,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> GitProcessOutcome:
        if "log" in argv or "blame" in argv:
            return GitProcessOutcome(
                argv=tuple(argv),
                return_code=-15,
                stdout=b"",
                stderr=b"",
                timed_out=True,
                stdout_truncated=False,
                stderr_truncated=False,
                duration_ms=timeout_seconds * 1000,
            )
        return self.delegate.run(
            repository_root,
            argv,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )


def test_query_timeouts_are_partial_not_no_match(tmp_path: Path) -> None:
    dag = _build_git_dag(tmp_path)
    runner = _TimeoutQueryRunner()
    preflight = preflight_git_repository(
        dag.root,
        dag.commits["merge_resolution"],
        runner=runner,
    )
    manifest = run_git_queries(
        dag.root,
        case_id="CASE-timeout",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=dag.commits["merge_resolution"],
        purpose="ambiguity_resolving",
        anchors=_anchors(),
        preflight=preflight,
        runner=runner,
    )
    assert manifest.status == "partial"
    assert manifest.status != "bounded_no_match"
    assert all(
        item.status == "partial"
        for item in manifest.family_results
        if item.status != "not_required"
    )
    assert all(item.timed_out for item in manifest.command_results)


def test_query_plan_keeps_repository_values_as_argv_data() -> None:
    value = "--all; touch NEVER_EXECUTED"
    anchor = GitAnchor(
        anchor_id="ANCHOR-argv",
        kind="literal",
        path="--odd;name.py",
        value=value,
        origin_ids=["ORIGIN-argv"],
        precise_origin_ids=["ORIGIN-argv"],
    )
    plan = plan_git_queries("1" * 40, [anchor])
    literal = next(item for item in plan if item.family == "pickaxe_literal")
    assert value in literal.argv
    assert literal.argv[-2:] == ["--", "--odd;name.py"]
    regex = next(item for item in plan if item.family == "pickaxe_regex")
    assert value not in regex.argv
    assert regex.argv[-2:] == ["--", "--odd;name.py"]
    with pytest.raises(ValidationError, match="safe repository-relative"):
        GitAnchor(
            anchor_id="bad",
            kind="path",
            path="../escape.py",
            origin_ids=["ORIGIN-bad"],
        )


def test_query_plan_honors_asymmetric_per_family_minima() -> None:
    anchors = [
        GitAnchor(
            anchor_id=f"ANCHOR-symbol-{ordinal}",
            kind="symbol",
            path=f"src/rule_{ordinal}.py",
            value=f"RULE_{ordinal}",
            origin_ids=[f"ORIGIN-symbol-{ordinal}"],
            precise_origin_ids=[f"ORIGIN-symbol-{ordinal}"],
        )
        for ordinal in range(3)
    ]
    anchors.append(
        GitAnchor(
            anchor_id="ANCHOR-line",
            kind="line",
            path="src/line_rule.py",
            start_line=1,
            end_line=2,
            origin_ids=["ORIGIN-line"],
        )
    )
    limits = GitLimits(
        min_path_history_queries=2,
        min_blame_queries=1,
        min_pickaxe_literal_queries=3,
        min_pickaxe_regex_queries=0,
        max_queries_per_family=4,
        max_total_commands=6,
    )
    plan = plan_git_queries("1" * 40, anchors, limits=limits)
    counts = {
        family: sum(item.family == family for item in plan)
        for family in (
            "path_history",
            "rename_tracking",
            "blame",
            "pickaxe_literal",
            "pickaxe_regex",
        )
    }
    assert counts == {
        "path_history": 2,
        "rename_tracking": 0,
        "blame": 1,
        "pickaxe_literal": 3,
        "pickaxe_regex": 0,
    }
    with pytest.raises(
        ValidationError, match="minimum Git family quotas exceed command budget"
    ):
        GitLimits(
            min_path_history_queries=2,
            min_blame_queries=1,
            min_pickaxe_literal_queries=3,
            min_pickaxe_regex_queries=0,
            max_queries_per_family=4,
            max_total_commands=5,
        )


def test_query_residual_and_commit_admission_prioritize_strong_families(
    tmp_path: Path,
) -> None:
    anchors = [
        GitAnchor(
            anchor_id="ANCHOR-strong",
            kind="symbol",
            path="src/rule.py",
            value="RULE",
            origin_ids=["ORIGIN-strong"],
            precise_origin_ids=["ORIGIN-strong"],
        ),
        GitAnchor(
            anchor_id="ANCHOR-line-only",
            kind="line",
            path="src/rule.py",
            start_line=1,
            end_line=1,
            origin_ids=["ORIGIN-strong"],
        ),
    ]
    residual_limits = GitLimits(
        min_path_history_queries=0,
        min_blame_queries=0,
        min_pickaxe_literal_queries=0,
        min_pickaxe_regex_queries=0,
        max_total_commands=4,
    )
    assert [
        item.family
        for item in plan_git_queries(
            "1" * 40,
            anchors,
            limits=residual_limits,
        )
    ] == ["pickaxe_literal", "pickaxe_regex", "blame", "rename_tracking"]

    active_anchors = [
        GitAnchor(
            anchor_id=f"ANCHOR-active-{ordinal}",
            kind="symbol",
            path=f"src/active_{ordinal}.py",
            value=f"RULE_{ordinal}",
            origin_ids=[f"ORIGIN-active-{ordinal}"],
            precise_origin_ids=[f"ORIGIN-active-{ordinal}"],
        )
        for ordinal in range(4)
    ]
    active_plan = plan_git_queries(
        "1" * 40,
        active_anchors,
        limits=GitLimits(max_total_commands=10, max_queries_per_family=4),
    )
    active_counts = Counter(item.family for item in active_plan)
    assert len(active_plan) == 10
    assert active_counts["rename_tracking"] >= 1
    assert (
        active_counts["pickaxe_literal"] + active_counts["pickaxe_regex"]
        > active_counts["path_history"]
    )

    root = tmp_path / "strong-admission"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _write_rule(root, rule="strict", condition="enabled", guard=True)
    strong_commit = _commit(root, "introduce rule", "2003-01-01T00:00:00+0000")
    with (root / "src/rule.py").open("a", encoding="utf-8") as stream:
        stream.write("\nUNRELATED = True\n")
    latest_commit = _commit(root, "unrelated path hit", "2003-01-02T00:00:00+0000")
    preflight = preflight_git_repository(root, latest_commit)
    manifest = run_git_queries(
        root,
        case_id="CASE-strong-admission",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=latest_commit,
        purpose="augmentative",
        anchors=anchors,
        preflight=preflight,
        limits=GitLimits(max_total_candidate_commits=1),
    )
    assert manifest.status == "partial"
    assert manifest.commit_ids == [strong_commit]
    assert "case_commit_limit" in manifest.reason_codes


def test_changed_line_matching_preserves_prefixes_and_is_symbol_boundary_safe() -> None:
    plus_literal = GitAnchor(
        anchor_id="ANCHOR-plus-literal",
        kind="literal",
        path="src/rule.py",
        value="++VALUE",
        origin_ids=["ORIGIN-plus-literal"],
        precise_origin_ids=["ORIGIN-plus-literal"],
    )
    minus_literal = GitAnchor(
        anchor_id="ANCHOR-minus-literal",
        kind="literal",
        path="src/rule.py",
        value="--flag",
        origin_ids=["ORIGIN-minus-literal"],
        precise_origin_ids=["ORIGIN-minus-literal"],
    )
    symbol = GitAnchor(
        anchor_id="ANCHOR-symbol",
        kind="symbol",
        path="src/rule.py",
        value="RULE",
        origin_ids=["ORIGIN-symbol"],
        precise_origin_ids=["ORIGIN-symbol"],
    )
    hunk = (
        "@@ -1 +1 @@\n"
        "---flag\n"
        "+++VALUE\n"
        "-preRULE = 1\n"
        "+RULE_suffix = 2\n"
        "\\ No newline at end of file\n"
    )
    lines = _changed_source_lines(hunk)
    assert lines[:2] == ("--flag", "++VALUE")
    assert _anchor_hits_changed_line(plus_literal, lines)
    assert _anchor_hits_changed_line(minus_literal, lines)
    assert not _anchor_hits_changed_line(symbol, lines)
    assert _anchor_hits_changed_line(symbol, (*lines, "RULE = 3"))
    assert not _anchor_hits_changed_line(symbol, ("规则RULE值 = 3",))
    assert not _anchor_hits_changed_line(symbol, ("RULE\N{COMBINING ACUTE ACCENT} = 3",))

    for after_path in ("src/original.py", "src/copied.py"):
        copy = _RawChange(
            parent_id="1" * 40,
            old_blob_id="2" * 40,
            new_blob_id="2" * 40,
            status="C",
            before_path="src/original.py",
            after_path=after_path,
        )
        with pytest.raises(ValueError, match="copy status is unsupported"):
            _change_kind(copy)


def test_policy_filtered_path_history_is_complete_bounded_no_match(
    tmp_path: Path,
) -> None:
    root = tmp_path / "policy-filtered"
    root.mkdir()
    _git(root, "init", "-b", "main")
    (root / "README.md").write_text("# Fixture\n", encoding="utf-8")
    cutoff = _commit(root, "fixture", "2004-01-01T00:00:00+0000")
    bundle = collect_git_evidence(
        root,
        case_id="CASE-policy-filtered",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=cutoff,
        purpose="augmentative",
        anchors=[
            GitAnchor(
                anchor_id="ANCHOR-never",
                kind="symbol",
                path="README.md",
                value="NEVER_PRESENT",
                origin_ids=["ORIGIN-never"],
                precise_origin_ids=["ORIGIN-never"],
            )
        ],
    )
    assert bundle.manifest.status == "complete"
    assert bundle.status == "bounded_no_match"
    assert bundle.slices == []
    assert bundle.skipped_change_reasons == []
    assert bundle.selection_audit.filter_reason_counts[
        "weak_path_only_without_changed_line_anchor_match"
    ] > 0
    assert "slice_selection_complete_no_match" in bundle.reason_codes
    topology_payload = bundle.topology.model_dump(
        mode="python", exclude={"artifact_digest"}
    )
    topology_payload["stdout_sha256"] = "f" * 64
    bundle_payload = bundle.model_dump(
        mode="python", exclude={"bundle_id", "artifact_digest"}
    )
    bundle_payload["topology"] = _make_topology(topology_payload).model_dump(
        mode="python"
    )
    with pytest.raises(ValidationError, match="empty Git selection topology"):
        _make_bundle(bundle_payload)


def test_blame_line_numbers_cannot_admit_an_unmatched_hunk(tmp_path: Path) -> None:
    root = tmp_path / "blame-filtered"
    root.mkdir()
    _git(root, "init", "-b", "main")
    (root / "README.md").write_text("# Fixture\n", encoding="utf-8")
    cutoff = _commit(root, "fixture", "2004-02-01T00:00:00+0000")
    bundle = collect_git_evidence(
        root,
        case_id="CASE-blame-filtered",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=cutoff,
        purpose="augmentative",
        anchors=[
            GitAnchor(
                anchor_id="ANCHOR-line",
                kind="line",
                path="README.md",
                start_line=1,
                end_line=1,
                origin_ids=["ORIGIN-blame"],
            ),
            GitAnchor(
                anchor_id="ANCHOR-exact-but-absent",
                kind="literal",
                path="README.md",
                value="NEVER_PRESENT",
                origin_ids=["ORIGIN-blame"],
                precise_origin_ids=["ORIGIN-blame"],
            ),
        ],
    )
    assert bundle.status == "bounded_no_match"
    assert bundle.slices == []
    assert bundle.selection_audit.filter_reason_counts[
        "strong_query_without_changed_line_anchor_match"
    ] > 0


def test_same_path_origin_without_precise_anchor_cannot_be_masked(
    tmp_path: Path,
) -> None:
    root = tmp_path / "origin-coverage"
    root.mkdir()
    _git(root, "init", "-b", "main")
    (root / "README.md").write_text("# Fixture\n", encoding="utf-8")
    cutoff = _commit(root, "fixture", "2005-01-01T00:00:00+0000")
    bundle = collect_git_evidence(
        root,
        case_id="CASE-origin-coverage",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=cutoff,
        purpose="ambiguity_resolving",
        anchors=[
            GitAnchor(
                anchor_id="ANCHOR-exact",
                kind="literal",
                path="README.md",
                value="# Fixture",
                origin_ids=["ORIGIN-exact"],
                precise_origin_ids=["ORIGIN-exact"],
            ),
            GitAnchor(
                anchor_id="ANCHOR-path-only",
                kind="path",
                path="README.md",
                origin_ids=["ORIGIN-uncovered"],
            ),
        ],
    )
    assert bundle.slices
    assert bundle.status == "partial"
    assert bundle.selection_audit.uncovered_origin_ids == ["ORIGIN-uncovered"]
    assert "slice_selection_origin_without_precise_anchor" in bundle.reason_codes


def test_slice_provenance_is_path_local_and_exact_replay_rejects_reseal(
    tmp_path: Path,
) -> None:
    from repo_constraint_memory_v3.unified.validation import (
        _git_object_replay_errors,
    )

    root = tmp_path / "path-local"
    root.mkdir()
    _git(root, "init", "-b", "main")
    (root / "src").mkdir()
    (root / "src/a.py").write_text("RULE = 1\n", encoding="utf-8")
    (root / "src/b.py").write_text("OTHER = 1\n", encoding="utf-8")
    cutoff = _commit(root, "both rules", "2006-01-01T00:00:00+0000")
    anchors = [
        GitAnchor(
            anchor_id="ANCHOR-a",
            kind="symbol",
            path="src/a.py",
            value="RULE",
            origin_ids=["ORIGIN-a"],
            precise_origin_ids=["ORIGIN-a"],
        ),
        GitAnchor(
            anchor_id="ANCHOR-a-line",
            kind="line",
            path="src/a.py",
            start_line=1,
            end_line=1,
            origin_ids=["ORIGIN-a"],
        ),
        GitAnchor(
            anchor_id="ANCHOR-b",
            kind="symbol",
            path="src/b.py",
            value="OTHER",
            origin_ids=["ORIGIN-b"],
            precise_origin_ids=["ORIGIN-b"],
        ),
    ]
    bundle = collect_git_evidence(
        root,
        case_id="CASE-path-local",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=cutoff,
        purpose="augmentative",
        anchors=anchors,
        limits=_full_limits(),
    )
    a_slice = next(item for item in bundle.slices if item.after_path == "src/a.py")
    assert "ANCHOR-a" in a_slice.anchor_ids
    assert "ANCHOR-a-line" in a_slice.anchor_ids
    assert "ANCHOR-b" not in a_slice.anchor_ids

    # Coordinated narrowing can satisfy the local schema, but exact replay must
    # restore every path-local command/anchor and reject the resealed payload.
    narrowed = a_slice.model_dump(
        mode="python", exclude={"item_id", "artifact_digest"}
    )
    narrowed["anchor_ids"] = ["ANCHOR-a"]
    matching_results = [
        result
        for result in bundle.manifest.command_results
        if cutoff in result.commit_ids and "ANCHOR-a" in result.anchor_ids
    ]
    narrowed["query_command_ids"] = sorted(
        result.command_id for result in matching_results
    )
    narrowed["query_families"] = sorted(
        {result.family for result in matching_results}
    )
    resealed_slice = _make_slice(narrowed)
    payload = bundle.model_dump(
        mode="python", exclude={"bundle_id", "artifact_digest"}
    )
    payload["slices"] = [
        resealed_slice.model_dump(mode="python")
        if item["item_id"] == a_slice.item_id
        else item
        for item in payload["slices"]
    ]
    resealed_bundle = _make_bundle(payload)
    errors = _git_object_replay_errors(
        root,
        cutoff,
        {resealed_bundle.bundle_id: resealed_bundle},
    )
    assert any("strong-slice selection/provenance differs" in item for item in errors)


def test_commit_message_context_is_bounded_and_typed(tmp_path: Path) -> None:
    root = tmp_path / "long-message-repository"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _write_rule(root, rule="strict", condition="enabled", guard=True)
    commit_id = _commit(
        root,
        "x" * 80,
        "2001-02-01T00:00:00+0000",
    )
    bundle = collect_git_evidence(
        root,
        case_id="CASE-long-message",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=commit_id,
        purpose="augmentative",
        anchors=[
            GitAnchor(
                anchor_id="ANCHOR-long-message",
                kind="symbol",
                path="src/rule.py",
                value="RULE",
                origin_ids=["ORIGIN-long-message"],
                precise_origin_ids=["ORIGIN-long-message"],
            )
        ],
        limits=_full_limits(max_commit_message_bytes=64),
    )
    assert bundle.status == "partial"
    assert "commit_message_context_truncated" in bundle.reason_codes
    assert bundle.slices
    assert {
        item.commit_message_context_status for item in bundle.slices
    } == {"truncated"}
    assert all(item.commit_message_context is None for item in bundle.slices)
    assert all(item.commit_message_context_sha256 is None for item in bundle.slices)


def test_manifest_and_slice_digests_detect_tampering(tmp_path: Path) -> None:
    dag = _build_git_dag(tmp_path)
    bundle = collect_git_evidence(
        dag.root,
        case_id="CASE-tamper",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=dag.commits["merge_resolution"],
        purpose="augmentative",
        anchors=_anchors(),
        limits=_full_limits(),
    )
    manifest = bundle.manifest.model_dump(mode="python")
    manifest["reason_codes"] = ["tampered"]
    with pytest.raises(
        ValidationError, match="manifest (?:ID|digest|reasons.*algebra)"
    ):
        GitQueryManifest.model_validate(manifest)

    anchor_manifest = bundle.manifest.model_dump(mode="python")
    anchor_manifest["anchors"][0]["origin_ids"] = ["ORIGIN-tampered"]
    if anchor_manifest["anchors"][0]["precise_origin_ids"]:
        anchor_manifest["anchors"][0]["precise_origin_ids"] = [
            "ORIGIN-tampered"
        ]
    with pytest.raises(ValidationError, match="manifest (?:ID|digest)"):
        GitQueryManifest.model_validate(anchor_manifest)

    slice_payload = bundle.slices[0].model_dump(mode="python")
    slice_payload["hunk"] = (slice_payload["hunk"] or "") + "tampered\n"
    with pytest.raises(ValidationError, match="patch digest"):
        GitChangeSlice.model_validate(slice_payload)

    message_payload = bundle.slices[0].model_dump(mode="python")
    message_payload["commit_message_context"] = "tampered message"
    with pytest.raises(ValidationError, match="commit message context digest"):
        GitChangeSlice.model_validate(message_payload)

    audit_payload = bundle.model_dump(
        mode="python", exclude={"bundle_id", "artifact_digest"}
    )
    audit_payload["selection_audit"]["filtered_count"] += 1
    with pytest.raises(ValidationError, match="filtered count differs"):
        _make_bundle(audit_payload)


def test_resealed_manifest_and_slice_provenance_algebra_is_rejected(
    tmp_path: Path,
) -> None:
    dag = _build_git_dag(tmp_path)
    bundle = collect_git_evidence(
        dag.root,
        case_id="CASE-algebra",
        snapshot_digest=_SNAPSHOT,
        cutoff_commit=dag.commits["merge_resolution"],
        purpose="augmentative",
        anchors=_anchors(),
        limits=_full_limits(),
    )
    assert bundle.manifest.command_results
    assert bundle.slices

    result_tamper = bundle.manifest.model_dump(
        mode="python", exclude={"manifest_id", "artifact_digest"}
    )
    result_tamper["command_results"][0]["argv"] = ["log", "--tampered"]
    with pytest.raises(ValidationError, match="result differs from command spec"):
        _manifest(result_tamper)

    anchor_tamper = bundle.manifest.model_dump(
        mode="python", exclude={"manifest_id", "artifact_digest"}
    )
    anchor_tamper["anchors"][0]["path"] = "src/other.py"
    with pytest.raises(
        ValidationError, match="commands differ from canonical anchors"
    ):
        _manifest(anchor_tamper)

    family_tamper = bundle.manifest.model_dump(
        mode="python", exclude={"manifest_id", "artifact_digest"}
    )
    family_tamper["family_results"][0]["returned_count"] += 1
    with pytest.raises(ValidationError, match="returned count differs"):
        _manifest(family_tamper)

    original_slice = next(
        item
        for item in bundle.slices
        if item.selection_reason == "strong_changed_line_anchor_match"
    )
    slice_payload = original_slice.model_dump(
        mode="python", exclude={"item_id", "artifact_digest"}
    )
    slice_payload["query_families"] = ["rename_tracking"]
    resealed_slice = _make_slice(slice_payload)
    bundle_payload = bundle.model_dump(
        mode="python", exclude={"bundle_id", "artifact_digest"}
    )
    bundle_payload["slices"] = [
        resealed_slice.model_dump(mode="python"),
        *[
            item
            for item in bundle_payload["slices"]
            if item["item_id"] != original_slice.item_id
        ],
    ]
    with pytest.raises(ValidationError, match="query provenance differs"):
        _make_bundle(bundle_payload)
