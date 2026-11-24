"""Deterministic, candidate-conditioned, read-only Git evidence acquisition.

This module deliberately owns no candidate discovery or semantic policy.  It
accepts host-validated anchors, binds every command to one exact HEAD commit,
and returns bounded, replayable query facts and change slices.  Repository
content and Git metadata are untrusted inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol, Sequence

from pydantic import Field, field_validator, model_validator

from repo_constraint_memory_v3.models import (
    NonEmptyString,
    SCHEMA_VERSION,
    SnapshotDigest,
    StrictModel,
)


GIT_EVIDENCE_VERSION = "git-evidence-1.2"
GIT_RAW_EXECUTION_VERSION = "git-raw-execution-1.0"
GIT_SLICE_SELECTION_VERSION = "git-strong-slice-selection-1.0"
GitObjectId = Annotated[
    str,
    Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"),
]
GitQueryFamily = Literal[
    "path_history",
    "blame",
    "pickaxe_literal",
    "pickaxe_regex",
    "rename_tracking",
]
GitEvidenceStatus = Literal[
    "not_required",
    "complete",
    "bounded_no_match",
    "partial",
    "unavailable",
    "failed",
]
GitQueryPurpose = Literal["augmentative", "ambiguity_resolving"]
CommitMessageContextStatus = Literal[
    "complete",
    "truncated",
    "unavailable",
    "invalid_utf8",
    "invalid_format",
]

_FAMILY_ORDER: tuple[GitQueryFamily, ...] = (
    "path_history",
    "blame",
    "pickaxe_literal",
    "pickaxe_regex",
    "rename_tracking",
)
_FAMILY_ADMISSION_ORDER: tuple[GitQueryFamily, ...] = (
    "pickaxe_literal",
    "pickaxe_regex",
    "blame",
    "rename_tracking",
    "path_history",
)
_STRONG_FAMILIES = frozenset({"pickaxe_literal", "pickaxe_regex", "blame"})
_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_BLAME_HEADER_RE = re.compile(
    r"^\^?([0-9a-f]{40}|[0-9a-f]{64}) [0-9]+ [0-9]+(?: [0-9]+)?$"
)
_RAW_CHANGE_RE = re.compile(
    r"^:([0-7]{6}) ([0-7]{6}) ([0-9a-f]{40}|[0-9a-f]{64}) "
    r"([0-9a-f]{40}|[0-9a-f]{64}) ([A-Z])([0-9]{0,3})$"
)
_HUNK_RE = re.compile(
    r"^@@ -([0-9]+)(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}-{_digest(value)[:24]}"


def _safe_repository_path(value: str) -> bool:
    if (
        not value
        or "\\" in value
        or len(value.encode("utf-8")) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and value == path.as_posix()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


class GitLimits(StrictModel):
    """Frozen host-owned resource limits for one candidate case."""

    command_timeout_seconds: Annotated[int, Field(ge=1, le=600)] = 20
    object_scan_timeout_seconds: Annotated[int, Field(ge=1, le=3600)] = 120
    max_command_output_bytes: Annotated[
        int, Field(ge=1024, le=100_000_000)
    ] = 2_000_000
    max_object_scan_output_bytes: Annotated[
        int, Field(ge=1024, le=1_000_000_000)
    ] = 64_000_000
    min_path_history_queries: Annotated[int, Field(ge=0, le=64)] = 1
    min_blame_queries: Annotated[int, Field(ge=0, le=64)] = 1
    min_pickaxe_literal_queries: Annotated[int, Field(ge=0, le=64)] = 1
    min_pickaxe_regex_queries: Annotated[int, Field(ge=0, le=64)] = 1
    max_queries_per_family: Annotated[int, Field(ge=1, le=64)] = 4
    max_total_commands: Annotated[int, Field(ge=1, le=256)] = 20
    max_results_per_query: Annotated[int, Field(ge=1, le=10_000)] = 16
    max_results_per_family: Annotated[int, Field(ge=1, le=100_000)] = 32
    max_total_candidate_commits: Annotated[
        int, Field(ge=1, le=100_000)
    ] = 64
    max_anchor_chars: Annotated[int, Field(ge=1, le=4096)] = 512
    max_blame_lines: Annotated[int, Field(ge=1, le=10_000)] = 200
    rename_similarity_percent: Annotated[int, Field(ge=1, le=100)] = 50
    patch_context_lines: Annotated[int, Field(ge=0, le=100)] = 3
    max_changed_paths_per_commit: Annotated[int, Field(ge=1, le=10_000)] = 128
    max_patch_bytes: Annotated[int, Field(ge=1024, le=100_000_000)] = 500_000
    max_total_patch_bytes: Annotated[
        int, Field(ge=1024, le=1_000_000_000)
    ] = 4_000_000
    max_change_slices: Annotated[int, Field(ge=1, le=100_000)] = 128
    max_commit_message_bytes: Annotated[
        int, Field(ge=64, le=1_000_000)
    ] = 4096

    @model_validator(mode="after")
    def coherent_query_bounds(self) -> GitLimits:
        minima = (
            self.min_path_history_queries,
            self.min_blame_queries,
            self.min_pickaxe_literal_queries,
            self.min_pickaxe_regex_queries,
        )
        if any(item > self.max_queries_per_family for item in minima):
            raise ValueError("minimum Git family quota exceeds maximum")
        if sum(minima) > self.max_total_commands:
            raise ValueError("minimum Git family quotas exceed command budget")
        if self.max_patch_bytes > self.max_total_patch_bytes:
            raise ValueError("per-patch Git budget exceeds case patch budget")
        return self


class GitAnchor(StrictModel):
    """One host-validated current-evidence anchor; never a raw regex."""

    anchor_id: NonEmptyString
    kind: Literal["path", "line", "literal", "symbol"]
    path: NonEmptyString
    value: str | None = None
    start_line: Annotated[int, Field(ge=1)] | None = None
    end_line: Annotated[int, Field(ge=1)] | None = None
    origin_ids: Annotated[list[NonEmptyString], Field(min_length=1)]
    precise_origin_ids: list[NonEmptyString] = Field(default_factory=list)

    @field_validator("origin_ids", "precise_origin_ids")
    @classmethod
    def canonical_origins(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("Git anchor origin IDs must be sorted and unique")
        return value

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        if not _safe_repository_path(value):
            raise ValueError("Git anchor path must be safe repository-relative POSIX")
        return value

    @field_validator("value")
    @classmethod
    def safe_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not value
            or "\x00" in value
            or "\n" in value
            or "\r" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("Git anchor value must be a bounded single line")
        return value

    @model_validator(mode="after")
    def shape(self) -> GitAnchor:
        if set(self.precise_origin_ids) - set(self.origin_ids):
            raise ValueError("precise Git anchor origins must belong to the anchor")
        if self.kind in {"literal", "symbol"}:
            if self.value is None or self.start_line is not None or self.end_line is not None:
                raise ValueError("literal/symbol anchors require only value and path")
            if self.kind == "literal" and self.precise_origin_ids != self.origin_ids:
                raise ValueError("literal Git anchors must be exact for every origin")
        elif self.kind == "line":
            if (
                self.value is not None
                or self.start_line is None
                or self.end_line is None
                or self.end_line < self.start_line
            ):
                raise ValueError("line anchors require an ordered line interval")
        elif (
            self.value is not None
            or self.start_line is not None
            or self.end_line is not None
        ):
            raise ValueError("path anchors cannot carry value or line fields")
        if self.kind in {"path", "line"} and self.precise_origin_ids:
            raise ValueError("path/line Git anchors cannot claim precise origins")
        return self


class GitPreflightCheck(StrictModel):
    cache_hit: bool = False
    name: NonEmptyString
    argv: list[str]
    status: Literal["passed", "failed", "truncated", "timed_out"]
    return_code: int | None

    @model_validator(mode="after")
    def cached_success_only(self) -> GitPreflightCheck:
        if self.cache_hit and self.return_code != 0:
            raise ValueError("Git preflight raw cache cannot reuse a failed process")
        return self


class GitPreflight(StrictModel):
    schema_version: str = SCHEMA_VERSION
    evidence_version: Literal["git-evidence-1.2"] = GIT_EVIDENCE_VERSION
    repository_root: NonEmptyString
    requested_cutoff_commit: NonEmptyString
    status: Literal["not_required", "complete", "unavailable", "failed"]
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]
    git_executable: str | None
    git_version: str | None
    object_format: Literal["sha1", "sha256"] | None
    head_commit: GitObjectId | None
    cutoff_commit: GitObjectId | None
    clean: bool | None
    shallow: bool | None
    partial_clone: bool | None
    alternates: bool | None
    reachable_objects_complete: bool | None
    reachable_object_count: Annotated[int, Field(ge=0)] | None
    checks: list[GitPreflightCheck]


class GitCommandSpec(StrictModel):
    command_id: Annotated[str, Field(pattern=r"^GITCMD-[0-9a-f]{24}$")]
    family: GitQueryFamily
    anchor_ids: Annotated[list[NonEmptyString], Field(min_length=1)]
    argv: Annotated[list[NonEmptyString], Field(min_length=1)]
    timeout_seconds: Annotated[int, Field(ge=1)]
    max_output_bytes: Annotated[int, Field(ge=1)]
    max_results: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def canonical_identity(self) -> GitCommandSpec:
        if self.anchor_ids != sorted(set(self.anchor_ids)):
            raise ValueError("Git command anchor IDs must be sorted and unique")
        payload = self.model_dump(mode="json", exclude={"command_id"})
        if self.command_id != _stable_id("GITCMD", payload):
            raise ValueError("incorrect Git command ID")
        return self


class GitCommandResult(StrictModel):
    cache_hit: bool = False
    command_id: Annotated[str, Field(pattern=r"^GITCMD-[0-9a-f]{24}$")]
    family: GitQueryFamily
    anchor_ids: Annotated[list[NonEmptyString], Field(min_length=1)]
    argv: Annotated[list[NonEmptyString], Field(min_length=1)]
    status: Literal["complete", "bounded_no_match", "partial", "failed"]
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]
    return_code: int | None
    timed_out: bool
    output_truncated: bool
    returned_count: Annotated[int, Field(ge=0)]
    retained_count: Annotated[int, Field(ge=0)]
    truncated_count: Annotated[int, Field(ge=0)]
    commit_ids: list[GitObjectId]
    stdout_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    duration_ms: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def count_contract(self) -> GitCommandResult:
        if self.cache_hit and (self.return_code != 0 or self.timed_out or self.output_truncated):
            raise ValueError("Git query raw cache can only reuse a complete successful process")
        if self.anchor_ids != sorted(set(self.anchor_ids)):
            raise ValueError("Git result anchor IDs must be sorted and unique")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("Git result reason codes must be canonical")
        if len(self.commit_ids) != len(set(self.commit_ids)):
            raise ValueError("Git result commit IDs must be unique")
        if self.retained_count != len(self.commit_ids):
            raise ValueError("Git command retained count differs from commit IDs")
        if self.returned_count != self.retained_count + self.truncated_count:
            raise ValueError("Git command count algebra is invalid")
        fatal = bool(
            {
                "git_command_start_failed",
                "git_command_nonzero_exit",
                "git_command_output_invalid",
            }
            & set(self.reason_codes)
        )
        incomplete = bool(
            {
                "git_command_timeout",
                "git_command_output_limit",
                "git_command_stderr_limit",
                "git_query_result_limit",
            }
            & set(self.reason_codes)
        )
        expected_status = (
            "failed"
            if fatal
            else "partial"
            if incomplete
            else "bounded_no_match"
            if not self.commit_ids
            else "complete"
        )
        if self.status != expected_status:
            raise ValueError("Git command status differs from result algebra")
        if self.timed_out != ("git_command_timeout" in self.reason_codes):
            raise ValueError("Git command timeout flag/reason mismatch")
        if bool(self.truncated_count) != (
            "git_query_result_limit" in self.reason_codes
        ):
            raise ValueError("Git command result truncation reason mismatch")
        if self.output_truncated != bool(
            {"git_command_output_limit", "git_command_stderr_limit"}
            & set(self.reason_codes)
        ):
            raise ValueError("Git command output truncation flag/reason mismatch")
        if self.return_code not in {0} and not self.timed_out:
            if "git_command_nonzero_exit" not in self.reason_codes:
                raise ValueError("Git command nonzero exit reason is missing")
        elif "git_command_nonzero_exit" in self.reason_codes:
            raise ValueError("Git command nonzero exit reason is contradictory")
        completion_reason = (
            "query_complete_no_match"
            if self.status == "bounded_no_match"
            else "query_complete"
            if self.status == "complete"
            else None
        )
        if completion_reason is not None and completion_reason not in self.reason_codes:
            raise ValueError("Git command completion reason is missing")
        if self.status not in {"complete", "bounded_no_match"} and (
            {"query_complete", "query_complete_no_match"} & set(self.reason_codes)
        ):
            raise ValueError("incomplete Git command carries a completion reason")
        return self


class GitQueryFamilyResult(StrictModel):
    family: GitQueryFamily
    status: GitEvidenceStatus
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]
    planned_count: Annotated[int, Field(ge=0)]
    omitted_count: Annotated[int, Field(ge=0)]
    executed_count: Annotated[int, Field(ge=0)]
    failed_count: Annotated[int, Field(ge=0)]
    returned_count: Annotated[int, Field(ge=0)]
    deduplicated_count: Annotated[int, Field(ge=0)]
    truncated_count: Annotated[int, Field(ge=0)]
    commit_ids: list[GitObjectId]
    command_ids: list[Annotated[str, Field(pattern=r"^GITCMD-[0-9a-f]{24}$")]]

    @model_validator(mode="after")
    def canonical_lists(self) -> GitQueryFamilyResult:
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("Git family reason codes must be canonical")
        if len(self.commit_ids) != len(set(self.commit_ids)):
            raise ValueError("Git family commit IDs must be unique")
        if len(self.command_ids) != len(set(self.command_ids)):
            raise ValueError("Git family command IDs must be unique")
        if self.deduplicated_count != len(self.commit_ids):
            raise ValueError("Git family deduplicated count differs from commits")
        return self


class GitQueryManifest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    evidence_version: Literal["git-evidence-1.2"] = GIT_EVIDENCE_VERSION
    manifest_id: Annotated[str, Field(pattern=r"^GITMAN-[0-9a-f]{24}$")]
    case_id: NonEmptyString
    snapshot_digest: SnapshotDigest
    temporal_cutoff_commit: GitObjectId
    purpose: GitQueryPurpose
    status: GitEvidenceStatus
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]
    limits: GitLimits
    anchors: list[GitAnchor]
    commands: list[GitCommandSpec]
    command_results: list[GitCommandResult]
    family_results: list[GitQueryFamilyResult]
    returned_count: Annotated[int, Field(ge=0)]
    deduplicated_count: Annotated[int, Field(ge=0)]
    truncated_count: Annotated[int, Field(ge=0)]
    commit_ids: list[GitObjectId]
    artifact_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def identity_and_digest(self) -> GitQueryManifest:
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("Git manifest reason codes must be canonical")
        if [item.family for item in self.family_results] != list(_FAMILY_ORDER):
            raise ValueError("Git manifest family results are not canonical")
        anchor_ids = [item.anchor_id for item in self.anchors]
        if anchor_ids != sorted(set(anchor_ids)):
            raise ValueError("Git manifest anchors must be sorted and unique")
        known_anchor_ids = set(anchor_ids)
        command_ids = [item.command_id for item in self.commands]
        if len(command_ids) != len(set(command_ids)):
            raise ValueError("Git manifest command IDs must be unique")
        result_ids = [item.command_id for item in self.command_results]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("Git manifest result IDs must be unique")
        command_by_id = {item.command_id: item for item in self.commands}
        if set(result_ids) - set(command_ids):
            raise ValueError("Git manifest result references an unknown command")
        if self.command_results and result_ids != command_ids:
            raise ValueError("Git manifest results must exactly cover commands")
        expected_commands = plan_git_queries(
            self.temporal_cutoff_commit,
            self.anchors,
            limits=self.limits,
        )
        if self.commands != expected_commands:
            raise ValueError("Git manifest commands differ from canonical anchors")
        for command in self.commands:
            if set(command.anchor_ids) - known_anchor_ids:
                raise ValueError("Git command references an unknown manifest anchor")
            if (
                command.timeout_seconds != self.limits.command_timeout_seconds
                or command.max_output_bytes
                != self.limits.max_command_output_bytes
                or command.max_results != self.limits.max_results_per_query
            ):
                raise ValueError("Git command bounds differ from manifest limits")
        for result in self.command_results:
            command = command_by_id[result.command_id]
            if (
                result.family != command.family
                or result.anchor_ids != command.anchor_ids
                or result.argv != command.argv
            ):
                raise ValueError("Git command result differs from command spec")
            if result.retained_count > command.max_results:
                raise ValueError("Git command retained results exceed its bound")

        for family_result in self.family_results:
            family_commands = [
                item for item in self.commands if item.family == family_result.family
            ]
            family_command_results = [
                item
                for item in self.command_results
                if item.family == family_result.family
            ]
            if family_result.planned_count != len(family_commands):
                raise ValueError("Git family planned count differs from commands")
            if family_result.executed_count != len(family_command_results):
                raise ValueError("Git family executed count differs from results")
            if family_result.command_ids != [
                item.command_id for item in family_command_results
            ]:
                raise ValueError("Git family command IDs differ from results")
            if family_result.failed_count != sum(
                item.status == "failed" for item in family_command_results
            ):
                raise ValueError("Git family failed count differs from results")
            if family_result.returned_count != sum(
                item.returned_count for item in family_command_results
            ):
                raise ValueError("Git family returned count differs from results")

            observed: list[str] = []
            seen: set[str] = set()
            for result in family_command_results:
                for commit_id in result.commit_ids:
                    if commit_id not in seen:
                        seen.add(commit_id)
                        observed.append(commit_id)
            family_limit_truncated = max(
                0, len(observed) - self.limits.max_results_per_family
            )
            retained = observed[: self.limits.max_results_per_family]
            if family_result.commit_ids != retained:
                raise ValueError("Git family commits differ from result aggregation")
            expected_truncated = (
                sum(item.truncated_count for item in family_command_results)
                + family_limit_truncated
            )
            if family_result.truncated_count != expected_truncated:
                raise ValueError("Git family truncated count differs from results")

            # A non-empty result set follows one exact aggregation branch.
            # Empty sets may instead represent preflight failure or a family
            # fully omitted by the bounded host plan; both are bound again by
            # the enclosing bundle and host-plan replay.
            if family_command_results:
                reasons = {
                    reason
                    for item in family_command_results
                    for reason in item.reason_codes
                }
                if family_result.omitted_count:
                    reasons.add("query_plan_budget")
                if family_limit_truncated:
                    reasons.add("family_result_limit")
                failed_count = sum(
                    item.status == "failed" for item in family_command_results
                )
                partial = any(
                    item.status == "partial" for item in family_command_results
                )
                expected_status: GitEvidenceStatus = (
                    "failed"
                    if failed_count == len(family_command_results)
                    else "partial"
                    if (
                        failed_count
                        or partial
                        or family_result.omitted_count
                        or family_limit_truncated
                    )
                    else "bounded_no_match"
                    if not retained
                    else "complete"
                )
                if family_result.status != expected_status:
                    raise ValueError("Git family status differs from result algebra")
                if family_result.reason_codes != sorted(reasons or {"query_complete"}):
                    raise ValueError("Git family reasons differ from result algebra")
            elif any(
                (
                    family_result.failed_count,
                    family_result.returned_count,
                    family_result.deduplicated_count,
                    family_result.truncated_count,
                    len(family_result.commit_ids),
                    len(family_result.command_ids),
                )
            ):
                raise ValueError("unexecuted Git family carries result aggregates")

        if self.deduplicated_count != len(self.commit_ids):
            raise ValueError("Git manifest deduplicated count differs from commits")
        if len(self.commit_ids) != len(set(self.commit_ids)):
            raise ValueError("Git manifest commit IDs must be unique")
        if self.returned_count != sum(
            item.returned_count for item in self.command_results
        ):
            raise ValueError("Git manifest returned count differs from results")
        global_commits: list[str] = []
        global_seen: set[str] = set()
        family_result_by_family = {
            item.family: item for item in self.family_results
        }
        for family in _FAMILY_ADMISSION_ORDER:
            family_result = family_result_by_family[family]
            for commit_id in family_result.commit_ids:
                if commit_id not in global_seen:
                    global_seen.add(commit_id)
                    global_commits.append(commit_id)
        total_limit_truncated = max(
            0, len(global_commits) - self.limits.max_total_candidate_commits
        )
        expected_commits = global_commits[: self.limits.max_total_candidate_commits]
        if self.commit_ids != expected_commits:
            raise ValueError("Git manifest commits differ from family aggregation")
        if self.truncated_count != (
            sum(item.truncated_count for item in self.family_results)
            + total_limit_truncated
        ):
            raise ValueError("Git manifest truncated count differs from families")

        special_reason_sets = {
            ("git_not_required",): "not_required",
            ("git_preflight_not_complete",): None,
            ("required_git_query_has_no_anchor",): "failed",
        }
        reason_key = tuple(self.reason_codes)
        if reason_key in special_reason_sets:
            expected_special = special_reason_sets[reason_key]
            if expected_special is not None and self.status != expected_special:
                raise ValueError("Git manifest special status/reason mismatch")
            if any(
                (
                    self.command_results,
                    self.returned_count,
                    self.deduplicated_count,
                    self.truncated_count,
                    self.commit_ids,
                )
            ):
                raise ValueError("blocked Git manifest carries query aggregates")
            for family_result in self.family_results:
                if reason_key == ("git_not_required",):
                    expected_family_status: GitEvidenceStatus = "not_required"
                    expected_family_reason = ["git_not_required"]
                elif reason_key == ("required_git_query_has_no_anchor",):
                    expected_family_status = "not_required"
                    expected_family_reason = ["no_applicable_anchor"]
                elif family_result.planned_count or family_result.omitted_count:
                    expected_family_status = self.status
                    expected_family_reason = ["git_preflight_not_complete"]
                else:
                    expected_family_status = "not_required"
                    expected_family_reason = ["no_applicable_anchor"]
                if (
                    family_result.status != expected_family_status
                    or family_result.reason_codes != expected_family_reason
                ):
                    raise ValueError(
                        "blocked Git family status/reason differs from manifest"
                    )
        else:
            for family_result in self.family_results:
                if family_result.executed_count:
                    continue
                if family_result.planned_count:
                    raise ValueError("planned Git command was not executed")
                if family_result.omitted_count:
                    if (
                        family_result.status != "partial"
                        or family_result.reason_codes != ["query_plan_budget"]
                    ):
                        raise ValueError("omitted Git family algebra is invalid")
                elif (
                    family_result.status != "not_required"
                    or family_result.reason_codes != ["no_applicable_anchor"]
                ):
                    raise ValueError("inapplicable Git family algebra is invalid")
            applicable = [
                item for item in self.family_results if item.status != "not_required"
            ]
            overall_reasons = {
                reason for item in applicable for reason in item.reason_codes
            }
            if total_limit_truncated:
                overall_reasons.add("case_commit_limit")
            expected_status = (
                "failed"
                if applicable
                and all(item.status == "failed" for item in applicable)
                else "partial"
                if any(
                    item.status in {"partial", "failed"} for item in applicable
                )
                or total_limit_truncated
                else "complete"
                if self.commit_ids
                else "bounded_no_match"
            )
            if expected_status == "bounded_no_match":
                overall_reasons.add("all_applicable_queries_complete_no_match")
            if self.status != expected_status:
                raise ValueError("Git manifest status differs from family algebra")
            if self.reason_codes != sorted(overall_reasons or {"query_complete"}):
                raise ValueError("Git manifest reasons differ from family algebra")
        payload = self.model_dump(
            mode="json", exclude={"manifest_id", "artifact_digest"}
        )
        if self.manifest_id != _stable_id("GITMAN", payload):
            raise ValueError("incorrect Git query manifest ID")
        digest_payload = self.model_dump(mode="json", exclude={"artifact_digest"})
        if self.artifact_digest != _digest(digest_payload):
            raise ValueError("incorrect Git query manifest digest")
        return self


GitSliceRetentionReason = Literal[
    "strong_changed_line_anchor_match",
    "weak_changed_line_anchor_match",
    "direct_pure_rename",
]
GitSliceFilterReason = Literal[
    "merge_inherited_or_ambiguous",
    "change_path_not_anchored",
    "query_provenance_not_path_local",
    "strong_query_without_changed_line_anchor_match",
    "weak_path_only_without_changed_line_anchor_match",
]


class GitSliceSelectionAudit(StrictModel):
    selection_version: Literal["git-strong-slice-selection-1.0"] = (
        GIT_SLICE_SELECTION_VERSION
    )
    evaluated_count: Annotated[int, Field(ge=0)]
    eligible_count: Annotated[int, Field(ge=0)]
    emitted_count: Annotated[int, Field(ge=0)]
    capacity_omitted_count: Annotated[int, Field(ge=0)]
    filtered_count: Annotated[int, Field(ge=0)]
    retention_reason_counts: dict[
        GitSliceRetentionReason, Annotated[int, Field(ge=1)]
    ]
    filter_reason_counts: dict[GitSliceFilterReason, Annotated[int, Field(ge=1)]]
    uncovered_origin_ids: list[NonEmptyString]
    incomplete_reason_codes: list[NonEmptyString]

    @model_validator(mode="after")
    def count_and_coverage_algebra(self) -> GitSliceSelectionAudit:
        if self.uncovered_origin_ids != sorted(set(self.uncovered_origin_ids)):
            raise ValueError("Git selection uncovered origins must be canonical")
        if self.incomplete_reason_codes != sorted(
            set(self.incomplete_reason_codes)
        ):
            raise ValueError("Git selection incomplete reasons must be canonical")
        if self.eligible_count != sum(self.retention_reason_counts.values()):
            raise ValueError("Git selection eligible count differs from reasons")
        if self.filtered_count != sum(self.filter_reason_counts.values()):
            raise ValueError("Git selection filtered count differs from reasons")
        if self.evaluated_count != self.eligible_count + self.filtered_count:
            raise ValueError("Git selection evaluated count algebra is invalid")
        if self.eligible_count != self.emitted_count + self.capacity_omitted_count:
            raise ValueError("Git selection capacity count algebra is invalid")
        expected_incomplete = (
            ["slice_selection_origin_without_precise_anchor"]
            if self.uncovered_origin_ids
            else []
        )
        if self.incomplete_reason_codes != expected_incomplete:
            raise ValueError("Git selection origin coverage algebra is invalid")
        return self


class GitChangeSlice(StrictModel):
    schema_version: str = SCHEMA_VERSION
    evidence_version: Literal["git-evidence-1.2"] = GIT_EVIDENCE_VERSION
    item_id: Annotated[str, Field(pattern=r"^GITITEM-[0-9a-f]{24}$")]
    case_id: NonEmptyString
    snapshot_digest: SnapshotDigest
    commit_id: GitObjectId
    parent_id: GitObjectId | None
    commit_parents: list[GitObjectId]
    commit_time: int
    commit_message_context_status: CommitMessageContextStatus
    commit_message_context: str | None
    commit_message_context_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ] | None
    comparison_kind: Literal["root", "ordinary", "merge_resolution"]
    change_kind: Literal[
        "introduction",
        "modification",
        "removal",
        "rename",
        "rename_modification",
    ]
    before_path: str | None
    after_path: str | None
    old_blob_id: GitObjectId | None
    new_blob_id: GitObjectId | None
    hunk_header: str | None
    hunk: str | None
    patch_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    query_command_ids: Annotated[
        list[Annotated[str, Field(pattern=r"^GITCMD-[0-9a-f]{24}$")]],
        Field(min_length=1),
    ]
    query_families: Annotated[list[GitQueryFamily], Field(min_length=1)]
    anchor_ids: Annotated[list[NonEmptyString], Field(min_length=1)]
    selection_reason: GitSliceRetentionReason
    selection_anchor_ids: Annotated[list[NonEmptyString], Field(min_length=1)]
    selection_origin_ids: Annotated[list[NonEmptyString], Field(min_length=1)]
    artifact_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("before_path", "after_path")
    @classmethod
    def safe_optional_path(cls, value: str | None) -> str | None:
        if value is not None and not _safe_repository_path(value):
            raise ValueError("change-slice path is unsafe")
        return value

    @model_validator(mode="after")
    def shape_identity_and_digest(self) -> GitChangeSlice:
        for values in (
            self.query_command_ids,
            self.query_families,
            self.anchor_ids,
            self.selection_anchor_ids,
            self.selection_origin_ids,
        ):
            if values != sorted(set(values)):
                raise ValueError("Git change-slice provenance must be canonical")
        if set(self.selection_anchor_ids) - set(self.anchor_ids):
            raise ValueError("Git selection anchors must belong to query provenance")
        if self.parent_id is not None and self.parent_id not in self.commit_parents:
            raise ValueError("change-slice parent is not a commit parent")
        if self.comparison_kind == "root" and self.parent_id is not None:
            raise ValueError("root change slice cannot carry a parent")
        if self.before_path is None and self.after_path is None:
            raise ValueError("change slice requires a before or after path")
        if self.commit_message_context_status == "complete":
            if (
                self.commit_message_context is None
                or self.commit_message_context_sha256 is None
            ):
                raise ValueError("complete commit message context requires content")
            encoded_context = self.commit_message_context.encode("utf-8")
            if len(encoded_context) > 1_000_000:
                raise ValueError("commit message context exceeds schema bound")
            if any(character in self.commit_message_context for character in "\x00\r\n"):
                raise ValueError("commit message context contains a forbidden control")
            if self.commit_message_context_sha256 != hashlib.sha256(
                encoded_context
            ).hexdigest():
                raise ValueError("commit message context digest does not match")
        elif (
            self.commit_message_context is not None
            or self.commit_message_context_sha256 is not None
        ):
            raise ValueError("skipped commit message context cannot carry content")
        if self.patch_sha256 != hashlib.sha256(
            (self.hunk or "").encode("utf-8")
        ).hexdigest():
            raise ValueError("change-slice patch digest does not match its hunk")
        if self.selection_reason == "direct_pure_rename":
            if self.change_kind != "rename" or self.hunk is not None:
                raise ValueError("direct pure rename selection must be hunk-free")
        elif self.hunk is None:
            raise ValueError("changed-line selection requires a patch hunk")
        identity = self.model_dump(
            mode="json",
            exclude={
                "item_id",
                "artifact_digest",
                "query_command_ids",
                "query_families",
                "anchor_ids",
                "selection_reason",
                "selection_anchor_ids",
                "selection_origin_ids",
            },
        )
        if self.item_id != _stable_id("GITITEM", identity):
            raise ValueError("incorrect Git change-slice ID")
        digest_payload = self.model_dump(mode="json", exclude={"artifact_digest"})
        if self.artifact_digest != _digest(digest_payload):
            raise ValueError("incorrect Git change-slice digest")
        return self


class GitTopologyGroup(StrictModel):
    ordinal: Annotated[int, Field(ge=0)]
    commit_ids: Annotated[list[GitObjectId], Field(min_length=1)]

    @model_validator(mode="after")
    def canonical_members(self) -> GitTopologyGroup:
        if self.commit_ids != sorted(set(self.commit_ids)):
            raise ValueError("Git topology group members must be sorted and unique")
        return self


class GitIncomparablePair(StrictModel):
    left_commit_id: GitObjectId
    right_commit_id: GitObjectId

    @model_validator(mode="after")
    def canonical_pair(self) -> GitIncomparablePair:
        if self.left_commit_id >= self.right_commit_id:
            raise ValueError("incomparable Git pair must be canonically ordered")
        return self


class GitTopologyOrder(StrictModel):
    status: Literal["not_required", "complete", "partial", "failed"]
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]
    traversal_argv: list[str]
    groups: list[GitTopologyGroup]
    incomparable_pairs: list[GitIncomparablePair]
    stdout_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    artifact_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def coverage_and_digest(self) -> GitTopologyOrder:
        ordinals = [item.ordinal for item in self.groups]
        if ordinals != list(range(len(ordinals))):
            raise ValueError("Git topology group ordinals must be contiguous")
        commits = [commit for item in self.groups for commit in item.commit_ids]
        if len(commits) != len(set(commits)):
            raise ValueError("Git topology commit appears in multiple groups")
        if self.status == "complete" and not commits:
            raise ValueError("complete Git topology requires at least one commit")
        if self.artifact_digest != _digest(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        ):
            raise ValueError("incorrect Git topology digest")
        return self


class GitRawExecution(StrictModel):
    receipt_version: Literal["git-raw-execution-1.0"] = GIT_RAW_EXECUTION_VERSION
    logical_calls: int = Field(ge=0)
    physical_calls: int = Field(ge=0)
    cache_hits: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_close(self) -> GitRawExecution:
        if self.logical_calls != self.physical_calls + self.cache_hits:
            raise ValueError("Git raw execution counters do not close")
        return self


class GitTemporalBundle(StrictModel):
    schema_version: str = SCHEMA_VERSION
    evidence_version: Literal["git-evidence-1.2"] = GIT_EVIDENCE_VERSION
    bundle_id: Annotated[str, Field(pattern=r"^GITBUNDLE-[0-9a-f]{24}$")]
    case_id: NonEmptyString
    snapshot_digest: SnapshotDigest
    temporal_cutoff_commit: GitObjectId
    status: GitEvidenceStatus
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]
    preflight: GitPreflight
    manifest: GitQueryManifest
    topology: GitTopologyOrder
    slices: list[GitChangeSlice]
    selection_audit: GitSliceSelectionAudit
    skipped_change_reasons: list[NonEmptyString]
    raw_execution: GitRawExecution
    raw_execution_scope: Literal["collection_only", "collection_with_preflight"]
    artifact_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def identity_and_digest(self) -> GitTemporalBundle:
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("Git bundle reason codes must be canonical")
        if self.skipped_change_reasons != sorted(
            set(self.skipped_change_reasons)
        ):
            raise ValueError("Git skipped-change reasons must be canonical")
        if self.case_id != self.manifest.case_id:
            raise ValueError("Git bundle case differs from manifest")
        if self.snapshot_digest != self.manifest.snapshot_digest:
            raise ValueError("Git bundle snapshot differs from manifest")
        if self.temporal_cutoff_commit != self.manifest.temporal_cutoff_commit:
            raise ValueError("Git bundle cutoff differs from manifest")
        if self.preflight.requested_cutoff_commit != self.temporal_cutoff_commit:
            raise ValueError("Git preflight cutoff differs from bundle")
        if self.preflight.status == "complete":
            if (
                self.preflight.head_commit != self.temporal_cutoff_commit
                or self.preflight.cutoff_commit != self.temporal_cutoff_commit
                or self.preflight.clean is not True
                or self.preflight.shallow is not False
                or self.preflight.partial_clone is not False
                or self.preflight.alternates is not False
                or self.preflight.reachable_objects_complete is not True
            ):
                raise ValueError("complete Git preflight facts are inconsistent")
        elif self.preflight.status == "not_required":
            if self.manifest.status != "not_required":
                raise ValueError("not-required Git preflight/manifest mismatch")
        else:
            expected_manifest_status: GitEvidenceStatus = (
                "failed" if self.preflight.status == "failed" else "unavailable"
            )
            if self.manifest.status != expected_manifest_status:
                raise ValueError("incomplete Git preflight/manifest mismatch")

        if len({item.item_id for item in self.slices}) != len(self.slices):
            raise ValueError("Git bundle slice IDs must be unique")
        if self.selection_audit.emitted_count != len(self.slices):
            raise ValueError("Git selection emitted count differs from bundle slices")
        if self.selection_audit.capacity_omitted_count != (
            self.selection_audit.eligible_count - len(self.slices)
        ):
            raise ValueError("Git selection capacity omission differs from slices")
        expected_uncovered = sorted(
            {
                origin_id
                for anchor in self.manifest.anchors
                for origin_id in anchor.origin_ids
            }
            - {
                origin_id
                for anchor in self.manifest.anchors
                for origin_id in anchor.precise_origin_ids
            }
        )
        if self.selection_audit.uncovered_origin_ids != expected_uncovered:
            raise ValueError("Git selection origin coverage differs from manifest")
        emitted_reason_counts = Counter(
            item.selection_reason for item in self.slices
        )
        if any(
            emitted_reason_counts[reason] > count
            for reason, count in self.selection_audit.retention_reason_counts.items()
        ) or set(emitted_reason_counts) - set(
            self.selection_audit.retention_reason_counts
        ):
            raise ValueError("Git selection emitted reasons exceed eligible reasons")
        if not self.selection_audit.capacity_omitted_count and dict(
            sorted(emitted_reason_counts.items())
        ) != self.selection_audit.retention_reason_counts:
            raise ValueError("Git selection reasons differ from emitted slices")
        if (
            "change_slice_limit" in self.skipped_change_reasons
        ) != bool(self.selection_audit.capacity_omitted_count):
            raise ValueError("Git selection capacity reason algebra is invalid")
        if set(self.skipped_change_reasons) & {
            "merge_inherited_or_ambiguous",
            "change_path_not_anchored",
            "query_provenance_not_path_local",
            "strong_query_without_changed_line_anchor_match",
            "weak_path_only_without_changed_line_anchor_match",
        }:
            raise ValueError("Git policy filters cannot be resource skips")
        if not self.manifest.commit_ids and self.selection_audit.evaluated_count:
            raise ValueError("Git selection evaluated commits absent from manifest")
        command_results = {
            item.command_id: item for item in self.manifest.command_results
        }
        for item in self.slices:
            if (
                item.case_id != self.case_id
                or item.snapshot_digest != self.snapshot_digest
            ):
                raise ValueError("Git slice case/snapshot differs from bundle")
            if item.commit_id not in self.manifest.commit_ids:
                raise ValueError("Git slice commit is outside the manifest")
            if item.query_command_ids != sorted(set(item.query_command_ids)):
                raise ValueError("Git slice command provenance is not canonical")
            if item.query_families != sorted(set(item.query_families)):
                raise ValueError("Git slice family provenance is not canonical")
            if item.anchor_ids != sorted(set(item.anchor_ids)):
                raise ValueError("Git slice anchor provenance is not canonical")
            anchor_by_id = {
                anchor.anchor_id: anchor for anchor in self.manifest.anchors
            }
            provenance_results = [
                result
                for result in self.manifest.command_results
                if item.commit_id in result.commit_ids
                and set(result.anchor_ids) & set(item.anchor_ids)
            ]
            expected_command_ids = sorted(
                result.command_id for result in provenance_results
            )
            expected_families = sorted(
                {result.family for result in provenance_results}
            )
            expected_anchor_ids = sorted(
                {
                    anchor_id
                    for result in provenance_results
                    for anchor_id in result.anchor_ids
                    if anchor_id in item.anchor_ids
                }
            )
            if (
                item.query_command_ids != expected_command_ids
                or item.query_families != expected_families
                or item.anchor_ids != expected_anchor_ids
                or any(
                    command_id not in command_results
                    for command_id in item.query_command_ids
                )
            ):
                raise ValueError("Git slice query provenance differs from manifest")
            if any(
                anchor_id not in anchor_by_id
                for anchor_id in item.selection_anchor_ids
            ):
                raise ValueError("Git selection references an unknown anchor")
            expected_selection_origins = sorted(
                {
                    origin_id
                    for anchor_id in item.selection_anchor_ids
                    for origin_id in (
                        anchor_by_id[anchor_id].origin_ids
                        if item.selection_reason == "direct_pure_rename"
                        else anchor_by_id[anchor_id].precise_origin_ids
                    )
                }
            )
            if item.selection_origin_ids != expected_selection_origins:
                raise ValueError("Git selection origin provenance differs from anchors")

        topology_commits = {
            commit_id
            for group in self.topology.groups
            for commit_id in group.commit_ids
        }
        slice_commits = {item.commit_id for item in self.slices}
        if self.topology.status == "complete" and topology_commits != slice_commits:
            raise ValueError("Git topology does not exactly cover slice commits")
        if self.topology.traversal_argv and self.topology.traversal_argv != [
            "rev-list",
            "--topo-order",
            "--reverse",
            "--parents",
            self.temporal_cutoff_commit,
        ]:
            raise ValueError("Git topology traversal command differs from adapter")

        if any(
            item.commit_message_context is not None
            and len(item.commit_message_context.encode("utf-8"))
            > self.manifest.limits.max_commit_message_bytes
            for item in self.slices
        ):
            raise ValueError("Git commit message context exceeds profile bound")

        if self.manifest.status in {
            "not_required",
            "unavailable",
            "failed",
        }:
            if self.status != self.manifest.status:
                raise ValueError("terminal Git bundle status differs from manifest")
            if self.reason_codes != self.manifest.reason_codes:
                raise ValueError("terminal Git bundle reasons differ from manifest")
            if self.slices or self.skipped_change_reasons:
                raise ValueError("terminal Git bundle cannot carry change slices")
            expected_topology_status = (
                "not_required"
                if self.manifest.status == "not_required"
                else "failed"
                if self.manifest.status == "failed"
                else "partial"
            )
            expected_topology_reason = (
                ["git_not_required"]
                if self.manifest.status == "not_required"
                else ["git_query_has_no_temporal_items"]
            )
            if (
                self.topology.status != expected_topology_status
                or self.topology.reason_codes != expected_topology_reason
                or self.topology.traversal_argv
                or self.topology.groups
                or self.topology.incomparable_pairs
                or self.topology.stdout_sha256
                != hashlib.sha256(b"").hexdigest()
            ):
                raise ValueError("terminal Git bundle topology algebra is invalid")
        elif self.manifest.status == "bounded_no_match":
            expected_status: GitEvidenceStatus = (
                "partial"
                if self.selection_audit.incomplete_reason_codes
                else "bounded_no_match"
            )
            expected_reasons = sorted(
                {
                    *self.manifest.reason_codes,
                    *self.selection_audit.incomplete_reason_codes,
                }
            )
            if (
                self.status != expected_status
                or self.reason_codes != expected_reasons
                or self.slices
                or self.skipped_change_reasons
            ):
                raise ValueError("no-match Git bundle selection algebra is invalid")
            if (
                self.topology.status != "partial"
                or self.topology.reason_codes
                != ["git_query_has_no_temporal_items"]
                or self.topology.traversal_argv
                or self.topology.groups
                or self.topology.incomparable_pairs
                or self.topology.stdout_sha256
                != hashlib.sha256(b"").hexdigest()
            ):
                raise ValueError("no-match Git bundle topology algebra is invalid")
        else:
            expected_status = self.manifest.status
            expected_reasons = set(self.manifest.reason_codes)
            expected_reasons.update(self.selection_audit.incomplete_reason_codes)
            if self.skipped_change_reasons:
                expected_reasons.update(self.skipped_change_reasons)
                if expected_status == "complete":
                    expected_status = "partial"
            if self.selection_audit.incomplete_reason_codes and (
                expected_status == "complete"
            ):
                expected_status = "partial"
            complete_policy_no_match = (
                self.manifest.status == "complete"
                and not self.slices
                and not self.skipped_change_reasons
                and not self.selection_audit.incomplete_reason_codes
                and self.selection_audit.eligible_count == 0
            )
            if complete_policy_no_match:
                expected_status = "bounded_no_match"
                expected_reasons.add("slice_selection_complete_no_match")
            elif self.manifest.commit_ids and not self.slices:
                expected_reasons.add("no_valid_change_slice")
                if expected_status == "complete":
                    expected_status = "partial"
            if self.slices and self.topology.status != "complete":
                expected_reasons.update(self.topology.reason_codes)
                if expected_status == "complete":
                    expected_status = "partial"
            if not self.slices and (
                self.topology.status != "partial"
                or self.topology.reason_codes != ["no_temporal_items_for_topology"]
                or self.topology.traversal_argv
                or self.topology.groups
                or self.topology.incomparable_pairs
                or self.topology.stdout_sha256
                != hashlib.sha256(b"").hexdigest()
            ):
                raise ValueError("empty Git selection topology algebra is invalid")
            if self.status != expected_status:
                raise ValueError("Git bundle status differs from slice algebra")
            if self.reason_codes != sorted(expected_reasons or {"git_evidence_complete"}):
                raise ValueError("Git bundle reasons differ from slice algebra")
        identity = self.model_dump(
            mode="json", exclude={"bundle_id", "artifact_digest"}
        )
        if self.bundle_id != _stable_id("GITBUNDLE", identity):
            raise ValueError("incorrect Git temporal bundle ID")
        digest_payload = self.model_dump(mode="json", exclude={"artifact_digest"})
        if self.artifact_digest != _digest(digest_payload):
            raise ValueError("incorrect Git temporal bundle digest")
        return self


@dataclass(frozen=True)
class GitProcessOutcome:
    argv: tuple[str, ...]
    return_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool
    duration_ms: int
    start_failed: bool = False
    cache_hit: bool = False


class GitRunner(Protocol):
    executable: str | None

    def run(
        self,
        repository_root: Path,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> GitProcessOutcome: ...


class RunLocalGitRunner:
    """Bounded success-only raw reuse within one frozen snapshot invocation.

    Every caller still performs its own parsing, selection and logical budget.
    Resource boundaries participate in the key; failures never enter the cache.
    """

    def __init__(self, runner: GitRunner, *, snapshot_digest: str) -> None:
        self.runner = runner
        self.executable = runner.executable
        self.snapshot_digest = snapshot_digest
        self._cache: dict[tuple[object, ...], GitProcessOutcome] = {}
        self._bytes = 0
        self.logical_calls = 0
        self.physical_calls = 0
        self.cache_hits = 0

    def run(self, repository_root: Path, argv: Sequence[str], *, timeout_seconds: int, max_output_bytes: int) -> GitProcessOutcome:
        key = (repository_root.resolve().as_posix(), self.snapshot_digest, tuple(argv), timeout_seconds, max_output_bytes)
        self.logical_calls += 1
        if key in self._cache:
            self.cache_hits += 1
            return replace(self._cache[key], duration_ms=0, cache_hit=True)
        self.physical_calls += 1
        result = self.runner.run(repository_root, argv, timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes)
        size = len(result.stdout) + len(result.stderr)
        if (result.return_code == 0 and not result.start_failed and not result.timed_out
                and not result.stdout_truncated and not result.stderr_truncated
                and len(self._cache) < 4096 and self._bytes + size <= 32 * 1024 * 1024):
            self._cache[key] = result
            self._bytes += size
        return result

    def counters(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in ("logical_calls", "physical_calls", "cache_hits")}


class BoundedGitRunner:
    """Run a fixed Git executable without shell interpolation or lazy fetch."""

    def __init__(self, executable: str | Path | None = None) -> None:
        selected = str(executable) if executable is not None else shutil.which("git")
        self.executable = (
            Path(selected).resolve().as_posix() if selected is not None else None
        )

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            process.wait()

    def run(
        self,
        repository_root: Path,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> GitProcessOutcome:
        if self.executable is None:
            return GitProcessOutcome(
                argv=tuple(argv), return_code=None, stdout=b"", stderr=b"",
                timed_out=False, stdout_truncated=False, stderr_truncated=False,
                duration_ms=0, start_failed=True,
            )
        if any(not isinstance(argument, str) or "\x00" in argument for argument in argv):
            raise ValueError("Git argv must contain only NUL-free strings")
        root = repository_root.resolve()
        command = [
            self.executable,
            "--no-pager",
            "--no-optional-locks",
            "--no-replace-objects",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "diff.external=",
            "-C",
            root.as_posix(),
            *argv,
        ]
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="norm4code-git-") as raw_temp:
            isolated = Path(raw_temp)
            environment = {
                "HOME": isolated.as_posix(),
                "TMPDIR": isolated.as_posix(),
                "LANG": "C",
                "LC_ALL": "C",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_NO_LAZY_FETCH": "1",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_PAGER": "cat",
            }
            try:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=environment,
                    start_new_session=True,
                )
            except OSError:
                return GitProcessOutcome(
                    argv=tuple(argv), return_code=None, stdout=b"", stderr=b"",
                    timed_out=False, stdout_truncated=False,
                    stderr_truncated=False,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    start_failed=True,
                )
            assert process.stdout is not None and process.stderr is not None
            stdout = bytearray()
            stderr = bytearray()
            flags = {"stdout": False, "stderr": False}
            lock = threading.Lock()

            def drain(stream, buffer: bytearray, key: str, limit: int) -> None:
                observed = 0
                while True:
                    chunk = stream.read(65_536)
                    if not chunk:
                        break
                    observed += len(chunk)
                    remaining = limit - len(buffer)
                    if remaining > 0:
                        buffer.extend(chunk[:remaining])
                    if observed > limit:
                        with lock:
                            flags[key] = True
                        break

            stderr_limit = min(max_output_bytes, 65_536)
            readers = (
                threading.Thread(
                    target=drain,
                    args=(process.stdout, stdout, "stdout", max_output_bytes),
                    daemon=True,
                ),
                threading.Thread(
                    target=drain,
                    args=(process.stderr, stderr, "stderr", stderr_limit),
                    daemon=True,
                ),
            )
            for reader in readers:
                reader.start()
            timed_out = False
            while process.poll() is None:
                if time.monotonic() - started > timeout_seconds:
                    timed_out = True
                    self._terminate(process)
                    break
                with lock:
                    exceeded = flags["stdout"] or flags["stderr"]
                if exceeded:
                    self._terminate(process)
                    break
                time.sleep(0.01)
            for reader in readers:
                reader.join(timeout=1)
            process.stdout.close()
            process.stderr.close()
            return GitProcessOutcome(
                argv=tuple(argv),
                return_code=process.poll(),
                stdout=bytes(stdout),
                stderr=bytes(stderr),
                timed_out=timed_out,
                stdout_truncated=flags["stdout"],
                stderr_truncated=flags["stderr"],
                duration_ms=int((time.monotonic() - started) * 1000),
            )


def _check_record(
    name: str,
    argv: Sequence[str],
    outcome: GitProcessOutcome,
    *,
    accepted_return_codes: frozenset[int] = frozenset({0}),
) -> GitPreflightCheck:
    if outcome.timed_out:
        status = "timed_out"
    elif outcome.stdout_truncated or outcome.stderr_truncated:
        status = "truncated"
    elif outcome.return_code not in accepted_return_codes:
        status = "failed"
    else:
        status = "passed"
    return GitPreflightCheck(
        cache_hit=outcome.cache_hit,
        name=name,
        argv=list(argv),
        status=status,
        return_code=outcome.return_code,
    )


def _decode_ascii(outcome: GitProcessOutcome) -> str | None:
    if outcome.return_code != 0 or outcome.timed_out or outcome.stdout_truncated:
        return None
    try:
        return outcome.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError:
        return None


def preflight_git_repository(
    repository_root: Path,
    cutoff_commit: str,
    *,
    limits: GitLimits | None = None,
    runner: GitRunner | None = None,
) -> GitPreflight:
    """Prove that one clean full repository is bound to the requested HEAD."""

    effective_limits = limits or GitLimits()
    effective_runner = runner or BoundedGitRunner()
    root = repository_root.resolve()
    checks: list[GitPreflightCheck] = []
    base = {
        "schema_version": SCHEMA_VERSION,
        "evidence_version": GIT_EVIDENCE_VERSION,
        "repository_root": root.as_posix(),
        "requested_cutoff_commit": cutoff_commit,
        "git_executable": effective_runner.executable,
        "git_version": None,
        "object_format": None,
        "head_commit": None,
        "cutoff_commit": None,
        "clean": None,
        "shallow": None,
        "partial_clone": None,
        "alternates": None,
        "reachable_objects_complete": None,
        "reachable_object_count": None,
    }

    def finish(
        status: Literal["complete", "unavailable", "failed"],
        *reasons: str,
        **updates: object,
    ) -> GitPreflight:
        return GitPreflight.model_validate(
            base
            | updates
            | {
                "status": status,
                "reason_codes": list(reasons),
                "checks": [item.model_dump(mode="json") for item in checks],
            }
        )

    if not root.is_dir():
        return finish("unavailable", "repository_root_unavailable")
    if effective_runner.executable is None:
        return finish("unavailable", "git_executable_unavailable")

    def run_check(
        name: str,
        argv: Sequence[str],
        *,
        timeout: int | None = None,
        output_limit: int | None = None,
        accepted: frozenset[int] = frozenset({0}),
    ) -> GitProcessOutcome:
        outcome = effective_runner.run(
            root,
            argv,
            timeout_seconds=timeout or effective_limits.command_timeout_seconds,
            max_output_bytes=output_limit or effective_limits.max_command_output_bytes,
        )
        checks.append(
            _check_record(name, argv, outcome, accepted_return_codes=accepted)
        )
        return outcome

    version_outcome = run_check("git_version", ["--version"])
    version = _decode_ascii(version_outcome)
    if version is None or not version.startswith("git version "):
        return finish("unavailable", "git_version_unavailable")
    base["git_version"] = version.removeprefix("git version ")

    top_outcome = run_check("repository_top_level", ["rev-parse", "--show-toplevel"])
    top = _decode_ascii(top_outcome)
    if top is None:
        return finish("unavailable", "not_a_git_worktree")
    try:
        top_path = Path(top).resolve(strict=True)
    except OSError:
        return finish("unavailable", "git_top_level_unreadable")
    if top_path != root:
        return finish("unavailable", "authorized_root_not_git_top_level")

    format_outcome = run_check(
        "object_format", ["rev-parse", "--show-object-format"]
    )
    object_format = _decode_ascii(format_outcome)
    if object_format not in {"sha1", "sha256"}:
        return finish("failed", "unsupported_or_invalid_object_format")
    base["object_format"] = object_format
    expected_length = 40 if object_format == "sha1" else 64
    if not _OID_RE.fullmatch(cutoff_commit) or len(cutoff_commit) != expected_length:
        return finish("unavailable", "invalid_cutoff_object_id")

    head_outcome = run_check(
        "head_commit", ["rev-parse", "--verify", "HEAD^{commit}"]
    )
    head = _decode_ascii(head_outcome)
    if head is None or not _OID_RE.fullmatch(head):
        return finish("unavailable", "head_commit_unavailable")
    base["head_commit"] = head
    if head != cutoff_commit:
        return finish("unavailable", "cutoff_not_bound_to_head")

    cutoff_outcome = run_check(
        "cutoff_commit",
        ["rev-parse", "--verify", f"{cutoff_commit}^{{commit}}"],
    )
    resolved_cutoff = _decode_ascii(cutoff_outcome)
    if resolved_cutoff != cutoff_commit:
        return finish("unavailable", "cutoff_commit_unreadable")
    base["cutoff_commit"] = cutoff_commit

    shallow_outcome = run_check(
        "shallow_repository", ["rev-parse", "--is-shallow-repository"]
    )
    shallow_text = _decode_ascii(shallow_outcome)
    if shallow_text not in {"true", "false"}:
        return finish("failed", "shallow_state_unreadable")
    shallow = shallow_text == "true"
    base["shallow"] = shallow
    if shallow:
        return finish("unavailable", "shallow_repository")

    partial_argv = [
        "config",
        "--local",
        "--get-regexp",
        r"^(extensions\.partialclone|remote\..*\.promisor)$",
    ]
    partial_outcome = run_check(
        "partial_clone_configuration",
        partial_argv,
        accepted=frozenset({0, 1}),
    )
    if (
        partial_outcome.timed_out
        or partial_outcome.stdout_truncated
        or partial_outcome.return_code not in {0, 1}
    ):
        return finish("failed", "partial_clone_state_unreadable")
    partial_clone = bool(partial_outcome.stdout.strip())
    base["partial_clone"] = partial_clone
    if partial_clone:
        return finish("unavailable", "partial_clone_repository")

    common_outcome = run_check(
        "common_git_directory", ["rev-parse", "--git-common-dir"]
    )
    common_text = _decode_ascii(common_outcome)
    if common_text is None:
        return finish("failed", "git_common_directory_unreadable")
    common_path = Path(common_text)
    if not common_path.is_absolute():
        common_path = root / common_path
    try:
        common_path = common_path.resolve(strict=True)
    except OSError:
        return finish("failed", "git_common_directory_unreadable")
    alternates = any(
        path.exists() or path.is_symlink()
        for path in (
            common_path / "objects" / "info" / "alternates",
            common_path / "objects" / "info" / "http-alternates",
        )
    )
    base["alternates"] = alternates
    if alternates:
        return finish("unavailable", "object_alternates_not_allowed")

    worktree_outcome = run_check(
        "clean_worktree",
        ["diff", "--quiet", "--no-ext-diff", cutoff_commit, "--"],
        accepted=frozenset({0, 1}),
    )
    index_outcome = run_check(
        "clean_index",
        ["diff", "--cached", "--quiet", "--no-ext-diff", cutoff_commit, "--"],
        accepted=frozenset({0, 1}),
    )
    untracked_outcome = run_check(
        "untracked_files",
        ["ls-files", "--others", "--exclude-standard", "-z"],
    )
    if any(
        outcome.timed_out or outcome.stdout_truncated
        for outcome in (worktree_outcome, index_outcome, untracked_outcome)
    ):
        return finish("failed", "worktree_state_incomplete")
    if worktree_outcome.return_code not in {0, 1} or index_outcome.return_code not in {0, 1}:
        return finish("failed", "worktree_state_unreadable")
    clean = (
        worktree_outcome.return_code == 0
        and index_outcome.return_code == 0
        and not untracked_outcome.stdout
    )
    base["clean"] = clean
    if not clean:
        return finish("unavailable", "working_tree_not_clean")

    scan_outcome = run_check(
        "reachable_objects",
        ["rev-list", "--objects", "--missing=print", cutoff_commit],
        timeout=effective_limits.object_scan_timeout_seconds,
        output_limit=effective_limits.max_object_scan_output_bytes,
    )
    if scan_outcome.timed_out or scan_outcome.stdout_truncated:
        return finish(
            "unavailable",
            "reachable_object_scan_incomplete",
            reachable_objects_complete=False,
        )
    if scan_outcome.return_code != 0:
        return finish(
            "unavailable",
            "reachable_object_scan_failed",
            reachable_objects_complete=False,
        )
    missing = []
    object_count = 0
    for line in scan_outcome.stdout.splitlines():
        if not line:
            continue
        object_count += 1
        prefix = line.split(b" ", 1)[0]
        if prefix.startswith(b"?") and _OID_RE.fullmatch(
            prefix[1:].decode("ascii", errors="ignore")
        ):
            missing.append(prefix[1:])
    base["reachable_object_count"] = object_count
    base["reachable_objects_complete"] = not missing
    if missing:
        return finish("unavailable", "reachable_objects_missing")
    return finish("complete", "preflight_complete")


def _command_spec(
    family: GitQueryFamily,
    anchor_ids: Sequence[str],
    argv: Sequence[str],
    limits: GitLimits,
) -> GitCommandSpec:
    payload = {
        "family": family,
        "anchor_ids": sorted(set(anchor_ids)),
        "argv": list(argv),
        "timeout_seconds": limits.command_timeout_seconds,
        "max_output_bytes": limits.max_command_output_bytes,
        "max_results": limits.max_results_per_query,
    }
    return GitCommandSpec(
        command_id=_stable_id("GITCMD", payload),
        **payload,
    )


def _query_candidates(
    cutoff_commit: str,
    anchors: Sequence[GitAnchor],
    limits: GitLimits,
) -> dict[GitQueryFamily, list[GitCommandSpec]]:
    if len({anchor.anchor_id for anchor in anchors}) != len(anchors):
        raise ValueError("Git anchor IDs must be unique")
    for anchor in anchors:
        if anchor.value is not None and len(anchor.value) > limits.max_anchor_chars:
            raise ValueError("Git anchor exceeds configured character limit")
        if (
            anchor.kind == "line"
            and anchor.end_line is not None
            and anchor.start_line is not None
            and anchor.end_line - anchor.start_line + 1 > limits.max_blame_lines
        ):
            raise ValueError("Git blame anchor exceeds configured line limit")
    by_path: dict[str, list[str]] = {}
    for anchor in sorted(anchors, key=lambda item: item.anchor_id):
        by_path.setdefault(anchor.path, []).append(anchor.anchor_id)
    result: dict[GitQueryFamily, list[GitCommandSpec]] = {
        family: [] for family in _FAMILY_ORDER
    }
    count = limits.max_results_per_query + 1
    for path, anchor_ids in sorted(by_path.items()):
        common = [
            "--literal-pathspecs",
            "log",
            "--topo-order",
            "--format=%H",
            f"--max-count={count}",
        ]
        result["path_history"].append(
            _command_spec(
                "path_history",
                anchor_ids,
                [*common, cutoff_commit, "--", path],
                limits,
            )
        )
        result["rename_tracking"].append(
            _command_spec(
                "rename_tracking",
                anchor_ids,
                [
                    "--literal-pathspecs",
                    "log",
                    "--follow",
                    f"--find-renames={limits.rename_similarity_percent}%",
                    "--topo-order",
                    "--format=%H",
                    f"--max-count={count}",
                    cutoff_commit,
                    "--",
                    path,
                ],
                limits,
            )
        )
    for anchor in sorted(anchors, key=lambda item: item.anchor_id):
        if anchor.kind == "line":
            assert anchor.start_line is not None and anchor.end_line is not None
            result["blame"].append(
                _command_spec(
                    "blame",
                    [anchor.anchor_id],
                    [
                        "--literal-pathspecs",
                        "blame",
                        "--incremental",
                        "-L",
                        f"{anchor.start_line},{anchor.end_line}",
                        cutoff_commit,
                        "--",
                        anchor.path,
                    ],
                    limits,
                )
            )
        if anchor.kind in {"literal", "symbol"}:
            assert anchor.value is not None
            result["pickaxe_literal"].append(
                _command_spec(
                    "pickaxe_literal",
                    [anchor.anchor_id],
                    [
                        "--literal-pathspecs",
                        "log",
                        "--topo-order",
                        "--format=%H",
                        f"--max-count={count}",
                        "-S",
                        anchor.value,
                        cutoff_commit,
                        "--",
                        anchor.path,
                    ],
                    limits,
                )
            )
            # re.escape makes the repository-derived value data, not regex syntax.
            result["pickaxe_regex"].append(
                _command_spec(
                    "pickaxe_regex",
                    [anchor.anchor_id],
                    [
                        "--literal-pathspecs",
                        "log",
                        "--topo-order",
                        "--format=%H",
                        f"--max-count={count}",
                        "-G",
                        re.escape(anchor.value),
                        cutoff_commit,
                        "--",
                        anchor.path,
                    ],
                    limits,
                )
            )
    return result


def plan_git_queries(
    cutoff_commit: str,
    anchors: Sequence[GitAnchor],
    *,
    limits: GitLimits | None = None,
) -> list[GitCommandSpec]:
    """Create a stable, family-fair fixed-argv query plan."""

    effective_limits = limits or GitLimits()
    candidates = _query_candidates(cutoff_commit, anchors, effective_limits)
    selected: list[GitCommandSpec] = []
    per_family = {
        family: items[: effective_limits.max_queries_per_family]
        for family, items in candidates.items()
    }
    minimums: dict[GitQueryFamily, int] = {
        "path_history": effective_limits.min_path_history_queries,
        # Rename tracking remains eligible for residual budget but has no
        # frozen family minimum of its own.
        "rename_tracking": 0,
        "blame": effective_limits.min_blame_queries,
        "pickaxe_literal": effective_limits.min_pickaxe_literal_queries,
        "pickaxe_regex": effective_limits.min_pickaxe_regex_queries,
    }
    for ordinal in range(max(minimums.values(), default=0)):
        for family in _FAMILY_ORDER:
            if len(selected) >= effective_limits.max_total_commands:
                return selected
            if ordinal < minimums[family] and ordinal < len(per_family[family]):
                selected.append(per_family[family][ordinal])
    residual = {
        family: per_family[family][minimums[family] :]
        for family in _FAMILY_ADMISSION_ORDER
    }
    for ordinal in range(max((len(items) for items in residual.values()), default=0)):
        for family in _FAMILY_ADMISSION_ORDER:
            if len(selected) >= effective_limits.max_total_commands:
                return selected
            if ordinal < len(residual[family]):
                selected.append(residual[family][ordinal])
    return selected


def _parse_query_commit_ids(
    family: GitQueryFamily,
    content: bytes,
) -> tuple[list[str], bool]:
    """Parse only commit IDs; blame author metadata is intentionally discarded."""

    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return [], False
    values: list[str] = []
    seen: set[str] = set()
    if family == "blame":
        for line in text.splitlines():
            match = _BLAME_HEADER_RE.fullmatch(line)
            if match and match.group(1) not in seen:
                seen.add(match.group(1))
                values.append(match.group(1))
        # Incremental blame has metadata lines, so unmatched lines are expected.
        return values, True
    for line in text.splitlines():
        value = line.strip()
        if not value:
            continue
        if not _OID_RE.fullmatch(value):
            return [], False
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values, True


def _execute_query(
    repository_root: Path,
    spec: GitCommandSpec,
    runner: GitRunner,
) -> GitCommandResult:
    outcome = runner.run(
        repository_root,
        spec.argv,
        timeout_seconds=spec.timeout_seconds,
        max_output_bytes=spec.max_output_bytes,
    )
    parsed, parse_ok = _parse_query_commit_ids(spec.family, outcome.stdout)
    reasons: list[str] = []
    if outcome.start_failed:
        reasons.append("git_command_start_failed")
    if outcome.timed_out:
        reasons.append("git_command_timeout")
    if outcome.stdout_truncated:
        reasons.append("git_command_output_limit")
    if outcome.stderr_truncated:
        reasons.append("git_command_stderr_limit")
    if outcome.return_code not in {0} and not outcome.timed_out:
        reasons.append("git_command_nonzero_exit")
    if not parse_ok:
        reasons.append("git_command_output_invalid")
    returned_count = len(parsed)
    retained = parsed[: spec.max_results]
    observed_truncated = max(0, returned_count - len(retained))
    if observed_truncated:
        reasons.append("git_query_result_limit")
    fatal = (
        outcome.start_failed
        or (outcome.return_code not in {0} and not outcome.timed_out)
        or not parse_ok
    )
    incomplete = (
        outcome.timed_out
        or outcome.stdout_truncated
        or outcome.stderr_truncated
        or bool(observed_truncated)
    )
    if fatal:
        status = "failed"
    elif incomplete:
        status = "partial"
    elif not retained:
        status = "bounded_no_match"
        reasons.append("query_complete_no_match")
    else:
        status = "complete"
        reasons.append("query_complete")
    return GitCommandResult(
        cache_hit=outcome.cache_hit,
        command_id=spec.command_id,
        family=spec.family,
        anchor_ids=spec.anchor_ids,
        argv=spec.argv,
        status=status,
        reason_codes=sorted(set(reasons)),
        return_code=outcome.return_code,
        timed_out=outcome.timed_out,
        output_truncated=outcome.stdout_truncated or outcome.stderr_truncated,
        returned_count=returned_count,
        retained_count=len(retained),
        truncated_count=observed_truncated,
        commit_ids=retained,
        stdout_sha256=hashlib.sha256(outcome.stdout).hexdigest(),
        duration_ms=outcome.duration_ms,
    )


def _manifest(payload: dict[str, object]) -> GitQueryManifest:
    identity_payload = dict(payload)
    manifest_id = _stable_id("GITMAN", identity_payload)
    with_id = identity_payload | {"manifest_id": manifest_id}
    return GitQueryManifest.model_validate(
        with_id | {"artifact_digest": _digest(with_id)}
    )


def _empty_family_result(
    family: GitQueryFamily,
    status: GitEvidenceStatus,
    reason: str,
    *,
    planned_count: int = 0,
    omitted_count: int = 0,
) -> GitQueryFamilyResult:
    return GitQueryFamilyResult(
        family=family,
        status=status,
        reason_codes=[reason],
        planned_count=planned_count,
        omitted_count=omitted_count,
        executed_count=0,
        failed_count=0,
        returned_count=0,
        deduplicated_count=0,
        truncated_count=0,
        commit_ids=[],
        command_ids=[],
    )


def run_git_queries(
    repository_root: Path,
    *,
    case_id: str,
    snapshot_digest: str,
    cutoff_commit: str,
    purpose: GitQueryPurpose,
    anchors: Sequence[GitAnchor],
    preflight: GitPreflight,
    limits: GitLimits | None = None,
    runner: GitRunner | None = None,
) -> GitQueryManifest:
    """Execute the deterministic fixed-family plan and preserve exact coverage."""

    effective_limits = limits or GitLimits()
    effective_runner = runner or BoundedGitRunner(preflight.git_executable)
    canonical_anchors = sorted(anchors, key=lambda item: item.anchor_id)
    candidates = _query_candidates(
        cutoff_commit, canonical_anchors, effective_limits
    )
    commands = plan_git_queries(
        cutoff_commit, canonical_anchors, limits=effective_limits
    )
    selected_by_family = {
        family: [item for item in commands if item.family == family]
        for family in _FAMILY_ORDER
    }
    omitted_by_family = {
        family: max(0, len(candidates[family]) - len(selected_by_family[family]))
        for family in _FAMILY_ORDER
    }
    if preflight.status != "complete":
        blocked_status: GitEvidenceStatus = (
            "failed" if preflight.status == "failed" else "unavailable"
        )
        families = [
            _empty_family_result(
                family,
                "not_required" if not candidates[family] else blocked_status,
                "no_applicable_anchor"
                if not candidates[family]
                else "git_preflight_not_complete",
                planned_count=len(selected_by_family[family]),
                omitted_count=omitted_by_family[family],
            )
            for family in _FAMILY_ORDER
        ]
        return _manifest(
            {
                "schema_version": SCHEMA_VERSION,
                "evidence_version": GIT_EVIDENCE_VERSION,
                "case_id": case_id,
                "snapshot_digest": snapshot_digest,
                "temporal_cutoff_commit": cutoff_commit,
                "purpose": purpose,
                "status": blocked_status,
                "reason_codes": ["git_preflight_not_complete"],
                "limits": effective_limits.model_dump(mode="json"),
                "anchors": [
                    item.model_dump(mode="json") for item in canonical_anchors
                ],
                "commands": [item.model_dump(mode="json") for item in commands],
                "command_results": [],
                "family_results": [item.model_dump(mode="json") for item in families],
                "returned_count": 0,
                "deduplicated_count": 0,
                "truncated_count": 0,
                "commit_ids": [],
            }
        )
    if not anchors:
        families = [
            _empty_family_result(family, "not_required", "no_applicable_anchor")
            for family in _FAMILY_ORDER
        ]
        return _manifest(
            {
                "schema_version": SCHEMA_VERSION,
                "evidence_version": GIT_EVIDENCE_VERSION,
                "case_id": case_id,
                "snapshot_digest": snapshot_digest,
                "temporal_cutoff_commit": cutoff_commit,
                "purpose": purpose,
                "status": "failed",
                "reason_codes": ["required_git_query_has_no_anchor"],
                "limits": effective_limits.model_dump(mode="json"),
                "anchors": [],
                "commands": [],
                "command_results": [],
                "family_results": [item.model_dump(mode="json") for item in families],
                "returned_count": 0,
                "deduplicated_count": 0,
                "truncated_count": 0,
                "commit_ids": [],
            }
        )

    results = [
        _execute_query(repository_root.resolve(), spec, effective_runner)
        for spec in commands
    ]
    families: list[GitQueryFamilyResult] = []
    global_commits: list[str] = []
    global_seen: set[str] = set()
    overall_reasons: set[str] = set()
    for family in _FAMILY_ORDER:
        family_specs = selected_by_family[family]
        family_results = [item for item in results if item.family == family]
        if not candidates[family]:
            families.append(
                _empty_family_result(family, "not_required", "no_applicable_anchor")
            )
            continue
        observed: list[str] = []
        seen: set[str] = set()
        returned_count = sum(item.returned_count for item in family_results)
        truncated_count = sum(item.truncated_count for item in family_results)
        for item in family_results:
            for commit_id in item.commit_ids:
                if commit_id not in seen:
                    seen.add(commit_id)
                    observed.append(commit_id)
        family_limit_truncated = max(
            0, len(observed) - effective_limits.max_results_per_family
        )
        retained = observed[: effective_limits.max_results_per_family]
        truncated_count += family_limit_truncated
        failed_count = sum(item.status == "failed" for item in family_results)
        partial = any(item.status == "partial" for item in family_results)
        omitted = omitted_by_family[family]
        reasons = {
            reason for item in family_results for reason in item.reason_codes
        }
        if omitted:
            reasons.add("query_plan_budget")
        if family_limit_truncated:
            reasons.add("family_result_limit")
        if failed_count == len(family_results) and family_results:
            status: GitEvidenceStatus = "failed"
        elif failed_count or partial or omitted or family_limit_truncated:
            status = "partial"
        elif not retained:
            status = "bounded_no_match"
        else:
            status = "complete"
        family_item = GitQueryFamilyResult(
            family=family,
            status=status,
            reason_codes=sorted(reasons) or ["query_complete"],
            planned_count=len(family_specs),
            omitted_count=omitted,
            executed_count=len(family_results),
            failed_count=failed_count,
            returned_count=returned_count,
            deduplicated_count=len(retained),
            truncated_count=truncated_count,
            commit_ids=retained,
            command_ids=[item.command_id for item in family_results],
        )
        families.append(family_item)
        overall_reasons.update(family_item.reason_codes)

    family_by_name = {item.family: item for item in families}
    for family in _FAMILY_ADMISSION_ORDER:
        for commit_id in family_by_name[family].commit_ids:
            if commit_id not in global_seen:
                global_seen.add(commit_id)
                global_commits.append(commit_id)

    total_limit_truncated = max(
        0, len(global_commits) - effective_limits.max_total_candidate_commits
    )
    commits = global_commits[: effective_limits.max_total_candidate_commits]
    if total_limit_truncated:
        overall_reasons.add("case_commit_limit")
    applicable = [item for item in families if item.status != "not_required"]
    if applicable and all(item.status == "failed" for item in applicable):
        overall_status: GitEvidenceStatus = "failed"
    elif any(item.status in {"partial", "failed"} for item in applicable) or total_limit_truncated:
        overall_status = "partial"
    elif commits:
        overall_status = "complete"
    else:
        overall_status = "bounded_no_match"
        overall_reasons.add("all_applicable_queries_complete_no_match")
    return _manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_version": GIT_EVIDENCE_VERSION,
            "case_id": case_id,
            "snapshot_digest": snapshot_digest,
            "temporal_cutoff_commit": cutoff_commit,
            "purpose": purpose,
            "status": overall_status,
            "reason_codes": sorted(overall_reasons) or ["query_complete"],
            "limits": effective_limits.model_dump(mode="json"),
            "anchors": [
                item.model_dump(mode="json") for item in canonical_anchors
            ],
            "commands": [item.model_dump(mode="json") for item in commands],
            "command_results": [item.model_dump(mode="json") for item in results],
            "family_results": [item.model_dump(mode="json") for item in families],
            "returned_count": sum(item.returned_count for item in results),
            "deduplicated_count": len(commits),
            "truncated_count": sum(item.truncated_count for item in families)
            + total_limit_truncated,
            "commit_ids": commits,
        }
    )


@dataclass(frozen=True)
class _CommitMetadata:
    commit_id: str
    parents: tuple[str, ...]
    commit_time: int
    commit_message_context_status: CommitMessageContextStatus
    commit_message_context: str | None
    commit_message_context_sha256: str | None


@dataclass(frozen=True)
class _RawChange:
    parent_id: str | None
    old_blob_id: str | None
    new_blob_id: str | None
    status: str
    before_path: str | None
    after_path: str | None


@dataclass(frozen=True)
class _SelectedSliceCandidate:
    item: GitChangeSlice
    priority: int
    encounter_ordinal: int


def _run_internal(
    runner: GitRunner,
    root: Path,
    argv: Sequence[str],
    limits: GitLimits,
    *,
    output_limit: int | None = None,
) -> GitProcessOutcome:
    return runner.run(
        root,
        argv,
        timeout_seconds=limits.command_timeout_seconds,
        max_output_bytes=output_limit or limits.max_command_output_bytes,
    )


def _commit_metadata(
    runner: GitRunner,
    root: Path,
    commit_id: str,
    cutoff_commit: str,
    limits: GitLimits,
) -> tuple[_CommitMetadata | None, str | None]:
    ancestor = _run_internal(
        runner,
        root,
        ["merge-base", "--is-ancestor", commit_id, cutoff_commit],
        limits,
    )
    if ancestor.return_code != 0 or ancestor.timed_out:
        return None, "query_commit_not_reachable_from_cutoff"
    outcome = _run_internal(
        runner,
        root,
        ["show", "-s", "--format=%H%x00%P%x00%ct", commit_id],
        limits,
    )
    if (
        outcome.return_code != 0
        or outcome.timed_out
        or outcome.stdout_truncated
    ):
        return None, "commit_metadata_unavailable"
    try:
        fields = outcome.stdout.decode("ascii", errors="strict").strip().split("\x00")
        if len(fields) != 3 or fields[0] != commit_id:
            raise ValueError
        parents = tuple(value for value in fields[1].split() if value)
        if any(not _OID_RE.fullmatch(value) for value in parents):
            raise ValueError
        commit_time = int(fields[2])
    except (UnicodeDecodeError, ValueError):
        return None, "commit_metadata_invalid"
    message_outcome = _run_internal(
        runner,
        root,
        ["show", "-s", "--format=%s%x00", commit_id],
        limits,
        output_limit=limits.max_commit_message_bytes + 2,
    )
    message_status: CommitMessageContextStatus
    message_context: str | None = None
    message_digest: str | None = None
    if (
        message_outcome.return_code != 0
        or message_outcome.timed_out
        or message_outcome.start_failed
    ):
        message_status = "unavailable"
    elif message_outcome.stdout_truncated:
        message_status = "truncated"
    elif (
        not message_outcome.stdout.endswith(b"\x00\n")
        or message_outcome.stdout.count(b"\x00") != 1
    ):
        message_status = "invalid_format"
    else:
        raw_message = message_outcome.stdout[:-2]
        if len(raw_message) > limits.max_commit_message_bytes:
            message_status = "truncated"
        else:
            try:
                decoded_message = raw_message.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                message_status = "invalid_utf8"
            else:
                if any(character in decoded_message for character in "\x00\r\n"):
                    message_status = "invalid_format"
                else:
                    message_status = "complete"
                    message_context = decoded_message
                    message_digest = hashlib.sha256(raw_message).hexdigest()
    return (
        _CommitMetadata(
            commit_id,
            parents,
            commit_time,
            message_status,
            message_context,
            message_digest,
        ),
        None,
    )


def _parse_raw_changes(
    content: bytes,
    parent_id: str | None,
) -> tuple[list[_RawChange], bool]:
    chunks = content.split(b"\x00")
    index = 0
    result: list[_RawChange] = []
    try:
        while index < len(chunks):
            if not chunks[index]:
                index += 1
                continue
            header = chunks[index].decode("ascii", errors="strict")
            index += 1
            match = _RAW_CHANGE_RE.fullmatch(header)
            if match is None or index >= len(chunks):
                return [], False
            old_oid, new_oid, status = match.group(3), match.group(4), match.group(5)
            first_path = chunks[index].decode("utf-8", errors="strict")
            index += 1
            if not _safe_repository_path(first_path):
                return [], False
            second_path: str | None = None
            if status in {"R", "C"}:
                if index >= len(chunks):
                    return [], False
                second_path = chunks[index].decode("utf-8", errors="strict")
                index += 1
                if not _safe_repository_path(second_path):
                    return [], False
            before_path = None if status == "A" else first_path
            after_path = None if status == "D" else (second_path or first_path)
            result.append(
                _RawChange(
                    parent_id=parent_id,
                    old_blob_id=None if set(old_oid) == {"0"} else old_oid,
                    new_blob_id=None if set(new_oid) == {"0"} else new_oid,
                    status=status,
                    before_path=before_path,
                    after_path=after_path,
                )
            )
    except UnicodeDecodeError:
        return [], False
    return result, True


def _changes_for_commit(
    runner: GitRunner,
    root: Path,
    metadata: _CommitMetadata,
    limits: GitLimits,
) -> tuple[dict[str | None, list[_RawChange]] | None, str | None]:
    comparisons: tuple[str | None, ...] = metadata.parents or (None,)
    result: dict[str | None, list[_RawChange]] = {}
    for parent_id in comparisons:
        if parent_id is None:
            argv = [
                "diff-tree",
                "--root",
                "--no-commit-id",
                "-r",
                "-z",
                "--raw",
                "--no-abbrev",
                f"--find-renames={limits.rename_similarity_percent}%",
                metadata.commit_id,
            ]
        else:
            argv = [
                "diff-tree",
                "--no-commit-id",
                "-r",
                "-z",
                "--raw",
                "--no-abbrev",
                f"--find-renames={limits.rename_similarity_percent}%",
                parent_id,
                metadata.commit_id,
            ]
        outcome = _run_internal(runner, root, argv, limits)
        if (
            outcome.return_code != 0
            or outcome.timed_out
            or outcome.stdout_truncated
        ):
            return None, "change_scan_incomplete"
        changes, valid = _parse_raw_changes(outcome.stdout, parent_id)
        if not valid:
            return None, "change_scan_invalid"
        if len(changes) > limits.max_changed_paths_per_commit:
            return None, "changed_path_limit"
        result[parent_id] = changes
    return result, None


def _conservative_changes(
    metadata: _CommitMetadata,
    comparisons: dict[str | None, list[_RawChange]],
) -> tuple[list[tuple[_RawChange, str]], list[str]]:
    if not metadata.parents:
        return [(item, "root") for item in comparisons.get(None, [])], []
    if len(metadata.parents) == 1:
        return [
            (item, "ordinary") for item in comparisons.get(metadata.parents[0], [])
        ], []
    # A merge slice is admitted only when the resulting path/blob differs from
    # every parent.  A change already present in any parent is inherited, not a
    # new independent temporal event.  This decision is deliberately per path:
    # one merge may contain both a genuine conflict resolution and unrelated
    # files inherited unchanged from either parent.
    first_parent = metadata.parents[0]
    admitted: list[tuple[_RawChange, str]] = []
    skipped: list[str] = []
    for candidate in comparisons.get(first_parent, []):
        if _is_conservative_merge_resolution(metadata, comparisons, candidate):
            admitted.append((candidate, "merge_resolution"))
        else:
            skipped.append("merge_inherited_or_ambiguous")
    return admitted, skipped


def _change_result_identity(change: _RawChange) -> tuple[str, str | None]:
    """Identify the resulting tree entry, including a path-specific deletion."""

    path = change.after_path if change.after_path is not None else change.before_path
    assert path is not None
    return path, change.new_blob_id


def _is_conservative_merge_resolution(
    metadata: _CommitMetadata,
    comparisons: dict[str | None, list[_RawChange]],
    candidate: _RawChange,
) -> bool:
    """Prove one merge result differs from every parent without ambiguity.

    Every parent comparison must contain exactly one change producing the same
    final path/blob (or the same path-specific deletion).  Requiring a unique
    match prevents a coincidental change elsewhere in a merge from admitting
    the saved slice.
    """

    if len(metadata.parents) < 2 or candidate.parent_id != metadata.parents[0]:
        return False
    key = _change_result_identity(candidate)
    for index, parent_id in enumerate(metadata.parents):
        matches = [
            item
            for item in comparisons.get(parent_id, [])
            if _change_result_identity(item) == key
        ]
        if len(matches) != 1:
            return False
        if index == 0 and matches[0] != candidate:
            return False
    return True


def _patch_hunks(
    runner: GitRunner,
    root: Path,
    metadata: _CommitMetadata,
    change: _RawChange,
    limits: GitLimits,
) -> tuple[list[tuple[str, str]], int, str | None]:
    paths = sorted(
        {
            value
            for value in (change.before_path, change.after_path)
            if value is not None
        }
    )
    common = [
        "--literal-pathspecs",
        "--no-pager",
    ]
    if change.parent_id is None:
        argv = [
            *common,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "-r",
            "-p",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            f"--find-renames={limits.rename_similarity_percent}%",
            f"--unified={limits.patch_context_lines}",
            metadata.commit_id,
            "--",
            *paths,
        ]
    else:
        argv = [
            *common,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            f"--find-renames={limits.rename_similarity_percent}%",
            f"--unified={limits.patch_context_lines}",
            change.parent_id,
            metadata.commit_id,
            "--",
            *paths,
        ]
    outcome = _run_internal(
        runner,
        root,
        argv,
        limits,
        output_limit=limits.max_patch_bytes,
    )
    if outcome.return_code != 0 or outcome.timed_out or outcome.stdout_truncated:
        return [], len(outcome.stdout), "patch_incomplete"
    try:
        text = outcome.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return [], len(outcome.stdout), "patch_non_utf8"
    hunks: list[tuple[str, str]] = []
    header: str | None = None
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("@@ "):
            if header is not None:
                hunks.append((header, "".join(lines)))
            header = line.rstrip("\r\n")
            lines = [line]
        elif header is not None and line.startswith("diff --git "):
            hunks.append((header, "".join(lines)))
            header = None
            lines = []
        elif header is not None:
            lines.append(line)
    if header is not None:
        hunks.append((header, "".join(lines)))
    if any(_HUNK_RE.match(header) is None for header, _ in hunks):
        return [], len(outcome.stdout), "patch_hunk_invalid"
    return hunks, len(outcome.stdout), None


def _change_kind(change: _RawChange) -> str:
    if change.status == "C":
        raise ValueError("Git copy status is unsupported")
    if change.status == "A" and change.before_path is None:
        if change.after_path is None:
            raise ValueError("Git addition requires an after path")
        return "introduction"
    if change.status == "D" and change.after_path is None:
        if change.before_path is None:
            raise ValueError("Git removal requires a before path")
        return "removal"
    if change.status == "R":
        if (
            change.before_path is None
            or change.after_path is None
            or change.before_path == change.after_path
        ):
            raise ValueError("Git rename requires two distinct paths")
        if change.old_blob_id == change.new_blob_id:
            return "rename"
        return "rename_modification"
    if change.status in {"M", "T"}:
        if (
            change.before_path is None
            or change.after_path is None
            or change.before_path != change.after_path
        ):
            raise ValueError("Git modification requires one stable path")
        return "modification"
    raise ValueError("Git change status/shape is unsupported")


def _empty_selection_audit(
    anchors: Sequence[GitAnchor] = (),
) -> GitSliceSelectionAudit:
    all_origins = {
        origin_id for anchor in anchors for origin_id in anchor.origin_ids
    }
    precise_origins = {
        origin_id
        for anchor in anchors
        for origin_id in anchor.precise_origin_ids
    }
    uncovered = sorted(all_origins - precise_origins)
    return GitSliceSelectionAudit(
        evaluated_count=0,
        eligible_count=0,
        emitted_count=0,
        capacity_omitted_count=0,
        filtered_count=0,
        retention_reason_counts={},
        filter_reason_counts={},
        uncovered_origin_ids=uncovered,
        incomplete_reason_codes=(
            ["slice_selection_origin_without_precise_anchor"]
            if uncovered
            else []
        ),
    )


def _changed_source_lines(hunk: str) -> tuple[str, ...]:
    """Return added/removed source lines from one parsed ``@@`` hunk."""

    values: list[str] = []
    for line in hunk.splitlines():
        # ``_patch_hunks`` starts at the @@ header and never includes file-level
        # ---/+++ headers. Consequently every subsequent leading +/- is the
        # diff marker even when source itself starts with ++ or --.
        if line.startswith("@@") or line == r"\ No newline at end of file":
            continue
        if line.startswith(("+", "-")):
            values.append(line[1:])
    return tuple(values)


def _anchor_hits_changed_line(anchor: GitAnchor, lines: Sequence[str]) -> bool:
    if not anchor.precise_origin_ids or anchor.value is None:
        return False
    if anchor.kind == "literal":
        return any(anchor.value in line for line in lines)
    if anchor.kind != "symbol":
        return False

    def identifier_continue(character: str) -> bool:
        # ``str.isidentifier`` covers Unicode combining marks that ``\w`` does
        # not consistently model as an identifier continuation.
        return ("a" + character).isidentifier()

    for line in lines:
        start = 0
        while True:
            index = line.find(anchor.value, start)
            if index < 0:
                break
            end = index + len(anchor.value)
            left_safe = index == 0 or not identifier_continue(line[index - 1])
            right_safe = end == len(line) or not identifier_continue(line[end])
            if left_safe and right_safe:
                return True
            start = index + 1
    return False


def _anchor_paths(
    anchors: Sequence[GitAnchor],
    comparisons_by_commit: dict[str, dict[str | None, list[_RawChange]]],
) -> dict[str, set[str]]:
    paths = {anchor.anchor_id: {anchor.path} for anchor in anchors}
    changed = True
    while changed:
        changed = False
        for comparisons in comparisons_by_commit.values():
            for changes in comparisons.values():
                for item in changes:
                    if item.status != "R" or item.after_path is None:
                        continue
                    for anchor_id, aliases in paths.items():
                        if (
                            item.after_path in aliases
                            and item.before_path is not None
                            and item.before_path not in aliases
                        ):
                            aliases.add(item.before_path)
                            changed = True
    return paths


def _path_local_results(
    manifest: GitQueryManifest,
    *,
    commit_id: str,
    change: _RawChange,
    paths_by_anchor: dict[str, set[str]],
) -> tuple[list[GitCommandResult], set[str]]:
    change_paths = {
        path for path in (change.before_path, change.after_path) if path is not None
    }
    local_anchor_ids = {
        anchor_id
        for anchor_id, paths in paths_by_anchor.items()
        if paths & change_paths
    }
    results = [
        result
        for result in manifest.command_results
        if commit_id in result.commit_ids
        and set(result.anchor_ids) & local_anchor_ids
    ]
    return results, local_anchor_ids


def _slice_selection(
    *,
    change: _RawChange,
    hunk: str | None,
    local_results: Sequence[GitCommandResult],
    anchor_by_id: dict[str, GitAnchor],
) -> tuple[
    GitSliceRetentionReason | None,
    list[str],
    list[str],
    GitSliceFilterReason | None,
]:
    provenance_anchor_ids = {
        anchor_id for result in local_results for anchor_id in result.anchor_ids
    }
    rename_anchor_ids = sorted(
        {
            anchor_id
            for result in local_results
            if result.family == "rename_tracking"
            for anchor_id in result.anchor_ids
            if anchor_id in provenance_anchor_ids
        }
    )
    if (
        change.status == "R"
        and _change_kind(change) == "rename"
        and hunk is None
        and rename_anchor_ids
    ):
        origins = sorted(
            {
                origin_id
                for anchor_id in rename_anchor_ids
                for origin_id in anchor_by_id[anchor_id].origin_ids
            }
        )
        return "direct_pure_rename", rename_anchor_ids, origins, None
    if hunk is None:
        families = {result.family for result in local_results}
        reason: GitSliceFilterReason = (
            "strong_query_without_changed_line_anchor_match"
            if families & _STRONG_FAMILIES
            else "weak_path_only_without_changed_line_anchor_match"
        )
        return None, [], [], reason

    lines = _changed_source_lines(hunk)
    matched_anchor_ids = sorted(
        anchor_id
        for anchor_id in provenance_anchor_ids
        if _anchor_hits_changed_line(anchor_by_id[anchor_id], lines)
    )
    if not matched_anchor_ids:
        families = {result.family for result in local_results}
        reason = (
            "strong_query_without_changed_line_anchor_match"
            if families & _STRONG_FAMILIES
            else "weak_path_only_without_changed_line_anchor_match"
        )
        return None, [], [], reason
    matched_set = set(matched_anchor_ids)
    strong = any(
        result.family in _STRONG_FAMILIES
        and bool(set(result.anchor_ids) & matched_set)
        for result in local_results
    )
    origins = sorted(
        {
            origin_id
            for anchor_id in matched_anchor_ids
            for origin_id in anchor_by_id[anchor_id].precise_origin_ids
        }
    )
    return (
        "strong_changed_line_anchor_match"
        if strong
        else "weak_changed_line_anchor_match",
        matched_anchor_ids,
        origins,
        None,
    )


def _collect_change_slices(
    runner: GitRunner,
    root: Path,
    manifest: GitQueryManifest,
) -> tuple[list[GitChangeSlice], GitSliceSelectionAudit, list[str]]:
    """Replay the bounded strong-slice policy from exact commits and anchors."""

    limits = manifest.limits
    metadata_by_commit: dict[str, _CommitMetadata] = {}
    comparisons_by_commit: dict[str, dict[str | None, list[_RawChange]]] = {}
    skipped: list[str] = []
    for commit_id in manifest.commit_ids:
        metadata, reason = _commit_metadata(
            runner,
            root,
            commit_id,
            manifest.temporal_cutoff_commit,
            limits,
        )
        if metadata is None:
            skipped.append(reason or "commit_metadata_unavailable")
            continue
        comparisons, reason = _changes_for_commit(runner, root, metadata, limits)
        if comparisons is None:
            skipped.append(reason or "change_scan_incomplete")
            continue
        metadata_by_commit[commit_id] = metadata
        comparisons_by_commit[commit_id] = comparisons
        if metadata.commit_message_context_status != "complete":
            skipped.append(
                "commit_message_context_" + metadata.commit_message_context_status
            )

    anchors = manifest.anchors
    anchor_by_id = {anchor.anchor_id: anchor for anchor in anchors}
    paths_by_anchor = _anchor_paths(anchors, comparisons_by_commit)
    filter_counts: Counter[str] = Counter()
    work: list[
        tuple[
            int,
            int,
            int,
            _CommitMetadata,
            _RawChange,
            str,
            list[GitCommandResult],
        ]
    ] = []
    for commit_ordinal, commit_id in enumerate(manifest.commit_ids):
        metadata = metadata_by_commit.get(commit_id)
        comparisons = comparisons_by_commit.get(commit_id)
        if metadata is None or comparisons is None:
            continue
        admitted, merge_filters = _conservative_changes(metadata, comparisons)
        filter_counts.update(merge_filters)
        for change_ordinal, (change, comparison_kind) in enumerate(admitted):
            try:
                _change_kind(change)
            except ValueError:
                skipped.append("unsupported_change_status")
                continue
            local_results, local_anchor_ids = _path_local_results(
                manifest,
                commit_id=commit_id,
                change=change,
                paths_by_anchor=paths_by_anchor,
            )
            change_paths = {
                path
                for path in (change.before_path, change.after_path)
                if path is not None
            }
            anchored = any(
                paths & change_paths for paths in paths_by_anchor.values()
            )
            if not anchored:
                filter_counts["change_path_not_anchored"] += 1
                continue
            if not local_results or not local_anchor_ids:
                filter_counts["query_provenance_not_path_local"] += 1
                continue
            families = {result.family for result in local_results}
            priority = (
                0
                if families & _STRONG_FAMILIES
                else 1
                if "rename_tracking" in families
                else 2
            )
            work.append(
                (
                    priority,
                    commit_ordinal,
                    change_ordinal,
                    metadata,
                    change,
                    comparison_kind,
                    local_results,
                )
            )

    work.sort(key=lambda item: item[:3])
    eligible: dict[str, _SelectedSliceCandidate] = {}
    total_patch_bytes = 0
    encounter_ordinal = 0
    for priority, _, _, metadata, change, comparison_kind, local_results in work:
        if total_patch_bytes >= limits.max_total_patch_bytes:
            skipped.append("case_patch_byte_limit")
            break
        hunks, patch_bytes, patch_reason = _patch_hunks(
            runner, root, metadata, change, limits
        )
        total_patch_bytes += patch_bytes
        if total_patch_bytes > limits.max_total_patch_bytes:
            skipped.append("case_patch_byte_limit")
            break
        if patch_reason is not None:
            skipped.append(patch_reason)
            continue
        if not hunks and change.status != "R":
            skipped.append("non_text_change_without_hunk")
            continue
        evidence_hunks: list[tuple[str | None, str | None]] = (
            [(None, None)] if not hunks else list(hunks)
        )
        local_anchor_ids = sorted(
            {
                anchor_id
                for result in local_results
                for anchor_id in result.anchor_ids
                if anchor_id in anchor_by_id
            }
        )
        for hunk_header, hunk in evidence_hunks:
            selection_reason, selection_anchor_ids, selection_origin_ids, filtered = (
                _slice_selection(
                    change=change,
                    hunk=hunk,
                    local_results=local_results,
                    anchor_by_id=anchor_by_id,
                )
            )
            if filtered is not None:
                filter_counts[filtered] += 1
                continue
            assert selection_reason is not None
            payload = {
                "schema_version": SCHEMA_VERSION,
                "evidence_version": GIT_EVIDENCE_VERSION,
                "case_id": manifest.case_id,
                "snapshot_digest": manifest.snapshot_digest,
                "commit_id": metadata.commit_id,
                "parent_id": change.parent_id,
                "commit_parents": list(metadata.parents),
                "commit_time": metadata.commit_time,
                "commit_message_context_status": (
                    metadata.commit_message_context_status
                ),
                "commit_message_context": metadata.commit_message_context,
                "commit_message_context_sha256": (
                    metadata.commit_message_context_sha256
                ),
                "comparison_kind": comparison_kind,
                "change_kind": _change_kind(change),
                "before_path": change.before_path,
                "after_path": change.after_path,
                "old_blob_id": change.old_blob_id,
                "new_blob_id": change.new_blob_id,
                "hunk_header": hunk_header,
                "hunk": hunk,
                "patch_sha256": hashlib.sha256(
                    (hunk or "").encode("utf-8")
                ).hexdigest(),
                "query_command_ids": sorted(
                    {result.command_id for result in local_results}
                ),
                "query_families": sorted(
                    {result.family for result in local_results}
                ),
                "anchor_ids": local_anchor_ids,
                "selection_reason": selection_reason,
                "selection_anchor_ids": selection_anchor_ids,
                "selection_origin_ids": selection_origin_ids,
            }
            item = _make_slice(payload)
            previous = eligible.get(item.item_id)
            if previous is None:
                eligible[item.item_id] = _SelectedSliceCandidate(
                    item=item,
                    priority=(
                        0
                        if selection_reason == "strong_changed_line_anchor_match"
                        else 1
                        if selection_reason == "direct_pure_rename"
                        else 2
                    ),
                    encounter_ordinal=encounter_ordinal,
                )
                encounter_ordinal += 1
                continue
            previous_payload = previous.item.model_dump(
                mode="python", exclude={"item_id", "artifact_digest"}
            )
            previous_payload["query_command_ids"] = sorted(
                set(previous.item.query_command_ids) | set(payload["query_command_ids"])
            )
            previous_payload["query_families"] = sorted(
                set(previous.item.query_families) | set(payload["query_families"])
            )
            previous_payload["anchor_ids"] = sorted(
                set(previous.item.anchor_ids) | set(local_anchor_ids)
            )
            if selection_reason == "strong_changed_line_anchor_match" or (
                previous.item.selection_reason == "weak_changed_line_anchor_match"
                and selection_reason == "direct_pure_rename"
            ):
                previous_payload["selection_reason"] = selection_reason
                previous_payload["selection_anchor_ids"] = selection_anchor_ids
                previous_payload["selection_origin_ids"] = selection_origin_ids
            elif selection_reason == previous.item.selection_reason:
                previous_payload["selection_anchor_ids"] = sorted(
                    set(previous.item.selection_anchor_ids)
                    | set(selection_anchor_ids)
                )
                previous_payload["selection_origin_ids"] = sorted(
                    set(previous.item.selection_origin_ids)
                    | set(selection_origin_ids)
                )
            merged = _make_slice(previous_payload)
            eligible[item.item_id] = _SelectedSliceCandidate(
                item=merged,
                priority=min(
                    previous.priority,
                    0
                    if selection_reason == "strong_changed_line_anchor_match"
                    else 1
                    if selection_reason == "direct_pure_rename"
                    else 2,
                ),
                encounter_ordinal=previous.encounter_ordinal,
            )

    ordered_eligible = sorted(
        eligible.values(),
        key=lambda item: (item.priority, item.encounter_ordinal, item.item.item_id),
    )
    emitted = ordered_eligible[: limits.max_change_slices]
    capacity_omitted = max(0, len(ordered_eligible) - len(emitted))
    if capacity_omitted:
        skipped.append("change_slice_limit")
    retention_counts: Counter[str] = Counter(
        item.item.selection_reason for item in ordered_eligible
    )
    all_origins = {
        origin_id for anchor in anchors for origin_id in anchor.origin_ids
    }
    precise_origins = {
        origin_id
        for anchor in anchors
        for origin_id in anchor.precise_origin_ids
    }
    uncovered = sorted(all_origins - precise_origins)
    audit = GitSliceSelectionAudit(
        evaluated_count=len(ordered_eligible) + sum(filter_counts.values()),
        eligible_count=len(ordered_eligible),
        emitted_count=len(emitted),
        capacity_omitted_count=capacity_omitted,
        filtered_count=sum(filter_counts.values()),
        retention_reason_counts=dict(sorted(retention_counts.items())),
        filter_reason_counts=dict(sorted(filter_counts.items())),
        uncovered_origin_ids=uncovered,
        incomplete_reason_codes=(
            ["slice_selection_origin_without_precise_anchor"]
            if uncovered
            else []
        ),
    )
    return [item.item for item in emitted], audit, sorted(set(skipped))


def _make_slice(payload: dict[str, object]) -> GitChangeSlice:
    identity_payload = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "query_command_ids",
            "query_families",
            "anchor_ids",
            "selection_reason",
            "selection_anchor_ids",
            "selection_origin_ids",
        }
    }
    item_id = _stable_id("GITITEM", identity_payload)
    with_id = payload | {"item_id": item_id}
    return GitChangeSlice.model_validate(
        with_id | {"artifact_digest": _digest(with_id)}
    )


def _make_bundle(payload: dict[str, object]) -> GitTemporalBundle:
    payload = {
        "raw_execution": {"logical_calls": 0, "physical_calls": 0, "cache_hits": 0},
        "raw_execution_scope": "collection_only",
        **payload,
    }
    payload["raw_execution"] = GitRawExecution.model_validate(payload["raw_execution"]).model_dump(mode="json")
    bundle_id = _stable_id("GITBUNDLE", payload)
    with_id = payload | {"bundle_id": bundle_id}
    return GitTemporalBundle.model_validate(
        with_id | {"artifact_digest": _digest(with_id)}
    )


def _make_topology(payload: dict[str, object]) -> GitTopologyOrder:
    return GitTopologyOrder.model_validate(
        payload | {"artifact_digest": _digest(payload)}
    )


def _empty_topology(status: Literal["not_required", "partial", "failed"], reason: str) -> GitTopologyOrder:
    payload = {
        "status": status,
        "reason_codes": [reason],
        "traversal_argv": [],
        "groups": [],
        "incomparable_pairs": [],
        "stdout_sha256": hashlib.sha256(b"").hexdigest(),
    }
    return _make_topology(payload)


def _topology_order(
    runner: GitRunner,
    root: Path,
    cutoff_commit: str,
    selected_commits: Sequence[str],
    limits: GitLimits,
) -> GitTopologyOrder:
    """Build exact ancestor layers without using commit timestamps.

    Members of one layer are pairwise incomparable.  Pairs in different layers
    can also be incomparable, so every such pair is retained explicitly.
    """

    selected = set(selected_commits)
    if not selected:
        return _empty_topology("partial", "no_temporal_items_for_topology")
    argv = ["rev-list", "--topo-order", "--reverse", "--parents", cutoff_commit]
    outcome = runner.run(
        root,
        argv,
        timeout_seconds=limits.object_scan_timeout_seconds,
        max_output_bytes=limits.max_object_scan_output_bytes,
    )
    base = {
        "traversal_argv": argv,
        "stdout_sha256": hashlib.sha256(outcome.stdout).hexdigest(),
    }
    if outcome.timed_out or outcome.stdout_truncated or outcome.stderr_truncated:
        return _make_topology(
            base
            | {
                "status": "partial",
                "reason_codes": ["topology_traversal_incomplete"],
                "groups": [],
                "incomparable_pairs": [],
            }
        )
    if outcome.return_code != 0:
        return _make_topology(
            base
            | {
                "status": "failed",
                "reason_codes": ["topology_traversal_failed"],
                "groups": [],
                "incomparable_pairs": [],
            }
        )
    try:
        lines = outcome.stdout.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError:
        return _make_topology(
            base
            | {
                "status": "failed",
                "reason_codes": ["topology_traversal_invalid"],
                "groups": [],
                "incomparable_pairs": [],
            }
        )
    selected_ancestors: dict[str, set[str]] = {}
    observed_selected: set[str] = set()
    for line in lines:
        fields = line.split()
        if not fields or any(not _OID_RE.fullmatch(value) for value in fields):
            return _make_topology(
                base
                | {
                    "status": "failed",
                    "reason_codes": ["topology_traversal_invalid"],
                    "groups": [],
                    "incomparable_pairs": [],
                }
            )
        commit_id, parents = fields[0], fields[1:]
        ancestors: set[str] = set()
        for parent_id in parents:
            ancestors.update(selected_ancestors.get(parent_id, set()))
            if parent_id in selected:
                ancestors.add(parent_id)
        selected_ancestors[commit_id] = ancestors
        if commit_id in selected:
            observed_selected.add(commit_id)
    if observed_selected != selected:
        return _make_topology(
            base
            | {
                "status": "partial",
                "reason_codes": ["topology_selected_commit_missing"],
                "groups": [],
                "incomparable_pairs": [],
            }
        )
    remaining = set(selected)
    groups: list[dict[str, object]] = []
    ordinal = 0
    while remaining:
        layer = sorted(
            commit_id
            for commit_id in remaining
            if not (selected_ancestors[commit_id] & remaining)
        )
        if not layer:
            return _make_topology(
                base
                | {
                    "status": "failed",
                    "reason_codes": ["topology_cycle_or_invalid_ancestry"],
                    "groups": [],
                    "incomparable_pairs": [],
                }
            )
        groups.append({"ordinal": ordinal, "commit_ids": layer})
        remaining.difference_update(layer)
        ordinal += 1
    incomparable: list[dict[str, str]] = []
    ordered = sorted(selected)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if (
                left not in selected_ancestors[right]
                and right not in selected_ancestors[left]
            ):
                incomparable.append(
                    {"left_commit_id": left, "right_commit_id": right}
                )
    return _make_topology(
        base
        | {
            "status": "complete",
            "reason_codes": ["topology_complete"],
            "groups": groups,
            "incomparable_pairs": incomparable,
        }
    )


def _not_required_bundle(
    repository_root: Path,
    *,
    case_id: str,
    snapshot_digest: str,
    cutoff_commit: str,
    purpose: GitQueryPurpose,
    limits: GitLimits,
) -> GitTemporalBundle:
    preflight = GitPreflight(
        repository_root=repository_root.resolve().as_posix(),
        requested_cutoff_commit=cutoff_commit,
        status="not_required",
        reason_codes=["git_not_required"],
        git_executable=None,
        git_version=None,
        object_format=None,
        head_commit=None,
        cutoff_commit=None,
        clean=None,
        shallow=None,
        partial_clone=None,
        alternates=None,
        reachable_objects_complete=None,
        reachable_object_count=None,
        checks=[],
    )
    families = [
        _empty_family_result(family, "not_required", "git_not_required")
        for family in _FAMILY_ORDER
    ]
    manifest = _manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_version": GIT_EVIDENCE_VERSION,
            "case_id": case_id,
            "snapshot_digest": snapshot_digest,
            "temporal_cutoff_commit": cutoff_commit,
            "purpose": purpose,
            "status": "not_required",
            "reason_codes": ["git_not_required"],
            "limits": limits.model_dump(mode="json"),
            "anchors": [],
            "commands": [],
            "command_results": [],
            "family_results": [item.model_dump(mode="json") for item in families],
            "returned_count": 0,
            "deduplicated_count": 0,
            "truncated_count": 0,
            "commit_ids": [],
        }
    )
    return _make_bundle(
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_version": GIT_EVIDENCE_VERSION,
            "case_id": case_id,
            "snapshot_digest": snapshot_digest,
            "temporal_cutoff_commit": cutoff_commit,
            "status": "not_required",
            "reason_codes": ["git_not_required"],
            "preflight": preflight.model_dump(mode="json"),
            "manifest": manifest.model_dump(mode="json"),
            "topology": _empty_topology(
                "not_required", "git_not_required"
            ).model_dump(mode="json"),
            "slices": [],
            "selection_audit": _empty_selection_audit().model_dump(mode="json"),
            "skipped_change_reasons": [],
        }
    )


def collect_git_evidence(
    repository_root: Path,
    *, case_id: str, snapshot_digest: str, cutoff_commit: str,
    purpose: GitQueryPurpose, anchors: Sequence[GitAnchor],
    required: bool = True, limits: GitLimits | None = None,
    runner: GitRunner | None = None, preflight: GitPreflight | None = None,
) -> GitTemporalBundle:
    effective_runner = runner if isinstance(runner, RunLocalGitRunner) else RunLocalGitRunner(runner or BoundedGitRunner(), snapshot_digest=snapshot_digest)
    if effective_runner.snapshot_digest != snapshot_digest:
        raise ValueError("Git raw cache belongs to another snapshot")
    before = effective_runner.counters()
    bundle = _collect_git_evidence(
        repository_root, case_id=case_id, snapshot_digest=snapshot_digest,
        cutoff_commit=cutoff_commit, purpose=purpose, anchors=anchors,
        required=required, limits=limits, runner=effective_runner, preflight=preflight,
    )
    payload = bundle.model_dump(mode="json", exclude={"bundle_id", "artifact_digest"})
    payload["raw_execution"] = {key: value - before[key] for key, value in effective_runner.counters().items()}
    payload["raw_execution_scope"] = (
        "collection_with_preflight" if preflight is None and required else "collection_only"
    )
    return _make_bundle(payload)


def _collect_git_evidence(
    repository_root: Path,
    *,
    case_id: str,
    snapshot_digest: str,
    cutoff_commit: str,
    purpose: GitQueryPurpose,
    anchors: Sequence[GitAnchor],
    required: bool = True,
    limits: GitLimits | None = None,
    runner: GitRunner | None = None,
    preflight: GitPreflight | None = None,
) -> GitTemporalBundle:
    """Preflight, query, slice and deduplicate one candidate case."""

    effective_limits = limits or GitLimits()
    if not required:
        return _not_required_bundle(
            repository_root,
            case_id=case_id,
            snapshot_digest=snapshot_digest,
            cutoff_commit=cutoff_commit,
            purpose=purpose,
            limits=effective_limits,
        )
    effective_runner = runner or BoundedGitRunner()
    if preflight is None:
        preflight = preflight_git_repository(
            repository_root,
            cutoff_commit,
            limits=effective_limits,
            runner=effective_runner,
        )
    elif (
        preflight.repository_root != repository_root.resolve().as_posix()
        or preflight.requested_cutoff_commit != cutoff_commit
    ):
        raise ValueError("reused Git preflight belongs to another repository/cutoff")
    manifest = run_git_queries(
        repository_root,
        case_id=case_id,
        snapshot_digest=snapshot_digest,
        cutoff_commit=cutoff_commit,
        purpose=purpose,
        anchors=anchors,
        preflight=preflight,
        limits=effective_limits,
        runner=effective_runner,
    )
    if manifest.status in {"unavailable", "failed", "bounded_no_match"}:
        selection_audit = _empty_selection_audit(manifest.anchors)
        status = manifest.status
        reasons = set(manifest.reason_codes)
        if manifest.status == "bounded_no_match" and (
            selection_audit.incomplete_reason_codes
        ):
            status = "partial"
            reasons.update(selection_audit.incomplete_reason_codes)
        return _make_bundle(
            {
                "schema_version": SCHEMA_VERSION,
                "evidence_version": GIT_EVIDENCE_VERSION,
                "case_id": case_id,
                "snapshot_digest": snapshot_digest,
                "temporal_cutoff_commit": cutoff_commit,
                "status": status,
                "reason_codes": sorted(reasons),
                "preflight": preflight.model_dump(mode="json"),
                "manifest": manifest.model_dump(mode="json"),
                "topology": _empty_topology(
                    "failed" if manifest.status == "failed" else "partial",
                    "git_query_has_no_temporal_items",
                ).model_dump(mode="json"),
                "slices": [],
                "selection_audit": selection_audit.model_dump(mode="json"),
                "skipped_change_reasons": [],
            }
        )
    slices, selection_audit, skipped = _collect_change_slices(
        effective_runner,
        repository_root.resolve(),
        manifest,
    )

    topology = _topology_order(
        effective_runner,
        repository_root.resolve(),
        cutoff_commit,
        sorted({item.commit_id for item in slices}),
        effective_limits,
    )
    if topology.status == "complete":
        topology_rank = {
            commit_id: group.ordinal
            for group in topology.groups
            for commit_id in group.commit_ids
        }
        # Members in one topology layer are explicitly incomparable.  Their
        # lexical order is only canonical serialization, never temporal order.
        slices.sort(
            key=lambda item: (
                topology_rank[item.commit_id],
                item.commit_id,
                item.item_id,
            )
        )
    else:
        # No timestamp fallback: without a validated ancestor traversal there
        # is no semantic temporal order.  Stable IDs only canonicalize storage.
        slices.sort(key=lambda item: item.item_id)
    status = manifest.status
    reasons = set(manifest.reason_codes)
    reasons.update(selection_audit.incomplete_reason_codes)
    if skipped:
        reasons.update(skipped)
        if status == "complete":
            status = "partial"
    if selection_audit.incomplete_reason_codes and status == "complete":
        status = "partial"
    complete_policy_no_match = (
        manifest.status == "complete"
        and not slices
        and not skipped
        and not selection_audit.incomplete_reason_codes
        and selection_audit.eligible_count == 0
    )
    if complete_policy_no_match:
        status = "bounded_no_match"
        reasons.add("slice_selection_complete_no_match")
    elif manifest.commit_ids and not slices:
        reasons.add("no_valid_change_slice")
        if status == "complete":
            status = "partial"
    if slices and topology.status != "complete":
        reasons.update(topology.reason_codes)
        if status == "complete":
            status = "partial"
    return _make_bundle(
        {
            "schema_version": SCHEMA_VERSION,
            "evidence_version": GIT_EVIDENCE_VERSION,
            "case_id": case_id,
            "snapshot_digest": snapshot_digest,
            "temporal_cutoff_commit": cutoff_commit,
            "status": status,
            "reason_codes": sorted(reasons) or ["git_evidence_complete"],
            "preflight": preflight.model_dump(mode="json"),
            "manifest": manifest.model_dump(mode="json"),
            "topology": topology.model_dump(mode="json"),
            "slices": [item.model_dump(mode="json") for item in slices],
            "selection_audit": selection_audit.model_dump(mode="json"),
            "skipped_change_reasons": sorted(set(skipped)),
        }
    )


__all__ = [
    "BoundedGitRunner",
    "GIT_EVIDENCE_VERSION",
    "GIT_SLICE_SELECTION_VERSION",
    "GitAnchor",
    "GitChangeSlice",
    "GitCommandResult",
    "GitCommandSpec",
    "GitEvidenceStatus",
    "GitLimits",
    "GitPreflight",
    "GitPreflightCheck",
    "GitProcessOutcome",
    "GitQueryFamilyResult",
    "GitQueryManifest",
    "GitQueryPurpose",
    "GitRunner",
    "RunLocalGitRunner",
    "GitTemporalBundle",
    "GitSliceSelectionAudit",
    "GitTopologyGroup",
    "GitTopologyOrder",
    "GitIncomparablePair",
    "collect_git_evidence",
    "plan_git_queries",
    "preflight_git_repository",
    "run_git_queries",
]
