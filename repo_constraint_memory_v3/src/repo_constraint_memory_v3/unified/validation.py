"""Offline structural, policy, reference, and delivery validation.

The validator never invokes a provider and never performs Git history
discovery.  Git operations bind the caller-authorized repository top-level and
HEAD, then use only bounded exact lookups for commit/parent/path/patch facts
already named by saved temporal artifacts.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import mmap
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TypeVar

from pydantic import BaseModel, RootModel, ValidationError, create_model

from repo_constraint_memory_v3.inference import (
    InferenceIndex,
    InferenceRecord,
    contractual_request_bytes,
    max_provider_calls_per_request,
    protocol_manifest_for_readonly_replay,
)
from repo_constraint_memory_v3.models import (
    BuildConfig,
    RepositoryMetadata,
    SCHEMA_VERSION,
    StrictModel,
)
from repo_constraint_memory_v3.unified.consumer import (
    CanonicalClaimsFile,
    CanonicalCompletenessFile,
    CanonicalDeliveryReport,
    ConsumerArtifacts,
    build_consumer_artifacts,
)
from repo_constraint_memory_v3.unified.export import canonical_json_text
from repo_constraint_memory_v3.unified.git_evidence import (
    BoundedGitRunner,
    GitChangeSlice,
    GitQueryManifest,
    GitTemporalBundle,
)
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    canonical_bytes,
    content_digest,
    seal_payload,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    ChannelAssessment,
    CurrentEvidenceBundle,
    ImmutableArtifactRef,
    MemberDisposition,
    RefinementRecord,
    TemporalRelationMatrix,
    UnifiedBundle,
    UnifiedCaseFinding,
    UnifiedFunnel,
    UnifiedIndex,
)
from repo_constraint_memory_v3.unified.policy import (
    apply_case_policy,
    is_explicit_fast_path,
)
from repo_constraint_memory_v3.unified.publication import _bounded_tree_paths
from repo_constraint_memory_v3.unified.semantic import (
    allowed_verifier_evidence_refs,
    git_manifest_ref,
    replay_candidate_semantic_group_record,
    replay_temporal_evidence_record,
    replay_unified_case_verify_record,
)
from repo_constraint_memory_v3.unified.upstream import UnifiedUpstreamInputs


_MAX_FILES = 100_000
_STREAMING_JSON_THRESHOLD_BYTES = 8 * 1024 * 1024
_SUCCESS_CAPABILITY = {"complete", "bounded_no_match", "not_required"}
UNIFIED_VALIDATOR_VERSION = "unified-artifact-validator-3.2"
INCOMPLETE_PACKAGE_ERROR = (
    "complete package required but unified package is partial"
)
_CURRENT_INFERENCE_STAGES = frozenset(
    {
        "current_semantic_map",
        "current_hypothesis_reduce",
        "population_semantic_classify",
        "counterexample_verify",
    }
)
_UNIFIED_INFERENCE_STAGES = frozenset(
    {
        "candidate_semantic_group",
        "temporal_evidence_classify",
        "unified_case_verify",
    }
)
_BUDGET_EXHAUSTION_REASONS = frozenset(
    {
        "max_total_calls",
        "max_total_input_tokens",
        "max_total_output_tokens",
    }
)
T = TypeVar("T", bound=BaseModel)


_UncheckedUnifiedUpstreamInputs = create_model(
    "_UncheckedUnifiedUpstreamInputs",
    __base__=StrictModel,
    **{
        name: (field.annotation, field)
        for name, field in UnifiedUpstreamInputs.model_fields.items()
    },
)


def _is_unified_refinement_assessment(item: ChannelAssessment) -> bool:
    return (
        "unified_refinement_child" in item.typed_reasons
        or "unified_refinement_round:1" in item.provenance
    )


@dataclass(frozen=True)
class _FileSnapshot:
    """Identity needed to safely reopen one already-audited artifact."""

    size: int
    sha256: str


@dataclass(frozen=True)
class _TreeSnapshot:
    root: Path
    files: dict[str, _FileSnapshot]
    canonical_json_files: frozenset[str]
    json_content_digests: dict[str, str]
    json_value_digests: dict[str, dict[str, str]]
    directories: frozenset[str]
    modes: dict[str, int]
    total_bytes: int


@dataclass(frozen=True)
class _UpstreamReplayBinding:
    """Compact equality commitments retained across channel replay."""

    explicit_decision_digests: tuple[str, ...]
    explicit_verification_digest: str
    current_source_digest: str | None
    observation_shards_digest: str | None
    raw_candidate_events: int
    explicit_discovery_complete: bool


@dataclass
class _UnifiedOmissionLedger:
    """Consume saved compact unified omissions against deterministic replay."""

    remaining: dict[str, int]
    omission_reasons: dict[str, dict[str, int]]
    max_calls: int
    max_input_tokens: int
    max_output_tokens: int
    logical_calls: int = 0
    logical_input_tokens: int = 0
    logical_output_tokens: int = 0

    @classmethod
    def from_index(
        cls,
        index: InferenceIndex | None,
        llm_settings: object,
        errors: list[str],
    ) -> _UnifiedOmissionLedger:
        max_calls = int(getattr(llm_settings, "max_total_calls"))
        if index is None:
            return cls(
                remaining={},
                omission_reasons={},
                max_calls=max_calls,
                max_input_tokens=int(
                    getattr(llm_settings, "max_total_input_tokens")
                ),
                max_output_tokens=int(
                    getattr(llm_settings, "max_total_output_tokens")
                ),
            )
        if index.total_calls > max_calls:
            errors.append(
                "inference index total calls exceed configured hard limit"
            )
        if index.total_calls >= max_calls:
            budget_reason = "max_total_calls"
        elif index.total_input_tokens >= int(
            getattr(llm_settings, "max_total_input_tokens")
        ):
            budget_reason = "max_total_input_tokens"
        elif index.total_output_tokens >= int(
            getattr(llm_settings, "max_total_output_tokens")
        ):
            budget_reason = "max_total_output_tokens"
        else:
            budget_reason = None

        omitted_stages = [item for item in index.stages if item.omitted]
        if omitted_stages and budget_reason is None:
            errors.append(
                "inference omissions exist before any global budget is exhausted"
            )
        known_replayed = _CURRENT_INFERENCE_STAGES | _UNIFIED_INFERENCE_STAGES
        for item in omitted_stages:
            if item.stage not in known_replayed:
                errors.append(
                    "inference omission has no deterministic replay owner: "
                    f"{item.stage}"
                )
            if (
                budget_reason is None
                or item.omission_reasons != {budget_reason: item.omitted}
            ):
                errors.append(
                    "inference omission reason differs from the exhausted global "
                    f"budget: {item.stage}"
                )
        return cls(
            remaining={
                item.stage: item.omitted
                for item in omitted_stages
                if item.stage in _UNIFIED_INFERENCE_STAGES
            },
            omission_reasons={
                item.stage: dict(item.omission_reasons)
                for item in omitted_stages
                if item.stage in _UNIFIED_INFERENCE_STAGES
            },
            max_calls=max_calls,
            max_input_tokens=int(
                getattr(llm_settings, "max_total_input_tokens")
            ),
            max_output_tokens=int(
                getattr(llm_settings, "max_total_output_tokens")
            ),
        )

    def seed_channel_records(self, records: Sequence[InferenceRecord]) -> None:
        for record in records:
            if record.stage not in _UNIFIED_INFERENCE_STAGES:
                self.record_terminal(record, admitted=True)

    def exhaustion_reason(self) -> str | None:
        if self.logical_calls >= self.max_calls:
            return "max_total_calls"
        if self.logical_input_tokens >= self.max_input_tokens:
            return "max_total_input_tokens"
        if self.logical_output_tokens >= self.max_output_tokens:
            return "max_total_output_tokens"
        return None

    def can_admit_parallel_wave(
        self,
        logical_request_capacities: list[int],
        *,
        attempts_per_request: int,
    ) -> bool:
        """Replay the runtime's all-or-serial call-capacity decision."""

        if (
            len(logical_request_capacities) <= 1
            or attempts_per_request < 1
            or any(value < 1 for value in logical_request_capacities)
            or self.exhaustion_reason() is not None
        ):
            return False
        required = sum(logical_request_capacities) * attempts_per_request
        return self.logical_calls + required <= self.max_calls

    def record_terminal(
        self,
        record: InferenceRecord,
        *,
        admitted: bool,
    ) -> bool:
        if not admitted and self.exhaustion_reason() is not None:
            return False
        if self.logical_calls + record.usage.calls > self.max_calls:
            return False
        self.logical_calls += record.usage.calls
        self.logical_input_tokens += record.usage.input_tokens
        self.logical_output_tokens += record.usage.output_tokens
        return True

    def consume(self, stage: str) -> tuple[bool, str | None]:
        count = self.remaining.get(stage, 0)
        reason = self.exhaustion_reason()
        if count <= 0 or reason is None:
            return False, None
        reason_counts = self.omission_reasons.get(stage, {})
        if reason_counts.get(reason, 0) <= 0:
            return False, reason
        if count == 1:
            self.remaining.pop(stage, None)
        else:
            self.remaining[stage] = count - 1
        if reason_counts[reason] == 1:
            reason_counts.pop(reason)
        else:
            reason_counts[reason] -= 1
        if not reason_counts:
            self.omission_reasons.pop(stage, None)
        return True, reason

    def finish(self) -> list[str]:
        return [
            f"unconsumed compact inference omissions: {stage}={count}"
            for stage, count in sorted(self.remaining.items())
            if count
        ]


def _short_error(error: Exception) -> str:
    return str(error).splitlines()[0][:300] or type(error).__name__


def _read_regular(path: Path, *, remaining: int) -> tuple[bytes | None, str | None]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            return None, "not a regular file"
        if info.st_size > remaining:
            return None, "artifact byte bound exceeded"
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            value = handle.read(remaining + 1)
        if len(value) > remaining:
            return None, "artifact byte bound exceeded"
        return value, None
    except OSError as error:
        return None, f"safe read failed: {_short_error(error)}"
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _json_no_duplicates(text: str) -> object:
    def hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(
        text,
        object_pairs_hook=hook,
        parse_constant=reject_constant,
    )


class _CanonicalJsonError(ValueError):
    """A large JSON artifact is malformed or outside the canonical grammar."""


class _CanonicalJsonScanner:
    """Bounded-memory verifier for the pretty canonical JSON export format."""

    def __init__(
        self,
        data: mmap.mmap,
        *,
        excluded_top_key: str | None = None,
        captured_top_keys: frozenset[str] = frozenset(),
    ) -> None:
        self._data = data
        self._length = len(data)
        self._position = 0
        self._excluded_top_key = excluded_top_key
        self._content_hasher = (
            hashlib.sha256() if excluded_top_key is not None else None
        )
        self._captured_top_keys = captured_top_keys
        self._capture_hasher: object | None = None
        self._captured_digests: dict[str, str] = {}

    def validate(self) -> tuple[str | None, dict[str, str]]:
        self._parse_value(depth=0, emit=True, top_level=True)
        if self._position >= self._length or self._data[self._position] != 0x0A:
            raise _CanonicalJsonError("canonical JSON requires one final newline")
        self._position += 1
        if self._position != self._length:
            raise _CanonicalJsonError("canonical JSON has trailing bytes")
        return (
            (
                self._content_hasher.hexdigest()
                if self._content_hasher is not None
                else None
            ),
            dict(self._captured_digests),
        )

    def _emit(self, value: bytes, *, enabled: bool) -> None:
        if enabled and self._content_hasher is not None:
            self._content_hasher.update(value)
        if enabled and self._capture_hasher is not None:
            self._capture_hasher.update(value)  # type: ignore[attr-defined]

    def _expect(self, value: bytes) -> None:
        end = self._position + len(value)
        if end > self._length or self._data[self._position : end] != value:
            raise _CanonicalJsonError("JSON token or canonical whitespace mismatch")
        self._position = end

    def _expect_indent(self, depth: int) -> None:
        count = depth * 2
        end = self._position + count
        if (
            end > self._length
            or self._data[self._position : end] != b" " * count
        ):
            raise _CanonicalJsonError("JSON indentation is not canonical")
        self._position = end

    def _parse_value(
        self,
        *,
        depth: int,
        emit: bool,
        top_level: bool = False,
    ) -> None:
        if self._position >= self._length:
            raise _CanonicalJsonError("truncated JSON value")
        marker = self._data[self._position]
        if marker == 0x7B:
            self._parse_object(depth=depth, emit=emit, top_level=top_level)
        elif marker == 0x5B:
            self._parse_array(depth=depth, emit=emit)
        elif marker == 0x22:
            _, raw = self._parse_string()
            self._emit(raw, enabled=emit)
        elif marker == 0x2D or 0x30 <= marker <= 0x39:
            raw = self._parse_number()
            self._emit(raw, enabled=emit)
        elif marker == 0x74:
            self._expect(b"true")
            self._emit(b"true", enabled=emit)
        elif marker == 0x66:
            self._expect(b"false")
            self._emit(b"false", enabled=emit)
        elif marker == 0x6E:
            self._expect(b"null")
            self._emit(b"null", enabled=emit)
        else:
            raise _CanonicalJsonError("invalid JSON value")

    def _parse_object(self, *, depth: int, emit: bool, top_level: bool) -> None:
        self._position += 1
        self._emit(b"{", enabled=emit)
        if self._position < self._length and self._data[self._position] == 0x7D:
            self._position += 1
            self._emit(b"}", enabled=emit)
            return
        self._expect(b"\n")
        prior_key: str | None = None
        emitted_members = 0
        while True:
            self._expect_indent(depth + 1)
            key, raw_key = self._parse_string()
            if prior_key is not None and key <= prior_key:
                raise _CanonicalJsonError(
                    "JSON object keys are duplicated or not canonically sorted"
                )
            prior_key = key
            self._expect(b": ")
            excluded = top_level and key == self._excluded_top_key
            member_emit = emit and not excluded
            if member_emit:
                if emitted_members:
                    self._emit(b",", enabled=True)
                self._emit(raw_key, enabled=True)
                self._emit(b":", enabled=True)
                emitted_members += 1
            prior_capture = self._capture_hasher
            if top_level and key in self._captured_top_keys:
                self._capture_hasher = hashlib.sha256()
            self._parse_value(depth=depth + 1, emit=member_emit)
            if top_level and key in self._captured_top_keys:
                assert self._capture_hasher is not None
                self._captured_digests[key] = (
                    self._capture_hasher.hexdigest()  # type: ignore[attr-defined]
                )
                self._capture_hasher = prior_capture
            if self._position >= self._length:
                raise _CanonicalJsonError("truncated JSON object")
            marker = self._data[self._position]
            if marker == 0x2C:
                self._position += 1
                self._expect(b"\n")
                continue
            if marker != 0x0A:
                raise _CanonicalJsonError("JSON object separators are not canonical")
            self._position += 1
            self._expect_indent(depth)
            self._expect(b"}")
            self._emit(b"}", enabled=emit)
            return

    def _parse_array(self, *, depth: int, emit: bool) -> None:
        self._position += 1
        self._emit(b"[", enabled=emit)
        if self._position < self._length and self._data[self._position] == 0x5D:
            self._position += 1
            self._emit(b"]", enabled=emit)
            return
        self._expect(b"\n")
        ordinal = 0
        while True:
            self._expect_indent(depth + 1)
            if ordinal:
                self._emit(b",", enabled=emit)
            self._parse_value(depth=depth + 1, emit=emit)
            ordinal += 1
            if self._position >= self._length:
                raise _CanonicalJsonError("truncated JSON array")
            marker = self._data[self._position]
            if marker == 0x2C:
                self._position += 1
                self._expect(b"\n")
                continue
            if marker != 0x0A:
                raise _CanonicalJsonError("JSON array separators are not canonical")
            self._position += 1
            self._expect_indent(depth)
            self._expect(b"]")
            self._emit(b"]", enabled=emit)
            return

    def _parse_string(self) -> tuple[str, bytes]:
        start = self._position
        search_from = start + 1
        while True:
            end = self._data.find(b'"', search_from)
            if end < 0:
                raise _CanonicalJsonError("unterminated JSON string")
            backslashes = 0
            cursor = end - 1
            while cursor > start and self._data[cursor] == 0x5C:
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                break
            search_from = end + 1
        self._position = end + 1
        raw = self._data[start : self._position]
        try:
            decoded_raw = raw.decode("utf-8", errors="strict")
            value = json.loads(decoded_raw)
            canonical = json.dumps(value, ensure_ascii=False).encode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError, ValueError) as error:
            raise _CanonicalJsonError("invalid JSON string") from error
        if not isinstance(value, str) or canonical != raw:
            raise _CanonicalJsonError("JSON string is not canonical")
        return value, raw

    def _parse_number(self) -> bytes:
        start = self._position
        while self._position < self._length and self._data[self._position] not in (
            0x0A,
            0x2C,
            0x5D,
            0x7D,
        ):
            self._position += 1
        raw = self._data[start : self._position]
        try:
            text = raw.decode("ascii", errors="strict")
            value = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise _CanonicalJsonError("invalid JSON number") from error
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise _CanonicalJsonError("invalid finite JSON number")
        canonical = json.dumps(value, ensure_ascii=False, allow_nan=False).encode(
            "ascii"
        )
        if raw != canonical:
            raise _CanonicalJsonError("JSON number is not canonical")
        return raw


def _inspect_large_json(
    path: Path,
    *,
    remaining: int,
    exclude_top_digest: bool,
) -> tuple[
    _FileSnapshot | None,
    str | None,
    str | None,
    dict[str, str],
]:
    """Inspect one large JSON file without materializing its object graph.

    The untrusted source is copied through a bounded stream into a private
    temporary file before mmap parsing.  Mapping the mutable source directly
    would let a concurrent truncate terminate the validator with SIGBUS rather
    than produce a typed invalid-artifact result.
    """

    descriptor: int | None = None
    mapped: mmap.mmap | None = None
    temporary = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            return None, "not a regular file", None, {}
        if info.st_size > remaining:
            return None, "artifact byte bound exceeded", None, {}
        if info.st_size == 0:
            return (
                _FileSnapshot(size=0, sha256=hashlib.sha256(b"").hexdigest()),
                "invalid JSON: empty artifact",
                None,
                {},
            )
        temporary = tempfile.TemporaryFile(mode="w+b")
        raw_hasher = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        utf8_error = False
        observed = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, max(1, info.st_size - observed + 1)),
            )
            if not chunk:
                break
            observed += len(chunk)
            if observed > info.st_size:
                return None, "artifact changed during safe read", None, {}
            raw_hasher.update(chunk)
            temporary.write(chunk)
            if not utf8_error:
                try:
                    decoder.decode(chunk, final=False)
                except UnicodeDecodeError:
                    utf8_error = True
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino)
            or after.st_size != info.st_size
            or after.st_mtime_ns != info.st_mtime_ns
            or after.st_ctime_ns != info.st_ctime_ns
            or observed != info.st_size
        ):
            return None, "artifact changed during safe read", None, {}
        if not utf8_error:
            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                utf8_error = True
        metadata = _FileSnapshot(size=info.st_size, sha256=raw_hasher.hexdigest())
        if utf8_error:
            return metadata, "artifact is not UTF-8", None, {}
        temporary.flush()
        mapped = mmap.mmap(temporary.fileno(), 0, access=mmap.ACCESS_READ)
        content_digest_value, value_digests = _CanonicalJsonScanner(
            mapped,
            excluded_top_key=("artifact_digest" if exclude_top_digest else None),
            captured_top_keys=(
                frozenset(
                    {
                        "current_source",
                        "explicit_verification",
                        "observation_shards",
                    }
                )
                if exclude_top_digest
                else frozenset()
            ),
        ).validate()
        return metadata, None, content_digest_value, value_digests
    except (_CanonicalJsonError, RecursionError) as error:
        metadata = (
            _FileSnapshot(size=info.st_size, sha256=raw_hasher.hexdigest())
            if "info" in locals() and "raw_hasher" in locals()
            else None
        )
        return metadata, f"invalid JSON: {_short_error(error)}", None, {}
    except OSError as error:
        return None, f"safe read failed: {_short_error(error)}", None, {}
    finally:
        if mapped is not None:
            mapped.close()
        if temporary is not None:
            temporary.close()
        if descriptor is not None:
            os.close(descriptor)


def _snapshot_tree(
    run_directory: Path,
    *,
    max_artifact_bytes: int,
) -> tuple[_TreeSnapshot | None, list[str]]:
    errors: list[str] = []
    lexical_root = run_directory.absolute()
    try:
        lexical_before = lexical_root.lstat()
    except OSError as error:
        return None, [f"run directory is unavailable: {_short_error(error)}"]
    if stat.S_ISLNK(lexical_before.st_mode):
        return None, ["run directory is a symlink"]
    if not stat.S_ISDIR(lexical_before.st_mode):
        return None, ["run directory is not a directory"]
    try:
        root = lexical_root.resolve(strict=True)
        lexical_resolved = lexical_root.lstat()
        root_before = root.lstat()
    except OSError as error:
        return None, [f"run directory is unavailable: {_short_error(error)}"]
    if (
        not stat.S_ISDIR(lexical_resolved.st_mode)
        or (lexical_resolved.st_dev, lexical_resolved.st_ino)
        != (lexical_before.st_dev, lexical_before.st_ino)
        or not stat.S_ISDIR(root_before.st_mode)
        or (root_before.st_dev, root_before.st_ino)
        != (lexical_before.st_dev, lexical_before.st_ino)
    ):
        return None, ["run directory changed during safe resolution"]
    files: dict[str, _FileSnapshot] = {}
    canonical_json_files: set[str] = set()
    json_content_digests: dict[str, str] = {}
    json_value_digests: dict[str, dict[str, str]] = {}
    directories: set[str] = set()
    modes: dict[str, int] = {}
    total = 0
    try:
        entries = _bounded_tree_paths(
            root,
            label="artifact tree",
            max_entries=_MAX_FILES,
        )
    except ValueError as error:
        return None, [str(error)]
    for path in entries:
        relative = path.relative_to(root).as_posix()
        pure = PurePosixPath(relative)
        if any(part in {"", ".", ".."} for part in pure.parts):
            errors.append(f"unsafe artifact path: {relative}")
            continue
        if path.is_symlink():
            errors.append(f"artifact is a symlink: {relative}")
            continue
        try:
            info = path.stat(follow_symlinks=False)
        except OSError as error:
            errors.append(f"artifact stat failed: {relative}: {_short_error(error)}")
            continue
        modes[relative] = info.st_mode
        if stat.S_ISDIR(info.st_mode):
            directories.add(relative)
            continue
        if not stat.S_ISREG(info.st_mode):
            errors.append(f"artifact is not a regular file: {relative}")
            continue
        remaining = max_artifact_bytes - total
        if (
            relative.endswith(".json")
            and info.st_size >= _STREAMING_JSON_THRESHOLD_BYTES
        ):
            (
                metadata,
                inspection_error,
                json_content_digest,
                value_digests,
            ) = _inspect_large_json(
                path,
                remaining=max(0, remaining),
                exclude_top_digest=(
                    relative == "canonical/audit/upstream-inputs.json"
                ),
            )
            if metadata is None:
                errors.append(
                    f"{relative}: {inspection_error or 'safe read failed'}"
                )
                continue
            total += metadata.size
            files[relative] = metadata
            if inspection_error is not None:
                if inspection_error == "artifact is not UTF-8":
                    errors.append(f"artifact is not UTF-8: {relative}")
                else:
                    label = (
                        "build_report.json parse error"
                        if relative == "build_report.json"
                        else f"invalid JSON artifact: {relative}"
                    )
                    errors.append(f"{label}: {inspection_error}")
                continue
            canonical_json_files.add(relative)
            if json_content_digest is not None:
                json_content_digests[relative] = json_content_digest
            if value_digests:
                json_value_digests[relative] = value_digests
            continue
        data, read_error = _read_regular(path, remaining=max(0, remaining))
        if read_error is not None or data is None:
            errors.append(f"{relative}: {read_error or 'safe read failed'}")
            continue
        total += len(data)
        files[relative] = _FileSnapshot(
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            errors.append(f"artifact is not UTF-8: {relative}")
            continue
        if relative.endswith(".json"):
            try:
                value = _json_no_duplicates(text)
            except (json.JSONDecodeError, ValueError, RecursionError) as error:
                label = (
                    "build_report.json parse error"
                    if relative == "build_report.json"
                    else f"invalid JSON artifact: {relative}"
                )
                errors.append(f"{label}: {_short_error(error)}")
                continue
            try:
                canonical = canonical_json_text(value).encode("utf-8")
            except (ValueError, RecursionError) as error:
                label = (
                    "build_report.json parse error"
                    if relative == "build_report.json"
                    else f"invalid JSON artifact: {relative}"
                )
                errors.append(f"{label}: {_short_error(error)}")
                continue
            if canonical != data:
                errors.append(f"JSON artifact is not canonical: {relative}")
                continue
            canonical_json_files.add(relative)
    try:
        lexical_after = lexical_root.lstat()
        root_after = root.lstat()
    except OSError as error:
        errors.append(f"run directory changed during snapshot: {_short_error(error)}")
    else:
        expected_identity = (root_before.st_dev, root_before.st_ino)
        if (
            stat.S_ISLNK(lexical_after.st_mode)
            or not stat.S_ISDIR(lexical_after.st_mode)
            or not stat.S_ISDIR(root_after.st_mode)
            or (lexical_after.st_dev, lexical_after.st_ino) != expected_identity
            or (root_after.st_dev, root_after.st_ino) != expected_identity
        ):
            errors.append("run directory changed during snapshot")
    return (
        _TreeSnapshot(
            root=root,
            files=files,
            canonical_json_files=frozenset(canonical_json_files),
            json_content_digests=json_content_digests,
            json_value_digests=json_value_digests,
            directories=frozenset(directories),
            modes=modes,
            total_bytes=total,
        ),
        errors,
    )


def _mapping(value: object, label: str, errors: list[str]) -> dict[str, object] | None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        errors.append(f"{label} must be a JSON object")
        return None
    return value


def _parse_model(
    model: type[T],
    value: object,
    label: str,
    errors: list[str],
) -> T | None:
    try:
        return model.model_validate(value)
    except (ValidationError, ValueError) as error:
        errors.append(f"{label} failed validation: {_short_error(error)}")
        return None


def _snapshot_bytes(
    tree: _TreeSnapshot,
    relative: str,
    errors: list[str],
) -> bytes | None:
    """Reopen an artifact without retaining the whole audited tree in memory.

    The initial traversal authenticates every regular file by size and SHA-256.
    Every later read is checked against that identity, so replacing a file
    between traversal and replay is rejected rather than silently validating a
    different tree.
    """

    metadata = tree.files.get(relative)
    if metadata is None:
        return None
    data, read_error = _read_regular(
        tree.root / PurePosixPath(relative),
        remaining=metadata.size,
    )
    if (
        read_error is not None
        or data is None
        or len(data) != metadata.size
        or hashlib.sha256(data).hexdigest() != metadata.sha256
    ):
        errors.append(f"artifact changed after snapshot: {relative}")
        return None
    return data


def _tree_unchanged_errors(tree: _TreeSnapshot) -> list[str]:
    """Recheck the initial closed tree after all lazy replay reads finish.

    Reusing the bounded snapshot scanner preserves the streaming path for
    large JSON instead of materializing every artifact merely to compare its
    digest a second time.
    """

    current, snapshot_errors = _snapshot_tree(
        tree.root,
        max_artifact_bytes=max(1, tree.total_bytes),
    )
    errors = [
        f"artifact final snapshot failed: {item}" for item in snapshot_errors
    ]
    if current is None:
        return sorted(set(errors))
    expected_files = set(tree.files)
    actual_files = set(current.files)
    if actual_files != expected_files:
        errors.append(
            "artifact file set changed after snapshot: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"foreign={sorted(actual_files - expected_files)}"
        )
    if current.directories != tree.directories:
        errors.append(
            "artifact directory set changed after snapshot: "
            f"missing={sorted(tree.directories - current.directories)}, "
            f"foreign={sorted(current.directories - tree.directories)}"
        )
    for relative in sorted(expected_files & actual_files):
        if current.files[relative] != tree.files[relative]:
            errors.append(f"artifact changed after snapshot: {relative}")
    return sorted(set(errors))


def _json_value(
    tree: _TreeSnapshot,
    relative: str,
    errors: list[str],
    *,
    required: bool = False,
) -> object | None:
    if relative not in tree.files:
        if required:
            errors.append(f"required artifact is missing: {relative}")
        return None
    if relative not in tree.canonical_json_files:
        if required:
            errors.append(f"required JSON artifact is unreadable: {relative}")
        return None
    data = _snapshot_bytes(tree, relative, errors)
    if data is None:
        return None
    try:
        # The exact bytes were already checked for UTF-8, duplicate keys,
        # finite values, and canonical encoding during the initial traversal.
        return _json_no_duplicates(data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        errors.append(f"artifact changed after snapshot: {relative}")
        return None


def _parse_model_file(
    tree: _TreeSnapshot,
    relative: str,
    model: type[T],
    errors: list[str],
    *,
    required: bool = False,
    label: str | None = None,
) -> T | None:
    """Validate a canonical JSON artifact without first building a Python DOM."""

    display = label or relative
    if relative not in tree.canonical_json_files:
        if required:
            errors.append(f"{display} failed validation: artifact is unreadable")
        return None
    data = _snapshot_bytes(tree, relative, errors)
    if data is None:
        return None
    try:
        return model.model_validate_json(data)
    except (ValidationError, ValueError) as error:
        errors.append(f"{display} failed validation: {_short_error(error)}")
        return None


def _required_json(tree: _TreeSnapshot, relative: str, errors: list[str]) -> object | None:
    return _json_value(tree, relative, errors, required=True)


def _collection(
    tree: _TreeSnapshot,
    directory: str,
    model: type[T],
    errors: list[str],
) -> list[T]:
    if directory not in tree.directories:
        errors.append(f"required artifact directory is missing: {directory}")
        return []
    prefix = directory + "/"
    values: list[T] = []
    for relative in sorted(
        path for path in tree.files if path.startswith(prefix)
    ):
        suffix = relative[len(prefix) :]
        if "/" in suffix or not suffix.endswith(".json"):
            errors.append(f"unexpected artifact in {directory}: {relative}")
            continue
        parsed = _parse_model_file(tree, relative, model, errors)
        if parsed is not None:
            values.append(parsed)
    return values


def _wrapper_list(
    tree: _TreeSnapshot,
    relative: str,
    key: str,
    model: type[T],
    snapshot_digest: str,
    errors: list[str],
) -> list[T]:
    raw = _required_json(tree, relative, errors)
    mapping = _mapping(raw, relative, errors) if raw is not None else None
    if mapping is None:
        return []
    if mapping.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{relative} schema mismatch")
    if mapping.get("snapshot_digest") != snapshot_digest:
        errors.append(f"{relative} snapshot mismatch")
    raw_items = mapping.get(key)
    if not isinstance(raw_items, list):
        errors.append(f"{relative}.{key} must be a list")
        return []
    result: list[T] = []
    for ordinal, item in enumerate(raw_items):
        parsed = _parse_model(model, item, f"{relative}:{ordinal}", errors)
        if parsed is not None:
            result.append(parsed)
    return result


def _git_head_errors(repository_root: Path, expected_commit: str | None) -> list[str]:
    if expected_commit is None:
        # Non-Git fixed snapshots remain valid, but are necessarily partial for
        # any case whose policy requires temporal evidence.  File-manifest
        # identity is replayed separately from the saved discovery artifact.
        return []
    if repository_root.is_symlink():
        return ["repository root is a symlink"]
    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        return [f"repository root is unavailable: {_short_error(error)}"]
    if not root.is_dir():
        return ["repository root is not a directory"]
    runner = BoundedGitRunner()
    errors: list[str] = []
    top = runner.run(
        root,
        ["rev-parse", "--show-toplevel"],
        timeout_seconds=10,
        max_output_bytes=4096,
    )
    if top.return_code != 0 or top.timed_out or top.stdout_truncated:
        return ["repository Git top-level is unavailable"]
    try:
        actual_top = Path(top.stdout.decode("utf-8", errors="strict").strip()).resolve(
            strict=True
        )
    except (OSError, UnicodeDecodeError):
        return ["repository Git top-level is invalid"]
    if actual_top != root:
        errors.append("caller repository root differs from Git top-level")
    head = runner.run(
        root,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        timeout_seconds=10,
        max_output_bytes=4096,
    )
    if head.return_code != 0 or head.timed_out or head.stdout_truncated:
        errors.append("repository HEAD is unavailable")
    else:
        try:
            actual_head = head.stdout.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError:
            actual_head = ""
        if actual_head != expected_commit:
            errors.append("repository HEAD differs from unified snapshot commit")
    return errors


def _consumer_repository_name(
    tree: _TreeSnapshot,
    repository_root: Path,
    config: BuildConfig | None,
    index: UnifiedIndex,
    errors: list[str],
    *,
    required: bool,
) -> str:
    """Resolve the explicit repository identity without conflating it with a path.

    A configured ``repository.name`` is a stable user-selected identity and may
    legitimately differ from the checkout directory name.  Final runs project
    the strictly typed repository metadata saved in ``build_report.json`` only
    after binding it to the caller-authorized checkout, snapshot, commit, and
    preserved config semantics.  Synthetic audit-only fixtures may omit the
    report and fall back to the same deterministic discovery rule.
    """

    fallback = (
        config.repository.name
        if config is not None and config.repository.name is not None
        else repository_root.resolve().name
    )
    raw_report = (
        _required_json(tree, "build_report.json", errors)
        if required
        else _json_value(tree, "build_report.json", errors)
    )
    if not isinstance(raw_report, dict):
        return fallback
    raw_repository = raw_report.get("repository")
    metadata = _parse_model(
        RepositoryMetadata,
        raw_repository,
        "build_report.json repository",
        errors,
    )
    if metadata is None:
        return fallback
    valid = True
    try:
        saved_root = Path(metadata.root).resolve(strict=True)
        caller_root = repository_root.resolve(strict=True)
    except OSError as error:
        errors.append(
            "build_report repository root cannot be bound to caller: "
            f"{_short_error(error)}"
        )
        valid = False
    else:
        if saved_root != caller_root:
            errors.append("build_report repository root differs from caller root")
            valid = False
    if metadata.snapshot_digest != index.snapshot_digest:
        errors.append("build_report repository snapshot differs from audit index")
        valid = False
    if metadata.git_head_for_reference != index.snapshot_commit:
        errors.append("build_report repository commit differs from audit index")
        valid = False
    if metadata.name != fallback:
        errors.append(
            "build_report repository name differs from preserved config identity"
        )
        valid = False
    return metadata.name if valid else fallback


def _owned_git_artifacts(
    tree: _TreeSnapshot,
    snapshot_digest: str,
    case_ids: set[str],
    errors: list[str],
) -> tuple[list[ImmutableArtifactRef], dict[str, BaseModel]]:
    refs_raw = _required_json(tree, "canonical/audit/git/refs.json", errors)
    refs_mapping = (
        _mapping(refs_raw, "canonical/audit/git/refs.json", errors)
        if refs_raw is not None
        else None
    )
    if refs_mapping is None:
        return [], {}
    if refs_mapping.get("schema_version") != SCHEMA_VERSION:
        errors.append("canonical/audit/git/refs.json schema mismatch")
    if refs_mapping.get("snapshot_digest") != snapshot_digest:
        errors.append("canonical/audit/git/refs.json snapshot mismatch")
    raw_refs = refs_mapping.get("refs")
    if not isinstance(raw_refs, list):
        errors.append("canonical/audit/git/refs.json refs must be a list")
        return [], {}
    refs: list[ImmutableArtifactRef] = []
    for ordinal, raw in enumerate(raw_refs):
        ref = _parse_model(
            ImmutableArtifactRef,
            raw,
            f"canonical/audit/git/refs.json:{ordinal}",
            errors,
        )
        if ref is not None:
            refs.append(ref)
    if refs != sorted(refs, key=lambda item: (item.kind, item.stable_id)):
        errors.append("Git refs are not canonically sorted")
    if len({item.stable_id for item in refs}) != len(refs):
        errors.append("Git refs contain duplicate stable IDs")
    parsed: dict[str, BaseModel] = {}
    model_by_kind: dict[str, type[BaseModel]] = {
        "git_temporal_bundle": GitTemporalBundle,
        "git_manifest": GitQueryManifest,
        "git_change_slice": GitChangeSlice,
    }
    directory_by_kind = {
        "git_temporal_bundle": "bundles",
        "git_manifest": "manifests",
        "git_change_slice": "slices",
    }
    identity_by_kind = {
        "git_temporal_bundle": "bundle_id",
        "git_manifest": "manifest_id",
        "git_change_slice": "item_id",
    }
    for ref in refs:
        if ref.kind not in model_by_kind:
            errors.append(f"unsupported Git ref kind: {ref.kind}")
            continue
        if ref.snapshot_digest != snapshot_digest:
            errors.append(f"Git ref snapshot mismatch: {ref.stable_id}")
        expected_path = (
            "canonical/audit/git/"
            f"{directory_by_kind[ref.kind]}/{ref.stable_id}.json"
        )
        if ref.artifact_path != expected_path:
            errors.append(f"Git ref path is noncanonical: {ref.stable_id}")
        raw = _required_json(tree, expected_path, errors)
        if raw is None:
            continue
        mapping = _mapping(raw, expected_path, errors)
        if mapping is None:
            continue
        if mapping.get(identity_by_kind[ref.kind]) != ref.stable_id:
            errors.append(f"Git payload identity mismatch: {ref.stable_id}")
        if mapping.get("snapshot_digest") != snapshot_digest:
            errors.append(f"Git payload snapshot mismatch: {ref.stable_id}")
        if mapping.get("artifact_digest") != ref.artifact_digest:
            errors.append(f"Git payload/ref digest mismatch: {ref.stable_id}")
        if artifact_digest(mapping) != ref.artifact_digest:
            errors.append(f"Git payload content digest mismatch: {ref.stable_id}")
        value = _parse_model(
            model_by_kind[ref.kind], mapping, expected_path, errors
        )
        if value is not None:
            parsed[ref.stable_id] = value
            case_id = getattr(value, "case_id", None)
            if case_id not in case_ids:
                errors.append(f"Git payload references foreign case: {ref.stable_id}")
    bundles = [item for item in parsed.values() if isinstance(item, GitTemporalBundle)]
    bundle_case_ids = [item.case_id for item in bundles]
    if len(bundle_case_ids) != len(set(bundle_case_ids)):
        errors.append("multiple Git temporal bundles exist for one case")
    expected_refs: list[ImmutableArtifactRef] = []
    for bundle in bundles:
        expected_refs.extend(
            [
                ImmutableArtifactRef(
                    kind="git_temporal_bundle",
                    stable_id=bundle.bundle_id,
                    snapshot_digest=bundle.snapshot_digest,
                    artifact_path=(
                        f"canonical/audit/git/bundles/{bundle.bundle_id}.json"
                    ),
                    artifact_digest=bundle.artifact_digest,
                ),
                ImmutableArtifactRef(
                    kind="git_manifest",
                    stable_id=bundle.manifest.manifest_id,
                    snapshot_digest=bundle.manifest.snapshot_digest,
                    artifact_path=(
                        "canonical/audit/git/manifests/"
                        f"{bundle.manifest.manifest_id}.json"
                    ),
                    artifact_digest=bundle.manifest.artifact_digest,
                ),
                *[
                    ImmutableArtifactRef(
                        kind="git_change_slice",
                        stable_id=item.item_id,
                        snapshot_digest=item.snapshot_digest,
                        artifact_path=(
                            f"canonical/audit/git/slices/{item.item_id}.json"
                        ),
                        artifact_digest=item.artifact_digest,
                    )
                    for item in bundle.slices
                ],
            ]
        )
        manifest = parsed.get(bundle.manifest.manifest_id)
        if manifest is None or manifest.model_dump(mode="json") != bundle.manifest.model_dump(
            mode="json"
        ):
            errors.append(f"Git bundle manifest closure failed: {bundle.bundle_id}")
        separate_slices = {
            item.item_id: item
            for item in parsed.values()
            if isinstance(item, GitChangeSlice)
        }
        for item in bundle.slices:
            saved = separate_slices.get(item.item_id)
            if saved is None or saved.model_dump(mode="json") != item.model_dump(
                mode="json"
            ):
                errors.append(f"Git bundle slice closure failed: {item.item_id}")
    expected_refs.sort(key=lambda item: (item.kind, item.stable_id))
    if refs != expected_refs:
        errors.append(
            "Git refs differ from the exact temporal-bundle artifact closure"
        )
    return refs, parsed


def _git_object_replay_errors(
    repository_root: Path,
    snapshot_commit: str | None,
    git_values: Mapping[str, BaseModel],
) -> list[str]:
    """Rebuild saved slices through exact object lookups, never search queries."""

    from repo_constraint_memory_v3.unified.git_evidence import (
        _collect_change_slices,
    )

    bundles = [
        item for item in git_values.values() if isinstance(item, GitTemporalBundle)
    ]
    if not bundles:
        return []
    if snapshot_commit is None:
        return ["saved Git evidence exists without a snapshot commit"]
    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        return [f"Git object replay repository is unavailable: {_short_error(error)}"]
    runner = BoundedGitRunner()
    errors: list[str] = []
    for bundle in sorted(bundles, key=lambda item: item.bundle_id):
        if bundle.preflight.repository_root != root.as_posix():
            errors.append(
                f"Git preflight repository differs from caller root: {bundle.bundle_id}"
            )
        if (
            bundle.temporal_cutoff_commit != snapshot_commit
            or bundle.manifest.temporal_cutoff_commit != snapshot_commit
        ):
            errors.append(f"Git bundle cutoff differs from snapshot: {bundle.bundle_id}")
            continue
        manifest_commits = set(bundle.manifest.commit_ids)
        if any(item.commit_id not in manifest_commits for item in bundle.slices):
            errors.append(f"Git bundle contains a slice outside its manifest: {bundle.bundle_id}")
        if bundle.manifest.commit_ids:
            replayed_slices, replayed_audit, replayed_skips = (
                _collect_change_slices(runner, root, bundle.manifest)
            )
            replayed_by_id = {
                item.item_id: item.model_dump(mode="json")
                for item in replayed_slices
            }
            saved_by_id = {
                item.item_id: item.model_dump(mode="json")
                for item in bundle.slices
            }
            if replayed_by_id != saved_by_id:
                errors.append(
                    "Git strong-slice selection/provenance differs from exact "
                    f"object replay: {bundle.bundle_id}"
                )
            if replayed_audit != bundle.selection_audit:
                errors.append(
                    f"Git slice-selection audit differs from replay: {bundle.bundle_id}"
                )
            if replayed_skips != bundle.skipped_change_reasons:
                errors.append(
                    f"Git resource/technical skips differ from replay: {bundle.bundle_id}"
                )
    return errors


def _git_preflight_plan_errors(bundle: GitTemporalBundle) -> list[str]:
    """Bind saved preflight audit rows to the fixed adapter command prefix."""

    cutoff = bundle.temporal_cutoff_commit
    canonical = [
        ("git_version", ["--version"]),
        ("repository_top_level", ["rev-parse", "--show-toplevel"]),
        ("object_format", ["rev-parse", "--show-object-format"]),
        ("head_commit", ["rev-parse", "--verify", "HEAD^{commit}"]),
        ("cutoff_commit", ["rev-parse", "--verify", f"{cutoff}^{{commit}}"]),
        ("shallow_repository", ["rev-parse", "--is-shallow-repository"]),
        (
            "partial_clone_configuration",
            [
                "config",
                "--local",
                "--get-regexp",
                r"^(extensions\.partialclone|remote\..*\.promisor)$",
            ],
        ),
        ("common_git_directory", ["rev-parse", "--git-common-dir"]),
        (
            "clean_worktree",
            ["diff", "--quiet", "--no-ext-diff", cutoff, "--"],
        ),
        (
            "clean_index",
            ["diff", "--cached", "--quiet", "--no-ext-diff", cutoff, "--"],
        ),
        ("untracked_files", ["ls-files", "--others", "--exclude-standard", "-z"]),
        (
            "reachable_objects",
            ["rev-list", "--objects", "--missing=print", cutoff],
        ),
    ]
    observed = [(item.name, item.argv) for item in bundle.preflight.checks]
    if observed != canonical[: len(observed)]:
        return [f"Git preflight command plan differs from adapter: {bundle.bundle_id}"]
    if bundle.preflight.status == "complete" and len(observed) != len(canonical):
        return [f"complete Git preflight omits adapter checks: {bundle.bundle_id}"]
    return []


def _expected_audit_files(
    bundle: UnifiedBundle,
    refs: list[ImmutableArtifactRef],
    *,
    include_upstream_inputs: bool,
) -> set[str]:
    paths = {
        "canonical/audit/index.json",
        "canonical/audit/member-dispositions.json",
        "canonical/audit/funnel.json",
        "canonical/audit/git/refs.json",
    }
    if include_upstream_inputs:
        paths.add("canonical/audit/upstream-inputs.json")
    paths.update(
        f"canonical/audit/channel-assessments/{item.assessment_id}.json"
        for item in bundle.assessments
    )
    paths.update(
        f"canonical/audit/cases/{item.case_id}.json" for item in bundle.cases
    )
    paths.update(
        f"canonical/audit/current-evidence/{item.bundle_id}.json"
        for item in bundle.current_evidence
    )
    paths.update(
        f"canonical/audit/git/relations/{item.matrix_id}.json"
        for item in bundle.temporal_matrices
    )
    paths.update(
        f"canonical/audit/findings/{item.finding_id}.json"
        for item in bundle.findings
    )
    paths.update(
        f"canonical/audit/refinements/{item.refinement_id}.json"
        for item in bundle.refinements
    )
    paths.update(item.artifact_path for item in refs)
    return paths


def _parse_large_upstream_inputs(
    tree: _TreeSnapshot,
    relative: str,
    errors: list[str],
) -> UnifiedUpstreamInputs | None:
    """Validate a large envelope after its digest was streamed at snapshot time.

    The public model's final validator recomputes its self-digest by dumping the
    complete nested model graph.  For a canonical file whose digest has already
    been independently streamed, repeat the remaining outer binding checks
    without that second full materialization.  Nested field models still run
    their normal strict validators and self-digest checks.
    """

    from repo_constraint_memory_v3.explicit_verification import (
        explicit_candidate_root_id,
    )

    data = _snapshot_bytes(tree, relative, errors)
    if data is None:
        return None
    try:
        unchecked = _UncheckedUnifiedUpstreamInputs.model_validate_json(data)
        value = UnifiedUpstreamInputs.model_construct(
            _fields_set=unchecked.model_fields_set,
            **unchecked.__dict__,
        )
        if value.current_source is not None and (
            value.current_source.snapshot_digest != value.snapshot_digest
        ):
            raise ValueError("current-source envelope snapshot mismatch")
        if value.observation_shards is not None and (
            value.observation_shards.snapshot_digest != value.snapshot_digest
        ):
            raise ValueError("observation-shard envelope snapshot mismatch")
        for decision in value.explicit_decisions:
            if (
                decision.proposition is not None
                and decision.proposition.snapshot_digest != value.snapshot_digest
            ):
                raise ValueError("explicit proposition snapshot mismatch")
        explicit_roots = {
            explicit_candidate_root_id(item.candidate_fingerprint)
            for item in value.explicit_decisions
            if item.raw_candidate is not None
        }
        verification_roots = {
            item.candidate_root_id
            for item in value.explicit_verification.decisions
        }
        if not verification_roots <= explicit_roots:
            raise ValueError(
                "explicit verification references an unknown candidate root"
            )
        if value.artifact_digest != tree.json_content_digests.get(relative):
            raise ValueError("incorrect unified upstream artifact digest")
        return value
    except (ValidationError, ValueError) as error:
        errors.append(f"{relative} failed validation: {_short_error(error)}")
        return None


def _upstream_inputs(
    tree: _TreeSnapshot,
    index: UnifiedIndex,
    *,
    required: bool,
    errors: list[str],
) -> UnifiedUpstreamInputs | None:
    relative = "canonical/audit/upstream-inputs.json"
    if relative not in tree.files:
        if required:
            errors.append(
                "required unified upstream input envelope is missing: "
                f"{relative}"
            )
        return None
    if relative not in tree.canonical_json_files:
        errors.append(f"required JSON artifact is unreadable: {relative}")
        return None
    value = (
        _parse_large_upstream_inputs(tree, relative, errors)
        if relative in tree.json_content_digests
        else _parse_model_file(
            tree,
            relative,
            UnifiedUpstreamInputs,
            errors,
        )
    )
    if value is None:
        return None
    if value.snapshot_digest != index.snapshot_digest:
        errors.append("unified upstream input snapshot mismatch")
    if value.profile_digest != index.profile_digest:
        errors.append("unified upstream input profile mismatch")
    if value.snapshot_commit != index.snapshot_commit:
        errors.append("unified upstream input snapshot commit mismatch")
    return value


def _upstream_replay_binding(
    upstream: UnifiedUpstreamInputs,
    *,
    streamed_value_digests: Mapping[str, str] | None = None,
) -> _UpstreamReplayBinding:
    """Reduce a validated envelope to collision-resistant replay commitments."""

    streamed = streamed_value_digests or {}
    return _UpstreamReplayBinding(
        explicit_decision_digests=tuple(
            sorted(
                _stream_content_digest(item)
                for item in upstream.explicit_decisions
            )
        ),
        explicit_verification_digest=streamed.get(
            "explicit_verification"
        ) or _stream_content_digest(upstream.explicit_verification),
        current_source_digest=(
            streamed.get("current_source")
            or _stream_content_digest(upstream.current_source)
            if upstream.current_source is not None
            else None
        ),
        observation_shards_digest=(
            streamed.get("observation_shards")
            or _stream_content_digest(upstream.observation_shards)
            if upstream.observation_shards is not None
            else None
        ),
        raw_candidate_events=upstream.raw_candidate_events,
        explicit_discovery_complete=upstream.explicit_discovery_complete,
    )


def _stream_content_digest(value: object) -> str:
    """Hash the ids.py canonical JSON form without constructing a full dump."""

    hasher = hashlib.sha256()

    def emit(item: object) -> None:
        # ``RootModel.model_dump(mode="json")`` serializes the root value
        # directly.  Treating it as an ordinary BaseModel would introduce a
        # synthetic ``{"root": ...}`` wrapper and make the streamed digest
        # disagree with the canonical persisted JSON.
        if isinstance(item, RootModel):
            emit(item.root)
            return
        if isinstance(item, BaseModel):
            hasher.update(b"{")
            emitted = 0
            for name in sorted(type(item).model_fields):
                field = type(item).model_fields[name]
                if field.exclude:
                    continue
                if emitted:
                    hasher.update(b",")
                hasher.update(
                    json.dumps(name, ensure_ascii=False).encode("utf-8")
                )
                hasher.update(b":")
                emit(getattr(item, name))
                emitted += 1
            hasher.update(b"}")
            return
        if isinstance(item, Enum):
            emit(item.value)
            return
        if isinstance(item, Path):
            emit(item.as_posix())
            return
        if isinstance(item, Mapping):
            hasher.update(b"{")
            emitted = 0
            for key, nested in sorted(item.items(), key=lambda pair: str(pair[0])):
                if emitted:
                    hasher.update(b",")
                hasher.update(
                    json.dumps(str(key), ensure_ascii=False).encode("utf-8")
                )
                hasher.update(b":")
                emit(nested)
                emitted += 1
            hasher.update(b"}")
            return
        if isinstance(item, (list, tuple)):
            hasher.update(b"[")
            for ordinal, nested in enumerate(item):
                if ordinal:
                    hasher.update(b",")
                emit(nested)
            hasher.update(b"]")
            return
        if isinstance(item, (set, frozenset)):
            encoded = sorted(canonical_bytes(nested) for nested in item)
            hasher.update(b"[")
            for ordinal, nested in enumerate(encoded):
                if ordinal:
                    hasher.update(b",")
                hasher.update(nested)
            hasher.update(b"]")
            return
        try:
            hasher.update(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"unsupported canonical digest value: {type(item).__name__}"
            ) from error

    emit(value)
    return hasher.hexdigest()


def _upstream_bundle_errors(
    upstream: UnifiedUpstreamInputs,
    bundle: UnifiedBundle,
    grouping_execution: object | None,
    git_values: Mapping[str, BaseModel],
) -> list[str]:
    """Bind authenticated upstream inputs to their deterministic projections."""

    from repo_constraint_memory_v3.unified.inputs import (
        build_channel_assessment_result,
    )
    from repo_constraint_memory_v3.unified.pipeline import _build_unified_funnel
    from repo_constraint_memory_v3.unified.upstream import _runtime_inputs

    errors: list[str] = []
    if upstream.raw_candidate_events != bundle.funnel.raw_candidate_events:
        errors.append("unified upstream raw candidate count differs from funnel")
    try:
        runtime = _runtime_inputs(upstream)
        assessment_build = build_channel_assessment_result(
            explicit_decisions=runtime.explicit_decisions,
            explicit_verification=runtime.explicit_verification,
            current_source=runtime.current_source,
            snapshot_digest=bundle.index.snapshot_digest,
            profile_digest=bundle.index.profile_digest,
        )
        expected_assessments = assessment_build.assessments
    except (KeyError, TypeError, ValueError) as error:
        errors.append(
            "unified upstream assessment reconstruction failed: "
            f"{_short_error(error)}"
        )
    else:
        actual_roots = tuple(
            sorted(
                (
                    item
                    for item in bundle.assessments
                    if not _is_unified_refinement_assessment(item)
                ),
                key=lambda item: item.assessment_id,
            )
        )
        if expected_assessments != actual_roots:
            errors.append(
                "channel assessments differ from unified upstream input envelope"
            )
        current_raw_events = (
            runtime.current_source.index.candidate_events
            if runtime.current_source is not None
            else 0
        )
        explicit_raw_events = upstream.raw_candidate_events - current_raw_events
        if explicit_raw_events < 0:
            errors.append(
                "unified upstream raw candidate count is below current events"
            )
        else:
            try:
                if grouping_execution is None:
                    raise ValueError("grouping execution replay is unavailable")
                replayed_funnel = _build_unified_funnel(
                    cases=bundle.cases,
                    current_evidence=bundle.current_evidence,
                    git_bundles=[item for item in git_values.values() if isinstance(item, GitTemporalBundle)],
                    assessments=bundle.assessments,
                    claims=bundle.canonical_claims,
                    dispositions=bundle.member_dispositions,
                    grouping_execution=grouping_execution,
                    raw_candidate_events_by_channel={
                        "explicit": explicit_raw_events,
                        "implicit_current": current_raw_events,
                    },
                    assessment_omissions=assessment_build.omissions,
                )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(
                    "unified upstream funnel reconstruction failed: "
                    f"{_short_error(error)}"
                )
            else:
                if replayed_funnel != bundle.funnel:
                    errors.append(
                        "unified funnel differs from upstream omission replay"
                    )

        capability_by_name = {
            item.capability: item for item in bundle.completeness.capabilities
        }
        assessment_capability = capability_by_name.get(
            "channel_assessment_construction"
        )
        expected_partial = any(
            item.affects_completeness for item in assessment_build.omissions
        )
        if (
            assessment_capability is None
            or assessment_capability.requirement_mode != "required"
            or assessment_capability.status
            != ("partial" if expected_partial else "complete")
            or assessment_capability.reason_codes
            != (
                ["channel_assessment_construction_incomplete"]
                if expected_partial
                else []
            )
        ):
            errors.append(
                "channel-assessment completeness differs from upstream replay"
            )

    expected_dependency_status = (
        "complete" if upstream.explicit_discovery_complete else "partial"
    )
    for current in bundle.current_evidence:
        for dependency in current.dependencies:
            if dependency.capability != "explicit_discovery_for_current_conflicts":
                continue
            if (
                dependency.requirement_mode != "augmentative"
                or dependency.status != expected_dependency_status
            ):
                errors.append(
                    "current evidence differs from upstream explicit-discovery "
                    f"binding: {current.case_id}"
                )
    return errors


def _reference_closure(
    bundle: UnifiedBundle,
    git_values: dict[str, BaseModel],
    tree: _TreeSnapshot,
    errors: list[str],
) -> None:
    assessments = {item.assessment_id: item for item in bundle.assessments}
    case_by_id = {item.case_id: item for item in bundle.cases}
    current_by_case = {item.case_id: item for item in bundle.current_evidence}
    matrix_by_case = {item.case_id: item for item in bundle.temporal_matrices}
    finding_by_case = {item.case_id: item for item in bundle.findings}
    if len(current_by_case) != len(bundle.current_evidence):
        errors.append("multiple current-evidence bundles exist for one case")
    if len(matrix_by_case) != len(bundle.temporal_matrices):
        errors.append("multiple temporal matrices exist for one case")
    if len(finding_by_case) != len(bundle.findings):
        errors.append("multiple unified findings exist for one case")
    if set(current_by_case) != set(case_by_id):
        errors.append("current-evidence/case closure is incomplete")
    if set(finding_by_case) != set(case_by_id):
        errors.append("finding/case closure is incomplete")
    git_bundles_by_case = {
        item.case_id: item
        for item in git_values.values()
        if isinstance(item, GitTemporalBundle)
    }

    _refinement_closure(bundle, git_values, errors)
    for case in bundle.cases:
        member_ids = {item.member_id for item in case.members}
        for member in case.members:
            assessment = assessments.get(member.assessment_id)
            if assessment is None:
                errors.append(f"case member assessment is missing: {member.member_id}")
                continue
            expected_path = (
                "canonical/audit/channel-assessments/"
                f"{assessment.assessment_id}.json"
            )
            if (
                member.assessment_ref.artifact_path != expected_path
                or member.assessment_ref.artifact_digest
                != assessment.artifact_digest
            ):
                errors.append(f"case assessment ref mismatch: {member.member_id}")
        current = current_by_case.get(case.case_id)
        if current is not None:
            if any(
                not set(item.member_ids) <= member_ids for item in current.items
            ):
                errors.append(f"current evidence references foreign member: {case.case_id}")
        matrix = matrix_by_case.get(case.case_id)
        if matrix is not None:
            if set(matrix.member_ids) != member_ids:
                errors.append(f"temporal matrix member closure failed: {case.case_id}")
            git_bundle = git_bundles_by_case.get(case.case_id)
            available_items = (
                {item.item_id for item in git_bundle.slices}
                if git_bundle is not None
                else set()
            )
            if not set(matrix.item_ids) <= available_items:
                errors.append(f"temporal matrix item closure failed: {case.case_id}")
            if matrix.status == "complete" and set(matrix.item_ids) != available_items:
                errors.append(f"complete temporal matrix omits Git items: {case.case_id}")
        finding = finding_by_case.get(case.case_id)
        if finding is not None:
            if {item.member_id for item in finding.member_findings} != member_ids:
                errors.append(f"finding member closure failed: {case.case_id}")
            known_evidence: set[str] = set()
            if current is not None:
                case_assessments = [
                    assessment
                    for member in case.members
                    if (assessment := assessments.get(member.assessment_id))
                    is not None
                ]
                known_evidence = allowed_verifier_evidence_refs(
                    case_assessments,
                    current,
                    matrix,
                )
            for item in finding.member_findings:
                if not set(item.evidence_refs) <= known_evidence:
                    errors.append(
                        f"finding evidence closure failed: {case.case_id}:{item.member_id}"
                    )

    all_member_ids = {
        member.member_id for case in bundle.cases for member in case.members
    }
    for claim in bundle.canonical_claims:
        if claim.case_id not in case_by_id or not set(
            claim.contributing_member_ids
        ) <= all_member_ids:
            errors.append(f"canonical claim member/case closure failed: {claim.claim_id}")
            continue
        current = current_by_case.get(claim.case_id)
        if current is None:
            continue
        item_by_id = {item.evidence_id: item for item in current.items}
        origin_by_id = {item.origin_id: item for item in current.origins}
        temporal_slices = {
            item.item_id: item
            for item in git_bundles_by_case.get(claim.case_id, ()).slices
        } if claim.case_id in git_bundles_by_case else {}
        for evidence in [*claim.support, *claim.counterevidence]:
            if evidence.kind == "temporal":
                temporal = temporal_slices.get(evidence.evidence_id)
                if temporal is None:
                    errors.append(
                        f"canonical temporal evidence is missing: {evidence.evidence_id}"
                    )
                elif (
                    evidence.path != (temporal.after_path or temporal.before_path)
                    or evidence.artifact_ref.kind != "git_change_slice"
                    or evidence.artifact_ref.stable_id != temporal.item_id
                    or evidence.artifact_ref.snapshot_digest
                    != temporal.snapshot_digest
                    or evidence.artifact_ref.artifact_digest
                    != temporal.artifact_digest
                    or evidence.artifact_ref.artifact_path
                    != f"canonical/audit/git/slices/{temporal.item_id}.json"
                ):
                    errors.append(
                        "canonical temporal evidence projection mismatch: "
                        f"{evidence.evidence_id}"
                    )
                continue
            item = item_by_id.get(evidence.evidence_id)
            origin = origin_by_id.get(evidence.origin_id)
            if item is None or origin is None or item.origin_id != evidence.origin_id:
                errors.append(
                    f"canonical current evidence closure failed: {evidence.evidence_id}"
                )
                continue
            if (
                evidence.path != origin.path
                or evidence.line_start != origin.line_start
                or evidence.line_end != origin.line_end
                or evidence.artifact_ref != item.artifact_ref
            ):
                errors.append(
                    f"canonical evidence projection mismatch: {evidence.evidence_id}"
                )

    inference_refs: list[tuple[ImmutableArtifactRef, str | None]] = []
    for assessment in bundle.assessments:
        for ref in [
            *assessment.support_refs,
            *assessment.counterevidence_refs,
            *assessment.coverage_refs,
            *assessment.lineage_refs,
            *assessment.extension_refs,
        ]:
            if ref.kind == "inference_request":
                inference_refs.append((ref, None))
    for matrix in bundle.temporal_matrices:
        inference_refs.extend((ref, "temporal_evidence_classify") for ref in matrix.inference_refs)
    for finding in bundle.findings:
        if finding.inference_ref is not None:
            inference_refs.append((finding.inference_ref, "unified_case_verify"))
    observed_inference: dict[str, InferenceRecord] = {}
    for ref, expected_stage in inference_refs:
        if ref.artifact_path not in tree.files:
            errors.append(f"inference record is missing: {ref.stable_id}")
            continue
        record = _parse_model_file(
            tree,
            ref.artifact_path,
            InferenceRecord,
            errors,
        )
        if record is None:
            continue
        if record.request_id != ref.stable_id:
            errors.append(f"inference record identity mismatch: {ref.stable_id}")
        if record.snapshot_digest != bundle.index.snapshot_digest:
            errors.append(f"inference record snapshot mismatch: {ref.stable_id}")
        if record.record_digest != ref.artifact_digest:
            errors.append(f"inference record digest mismatch: {ref.stable_id}")
        if expected_stage is not None and record.stage != expected_stage:
            errors.append(f"inference record stage mismatch: {ref.stable_id}")
        prior = observed_inference.get(ref.stable_id)
        if prior is not None and prior != record:
            errors.append(f"inference ref resolves inconsistently: {ref.stable_id}")
        observed_inference[ref.stable_id] = record


def _inference_ref_for_record(record: InferenceRecord) -> ImmutableArtifactRef:
    return ImmutableArtifactRef(
        kind="inference_request",
        stable_id=record.request_id,
        snapshot_digest=record.snapshot_digest,
        artifact_path=f"inference/requests/{record.request_id}.json",
        artifact_digest=record.record_digest,
    )


def _replay_round_zero_grouping(
    bundle: UnifiedBundle,
    assessments: list[ChannelAssessment],
    anchors_by_member: Mapping[str, object],
    records: list[InferenceRecord],
    settings: object,
    omission_ledger: _UnifiedOmissionLedger | None = None,
    max_concurrency: int = 1,
    max_attempts_per_request: int = 1,
) -> tuple[
    list[str],
    tuple[CandidateCase, ...],
    CandidateGroupingExecution | None,
]:
    """Rebuild the exact bounded grouping plan and its round-zero cases."""

    from repo_constraint_memory_v3.inference import InferenceInputRef
    from repo_constraint_memory_v3.unified.grouping import (
        assemble_candidate_cases,
        mark_grouping_pairs_incomplete,
        plan_candidate_grouping,
        plan_grouping_batches,
        resolve_direct_grouping,
    )
    from repo_constraint_memory_v3.unified.ids import content_digest
    from repo_constraint_memory_v3.unified.pipeline import (
        CandidateGroupingExecution,
        _build_unified_funnel,
        _case_members,
        _grouping_limits,
        _with_case_marker,
    )
    from repo_constraint_memory_v3.unified.semantic import (
        CandidateSemanticGroupResponse,
        _candidate_group_request_ordinal,
        _validate_saved_request,
        build_candidate_semantic_group_prompt,
    )

    errors: list[str] = []
    members = _case_members(assessments)
    grouping_records = [
        item for item in records if item.stage == "candidate_semantic_group"
    ]
    if not members:
        if grouping_records:
            errors.append("candidate grouping records exist without root members")
        mode = getattr(settings, "candidate_grouping_mode")
        return (
            errors,
            (),
            CandidateGroupingExecution(
                mode=mode,
                status="not_required",
                reason=(
                    "candidate_grouping_mode_singleton"
                    if mode == "singleton"
                    else "no_candidate_members"
                ),
                planned_pairs=0,
                planned_calls=0,
                terminal_calls=0,
                case_count=0,
            ),
        )
    try:
        plan = plan_candidate_grouping(
            members,
            anchors_by_member=anchors_by_member,  # type: ignore[arg-type]
            limits=_grouping_limits(settings),  # type: ignore[arg-type]
            mode=getattr(settings, "candidate_grouping_mode"),
        )
    except (KeyError, TypeError, ValueError) as error:
        return (
            [f"candidate grouping plan replay failed: {_short_error(error)}"],
            (),
            None,
        )

    records_by_batch: dict[
        tuple[str, int, tuple[str, ...]], list[InferenceRecord]
    ] = {}
    for record in grouping_records:
        input_ids = tuple(
            item.stable_id
            for item in record.input_manifest
            if item.kind == "candidate_case_member"
        )
        key = (record.mode or "", record.request_ordinal, input_ids)
        records_by_batch.setdefault(key, []).append(record)

    relations = []
    used_records: set[str] = set()
    members_by_id = {item.member_id: item for item in plan.members}

    def replay_batches(batch_plan):
        replayed_relations = []
        failed_pairs = set(batch_plan.omitted_pairs)
        concurrency = max(1, max_concurrency)

        def replay_batch(batch, *, admitted: bool) -> None:
            key = (
                f"blind_pair_relation_{batch.phase}",
                _candidate_group_request_ordinal(batch.phase, batch.pairs),
                batch.member_ids,
            )
            candidates = records_by_batch.get(key, [])
            used_records.update(item.request_id for item in candidates)
            if not candidates and omission_ledger is not None and not admitted:
                consumed, _ = omission_ledger.consume(
                    "candidate_semantic_group"
                )
                if consumed:
                    failed_pairs.update(batch.pairs)
                    return
            if len(candidates) != 1:
                errors.append(
                    "candidate grouping batch has missing or duplicate records: "
                    f"{batch.phase}:"
                    + ",".join(
                        f"{left}/{right}" for left, right in batch.pairs
                    )
                )
                failed_pairs.update(batch.pairs)
                return
            record = candidates[0]
            if (
                omission_ledger is not None
                and not omission_ledger.record_terminal(record, admitted=admitted)
            ):
                errors.append(
                    "candidate grouping record appears after global "
                    f"budget closure: {batch.phase}"
                )
            ordered_members = [
                members_by_id[item] for item in batch.member_ids
            ]
            inputs = [
                InferenceInputRef(
                    kind="candidate_case_member",
                    stable_id=item.member_id,
                    digest=content_digest(item.model_dump(mode="json")),
                    snapshot_digest=item.snapshot_digest,
                )
                for item in ordered_members
            ]
            try:
                request_payload = build_candidate_semantic_group_prompt(
                    ordered_members,
                    required_pairs=batch.pairs,
                    phase=batch.phase,
                )
                if (
                    contractual_request_bytes(
                        request_payload,
                        CandidateSemanticGroupResponse,
                    )
                    > int(getattr(settings, "max_group_request_bytes"))
                ):
                    raise ValueError(
                        "planned candidate grouping batch exceeds request-byte "
                        "bound"
                    )
                _validate_saved_request(
                    record,
                    inputs=inputs,
                    request_ordinal=_candidate_group_request_ordinal(
                        batch.phase, batch.pairs
                    ),
                    request_payload=request_payload,
                    response_model=CandidateSemanticGroupResponse,
                )
                if (
                    "max_candidate_group_request_bytes"
                    in record.incomplete_reasons
                ):
                    raise ValueError(
                        "planned candidate grouping record carries a forged "
                        "byte-gate skip"
                    )
                if record.terminal_status == "success":
                    replayed_relations.extend(
                        replay_candidate_semantic_group_record(
                            record,
                            ordered_members,
                            expected_pairs=batch.pairs,
                            phase=batch.phase,
                        )
                    )
                else:
                    failed_pairs.update(batch.pairs)
            except (ValidationError, ValueError, TypeError) as error:
                errors.append(
                    f"{record.request_id}: candidate grouping response replay "
                    f"failed: {_short_error(error)}"
                )
                failed_pairs.update(batch.pairs)

        for offset in range(0, len(batch_plan.batches), concurrency):
            wave = batch_plan.batches[offset : offset + concurrency]
            parallel_admitted = (
                omission_ledger is not None
                and omission_ledger.can_admit_parallel_wave(
                    [1] * len(wave),
                    attempts_per_request=max_attempts_per_request,
                )
            )
            for batch in wave:
                admitted = (
                    True
                    if omission_ledger is None or parallel_admitted
                    else omission_ledger.exhaustion_reason() is None
                )
                replay_batch(batch, admitted=admitted)
        return replayed_relations, failed_pairs

    direct_batch_plan = plan_grouping_batches(
        plan, plan.semantic_pairs, phase="direct"
    )
    direct_pair_count = len(plan.semantic_pairs)
    direct_relations, failed_direct_pairs = replay_batches(direct_batch_plan)
    relations.extend(direct_relations)
    try:
        plan, closure_pairs = resolve_direct_grouping(
            plan,
            direct_relations,
            failed_pairs=sorted(failed_direct_pairs),
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"candidate grouping direct replay failed: {_short_error(error)}")
        return errors, (), None
    closure_batch_plan = plan_grouping_batches(
        plan, closure_pairs, phase="closure"
    )
    closure_relations, failed_closure_pairs = replay_batches(closure_batch_plan)
    relations.extend(closure_relations)
    if failed_closure_pairs:
        try:
            plan = mark_grouping_pairs_incomplete(
                plan,
                sorted(failed_closure_pairs),
                reason_code="candidate_semantic_group_failed",
            )
        except ValueError as error:
            errors.append(
                f"candidate grouping failure replay failed: {_short_error(error)}"
            )
            return errors, (), None

    orphaned = sorted(
        item.request_id
        for item in grouping_records
        if item.request_id not in used_records
    )
    if orphaned:
        errors.append(f"orphaned candidate grouping records: {orphaned}")
    try:
        grouping = assemble_candidate_cases(
            plan,
            relations,
            profile_digest=bundle.index.profile_digest,
            required_capabilities=("current_evidence",),
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"candidate case assembly replay failed: {_short_error(error)}")
        return errors, (), None
    expected_cases = list(grouping.cases)
    max_cases = int(getattr(settings, "max_cases"))
    if (
        getattr(settings, "candidate_grouping_mode") == "semantic"
        and len(expected_cases) > max_cases
    ):
        bounded: list[CandidateCase] = []
        for ordinal, case in enumerate(expected_cases):
            if ordinal < max_cases:
                bounded.append(case)
                continue
            payload = case.model_dump(mode="python", exclude={"artifact_digest"})
            payload.update(
                {
                    "grouping_status": "incomplete",
                    "grouping_reason_codes": sorted(
                        {*case.grouping_reason_codes, "max_cases_exceeded"}
                    ),
                }
            )
            bounded.append(CandidateCase.model_validate(seal_payload(payload)))
        expected_cases = bounded
    complete_refinement_parents = {
        item.case_id for item in bundle.refinements if item.status == "complete"
    }
    expected_cases = [
        (
            _with_case_marker(item, "refinement_parent_superseded")
            if item.case_id in complete_refinement_parents
            else item
        )
        for item in expected_cases
    ]
    refinement_assessments = {
        item.assessment_id
        for item in bundle.assessments
        if _is_unified_refinement_assessment(item)
    }
    saved_round_zero = tuple(
        sorted(
            (
                case
                for case in bundle.cases
                if not any(
                    member.assessment_id in refinement_assessments
                    for member in case.members
                )
            ),
            key=lambda item: item.case_id,
        )
    )
    expected_round_zero = tuple(sorted(expected_cases, key=lambda item: item.case_id))
    if expected_round_zero != saved_round_zero:
        errors.append("round-zero candidate cases differ from grouping replay")
    # ``resolve_direct_grouping`` rewrites ``plan.semantic_pairs`` to include
    # successful direct cells plus closure cells.  Preserve the frozen direct
    # denominator from before that rewrite; adding closure to the rewritten
    # plan would double-count closure pairs in validator replay.
    planned_pairs = direct_pair_count + len(closure_pairs)
    planned_calls = len(direct_batch_plan.batches) + len(
        closure_batch_plan.batches
    )
    terminal_calls = len(used_records)
    mode = getattr(settings, "candidate_grouping_mode")
    grouping_execution = CandidateGroupingExecution(
        mode=mode,
        status=(
            "not_required"
            if mode == "singleton"
            else (
                "complete"
                if terminal_calls == planned_calls
                and all(
                    item.grouping_status == "complete"
                    for item in expected_round_zero
                )
                else "partial"
            )
        ),
        reason=(
            "candidate_grouping_mode_singleton"
            if mode == "singleton"
            else None
        ),
        planned_pairs=planned_pairs,
        planned_calls=planned_calls,
        terminal_calls=terminal_calls,
        case_count=len(expected_round_zero),
    )
    return errors, expected_round_zero, grouping_execution


def _unified_inference_replay_errors(
    bundle: UnifiedBundle,
    git_values: Mapping[str, BaseModel],
    records: list[InferenceRecord],
) -> list[str]:
    """Replay saved normalized unified responses without provider or Git I/O."""

    errors: list[str] = []
    record_by_id = {item.request_id: item for item in records}
    if len(record_by_id) != len(records):
        errors.append("unified inference replay records contain duplicate IDs")
        return errors

    members_by_id = {}
    for case in bundle.cases:
        for member in case.members:
            prior = members_by_id.get(member.member_id)
            if prior is not None and prior != member:
                errors.append(
                    f"case member resolves inconsistently: {member.member_id}"
                )
            members_by_id[member.member_id] = member

    saved_semantic_relations = {}
    case_ids_by_member: dict[str, set[str]] = {}
    for case in bundle.cases:
        for member in case.members:
            case_ids_by_member.setdefault(member.member_id, set()).add(case.case_id)
        for relation in case.relations:
            if relation.source != "semantic_grouping":
                continue
            pair = (relation.left_member_id, relation.right_member_id)
            prior = saved_semantic_relations.get(pair)
            if prior is not None and prior != relation:
                errors.append(
                    "semantic grouping relation resolves inconsistently: "
                    f"{pair[0]} / {pair[1]}"
                )
            saved_semantic_relations[pair] = relation

    for record in sorted(records, key=lambda item: item.request_id):
        if (
            record.stage != "candidate_semantic_group"
            or record.terminal_status != "success"
        ):
            continue
        input_ids = [
            item.stable_id
            for item in record.input_manifest
            if item.kind == "candidate_case_member"
        ]
        members = [members_by_id[item] for item in input_ids if item in members_by_id]
        if len(members) != len(input_ids):
            errors.append(f"{record.request_id}: grouping input member is missing")
            continue
        try:
            replayed = replay_candidate_semantic_group_record(
                record,
                members,
            )
        except (ValidationError, ValueError, TypeError) as error:
            errors.append(
                f"{record.request_id}: candidate grouping response replay failed: "
                f"{_short_error(error)}"
            )
            continue
        for relation in replayed:
            pair = (relation.left_member_id, relation.right_member_id)
            shared_cases = case_ids_by_member.get(pair[0], set()) & (
                case_ids_by_member.get(pair[1], set())
            )
            if relation.relation == "distinct":
                # A distinct pair can remain in one case through another
                # related/uncertain edge; case membership is a connected
                # component, not an equivalence partition.
                if shared_cases and saved_semantic_relations.get(pair) != relation:
                    errors.append(
                        "saved candidate grouping relation differs from response: "
                        f"{pair[0]} / {pair[1]}"
                    )
                continue
            if not shared_cases or saved_semantic_relations.get(pair) != relation:
                errors.append(
                    "saved candidate grouping relation differs from response: "
                    f"{pair[0]} / {pair[1]}"
                )

    cases_by_id = {item.case_id: item for item in bundle.cases}
    slices_by_id = {
        item.item_id: item
        for item in git_values.values()
        if isinstance(item, GitChangeSlice)
    }
    git_by_case = {
        item.case_id: item
        for item in git_values.values()
        if isinstance(item, GitTemporalBundle)
    }
    matrices_by_id = {item.matrix_id: item for item in bundle.temporal_matrices}
    for matrix in bundle.temporal_matrices:
        if matrix.status != "complete" or not matrix.inference_refs:
            continue
        case = cases_by_id.get(matrix.case_id)
        temporal_bundle = git_by_case.get(matrix.case_id)
        if case is None or temporal_bundle is None:
            errors.append(f"temporal replay case is missing: {matrix.case_id}")
            continue
        manifest_ref = git_manifest_ref(temporal_bundle)
        if matrix.git_manifest_ref != manifest_ref:
            errors.append(f"temporal matrix Git manifest differs: {matrix.matrix_id}")
            continue
        replayed_relations = []
        replayed_refs = []
        failed = False
        for ref in matrix.inference_refs:
            record = record_by_id.get(ref.stable_id)
            if record is None:
                errors.append(f"temporal replay record is missing: {ref.stable_id}")
                failed = True
                continue
            item_ids = [
                item.stable_id
                for item in record.input_manifest
                if item.kind == "temporal_evidence_item"
            ]
            slices = [slices_by_id[item] for item in item_ids if item in slices_by_id]
            if len(slices) != len(item_ids):
                errors.append(f"{record.request_id}: temporal replay slice is missing")
                failed = True
                continue
            try:
                replayed_relations.extend(
                    replay_temporal_evidence_record(
                        record,
                        case,
                        slices,
                        manifest_ref,
                    )
                )
            except (ValidationError, ValueError, TypeError) as error:
                errors.append(
                    f"{record.request_id}: temporal response replay failed: "
                    f"{_short_error(error)}"
                )
                failed = True
                continue
            replayed_refs.append(_inference_ref_for_record(record))
        if failed:
            continue
        replayed_relations.sort(key=lambda item: (item.item_id, item.member_id))
        payload: dict[str, object] = {
            "matrix_id": matrix.matrix_id,
            "case_id": case.case_id,
            "snapshot_digest": case.snapshot_digest,
            "profile_digest": case.profile_digest,
            "git_manifest_ref": manifest_ref.model_dump(mode="json"),
            "item_ids": sorted({item.item_id for item in replayed_relations}),
            "member_ids": sorted(item.member_id for item in case.members),
            "relations": [
                item.model_dump(mode="json") for item in replayed_relations
            ],
            "status": "complete",
            "incomplete_reasons": [],
            "inference_refs": [
                item.model_dump(mode="json") for item in replayed_refs
            ],
        }
        payload["artifact_digest"] = artifact_digest(payload)
        try:
            replayed_matrix = TemporalRelationMatrix.model_validate(payload)
        except (ValidationError, ValueError) as error:
            errors.append(
                f"{matrix.matrix_id}: temporal batch merge replay failed: "
                f"{_short_error(error)}"
            )
            continue
        if replayed_matrix != matrix:
            errors.append(
                f"saved temporal relation matrix differs from response: {matrix.matrix_id}"
            )

    assessments_by_id = {item.assessment_id: item for item in bundle.assessments}
    current_by_id = {item.bundle_id: item for item in bundle.current_evidence}
    for finding in bundle.findings:
        if finding.status != "complete" or finding.inference_ref is None:
            continue
        record = record_by_id.get(finding.inference_ref.stable_id)
        case = cases_by_id.get(finding.case_id)
        if record is None or case is None:
            errors.append(f"finding replay input is missing: {finding.finding_id}")
            continue
        current_ids = [
            item.stable_id
            for item in record.input_manifest
            if item.kind == "current_evidence_bundle"
        ]
        temporal_ids = [
            item.stable_id
            for item in record.input_manifest
            if item.kind == "temporal_evidence_bundle"
        ]
        if len(current_ids) != 1 or len(temporal_ids) > 1:
            errors.append(f"finding replay manifest is invalid: {finding.finding_id}")
            continue
        current = current_by_id.get(current_ids[0])
        temporal = matrices_by_id.get(temporal_ids[0]) if temporal_ids else None
        if current is None or (temporal_ids and temporal is None):
            errors.append(f"finding replay evidence is missing: {finding.finding_id}")
            continue
        assessments = [
            assessments_by_id[item.assessment_id]
            for item in case.members
            if item.assessment_id in assessments_by_id
        ]
        if len(assessments) != len(case.members):
            errors.append(f"finding replay assessment is missing: {finding.finding_id}")
            continue
        try:
            replayed_finding = replay_unified_case_verify_record(
                record,
                case,
                assessments,
                current,
                temporal,
                verification_round=finding.verification_round,
            )
        except (ValidationError, ValueError, TypeError) as error:
            errors.append(
                f"{record.request_id}: unified finding response replay failed: "
                f"{_short_error(error)}"
            )
            continue
        if replayed_finding != finding:
            errors.append(
                f"saved unified finding differs from response: {finding.finding_id}"
            )
    return errors


@dataclass(frozen=True)
class _UnifiedBudgetClosureReplay:
    temporal_omission_reasons: dict[
        tuple[str, frozenset[str], frozenset[str]], str
    ]
    verifier_omission_case_reasons: dict[str, str]
    git_skip_case_reasons: dict[str, str]
    errors: tuple[str, ...]


def _replay_unified_budget_timeline(
    bundle: UnifiedBundle,
    git_values: Mapping[str, BaseModel],
    records: Sequence[InferenceRecord],
    *,
    settings: object,
    round_zero_routes: Mapping[str, str],
    git_enabled: bool,
    snapshot_commit_available: bool,
    max_concurrency: int,
    max_attempts_per_request: int,
    ledger: _UnifiedOmissionLedger,
) -> _UnifiedBudgetClosureReplay:
    """Replay operation admission order before payload-level validation.

    The global budget is checked between grouping, temporal, and initial
    verifier concurrency waves.  Records within an already admitted wave
    (including a temporal bisection subtree) remain legal even when an earlier
    response in that wave reaches a token limit.  Refinement children retain a
    dependent second-phase barrier and execute serially.
    """

    errors: list[str] = []
    temporal_omissions: dict[
        tuple[str, frozenset[str], frozenset[str]], str
    ] = {}
    verifier_omissions: dict[str, str] = {}
    git_skips: dict[str, str] = {}
    cases_by_id = {item.case_id: item for item in bundle.cases}
    current_by_case = {item.case_id: item for item in bundle.current_evidence}
    findings_by_case = {item.case_id: item for item in bundle.findings}
    git_by_case = {
        item.case_id: item
        for item in git_values.values()
        if isinstance(item, GitTemporalBundle)
    }
    assessments_by_id = {
        item.assessment_id: item for item in bundle.assessments
    }
    temporal_by_inputs: dict[
        tuple[str, frozenset[str], frozenset[str]], list[InferenceRecord]
    ] = {}
    for record in records:
        if record.stage != "temporal_evidence_classify":
            continue
        members = frozenset(
            item.stable_id
            for item in record.input_manifest
            if item.kind == "candidate_case_member"
        )
        slices = frozenset(
            item.stable_id
            for item in record.input_manifest
            if item.kind == "temporal_evidence_item"
        )
        manifests = [
            item.stable_id
            for item in record.input_manifest
            if item.kind == "git_query_manifest"
        ]
        manifest_id = manifests[0] if len(manifests) == 1 else ""
        temporal_by_inputs.setdefault((manifest_id, members, slices), []).append(
            record
        )

    verifier_by_case: dict[str, list[InferenceRecord]] = {}
    member_case = {
        frozenset(item.member_id for item in case.members): case.case_id
        for case in bundle.cases
    }
    for record in records:
        if record.stage != "unified_case_verify":
            continue
        members = frozenset(
            item.stable_id
            for item in record.input_manifest
            if item.kind == "candidate_case_member"
        )
        case_id = member_case.get(members)
        if case_id is not None:
            verifier_by_case.setdefault(case_id, []).append(record)

    def expected_git(case_id: str, *, round_one: bool) -> bool:
        if not git_enabled or not snapshot_commit_available:
            return False
        if not round_one:
            route = round_zero_routes.get(case_id)
            return route is not None and not route.startswith("not_required")
        case = cases_by_id.get(case_id)
        current = current_by_case.get(case_id)
        if case is None or current is None:
            return False
        from repo_constraint_memory_v3.unified.pipeline import _route_case

        try:
            route = _route_case(case, assessments_by_id, current)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(
                "refinement Git route replay failed for "
                f"{case_id}: {_short_error(error)}"
            )
            return False
        return route is not None and not route.startswith("not_required")

    TemporalReplayBatch = tuple[
        CandidateCase,
        str,
        frozenset[str],
        list[GitChangeSlice],
    ]

    def freeze_git_and_temporal(
        case: CandidateCase,
        *,
        round_one: bool,
    ) -> list[TemporalReplayBatch]:
        git_bundle = git_by_case.get(case.case_id)
        needs_git = expected_git(case.case_id, round_one=round_one)
        boundary_reason = ledger.exhaustion_reason()
        if needs_git and git_bundle is None:
            if boundary_reason is not None:
                git_skips[case.case_id] = boundary_reason
            return []
        if git_bundle is None:
            return []
        if needs_git and boundary_reason is not None:
            errors.append(
                "Git bundle appears after global inference budget closure: "
                f"{case.case_id}"
            )
        ordered = sorted(git_bundle.slices, key=lambda item: item.item_id)
        maximum = int(getattr(settings, "max_temporal_items_per_case"))
        if not ordered or len(ordered) > maximum:
            return []
        member_count = len(case.members)
        max_cells = int(getattr(settings, "max_temporal_cells_per_request"))
        if member_count < 1 or max_cells < member_count:
            return []
        batch_size = min(
            int(getattr(settings, "max_temporal_batch_items")),
            max_cells // member_count,
        )
        batches = [
            ordered[start : start + batch_size]
            for start in range(0, len(ordered), batch_size)
        ]
        manifest_id = git_manifest_ref(git_bundle).stable_id
        member_key = frozenset(item.member_id for item in case.members)
        return [
            (case, manifest_id, member_key, batch)
            for batch in batches
        ]

    def consume_tree(
        case: CandidateCase,
        manifest_id: str,
        member_key: frozenset[str],
        batch: list[GitChangeSlice],
        *,
        admitted: bool,
        permit_admitted: bool,
        recursive_child: bool = False,
    ) -> None:
        key = (
            manifest_id,
            member_key,
            frozenset(item.item_id for item in batch),
        )
        candidates = temporal_by_inputs.get(key, [])
        if not candidates:
            if not admitted:
                consumed, reason = ledger.consume(
                    "temporal_evidence_classify"
                )
                if consumed and reason is not None:
                    temporal_omissions[key] = reason
            return
        if len(candidates) != 1:
            return
        record = candidates[0]
        if not admitted and recursive_child:
            boundary_reason = ledger.exhaustion_reason()
            if (
                boundary_reason is None
                or record.terminal_status != "failure"
                or record.usage.calls != 0
                or record.usage.input_tokens != 0
                or record.usage.output_tokens != 0
                or record.attempts
                or boundary_reason not in record.incomplete_reasons
            ):
                errors.append(
                    "temporal recursive record crosses serial budget "
                    f"closure: {case.case_id}:{sorted(key[2])}"
                )
                return
            # Runtime still materializes one canonical zero-attempt child
            # failure after its parent crosses a budget boundary. It consumes
            # no usage and proves why the bisection stops.
            ledger.record_terminal(record, admitted=True)
        elif not ledger.record_terminal(record, admitted=admitted):
            errors.append(
                "temporal record appears after global budget closure: "
                f"{case.case_id}:{sorted(key[2])}"
            )
            return
        nonrecoverable = any(
            reason in _BUDGET_EXHAUSTION_REASONS
            or reason.startswith("model_profile_violation:")
            for reason in record.incomplete_reasons
        )
        if (
            record.terminal_status != "success"
            and len(batch) > 1
            and not nonrecoverable
        ):
            midpoint = len(batch) // 2
            for child in (batch[:midpoint], batch[midpoint:]):
                child_admitted = (
                    True
                    if permit_admitted
                    else ledger.exhaustion_reason() is None
                )
                consume_tree(
                    case,
                    manifest_id,
                    member_key,
                    child,
                    admitted=child_admitted,
                    permit_admitted=permit_admitted,
                    recursive_child=True,
                )

    def consume_temporal_batches(
        batches: Sequence[TemporalReplayBatch],
    ) -> None:
        concurrency = max(1, max_concurrency)
        for offset in range(0, len(batches), concurrency):
            wave = batches[offset : offset + concurrency]
            parallel_admitted = ledger.can_admit_parallel_wave(
                [2 * len(batch[3]) - 1 for batch in wave],
                attempts_per_request=max_attempts_per_request,
            )
            for case, manifest_id, member_key, batch in wave:
                admitted = (
                    parallel_admitted
                    or ledger.exhaustion_reason() is None
                )
                consume_tree(
                    case,
                    manifest_id,
                    member_key,
                    batch,
                    admitted=admitted,
                    permit_admitted=parallel_admitted,
                )

    def consume_git_and_temporal(case: CandidateCase, *, round_one: bool) -> None:
        consume_temporal_batches(
            freeze_git_and_temporal(case, round_one=round_one)
        )

    refinement_children: dict[str, list[str]] = {}
    finding_case_by_id = {item.finding_id: item.case_id for item in bundle.findings}
    for refinement in sorted(
        bundle.refinements,
        key=lambda item: (item.case_id, item.parent_member_id),
    ):
        for finding_id in refinement.finding_ids:
            child_case_id = finding_case_by_id.get(finding_id)
            if child_case_id is not None:
                refinement_children.setdefault(refinement.case_id, []).append(
                    child_case_id
                )

    refinement_assessment_ids = {
        item.assessment_id
        for item in bundle.assessments
        if _is_unified_refinement_assessment(item)
    }

    def verifier_expected(case: CandidateCase, *, round_one: bool) -> bool:
        current = current_by_case.get(case.case_id)
        finding = findings_by_case.get(case.case_id)
        if current is None or finding is None:
            return False
        if round_one:
            reroute = next(
                (
                    item
                    for item in current.dependencies
                    if item.capability == "refinement_git_reroute"
                ),
                None,
            )
            return (
                current.integrity == "valid"
                and reroute is not None
                and reroute.status == "complete"
            )
        route = round_zero_routes.get(case.case_id)
        return (
            route is not None
            and route
            not in {
                "not_required_fast_path",
                "not_required_hard_invalid",
                "not_required_current_incomplete",
            }
            and case.grouping_status != "incomplete"
            and current.integrity != "invalid"
        )

    def consume_verifier(
        case: CandidateCase,
        *,
        round_one: bool,
        admitted_wave: bool | None = None,
    ) -> None:
        if not verifier_expected(case, round_one=round_one):
            return
        candidates = verifier_by_case.get(case.case_id, [])
        admitted = (
            ledger.exhaustion_reason() is None
            if admitted_wave is None
            else admitted_wave
        )
        if not candidates:
            if not admitted:
                consumed, reason = ledger.consume("unified_case_verify")
                if consumed and reason is not None:
                    verifier_omissions[case.case_id] = reason
            return
        if len(candidates) == 1 and not ledger.record_terminal(
            candidates[0], admitted=admitted
        ):
            errors.append(
                "verifier record appears after global budget closure: "
                f"{case.case_id}"
            )

    base_cases = [
        cases_by_id[case_id]
        for case_id in sorted(round_zero_routes)
        if case_id in cases_by_id
    ]
    round_zero_temporal_batches = [
        batch
        for case in base_cases
        for batch in freeze_git_and_temporal(case, round_one=False)
    ]
    consume_temporal_batches(round_zero_temporal_batches)
    initial_verifier_cases = [
        case
        for case in base_cases
        if verifier_expected(case, round_one=False)
    ]
    concurrency = max(1, max_concurrency)
    for offset in range(0, len(initial_verifier_cases), concurrency):
        wave = initial_verifier_cases[offset : offset + concurrency]
        parallel_admitted = ledger.can_admit_parallel_wave(
            [1] * len(wave),
            attempts_per_request=max_attempts_per_request,
        )
        for case in wave:
            admitted_wave = (
                parallel_admitted
                or ledger.exhaustion_reason() is None
            )
            consume_verifier(
                case,
                round_one=False,
                admitted_wave=admitted_wave,
            )
    # Refinement remains a dependent second phase: every initial finding is
    # terminal before any child reacquires current/Git evidence or verifies.
    for case in base_cases:
        for child_case_id in refinement_children.get(case.case_id, []):
            child = cases_by_id.get(child_case_id)
            if child is None or not any(
                member.assessment_id in refinement_assessment_ids
                for member in child.members
            ):
                continue
            consume_git_and_temporal(child, round_one=True)
            consume_verifier(child, round_one=True)

    return _UnifiedBudgetClosureReplay(
        temporal_omission_reasons=temporal_omissions,
        verifier_omission_case_reasons=verifier_omissions,
        git_skip_case_reasons=git_skips,
        errors=tuple(errors),
    )


def _refinement_budget_dependency_errors(
    bundle: UnifiedBundle,
    git_values: Mapping[str, BaseModel],
    git_skip_case_reasons: Mapping[str, str],
) -> list[str]:
    """Bind compact round-one Git skips to exact child dependencies.

    The operation timeline proves *when* a child Git acquisition may be
    omitted.  This second projection proves that the child artifact reports
    that exact host decision, rather than allowing a coordinated reseal to
    replace the typed reason while retaining the same omitted verifier.
    """

    from repo_constraint_memory_v3.unified.pipeline import (
        _dependency_for_git,
        _dependency_for_temporal,
        _route_case,
    )

    errors: list[str] = []
    assessments_by_id = {
        item.assessment_id: item for item in bundle.assessments
    }
    refinement_assessment_ids = {
        item.assessment_id
        for item in bundle.assessments
        if _is_unified_refinement_assessment(item)
    }
    current_by_case = {
        item.case_id: item for item in bundle.current_evidence
    }
    git_by_case = {
        item.case_id: item
        for item in git_values.values()
        if isinstance(item, GitTemporalBundle)
    }
    matrix_by_case = {
        item.case_id: item for item in bundle.temporal_matrices
    }
    budget_markers = {
        "git_not_run_after_inference_budget_exhausted",
        "temporal_not_run_after_inference_budget_exhausted",
    }

    for case in sorted(bundle.cases, key=lambda item: item.case_id):
        if not any(
            member.assessment_id in refinement_assessment_ids
            for member in case.members
        ):
            continue
        current = current_by_case.get(case.case_id)
        if current is None:
            continue
        direct_dependencies = {
            item.capability: item
            for item in current.dependencies
            if item.capability
            in {
                "git_temporal_evidence",
                "temporal_evidence_classification",
            }
        }
        skip_reason = git_skip_case_reasons.get(case.case_id)
        if skip_reason is None:
            if any(
                budget_markers & set(item.reason_codes)
                for item in direct_dependencies.values()
            ):
                errors.append(
                    "refinement budget-skip dependency appears without a "
                    f"replayed budget boundary: {case.case_id}"
                )
            continue

        try:
            route = _route_case(case, assessments_by_id, current)
            expected_git = _dependency_for_git(
                route,
                None,
                host_skip_reason=skip_reason,
            )
            expected_temporal = _dependency_for_temporal(
                route,
                None,
                None,
                host_skip_reason=skip_reason,
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(
                "refinement budget dependency replay failed for "
                f"{case.case_id}: {_short_error(error)}"
            )
            continue
        if route.startswith("not_required"):
            errors.append(
                "refinement Git omission exists for a no-Git route: "
                f"{case.case_id}"
            )
        if git_by_case.get(case.case_id) is not None:
            errors.append(
                "refinement Git bundle exists after compact budget skip: "
                f"{case.case_id}"
            )
        if matrix_by_case.get(case.case_id) is not None:
            errors.append(
                "refinement temporal matrix exists after compact Git skip: "
                f"{case.case_id}"
            )
        for record in bundle.refinements:
            if (
                current.bundle_id in record.reacquired_current_bundle_ids
                and record.git_bundle_refs
            ):
                errors.append(
                    "refinement Git refs exist after compact budget skip: "
                    f"{record.refinement_id}"
                )
        if direct_dependencies.get("git_temporal_evidence") != expected_git:
            errors.append(
                "refinement Git budget dependency differs from replay: "
                f"{case.case_id}"
            )
        if (
            direct_dependencies.get("temporal_evidence_classification")
            != expected_temporal
        ):
            errors.append(
                "refinement temporal budget dependency differs from replay: "
                f"{case.case_id}"
            )
    return errors


def _exhaustive_unified_inference_replay_errors(
    bundle: UnifiedBundle,
    git_values: Mapping[str, BaseModel],
    records: list[InferenceRecord],
    *,
    settings: object,
    round_zero_routes: Mapping[str, str],
    omission_ledger: _UnifiedOmissionLedger | None = None,
    git_enabled: bool = True,
    snapshot_commit_available: bool = True,
    max_concurrency: int = 1,
    max_attempts_per_request: int = 1,
    budget_git_skip_reasons: dict[str, str] | None = None,
) -> list[str]:
    """Bind every temporal/verifier record to one planned case operation."""

    from repo_constraint_memory_v3.inference import InferenceInputRef
    from repo_constraint_memory_v3.unified.ids import (
        content_digest,
        make_temporal_matrix_id,
    )
    from repo_constraint_memory_v3.unified.semantic import (
        TemporalEvidenceClassificationResponse,
        _stable_request_ordinal,
        _validate_saved_request,
        _verifier_provider_response_model,
        build_temporal_evidence_prompt,
        build_unified_case_verify_prompt,
    )
    from repo_constraint_memory_v3.unified.pipeline import _host_finding
    from repo_constraint_memory_v3.unified.refinement import _failed_finding

    errors: list[str] = []
    closure_replay = (
        _replay_unified_budget_timeline(
            bundle,
            git_values,
            records,
            settings=settings,
            round_zero_routes=round_zero_routes,
            git_enabled=git_enabled,
            snapshot_commit_available=snapshot_commit_available,
            max_concurrency=max_concurrency,
            max_attempts_per_request=max_attempts_per_request,
            ledger=omission_ledger,
        )
        if omission_ledger is not None
        else _UnifiedBudgetClosureReplay(
            temporal_omission_reasons={},
            verifier_omission_case_reasons={},
            git_skip_case_reasons={},
            errors=(),
        )
    )
    errors.extend(closure_replay.errors)
    if budget_git_skip_reasons is not None:
        budget_git_skip_reasons.update(closure_replay.git_skip_case_reasons)
    cases_by_id = {item.case_id: item for item in bundle.cases}
    assessments_by_id = {item.assessment_id: item for item in bundle.assessments}
    current_by_case = {item.case_id: item for item in bundle.current_evidence}
    matrices_by_case = {item.case_id: item for item in bundle.temporal_matrices}
    findings_by_case = {item.case_id: item for item in bundle.findings}
    git_by_case = {
        item.case_id: item
        for item in git_values.values()
        if isinstance(item, GitTemporalBundle)
    }

    temporal_records = [
        item for item in records if item.stage == "temporal_evidence_classify"
    ]
    temporal_by_inputs: dict[
        tuple[str, frozenset[str], frozenset[str]], list[InferenceRecord]
    ] = {}
    for record in temporal_records:
        members = frozenset(
            item.stable_id
            for item in record.input_manifest
            if item.kind == "candidate_case_member"
        )
        slices = frozenset(
            item.stable_id
            for item in record.input_manifest
            if item.kind == "temporal_evidence_item"
        )
        manifests = [
            item.stable_id
            for item in record.input_manifest
            if item.kind == "git_query_manifest"
        ]
        manifest_id = manifests[0] if len(manifests) == 1 else ""
        temporal_by_inputs.setdefault((manifest_id, members, slices), []).append(record)
    used_temporal: set[str] = set()
    for case in sorted(bundle.cases, key=lambda item: item.case_id):
        temporal_bundle = git_by_case.get(case.case_id)
        ordered = (
            []
            if temporal_bundle is None
            else sorted(temporal_bundle.slices, key=lambda item: item.item_id)
        )
        saved_matrix = matrices_by_case.get(case.case_id)
        if not ordered:
            if saved_matrix is not None:
                errors.append(
                    f"temporal matrix exists without planned slices: {case.case_id}"
                )
            continue
        assert temporal_bundle is not None
        manifest_ref = git_manifest_ref(temporal_bundle)
        if saved_matrix is not None and saved_matrix.git_manifest_ref != manifest_ref:
            errors.append(f"temporal matrix Git manifest differs: {case.case_id}")
        maximum = int(getattr(settings, "max_temporal_items_per_case"))
        if len(ordered) > maximum:
            if saved_matrix is not None:
                errors.append(
                    f"temporal matrix exists beyond total item bound: {case.case_id}"
                )
            continue
        member_ids = [item.member_id for item in case.members]
        max_cells = int(getattr(settings, "max_temporal_cells_per_request"))
        if max_cells < len(member_ids):
            errors.append(
                "temporal cell bound cannot accommodate one item: "
                f"{case.case_id}"
            )
            continue
        batch_size = min(
            int(getattr(settings, "max_temporal_batch_items")),
            max_cells // len(member_ids),
        )
        batches = [
            ordered[start : start + batch_size]
            for start in range(0, len(ordered), batch_size)
        ]
        successful_relations = []
        successful_records: list[InferenceRecord] = []
        failed_records: list[InferenceRecord] = []
        omitted_batch_count = 0
        omitted_reasons: set[str] = set()
        replay_failed = False
        member_key = frozenset(item.member_id for item in case.members)

        def replay_batch_tree(
            batch: list[GitChangeSlice],
            *,
            allow_compact_omission: bool = False,
        ) -> bool:
            """Consume the exact live parent/child recovery tree for a batch."""

            nonlocal omitted_batch_count

            key = (
                manifest_ref.stable_id,
                member_key,
                frozenset(item.item_id for item in batch),
            )
            candidates = temporal_by_inputs.get(key, [])
            if not candidates and allow_compact_omission:
                reason = closure_replay.temporal_omission_reasons.get(key)
                if reason is not None:
                    omitted_batch_count += 1
                    omitted_reasons.add(reason)
                    return True
            if len(candidates) != 1:
                errors.append(
                    "temporal batch has missing or duplicate records: "
                    f"{case.case_id}:{sorted(key[2])}"
                )
                return False
            record = candidates[0]
            used_temporal.add(record.request_id)
            inputs = [
                InferenceInputRef(
                    kind="git_query_manifest",
                    stable_id=manifest_ref.stable_id,
                    digest=manifest_ref.artifact_digest,
                    snapshot_digest=case.snapshot_digest,
                ),
                *(
                    InferenceInputRef(
                        kind="candidate_case_member",
                        stable_id=item.member_id,
                        digest=content_digest(item.model_dump(mode="json")),
                        snapshot_digest=case.snapshot_digest,
                    )
                    for item in case.members
                ),
                *(
                    InferenceInputRef(
                        kind="temporal_evidence_item",
                        stable_id=item.item_id,
                        digest=item.artifact_digest,
                        snapshot_digest=case.snapshot_digest,
                    )
                    for item in batch
                ),
            ]
            item_ids = [item.item_id for item in batch]
            try:
                if record.mode != "complete_item_member_matrix":
                    raise ValueError("temporal record mode differs")
                request_payload = build_temporal_evidence_prompt(case, batch)
                request_over_limit = (
                    contractual_request_bytes(
                        request_payload,
                        TemporalEvidenceClassificationResponse,
                    )
                    > int(getattr(settings, "max_temporal_request_bytes"))
                )
                _validate_saved_request(
                    record,
                    inputs=inputs,
                    request_ordinal=_stable_request_ordinal(
                        "temporal_evidence_classify",
                        [case.case_id, *item_ids, *member_ids],
                    ),
                    request_payload=request_payload,
                    response_model=TemporalEvidenceClassificationResponse,
                )
                request_gate_reason = (
                    "max_temporal_classifier_request_bytes"
                )
                if request_over_limit:
                    zero_usage = (
                        record.usage.calls,
                        record.usage.input_tokens,
                        record.usage.output_tokens,
                        record.usage.reasoning_tokens,
                        record.usage.latency_ms,
                    ) == (0, 0, 0, 0, 0)
                    if (
                        record.terminal_status != "failure"
                        or record.structured_response is not None
                        or record.attempts
                        or not zero_usage
                        or record.incomplete_reasons != [request_gate_reason]
                    ):
                        raise ValueError(
                            "over-limit temporal request is not the exact "
                            "pre-provider typed skip"
                        )
                elif request_gate_reason in record.incomplete_reasons:
                    raise ValueError(
                        "within-limit temporal request carries a forged "
                        "byte-gate skip"
                    )
                if record.terminal_status == "success":
                    successful_relations.extend(
                        replay_temporal_evidence_record(
                            record,
                            case,
                            batch,
                            manifest_ref,
                        )
                    )
                    successful_records.append(record)
                    return True
            except (ValidationError, ValueError, TypeError) as error:
                errors.append(
                    f"{record.request_id}: temporal response replay failed: "
                    f"{_short_error(error)}"
                )
                return False

            nonrecoverable = any(
                reason
                in {
                    "max_total_calls",
                    "max_total_input_tokens",
                    "max_total_output_tokens",
                }
                or reason.startswith("model_profile_violation:")
                for reason in record.incomplete_reasons
            )
            if len(batch) > 1 and not nonrecoverable:
                midpoint = len(batch) // 2
                # Do not short-circuit: live planned both children, so a
                # missing/tampered left child must not hide an orphaned or
                # independently invalid right child.
                left_valid = replay_batch_tree(batch[:midpoint])
                right_valid = replay_batch_tree(batch[midpoint:])
                return left_valid and right_valid
            failed_records.append(record)
            return True

        for batch in batches:
            if not replay_batch_tree(batch, allow_compact_omission=True):
                replay_failed = True
        if replay_failed:
            continue
        relations = sorted(
            successful_relations, key=lambda item: (item.item_id, item.member_id)
        )
        if not failed_records and not omitted_batch_count:
            status = "complete"
            reasons: list[str] = []
        else:
            reason_set = {
                "temporal_batch_incomplete",
                *(
                    reason
                    for failed_record in failed_records
                    for reason in (
                        failed_record.incomplete_reasons
                        or [
                            "temporal_classifier_"
                            f"{failed_record.terminal_status}"
                        ]
                    )
                ),
            }
            if omitted_batch_count:
                reason_set.update(
                    {
                        "temporal_batches_unprocessed",
                        *omitted_reasons,
                    }
                )
            status = "partial"
            reasons = sorted(reason_set)
        all_item_ids = sorted(item.item_id for item in ordered)
        all_member_ids = sorted(item.member_id for item in case.members)
        payload: dict[str, object] = {
            "matrix_id": make_temporal_matrix_id(
                case.snapshot_digest,
                case.profile_digest,
                case.case_id,
                all_item_ids,
                all_member_ids,
            ),
            "case_id": case.case_id,
            "snapshot_digest": case.snapshot_digest,
            "profile_digest": case.profile_digest,
            "git_manifest_ref": manifest_ref.model_dump(mode="json"),
            "item_ids": all_item_ids,
            "member_ids": all_member_ids,
            "relations": [item.model_dump(mode="json") for item in relations],
            "status": status,
            "incomplete_reasons": reasons,
            "inference_refs": [
                _inference_ref_for_record(item).model_dump(mode="json")
                for item in sorted(
                    successful_records,
                    key=lambda value: value.request_id,
                )
            ],
        }
        payload["artifact_digest"] = artifact_digest(payload)
        try:
            expected_matrix = TemporalRelationMatrix.model_validate(payload)
        except (ValidationError, ValueError) as error:
            errors.append(
                f"temporal matrix replay failed for {case.case_id}: "
                f"{_short_error(error)}"
            )
            continue
        if saved_matrix != expected_matrix:
            errors.append(
                f"saved temporal relation matrix differs from response: {case.case_id}"
            )
    orphaned_temporal = sorted(
        item.request_id
        for item in temporal_records
        if item.request_id not in used_temporal
    )
    if orphaned_temporal:
        errors.append(f"orphaned temporal classifier records: {orphaned_temporal}")

    refinement_assessment_ids = {
        item.assessment_id
        for item in bundle.assessments
        if _is_unified_refinement_assessment(item)
    }
    verifier_records = [
        item for item in records if item.stage == "unified_case_verify"
    ]
    verifier_by_inputs: dict[
        tuple[frozenset[str], str], list[InferenceRecord]
    ] = {}
    for record in verifier_records:
        members = frozenset(
            item.stable_id
            for item in record.input_manifest
            if item.kind == "candidate_case_member"
        )
        current_ids = [
            item.stable_id
            for item in record.input_manifest
            if item.kind == "current_evidence_bundle"
        ]
        if len(current_ids) != 1:
            errors.append(f"verifier record current input is invalid: {record.request_id}")
            continue
        verifier_by_inputs.setdefault((members, current_ids[0]), []).append(record)
    used_verifiers: set[str] = set()
    for case in sorted(bundle.cases, key=lambda item: item.case_id):
        current = current_by_case.get(case.case_id)
        finding = findings_by_case.get(case.case_id)
        if current is None or finding is None:
            continue
        is_round_one = any(
            item.assessment_id in refinement_assessment_ids for item in case.members
        )
        verification_round = 1 if is_round_one else 0
        route: str | None = None
        if is_round_one:
            reroute = next(
                (
                    item
                    for item in current.dependencies
                    if item.capability == "refinement_git_reroute"
                ),
                None,
            )
            expected = (
                current.integrity == "valid"
                and reroute is not None
                and reroute.status == "complete"
            )
        else:
            route = round_zero_routes.get(case.case_id)
            expected = (
                route is not None
                and route
                not in {
                    "not_required_fast_path",
                    "not_required_hard_invalid",
                    "not_required_current_incomplete",
                }
                and case.grouping_status != "incomplete"
                and current.integrity != "invalid"
            )
        key = (
            frozenset(item.member_id for item in case.members),
            current.bundle_id,
        )
        candidates = verifier_by_inputs.get(key, [])
        if not expected:
            if candidates:
                errors.append(
                    f"verifier record exists for host-resolved case: {case.case_id}"
                )
            if (
                not is_round_one
                and route == "not_required_current_incomplete"
            ):
                expected_host_finding = _host_finding(
                    case,
                    current,
                    profile_digest=bundle.index.profile_digest,
                    status="not_required",
                    reason_code="required_current_capability_incomplete",
                    detail_reason_codes=tuple(current.incomplete_reasons),
                )
                if finding != expected_host_finding:
                    errors.append(
                        "required-current gap has no exact host finding: "
                        f"{case.case_id}"
                    )
            elif not is_round_one and route == "not_required_fast_path":
                expected_host_finding = _host_finding(
                    case,
                    current,
                    profile_digest=bundle.index.profile_digest,
                    status="not_required",
                    reason_code="explicit_local_positive_fast_path",
                )
                if finding != expected_host_finding:
                    errors.append(
                        "explicit local-positive fast path has no exact host "
                        f"finding: {case.case_id}"
                    )
            continue
        if not candidates:
            reason = closure_replay.verifier_omission_case_reasons.get(
                case.case_id
            )
            if reason is not None:
                expected_host_finding = (
                    _failed_finding(
                        case=case,
                        reason_codes=(
                            "round_one_verifier_budget_exhausted",
                            reason,
                        ),
                    )
                    if verification_round == 1
                    else _host_finding(
                        case,
                        current,
                        profile_digest=bundle.index.profile_digest,
                        status="failed",
                        reason_code=(
                            "unified_case_verifier_budget_exhausted"
                        ),
                        detail_reason_codes=(reason,),
                    )
                )
                if finding != expected_host_finding:
                    errors.append(
                        "compact verifier omission has no exact host finding: "
                        f"{case.case_id}"
                    )
                continue
        if len(candidates) != 1:
            errors.append(
                f"case has missing or duplicate verifier records: {case.case_id}"
            )
            continue
        record = candidates[0]
        used_verifiers.add(record.request_id)
        matrix = matrices_by_case.get(case.case_id)
        assessments = [
            assessments_by_id[item.assessment_id] for item in case.members
        ]
        inputs = [
            *(
                InferenceInputRef(
                    kind="candidate_case_member",
                    stable_id=item.member_id,
                    digest=content_digest(item.model_dump(mode="json")),
                    snapshot_digest=case.snapshot_digest,
                )
                for item in case.members
            ),
            InferenceInputRef(
                kind="current_evidence_bundle",
                stable_id=current.bundle_id,
                digest=current.artifact_digest,
                snapshot_digest=case.snapshot_digest,
            ),
        ]
        if matrix is not None:
            inputs.append(
                InferenceInputRef(
                    kind="temporal_evidence_bundle",
                    stable_id=matrix.matrix_id,
                    digest=matrix.artifact_digest,
                    snapshot_digest=case.snapshot_digest,
                )
            )
        identifiers = [
            case.case_id,
            current.bundle_id,
            *(item.member_id for item in case.members),
        ]
        if matrix is not None:
            identifiers.append(matrix.matrix_id)
        try:
            if record.mode != f"candidate_case_round_{verification_round}":
                raise ValueError("verifier record round/mode differs")
            request_payload = build_unified_case_verify_prompt(
                case,
                assessments,
                current,
                matrix,
                verification_round=verification_round,
            )
            provider_response_model = _verifier_provider_response_model(
                tuple(
                    sorted(
                        allowed_verifier_evidence_refs(
                            assessments,
                            current,
                            matrix,
                        )
                    )
                )
            )
            request_over_limit = (
                contractual_request_bytes(
                    request_payload,
                    provider_response_model,
                )
                > settings.max_case_input_bytes
            )
            _validate_saved_request(
                record,
                inputs=inputs,
                request_ordinal=_stable_request_ordinal(
                    f"unified_case_verify:{verification_round}", identifiers
                ),
                request_payload=request_payload,
                response_model=provider_response_model,
            )
            request_gate_reason = "max_unified_verifier_request_bytes"
            if request_over_limit:
                zero_usage = (
                    record.usage.calls,
                    record.usage.input_tokens,
                    record.usage.output_tokens,
                    record.usage.reasoning_tokens,
                    record.usage.latency_ms,
                ) == (0, 0, 0, 0, 0)
                if (
                    record.terminal_status != "failure"
                    or record.structured_response is not None
                    or record.attempts
                    or not zero_usage
                    or record.incomplete_reasons != [request_gate_reason]
                ):
                    raise ValueError(
                        "over-limit verifier request is not the exact "
                        "pre-provider typed skip"
                    )
                expected_gate_finding = (
                    _failed_finding(
                        case=case,
                        reason_codes=("round_one_verifier_failed",),
                    )
                    if verification_round == 1
                    else _host_finding(
                        case,
                        current,
                        profile_digest=bundle.index.profile_digest,
                        status="failed",
                        reason_code="unified_case_verifier_failed",
                    )
                )
                if finding != expected_gate_finding:
                    raise ValueError(
                        "over-limit verifier request has no exact failed finding"
                    )
                continue
            if request_gate_reason in record.incomplete_reasons:
                raise ValueError(
                    "within-limit verifier request carries a forged byte-gate skip"
                )
            if record.terminal_status == "success":
                replayed = replay_unified_case_verify_record(
                    record,
                    case,
                    assessments,
                    current,
                    matrix,
                    verification_round=verification_round,  # type: ignore[arg-type]
                )
                if replayed != finding:
                    errors.append(
                        f"saved unified finding differs from response: {case.case_id}"
                    )
            elif finding.status != "failed" or finding.inference_ref is not None:
                errors.append(
                    f"failed verifier record has no truthful host finding: {case.case_id}"
                )
        except (ValidationError, ValueError, TypeError, KeyError) as error:
            errors.append(
                f"{record.request_id}: unified finding response replay failed: "
                f"{_short_error(error)}"
            )
    orphaned_verifiers = sorted(
        item.request_id
        for item in verifier_records
        if item.request_id not in used_verifiers
    )
    if orphaned_verifiers:
        errors.append(f"orphaned unified verifier records: {orphaned_verifiers}")
    return errors


def _refinement_closure(
    bundle: UnifiedBundle,
    git_values: Mapping[str, BaseModel],
    errors: list[str],
) -> None:
    """Replay the host-owned, one-round refinement lineage graph."""

    from repo_constraint_memory_v3.unified.refinement import (
        _make_child_assessment,
        _make_child_member,
        _proposal_key,
        rebind_child_git_need,
        validate_refinement_proposals,
    )

    assessments = {item.assessment_id: item for item in bundle.assessments}
    cases = {item.case_id: item for item in bundle.cases}
    current_by_id = {item.bundle_id: item for item in bundle.current_evidence}
    current_by_case = {item.case_id: item for item in bundle.current_evidence}
    finding_by_id = {item.finding_id: item for item in bundle.findings}
    finding_by_case = {item.case_id: item for item in bundle.findings}
    bundle_git_refs = {item.stable_id: item for item in bundle.git_bundle_refs}
    member_occurrences: dict[str, list[tuple[CandidateCase, object]]] = {}
    for case in bundle.cases:
        for member in case.members:
            member_occurrences.setdefault(member.member_id, []).append((case, member))

    recorded_child_ids: set[str] = set()
    recorded_parent_ids: set[str] = set()
    child_case_ids: set[str] = set()
    for record in bundle.refinements:
        parent_case = cases.get(record.case_id)
        if parent_case is None:
            errors.append(
                f"refinement parent case is missing: {record.refinement_id}"
            )
            continue
        parent_matches = [
            item
            for item in parent_case.members
            if item.member_id == record.parent_member_id
        ]
        if len(parent_matches) != 1:
            errors.append(
                f"refinement parent member closure failed: {record.refinement_id}"
            )
            continue
        if record.parent_member_id in recorded_parent_ids:
            errors.append(
                f"refinement parent is executed more than once: {record.parent_member_id}"
            )
        recorded_parent_ids.add(record.parent_member_id)
        parent_member = parent_matches[0]
        parent_assessment = assessments.get(parent_member.assessment_id)
        if (
            parent_assessment is None
            or _is_unified_refinement_assessment(parent_assessment)
            or parent_assessment.candidate_version > 1
        ):
            errors.append(
                f"refinement parent assessment is not round zero: {record.refinement_id}"
            )
            continue

        parent_finding = finding_by_case.get(parent_case.case_id)
        parent_proposals = (
            []
            if parent_finding is None or parent_finding.verification_round != 0
            else [
                item
                for item in parent_finding.refinement_proposals
                if item.parent_member_id == parent_member.member_id
            ]
        )
        if len(parent_proposals) != len(record.child_member_ids):
            errors.append(
                f"refinement proposal/child cardinality mismatch: {record.refinement_id}"
            )

        candidate_child_cases = [
            case
            for case in bundle.cases
            if set(record.child_member_ids)
            <= {item.member_id for item in case.members}
        ]
        if len(candidate_child_cases) != 1:
            errors.append(
                f"refinement child case closure failed: {record.refinement_id}"
            )
            continue
        child_case = candidate_child_cases[0]
        child_case_ids.add(child_case.case_id)
        parent_inactive = "refinement_parent_superseded" in (
            parent_case.grouping_reason_codes
        )
        child_review = "refinement_review_branch" in (
            child_case.grouping_reason_codes
        )
        if record.status == "complete" and (not parent_inactive or child_review):
            errors.append(
                f"complete refinement active-case markers are invalid: {record.refinement_id}"
            )
        if record.status == "review" and (parent_inactive or not child_review):
            errors.append(
                f"review refinement active-case markers are invalid: {record.refinement_id}"
            )
        expected_members = {
            item.member_id
            for item in parent_case.members
            if item.member_id != parent_member.member_id
        } | set(record.child_member_ids)
        if {item.member_id for item in child_case.members} != expected_members:
            errors.append(
                f"refinement child case is not a parent replacement: {record.refinement_id}"
            )

        # The proposed judgment belongs to the final child, with citations
        # rebound from parent evidence by exact immutable provenance. Replaying
        # only semantics/version would allow a resealed child to inherit or
        # invent affirmative demand that no terminal model response established.
        parent_current = current_by_case.get(parent_case.case_id)
        child_current = current_by_case.get(child_case.case_id)
        expected_children: dict[str, ChannelAssessment] = {}
        if parent_current is None or child_current is None:
            errors.append(f"refinement Git need replay evidence missing: {record.refinement_id}")
        else:
            validated, proposal_errors = validate_refinement_proposals(
                case=parent_case,
                assessments=assessments,
                current_evidence=parent_current,
                proposals=parent_proposals,
                max_children=2,
                parent_finding=parent_finding,
            )
            if proposal_errors:
                errors.append(f"refinement Git need/proposal binding invalid: {record.refinement_id}")
            else:
                for ordinal, proposal in enumerate(sorted(validated, key=_proposal_key)):
                    child = _make_child_assessment(parent=parent_assessment, proposal=proposal, ordinal=ordinal)
                    member = _make_child_member(child)
                    child = rebind_child_git_need(
                        child,
                        parent_current=parent_current,
                        child_current=child_current,
                        child_member_id=member.member_id,
                    )
                    expected_children[member.member_id] = child

        child_semantics: list[str] = []
        for child_id in record.child_member_ids:
            if child_id in recorded_child_ids:
                errors.append(f"refinement child is reused: {child_id}")
            recorded_child_ids.add(child_id)
            occurrences = member_occurrences.get(child_id, [])
            if len(occurrences) != 1 or occurrences[0][0].case_id != child_case.case_id:
                errors.append(f"refinement child member closure failed: {child_id}")
                continue
            child_member = occurrences[0][1]
            child_assessment = assessments.get(child_member.assessment_id)
            if child_assessment is None:
                errors.append(f"refinement child assessment is missing: {child_id}")
                continue
            if expected_children.get(child_id) != child_assessment:
                errors.append(f"refinement child assessment/Git need differs from rebound proposal: {child_id}")
            if (
                child_assessment.candidate_version
                != parent_assessment.candidate_version + 1
                or child_assessment.parent_candidate_id
                != parent_assessment.assessment_id
                or child_assessment.channel != parent_assessment.channel
                or child_assessment.snapshot_digest != parent_assessment.snapshot_digest
                or child_assessment.profile_digest != parent_assessment.profile_digest
                or child_member.semantics != child_assessment.semantics
                or child_assessment.semantics.normalized_action
                != parent_assessment.semantics.normalized_action
                or child_assessment.semantics.polarity
                != parent_assessment.semantics.polarity
            ):
                errors.append(f"refinement child lineage mismatch: {child_id}")
            child_semantics.append(
                content_digest(child_assessment.semantics.model_dump(mode="json"))
            )

        proposal_semantics = sorted(
            content_digest(item.semantics.model_dump(mode="json"))
            for item in parent_proposals
        )
        if sorted(child_semantics) != proposal_semantics:
            errors.append(
                f"refinement child semantics differ from proposal: {record.refinement_id}"
            )
        expected_change_kinds = sorted(
            {
                change
                for proposal in parent_proposals
                for change in proposal.change_kinds
            }
        )
        if record.change_kinds != expected_change_kinds:
            errors.append(
                f"refinement change kinds differ from proposal: {record.refinement_id}"
            )

        current_values = [
            current_by_id[item]
            for item in record.reacquired_current_bundle_ids
            if item in current_by_id
        ]
        if (
            len(current_values) != len(record.reacquired_current_bundle_ids)
            or len(current_values) != 1
            or current_values[0].case_id != child_case.case_id
        ):
            errors.append(
                f"refinement current-evidence closure failed: {record.refinement_id}"
            )
            current_value = None
        else:
            current_value = current_values[0]

        finding_values = [
            finding_by_id[item]
            for item in record.finding_ids
            if item in finding_by_id
        ]
        if (
            len(finding_values) != len(record.finding_ids)
            or len(finding_values) != 1
            or finding_values[0].case_id != child_case.case_id
            or finding_values[0].verification_round != 1
        ):
            errors.append(
                f"refinement round-one finding closure failed: {record.refinement_id}"
            )
            finding_value = None
        else:
            finding_value = finding_values[0]

        for ref in record.git_bundle_refs:
            if bundle_git_refs.get(ref.stable_id) != ref:
                errors.append(
                    f"refinement Git ref closure failed: {record.refinement_id}:{ref.stable_id}"
                )
        expected_git_refs = sorted(
            (
                ref
                for ref in bundle.git_bundle_refs
                if getattr(git_values.get(ref.stable_id), "case_id", None)
                == child_case.case_id
            ),
            key=lambda item: (item.kind, item.stable_id),
        )
        if record.git_bundle_refs != expected_git_refs:
            errors.append(
                f"refinement Git reroute refs are incomplete: {record.refinement_id}"
            )

        if record.status == "complete":
            reroute = (
                None
                if current_value is None
                else next(
                    (
                        item
                        for item in current_value.dependencies
                        if item.capability == "refinement_git_reroute"
                    ),
                    None,
                )
            )
            if (
                current_value is None
                or current_value.status != "complete"
                or current_value.integrity != "valid"
                or current_value.incomplete_reasons
                or reroute is None
                or reroute.requirement_mode != "required"
                or reroute.status != "complete"
            ):
                errors.append(
                    f"complete refinement lacks complete reacquisition: {record.refinement_id}"
                )
            if (
                finding_value is None
                or finding_value.status != "complete"
                or finding_value.refinement_proposals
            ):
                errors.append(
                    f"complete refinement lacks terminal round-one finding: {record.refinement_id}"
                )

    unified_child_assessment_ids = {
        item.assessment_id
        for item in bundle.assessments
        if _is_unified_refinement_assessment(item)
    }
    recorded_child_assessment_ids = {
        member.assessment_id
        for case in bundle.cases
        for member in case.members
        if member.member_id in recorded_child_ids
    }
    if unified_child_assessment_ids != recorded_child_assessment_ids:
        errors.append("refinement assessments are orphaned or unrecorded")
    refined_case_ids = {
        case.case_id
        for case in bundle.cases
        if any(
            assessments.get(member.assessment_id) is not None
            and _is_unified_refinement_assessment(
                assessments[member.assessment_id]
            )
            for member in case.members
        )
    }
    if refined_case_ids != child_case_ids:
        errors.append("refinement child cases are orphaned or unrecorded")

    refinement_capabilities = [
        item
        for item in bundle.completeness.capabilities
        if item.capability == "unified_refinement"
    ]
    refinement_contract_present = bool(
        bundle.refinements
        or unified_child_assessment_ids
        or any(item.refinement_proposals for item in bundle.findings)
    )
    if refinement_contract_present and len(refinement_capabilities) != 1:
        errors.append("unified refinement capability is missing or duplicated")
    elif refinement_contract_present:
        capability = refinement_capabilities[0]
        proposal_parent_ids = {
            proposal.parent_member_id
            for finding in bundle.findings
            for proposal in finding.refinement_proposals
        }
        proposed = bool(proposal_parent_ids)
        # A host-invalid proposal intentionally has no RefinementRecord.  The
        # status therefore cannot be replayed from the materialized records
        # alone: their parent set must cover the proposal denominator too.
        complete = (
            bool(bundle.refinements)
            and proposal_parent_ids == recorded_parent_ids
            and all(item.status == "complete" for item in bundle.refinements)
            and not unified_child_assessment_ids - recorded_child_assessment_ids
        )
        if not proposed and (
            capability.requirement_mode != "not_required"
            or capability.status != "not_required"
        ):
            errors.append("unneeded refinement capability is not not-required")
        if proposed and complete and capability.status != "complete":
            errors.append("complete refinement capability is not complete")
        if proposed and not complete and capability.status in _SUCCESS_CAPABILITY:
            errors.append("incomplete refinement capability claims success")


def _recompute_policy(
    bundle: UnifiedBundle,
    git_values: Mapping[str, BaseModel],
    errors: list[str],
) -> None:
    from repo_constraint_memory_v3.unified.policy import (
        CanonicalPolicyAccumulator,
    )

    assessments = {item.assessment_id: item for item in bundle.assessments}
    current_by_case = {item.case_id: item for item in bundle.current_evidence}
    matrix_by_case = {item.case_id: item for item in bundle.temporal_matrices}
    finding_by_case = {item.case_id: item for item in bundle.findings}
    git_by_case = {
        item.case_id: item
        for item in git_values.values()
        if isinstance(item, GitTemporalBundle)
    }
    accumulator = CanonicalPolicyAccumulator()

    case_by_id = {item.case_id: item for item in bundle.cases}
    child_case_by_refinement: dict[str, CandidateCase] = {}
    child_case_ids: set[str] = set()
    refinements_by_parent: dict[str, list[RefinementRecord]] = {}
    for refinement in bundle.refinements:
        candidates = [
            case
            for case in bundle.cases
            if set(refinement.child_member_ids)
            <= {item.member_id for item in case.members}
        ]
        if len(candidates) != 1:
            continue
        child_case_by_refinement[refinement.refinement_id] = candidates[0]
        child_case_ids.add(candidates[0].case_id)
        refinements_by_parent.setdefault(refinement.case_id, []).append(refinement)

    for case in sorted(
        (item for item in bundle.cases if item.case_id not in child_case_ids),
        key=lambda item: item.case_id,
    ):
        current = current_by_case.get(case.case_id)
        if current is None:
            continue
        finding = finding_by_case.get(case.case_id)
        finding_for_policy = finding
        try:
            if is_explicit_fast_path(
                case=case,
                assessments=assessments,
                current_evidence=current,
            ):
                finding_for_policy = None
            result = apply_case_policy(
                case=case,
                assessments=assessments,
                current_evidence=current,
                temporal_matrix=matrix_by_case.get(case.case_id),
                git_bundle=git_by_case.get(case.case_id),
                finding=finding_for_policy,
            )
        except (KeyError, ValueError) as error:
            errors.append(
                f"frozen policy replay failed for {case.case_id}: {_short_error(error)}"
            )
            continue
        accumulator.absorb(result)

        refinements = sorted(
            refinements_by_parent.get(case.case_id, []),
            key=lambda item: item.refinement_id,
        )
        execution_complete = bool(refinements) and all(
            item.status == "complete" for item in refinements
        )
        for refinement in refinements:
            child_case = child_case_by_refinement.get(refinement.refinement_id)
            if child_case is None:
                continue
            child_current = current_by_case.get(child_case.case_id)
            child_finding = finding_by_case.get(child_case.case_id)
            if child_current is None or child_finding is None:
                continue
            try:
                child_result = apply_case_policy(
                    case=child_case,
                    assessments=assessments,
                    current_evidence=child_current,
                    temporal_matrix=matrix_by_case.get(child_case.case_id),
                    git_bundle=git_by_case.get(child_case.case_id),
                    finding=child_finding,
                )
            except (KeyError, ValueError) as error:
                errors.append(
                    "frozen refinement policy replay failed for "
                    f"{child_case.case_id}: {_short_error(error)}"
                )
                continue
            accumulator.absorb(
                child_result,
                include_claim_member_ids=(None if execution_complete else set()),
                prefer_dispositions=execution_complete,
                supersedes_case_id=(case.case_id if execution_complete else None),
            )

        for refinement in refinements:
            if refinement.status != "complete":
                continue
            try:
                accumulator.supersede_member(
                    parent_member_id=refinement.parent_member_id,
                    child_member_ids=refinement.child_member_ids,
                )
            except KeyError:
                errors.append(
                    "refinement parent disposition is missing during policy replay: "
                    f"{refinement.parent_member_id}"
                )

    aggregation = accumulator.finalize()
    expected_claims = list(aggregation.claims)
    expected_dispositions = list(aggregation.dispositions)
    actual_claims = sorted(bundle.canonical_claims, key=lambda item: item.claim_id)
    actual_dispositions = sorted(
        bundle.member_dispositions, key=lambda item: item.member_id
    )
    if [item.model_dump(mode="json") for item in expected_claims] != [
        item.model_dump(mode="json") for item in actual_claims
    ]:
        errors.append("canonical claims differ from frozen policy replay")
    if [item.model_dump(mode="json") for item in expected_dispositions] != [
        item.model_dump(mode="json") for item in actual_dispositions
    ]:
        errors.append("member dispositions differ from frozen policy replay")
    capability = next(
        (
            item
            for item in bundle.completeness.capabilities
            if item.capability == "canonical_identity_resolution"
        ),
        None,
    )
    if capability is None:
        errors.append("canonical identity resolution capability is missing")
    elif (
        capability.requirement_mode != "required"
        or capability.status
        != ("partial" if aggregation.collision_claim_ids else "complete")
        or capability.reason_codes != list(aggregation.reason_codes)
    ):
        errors.append(
            "canonical identity resolution capability differs from policy replay"
        )


def _completeness_errors(bundle: UnifiedBundle, *, require_complete: bool) -> list[str]:
    errors: list[str] = []
    completeness = bundle.completeness
    if not completeness.structurally_valid or completeness.package_status == "invalid":
        errors.append("unified package is structurally invalid")
        return errors
    gaps = [
        item
        for item in completeness.capabilities
        if item.status not in _SUCCESS_CAPABILITY
    ]
    expected_status = "partial" if gaps else "complete"
    if completeness.package_status != expected_status:
        errors.append("unified package status does not reflect capability gaps")
    expected_reasons = sorted(
        {reason for item in gaps for reason in item.reason_codes}
    )
    if completeness.reasons != expected_reasons:
        errors.append("unified completeness reasons do not equal capability gaps")
    if gaps and not expected_reasons:
        errors.append("partial capability gap has no typed reason")
    if completeness.omitted_possible != bool(gaps):
        errors.append("omitted_possible does not reflect package completeness")
    if require_complete and completeness.package_status != "complete":
        errors.append(INCOMPLETE_PACKAGE_ERROR)
    return errors


def _artifact_set_errors(
    bundle: UnifiedBundle,
    git_values: dict[str, BaseModel],
) -> list[str]:
    git_bundles = [
        item for item in git_values.values() if isinstance(item, GitTemporalBundle)
    ]
    digests = sorted(
        [
            *(item.artifact_digest for item in bundle.assessments),
            *(item.artifact_digest for item in bundle.cases),
            *(item.artifact_digest for item in bundle.current_evidence),
            *(item.artifact_digest for item in bundle.temporal_matrices),
            *(item.artifact_digest for item in bundle.findings),
            *(item.artifact_digest for item in bundle.refinements),
            *(item.artifact_digest for item in bundle.canonical_claims),
            *(item.artifact_digest for item in git_bundles),
            content_digest(bundle.funnel.model_dump(mode="json")),
            content_digest(bundle.completeness.model_dump(mode="json")),
        ]
    )
    if bundle.index.artifact_set_digest != content_digest(digests):
        return ["unified index artifact-set digest mismatch"]
    return []


def _reconstructed_bundle_reference_errors(bundle: UnifiedBundle) -> list[str]:
    """Replay ``UnifiedBundle`` cross-object checks without a giant re-dump.

    The run does not contain a serialized ``UnifiedBundle`` or an externally
    anchored digest for that transient aggregate.  Every persisted component
    has already passed its own schema/digest validator, and the index carries
    the independently replayed artifact-set digest.  Re-serializing all those
    components merely to create and then verify a self-generated aggregate
    digest duplicates the largest object graph.  Keep every substantive global
    reference and algebra check from ``UnifiedBundle`` here instead.
    """

    errors: list[str] = []
    if bundle.index.channel_assessment_ids != sorted(
        item.assessment_id for item in bundle.assessments
    ):
        errors.append("unified assessment index mismatch")
    if bundle.index.case_ids != sorted(item.case_id for item in bundle.cases):
        errors.append("unified case index mismatch")
    if bundle.index.finding_ids != sorted(
        item.finding_id for item in bundle.findings
    ):
        errors.append("unified finding index mismatch")
    if bundle.index.canonical_claim_ids != sorted(
        item.claim_id for item in bundle.canonical_claims
    ):
        errors.append("unified canonical index mismatch")

    member_ids = {
        item.member_id for case in bundle.cases for item in case.members
    }
    inactive_markers = {
        "refinement_parent_superseded",
        "refinement_review_branch",
    }
    active_cases = [
        case
        for case in bundle.cases
        if not inactive_markers.intersection(case.grouping_reason_codes)
    ]
    active_members = [
        item.member_id for case in active_cases for item in case.members
    ]
    if len(active_members) != len(set(active_members)):
        errors.append("a member belongs to multiple active cases")
    active_case_ids = {item.case_id for item in active_cases}
    active_case_members = {
        item.case_id: {member.member_id for member in item.members}
        for item in active_cases
    }
    for claim in bundle.canonical_claims:
        if (
            claim.case_id not in active_case_ids
            or not set(claim.contributing_member_ids)
            <= active_case_members[claim.case_id]
        ):
            errors.append("canonical claim references an inactive or foreign case")
            break

    disposition_ids = [
        item.member_id for item in bundle.member_dispositions
    ]
    if (
        len(disposition_ids) != len(set(disposition_ids))
        or set(disposition_ids) != member_ids
    ):
        errors.append("every case member requires one final disposition")
    all_case_members = {
        item.case_id: {member.member_id for member in item.members}
        for item in bundle.cases
    }
    claim_ids = {item.claim_id for item in bundle.canonical_claims}
    git_ref_keys = {
        (
            item.kind,
            item.stable_id,
            item.snapshot_digest,
            item.artifact_path,
            item.artifact_digest,
        )
        for item in bundle.git_bundle_refs
    }
    for matrix in bundle.temporal_matrices:
        if matrix.git_manifest_ref is None:
            continue
        ref = matrix.git_manifest_ref
        if (
            ref.kind,
            ref.stable_id,
            ref.snapshot_digest,
            ref.artifact_path,
            ref.artifact_digest,
        ) not in git_ref_keys:
            errors.append("temporal matrix references a foreign Git manifest")
            break
    for disposition in bundle.member_dispositions:
        if (
            disposition.case_id not in all_case_members
            or disposition.member_id
            not in all_case_members[disposition.case_id]
            or not set(disposition.canonical_claim_ids) <= claim_ids
        ):
            errors.append("member disposition references a foreign case or claim")
            break
    if (
        bundle.funnel.accepted
        + bundle.funnel.review
        + bundle.funnel.rejected
        != len(bundle.canonical_claims)
    ):
        errors.append("unified funnel canonical count algebra is invalid")
    return errors


def _parent_directories(paths: set[str]) -> set[str]:
    directories: set[str] = set()
    for relative in paths:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _consumer_errors(
    tree: _TreeSnapshot,
    bundle: UnifiedBundle,
    *,
    repository_name: str,
    git_values: Mapping[str, BaseModel],
    canonical_file: CanonicalClaimsFile,
    completeness_file: CanonicalCompletenessFile,
    delivery_report: CanonicalDeliveryReport,
) -> tuple[list[str], ConsumerArtifacts | None]:
    errors: list[str] = []
    artifacts: ConsumerArtifacts | None = None
    try:
        artifacts = build_consumer_artifacts(
            bundle=bundle,
            repository_name=repository_name,
            profiles=delivery_report.requested_consumer_profiles,
            git_artifacts=git_values,
        )
        expected_canonical = {
            "canonical/claims.json": artifacts.canonical_claims,
            "canonical/completeness.json": artifacts.completeness,
            "canonical/delivery_report.json": artifacts.delivery_report,
        }
        for relative, expected in expected_canonical.items():
            if _snapshot_bytes(tree, relative, errors) != canonical_json_text(
                expected
            ).encode("utf-8"):
                errors.append(
                    f"canonical machine artifact differs from replay: {relative}"
                )
        if canonical_file != artifacts.canonical_claims:
            errors.append("canonical claims model differs from bundle replay")
        if completeness_file != artifacts.completeness:
            errors.append("canonical completeness model differs from bundle replay")
        if delivery_report != artifacts.delivery_report:
            errors.append("canonical delivery report differs from bundle replay")

        expected_files: dict[str, bytes] = {}
        expected_directories = {"consumers"}
        for profile, plan in artifacts.views.items():
            prefix = f"consumers/{profile}"
            expected_directories.add(prefix)
            expected_directories.update(
                f"{prefix}/{relative}" for relative in plan.directories
            )
            expected_files.update(
                {
                    f"{prefix}/{relative}": data
                    for relative, data in plan.files.items()
                }
            )
        actual_files = {
            path for path in tree.files if path.startswith("consumers/")
        }
        actual_directories = {
            path
            for path in tree.directories
            if path == "consumers" or path.startswith("consumers/")
        }
        if actual_files != set(expected_files):
            errors.append(
                "consumer closed-tree file set differs from replay: "
                f"missing={sorted(set(expected_files) - actual_files)}, "
                f"foreign={sorted(actual_files - set(expected_files))}"
            )
        if actual_directories != expected_directories:
            errors.append(
                "consumer closed-tree directory set differs from replay: "
                f"missing={sorted(expected_directories - actual_directories)}, "
                f"foreign={sorted(actual_directories - expected_directories)}"
            )
        for relative, expected in expected_files.items():
            if _snapshot_bytes(tree, relative, errors) != expected:
                errors.append(f"consumer generated bytes differ from replay: {relative}")
    except ValueError as error:
        errors.append(f"consumer surface reconstruction failed: {_short_error(error)}")
    if any(
        PurePosixPath(path).name == "by-trigger"
        for path in [*tree.files, *tree.directories]
    ):
        errors.append("consumer leakage: by-trigger must not exist")
    return errors, artifacts


_UNIFIED_REPORT_STAGES = {
    "candidate_semantic_group",
    "temporal_evidence_classify",
    "unified_case_verify",
}


def _report_usage(records: list[InferenceRecord]) -> dict[str, int | None]:
    return {
        "record_count": len(records),
        "calls": sum(item.usage.calls for item in records),
        "input_tokens": sum(item.usage.input_tokens for item in records),
        "output_tokens": sum(item.usage.output_tokens for item in records),
        "reasoning_tokens": (
            sum(item.usage.reasoning_tokens or 0 for item in records)
            if all(item.usage.reasoning_tokens is not None for item in records)
            else None
        ),
    }


def _build_report_closure_errors(
    tree: _TreeSnapshot,
    bundle: UnifiedBundle,
    *,
    consumer_artifacts: ConsumerArtifacts | None,
) -> list[str]:
    """Recompute run-dependent report fields from the exported closure."""

    errors: list[str] = []
    report = _json_value(tree, "build_report.json", errors)
    if not isinstance(report, dict):
        return [*errors, "build_report.json is missing or is not an object"]
    entrypoint = report.get("entrypoint")
    if entrypoint not in {"full", "build-unified"}:
        errors.append("build_report entrypoint is missing or invalid")
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("build_report schema version differs from active schema")
    if report.get("snapshot_digest") != bundle.index.snapshot_digest:
        errors.append("build_report snapshot differs from unified bundle")
    saved_config = _json_value(tree, "config.json", errors)
    if not isinstance(saved_config, dict) or report.get("config") != saved_config:
        errors.append("build_report config differs from saved config")
    versions = report.get("versions")
    if not isinstance(versions, dict):
        errors.append("build_report versions are missing")
        versions = {}
    if versions.get("unified_contract") != bundle.index.contract_version:
        errors.append("build_report unified contract differs from audit index")
    if versions.get("unified_policy") != bundle.index.policy_version:
        errors.append("build_report unified policy differs from audit index")
    expected_complete = bundle.completeness.package_status == "complete"
    if report.get("complete") is not expected_complete:
        errors.append("build_report complete differs from unified completeness")
    if report.get("incomplete_reasons") != sorted(bundle.completeness.reasons):
        errors.append(
            "build_report incomplete reasons differ from unified completeness"
        )

    delivered = [
        item
        for item in bundle.canonical_claims
        if item.epistemic_status == "accepted"
        and item.delivery_policy != "reference_only"
    ]
    expected_unified: dict[str, object] = {
        "bundle_id": bundle.index.bundle_id,
        "profile_digest": bundle.index.profile_digest,
        "case_count": len(bundle.cases),
        "assessment_count": len(bundle.assessments),
        "temporal_matrix_count": len(bundle.temporal_matrices),
        "finding_count": len(bundle.findings),
        "refinement_count": len(bundle.refinements),
        "canonical_counts": {
            "accepted": bundle.index.accepted_count,
            "review": bundle.index.review_count,
            "rejected": bundle.index.rejected_count,
            "delivered": len(delivered),
        },
        "completeness": bundle.completeness.model_dump(mode="json"),
    }
    unified_report = report.get("unified")
    if not isinstance(unified_report, dict):
        errors.append("build_report unified summary is missing")
    else:
        for key, expected in expected_unified.items():
            if unified_report.get(key) != expected:
                errors.append(f"build_report unified {key} differs from replay")

    expected_consumer_surface = (
        None
        if consumer_artifacts is None
        else consumer_artifacts.report_metadata()
    )
    if expected_consumer_surface is None:
        errors.append("build_report consumer surface could not be replayed")
    elif report.get("consumer_surface") != expected_consumer_surface:
        errors.append("build_report consumer surface differs from replay")

    inference_index = _parse_model_file(
        tree,
        "inference/index.json",
        InferenceIndex,
        errors,
        required=True,
        label="inference/index.json for build_report replay",
    )
    if inference_index is None:
        return errors
    if (
        report.get("package_version")
        != inference_index.protocol_manifest.package_version
    ):
        errors.append("build_report package version differs from protocol manifest")
    if (
        versions.get("protocol_manifest")
        != inference_index.protocol_manifest.protocol_manifest_digest
    ):
        errors.append("build_report protocol manifest differs from inference index")
    records: list[InferenceRecord] = []
    for reference in inference_index.requests:
        record = _parse_model_file(
            tree,
            reference.artifact_path,
            InferenceRecord,
            errors,
            required=True,
            label=f"{reference.artifact_path} for build_report replay",
        )
        if record is not None:
            records.append(record)
    if len(records) != len(inference_index.requests):
        return errors

    try:
        from repo_constraint_memory_v3.exporters import (
            execution_model_run_summary,
        )
        from repo_constraint_memory_v3.source_acquisition import (
            load_source_acquisition,
        )

        report_config = BuildConfig.model_validate(saved_config)
        report_acquisition = load_source_acquisition(
            tree.root,
            expected_stage_settings=(
                report_config.agent.stages["source_discovery"]
            ),
        )
        expected_execution_model = execution_model_run_summary(
            report_config,
            report_acquisition,
            records,
        )
    except (OSError, KeyError, RuntimeError, ValidationError, ValueError) as error:
        errors.append(
            "build_report execution-model summary could not be replayed: "
            f"{_short_error(error)}"
        )
    else:
        if report.get("execution_model") != expected_execution_model:
            errors.append("build_report execution model summary differs from replay")

        grouping_report = (
            unified_report.get("candidate_grouping")
            if isinstance(unified_report, dict)
            else None
        )
        if not isinstance(grouping_report, dict):
            errors.append("build_report candidate grouping summary is missing")
        else:
            grouping_stage = next(
                (
                    item
                    for item in inference_index.stages
                    if item.stage == "candidate_semantic_group"
                ),
                None,
            )
            planned_grouping_calls = (
                grouping_stage.planned if grouping_stage is not None else 0
            )
            terminal_grouping_calls = (
                (
                    grouping_stage.success
                    + grouping_stage.uncertain
                    + grouping_stage.failure
                )
                if grouping_stage is not None
                else 0
            )
            refinement_assessment_ids = {
                item.assessment_id
                for item in bundle.assessments
                if _is_unified_refinement_assessment(item)
            }
            round_zero_cases = [
                case
                for case in bundle.cases
                if not any(
                    member.assessment_id in refinement_assessment_ids
                    for member in case.members
                )
            ]
            expected_mode = report_config.unified.candidate_grouping_mode
            if expected_mode == "singleton":
                expected_status = "not_required"
                expected_reason = "candidate_grouping_mode_singleton"
            elif not round_zero_cases:
                expected_status = "not_required"
                expected_reason = "no_candidate_members"
            else:
                expected_status = (
                    "complete"
                    if terminal_grouping_calls == planned_grouping_calls
                    and all(
                        case.grouping_status == "complete"
                        for case in round_zero_cases
                    )
                    else "partial"
                )
                expected_reason = None
            expected_grouping_fields = {
                "mode": expected_mode,
                "status": expected_status,
                "reason": expected_reason,
                "planned_pairs": bundle.funnel.grouping_planned_pairs,
                "planned_calls": planned_grouping_calls,
                "terminal_calls": terminal_grouping_calls,
                "case_count": len(round_zero_cases),
            }
            for key, expected in expected_grouping_fields.items():
                if grouping_report.get(key) != expected:
                    errors.append(
                        "build_report candidate grouping "
                        f"{key} differs from replay"
                    )
            planned_pairs = grouping_report.get("planned_pairs")
            if (
                isinstance(planned_pairs, bool)
                or not isinstance(planned_pairs, int)
                or planned_pairs < planned_grouping_calls
                or (
                    expected_status == "not_required"
                    and planned_pairs != 0
                )
            ):
                errors.append(
                    "build_report candidate grouping planned_pairs is invalid"
                )

    llm = report.get("llm")
    if not isinstance(llm, dict):
        errors.append("build_report llm summary is missing")
        return errors
    stable_llm = {
        "attempt_count": sum(len(item.attempts) for item in records),
        "schema_repairs": sum(
            attempt.repair for item in records for attempt in item.attempts
        ),
        "provider_retries": sum(
            attempt.provider_try > 0
            for item in records
            for attempt in item.attempts
        ),
        "output_limit_retries": sum(
            attempt.output_limit_retry > 0
            for item in records
            for attempt in item.attempts
        ),
        "input_tokens": inference_index.total_input_tokens,
        "output_tokens": inference_index.total_output_tokens,
        "reasoning_tokens": inference_index.total_reasoning_tokens,
        "record_count": len(inference_index.requests),
        "inference_set_digest": inference_index.inference_set_digest,
    }
    for key, expected in stable_llm.items():
        if llm.get(key) != expected:
            errors.append(f"build_report llm {key} differs from inference replay")

    if entrypoint == "build-unified":
        channel_records = [
            item for item in records if item.stage not in _UNIFIED_REPORT_STAGES
        ]
        unified_records = [
            item for item in records if item.stage in _UNIFIED_REPORT_STAGES
        ]
        if llm.get("prior_channel_usage") != _report_usage(channel_records):
            errors.append(
                "build_report prior channel usage differs from inference replay"
            )
        current_usage = llm.get("this_run_unified_usage")
        if not isinstance(current_usage, dict):
            errors.append("build_report this-run unified usage is missing")
        else:
            for key, expected in _report_usage(unified_records).items():
                if current_usage.get(key) != expected:
                    errors.append(
                        "build_report this-run unified usage differs from "
                        f"inference replay: {key}"
                    )
            for key in ("provider_calls", "cache_hits"):
                value = current_usage.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    errors.append(
                        f"build_report this-run unified {key} is invalid"
                    )
            if llm.get("this_run_provider_calls") != current_usage.get(
                "provider_calls"
            ):
                errors.append("build_report provider-call telemetry is inconsistent")
            if llm.get("this_run_cache_hits") != current_usage.get("cache_hits"):
                errors.append("build_report cache-hit telemetry is inconsistent")
    return errors


def _channel_replay_errors(
    tree: _TreeSnapshot,
    repository_root: Path,
    bundle: UnifiedBundle,
    upstream_binding: _UpstreamReplayBinding | None,
    git_values: Mapping[str, BaseModel],
) -> list[str]:
    """Replay both discovery channels and bind their immutable assessments.

    This deliberately reuses the existing deterministic observation/current-
    source replay and structured-inference validators.  It never invokes a
    provider and passes an empty explicit context so isolated and full builds
    share the schema-4.11 independent-channel contract.
    """

    from repo_constraint_memory_v3.adjudication import (
        CandidateDecision,
        candidate_fingerprint,
        finalize_candidate_decisions,
        prepare_candidate_decisions,
    )
    from repo_constraint_memory_v3.builder import (
        _apply_explicit_verification,
        _source_funnel,
        _verify_discovery,
        protocol_manifest_for_config,
    )
    from repo_constraint_memory_v3.discovery import DiscoveryError
    from repo_constraint_memory_v3.exporters import (
        _audit_text,
        evidence_relations_payload,
        execution_model_config_errors,
        execution_model_inference_errors,
        explicit_verification_payload,
    )
    from repo_constraint_memory_v3.current_source_run import _observation_coverage
    from repo_constraint_memory_v3.models import (
        AttemptAudit,
        AuditRecord,
        BuildConfig,
        CandidateAuditItem,
        DiscoveryArtifact,
        SourceSupplement,
    )
    from repo_constraint_memory_v3.observations import run_observations
    from repo_constraint_memory_v3.observations.semantic_units import (
        build_semantic_units,
        validate_semantic_unit_tree,
    )
    from repo_constraint_memory_v3.observations.validation import (
        validate_observation_tree,
    )
    from repo_constraint_memory_v3.run_validation import _validate_inference_tree
    from repo_constraint_memory_v3.semantic.validation import (
        validate_current_source_tree,
    )
    from repo_constraint_memory_v3.source_acquisition import (
        SourceAcquisitionError,
        load_source_acquisition,
        validate_acquisition_for_build,
    )
    from repo_constraint_memory_v3.source_plan import validate_source_plan
    from repo_constraint_memory_v3.unified.inputs import (
        build_blocking_anchors,
        build_case_evidence_inputs,
        build_case_input_index,
        build_channel_assessment_result,
        build_observation_anchor_index,
    )
    from repo_constraint_memory_v3.traceability import (
        TraceabilityArtifact,
        build_explicit_channel,
        make_traceability,
        not_run_channel,
    )
    from repo_constraint_memory_v3.unified.current_evidence import (
        CurrentEvidenceLimits,
        assemble_current_evidence,
    )
    from repo_constraint_memory_v3.unified.pipeline import (
        _build_unified_funnel,
        _case_members,
        _dependency_for_git,
        _dependency_for_temporal,
        _git_anchors,
        _git_limits,
        _route_case,
        _with_dependencies,
    )
    from repo_constraint_memory_v3.unified.git_evidence import (
        _query_candidates,
        plan_git_queries,
    )
    from repo_constraint_memory_v3.unified.upstream import (
        CandidateDecisionRecord,
        CurrentSourceAdapterInputs,
        ObservationShardInputs,
    )

    errors: list[str] = []
    config = _parse_model_file(
        tree,
        "config.json",
        BuildConfig,
        errors,
        required=True,
    )
    discovery = _parse_model_file(
        tree,
        "discovery.json",
        DiscoveryArtifact,
        errors,
        required=True,
    )
    supplement = _parse_model_file(
        tree,
        "source_supplement.json",
        SourceSupplement,
        errors,
        required=True,
    )
    if config is None or discovery is None or supplement is None:
        return errors
    try:
        acquisition = load_source_acquisition(
            tree.root,
            expected_stage_settings=config.agent.stages["source_discovery"],
        )
        validate_acquisition_for_build(acquisition, config)
    except SourceAcquisitionError as error:
        return [
            *errors,
            *(f"source acquisition replay failed: {item}" for item in error.errors),
        ]
    except (OSError, ValueError) as error:
        return [
            *errors,
            f"source acquisition replay failed: {_short_error(error)}",
        ]
    if acquisition.discovery != discovery:
        errors.append("source acquisition discovery differs from discovery.json")
    if acquisition.supplement != supplement:
        errors.append("source acquisition supplement differs from artifact")
    errors.extend(execution_model_config_errors(config, acquisition.model_profile))
    try:
        replayed_discovery = _verify_discovery(
            repository_root.resolve(), config, discovery
        )
    except (DiscoveryError, OSError, ValueError) as error:
        errors.append(f"fixed-snapshot discovery replay failed: {_short_error(error)}")
        return errors
    if replayed_discovery.repository.snapshot_digest != bundle.index.snapshot_digest:
        errors.append("channel snapshot differs from unified snapshot")
    if (
        replayed_discovery.repository.git_head_for_reference
        != bundle.index.snapshot_commit
    ):
        errors.append("channel snapshot commit differs from unified index")
    inference_index, records, results_by_unit, sources, units, verification = (
        _validate_inference_tree(
            tree.root,
            repository_root=repository_root.resolve(),
            discovery=replayed_discovery,
            supplement=supplement,
            config=config,
            agent_profile_digest=acquisition.acquisition.agent_profile_sha256,
            errors=errors,
        )
    )
    protocol = protocol_manifest_for_config(
        config,
        agent_profile_digest=acquisition.acquisition.agent_profile_sha256,
    )
    if inference_index is not None:
        protocol = protocol_manifest_for_readonly_replay(
            protocol,
            producer_package_version=inference_index.protocol_manifest.package_version,
        )
    if (
        protocol.execution_model_profile_id != acquisition.model_profile.profile_id
        or protocol.execution_model_profile_digest
        != acquisition.model_profile.profile_digest
    ):
        errors.append(
            "channel protocol execution-model profile differs from acquisition"
        )
    if protocol.protocol_manifest_digest != bundle.index.profile_digest:
        errors.append("channel protocol profile differs from unified profile")

    errors.extend(
        execution_model_inference_errors(records, acquisition.model_profile)
    )
    omission_ledger = _UnifiedOmissionLedger.from_index(
        inference_index,
        config.llm,
        errors,
    )
    omission_ledger.seed_channel_records(records)
    expected_current_omissions = (
        {
            item.stage: (item.omitted, item.omission_reasons)
            for item in inference_index.stages
            if item.stage in _CURRENT_INFERENCE_STAGES and item.omitted
        }
        if inference_index is not None
        else None
    )
    units_by_id = {item.unit_id: item for item in units}
    request_by_unit = {
        item.stable_id: record.request_id
        for record in records
        for item in record.input_manifest
        if item.kind == "explicit_unit"
    }
    raw_candidates = [
        (result.unit_id, result.reason_code, candidate)
        for result in results_by_unit.values()
        for candidate in getattr(result, "claims", ())
    ]
    decisions = prepare_candidate_decisions(
        raw_candidates,
        units_by_id=units_by_id,
        sources_by_path={item.path: item for item in sources},
        manifest_paths={item.path for item in replayed_discovery.manifest.files},
        snapshot_digest=replayed_discovery.repository.snapshot_digest,
        inference_request_by_unit=request_by_unit,
    )
    if verification is None:
        errors.append("explicit verification replay is unavailable")
    else:
        _apply_explicit_verification(decisions, verification)
    adjudication = finalize_candidate_decisions(decisions)

    # The saved channel artifact remains part of the self-contained replay
    # surface even though the unified envelope is the isolated-build input.
    # Compare the whole typed projection so coordinated additions or edits to
    # this legacy-named audit file cannot become unaudited bytes.
    if verification is not None:
        expected_verification = explicit_verification_payload(
            bundle.index.snapshot_digest,
            verification,
        )
        if _json_value(
            tree, "explicit_verification.json", errors
        ) != expected_verification:
            errors.append(
                "explicit_verification.json differs from deterministic replay"
            )

    observations = None
    current_source = None
    if config.observation.enabled:
        try:
            observations = run_observations(
                repository_root.resolve(),
                replayed_discovery,
                config.observation,
                cache_directory=None,
            ).bundle
        except (OSError, ValueError) as error:
            errors.append(f"observation replay failed: {_short_error(error)}")
        if observations is not None:
            errors.extend(
                validate_observation_tree(
                    tree.root,
                    repository_root=repository_root.resolve(),
                    discovery=replayed_discovery,
                    settings=config.observation,
                    coverage=_observation_coverage(observations),
                    semantic_units_enabled=config.semantic.enabled,
                )
            )
            if config.semantic.enabled:
                errors.extend(
                    validate_semantic_unit_tree(
                        tree.root,
                        repository_root=repository_root.resolve(),
                        discovery=replayed_discovery,
                        observations=observations,
                        settings=config.semantic,
                    )
                )
                try:
                    semantic_units = build_semantic_units(
                        repository_root.resolve(),
                        replayed_discovery,
                        observations,
                        config.semantic,
                    )
                except (OSError, TypeError, ValueError) as error:
                    errors.append(
                        f"semantic-unit replay failed: {_short_error(error)}"
                    )
                else:
                    current_errors, current_source = validate_current_source_tree(
                        tree.root,
                        repository_root=repository_root.resolve(),
                        discovery=replayed_discovery,
                        observations=observations,
                        semantic_units=semantic_units,
                        semantic_settings=config.semantic,
                        llm_settings=config.llm,
                        protocol_manifest_digest=protocol.protocol_manifest_digest,
                        inference_records=records,
                        explicit_context=(),
                        expected_stage_omissions=expected_current_omissions,
                    )
                    errors.extend(current_errors)
    elif "observations" in tree.directories:
        errors.append("observation artifacts exist while observation is disabled")

    explicit_records = [
        record for record in records if record.stage == "explicit_extract"
    ]
    sent_unit_ids = {
        item.stable_id
        for record in explicit_records
        for item in record.input_manifest
        if item.kind == "explicit_unit"
    }
    failed_unit_ids = {
        item.stable_id
        for record in explicit_records
        if record.terminal_status != "success"
        for item in record.input_manifest
        if item.kind == "explicit_unit"
    }
    dropped_unit_ids = {item.unit_id for item in units} - sent_unit_ids
    explicit_candidate_enumeration_complete = (
        replayed_discovery.complete
        and not dropped_unit_ids
        and not failed_unit_ids
    )
    expected_assessments: tuple[ChannelAssessment, ...] = ()
    assessment_omissions = ()
    round_zero_cases: tuple[CandidateCase, ...] = ()
    replayed_grouping_execution = None
    round_zero_routes: dict[str, str] = {}

    if verification is not None:
        assessment_build = build_channel_assessment_result(
            explicit_decisions=adjudication.decisions,
            explicit_verification=verification.result,
            current_source=current_source,
            snapshot_digest=bundle.index.snapshot_digest,
            profile_digest=bundle.index.profile_digest,
        )
        expected_assessments = assessment_build.assessments
        assessment_omissions = assessment_build.omissions
        actual_roots = tuple(
            sorted(
                (
                    item
                    for item in bundle.assessments
                    if not _is_unified_refinement_assessment(item)
                ),
                key=lambda item: item.assessment_id,
            )
        )
        if expected_assessments != actual_roots:
            errors.append(
                "channel assessments differ from deterministic upstream replay"
            )

        root_members = _case_members(expected_assessments)
        input_index = build_case_input_index(
            assessments=expected_assessments,
            explicit_decisions=adjudication.decisions,
            observations=observations,
            current_source=current_source,
        )
        anchors_by_member = build_blocking_anchors(
            input_index=input_index,
            members=root_members,
            assessments=expected_assessments,
            explicit_decisions=adjudication.decisions,
            current_source=current_source,
            observations=observations,
        )
        (
            grouping_errors,
            round_zero_cases,
            replayed_grouping_execution,
        ) = _replay_round_zero_grouping(
            bundle,
            list(expected_assessments),
            anchors_by_member,
            records,
            config.unified,
            omission_ledger,
            max_concurrency=config.llm.max_concurrency,
            max_attempts_per_request=max_provider_calls_per_request(config.llm),
        )
        errors.extend(grouping_errors)
        saved_current_by_case = {
            item.case_id: item for item in bundle.current_evidence
        }
        saved_matrix_by_case = {
            item.case_id: item for item in bundle.temporal_matrices
        }
        saved_git_by_case = {
            item.case_id: item
            for item in git_values.values()
            if isinstance(item, GitTemporalBundle)
        }
        assessment_by_id = {
            item.assessment_id: item for item in expected_assessments
        }
        all_assessments_by_id = {
            item.assessment_id: item for item in bundle.assessments
        }
        expected_git_limits = _git_limits(config.unified)
        cases_by_id = {item.case_id: item for item in bundle.cases}
        for git_bundle in saved_git_by_case.values():
            if git_bundle.manifest.limits != expected_git_limits:
                errors.append(
                    f"Git manifest limits differ from frozen profile: {git_bundle.case_id}"
                )
            errors.extend(_git_preflight_plan_errors(git_bundle))
            git_case = cases_by_id.get(git_bundle.case_id)
            git_current = saved_current_by_case.get(git_bundle.case_id)
            if git_case is None or git_current is None:
                continue
            try:
                git_route = _route_case(
                    git_case,
                    all_assessments_by_id,
                    git_current,
                )
            except (KeyError, ValueError) as error:
                errors.append(
                    f"Git route replay failed for {git_bundle.case_id}: "
                    f"{_short_error(error)}"
                )
            else:
                if git_route != git_bundle.manifest.purpose:
                    errors.append(
                        f"Git manifest purpose differs from case route: {git_bundle.case_id}"
                    )
            git_anchors = _git_anchors(
                git_case,
                git_current,
                max_blame_lines=expected_git_limits.max_blame_lines,
            )
            expected_manifest_anchors = (
                []
                if git_bundle.preflight.status == "not_required"
                else list(git_anchors)
            )
            if git_bundle.manifest.anchors != expected_manifest_anchors:
                errors.append(
                    f"Git manifest anchors differ from host plan: {git_bundle.case_id}"
                )
            expected_commands = (
                []
                if git_bundle.preflight.status == "not_required"
                else plan_git_queries(
                    bundle.index.snapshot_commit,
                    git_anchors,
                    limits=expected_git_limits,
                )
            )
            if git_bundle.manifest.commands != expected_commands:
                errors.append(
                    f"Git manifest command plan differs from host plan: {git_bundle.case_id}"
                )
            if git_bundle.preflight.status != "not_required":
                candidates = _query_candidates(
                    bundle.index.snapshot_commit,
                    git_anchors,
                    expected_git_limits,
                )
                selected_counts = {
                    family: sum(
                        item.family == family for item in expected_commands
                    )
                    for family in candidates
                }
                for family_result in git_bundle.manifest.family_results:
                    expected_omitted = max(
                        0,
                        len(candidates[family_result.family])
                        - selected_counts[family_result.family],
                    )
                    if (
                        family_result.planned_count
                        != selected_counts[family_result.family]
                        or family_result.omitted_count != expected_omitted
                    ):
                        errors.append(
                            "Git family plan counts differ from host plan: "
                            f"{git_bundle.case_id}:{family_result.family}"
                        )
        observation_anchor_index = build_observation_anchor_index(observations)
        base_evidence_by_case: dict[str, CurrentEvidenceBundle] = {}
        for case in round_zero_cases:
            try:
                evidence_inputs = build_case_evidence_inputs(
                    input_index=input_index,
                    case=case,
                    assessments=expected_assessments,
                    explicit_decisions=adjudication.decisions,
                    current_source=current_source,
                    observations=observations,
                    explicit_discovery_complete=(
                        explicit_candidate_enumeration_complete
                    ),
                    anchors_by_member=anchors_by_member,
                    observation_anchor_index=observation_anchor_index,
                    max_observation_context_items=(
                        config.unified.max_observation_context_items_per_case
                    ),
                )
                base_evidence = assemble_current_evidence(
                    case,
                    explicit_evidence=[
                        *evidence_inputs.explicit_evidence,
                        *evidence_inputs.observation_evidence,
                    ],
                    population_evidence=evidence_inputs.population_evidence,
                    denominators=evidence_inputs.denominators,
                    population_refs=evidence_inputs.population_refs,
                    falsification_refs=evidence_inputs.falsification_refs,
                    scope_condition_refs=evidence_inputs.scope_condition_refs,
                    dependencies=evidence_inputs.dependencies,
                    limits=CurrentEvidenceLimits(
                        max_items=config.unified.max_current_evidence_items,
                        max_bytes=config.unified.max_current_evidence_bytes,
                    ),
                ).bundle
                base_evidence_by_case[case.case_id] = base_evidence
                round_zero_routes[case.case_id] = _route_case(
                    case, assessment_by_id, base_evidence
                )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(
                    f"current-evidence replay failed for {case.case_id}: "
                    f"{_short_error(error)}"
                )

        budget_git_skip_reasons: dict[str, str] = {}
        errors.extend(
            _exhaustive_unified_inference_replay_errors(
                bundle,
                git_values,
                records,
                settings=config.unified,
                round_zero_routes=round_zero_routes,
                omission_ledger=omission_ledger,
                git_enabled=config.unified.git_enabled,
                snapshot_commit_available=(
                    bundle.index.snapshot_commit is not None
                ),
                max_concurrency=config.llm.max_concurrency,
                max_attempts_per_request=max_provider_calls_per_request(
                    config.llm
                ),
                budget_git_skip_reasons=budget_git_skip_reasons,
            )
        )
        errors.extend(
            _refinement_budget_dependency_errors(
                bundle,
                git_values,
                budget_git_skip_reasons,
            )
        )

        for case in round_zero_cases:
            base_evidence = base_evidence_by_case.get(case.case_id)
            route = round_zero_routes.get(case.case_id)
            if base_evidence is None or route is None:
                continue
            git_bundle = saved_git_by_case.get(case.case_id)
            matrix = saved_matrix_by_case.get(case.case_id)
            expected_git = (
                not route.startswith("not_required")
                and config.unified.git_enabled
                and bundle.index.snapshot_commit is not None
            )
            budget_skip_reason = budget_git_skip_reasons.get(case.case_id)
            try:
                expected_evidence = _with_dependencies(
                    base_evidence,
                    [
                        _dependency_for_git(
                            route,
                            git_bundle,
                            host_skip_reason=budget_skip_reason,
                        ),
                        _dependency_for_temporal(
                            route,
                            git_bundle,
                            matrix,
                            host_skip_reason=budget_skip_reason,
                        ),
                    ],
                )
            except ValueError as error:
                errors.append(
                    f"current-evidence replay failed for {case.case_id}: "
                    f"{_short_error(error)}"
                )
                continue
            if saved_current_by_case.get(case.case_id) != expected_evidence:
                errors.append(
                    f"round-zero current evidence differs from replay: {case.case_id}"
                )
            if expected_git and git_bundle is None and budget_skip_reason is None:
                errors.append(
                    f"Git bundle presence differs from deterministic route: {case.case_id}"
                )
            if not expected_git and git_bundle is not None:
                errors.append(
                    f"Git bundle presence differs from deterministic route: {case.case_id}"
                )
            if git_bundle is not None:
                replayed_anchors = _git_anchors(
                    case,
                    expected_evidence,
                    max_blame_lines=expected_git_limits.max_blame_lines,
                )
                expected_manifest_anchors = (
                    []
                    if git_bundle.preflight.status == "not_required"
                    else list(replayed_anchors)
                )
                if git_bundle.manifest.anchors != expected_manifest_anchors:
                    errors.append(
                        f"Git manifest anchors differ from host plan: {case.case_id}"
                    )
                expected_commands = (
                    []
                    if git_bundle.preflight.status == "not_required"
                    else plan_git_queries(
                        bundle.index.snapshot_commit,
                        replayed_anchors,
                        limits=git_bundle.manifest.limits,
                    )
                )
                if git_bundle.manifest.commands != expected_commands:
                    errors.append(
                        f"Git manifest command plan differs from host plan: {case.case_id}"
                    )

        errors.extend(omission_ledger.finish())
        # Also bind semantic relations inherited by refinement child cases to
        # the successful round-zero grouping responses.
        errors.extend(_unified_inference_replay_errors(bundle, git_values, records))

        explicit_channel_complete = (
            explicit_candidate_enumeration_complete
            and all(
                decision.raw_candidate is not None
                and len(decision.grounded)
                == len(decision.raw_candidate.evidence)
                for decision in adjudication.decisions
            )
            and verification.result.coverage.status == "complete"
        )
        explicit_traces, explicit_events, explicit_channel = build_explicit_channel(
            snapshot_digest=bundle.index.snapshot_digest,
            decisions=adjudication.decisions,
            extraction_complete=explicit_channel_complete,
        )
        expected_traceability = make_traceability(
            snapshot_digest=bundle.index.snapshot_digest,
            explicit_traces=explicit_traces,
            explicit_events=explicit_events,
            explicit_channel=explicit_channel,
            current_traces=(current_source.traces if current_source is not None else ()),
            current_events=(
                current_source.event_traces if current_source is not None else ()
            ),
            current_channel=(
                current_source.trace_channel
                if current_source is not None
                else not_run_channel("disabled_by_config")
            ),
        )
        actual_traceability = _parse_model(
            TraceabilityArtifact,
            _json_value(tree, "traceability.json", errors),
            "traceability.json",
            errors,
        )
        if actual_traceability is None:
            errors.append("traceability.json schema error")
        if (
            actual_traceability is not None
            and actual_traceability != expected_traceability
        ):
            errors.append("traceability.json differs from deterministic replay")
        report_value = _json_value(tree, "build_report.json", errors)
        if isinstance(report_value, dict) and report_value.get(
            "channel_funnels"
        ) != {
            "explicit": expected_traceability.explicit.model_dump(mode="json"),
            "current_source": expected_traceability.current_source.model_dump(
                mode="json"
            ),
        }:
            errors.append("build_report channel_funnels differ from traceability replay")

    if _json_value(
        tree, "evidence_relations.json", errors
    ) != evidence_relations_payload(
        bundle.index.snapshot_digest,
        adjudication.evidence_relations,
    ):
        errors.append("evidence_relations.json differs from adjudication replay")

    try:
        if replayed_grouping_execution is None:
            raise ValueError("grouping execution replay is unavailable")
        replayed_funnel = _build_unified_funnel(
            cases=bundle.cases,
            current_evidence=bundle.current_evidence,
            git_bundles=[item for item in git_values.values() if isinstance(item, GitTemporalBundle)],
            assessments=bundle.assessments,
            claims=bundle.canonical_claims,
            dispositions=bundle.member_dispositions,
            grouping_execution=replayed_grouping_execution,
            raw_candidate_events_by_channel={
                "explicit": len(raw_candidates),
                "implicit_current": (
                    current_source.index.candidate_events
                    if current_source is not None
                    else 0
                ),
            },
            assessment_omissions=assessment_omissions,
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"unified funnel replay failed: {_short_error(error)}")
    else:
        if bundle.funnel != replayed_funnel:
            errors.append("unified funnel differs from deterministic replay")

    decisions_by_unit: dict[str, list[CandidateDecision]] = {}
    for decision in adjudication.decisions:
        decisions_by_unit.setdefault(decision.unit_id, []).append(decision)
    for unit_id, result in results_by_unit.items():
        if (
            getattr(result, "classification", None) != "uncertain"
            or getattr(result, "claims", ())
        ):
            continue
        decisions_by_unit.setdefault(unit_id, []).append(
            CandidateDecision(
                unit_id=unit_id,
                candidate_fingerprint=candidate_fingerprint(unit_id, None),
                outcome="review",
                outcome_stage="semantics",
                reason_code=result.reason_code,
                raw_candidate=None,
                errors=["model returned a bounded uncertain abstention"],
            )
        )
    explicit_record_by_unit = {
        item.stable_id: record
        for record in explicit_records
        for item in record.input_manifest
        if item.kind == "explicit_unit"
    }
    expected_audit: list[AuditRecord] = []
    for unit in units:
        result = results_by_unit.get(unit.unit_id)
        local_decisions = decisions_by_unit.get(unit.unit_id, [])
        inference_record = explicit_record_by_unit.get(unit.unit_id)
        if inference_record is None:
            classification = "not_sent"
            local_outcome = "failed"
            reason = "resource_or_safety_deferred"
            validation_errors = [
                "unit was not sent because a bounded resource/safety limit applied"
            ]
        elif inference_record.terminal_status != "success" or result is None:
            classification = "failed"
            local_outcome = "failed"
            reason = "extraction_failed"
            validation_errors = list(inference_record.incomplete_reasons)
        elif result.classification == "no_claims":
            classification = result.classification
            local_outcome = "rejected"
            reason = result.reason_code
            validation_errors = []
        elif result.classification == "uncertain":
            classification = result.classification
            local_outcome = "review"
            reason = result.reason_code
            validation_errors = [
                error for decision in local_decisions for error in decision.errors
            ]
        else:
            classification = result.classification
            local_outcome = (
                "review"
                if any(item.outcome != "accepted" for item in local_decisions)
                else "accepted"
            )
            reason = result.reason_code
            validation_errors = [
                error
                for decision in local_decisions
                if decision.outcome != "accepted"
                for error in decision.errors
            ]
        expected_audit.append(
            AuditRecord(
                unit_id=unit.unit_id,
                path=unit.path,
                line_start=unit.line_start,
                line_end=unit.line_end,
                discovery_origin=unit.discovery_origin,
                inference_request_id=(
                    None if inference_record is None else inference_record.request_id
                ),
                model_classification=classification,  # type: ignore[arg-type]
                local_outcome=local_outcome,  # type: ignore[arg-type]
                reason_code=reason,
                claim_ids=sorted(
                    {
                        decision.claim_id
                        for decision in local_decisions
                        if decision.claim_id
                    }
                ),
                validation_errors=sorted(set(validation_errors)),
                attempts=[
                    AttemptAudit(
                        schema_attempt=item.schema_attempt,
                        output_limit_retry=item.output_limit_retry,
                        provider_try=item.provider_try,
                        repair=item.repair,
                        parsed=item.parsed,
                        provider_error=item.provider_error,
                        validation_errors=item.validation_errors,
                    )
                    for item in (
                        inference_record.attempts
                        if inference_record is not None
                        else []
                    )
                ],
                items=[
                    CandidateAuditItem(
                        candidate_fingerprint=decision.candidate_fingerprint,
                        verification_request_id=decision.verification_request_id,
                        verification_reason_codes=decision.verification_reason_codes,
                        verification_technical_reason_code=(
                            decision.verification_technical_reason_code
                        ),
                        local_outcome=decision.outcome,  # type: ignore[arg-type]
                        outcome_stage=decision.outcome_stage,  # type: ignore[arg-type]
                        reason_code=decision.reason_code,
                        rule=(
                            decision.raw_candidate.rule
                            if decision.raw_candidate is not None
                            else None
                        ),
                        claim_id=decision.claim_id,
                        validation_errors=sorted(set(decision.errors)),
                    )
                    for decision in local_decisions
                ],
            )
        )
    if _snapshot_bytes(tree, "audit.jsonl", errors) != _audit_text(
        expected_audit
    ).encode("utf-8"):
        errors.append("audit.jsonl differs from deterministic adjudication replay")

    if upstream_binding is not None:
        replayed_decisions = [
            CandidateDecisionRecord.from_runtime(item)
            for item in adjudication.decisions
        ]
        replayed_decision_digests = tuple(
            sorted(_stream_content_digest(item) for item in replayed_decisions)
        )
        if (
            replayed_decision_digests
            != upstream_binding.explicit_decision_digests
        ):
            errors.append(
                "unified upstream explicit decisions differ from channel replay"
            )
        if (
            verification is not None
            and _stream_content_digest(verification.result)
            != upstream_binding.explicit_verification_digest
        ):
            errors.append(
                "unified upstream explicit verification differs from channel replay"
            )
        replayed_current = (
            CurrentSourceAdapterInputs.from_runtime(current_source)
            if current_source is not None
            else None
        )
        replayed_current_digest = (
            _stream_content_digest(replayed_current)
            if replayed_current is not None
            else None
        )
        if replayed_current_digest != upstream_binding.current_source_digest:
            errors.append(
                "unified upstream current-source inputs differ from channel replay"
            )
        replayed_observations = (
            ObservationShardInputs(
                snapshot_digest=bundle.index.snapshot_digest,
                observation_set_digest=(
                    observations.index.observation_set_digest
                ),
                overall=observations.index.overall,
                incomplete_reasons=list(
                    observations.index.incomplete_reasons
                ),
                shards=list(observations.shards),
            )
            if observations is not None
            else None
        )
        replayed_observations_digest = (
            _stream_content_digest(replayed_observations)
            if replayed_observations is not None
            else None
        )
        if (
            replayed_observations_digest
            != upstream_binding.observation_shards_digest
        ):
            errors.append(
                "unified upstream observation shards differ from channel replay"
            )
        replayed_event_count = len(raw_candidates) + (
            current_source.index.candidate_events
            if current_source is not None
            else 0
        )
        if replayed_event_count != upstream_binding.raw_candidate_events:
            errors.append(
                "unified upstream raw candidate count differs from channel replay"
            )
        explicit_records = [
            record for record in records if record.stage == "explicit_extract"
        ]
        sent_unit_ids = {
            item.stable_id
            for record in explicit_records
            for item in record.input_manifest
            if item.kind == "explicit_unit"
        }
        failed_unit_ids = {
            item.stable_id
            for record in explicit_records
            if record.terminal_status != "success"
            for item in record.input_manifest
            if item.kind == "explicit_unit"
        }
        dropped_unit_ids = {item.unit_id for item in units} - sent_unit_ids
        explicit_candidate_enumeration_complete = (
            replayed_discovery.complete
            and not dropped_unit_ids
            and not failed_unit_ids
        )
        if (
            explicit_candidate_enumeration_complete
            != upstream_binding.explicit_discovery_complete
        ):
            errors.append(
                "unified upstream explicit completeness differs from channel replay"
            )

    report = _json_value(tree, "build_report.json", errors)
    if not isinstance(report, dict):
        errors.append("build_report.json is missing or is not an object")
    else:
        versions = report.get("versions")
        if not isinstance(versions, dict):
            errors.append("build_report versions are missing")
        else:
            if versions.get("protocol_manifest") != protocol.protocol_manifest_digest:
                errors.append("build_report protocol manifest differs from replay")
            if (
                versions.get("agent_profile")
                != acquisition.acquisition.agent_profile_sha256
            ):
                errors.append("build_report agent profile differs from replay")
        try:
            replayed_plan = validate_source_plan(
                replayed_discovery,
                supplement,
                config.discovery,
            )
            failed_or_omitted = set(units_by_id) - set(results_by_unit)
            expected_funnel = _source_funnel(
                replayed_plan,
                units,
                results_by_unit,
                adjudication,
                failed_or_omitted,
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(
                f"build_report source funnel replay failed: {_short_error(error)}"
            )
        else:
            if report.get("source_funnel") != expected_funnel:
                errors.append(
                    "build_report source funnel differs from deterministic replay"
                )
    return errors


def validate_unified_directory(
    run_directory: Path,
    repository_root: Path,
    *,
    require_complete: bool = False,
    max_artifact_bytes: int = 1_000_000_000,
    validate_channels: bool = True,
) -> list[str]:
    """Validate a published unified package without provider or history queries."""

    if (
        isinstance(max_artifact_bytes, bool)
        or not isinstance(max_artifact_bytes, int)
        or max_artifact_bytes <= 0
    ):
        return ["max_artifact_bytes must be a positive integer"]
    tree, errors = _snapshot_tree(
        run_directory, max_artifact_bytes=max_artifact_bytes
    )
    if tree is None:
        return sorted(set(errors))
    raw_config = (
        _required_json(tree, "config.json", errors)
        if validate_channels
        else _json_value(tree, "config.json", errors)
    )
    profile_config = (
        _parse_model(BuildConfig, raw_config, "config.json", errors)
        if raw_config is not None
        else None
    )
    if (
        profile_config is not None
        and tree.total_bytes
        > profile_config.unified.max_unified_artifact_bytes
    ):
        errors.append(
            "artifact tree bytes exceed unified profile bound: "
            f"{tree.total_bytes} > "
            f"{profile_config.unified.max_unified_artifact_bytes}"
        )
    allowed_top_level = {
        "agent",
        "audit.jsonl",
        "build_report.json",
        "canonical",
        "config.json",
        "consumers",
        "current-source",
        "discovery.json",
        "evidence_relations.json",
        "explicit_verification.json",
        "inference",
        "model",
        "observations",
        "source_acquisition.json",
        "source_supplement.json",
        "traceability.json",
    }
    actual_top_level = {
        path.split("/", 1)[0] for path in [*tree.files, *tree.directories]
    }
    unexpected_top_level = sorted(actual_top_level - allowed_top_level)
    if unexpected_top_level:
        errors.append(
            "unexpected top-level artifacts in unified run: "
            f"{unexpected_top_level}"
        )
    if any(
        path == "by-trigger" or path.startswith("by-trigger/")
        for path in [*tree.files, *tree.directories]
    ):
        errors.append("delivery leakage: by-trigger must not exist")
    if any(
        len(parts := PurePosixPath(path).parts) >= 3
        and parts[0] == "consumers"
        and parts[2] == "model"
        for path in [*tree.files, *tree.directories]
    ):
        errors.append("consumer leakage: execution-model artifacts must not be exposed")

    raw_index = _required_json(tree, "canonical/audit/index.json", errors)
    index = (
        _parse_model(
            UnifiedIndex,
            raw_index,
            "canonical/audit/index.json",
            errors,
        )
        if raw_index is not None
        else None
    )
    if index is None:
        return sorted(set(errors))
    errors.extend(_git_head_errors(repository_root, index.snapshot_commit))
    consumer_repository_name = _consumer_repository_name(
        tree,
        repository_root,
        profile_config,
        index,
        errors,
        required=validate_channels,
    )
    snapshot = index.snapshot_digest

    assessments = _collection(
        tree,
        "canonical/audit/channel-assessments",
        ChannelAssessment,
        errors,
    )
    cases = _collection(tree, "canonical/audit/cases", CandidateCase, errors)
    current = _collection(
        tree,
        "canonical/audit/current-evidence",
        CurrentEvidenceBundle,
        errors,
    )
    matrices = _collection(
        tree,
        "canonical/audit/git/relations",
        TemporalRelationMatrix,
        errors,
    )
    findings = _collection(
        tree,
        "canonical/audit/findings",
        UnifiedCaseFinding,
        errors,
    )
    refinements = _collection(
        tree,
        "canonical/audit/refinements",
        RefinementRecord,
        errors,
    )
    canonical_file = _parse_model_file(
        tree,
        "canonical/claims.json",
        CanonicalClaimsFile,
        errors,
        required=True,
        label="canonical/claims.json",
    )
    claims = [] if canonical_file is None else canonical_file.claims
    dispositions = _wrapper_list(
        tree,
        "canonical/audit/member-dispositions.json",
        "items",
        MemberDisposition,
        snapshot,
        errors,
    )
    raw_funnel = _required_json(tree, "canonical/audit/funnel.json", errors)
    funnel = (
        _parse_model(
            UnifiedFunnel,
            raw_funnel,
            "canonical/audit/funnel.json",
            errors,
        )
        if raw_funnel is not None
        else None
    )
    completeness_file = _parse_model_file(
        tree,
        "canonical/completeness.json",
        CanonicalCompletenessFile,
        errors,
        required=True,
        label="canonical/completeness.json",
    )
    completeness = None if completeness_file is None else completeness_file.state
    delivery_report = _parse_model_file(
        tree,
        "canonical/delivery_report.json",
        CanonicalDeliveryReport,
        errors,
        required=True,
        label="canonical/delivery_report.json",
    )
    refs, git_values = _owned_git_artifacts(
        tree, snapshot, {item.case_id for item in cases}, errors
    )
    errors.extend(
        _git_object_replay_errors(repository_root, index.snapshot_commit, git_values)
    )

    components = [
        index,
        funnel,
        completeness,
        canonical_file,
        completeness_file,
        delivery_report,
    ]
    if any(item is None for item in components):
        return sorted(set(errors))
    assert (
        funnel is not None
        and completeness is not None
        and canonical_file is not None
        and completeness_file is not None
        and delivery_report is not None
    )
    if (
        canonical_file.snapshot_digest != index.snapshot_digest
        or canonical_file.snapshot_commit != index.snapshot_commit
        or canonical_file.profile_digest != index.profile_digest
    ):
        errors.append("canonical claims identity differs from audit index")
    if (
        completeness_file.snapshot_digest != index.snapshot_digest
        or completeness_file.snapshot_commit != index.snapshot_commit
        or completeness_file.profile_digest != index.profile_digest
    ):
        errors.append("canonical completeness identity differs from audit index")
    if (
        delivery_report.snapshot_digest != index.snapshot_digest
        or delivery_report.snapshot_commit != index.snapshot_commit
        or delivery_report.profile_digest != index.profile_digest
    ):
        errors.append("canonical delivery report identity differs from audit index")
    bundle_payload = {
        "index": index,
        "assessments": assessments,
        "cases": cases,
        "current_evidence": current,
        "git_bundle_refs": refs,
        "temporal_matrices": matrices,
        "findings": findings,
        "refinements": refinements,
        "canonical_claims": claims,
        "member_dispositions": dispositions,
        "funnel": funnel,
        "completeness": completeness,
    }
    bundle = UnifiedBundle.model_construct(
        **bundle_payload,
        # Validation-only aggregate: no serialized bundle digest exists in the
        # run.  Component and artifact-set digests are verified independently.
        artifact_digest="0" * 64,
    )
    bundle_reference_errors = _reconstructed_bundle_reference_errors(bundle)
    errors.extend(
        f"reconstructed unified bundle failed validation: {item}"
        for item in bundle_reference_errors
    )
    if bundle_reference_errors:
        return sorted(set(errors))

    # Reconstruct the serialized grouping denominator for the envelope and
    # final algebra checks below.  Full channel validation independently
    # rebuilds the grouping plan and compares these funnel fields with that
    # replay, so this projection cannot make a tampered denominator trusted.
    from repo_constraint_memory_v3.unified.pipeline import (
        CandidateGroupingExecution,
    )

    replayed_grouping_execution = CandidateGroupingExecution(
        mode=funnel.candidate_grouping_mode,
        status=funnel.grouping_status,
        reason=funnel.grouping_reason,
        planned_pairs=funnel.grouping_planned_pairs,
        planned_calls=funnel.grouping_planned_calls,
        terminal_calls=funnel.grouping_terminal_calls,
        case_count=funnel.grouping_case_count,
    )

    upstream = _upstream_inputs(
        tree,
        index,
        required=validate_channels,
        errors=errors,
    )
    upstream_binding: _UpstreamReplayBinding | None = None
    if upstream is not None:
        errors.extend(
            _upstream_bundle_errors(
                upstream,
                bundle,
                replayed_grouping_execution,
                git_values,
            )
        )
        upstream_binding = _upstream_replay_binding(
            upstream,
            streamed_value_digests=tree.json_value_digests.get(
                "canonical/audit/upstream-inputs.json"
            ),
        )
        # Channel replay reconstructs large observation/current-source graphs.
        # Retain only the authenticated equality commitments across that phase.
        del upstream

    expected_audit = _expected_audit_files(
        bundle,
        refs,
        include_upstream_inputs=(
            "canonical/audit/upstream-inputs.json" in tree.files
        ),
    )
    expected_canonical = expected_audit | {
        "canonical/claims.json",
        "canonical/completeness.json",
        "canonical/delivery_report.json",
    }
    actual_canonical = {
        path for path in tree.files if path.startswith("canonical/")
    }
    missing = sorted(expected_canonical - actual_canonical)
    foreign = sorted(actual_canonical - expected_canonical)
    if missing:
        errors.append(f"canonical tree is missing files: {missing}")
    if foreign:
        errors.append(f"canonical tree contains foreign files: {foreign}")
    expected_canonical_directories = _parent_directories(expected_canonical) | {
        "canonical/audit/channel-assessments",
        "canonical/audit/cases",
        "canonical/audit/current-evidence",
        "canonical/audit/findings",
        "canonical/audit/git/bundles",
        "canonical/audit/git/manifests",
        "canonical/audit/git/relations",
        "canonical/audit/git/slices",
        "canonical/audit/refinements",
    }
    actual_canonical_directories = {
        path
        for path in tree.directories
        if path == "canonical" or path.startswith("canonical/")
    }
    if actual_canonical_directories != expected_canonical_directories:
        errors.append(
            "canonical closed-tree directory set differs from replay: "
            "missing="
            f"{sorted(expected_canonical_directories - actual_canonical_directories)}, "
            "foreign="
            f"{sorted(actual_canonical_directories - expected_canonical_directories)}"
        )

    snapshot_objects: list[tuple[str, str, str]] = []
    snapshot_objects.extend(
        ("assessment", item.assessment_id, item.snapshot_digest)
        for item in bundle.assessments
    )
    snapshot_objects.extend(
        ("case", item.case_id, item.snapshot_digest) for item in bundle.cases
    )
    snapshot_objects.extend(
        ("current evidence", item.bundle_id, item.snapshot_digest)
        for item in bundle.current_evidence
    )
    snapshot_objects.extend(
        ("temporal matrix", item.matrix_id, item.snapshot_digest)
        for item in bundle.temporal_matrices
    )
    snapshot_objects.extend(
        ("finding", item.finding_id, item.snapshot_digest)
        for item in bundle.findings
    )
    snapshot_objects.extend(
        ("refinement", item.refinement_id, item.snapshot_digest)
        for item in bundle.refinements
    )
    snapshot_objects.extend(
        ("claim", item.claim_id, item.snapshot_digest)
        for item in bundle.canonical_claims
    )
    for kind, stable_id, value in snapshot_objects:
        if value != snapshot:
            errors.append(f"{kind} snapshot mismatch: {stable_id}")
    if any(
        item.profile_digest != index.profile_digest
        for item in [
            *bundle.assessments,
            *bundle.cases,
            *bundle.current_evidence,
            *bundle.temporal_matrices,
            *bundle.findings,
            *bundle.canonical_claims,
        ]
    ):
        errors.append("unified artifact profile digest mismatch")

    if validate_channels:
        errors.extend(
            _channel_replay_errors(
                tree,
                repository_root,
                bundle,
                upstream_binding,
                git_values,
            )
        )
    _reference_closure(bundle, git_values, tree, errors)
    _recompute_policy(bundle, git_values, errors)
    errors.extend(_completeness_errors(bundle, require_complete=require_complete))
    errors.extend(_artifact_set_errors(bundle, git_values))
    consumer_errors, consumer_artifacts = _consumer_errors(
        tree,
        bundle,
        repository_name=consumer_repository_name,
        git_values=git_values,
        canonical_file=canonical_file,
        completeness_file=completeness_file,
        delivery_report=delivery_report,
    )
    errors.extend(consumer_errors)
    if "build_report.json" in tree.files:
        errors.extend(
            _build_report_closure_errors(
                tree,
                bundle,
                consumer_artifacts=consumer_artifacts,
            )
        )

    counts = {
        "accepted": sum(
            item.epistemic_status == "accepted" for item in bundle.canonical_claims
        ),
        "review": sum(
            item.epistemic_status == "review" for item in bundle.canonical_claims
        ),
        "rejected": sum(
            item.epistemic_status == "rejected" for item in bundle.canonical_claims
        ),
    }
    if (
        index.accepted_count != counts["accepted"]
        or index.review_count != counts["review"]
        or index.rejected_count != counts["rejected"]
    ):
        errors.append("unified index canonical counts mismatch")
    if (
        funnel.accepted != counts["accepted"]
        or funnel.review != counts["review"]
        or funnel.rejected != counts["rejected"]
    ):
        errors.append("unified funnel canonical counts mismatch")
    from repo_constraint_memory_v3.unified.inputs import (
        ASSESSMENT_CONSTRUCTION_ATTRITION,
        TERMINAL_INVALID_SCOPE_ATTRITION,
    )

    terminal_omissions = sum(
        funnel.attrition_reasons.get(reason, 0)
        for reason in (
            TERMINAL_INVALID_SCOPE_ATTRITION,
            ASSESSMENT_CONSTRUCTION_ATTRITION,
        )
    )
    if funnel.invalid_members != (
        sum(item.normalized_status == "invalid" for item in bundle.assessments)
        + terminal_omissions
    ):
        errors.append("unified funnel invalid-member count mismatch")
    if not terminal_omissions:
        if funnel.unique_candidate_roots != len(
            {
                (item.channel, item.candidate_root_id)
                for item in bundle.assessments
                if not _is_unified_refinement_assessment(item)
            }
        ):
            errors.append("unified funnel unique-root denominator mismatch")
        if funnel.refinement_versions != sum(
            item.candidate_version > 0 for item in bundle.assessments
        ):
            errors.append("unified funnel refinement-version count mismatch")
    try:
        from repo_constraint_memory_v3.unified.pipeline import (
            _build_unified_funnel,
        )

        replayed_funnel = (
            None
            if terminal_omissions
            else _build_unified_funnel(
                cases=bundle.cases,
                current_evidence=bundle.current_evidence,
                git_bundles=[item for item in git_values.values() if isinstance(item, GitTemporalBundle)],
                assessments=bundle.assessments,
                claims=bundle.canonical_claims,
                dispositions=bundle.member_dispositions,
                grouping_execution=replayed_grouping_execution,
                raw_candidate_events_by_channel={
                    channel: values["raw_candidate_events"]
                    for channel, values in funnel.by_channel.items()
                },
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"unified funnel algebra replay failed: {_short_error(error)}")
    else:
        # Candidate-local omissions have no rewritten semantic assessment by
        # design.  Their exact replay is bound above through the mandatory
        # upstream decision envelope; assessment-only algebra remains useful
        # for packages with no such terminal roots.
        if replayed_funnel is not None and funnel != replayed_funnel:
            errors.append("unified funnel differs from final disposition replay")
    if validate_channels and profile_config is not None:
        # HEAD alone does not bind a dirty worktree.  Recompute the complete
        # deterministic repository snapshot after all artifact replay so a
        # source-file change during validation cannot be accepted against the
        # earlier discovery result.
        from repo_constraint_memory_v3.builder import _verify_discovery
        from repo_constraint_memory_v3.discovery import (
            DiscoveryArtifact,
            DiscoveryError,
        )

        final_discovery = _parse_model_file(
            tree,
            "discovery.json",
            DiscoveryArtifact,
            errors,
            required=True,
            label="discovery.json for final snapshot replay",
        )
        if final_discovery is not None:
            try:
                _verify_discovery(
                    repository_root.resolve(),
                    profile_config,
                    final_discovery,
                )
            except (DiscoveryError, OSError, ValueError) as error:
                errors.append(
                    "repository changed during unified validation: "
                    f"{_short_error(error)}"
                )
    errors.extend(_tree_unchanged_errors(tree))
    errors.extend(_git_head_errors(repository_root, index.snapshot_commit))
    return sorted(set(errors))


__all__ = [
    "INCOMPLETE_PACKAGE_ERROR",
    "UNIFIED_VALIDATOR_VERSION",
    "validate_unified_directory",
]
