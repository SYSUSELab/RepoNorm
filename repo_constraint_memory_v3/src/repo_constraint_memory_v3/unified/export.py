"""Exclusive audit and delivery-safe exports for a unified bundle."""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from repo_constraint_memory_v3.inference import (
    InferenceIndex,
    InferenceRecord,
    ProtocolManifest,
    make_inference_index,
)
from repo_constraint_memory_v3.models import SCHEMA_VERSION
from repo_constraint_memory_v3.unified.consumer import (
    ConsumerArtifacts,
    ConsumerProfile,
    build_consumer_artifacts,
    claim_path_anchor,
    path_anchor,
)
from repo_constraint_memory_v3.unified.ids import artifact_digest
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CanonicalClaim,
    ChannelAssessment,
    CurrentEvidenceBundle,
    ImmutableArtifactRef,
    MemberDisposition,
    RefinementRecord,
    TemporalRelationMatrix,
    UnifiedBundle,
    UnifiedCaseFinding,
    UnifiedCompleteness,
    UnifiedFunnel,
    UnifiedIndex,
)
from repo_constraint_memory_v3.unified.publication import (
    _PublicationPreflight,
    _bounded_tree_paths,
    consume_initial_empty_directory,
    directory_tree_fingerprint,
    rename_directory_noreplace,
)
from repo_constraint_memory_v3.unified.upstream import UnifiedUpstreamInputs


UNIFIED_EXPORT_VERSION = "unified-export-3.2"
_SAFE_FILE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_FILE_SYNC_WORKERS = 8
# Linux permits fsync on read-only descriptors, including chmod'ed consumers.
# Keep the original writable-descriptor barriers on other platforms (Windows
# FlushFileBuffers requires write access). No runtime error is retried/fallbacked.
_DEFER_FILE_SYNC = sys.platform.startswith("linux")


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def canonical_json_text(value: object) -> str:
    """Return deterministic UTF-8 JSON text with a final newline."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def _safe_file_token(value: str, *, label: str) -> str:
    if value in {".", ".."} or _SAFE_FILE_TOKEN.fullmatch(value) is None:
        raise ValueError(f"unsafe {label} for generated artifact: {value!r}")
    return value


class _BoundedTreeWriter:
    def __init__(self, root: Path, *, max_bytes: int) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self.total_bytes = 0
        self.relative_paths: list[str] = []

    def sync_files(self) -> None:
        """Sync the complete private tree before it can be published.

        Deferring these barriers lets the filesystem batch dirty writes instead
        of forcing a journal commit between successive small-file creations.
        Every file still receives fsync. Fixed waves bound both outstanding
        tasks and descriptors; they do not use model inference concurrency.
        This does not add a directory/rename crash-durability guarantee.
        """
        if not _DEFER_FILE_SYNC:
            return

        def sync_one(relative: str) -> None:
            path = self.root.joinpath(*self._parts(relative))
            expected = path.lstat()
            if not stat.S_ISREG(expected.st_mode):
                raise ValueError(f"generated artifact is not a regular file: {relative}")
            fd = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            try:
                opened = os.fstat(fd)
                if not stat.S_ISREG(opened.st_mode) or (
                    opened.st_dev, opened.st_ino
                ) != (expected.st_dev, expected.st_ino):
                    raise ValueError(f"generated artifact changed during sync: {relative}")
                os.fsync(fd)
            finally:
                os.close(fd)

        with ThreadPoolExecutor(max_workers=_FILE_SYNC_WORKERS) as pool:
            for offset in range(0, len(self.relative_paths), _FILE_SYNC_WORKERS):
                # Drain this wave before submitting another. Executor shutdown
                # also joins in-flight syncs before failure cleanup removes staging.
                list(pool.map(
                    sync_one,
                    self.relative_paths[offset:offset + _FILE_SYNC_WORKERS],
                ))

    @staticmethod
    def _parts(relative: str) -> tuple[str, ...]:
        if "\\" in relative or "\x00" in relative:
            raise ValueError(f"unsafe generated relative path: {relative!r}")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(
            part in {"", ".", ".."} for part in pure.parts
        ):
            raise ValueError(f"unsafe generated relative path: {relative!r}")
        return pure.parts

    def directory(self, relative: str) -> Path:
        path = self.root.joinpath(*self._parts(relative))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_text(
        self,
        relative: str,
        text: str,
        *,
        read_only: bool = False,
    ) -> Path:
        data = text.encode("utf-8")
        prospective = self.total_bytes + len(data)
        if prospective > self.max_bytes:
            raise ValueError(
                "unified export exceeds max_unified_artifact_bytes: "
                f"{prospective}>{self.max_bytes}"
            )
        path = self.root.joinpath(*self._parts(relative))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise ValueError(f"refusing to overwrite generated artifact: {relative}")
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            if not _DEFER_FILE_SYNC:
                os.fsync(handle.fileno())
        if read_only:
            path.chmod(0o444)
        self.total_bytes = prospective
        self.relative_paths.append(relative)
        return path

    def write_json(
        self,
        relative: str,
        value: object,
        *,
        read_only: bool = False,
    ) -> Path:
        # ``json.dumps(..., indent=2)`` first retains every encoder fragment
        # and then joins them into one Unicode string.  One non-ASCII scalar
        # can make that whole string use a wide internal representation; the
        # large replay envelope consequently used far more memory than its
        # UTF-8 artifact.  ``iterencode`` emits exactly the same JSON chunks,
        # while this writer accounts and persists their UTF-8 bytes directly.
        # The model dump remains necessary for the existing sorted-key byte
        # contract, but neither a whole JSON string nor a second whole bytes
        # object is retained.
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        path = self.root.joinpath(*self._parts(relative))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise ValueError(f"refusing to overwrite generated artifact: {relative}")
        encoder = json.JSONEncoder(
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        written = 0
        created = False
        try:
            with path.open("xb") as handle:
                created = True
                for chunk in encoder.iterencode(value):
                    data = chunk.encode("utf-8")
                    prospective = self.total_bytes + written + len(data)
                    if prospective > self.max_bytes:
                        raise ValueError(
                            "unified export exceeds max_unified_artifact_bytes: "
                            f"{prospective}>{self.max_bytes}"
                        )
                    handle.write(data)
                    written += len(data)
                prospective = self.total_bytes + written + 1
                if prospective > self.max_bytes:
                    raise ValueError(
                        "unified export exceeds max_unified_artifact_bytes: "
                        f"{prospective}>{self.max_bytes}"
                    )
                handle.write(b"\n")
                written += 1
                handle.flush()
                if not _DEFER_FILE_SYNC:
                    os.fsync(handle.fileno())
        except Exception:
            if created:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise
        if read_only:
            path.chmod(0o444)
        self.total_bytes += written
        self.relative_paths.append(relative)
        return path

    def write_bytes(self, relative: str, value: bytes) -> Path:
        # Channel bytes are UTF-8 diagnostics, as previously enforced by the
        # upstream import boundary. Preserve their exact bytes in staging.
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"channel artifact is not UTF-8: {relative}") from error
        return self.write_text(relative, text)

    def copy_utf8_file(
        self,
        relative: str,
        source: Path,
        expected: os.stat_result,
    ) -> Path:
        """Copy one identity-pinned regular UTF-8 file with bounded memory."""

        prospective = self.total_bytes + expected.st_size
        if prospective > self.max_bytes:
            raise ValueError(
                f"upstream artifact exceeds remaining byte budget: {relative}"
            )
        path = self.root.joinpath(*self._parts(relative))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise ValueError(f"refusing to overwrite generated artifact: {relative}")
        source_fd: int | None = None
        created = False
        observed = 0
        try:
            source_fd = os.open(
                source,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(source_fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (expected.st_dev, expected.st_ino)
                or opened.st_size != expected.st_size
            ):
                raise ValueError(
                    f"upstream artifact changed during safe open: {relative}"
                )
            decoder = codecs.getincrementaldecoder("utf-8")("strict")
            with path.open("xb") as handle:
                created = True
                while True:
                    chunk = os.read(source_fd, 1_048_576)
                    if not chunk:
                        break
                    observed += len(chunk)
                    if observed > expected.st_size:
                        raise ValueError(
                            f"upstream artifact changed while being read: {relative}"
                        )
                    decoder.decode(chunk, final=False)
                    handle.write(chunk)
                decoder.decode(b"", final=True)
                after = os.fstat(source_fd)
                if (
                    (after.st_dev, after.st_ino)
                    != (expected.st_dev, expected.st_ino)
                    or after.st_size != expected.st_size
                    or after.st_mtime_ns != expected.st_mtime_ns
                    or observed != expected.st_size
                ):
                    raise ValueError(
                        f"upstream artifact changed while being read: {relative}"
                    )
                handle.flush()
                if not _DEFER_FILE_SYNC:
                    os.fsync(handle.fileno())
        except (OSError, UnicodeDecodeError) as error:
            if created:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise ValueError(
                f"upstream artifact is not safely readable UTF-8: {relative}"
            ) from error
        except Exception:
            if created:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise
        finally:
            if source_fd is not None:
                os.close(source_fd)
        self.total_bytes = prospective
        self.relative_paths.append(relative)
        return path


def _validate_bundle_for_export(bundle: UnifiedBundle) -> UnifiedBundle:
    # Reparse every persisted artifact so model_construct or in-memory mutation
    # cannot bypass its own digest and invariant validators at the writer edge.
    # Validate one top-level artifact at a time and discard the temporary model;
    # a whole-bundle dump/reparse duplicates the complete current-evidence graph
    # and keeps that duplicate alive throughout export.  The final aggregate
    # validation below reuses the now-checked instances while rerunning every
    # cross-artifact reference, algebra, and aggregate-digest check.
    singular = (
        (UnifiedIndex, bundle.index),
        (UnifiedFunnel, bundle.funnel),
        (UnifiedCompleteness, bundle.completeness),
    )
    collections = (
        (ChannelAssessment, bundle.assessments),
        (CandidateCase, bundle.cases),
        (CurrentEvidenceBundle, bundle.current_evidence),
        (ImmutableArtifactRef, bundle.git_bundle_refs),
        (TemporalRelationMatrix, bundle.temporal_matrices),
        (UnifiedCaseFinding, bundle.findings),
        (RefinementRecord, bundle.refinements),
        (CanonicalClaim, bundle.canonical_claims),
        (MemberDisposition, bundle.member_dispositions),
    )
    for model, value in singular:
        if not isinstance(value, BaseModel):
            raise ValueError("unified bundle contains a non-model artifact")
        model.model_validate(value.model_dump(mode="json"))
    for model, values in collections:
        for value in values:
            if not isinstance(value, BaseModel):
                raise ValueError("unified bundle contains a non-model artifact")
            model.model_validate(value.model_dump(mode="json"))
    validated = UnifiedBundle.model_validate(
        {
            field_name: getattr(bundle, field_name)
            for field_name in UnifiedBundle.model_fields
        }
    )
    if (
        not validated.completeness.structurally_valid
        or validated.completeness.package_status == "invalid"
    ):
        raise ValueError("structurally invalid unified bundle is not exportable")
    snapshot = validated.index.snapshot_digest
    profile = validated.index.profile_digest
    case_ids = {item.case_id for item in validated.cases}
    claim_ids = {item.claim_id for item in validated.canonical_claims}
    if len(claim_ids) != len(validated.canonical_claims):
        raise ValueError("canonical claims contain duplicate identities")
    for assessment in validated.assessments:
        if (
            assessment.snapshot_digest != snapshot
            or assessment.profile_digest != profile
        ):
            raise ValueError("assessment export snapshot/profile mismatch")
    for case in validated.cases:
        if case.snapshot_digest != snapshot or case.profile_digest != profile:
            raise ValueError("case export snapshot/profile mismatch")
    for current in validated.current_evidence:
        if (
            current.snapshot_digest != snapshot
            or current.profile_digest != profile
            or current.case_id not in case_ids
        ):
            raise ValueError("current-evidence export identity mismatch")
    for matrix in validated.temporal_matrices:
        if (
            matrix.snapshot_digest != snapshot
            or matrix.profile_digest != profile
            or matrix.case_id not in case_ids
        ):
            raise ValueError("temporal-matrix export identity mismatch")
    for finding in validated.findings:
        if (
            finding.snapshot_digest != snapshot
            or finding.profile_digest != profile
            or finding.case_id not in case_ids
        ):
            raise ValueError("finding export identity mismatch")
    for claim in validated.canonical_claims:
        if (
            claim.snapshot_digest != snapshot
            or claim.profile_digest != profile
            or claim.case_id not in case_ids
        ):
            raise ValueError("canonical claim export identity mismatch")
        for scope in claim.semantics.applies_to:
            claim_path_anchor(claim, scope)
    for disposition in validated.member_dispositions:
        if not set(disposition.canonical_claim_ids) <= claim_ids:
            raise ValueError("member disposition references a foreign claim")
    return validated


def _git_payload(
    value: object,
    *,
    stable_id: str,
) -> dict[str, object]:
    if isinstance(value, BaseModel):
        raw = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise ValueError(f"Git artifact {stable_id} is not a mapping/model")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"Git artifact {stable_id} has a non-string key")
    return raw


def _write_git_artifacts(
    writer: _BoundedTreeWriter,
    bundle: UnifiedBundle,
    git_artifacts: Mapping[str, object] | None,
) -> None:
    refs = sorted(
        bundle.git_bundle_refs,
        key=lambda item: (item.kind, item.stable_id),
    )
    ref_ids = [item.stable_id for item in refs]
    if len(ref_ids) != len(set(ref_ids)):
        raise ValueError("unified Git refs contain duplicate stable IDs")
    supplied = {} if git_artifacts is None else dict(git_artifacts)
    if not all(isinstance(key, str) for key in supplied):
        raise ValueError("Git artifact mapping keys must be strings")
    if set(supplied) != set(ref_ids):
        raise ValueError(
            "Git payload mapping differs from bundle refs; "
            f"missing={sorted(set(ref_ids) - set(supplied))}, "
            f"foreign={sorted(set(supplied) - set(ref_ids))}"
        )
    writer.write_json(
        "canonical/audit/git/refs.json",
        {
            "schema_version": SCHEMA_VERSION,
            "snapshot_digest": bundle.index.snapshot_digest,
            "refs": [item.model_dump(mode="json") for item in refs],
        },
    )
    for ref in refs:
        stable_id = _safe_file_token(ref.stable_id, label="Git artifact ID")
        if ref.kind not in {
            "git_temporal_bundle",
            "git_manifest",
            "git_change_slice",
        }:
            raise ValueError(f"unsupported Git audit ref kind: {ref.kind}")
        payload = _git_payload(supplied[ref.stable_id], stable_id=ref.stable_id)
        identity_field = {
            "git_temporal_bundle": "bundle_id",
            "git_manifest": "manifest_id",
            "git_change_slice": "item_id",
        }[ref.kind]
        if payload.get(identity_field) != ref.stable_id:
            raise ValueError(f"Git artifact identity differs from ref: {ref.stable_id}")
        if payload.get("snapshot_digest") != ref.snapshot_digest:
            raise ValueError(f"Git artifact snapshot differs from ref: {ref.stable_id}")
        if payload.get("artifact_digest") != ref.artifact_digest:
            raise ValueError(f"Git artifact digest differs from ref: {ref.stable_id}")
        if artifact_digest(payload) != ref.artifact_digest:
            raise ValueError(f"Git artifact content digest is invalid: {ref.stable_id}")
        directory = {
            "git_temporal_bundle": "bundles",
            "git_manifest": "manifests",
            "git_change_slice": "slices",
        }[ref.kind]
        writer.write_json(
            f"canonical/audit/git/{directory}/{stable_id}.json",
            payload,
        )


def make_export_inference_index(
    bundle: UnifiedBundle,
    *,
    protocol_manifest: ProtocolManifest,
    inference_records: Sequence[InferenceRecord],
    incomplete_reasons: Sequence[str],
    stage_omissions: Mapping[str, tuple[int, Mapping[str, int]]] | None = None,
) -> InferenceIndex:
    """Build the exact root inference index used by the atomic exporter."""

    if protocol_manifest.protocol_manifest_digest != bundle.index.profile_digest:
        raise ValueError("unified inference profile differs from bundle")
    ordered = sorted(inference_records, key=lambda item: item.request_id)
    request_ids = [item.request_id for item in ordered]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("unified inference records contain duplicate IDs")
    for record in ordered:
        if record.snapshot_digest != bundle.index.snapshot_digest:
            raise ValueError("unified inference record snapshot differs from bundle")
        if record.protocol_manifest_digest != bundle.index.profile_digest:
            raise ValueError("unified inference record profile differs from bundle")
    artifact_hashes: dict[str, str] = {}
    artifact_bytes = 0
    for item in ordered:
        content = canonical_json_text(item).encode("utf-8")
        artifact_hashes[item.request_id] = hashlib.sha256(content).hexdigest()
        artifact_bytes += len(content)
    index = make_inference_index(
        snapshot_digest=bundle.index.snapshot_digest,
        protocol_manifest=protocol_manifest,
        records=ordered,
        artifact_sha256_by_id=artifact_hashes,
        total_artifact_bytes=artifact_bytes,
        incomplete_reasons=sorted(set(incomplete_reasons)),
        stage_omissions=stage_omissions,
    )
    return index


def _write_inference_tree(
    writer: _BoundedTreeWriter,
    bundle: UnifiedBundle,
    *,
    protocol_manifest: ProtocolManifest | None,
    inference_records: Sequence[InferenceRecord] | None,
    incomplete_reasons: Sequence[str],
    stage_omissions: Mapping[str, tuple[int, Mapping[str, int]]] | None,
) -> None:
    if protocol_manifest is None and inference_records is None:
        return
    if protocol_manifest is None or inference_records is None:
        raise ValueError(
            "unified inference export requires both manifest and records"
        )
    ordered = sorted(inference_records, key=lambda item: item.request_id)
    index = make_export_inference_index(
        bundle,
        protocol_manifest=protocol_manifest,
        inference_records=ordered,
        incomplete_reasons=incomplete_reasons,
        stage_omissions=stage_omissions,
    )
    writer.directory("inference/requests")
    writer.write_json("inference/index.json", index)
    for record in ordered:
        writer.write_json(
            f"inference/requests/{record.request_id}.json",
            record,
        )


def _write_canonical_tree(
    writer: _BoundedTreeWriter,
    bundle: UnifiedBundle,
    *,
    consumer_artifacts: ConsumerArtifacts,
    git_artifacts: Mapping[str, object] | None,
    upstream_inputs: UnifiedUpstreamInputs | None,
) -> None:
    for relative in (
        "canonical/audit/channel-assessments",
        "canonical/audit/cases",
        "canonical/audit/current-evidence",
        "canonical/audit/git/manifests",
        "canonical/audit/git/bundles",
        "canonical/audit/git/slices",
        "canonical/audit/git/relations",
        "canonical/audit/findings",
        "canonical/audit/refinements",
    ):
        writer.directory(relative)
    writer.write_json("canonical/claims.json", consumer_artifacts.canonical_claims)
    writer.write_json(
        "canonical/completeness.json", consumer_artifacts.completeness
    )
    writer.write_json(
        "canonical/delivery_report.json", consumer_artifacts.delivery_report
    )
    writer.write_json("canonical/audit/index.json", bundle.index)
    if upstream_inputs is not None:
        if (
            upstream_inputs.snapshot_digest != bundle.index.snapshot_digest
            or upstream_inputs.profile_digest != bundle.index.profile_digest
            or upstream_inputs.snapshot_commit != bundle.index.snapshot_commit
        ):
            raise ValueError("unified upstream input binding differs from bundle")
        writer.write_json("canonical/audit/upstream-inputs.json", upstream_inputs)
    for assessment in sorted(
        bundle.assessments, key=lambda item: item.assessment_id
    ):
        stable_id = _safe_file_token(
            assessment.assessment_id, label="assessment ID"
        )
        writer.write_json(
            f"canonical/audit/channel-assessments/{stable_id}.json", assessment
        )
    for case in sorted(bundle.cases, key=lambda item: item.case_id):
        stable_id = _safe_file_token(case.case_id, label="case ID")
        writer.write_json(f"canonical/audit/cases/{stable_id}.json", case)
    for current in sorted(
        bundle.current_evidence, key=lambda item: item.bundle_id
    ):
        stable_id = _safe_file_token(
            current.bundle_id, label="current-evidence ID"
        )
        writer.write_json(
            f"canonical/audit/current-evidence/{stable_id}.json", current
        )
    _write_git_artifacts(writer, bundle, git_artifacts)
    for matrix in sorted(
        bundle.temporal_matrices,
        key=lambda item: (item.case_id, item.matrix_id),
    ):
        stable_id = _safe_file_token(matrix.matrix_id, label="temporal matrix ID")
        writer.write_json(
            f"canonical/audit/git/relations/{stable_id}.json", matrix
        )
    for finding in sorted(bundle.findings, key=lambda item: item.finding_id):
        stable_id = _safe_file_token(finding.finding_id, label="finding ID")
        writer.write_json(
            f"canonical/audit/findings/{stable_id}.json", finding
        )
    for refinement in sorted(
        bundle.refinements, key=lambda item: item.refinement_id
    ):
        stable_id = _safe_file_token(
            refinement.refinement_id, label="refinement ID"
        )
        writer.write_json(
            f"canonical/audit/refinements/{stable_id}.json", refinement
        )
    writer.write_json(
        "canonical/audit/member-dispositions.json",
        {
            "schema_version": SCHEMA_VERSION,
            "snapshot_digest": bundle.index.snapshot_digest,
            "items": [
                item.model_dump(mode="json")
                for item in sorted(
                    bundle.member_dispositions, key=lambda item: item.member_id
                )
            ],
        },
    )
    writer.write_json("canonical/audit/funnel.json", bundle.funnel)


def _write_consumer_tree(
    writer: _BoundedTreeWriter,
    artifacts: ConsumerArtifacts,
) -> None:
    for profile, plan in sorted(artifacts.views.items()):
        prefix = f"consumers/{profile}"
        for directory in sorted(plan.directories):
            writer.directory(f"{prefix}/{directory}")
        for relative, data in sorted(plan.files.items()):
            writer.write_text(
                f"{prefix}/{relative}",
                data.decode("utf-8", errors="strict"),
                read_only=True,
            )


def _copy_upstream_tree(
    writer: _BoundedTreeWriter,
    upstream_directory: Path | None,
    *,
    replace_inference: bool,
    replace_build_report: bool,
) -> None:
    if upstream_directory is None:
        return
    root = upstream_directory.resolve()
    if upstream_directory.is_symlink() or not root.is_dir():
        raise ValueError("validated upstream directory is missing or a symlink")
    replaced_roots = {
        "memory.json",
        "review.json",
        "catalog.json",
        "delivery_report.json",
        "index.md",
        "by-path",
        "by-trigger",
        "unified",
        "canonical",
        "consumers",
    }
    if replace_inference:
        replaced_roots.add("inference")
    if replace_build_report:
        replaced_roots.add("build_report.json")
    for path in _bounded_tree_paths(
        root,
        excluded_top_level=frozenset(replaced_roots),
        label="upstream tree",
    ):
        relative = path.relative_to(root).as_posix()
        if relative.split("/", 1)[0] in replaced_roots:
            continue
        try:
            before = path.lstat()
        except OSError as error:
            raise ValueError(f"cannot stat upstream artifact: {relative}") from error
        if stat.S_ISLNK(before.st_mode):
            raise ValueError(f"upstream artifact is a symlink: {relative}")
        if stat.S_ISDIR(before.st_mode):
            writer.directory(relative)
            continue
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"upstream artifact is not a regular file: {relative}")
        writer.copy_utf8_file(relative, path, before)


def _write_unified_outputs_impl(
    output_directory: Path,
    *,
    bundle: UnifiedBundle,
    max_artifact_bytes: int,
    repository_name: str | None = None,
    repository_root: Path | None = None,
    consumer_profiles: Sequence[ConsumerProfile] = ("accepted_only",),
    git_artifacts: Mapping[str, object] | None = None,
    upstream_directory: Path | None = None,
    channel_writer: Callable[[_BoundedTreeWriter], object] | None = None,
    upstream_inputs: UnifiedUpstreamInputs | None = None,
    protocol_manifest: ProtocolManifest | None = None,
    inference_records: Sequence[InferenceRecord] | None = None,
    inference_incomplete_reasons: Sequence[str] = (),
    inference_stage_omissions: (
        Mapping[str, tuple[int, Mapping[str, int]]] | None
    ) = None,
    build_report: Mapping[str, object] | None = None,
    prepublication_check: Callable[[], None] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Render and atomically publish a structurally checked tree.

    ``git_artifacts`` is keyed by the immutable Git manifest/slice stable IDs in
    ``bundle.git_bundle_refs``.  Every ref must have one digest-matching payload;
    extras and omissions are rejected.  Full offline replay is deliberately an
    independent validator responsibility and is never invoked by this writer.
    """

    started = time.monotonic()

    def progress(phase: str) -> None:
        if progress_callback is not None:
            progress_callback(
                f"publication phase={phase} elapsed={time.monotonic() - started:.3f}s"
            )

    progress("prepare")
    if (
        isinstance(max_artifact_bytes, bool)
        or not isinstance(max_artifact_bytes, int)
        or max_artifact_bytes <= 0
    ):
        raise ValueError("max_artifact_bytes must be a positive integer")
    if channel_writer is not None and upstream_directory is not None:
        raise ValueError("direct channel staging and upstream import are exclusive")
    validated = _validate_bundle_for_export(bundle)
    resolved_repository_name = repository_name
    if resolved_repository_name is None and repository_root is not None:
        resolved_repository_name = repository_root.resolve().name
    if not resolved_repository_name:
        resolved_repository_name = "repository"
    consumer_artifacts = build_consumer_artifacts(
        bundle=validated,
        repository_name=resolved_repository_name,
        profiles=consumer_profiles,
        git_artifacts=git_artifacts,
    )
    effective_build_report: dict[str, object] | None = None
    if build_report is not None:
        effective_build_report = dict(build_report)
        effective_build_report["consumer_surface"] = (
            consumer_artifacts.report_metadata()
        )
        if isinstance(build_report, dict):
            # Keep the caller-visible BuildResult report byte-equivalent to the
            # report written into the staged run.  Mapping-only inputs remain
            # immutable and are copied above.
            build_report.clear()
            build_report.update(effective_build_report)
    destination = output_directory.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(
            f"output path already exists; refusing to overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.publish.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError(f"another unified export is publishing: {destination}") from error
    os.close(lock_fd)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    published = False
    writer = _BoundedTreeWriter(staging, max_bytes=max_artifact_bytes)
    try:
        progress("write")
        if channel_writer is not None:
            channel_writer(writer)
        replace_inference = (
            protocol_manifest is not None or inference_records is not None
        )
        _copy_upstream_tree(
            writer,
            upstream_directory,
            replace_inference=replace_inference,
            replace_build_report=effective_build_report is not None,
        )
        _write_inference_tree(
            writer,
            validated,
            protocol_manifest=protocol_manifest,
            inference_records=inference_records,
            incomplete_reasons=inference_incomplete_reasons,
            stage_omissions=inference_stage_omissions,
        )
        if effective_build_report is not None:
            writer.write_json("build_report.json", effective_build_report)
        _write_canonical_tree(
            writer,
            validated,
            consumer_artifacts=consumer_artifacts,
            git_artifacts=git_artifacts,
            upstream_inputs=upstream_inputs,
        )
        _write_consumer_tree(writer, consumer_artifacts)
        progress(f"sync files={len(writer.relative_paths)} bytes={writer.total_bytes}")
        writer.sync_files()
        progress("check")
        staged_fingerprint = directory_tree_fingerprint(
            staging,
            max_bytes=max_artifact_bytes,
        )
        # No fallible observer callback runs after the final input/tree checks
        # or after rename. The caller's terminal result reports completion.
        progress("publish")
        if prepublication_check is not None:
            # Keep mutable external-input checks adjacent to the atomic rename.
            # The writer cannot infer the build's frozen discovery/upstream
            # identities, so production callers provide the narrow replay-free
            # check appropriate to their inputs.
            prepublication_check()
        if directory_tree_fingerprint(
            staging,
            max_bytes=max_artifact_bytes,
        ) != staged_fingerprint:
            raise ValueError("unified staging tree changed before publication")
        if repository_root is not None and _paths_overlap(
            destination, repository_root
        ):
            raise ValueError("unified output overlaps the target repository")
        if upstream_directory is not None and _paths_overlap(
            destination, upstream_directory
        ):
            raise ValueError("unified output overlaps the upstream run")
        if destination.exists() or destination.is_symlink():
            raise ValueError(
                f"output path appeared during export; refusing: {destination}"
            )
        rename_directory_noreplace(staging, destination)
        published = True
        return {
            relative: destination.joinpath(*PurePosixPath(relative).parts)
            for relative in sorted(writer.relative_paths)
        }
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def write_unified_outputs(
    output_directory: Path,
    *,
    bundle: UnifiedBundle,
    max_artifact_bytes: int,
    repository_root: Path,
    repository_name: str | None = None,
    consumer_profiles: Sequence[ConsumerProfile] = ("accepted_only",),
    git_artifacts: Mapping[str, object] | None = None,
    upstream_directory: Path | None = None,
    channel_writer: Callable[[_BoundedTreeWriter], object] | None = None,
    upstream_inputs: UnifiedUpstreamInputs | None = None,
    protocol_manifest: ProtocolManifest | None = None,
    inference_records: Sequence[InferenceRecord] | None = None,
    inference_incomplete_reasons: Sequence[str] = (),
    inference_stage_omissions: (
        Mapping[str, tuple[int, Mapping[str, int]]] | None
    ) = None,
    build_report: Mapping[str, object] | None = None,
    _destination_preflight: _PublicationPreflight | None = None,
    _prepublication_check: Callable[[], None] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Atomically publish a locally checked complete knowledge-package tree.

    The caller-authorized repository is mandatory at this production boundary;
    it protects repository/output separation.  Full offline replay is available
    only through ``validate``/``validate-unified`` and does not gate publication.
    """

    if repository_root is None:  # Defensive for untyped callers.
        raise ValueError("repository_root is required for protected publication")
    if output_directory.is_symlink():
        raise ValueError("unified output is a symlink")
    destination = output_directory.resolve(strict=False)
    repository = repository_root.resolve()
    if _paths_overlap(destination, repository):
        raise ValueError("unified output overlaps the target repository")
    if upstream_directory is not None:
        upstream = upstream_directory.resolve()
        if _paths_overlap(destination, upstream):
            raise ValueError("unified output overlaps the upstream run")
    if _destination_preflight is None:
        _destination_preflight = consume_initial_empty_directory(
            destination
        )
    elif (
        not isinstance(_destination_preflight, _PublicationPreflight)
        or _destination_preflight.destination
        != destination
    ):
        raise ValueError("unified output preflight does not match destination")
    if destination.exists() or destination.is_symlink():
        raise ValueError("unified output appeared after initial preflight")
    return _write_unified_outputs_impl(
        destination,
        bundle=bundle,
        max_artifact_bytes=max_artifact_bytes,
        repository_name=repository_name,
        repository_root=repository_root,
        consumer_profiles=consumer_profiles,
        git_artifacts=git_artifacts,
        upstream_directory=upstream_directory,
        channel_writer=channel_writer,
        upstream_inputs=upstream_inputs,
        protocol_manifest=protocol_manifest,
        inference_records=inference_records,
        inference_incomplete_reasons=inference_incomplete_reasons,
        inference_stage_omissions=inference_stage_omissions,
        build_report=build_report,
        prepublication_check=_prepublication_check,
        progress_callback=progress_callback,
    )


def _write_unvalidated_outputs_for_test(
    output_directory: Path,
    **kwargs: object,
) -> dict[str, Path]:
    """Materialize deliberately incomplete validator fixtures without bindings.

    This private helper exists only because validator unit tests need malformed
    trees without a caller-authorized repository.  Production builders must use
    :func:`write_unified_outputs`.
    """

    if "repository_root" in kwargs:
        raise ValueError("unvalidated test fixture cannot accept repository_root")
    return _write_unified_outputs_impl(
        output_directory,
        repository_root=None,
        **kwargs,  # type: ignore[arg-type]
    )


__all__ = [
    "UNIFIED_EXPORT_VERSION",
    "canonical_json_text",
    "make_export_inference_index",
    "path_anchor",
    "write_unified_outputs",
]
