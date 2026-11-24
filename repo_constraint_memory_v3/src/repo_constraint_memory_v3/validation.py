"""Offline replay helpers for accepted claims and typed exact evidence."""

from __future__ import annotations

import hashlib
import stat
import unicodedata
from pathlib import Path, PurePosixPath

from repo_constraint_memory_v3.adjudication import (
    claim_id_for_proposition,
    classify_execution_risk,
    delivery_policy_for,
    semantic_core,
)
from repo_constraint_memory_v3.discovery import (
    SourceFile,
    discover_repository,
    sha256_bytes,
    valid_relative_posix_path,
)
from repo_constraint_memory_v3.models import (
    CandidateEvidenceRange,
    Constraint,
    DiscoveryArtifact,
    EvidenceUnit,
    MemoryFile,
    SCHEMA_VERSION,
)
from repo_constraint_memory_v3.source_adapters import (
    AdapterError,
    ground_evidence_range,
)
from repo_constraint_memory_v3.spans import physical_line_spans


def _normal(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", value).casefold().split()
    )


def _claim_payload(claim: Constraint) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "claim_kind": claim.claim_kind,
        "normative_strength": claim.normative_strength,
        "applies_to": sorted(set(claim.applies_to)),
        "condition": _normal(claim.condition or ""),
    }
    payload["semantic_core"] = semantic_core(
        claim.rule,
        claim.claim_kind,
    )
    return payload


def _source_for_evidence(
    repository_root: Path,
    evidence,
) -> tuple[SourceFile, bytes, str]:
    if not valid_relative_posix_path(evidence.path):
        raise ValueError("unsafe evidence path")
    absolute = repository_root.joinpath(*PurePosixPath(evidence.path).parts)
    info = absolute.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("evidence path is not a regular file")
    raw = absolute.read_bytes()
    if sha256_bytes(raw) != evidence.source_content_sha256:
        raise ValueError("source content hash changed")
    content = raw.decode("utf-8", errors="strict").replace("\r\n", "\n")
    source = SourceFile(
        path=evidence.path,
        source_kind=evidence.source_kind,
        discovery_origin=evidence.discovery_origin,
        reason="run_validation_replay",
        referenced_by=(),
        content=content,
        content_sha256=evidence.source_content_sha256,
        size_bytes=len(raw),
    )
    return source, raw, content


def validate_memory_against_repository(
    memory: MemoryFile,
    repository_root: Path,
    discovery: DiscoveryArtifact,
) -> list[str]:
    """Replay snapshot, claim IDs, exact spans, adapters, risk, and policy."""

    errors: list[str] = []
    root = repository_root.resolve()
    try:
        current = discover_repository(
            root,
            discovery.limits,
            repository_name=discovery.repository.name,
        )
    except Exception as error:  # localize discovery errors for validate
        return [f"cannot replay repository discovery: {error}"]
    if current.repository.snapshot_digest != memory.repository.snapshot_digest:
        errors.append("repository snapshot digest no longer matches memory")
    if current.model_dump(mode="json") != discovery.model_dump(mode="json"):
        errors.append("saved discovery does not match deterministic rediscovery")
    manifest_paths = {item.path for item in discovery.manifest.files}
    for claim_index, claim in enumerate(memory.claims):
        prefix = f"claims[{claim_index}]"
        expected_id = claim_id_for_proposition(_claim_payload(claim))
        if expected_id != claim.id:
            errors.append(f"{prefix}: claim ID does not match proposition")
        replayed_evidence = []
        for evidence_index, evidence in enumerate(claim.evidence):
            evidence_prefix = f"{prefix}.evidence[{evidence_index}]"
            if evidence.path not in manifest_paths:
                errors.append(f"{evidence_prefix}: path is absent from manifest")
                continue
            try:
                source, _, content = _source_for_evidence(root, evidence)
            except (OSError, UnicodeDecodeError, ValueError) as error:
                errors.append(f"{evidence_prefix}: {error}")
                continue
            lines = content.splitlines(keepends=True)
            if evidence.line_end > len(lines):
                errors.append(f"{evidence_prefix}: line range exceeds source")
                continue
            quote = "".join(
                lines[evidence.line_start - 1 : evidence.line_end]
            )
            if quote != evidence.quote:
                errors.append(f"{evidence_prefix}: quote does not match source")
            if hashlib.sha256(quote.encode("utf-8")).hexdigest() != evidence.quote_sha256:
                errors.append(f"{evidence_prefix}: quote hash mismatch")
            spans = physical_line_spans(
                quote,
                source_content_sha256=evidence.source_content_sha256,
                line_start=evidence.line_start,
            )
            if [item.span_id for item in spans] != evidence.span_ids:
                errors.append(f"{evidence_prefix}: span IDs do not replay")
                continue
            unit_hash = hashlib.sha256(
                (
                    evidence.path
                    + "\0"
                    + str(evidence.line_start)
                    + "\0"
                    + evidence.quote_sha256
                ).encode("utf-8")
            ).hexdigest()
            unit = EvidenceUnit(
                unit_id=f"UNIT-{unit_hash[:24]}",
                path=evidence.path,
                source_kind=evidence.source_kind,
                discovery_origin=evidence.discovery_origin,
                line_start=evidence.line_start,
                line_end=evidence.line_end,
                default_scope=claim.applies_to,
                content=quote,
                content_sha256=evidence.quote_sha256,
                source_content_sha256=evidence.source_content_sha256,
                spans=spans,
            )
            selection = CandidateEvidenceRange(
                unit_id=unit.unit_id,
                span_start_id=spans[0].span_id,
                span_end_id=spans[-1].span_id,
            )
            try:
                grounded = ground_evidence_range(
                    selection,
                    units_by_id={unit.unit_id: unit},
                    sources_by_path={source.path: source},
                    allowed_unit_ids={unit.unit_id},
                )
            except AdapterError as error:
                errors.append(f"{evidence_prefix}: typed adapter replay failed: {error}")
                continue
            if grounded.evidence != evidence:
                errors.append(
                    f"{evidence_prefix}: typed evidence semantics do not replay"
                )
            replayed_evidence.append(evidence)
        if replayed_evidence:
            risk = classify_execution_risk(
                claim.claim_kind,
                claim.rule,
                replayed_evidence,
            )
            if risk != claim.execution_risk:
                errors.append(f"{prefix}: execution risk does not replay")
            policy = delivery_policy_for(
                claim.condition,
                risk,
            )
            if policy != claim.delivery_policy:
                errors.append(f"{prefix}: delivery policy does not replay")
    return errors
