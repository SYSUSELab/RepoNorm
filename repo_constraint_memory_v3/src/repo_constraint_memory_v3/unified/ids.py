"""Stable identifiers and canonical digests for unified-layer artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


UNIFIED_ID_CONTRACT_VERSION = "unified-id-1.0"


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    return value


def canonical_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value without presentation-dependent bytes."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_digest(value: object) -> str:
    """Return the full SHA-256 digest of a canonical payload."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def artifact_digest(
    value: BaseModel | Mapping[str, object],
    *,
    digest_field: str = "artifact_digest",
) -> str:
    """Digest a model or mapping while excluding its self-referential field."""

    if isinstance(value, BaseModel):
        payload: dict[str, object] = value.model_dump(
            mode="json", exclude={digest_field}
        )
    else:
        payload = dict(value)
        payload.pop(digest_field, None)
    return content_digest(payload)


def seal_payload(
    payload: Mapping[str, object],
    *,
    digest_field: str = "artifact_digest",
) -> dict[str, object]:
    """Copy a payload and add its canonical artifact digest."""

    sealed = dict(payload)
    sealed.pop(digest_field, None)
    sealed[digest_field] = content_digest(sealed)
    return sealed


def stable_id(prefix: str, payload: object) -> str:
    """Return a bounded, content-derived identifier under the unified contract."""

    if not prefix or not prefix.isascii() or not prefix.replace("_", "").isalnum():
        raise ValueError("stable ID prefix must be a non-empty ASCII token")
    return (
        f"{prefix.upper()}-"
        f"{content_digest([UNIFIED_ID_CONTRACT_VERSION, payload])[:24]}"
    )


def make_assessment_id(
    snapshot_digest: str,
    profile_digest: str,
    channel: str,
    candidate_root_id: str,
    candidate_version: int,
) -> str:
    return stable_id(
        "ASSESS",
        [
            snapshot_digest,
            profile_digest,
            channel,
            candidate_root_id,
            candidate_version,
        ],
    )


def make_member_id(snapshot_digest: str, assessment_id: str) -> str:
    return stable_id("MEMBER", [snapshot_digest, assessment_id])


def make_case_id(snapshot_digest: str, member_ids: Iterable[str]) -> str:
    return stable_id("CASE", [snapshot_digest, sorted(member_ids)])


def make_evidence_origin_id(
    snapshot_digest: str,
    origin_kind: str,
    path: str,
    span: object | None,
    source_content_digest: str,
) -> str:
    return stable_id(
        "EVORIGIN",
        [snapshot_digest, origin_kind, path, span, source_content_digest],
    )


def make_evidence_id(
    origin_id: str,
    role: str,
    member_ids: Iterable[str],
    artifact_id: str,
) -> str:
    return stable_id(
        "EVIDENCE",
        [origin_id, role, sorted(member_ids), artifact_id],
    )


def make_current_evidence_bundle_id(
    snapshot_digest: str,
    profile_digest: str,
    case_id: str,
) -> str:
    return stable_id("CURREV", [snapshot_digest, profile_digest, case_id])


def make_temporal_matrix_id(
    snapshot_digest: str,
    profile_digest: str,
    case_id: str,
    item_ids: Iterable[str],
    member_ids: Iterable[str],
) -> str:
    return stable_id(
        "TEMPMATRIX",
        [
            snapshot_digest,
            profile_digest,
            case_id,
            sorted(item_ids),
            sorted(member_ids),
        ],
    )


def make_finding_id(
    snapshot_digest: str,
    profile_digest: str,
    case_id: str,
    verification_round: int,
) -> str:
    return stable_id(
        "FINDING",
        [snapshot_digest, profile_digest, case_id, verification_round],
    )


def make_refinement_id(
    snapshot_digest: str,
    case_id: str,
    parent_member_id: str,
    child_version_ids: Iterable[str],
) -> str:
    return stable_id(
        "UREFINE",
        [snapshot_digest, case_id, parent_member_id, sorted(child_version_ids)],
    )


def make_canonical_claim_id(snapshot_digest: str, semantics: object) -> str:
    return stable_id("CLAIM", [snapshot_digest, semantics])


def make_bundle_id(snapshot_digest: str, profile_digest: str) -> str:
    return stable_id("UNIFIED", [snapshot_digest, profile_digest])


__all__ = [
    "UNIFIED_ID_CONTRACT_VERSION",
    "artifact_digest",
    "canonical_bytes",
    "content_digest",
    "make_assessment_id",
    "make_bundle_id",
    "make_canonical_claim_id",
    "make_case_id",
    "make_current_evidence_bundle_id",
    "make_evidence_id",
    "make_evidence_origin_id",
    "make_finding_id",
    "make_member_id",
    "make_refinement_id",
    "make_temporal_matrix_id",
    "seal_payload",
    "stable_id",
]
