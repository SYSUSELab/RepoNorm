"""Channel-neutral proposition envelopes and snapshot-bound typed references."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Mapping, Sequence

from pydantic import Field, field_validator, model_validator

from repo_constraint_memory_v3.models import (
    CandidateClaim,
    ClaimKind,
    Evidence,
    EvidenceUnit,
    NonEmptyString,
    NormativeStrength,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
)
from repo_constraint_memory_v3.observations.canonical import (
    model_payload_digest,
    payload_digest,
)

PROPOSITION_CONTRACT_VERSION = "common-proposition-1.1"
PropositionId = Annotated[str, Field(pattern=r"^PROP-[0-9a-f]{24}$")]


class TypedObjectRef(StrictModel):
    kind: Literal[
        "explicit_unit",
        "explicit_span",
        "explicit_evidence",
        "observation_file",
        "observation_entity",
        "observation_site",
        "observation_relation",
        "observation_clone",
        "semantic_unit",
        "semantic_packet",
        "semantic_observation",
        "norm_hypothesis",
        "inference_request",
        "population_spec",
        "population_result",
        "falsification_result",
    ]
    stable_id: NonEmptyString
    snapshot_digest: SnapshotDigest


class PropositionEnvelope(StrictModel):
    contract_version: Literal["common-proposition-1.1"]
    snapshot_digest: SnapshotDigest
    proposition_id: PropositionId
    rule: Annotated[str, Field(min_length=1, max_length=500)]
    category: Annotated[str, Field(min_length=1, max_length=120)] | None
    proposed_claim_kind: ClaimKind
    proposed_normative_strength: NormativeStrength
    applies_to: Annotated[list[NonEmptyString], Field(min_length=1)]
    condition: str | None
    origin_channels: Annotated[
        list[Literal["explicit", "implicit_current"]],
        Field(min_length=1),
    ]
    evidence_routes: Annotated[
        list[
            Literal[
                "explicit_typed_evidence",
                "current_structural_population",
                "current_semantic_population",
            ]
        ],
        Field(min_length=1),
    ]
    evidence_refs: list[TypedObjectRef]
    inference_refs: list[TypedObjectRef]
    population_refs: list[TypedObjectRef]
    limitation_refs: list[TypedObjectRef]
    completeness_refs: list[NonEmptyString]
    provenance: list[NonEmptyString]
    artifact_digest: Sha256Hex

    @field_validator("category")
    @classmethod
    def bounded_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or "\n" in stripped or "\r" in stripped:
            raise ValueError("category must be a bounded single line")
        return stripped

    @field_validator("condition")
    @classmethod
    def non_blank_condition(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("condition must be null or non-blank")
        return value

    @model_validator(mode="after")
    def canonical_and_digest(self) -> PropositionEnvelope:
        if len(set(self.origin_channels)) != len(self.origin_channels):
            raise ValueError("proposition origin channels must be unique")
        all_refs = (
            self.evidence_refs
            + self.inference_refs
            + self.population_refs
            + self.limitation_refs
        )
        if any(item.snapshot_digest != self.snapshot_digest for item in all_refs):
            raise ValueError("proposition reference snapshot does not match")
        if self.artifact_digest != model_payload_digest(
            self, digest_field="artifact_digest"
        ):
            raise ValueError("incorrect proposition artifact digest")
        return self


def _stable_id(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"PROP-{digest[:24]}"


def normalize_explicit_proposition(
    *,
    snapshot_digest: str,
    unit: EvidenceUnit,
    candidate: CandidateClaim,
    derived_kind: ClaimKind,
    derived_strength: NormativeStrength,
    canonical_scope: Sequence[str],
    canonical_condition: str,
    grounded_evidence: Sequence[Evidence],
    inference_request_id: str | None = None,
    category: str | None = None,
) -> PropositionEnvelope:
    """Normalize grounded explicit evidence without changing explicit policy."""

    evidence_refs = [
        TypedObjectRef(
            kind="explicit_evidence",
            stable_id=payload_digest(
                {
                    "path": item.path,
                    "span_ids": item.span_ids,
                    "quote_sha256": item.quote_sha256,
                }
            ),
            snapshot_digest=snapshot_digest,
        )
        for item in grounded_evidence
    ]
    inference_refs = (
        [
            TypedObjectRef(
                kind="inference_request",
                stable_id=inference_request_id,
                snapshot_digest=snapshot_digest,
            )
        ]
        if inference_request_id is not None
        else []
    )
    identity = {
        "contract_version": PROPOSITION_CONTRACT_VERSION,
        "snapshot_digest": snapshot_digest,
        "rule": candidate.rule,
        "proposed_claim_kind": derived_kind,
        "proposed_normative_strength": derived_strength,
        "applies_to": list(canonical_scope),
        "condition": canonical_condition or None,
        "origin_channels": ["explicit"],
        "evidence_routes": ["explicit_typed_evidence"],
        "evidence_refs": [
            item.model_dump(mode="json") for item in evidence_refs
        ],
    }
    payload: dict[str, object] = {
        "contract_version": PROPOSITION_CONTRACT_VERSION,
        "snapshot_digest": snapshot_digest,
        "proposition_id": _stable_id(identity),
        "rule": candidate.rule,
        "category": category,
        "proposed_claim_kind": derived_kind,
        "proposed_normative_strength": derived_strength,
        "applies_to": list(canonical_scope),
        "condition": canonical_condition or None,
        "origin_channels": ["explicit"],
        "evidence_routes": ["explicit_typed_evidence"],
        "evidence_refs": [
            item.model_dump(mode="json") for item in evidence_refs
        ],
        "inference_refs": [
            item.model_dump(mode="json") for item in inference_refs
        ],
        "population_refs": [],
        "limitation_refs": [],
        "completeness_refs": ["explicit_grounding_complete"],
        "provenance": [f"unit:{unit.unit_id}", f"path:{unit.path}"],
    }
    payload["artifact_digest"] = payload_digest(payload)
    return PropositionEnvelope.model_validate(payload)


def validate_explicit_proposition_route(proposition: PropositionEnvelope) -> None:
    """Validate the explicit channel's local proposition projection.

    Canonical acceptance is deliberately outside this substrate and belongs to
    the schema-4 unified policy.
    """

    if proposition.origin_channels != ["explicit"]:
        raise ValueError("explicit proposition has a non-explicit origin")
    if proposition.evidence_routes != ["explicit_typed_evidence"]:
        raise ValueError(
            "explicit proposition requires the explicit typed-evidence route"
        )
    if proposition.population_refs:
        raise ValueError("explicit proposition cannot carry population results")


def validate_proposition_references(
    proposition: PropositionEnvelope,
    manifests: Mapping[str, set[str]],
) -> None:
    """Resolve every typed reference in its declared snapshot manifest."""

    for reference in (
        proposition.evidence_refs
        + proposition.inference_refs
        + proposition.population_refs
        + proposition.limitation_refs
    ):
        if reference.stable_id not in manifests.get(reference.kind, set()):
            raise ValueError(
                f"unresolved {reference.kind} reference: {reference.stable_id}"
            )
