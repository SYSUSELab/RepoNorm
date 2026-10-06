"""Replay-safe upstream inputs for the unified evidence layer.

This envelope is deliberately narrower than a full build result.  It contains
only immutable channel artifacts consumed by :func:`run_unified_layer`; model
responses, provider telemetry, inference records, settings, and credentials are
not part of the contract.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from repo_constraint_memory_v3.adjudication import CandidateDecision
from repo_constraint_memory_v3.explicit_verification import (
    ExplicitVerificationResult,
    explicit_candidate_root_id,
)
from repo_constraint_memory_v3.models import (
    AcceptanceBasis,
    CandidateClaim,
    ClaimKind,
    Evidence,
    EvidenceRelation,
    LocalOutcome,
    NonEmptyString,
    NormativeStrength,
    OutcomeStage,
    SCHEMA_VERSION,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
    UnitId,
)
from repo_constraint_memory_v3.observations.engine import ObservationBundle
from repo_constraint_memory_v3.observations.models import PythonFileShard
from repo_constraint_memory_v3.propositions import PropositionEnvelope
from repo_constraint_memory_v3.semantic.models import (
    FalsificationResult,
    NormHypothesis,
    PopulationArtifact,
    PopulationMemberPacket,
    RefinementRecord,
    SemanticObservation,
)
from repo_constraint_memory_v3.semantic.validation import (
    CurrentSourceBundle,
    CurrentSourceIndex,
    CurrentSourcePolicyDecision,
)
from repo_constraint_memory_v3.source_adapters import GroundedEvidence
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest as compute_artifact_digest,
    canonical_bytes,
    seal_payload,
)
from repo_constraint_memory_v3.unified.models import ProfileDigest


UNIFIED_UPSTREAM_INPUT_VERSION = "unified-upstream-inputs-1.1"
MAX_UNIFIED_UPSTREAM_BYTES = 500_000_000

GitObjectId = Annotated[
    str,
    Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"),
]


class UnifiedUpstreamLoadError(ValueError):
    """An upstream artifact is unsafe, malformed, or bound to another run."""


class GroundedEvidenceRecord(StrictModel):
    """JSON form of the frozen dataclass used by explicit adjudication."""

    evidence: Evidence
    semantic_flags: list[NonEmptyString]
    contributor_context: bool
    hard_modal: bool
    soft_modal: bool
    hard_directive: bool
    soft_directive: bool
    example_context: bool

    @field_validator("semantic_flags")
    @classmethod
    def canonical_flags(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)):
            raise ValueError("grounded semantic flags must be sorted and unique")
        return values

    @classmethod
    def from_runtime(cls, value: GroundedEvidence) -> GroundedEvidenceRecord:
        return cls(
            evidence=value.evidence,
            semantic_flags=sorted(value.semantic_flags),
            contributor_context=value.contributor_context,
            hard_modal=value.hard_modal,
            soft_modal=value.soft_modal,
            hard_directive=value.hard_directive,
            soft_directive=value.soft_directive,
            example_context=value.example_context,
        )

    def to_runtime(self) -> GroundedEvidence:
        return GroundedEvidence(
            evidence=self.evidence,
            semantic_flags=frozenset(self.semantic_flags),
            contributor_context=self.contributor_context,
            hard_modal=self.hard_modal,
            soft_modal=self.soft_modal,
            hard_directive=self.hard_directive,
            soft_directive=self.soft_directive,
            example_context=self.example_context,
        )


class CandidateDecisionRecord(StrictModel):
    """Lossless, strict representation of every host-owned decision field."""

    unit_id: UnitId
    candidate_fingerprint: Sha256Hex
    outcome: LocalOutcome
    outcome_stage: OutcomeStage
    reason_code: NonEmptyString
    raw_candidate: CandidateClaim | None
    errors: list[str]
    grounded: list[GroundedEvidenceRecord]
    relations: list[EvidenceRelation]
    derived_kind: ClaimKind | None
    derived_strength: NormativeStrength | None
    acceptance_basis: list[AcceptanceBasis]
    semantic_core: str | None
    canonical_scope: list[str]
    canonical_condition: str
    condition_source: str
    host_eligible: bool
    verification_disposition: str | None
    verification_request_id: str | None
    verification_reason_codes: list[str]
    verification_technical_reason_code: str | None
    claim_id: str | None
    proposition: PropositionEnvelope | None

    @classmethod
    def from_runtime(cls, value: CandidateDecision) -> CandidateDecisionRecord:
        return cls(
            unit_id=value.unit_id,
            candidate_fingerprint=value.candidate_fingerprint,
            outcome=value.outcome,
            outcome_stage=value.outcome_stage,
            reason_code=value.reason_code,
            raw_candidate=value.raw_candidate,
            errors=list(value.errors),
            grounded=[
                GroundedEvidenceRecord.from_runtime(item)
                for item in value.grounded
            ],
            relations=list(value.relations),
            derived_kind=value.derived_kind,
            derived_strength=value.derived_strength,
            acceptance_basis=list(value.acceptance_basis),
            semantic_core=value.semantic_core,
            canonical_scope=list(value.canonical_scope),
            canonical_condition=value.canonical_condition,
            condition_source=value.condition_source,
            host_eligible=value.host_eligible,
            verification_disposition=value.verification_disposition,
            verification_request_id=value.verification_request_id,
            verification_reason_codes=list(value.verification_reason_codes),
            verification_technical_reason_code=(
                value.verification_technical_reason_code
            ),
            claim_id=value.claim_id,
            proposition=value.proposition,
        )

    def to_runtime(self) -> CandidateDecision:
        return CandidateDecision(
            unit_id=self.unit_id,
            candidate_fingerprint=self.candidate_fingerprint,
            outcome=self.outcome,
            outcome_stage=self.outcome_stage,
            reason_code=self.reason_code,
            raw_candidate=self.raw_candidate,
            errors=list(self.errors),
            grounded=[item.to_runtime() for item in self.grounded],
            relations=list(self.relations),
            derived_kind=self.derived_kind,
            derived_strength=self.derived_strength,
            acceptance_basis=list(self.acceptance_basis),
            semantic_core=self.semantic_core,
            canonical_scope=tuple(self.canonical_scope),
            canonical_condition=self.canonical_condition,
            condition_source=self.condition_source,
            host_eligible=self.host_eligible,
            verification_disposition=self.verification_disposition,
            verification_request_id=self.verification_request_id,
            verification_reason_codes=list(self.verification_reason_codes),
            verification_technical_reason_code=(
                self.verification_technical_reason_code
            ),
            claim_id=self.claim_id,
            proposition=self.proposition,
        )


def _unique(values: Sequence[str], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


class CurrentSourceAdapterInputs(StrictModel):
    """Minimal lossless current-source view consumed by unified adapters."""

    snapshot_digest: SnapshotDigest
    index: CurrentSourceIndex
    semantic_observations: list[SemanticObservation]
    hypotheses: list[NormHypothesis]
    populations: list[PopulationArtifact]
    member_packets: list[PopulationMemberPacket]
    falsifications: list[FalsificationResult]
    refinements: list[RefinementRecord]
    policy_decisions: list[CurrentSourcePolicyDecision]

    @model_validator(mode="after")
    def snapshot_and_identity_integrity(self) -> CurrentSourceAdapterInputs:
        if self.index.snapshot_digest != self.snapshot_digest:
            raise ValueError("current-source index snapshot mismatch")
        collections = (
            self.semantic_observations,
            self.hypotheses,
            self.populations,
            self.member_packets,
            self.falsifications,
            self.refinements,
            self.policy_decisions,
        )
        if any(
            item.snapshot_digest != self.snapshot_digest
            for items in collections
            for item in items
        ):
            raise ValueError("current-source artifact snapshot mismatch")

        _unique(
            [item.observation_id for item in self.semantic_observations],
            label="semantic observation IDs",
        )
        _unique(
            [item.hypothesis_id for item in self.hypotheses],
            label="hypothesis IDs",
        )
        _unique(
            [item.population_id for item in self.populations],
            label="population IDs",
        )
        _unique(
            [item.packet_id for item in self.member_packets],
            label="population packet IDs",
        )
        _unique(
            [item.falsification_id for item in self.falsifications],
            label="falsification IDs",
        )
        _unique(
            [item.refinement_id for item in self.refinements],
            label="refinement IDs",
        )
        _unique(
            [item.candidate_root_id for item in self.policy_decisions],
            label="current-source policy roots",
        )
        return self

    @classmethod
    def from_runtime(
        cls, value: CurrentSourceBundle
    ) -> CurrentSourceAdapterInputs:
        return cls(
            snapshot_digest=value.index.snapshot_digest,
            index=value.index,
            semantic_observations=list(value.semantic_map.observations),
            hypotheses=list(value.hypotheses),
            populations=list(value.populations),
            member_packets=list(value.member_packets),
            falsifications=list(value.falsifications),
            refinements=list(value.refinements),
            policy_decisions=list(value.policy_decisions),
        )


class ObservationShardInputs(StrictModel):
    """Snapshot binding for shards whose legacy schema has no snapshot field."""

    snapshot_digest: SnapshotDigest
    observation_set_digest: Sha256Hex
    overall: Literal["complete", "partial"]
    incomplete_reasons: list[NonEmptyString]
    shards: list[PythonFileShard]

    @model_validator(mode="after")
    def unique_shards(self) -> ObservationShardInputs:
        _unique([item.file_id for item in self.shards], label="observation file IDs")
        _unique([item.path for item in self.shards], label="observation paths")
        if self.incomplete_reasons != sorted(set(self.incomplete_reasons)):
            raise ValueError("observation incomplete reasons must be canonical")
        if self.overall == "complete" and self.incomplete_reasons:
            raise ValueError("complete observation projection cannot carry gaps")
        return self


class UnifiedUpstreamInputs(StrictModel):
    """Self-digested schema-4 input envelope for deterministic unified replay."""

    artifact_kind: Literal["unified_upstream_inputs"]
    schema_version: Literal["4.11-partition-only-required-evidence"]
    input_version: Literal["unified-upstream-inputs-1.1"]
    snapshot_digest: SnapshotDigest
    profile_digest: ProfileDigest
    snapshot_commit: GitObjectId | None
    explicit_decisions: list[CandidateDecisionRecord]
    explicit_verification: ExplicitVerificationResult
    current_source: CurrentSourceAdapterInputs | None
    observation_shards: ObservationShardInputs | None
    explicit_discovery_complete: Annotated[
        bool,
        Field(
            description=(
                "Candidate-enumeration completeness only: source discovery "
                "completed and every planned explicit unit produced one "
                "complete structured response. Known candidate grounding or "
                "semantic-verification failures remain candidate-local."
            )
        ),
    ]
    raw_candidate_events: Annotated[int, Field(ge=0)]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def bindings_and_digest(self) -> UnifiedUpstreamInputs:
        if self.current_source is not None and (
            self.current_source.snapshot_digest != self.snapshot_digest
        ):
            raise ValueError("current-source envelope snapshot mismatch")
        if self.observation_shards is not None and (
            self.observation_shards.snapshot_digest != self.snapshot_digest
        ):
            raise ValueError("observation-shard envelope snapshot mismatch")
        for decision in self.explicit_decisions:
            if (
                decision.proposition is not None
                and decision.proposition.snapshot_digest != self.snapshot_digest
            ):
                raise ValueError("explicit proposition snapshot mismatch")

        explicit_roots = {
            explicit_candidate_root_id(item.candidate_fingerprint)
            for item in self.explicit_decisions
            if item.raw_candidate is not None
        }
        verification_roots = {
            item.candidate_root_id
            for item in self.explicit_verification.decisions
        }
        if not verification_roots <= explicit_roots:
            raise ValueError(
                "explicit verification references an unknown candidate root"
            )
        if self.artifact_digest != compute_artifact_digest(self):
            raise ValueError("incorrect unified upstream artifact digest")
        return self


@dataclass(frozen=True)
class SemanticMapRuntimeView:
    observations: tuple[SemanticObservation, ...]


@dataclass(frozen=True)
class CurrentSourceRuntimeView:
    """Duck-typed replacement for the fields unified adapters consume."""

    index: CurrentSourceIndex
    semantic_map: SemanticMapRuntimeView
    hypotheses: tuple[NormHypothesis, ...]
    populations: tuple[PopulationArtifact, ...]
    member_packets: tuple[PopulationMemberPacket, ...]
    falsifications: tuple[FalsificationResult, ...]
    refinements: tuple[RefinementRecord, ...]
    policy_decisions: tuple[CurrentSourcePolicyDecision, ...]


@dataclass(frozen=True)
class ObservationRuntimeView:
    """Duck-typed replacement for the observation fields unified adapters use."""

    snapshot_digest: str
    observation_set_digest: str
    overall: Literal["complete", "partial"]
    incomplete_reasons: tuple[str, ...]
    shards: tuple[PythonFileShard, ...]


@dataclass(frozen=True)
class UnifiedRuntimeInputs(Mapping[str, object]):
    """Runtime values that can be passed directly as ``**inputs``."""

    snapshot_digest: str
    snapshot_commit: str | None
    profile_digest: str
    explicit_decisions: tuple[CandidateDecision, ...]
    explicit_verification: ExplicitVerificationResult
    current_source: CurrentSourceRuntimeView | None
    observations: ObservationRuntimeView | None
    explicit_discovery_complete: bool
    raw_candidate_events: int

    _KEYS: ClassVar[tuple[str, ...]] = (
        "snapshot_digest",
        "snapshot_commit",
        "profile_digest",
        "explicit_decisions",
        "explicit_verification",
        "current_source",
        "observations",
        "explicit_discovery_complete",
        "raw_candidate_events",
    )

    def __getitem__(self, key: str) -> object:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    def pipeline_kwargs(self) -> dict[str, object]:
        """Return a plain mapping for callers that prefer an explicit copy."""

        return {key: self[key] for key in self._KEYS}


def _runtime_inputs(value: UnifiedUpstreamInputs) -> UnifiedRuntimeInputs:
    current = value.current_source
    current_runtime = (
        CurrentSourceRuntimeView(
            index=current.index,
            semantic_map=SemanticMapRuntimeView(
                observations=tuple(current.semantic_observations)
            ),
            hypotheses=tuple(current.hypotheses),
            populations=tuple(current.populations),
            member_packets=tuple(current.member_packets),
            falsifications=tuple(current.falsifications),
            refinements=tuple(current.refinements),
            policy_decisions=tuple(current.policy_decisions),
        )
        if current is not None
        else None
    )
    observations = value.observation_shards
    return UnifiedRuntimeInputs(
        snapshot_digest=value.snapshot_digest,
        snapshot_commit=value.snapshot_commit,
        profile_digest=value.profile_digest,
        explicit_decisions=tuple(
            item.to_runtime() for item in value.explicit_decisions
        ),
        explicit_verification=value.explicit_verification,
        current_source=current_runtime,
        observations=(
            ObservationRuntimeView(
                snapshot_digest=observations.snapshot_digest,
                observation_set_digest=observations.observation_set_digest,
                overall=observations.overall,
                incomplete_reasons=tuple(observations.incomplete_reasons),
                shards=tuple(observations.shards),
            )
            if observations is not None
            else None
        ),
        explicit_discovery_complete=value.explicit_discovery_complete,
        raw_candidate_events=value.raw_candidate_events,
    )


def make_unified_upstream_inputs(
    *,
    snapshot_digest: str,
    profile_digest: str,
    snapshot_commit: str | None,
    explicit_decisions: Sequence[CandidateDecision],
    explicit_verification: ExplicitVerificationResult,
    current_source: CurrentSourceBundle | None,
    observations: ObservationBundle | None,
    explicit_discovery_complete: bool,
    raw_candidate_events: int | None = None,
) -> UnifiedUpstreamInputs:
    """Freeze exactly the live values consumed by ``run_unified_layer``."""

    if current_source is not None and (
        current_source.index.snapshot_digest != snapshot_digest
    ):
        raise ValueError("current-source input snapshot mismatch")
    if observations is not None and (
        observations.index.snapshot_digest != snapshot_digest
    ):
        raise ValueError("observation input snapshot mismatch")

    payload: dict[str, object] = {
        "artifact_kind": "unified_upstream_inputs",
        "schema_version": SCHEMA_VERSION,
        "input_version": UNIFIED_UPSTREAM_INPUT_VERSION,
        "snapshot_digest": snapshot_digest,
        "profile_digest": profile_digest,
        "snapshot_commit": snapshot_commit,
        "explicit_decisions": [
            CandidateDecisionRecord.from_runtime(item)
            for item in explicit_decisions
        ],
        "explicit_verification": explicit_verification,
        "current_source": (
            CurrentSourceAdapterInputs.from_runtime(current_source)
            if current_source is not None
            else None
        ),
        "observation_shards": (
            ObservationShardInputs(
                snapshot_digest=snapshot_digest,
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
        ),
        "explicit_discovery_complete": explicit_discovery_complete,
        "raw_candidate_events": (
            len(explicit_decisions)
            if raw_candidate_events is None
            else raw_candidate_events
        ),
    }
    return UnifiedUpstreamInputs.model_validate(seal_payload(payload))


def canonical_unified_upstream_bytes(value: UnifiedUpstreamInputs) -> bytes:
    """Return the unique UTF-8 JSON encoding used for persisted artifacts."""

    return canonical_bytes(value)


def canonical_unified_upstream_text(value: UnifiedUpstreamInputs) -> str:
    return canonical_unified_upstream_bytes(value).decode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise UnifiedUpstreamLoadError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> object:
    raise UnifiedUpstreamLoadError(f"non-finite JSON constant is forbidden: {value}")


def _read_regular_bounded(path: Path, *, max_bytes: int) -> bytes:
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    try:
        before = path.lstat()
    except OSError as error:
        raise UnifiedUpstreamLoadError(
            f"cannot stat unified upstream artifact: {error}"
        ) from error
    if stat.S_ISLNK(before.st_mode):
        raise UnifiedUpstreamLoadError("unified upstream artifact cannot be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise UnifiedUpstreamLoadError(
            "unified upstream artifact must be a regular file"
        )
    if before.st_size > max_bytes:
        raise UnifiedUpstreamLoadError("unified upstream artifact exceeds byte limit")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise UnifiedUpstreamLoadError(
            f"cannot open unified upstream artifact safely: {error}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise UnifiedUpstreamLoadError(
                "unified upstream artifact must remain a regular file"
            )
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise UnifiedUpstreamLoadError(
                "unified upstream artifact changed during safe open"
            )
        if opened.st_size > max_bytes:
            raise UnifiedUpstreamLoadError(
                "unified upstream artifact exceeds byte limit"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise UnifiedUpstreamLoadError(
                "unified upstream artifact exceeds byte limit"
            )
        if os.fstat(descriptor).st_size != len(data):
            raise UnifiedUpstreamLoadError(
                "unified upstream artifact changed while being read"
            )
        return data
    finally:
        os.close(descriptor)


def load_unified_upstream_artifact(
    path: Path | str,
    *,
    expected_snapshot_digest: str,
    expected_profile_digest: str,
    max_bytes: int = MAX_UNIFIED_UPSTREAM_BYTES,
) -> UnifiedUpstreamInputs:
    """Safely load and authenticate one immutable upstream envelope."""

    data = _read_regular_bounded(Path(path), max_bytes=max_bytes)
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise UnifiedUpstreamLoadError(
            "unified upstream artifact is not valid UTF-8"
        ) from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except UnifiedUpstreamLoadError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise UnifiedUpstreamLoadError(
            f"invalid unified upstream JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise UnifiedUpstreamLoadError("unified upstream root must be a JSON object")
    try:
        artifact = UnifiedUpstreamInputs.model_validate(payload)
    except ValidationError as error:
        raise UnifiedUpstreamLoadError(
            f"invalid unified upstream artifact: {error}"
        ) from error
    if artifact.snapshot_digest != expected_snapshot_digest:
        raise UnifiedUpstreamLoadError("unified upstream snapshot mismatch")
    if artifact.profile_digest != expected_profile_digest:
        raise UnifiedUpstreamLoadError("unified upstream profile mismatch")
    return artifact


def load_unified_upstream_inputs(
    path: Path | str,
    *,
    expected_snapshot_digest: str,
    expected_profile_digest: str,
    max_bytes: int = MAX_UNIFIED_UPSTREAM_BYTES,
) -> UnifiedRuntimeInputs:
    """Safely load, authenticate, bind, and reconstruct pipeline inputs."""

    artifact = load_unified_upstream_artifact(
        path,
        expected_snapshot_digest=expected_snapshot_digest,
        expected_profile_digest=expected_profile_digest,
        max_bytes=max_bytes,
    )
    return _runtime_inputs(artifact)


__all__ = [
    "MAX_UNIFIED_UPSTREAM_BYTES",
    "UNIFIED_UPSTREAM_INPUT_VERSION",
    "CandidateDecisionRecord",
    "CurrentSourceAdapterInputs",
    "CurrentSourceRuntimeView",
    "GroundedEvidenceRecord",
    "ObservationRuntimeView",
    "ObservationShardInputs",
    "SemanticMapRuntimeView",
    "UnifiedRuntimeInputs",
    "UnifiedUpstreamInputs",
    "UnifiedUpstreamLoadError",
    "canonical_unified_upstream_bytes",
    "canonical_unified_upstream_text",
    "load_unified_upstream_artifact",
    "load_unified_upstream_inputs",
    "make_unified_upstream_inputs",
]
