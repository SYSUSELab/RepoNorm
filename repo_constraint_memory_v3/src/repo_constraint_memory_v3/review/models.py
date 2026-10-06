"""Snapshot-bound developer decisions, independent of canonical truth."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from repo_constraint_memory_v3.models import Sha256Hex, SnapshotDigest, StrictModel
from repo_constraint_memory_v3.repository_paths import is_canonical_repository_path
from repo_constraint_memory_v3.unified.ids import artifact_digest

REVIEW_PROTOCOL_VERSION = "developer-review-1.0"


class ReviewEvidence(StrictModel):
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    start_line: Annotated[int, Field(ge=1)]
    end_line: Annotated[int, Field(ge=1)]
    description: Annotated[str, Field(max_length=8192)] = ""
    content_digest: Sha256Hex | None = None

    @model_validator(mode="after")
    def location(self) -> ReviewEvidence:
        if not is_canonical_repository_path(self.path):
            raise ValueError("review evidence requires a safe repository-relative path")
        if self.end_line < self.start_line:
            raise ValueError("review evidence end precedes start")
        return self


class ReviewDecision(StrictModel):
    claim_id: Annotated[str, Field(pattern=r"^CLAIM-[0-9a-f]{24}$")]
    claim_digest: Sha256Hex
    decision: Literal["accept", "reject"]
    reviewer_kind: Literal["developer", "model"]
    reviewer_id: Annotated[str, Field(min_length=1, max_length=1024)]
    rationale: Annotated[str, Field(max_length=32768)] = ""
    evidence: Annotated[list[ReviewEvidence], Field(max_length=128)] = Field(default_factory=list)
    execution_ref: str | None = None
    execution_digest: Sha256Hex | None = None
    decided_at: str
    version: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def model_provenance(self) -> ReviewDecision:
        if self.execution_ref is not None and not is_canonical_repository_path(self.execution_ref):
            raise ValueError("execution reference must be a safe relative path")
        if (self.execution_ref is None) != (self.execution_digest is None):
            raise ValueError("execution reference and digest must be bound together")
        if self.reviewer_kind == "model" and (not self.rationale.strip() or not self.evidence or not self.execution_ref):
            raise ValueError("model decisions require rationale, repository evidence and execution reference")
        return self


class ReviewSession(StrictModel):
    format_version: Literal["developer-review-1.0"] = REVIEW_PROTOCOL_VERSION
    session_id: Annotated[str, Field(pattern=r"^REVIEW-[0-9a-f]{32}$")]
    source_run: str
    repository_root: str
    repository: str
    snapshot_digest: SnapshotDigest
    snapshot_commit: str | None
    canonical_claims_digest: Sha256Hex
    completeness_digest: Sha256Hex
    source_tree_digest: Sha256Hex
    created_at: str
    claim_digests: dict[str, Sha256Hex]
    decisions: list[ReviewDecision] = Field(default_factory=list)
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def identity(self) -> ReviewSession:
        if list(self.claim_digests) != sorted(self.claim_digests):
            raise ValueError("review claim bindings must be sorted")
        versions: dict[str, int] = {}
        for item in self.decisions:
            if self.claim_digests.get(item.claim_id) != item.claim_digest:
                raise ValueError("decision differs from review claim binding")
            expected = versions.get(item.claim_id, 0) + 1
            if item.version != expected:
                raise ValueError("decision versions must be contiguous per claim")
            versions[item.claim_id] = item.version
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect review session digest")
        return self

    def latest_decisions(self) -> dict[str, ReviewDecision]:
        return {item.claim_id: item for item in self.decisions}

    def progress(self) -> dict[str, int]:
        latest = self.latest_decisions()
        return {
            "planned": len(self.claim_digests),
            "accepted": sum(item.decision == "accept" for item in latest.values()),
            "rejected": sum(item.decision == "reject" for item in latest.values()),
            "unprocessed": len(self.claim_digests) - len(latest),
        }


class ReviewConsumerManifest(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"] = "4.11-partition-only-required-evidence"
    format_version: Literal["review-consumer-manifest-1.0"] = "review-consumer-manifest-1.0"
    consumer_surface_version: Literal["review-consumer-surface-2.0"] = "review-consumer-surface-2.0"
    consumer_profile: Literal["accepted_after_review"] = "accepted_after_review"
    max_catalog_file_bytes: Literal[32768] = 32768
    max_rendered_compact_card_bytes: Literal[24576] = 24576
    repository: str
    snapshot_digest: SnapshotDigest
    snapshot_commit: str | None
    acquisition_profile_digest: Sha256Hex
    canonical_claims_digest: Sha256Hex
    completeness_digest: Sha256Hex
    source_tree_digest: Sha256Hex
    review_session_id: str
    review_decisions_digest: Sha256Hex
    package_status: Literal["complete", "partial"]
    canonical_accepted_count: Annotated[int, Field(ge=0)]
    canonical_review_count: Annotated[int, Field(ge=0)]
    review_accepted_count: Annotated[int, Field(ge=0)]
    review_rejected_count: Annotated[int, Field(ge=0)]
    review_unprocessed_count: Annotated[int, Field(ge=0)]
    exposed_accepted_count: Annotated[int, Field(ge=0)]
    delivered_claim_count: Annotated[int, Field(ge=0)]
    catalog_card_count: Annotated[int, Field(ge=0)]
    detail_file_count: Annotated[int, Field(ge=0)]
    markdown_file_count: Annotated[int, Field(ge=0)]
    task_tree_digest: Sha256Hex
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def counts_and_digest(self) -> ReviewConsumerManifest:
        if self.canonical_review_count != self.review_accepted_count + self.review_rejected_count + self.review_unprocessed_count:
            raise ValueError("review denominator differs from outcomes")
        if self.exposed_accepted_count != self.canonical_accepted_count + self.review_accepted_count:
            raise ValueError("review exposure count differs from accepted union")
        if self.catalog_card_count != self.exposed_accepted_count or self.detail_file_count != self.catalog_card_count:
            raise ValueError("review catalog/detail counts differ")
        if self.delivered_claim_count > self.exposed_accepted_count:
            raise ValueError("delivered count exceeds exposed claims")
        if artifact_digest(self) != self.artifact_digest:
            raise ValueError("incorrect review consumer manifest digest")
        return self
