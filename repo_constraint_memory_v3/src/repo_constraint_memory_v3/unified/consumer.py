"""Deterministic canonical files and static consumer-surface 3.0 views.

The functions in this module are deliberately provider-free and task-agnostic.
They project one structurally checked :class:`~UnifiedBundle` into canonical JSON
and isolated Markdown trees, and can replay a standalone tree byte-for-byte
against its immutable source run.  Repository text, model output, and saved
artifacts remain untrusted at every rendering boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast
from urllib.parse import quote

from pydantic import BaseModel, Field, model_validator

from repo_constraint_memory_v3.models import (
    BuildConfig,
    SCHEMA_VERSION,
    Sha256Hex,
    SnapshotDigest,
    StrictModel,
)
from repo_constraint_memory_v3.repository_paths import (
    MAX_CONSUMER_ANCHOR_RELATIVE_BYTES,
    MAX_PHYSICAL_PATH_COMPONENT_BYTES,
    escape_consumer_path_component,
    is_canonical_repository_path,
    is_consumer_routable_repository_path,
    is_safe_display_text,
)
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    content_digest,
    seal_payload,
)
from repo_constraint_memory_v3.unified.models import (
    CanonicalClaim,
    CanonicalEvidenceRef,
    CurrentEvidenceBundle,
    ImmutableArtifactRef,
    ProfileDigest,
    UnifiedBundle,
    UnifiedCaseFinding,
    UnifiedCompleteness,
)
from repo_constraint_memory_v3.unified.publication import (
    consume_initial_empty_directory,
    directory_tree_fingerprint,
    rename_directory_noreplace,
)


CANONICAL_CLAIMS_FORMAT_VERSION = "unified-canonical-claims-1.0"
CANONICAL_COMPLETENESS_FORMAT_VERSION = "unified-completeness-1.0"
CANONICAL_DELIVERY_REPORT_FORMAT_VERSION = "unified-delivery-report-2.0"
CONSUMER_MANIFEST_VERSION = "static-consumer-manifest-1.0"
CONSUMER_SURFACE_VERSION = "static-consumer-surface-3.0"
MAX_CATALOG_FILE_BYTES = 32_768
MAX_RENDERED_COMPACT_CARD_BYTES = 24_576
MAX_CATALOG_SHARDS = 9_999
MAX_CONSUMER_RELATIVE_PATH_BYTES = MAX_CONSUMER_ANCHOR_RELATIVE_BYTES

ConsumerProfile = Literal["accepted_only", "accepted_plus_review"]
_PROFILES: tuple[ConsumerProfile, ...] = (
    "accepted_only",
    "accepted_plus_review",
)
_SUCCESS_CAPABILITY = {"complete", "bounded_no_match", "not_required"}
_EXPLICIT_ORIGIN_KINDS = {
    "explicit_span",
    "configuration_atom",
    "command_atom",
    "workflow_atom",
}
_GLOB_COMPONENT = re.compile(r"[*?\[]")
_BACKTICK_RUN = re.compile(r"`+")
_SAFE_CLAIM_ID = re.compile(r"^CLAIM-[0-9a-f]{24}$")
_REVIEW_EXPLANATIONS: dict[str, str] = {
    "technical_or_coverage_gap": "Required evidence or coverage is incomplete.",
    "conflict_or_currentness": (
        "A material conflict or currentness question remains unresolved."
    ),
    "refinement_or_qualifier_needed": (
        "The rule, scope, condition, or exception still needs a supported "
        "refinement."
    ),
    "negative_basis_insufficient": (
        "Available negative evidence is insufficient for the proposed conclusion."
    ),
    "channel_disagreement": "Evidence channels disagree on this claim.",
    "factual_uncertainty": "The claim's factual interpretation remains uncertain.",
}
_AXIS_LABELS: tuple[tuple[str, str], ...] = (
    ("norm_relevance", "repository-coding relevance"),
    ("rule_support", "rule support"),
    ("scope_support", "scope support"),
    ("condition_support", "condition support"),
    ("temporal_status", "currentness"),
    ("evidence_sufficiency", "evidence sufficiency"),
)


def canonical_json_text(value: object) -> str:
    """Return canonical presentation JSON with one final newline."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


class CanonicalClaimsFile(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"] = SCHEMA_VERSION
    format_version: Literal["unified-canonical-claims-1.0"] = (
        CANONICAL_CLAIMS_FORMAT_VERSION
    )
    snapshot_digest: SnapshotDigest
    snapshot_commit: str | None
    profile_digest: ProfileDigest
    claims: list[CanonicalClaim]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def identity_and_digest(self) -> CanonicalClaimsFile:
        claim_ids = [item.claim_id for item in self.claims]
        if claim_ids != sorted(set(claim_ids)):
            raise ValueError("canonical claims must be uniquely sorted")
        if any(
            item.snapshot_digest != self.snapshot_digest
            or item.profile_digest != self.profile_digest
            for item in self.claims
        ):
            raise ValueError("canonical claim file identity mismatch")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect canonical claims file digest")
        return self


class CanonicalCompletenessFile(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"] = SCHEMA_VERSION
    format_version: Literal["unified-completeness-1.0"] = (
        CANONICAL_COMPLETENESS_FORMAT_VERSION
    )
    snapshot_digest: SnapshotDigest
    snapshot_commit: str | None
    profile_digest: ProfileDigest
    state: UnifiedCompleteness
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def identity_and_digest(self) -> CanonicalCompletenessFile:
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect canonical completeness digest")
        return self


class DeliveryExclusion(StrictModel):
    claim_id: str
    epistemic_status: Literal["accepted", "review", "rejected"]
    reason: Literal[
        "delivery_policy_reference_only",
        "epistemic_status_review",
        "epistemic_status_rejected",
    ]

    @model_validator(mode="after")
    def reason_matches_status(self) -> DeliveryExclusion:
        expected = (
            "delivery_policy_reference_only"
            if self.epistemic_status == "accepted"
            else f"epistemic_status_{self.epistemic_status}"
        )
        if self.reason != expected:
            raise ValueError("delivery exclusion reason differs from status")
        return self


class ConsumerProfileSummary(StrictModel):
    profile: ConsumerProfile
    exposed_accepted_count: Annotated[int, Field(ge=0)]
    experimentally_exposed_review_count: Annotated[int, Field(ge=0)]
    catalog_card_count: Annotated[int, Field(ge=0)]
    catalog_shard_count: Annotated[int, Field(ge=0, le=MAX_CATALOG_SHARDS)]
    detail_file_count: Annotated[int, Field(ge=0)]
    path_view_count: Annotated[int, Field(ge=0)]
    path_card_occurrence_count: Annotated[int, Field(ge=0)]
    path_unique_claim_count: Annotated[int, Field(ge=0)]
    markdown_file_count: Annotated[int, Field(ge=0)]
    tree_digest: Sha256Hex

    @model_validator(mode="after")
    def count_algebra(self) -> ConsumerProfileSummary:
        if self.catalog_card_count != (
            self.exposed_accepted_count
            + self.experimentally_exposed_review_count
        ):
            raise ValueError("consumer catalog exposure counts do not sum")
        if self.detail_file_count != self.catalog_card_count:
            raise ValueError("consumer detail count differs from catalog count")
        if self.path_unique_claim_count > self.path_card_occurrence_count:
            raise ValueError("consumer path unique count exceeds occurrences")
        minimum_markdown_files = (
            3
            + self.detail_file_count
            + self.path_view_count
            + self.catalog_shard_count
        )
        if self.markdown_file_count != minimum_markdown_files:
            raise ValueError("consumer Markdown count differs from projection counts")
        return self


class ConsumerManifest(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"] = SCHEMA_VERSION
    format_version: Literal["static-consumer-manifest-1.0"] = (
        CONSUMER_MANIFEST_VERSION
    )
    consumer_surface_version: Literal["static-consumer-surface-3.0"] = (
        CONSUMER_SURFACE_VERSION
    )
    repository: Annotated[str, Field(min_length=1, max_length=1_024)]
    snapshot_digest: SnapshotDigest
    snapshot_commit: str | None
    acquisition_profile_digest: ProfileDigest
    consumer_profile: ConsumerProfile
    canonical_claims_digest: Sha256Hex
    completeness_digest: Sha256Hex
    package_status: Literal["complete", "partial"]
    max_catalog_file_bytes: Literal[32768] = MAX_CATALOG_FILE_BYTES
    max_rendered_compact_card_bytes: Literal[24576] = (
        MAX_RENDERED_COMPACT_CARD_BYTES
    )
    requested_consumer_profiles: list[ConsumerProfile]
    emitted_consumer_profiles: list[ConsumerProfile]
    canonical_accepted_count: Annotated[int, Field(ge=0)]
    canonical_review_count: Annotated[int, Field(ge=0)]
    canonical_rejected_count: Annotated[int, Field(ge=0)]
    exposed_accepted_count: Annotated[int, Field(ge=0)]
    experimentally_exposed_review_count: Annotated[int, Field(ge=0)]
    delivered_claim_count: Annotated[int, Field(ge=0)]
    accepted_policy_counts: dict[str, Annotated[int, Field(ge=0)]]
    accepted_risk_counts: dict[str, Annotated[int, Field(ge=0)]]
    review_facet_counts: dict[str, Annotated[int, Field(ge=0)]]
    review_facet_counts_are_overlapping: Literal[True] = True
    catalog_card_count: Annotated[int, Field(ge=0)]
    catalog_shard_count: Annotated[int, Field(ge=0, le=MAX_CATALOG_SHARDS)]
    detail_file_count: Annotated[int, Field(ge=0)]
    path_view_count: Annotated[int, Field(ge=0)]
    path_card_occurrence_count: Annotated[int, Field(ge=0)]
    path_unique_claim_count: Annotated[int, Field(ge=0)]
    markdown_file_count: Annotated[int, Field(ge=0)]
    task_tree_digest: Sha256Hex
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def identity_counts_and_digest(self) -> ConsumerManifest:
        for values, label in (
            (self.requested_consumer_profiles, "requested consumer profiles"),
            (self.emitted_consumer_profiles, "emitted consumer profiles"),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"{label} must be uniquely sorted")
        if self.requested_consumer_profiles != self.emitted_consumer_profiles:
            raise ValueError("completed view set differs from requested profiles")
        if self.consumer_profile not in self.emitted_consumer_profiles:
            raise ValueError("consumer manifest profile was not emitted")
        if self.consumer_profile == "accepted_only":
            if self.experimentally_exposed_review_count:
                raise ValueError("accepted-only manifest exposes review claims")
        elif (
            self.experimentally_exposed_review_count
            != self.canonical_review_count
        ):
            raise ValueError("review-exposure manifest count is incomplete")
        if self.exposed_accepted_count != self.canonical_accepted_count:
            raise ValueError("consumer manifest omits accepted claims")
        if self.catalog_card_count != (
            self.exposed_accepted_count
            + self.experimentally_exposed_review_count
        ):
            raise ValueError("consumer manifest catalog counts do not sum")
        if self.detail_file_count != self.catalog_card_count:
            raise ValueError("consumer manifest detail count differs from catalog")
        if self.path_unique_claim_count > self.path_card_occurrence_count:
            raise ValueError("consumer manifest path unique count exceeds occurrences")
        if sum(self.accepted_policy_counts.values()) != self.canonical_accepted_count:
            raise ValueError("consumer manifest accepted policy counts do not sum")
        if sum(self.accepted_risk_counts.values()) != self.canonical_accepted_count:
            raise ValueError("consumer manifest accepted risk counts do not sum")
        for counts, label in (
            (self.accepted_policy_counts, "accepted policy counts"),
            (self.accepted_risk_counts, "accepted risk counts"),
            (self.review_facet_counts, "review facet counts"),
        ):
            if counts != dict(sorted(counts.items())):
                raise ValueError(f"{label} must be canonical")
        expected_markdown = (
            3
            + self.detail_file_count
            + self.path_view_count
            + self.catalog_shard_count
        )
        if self.markdown_file_count != expected_markdown:
            raise ValueError("consumer manifest Markdown count is invalid")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect consumer manifest digest")
        return self


class CanonicalDeliveryReport(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"] = SCHEMA_VERSION
    format_version: Literal["unified-delivery-report-2.0"] = (
        CANONICAL_DELIVERY_REPORT_FORMAT_VERSION
    )
    consumer_manifest_version: Literal["static-consumer-manifest-1.0"] = (
        CONSUMER_MANIFEST_VERSION
    )
    consumer_surface_version: Literal["static-consumer-surface-3.0"] = (
        CONSUMER_SURFACE_VERSION
    )
    max_catalog_file_bytes: Literal[32768] = MAX_CATALOG_FILE_BYTES
    max_rendered_compact_card_bytes: Literal[24576] = (
        MAX_RENDERED_COMPACT_CARD_BYTES
    )
    snapshot_digest: SnapshotDigest
    snapshot_commit: str | None
    profile_digest: ProfileDigest
    canonical_claims_digest: Sha256Hex
    completeness_digest: Sha256Hex
    accepted_claim_count: Annotated[int, Field(ge=0)]
    review_claim_count: Annotated[int, Field(ge=0)]
    rejected_claim_count: Annotated[int, Field(ge=0)]
    delivered_claim_count: Annotated[int, Field(ge=0)]
    delivered_claim_ids: list[str]
    accepted_policy_counts: dict[str, Annotated[int, Field(ge=0)]]
    accepted_risk_counts: dict[str, Annotated[int, Field(ge=0)]]
    review_facet_counts: dict[str, Annotated[int, Field(ge=0)]]
    review_facet_counts_are_overlapping: Literal[True] = True
    exclusions: list[DeliveryExclusion]
    requested_consumer_profiles: list[ConsumerProfile]
    emitted_consumer_profiles: list[ConsumerProfile]
    consumer_profiles: list[ConsumerProfileSummary]
    consumer_manifest_digests: dict[ConsumerProfile, Sha256Hex]
    artifact_digest: Sha256Hex

    @model_validator(mode="after")
    def canonical_counts_and_digest(self) -> CanonicalDeliveryReport:
        for values, label in (
            (self.delivered_claim_ids, "delivered claim IDs"),
            (self.requested_consumer_profiles, "requested consumer profiles"),
            (self.emitted_consumer_profiles, "emitted consumer profiles"),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"{label} must be uniquely sorted")
        if self.requested_consumer_profiles != self.emitted_consumer_profiles:
            raise ValueError("completed run did not emit every requested profile")
        summaries = [item.profile for item in self.consumer_profiles]
        if summaries != self.emitted_consumer_profiles:
            raise ValueError("consumer summaries differ from emitted profiles")
        if list(self.consumer_manifest_digests) != self.emitted_consumer_profiles:
            raise ValueError("consumer manifest digests differ from emitted profiles")
        if self.delivered_claim_count != len(self.delivered_claim_ids):
            raise ValueError("delivered claim count differs from IDs")
        exclusions = [item.claim_id for item in self.exclusions]
        if exclusions != sorted(set(exclusions)):
            raise ValueError("delivery exclusions must be uniquely sorted")
        for counts, label in (
            (self.accepted_policy_counts, "accepted policy counts"),
            (self.accepted_risk_counts, "accepted risk counts"),
            (self.review_facet_counts, "review facet counts"),
        ):
            if counts != dict(sorted(counts.items())):
                raise ValueError(f"{label} must be canonical")
        if sum(self.accepted_policy_counts.values()) != self.accepted_claim_count:
            raise ValueError("accepted policy counts do not sum")
        if sum(self.accepted_risk_counts.values()) != self.accepted_claim_count:
            raise ValueError("accepted risk counts do not sum")
        if self.artifact_digest != artifact_digest(self):
            raise ValueError("incorrect canonical delivery report digest")
        return self


@dataclass(frozen=True)
class ConsumerProjection:
    catalog_claims: tuple[CanonicalClaim, ...]
    detail_claims: tuple[CanonicalClaim, ...]
    path_claims: tuple[CanonicalClaim, ...]


@dataclass(frozen=True)
class ConsumerViewPlan:
    profile: ConsumerProfile
    files: Mapping[str, bytes]
    directories: tuple[str, ...]
    summary: ConsumerProfileSummary
    manifest: ConsumerManifest


@dataclass(frozen=True)
class ConsumerArtifacts:
    canonical_claims: CanonicalClaimsFile
    completeness: CanonicalCompletenessFile
    delivery_report: CanonicalDeliveryReport
    views: Mapping[ConsumerProfile, ConsumerViewPlan]

    def report_metadata(self) -> dict[str, object]:
        return {
            "surface_version": CONSUMER_SURFACE_VERSION,
            "manifest_version": CONSUMER_MANIFEST_VERSION,
            "max_catalog_file_bytes": MAX_CATALOG_FILE_BYTES,
            "max_rendered_compact_card_bytes": (
                MAX_RENDERED_COMPACT_CARD_BYTES
            ),
            "canonical_claims_format": CANONICAL_CLAIMS_FORMAT_VERSION,
            "canonical_claims_digest": self.canonical_claims.artifact_digest,
            "completeness_format": CANONICAL_COMPLETENESS_FORMAT_VERSION,
            "completeness_digest": self.completeness.artifact_digest,
            "delivery_report_format": CANONICAL_DELIVERY_REPORT_FORMAT_VERSION,
            "delivery_report_digest": self.delivery_report.artifact_digest,
            "requested_profiles": list(
                self.delivery_report.requested_consumer_profiles
            ),
            "emitted_profiles": list(
                self.delivery_report.emitted_consumer_profiles
            ),
            "profiles": [
                item.model_dump(mode="json")
                for item in self.delivery_report.consumer_profiles
            ],
            "manifest_digests": dict(
                self.delivery_report.consumer_manifest_digests
            ),
        }


def normalize_consumer_profiles(
    profiles: Sequence[str],
) -> tuple[ConsumerProfile, ...]:
    """Validate requested views and always include the conservative default."""

    supplied = set(profiles)
    if not supplied:
        supplied = {"accepted_only"}
    unknown = supplied - set(_PROFILES)
    if unknown:
        raise ValueError("unknown consumer profile(s): " + ", ".join(sorted(unknown)))
    supplied.add("accepted_only")
    return cast(tuple[ConsumerProfile, ...], tuple(sorted(supplied)))


def _consumer_profile(value: str) -> ConsumerProfile:
    if value not in _PROFILES:
        raise ValueError(f"unknown consumer profile: {value}")
    return cast(ConsumerProfile, value)


def project_claims(
    claims: Sequence[CanonicalClaim],
    profile: ConsumerProfile,
) -> ConsumerProjection:
    """Return the exact A/R/D projection sets for one stable profile."""

    _consumer_profile(profile)
    by_id: dict[str, CanonicalClaim] = {}
    for claim in claims:
        if claim.claim_id in by_id:
            raise ValueError("canonical claims contain duplicate IDs")
        by_id[claim.claim_id] = claim
    accepted = {
        claim_id: item
        for claim_id, item in by_id.items()
        if item.epistemic_status == "accepted"
    }
    review = {
        claim_id: item
        for claim_id, item in by_id.items()
        if item.epistemic_status == "review"
    }
    exposed = dict(accepted)
    if profile == "accepted_plus_review":
        exposed.update(review)
    catalog = tuple(exposed[key] for key in sorted(exposed))
    path = tuple(
        item
        for item in catalog
        if item.epistemic_status == "review"
        or item.delivery_policy
        in {"default_consider", "condition_matched_only"}
    )
    return ConsumerProjection(
        catalog_claims=catalog,
        detail_claims=catalog,
        path_claims=path,
    )


def _safe_display_text(value: str) -> bool:
    return is_safe_display_text(value)


def _canonical_repository_path(value: str) -> bool:
    return is_canonical_repository_path(value)


def _escape_consumer_component(component: str) -> str:
    return escape_consumer_path_component(component)


def _routable_repository_path(value: str) -> bool:
    return is_consumer_routable_repository_path(value)


def _path_anchor(
    pattern: str,
    *,
    allow_nonaccepted_directory_marker: bool,
) -> str:
    routed = pattern
    if not _routable_repository_path(routed):
        if (
            not allow_nonaccepted_directory_marker
            or not pattern.endswith("/")
            or not _routable_repository_path(pattern[:-1])
        ):
            raise ValueError(f"unsafe claim scope: {pattern!r}")
        routed = pattern[:-1]
    parts = PurePosixPath(routed).parts
    first_glob = next(
        (
            ordinal
            for ordinal, component in enumerate(parts)
            if _GLOB_COMPONENT.search(component) is not None
        ),
        None,
    )
    if first_glob is None:
        return PurePosixPath(*parts).as_posix()
    if first_glob == 0:
        return ""
    return PurePosixPath(*parts[:first_glob]).as_posix()


def path_anchor(pattern: str) -> str:
    """Return the hybrid logical anchor for one strict accepted scope."""

    return _path_anchor(pattern, allow_nonaccepted_directory_marker=False)


def claim_path_anchor(claim: CanonicalClaim, pattern: str) -> str:
    """Route a claim, allowing one truthful review-only trailing slash."""

    return _path_anchor(
        pattern,
        allow_nonaccepted_directory_marker=claim.epistemic_status != "accepted",
    )


def path_view_relative(anchor: str) -> str:
    """Return the collision-safe physical path for one logical anchor."""

    if not anchor:
        return "by-path/constraints.md"
    if not _canonical_repository_path(anchor):
        raise ValueError(f"unsafe path-view anchor: {anchor!r}")
    components = tuple(
        _escape_consumer_component(item) for item in PurePosixPath(anchor).parts
    )
    if any(
        len(item.encode("utf-8")) > MAX_PHYSICAL_PATH_COMPONENT_BYTES
        for item in components
    ):
        raise ValueError("path-view anchor component exceeds physical bound")
    relative = PurePosixPath(
        "by-path", *components, "constraints.md"
    ).as_posix()
    if len(relative.encode("utf-8")) > MAX_CONSUMER_RELATIVE_PATH_BYTES:
        raise ValueError("path-view relative path exceeds consumer bound")
    return relative


def claims_by_anchor(
    claims: Sequence[CanonicalClaim],
) -> dict[str, tuple[CanonicalClaim, ...]]:
    """Group routed claims by anchor with per-view claim-ID deduplication."""

    routed: dict[str, dict[str, CanonicalClaim]] = {}
    for claim in claims:
        for scope in claim.semantics.applies_to:
            anchor = claim_path_anchor(claim, scope)
            relative = path_view_relative(anchor)
            if len(relative.encode("utf-8")) > MAX_CONSUMER_RELATIVE_PATH_BYTES:
                raise ValueError("consumer path-view path exceeds frozen bound")
            existing = routed.setdefault(anchor, {}).get(claim.claim_id)
            if existing is not None and existing != claim:
                raise ValueError("one claim ID maps to inconsistent canonical records")
            routed[anchor][claim.claim_id] = claim
    return {
        anchor: tuple(items[key] for key in sorted(items))
        for anchor, items in sorted(routed.items())
    }


def _code_literal(value: str) -> str:
    if not _safe_display_text(value) or "\n" in value:
        raise ValueError("inline code literal must be safe and single-line")
    longest = max(
        (len(item.group(0)) for item in _BACKTICK_RUN.finditer(value)),
        default=0,
    )
    fence = "`" * max(1, longest + 1)
    padding = (
        " "
        if value.startswith(("`", " ")) or value.endswith(("`", " "))
        else ""
    )
    return f"{fence}{padding}{value}{padding}{fence}"


def _dynamic_text_fence(value: str, *, indent: str = "") -> list[str]:
    if not _safe_display_text(value):
        raise ValueError("consumer display text contains a forbidden control character")
    longest = max(
        (len(item.group(0)) for item in _BACKTICK_RUN.finditer(value)),
        default=0,
    )
    fence = "`" * max(3, longest + 1)
    return [
        f"{indent}{fence}text",
        *(f"{indent}{line}" for line in value.split("\n")),
        f"{indent}{fence}",
    ]


def _labeled_scalar(
    label: str,
    value: str | None,
    *,
    absent: str | None = None,
) -> list[str]:
    if value is None:
        if absent is None:
            raise ValueError(f"absent {label} has no explicit rendering")
        return [f"- **{label}:** {_code_literal(absent)}"]
    if not _safe_display_text(value):
        raise ValueError("consumer display text contains a forbidden control character")
    if "\n" not in value and not re.search(r"[\\`*_\[\]<>|&~]", value):
        return [f"- **{label}:** {value}"]
    return [f"- **{label}:**", *_dynamic_text_fence(value, indent="  ")]


def _detail_link(containing_relative: str, claim_id: str) -> str:
    if _SAFE_CLAIM_ID.fullmatch(claim_id) is None:
        raise ValueError("unsafe claim ID in detail link")
    parent = PurePosixPath(containing_relative).parent.as_posix()
    start = "." if parent == "." else parent
    target = f"claims/{claim_id}.md"
    link = posixpath.relpath(target, start=start)
    if link.startswith("/") or posixpath.normpath(posixpath.join(start, link)) != target:
        raise ValueError("generated detail link escapes the consumer tree")
    return quote(link, safe="/._-")


def _semantic_field_lines(
    claim: CanonicalClaim,
    *,
    containing_relative: str | None,
) -> list[str]:
    if claim.epistemic_status == "rejected":
        raise ValueError("rejected claim cannot be rendered in a consumer view")
    lines: list[str] = []
    if claim.epistemic_status == "review":
        if not claim.review_facets:
            raise ValueError("review claim requires typed uncertainty facets")
        lines.append("- **Status:** Unconfirmed (review)")
        lines.append("- **Uncertainty:**")
        for facet in claim.review_facets:
            explanation = _REVIEW_EXPLANATIONS.get(facet)
            if explanation is None:
                raise ValueError(f"review facet has no frozen explanation: {facet}")
            lines.append(f"  - {explanation}")
    lines.extend(_labeled_scalar("Rule", claim.semantics.rule))
    lines.append("- **Applies to:**")
    lines.extend(
        f"  - {_code_literal(item)}" for item in claim.semantics.applies_to
    )
    if claim.semantics.condition is not None:
        lines.extend(_labeled_scalar("Condition", claim.semantics.condition))
    if claim.semantics.exception is not None:
        lines.extend(_labeled_scalar("Exception", claim.semantics.exception))
    strength_label = (
        "Normative strength"
        if claim.epistemic_status == "accepted"
        else "Proposed normative strength"
    )
    lines.append(
        f"- **{strength_label}:** {_code_literal(claim.normative_strength)}"
    )
    if (
        claim.epistemic_status == "accepted"
        and claim.delivery_policy == "reference_only"
    ):
        lines.append(f"- **Use:** {_code_literal('reference_only')}")
    if claim.execution_risk in {
        "repository_wide_mutation",
        "external_side_effect",
    }:
        lines.append(
            f"- **Execution risk:** {_code_literal(claim.execution_risk)}"
        )
    if containing_relative is not None:
        link = _detail_link(containing_relative, claim.claim_id)
        lines.append(f"- **Detail:** [Open claim details]({link})")
    return lines


def render_claim_card(
    claim: CanonicalClaim,
    *,
    containing_relative: str,
) -> str:
    """Render one exact conditionally populated compact card."""

    text = "\n".join(
        [
            "## Rule",
            "",
            *_semantic_field_lines(
                claim,
                containing_relative=containing_relative,
            ),
        ]
    ) + "\n"
    size = len(text.encode("utf-8"))
    if size > MAX_RENDERED_COMPACT_CARD_BYTES:
        raise ValueError(
            "rendered compact card exceeds frozen preflight bound: "
            f"{claim.claim_id} {size}>{MAX_RENDERED_COMPACT_CARD_BYTES}"
        )
    return text


def render_catalog_entry(
    claim: CanonicalClaim,
    *,
    containing_relative: str,
    review_pending: bool | None = None,
) -> str:
    """Render an index entry retaining full rule and applicability semantics."""
    if claim.epistemic_status == "rejected":
        raise ValueError("rejected claim cannot be rendered in a consumer view")
    lines = ["## Rule", ""]
    pending = (
        claim.epistemic_status == "review"
        if review_pending is None
        else review_pending
    )
    if pending:
        lines.append("- **Status:** Unconfirmed (review)")
    lines.extend(_labeled_scalar("Rule", claim.semantics.rule))
    lines.append("- **Applies to:**")
    lines.extend(f"  - {_code_literal(item)}" for item in claim.semantics.applies_to)
    if claim.semantics.condition is not None:
        lines.extend(_labeled_scalar("Condition", claim.semantics.condition))
    if claim.semantics.exception is not None:
        lines.extend(_labeled_scalar("Exception", claim.semantics.exception))
    strength_label = "Proposed normative strength" if pending else "Normative strength"
    lines.append(f"- **{strength_label}:** {_code_literal(claim.normative_strength)}")
    if claim.delivery_policy == "reference_only":
        lines.append("- **Use:** `reference_only`")
    if claim.execution_risk in {"repository_wide_mutation", "external_side_effect"}:
        lines.append(f"- **Execution risk:** {_code_literal(claim.execution_risk)}")
    link = _detail_link(containing_relative, claim.claim_id)
    lines.append(f"- [Read applicability and details]({link})")
    text = "\n".join(lines) + "\n"
    if len(text.encode("utf-8")) > MAX_RENDERED_COMPACT_CARD_BYTES:
        raise ValueError("rendered compact card exceeds frozen preflight bound")
    return text


def _render_card_file(heading: str, cards: Sequence[str]) -> bytes:
    text = heading + "\n\n"
    if cards:
        text += "\n\n".join(item.rstrip("\n") for item in cards) + "\n"
    return text.encode("utf-8")


def _catalog_files(
    claims: Sequence[CanonicalClaim],
) -> tuple[dict[str, bytes], int]:
    catalog_cards = [
        render_catalog_entry(item, containing_relative="catalog.md")
        for item in claims
    ]
    unsharded = _render_card_file("# Norm4Code Claim Catalog", catalog_cards)
    if len(unsharded) <= MAX_CATALOG_FILE_BYTES:
        return {"catalog.md": unsharded}, 0

    shard_cards = [
        render_catalog_entry(item, containing_relative="catalog/0001.md")
        for item in claims
    ]
    groups: list[list[str]] = []
    current: list[str] = []
    for card in shard_cards:
        ordinal = len(groups) + 1
        if ordinal > MAX_CATALOG_SHARDS:
            raise ValueError("consumer catalog requires more than 9,999 shards")
        heading = f"# Norm4Code Claim Catalog — {ordinal:04d}"
        candidate = [*current, card]
        if len(_render_card_file(heading, candidate)) <= MAX_CATALOG_FILE_BYTES:
            current = candidate
            continue
        if not current:
            raise ValueError("one compact card cannot fit a bounded catalog shard")
        groups.append(current)
        current = [card]
        next_ordinal = len(groups) + 1
        if next_ordinal > MAX_CATALOG_SHARDS:
            raise ValueError("consumer catalog requires more than 9,999 shards")
        next_heading = f"# Norm4Code Claim Catalog — {next_ordinal:04d}"
        if len(_render_card_file(next_heading, current)) > MAX_CATALOG_FILE_BYTES:
            raise ValueError("one compact card cannot fit a bounded catalog shard")
    if current:
        groups.append(current)
    if len(groups) > MAX_CATALOG_SHARDS:
        raise ValueError("consumer catalog requires more than 9,999 shards")

    files: dict[str, bytes] = {}
    index_lines = ["# Norm4Code Claim Catalog", ""]
    claim_offset = 0
    for ordinal, group in enumerate(groups, start=1):
        relative = f"catalog/{ordinal:04d}.md"
        heading = f"# Norm4Code Claim Catalog — {ordinal:04d}"
        shard = _render_card_file(heading, group)
        if len(shard) > MAX_CATALOG_FILE_BYTES:
            raise ValueError("consumer catalog shard exceeds frozen byte bound")
        files[relative] = shard
        selected = claims[claim_offset : claim_offset + len(group)]
        if not selected:
            raise ValueError("empty catalog shard")
        scope = selected[0].semantics.applies_to[0]
        scope_preview = scope[:80] + ("…" if len(scope) > 80 else "")
        index_lines.append(
            f"- [Rules {claim_offset + 1}–{claim_offset + len(selected)}]({relative})"
            f" — scope preview: {_code_literal(scope_preview)}"
        )
        claim_offset += len(group)
    if claim_offset != len(claims):
        raise ValueError("catalog shard accounting differs from selected claims")
    index = ("\n".join(index_lines).rstrip() + "\n").encode("utf-8")
    if len(index) > MAX_CATALOG_FILE_BYTES:
        raise ValueError("consumer catalog shard index exceeds frozen byte bound")
    files["catalog.md"] = index
    return dict(sorted(files.items())), len(groups)


def _safe_markdown_destination(relative: str, *, start: str) -> str:
    if relative.startswith(("/", "\\")) or "\\" in relative:
        raise ValueError("generated Markdown destination is unsafe")
    normalized = posixpath.normpath(posixpath.join(start, relative))
    if normalized.startswith("../") or normalized == "..":
        raise ValueError("generated Markdown destination escapes consumer tree")
    return quote(relative, safe="/._-")


def _render_path_index(
    views: Mapping[str, Sequence[CanonicalClaim]],
) -> bytes:
    lines = ["# Norm4Code Path View Index", ""]
    for anchor, claims in views.items():
        relative = path_view_relative(anchor)
        link = posixpath.relpath(relative, start="by-path")
        encoded = _safe_markdown_destination(link, start="by-path")
        logical = "repository_root" if not anchor else anchor
        lines.append(
            f"- {_code_literal(logical)} — [Open view]({encoded}) — "
            f"{_code_literal(str(len(claims)))} claim(s)"
        )
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _render_path_view(
    anchor: str,
    claims: Sequence[CanonicalClaim],
) -> bytes:
    relative = path_view_relative(anchor)
    logical = "repository_root" if not anchor else anchor
    cards = [
        render_claim_card(item, containing_relative=relative) for item in claims
    ]
    prefix = (
        "# Norm4Code Path View\n\n"
        f"- **Logical anchor:** {_code_literal(logical)}\n\n"
    )
    text = prefix
    if cards:
        text += "\n\n".join(item.rstrip("\n") for item in cards) + "\n"
    return text.encode("utf-8")


def _location_text(evidence: CanonicalEvidenceRef) -> str:
    location = evidence.path
    if evidence.line_start is None:
        if evidence.line_end is not None:
            raise ValueError("canonical evidence location has an orphan line end")
        return location
    if evidence.line_end is None:
        return f"{location}:{evidence.line_start}"
    if evidence.line_end < evidence.line_start:
        raise ValueError("canonical evidence location is reversed")
    if evidence.line_end == evidence.line_start:
        return f"{location}:{evidence.line_start}"
    return f"{location}:{evidence.line_start}-{evidence.line_end}"


@dataclass(frozen=True)
class _ResolvedEvidence:
    ref: CanonicalEvidenceRef
    evidence_kind: str
    content_form: str
    content: str
    location: str
    location_key: tuple[object, ...]
    member_ids: tuple[str, ...]


@dataclass(frozen=True)
class _EvidenceGroup:
    evidence_kind: str
    content_form: str
    content: str
    locations: tuple[str, ...]
    location_keys: tuple[tuple[object, ...], ...]
    item_count: int
    member_ids: frozenset[str]
    population_support: tuple[int, int, bool] | None = None


def _current_evidence_maps(
    bundles: Sequence[CurrentEvidenceBundle],
) -> tuple[
    dict[str, dict[str, tuple[object, object]]],
    dict[str, CurrentEvidenceBundle],
]:
    by_case: dict[str, CurrentEvidenceBundle] = {}
    evidence_by_case: dict[str, dict[str, tuple[object, object]]] = {}
    for bundle in bundles:
        if bundle.case_id in by_case:
            raise ValueError("multiple current-evidence bundles for one case")
        by_case[bundle.case_id] = bundle
        origins = {item.origin_id: item for item in bundle.origins}
        if len(origins) != len(bundle.origins):
            raise ValueError("duplicate current-evidence origin in one case")
        items: dict[str, tuple[object, object]] = {}
        for item in bundle.items:
            if item.evidence_id in items:
                raise ValueError("current evidence ID is duplicated within its case")
            origin = origins.get(item.origin_id)
            if origin is None:
                raise ValueError("current evidence has no case-local origin")
            items[item.evidence_id] = (item, origin)
        evidence_by_case[bundle.case_id] = items
    return evidence_by_case, by_case


def _git_payload_by_id(
    git_artifacts: Mapping[str, object] | None,
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for stable_id, value in (git_artifacts or {}).items():
        if isinstance(value, BaseModel):
            payload = value.model_dump(mode="json")
        elif isinstance(value, Mapping):
            payload = dict(value)
        else:
            raise ValueError(f"Git artifact is not structured: {stable_id}")
        if not all(isinstance(key, str) for key in payload):
            raise ValueError(f"Git artifact has a non-string key: {stable_id}")
        if payload.get("artifact_digest") != artifact_digest(payload):
            raise ValueError(f"Git artifact digest is invalid: {stable_id}")
        result[stable_id] = payload
    return result


def _temporal_fact(payload: Mapping[str, object]) -> str:
    lines = [
        f"Change kind: {payload.get('change_kind')}",
        f"Comparison: {payload.get('comparison_kind')}",
    ]
    before = payload.get("before_path")
    after = payload.get("after_path")
    if isinstance(before, str):
        lines.append(f"Before path: {before}")
    if isinstance(after, str):
        lines.append(f"After path: {after}")
    if payload.get("commit_message_context_status") == "complete":
        context = payload.get("commit_message_context")
        if not isinstance(context, str) or not context:
            raise ValueError("complete temporal context has no bounded text")
        lines.append(f"Commit context: {context}")
    result = "\n".join(lines)
    if not _safe_display_text(result):
        raise ValueError("temporal evidence fact contains unsafe display text")
    return result


def _resolve_evidence(
    evidence: CanonicalEvidenceRef,
    *,
    current_items: Mapping[str, tuple[object, object]],
    git_items: Mapping[str, Mapping[str, object]],
) -> _ResolvedEvidence:
    location = _location_text(evidence)
    location_key = (
        evidence.path,
        evidence.line_start if evidence.line_start is not None else -1,
        evidence.line_end if evidence.line_end is not None else -1,
    )
    if evidence.kind != "temporal":
        bound = current_items.get(evidence.evidence_id)
        if bound is None:
            raise ValueError(
                f"canonical evidence does not resolve case-locally: "
                f"{evidence.evidence_id}"
            )
        item, origin = bound
        if getattr(item, "artifact_ref") != evidence.artifact_ref:
            raise ValueError("canonical evidence artifact binding differs")
        if getattr(item, "origin_id") != evidence.origin_id:
            raise ValueError("canonical evidence origin binding differs")
        if getattr(origin, "origin_id") != evidence.origin_id:
            raise ValueError("canonical evidence origin does not resolve")
        if (
            getattr(origin, "path") != evidence.path
            or getattr(origin, "line_start") != evidence.line_start
            or getattr(origin, "line_end") != evidence.line_end
        ):
            raise ValueError("canonical evidence source location differs")
        if getattr(item, "role") != evidence.role:
            raise ValueError("canonical evidence semantic role differs")
        description = str(getattr(item, "description"))
        if not _safe_display_text(description):
            raise ValueError("canonical evidence text contains unsafe controls")
        origin_kind = str(getattr(origin, "origin_kind"))
        if evidence.kind == "explicit":
            if origin_kind not in _EXPLICIT_ORIGIN_KINDS:
                raise ValueError("explicit canonical evidence has a non-explicit origin")
            if hashlib.sha256(description.encode("utf-8")).hexdigest() != getattr(
                item, "excerpt_digest"
            ):
                raise ValueError("explicit evidence excerpt digest differs")
            content_form = "exact_source_excerpt"
        else:
            if origin_kind in _EXPLICIT_ORIGIN_KINDS:
                raise ValueError("current observation aliases an explicit origin")
            content_form = "deterministic_fact"
        return _ResolvedEvidence(
            ref=evidence,
            evidence_kind=evidence.kind,
            content_form=content_form,
            content=description,
            location=location,
            location_key=location_key,
            member_ids=tuple(getattr(item, "member_ids")),
        )

    payload = git_items.get(evidence.evidence_id)
    if payload is None:
        payload = git_items.get(evidence.artifact_ref.stable_id)
    if payload is None:
        raise ValueError(
            f"canonical temporal evidence does not resolve: {evidence.evidence_id}"
        )
    if payload.get("item_id") != evidence.evidence_id:
        raise ValueError("temporal evidence identity differs")
    if payload.get("artifact_digest") != evidence.artifact_ref.artifact_digest:
        raise ValueError("temporal evidence artifact digest differs")
    if evidence.artifact_ref.stable_id != evidence.evidence_id:
        raise ValueError("temporal canonical ref does not bind its change slice")
    payload_path = payload.get("after_path") or payload.get("before_path")
    if payload_path != evidence.path:
        raise ValueError("temporal canonical evidence path differs")
    return _ResolvedEvidence(
        ref=evidence,
        evidence_kind=evidence.kind,
        content_form="deterministic_temporal_fact",
        content=_temporal_fact(payload),
        location=location,
        location_key=location_key,
        member_ids=(),
    )


def _group_evidence(
    evidence: Sequence[CanonicalEvidenceRef],
    *,
    current_items: Mapping[str, tuple[object, object]],
    git_items: Mapping[str, Mapping[str, object]],
) -> tuple[_EvidenceGroup, ...]:
    identities = [
        (
            item.evidence_id,
            item.origin_id,
            item.role,
            item.kind,
            item.artifact_ref.stable_id,
        )
        for item in evidence
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("canonical evidence list contains a duplicate item")
    resolved = [
        _resolve_evidence(
            item,
            current_items=current_items,
            git_items=git_items,
        )
        for item in evidence
    ]
    grouped: dict[tuple[str, str, str], list[_ResolvedEvidence]] = {}
    for item in resolved:
        grouped.setdefault(
            (item.evidence_kind, item.content_form, item.content), []
        ).append(item)
    groups: list[_EvidenceGroup] = []
    for (kind, content_form, content), items in grouped.items():
        ordered = sorted(
            items,
            key=lambda item: (*item.location_key, item.ref.evidence_id),
        )
        groups.append(
            _EvidenceGroup(
                evidence_kind=kind,
                content_form=content_form,
                content=content,
                locations=tuple(item.location for item in ordered),
                location_keys=tuple(item.location_key for item in ordered),
                item_count=len(ordered),
                member_ids=frozenset(
                    member_id for item in ordered for member_id in item.member_ids
                ),
            )
        )
    groups.sort(
        key=lambda item: (
            item.location_keys[0] if item.location_keys else ("", -1, -1),
            item.evidence_kind,
            hashlib.sha256(item.content.encode("utf-8")).hexdigest(),
            item.content_form,
        )
    )
    return tuple(groups)


def _population_support(
    claim: CanonicalClaim,
    bundle: CurrentEvidenceBundle,
) -> tuple[int, int, str, bool] | None:
    denominators = [
        item
        for item in bundle.denominators
        if item.member_id in set(claim.contributing_member_ids)
        and item.status == "complete"
    ]
    if len(denominators) != 1:
        return None
    denominator = denominators[0]
    if (
        denominator.census_enumeration_status != "complete"
        or denominator.sampling_plan_status != "complete"
    ):
        # A status label alone cannot turn an incomplete census or failed
        # sampling plan into a complete-looking population fraction.
        return None
    eligible = (
        denominator.support
        + denominator.counterexample
        + denominator.unknown
    )
    if eligible < denominator.support:
        raise ValueError("population support exceeds eligible denominator")
    return (
        denominator.support,
        eligible,
        denominator.member_id,
        denominator.sampling_mode == "deterministic_stratified_sample",
    )


def _attach_population_support(
    groups: tuple[_EvidenceGroup, ...],
    support: tuple[int, int, str, bool] | None,
) -> tuple[_EvidenceGroup, ...]:
    if support is None:
        return groups
    numerator, denominator, member_id, sampled = support
    matching = [
        ordinal
        for ordinal, group in enumerate(groups)
        if member_id in group.member_ids
    ]
    if len(matching) == 1:
        ordinal = matching[0]
        updated = list(groups)
        updated[ordinal] = replace(
            groups[ordinal],
            population_support=(numerator, denominator, sampled),
        )
        return tuple(updated)
    # A denominator is claim/member-bound rather than evidence-group-bound.
    # When several exact groups carry that member, choosing one would suggest
    # a more specific relationship than the canonical audit establishes.
    return groups


def _render_evidence_groups(
    heading: str,
    groups: Sequence[_EvidenceGroup],
    *,
    counterevidence: bool,
) -> list[str]:
    if not groups:
        return []
    lines = [f"## {heading}", ""]
    label = "Counterevidence" if counterevidence else "Evidence"
    for ordinal, group in enumerate(groups, start=1):
        lines.extend(
            [
                f"### {label} {ordinal} — {group.evidence_kind}",
                "",
                *_dynamic_text_fence(group.content),
                "",
                "- **Sources:**",
                *(f"  - {_code_literal(item)}" for item in group.locations),
            ]
        )
        lines.append("")
    return lines


def _findings_by_case(
    findings: Sequence[UnifiedCaseFinding],
) -> dict[str, tuple[UnifiedCaseFinding, ...]]:
    grouped: dict[str, list[UnifiedCaseFinding]] = {}
    for finding in findings:
        grouped.setdefault(finding.case_id, []).append(finding)
    return {
        case_id: tuple(
            sorted(
                items,
                key=lambda item: (item.verification_round, item.finding_id),
            )
        )
        for case_id, items in grouped.items()
    }


def _axis_value(field: str, raw: str) -> str | None:
    if field != "temporal_status":
        return raw if raw in {"fail", "uncertain"} else None
    if raw == "adverse_unresolved":
        return "fail"
    if raw in {"uncertain", "not_available"}:
        return "uncertain"
    return None


def _review_limitations(
    claim: CanonicalClaim,
    *,
    current: CurrentEvidenceBundle,
    findings: Sequence[UnifiedCaseFinding],
) -> list[str]:
    lines = ["## Why this remains review", ""]
    for facet in claim.review_facets:
        explanation = _REVIEW_EXPLANATIONS.get(facet)
        if explanation is None:
            raise ValueError(f"review facet has no frozen explanation: {facet}")
        lines.append(f"- {explanation}")
    values_by_axis: dict[str, set[str]] = {
        field: set() for field, _ in _AXIS_LABELS
    }
    member_ids = set(claim.contributing_member_ids)
    for finding in findings:
        if finding.status != "complete":
            continue
        for member in finding.member_findings:
            if member.member_id not in member_ids:
                continue
            for field, _ in _AXIS_LABELS:
                value = _axis_value(field, str(getattr(member, field)))
                if value is not None:
                    values_by_axis[field].add(value)
    for field, label in _AXIS_LABELS:
        values = values_by_axis[field]
        if not values:
            continue
        value = "fail" if "fail" in values else "uncertain"
        lines.append(f"- **{label}:** {_code_literal(value)}")
    required_incomplete = claim.dependency_status == "incomplete" or any(
        item.requirement_mode == "required" and item.status not in _SUCCESS_CAPABILITY
        for item in current.dependencies
    )
    if required_incomplete:
        lines.append("- Required dependency: incomplete")
    if current.status != "complete" and not required_incomplete:
        lines.append("- Evidence coverage remains incomplete.")
    if any(item.status == "failed" for item in findings):
        lines.append("- Required verification did not complete.")
    return lines


def _render_conflict_entry(target: CanonicalClaim) -> list[str]:
    link = quote(f"{target.claim_id}.md", safe="._-")
    prefix = f"- [Conflicting rule]({link}):"
    rule = target.semantics.rule
    return [prefix, *_dynamic_text_fence(rule, indent="  ")]


def _render_detail(
    claim: CanonicalClaim,
    *,
    exposed_ids: set[str],
    claim_by_id: Mapping[str, CanonicalClaim],
    current_items: Mapping[str, tuple[object, object]],
    current: CurrentEvidenceBundle,
    findings: Sequence[UnifiedCaseFinding],
    git_items: Mapping[str, Mapping[str, object]],
) -> bytes:
    if claim.epistemic_status == "rejected":
        raise ValueError("rejected claim cannot have a consumer detail")
    support_groups = _group_evidence(
        claim.support,
        current_items=current_items,
        git_items=git_items,
    )
    support_groups = _attach_population_support(
        support_groups,
        _population_support(claim, current),
    )
    counter_groups = _group_evidence(
        claim.counterevidence,
        current_items=current_items,
        git_items=git_items,
    )
    lines = [
        "# Rule details",
        "",
        *_semantic_field_lines(claim, containing_relative=None),
        "",
    ]
    lines.extend(
        _render_evidence_groups(
            "Supporting evidence",
            support_groups,
            counterevidence=False,
        )
    )
    lines.extend(
        _render_evidence_groups(
            "Counterevidence",
            counter_groups,
            counterevidence=True,
        )
    )
    visible_conflicts = [
        item for item in claim.conflicts_with if item in exposed_ids
    ]
    for conflict_id in claim.conflicts_with:
        if conflict_id not in claim_by_id:
            raise ValueError("canonical conflict target is missing")
    if visible_conflicts:
        lines.extend(["## Material conflicts", ""])
        for conflict_id in sorted(visible_conflicts):
            lines.extend(_render_conflict_entry(claim_by_id[conflict_id]))
        lines.append("")
    if claim.epistemic_status == "review":
        lines.extend(
            _review_limitations(
                claim,
                current=current,
                findings=findings,
            )
        )
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _render_start(
    *,
    repository_name: str,
    profile: ConsumerProfile,
    snapshot_commit: str | None,
    package_status: str,
    projection: ConsumerProjection,
) -> bytes:
    if package_status not in {"complete", "partial"}:
        raise ValueError("invalid package cannot produce a consumer START file")
    lines = [
        "# Repository knowledge",
        "",
        "- [Catalog](catalog.md): browse rules and their scopes.",
        "- [By path](by-path/index.md): browse rules associated with repository paths.",
        "- Follow a rule's detail link for applicability, sources, evidence, and limitations.",
        "",
    ]
    if package_status == "partial":
        lines.extend(["This collection may omit repository conventions.", ""])
    if any(item.epistemic_status == "review" for item in projection.catalog_claims):
        lines.extend([
            "Entries marked **Unconfirmed (review)** are uncertain proposals, not accepted norms.",
            "",
        ])
    return "\n".join(lines).encode("utf-8")


def _tree_directories(files: Mapping[str, bytes]) -> tuple[str, ...]:
    directories = {"by-path", "claims"}
    for relative in files:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError(f"unsafe consumer relative path: {relative}")
        parent = pure.parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return tuple(sorted(directories))


def consumer_tree_digest(
    files: Mapping[str, bytes],
    directories: Sequence[str] | None = None,
) -> str:
    """Digest only task-facing Markdown bytes; ignore manifest and modes.

    ``directories`` remains an accepted compatibility argument, but directory
    metadata deliberately does not enter the static-consumer-surface 3.0 tree
    identity.
    """

    del directories
    markdown: dict[str, tuple[str, int]] = {}
    for relative in sorted(files):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError(f"unsafe consumer relative path: {relative}")
        if pure.suffix != ".md":
            continue
        data = files[relative]
        if not isinstance(data, bytes):
            raise ValueError("consumer tree digest requires byte values")
        data.decode("utf-8", errors="strict")
        markdown[relative] = (
            hashlib.sha256(data).hexdigest(),
            len(data),
        )
    return _consumer_tree_digest_from_metadata(markdown)


def _consumer_tree_digest_from_metadata(
    markdown: Mapping[str, tuple[str, int]],
) -> str:
    return content_digest(
        {
            "markdown_files": [
                {
                    "path": relative,
                    "sha256": metadata[0],
                    "size": metadata[1],
                }
                for relative, metadata in sorted(markdown.items())
            ]
        }
    )


def make_canonical_claims_file(bundle: UnifiedBundle) -> CanonicalClaimsFile:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "format_version": CANONICAL_CLAIMS_FORMAT_VERSION,
        "snapshot_digest": bundle.index.snapshot_digest,
        "snapshot_commit": bundle.index.snapshot_commit,
        "profile_digest": bundle.index.profile_digest,
        "claims": sorted(bundle.canonical_claims, key=lambda item: item.claim_id),
    }
    return CanonicalClaimsFile.model_validate(seal_payload(payload))


def make_canonical_completeness_file(
    bundle: UnifiedBundle,
) -> CanonicalCompletenessFile:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "format_version": CANONICAL_COMPLETENESS_FORMAT_VERSION,
        "snapshot_digest": bundle.index.snapshot_digest,
        "snapshot_commit": bundle.index.snapshot_commit,
        "profile_digest": bundle.index.profile_digest,
        "state": bundle.completeness,
    }
    return CanonicalCompletenessFile.model_validate(seal_payload(payload))


def _aggregate_counts(claims: Sequence[CanonicalClaim]) -> dict[str, object]:
    accepted = [item for item in claims if item.epistemic_status == "accepted"]
    review = [item for item in claims if item.epistemic_status == "review"]
    rejected = [item for item in claims if item.epistemic_status == "rejected"]
    delivered = [
        item
        for item in accepted
        if item.delivery_policy in {"default_consider", "condition_matched_only"}
    ]
    return {
        "accepted": accepted,
        "review": review,
        "rejected": rejected,
        "delivered": delivered,
        "accepted_policy_counts": dict(
            sorted(Counter(item.delivery_policy for item in accepted).items())
        ),
        "accepted_risk_counts": dict(
            sorted(Counter(item.execution_risk for item in accepted).items())
        ),
        "review_facet_counts": dict(
            sorted(
                Counter(
                    facet for item in review for facet in item.review_facets
                ).items()
            )
        ),
    }


def _make_manifest(
    *,
    repository_name: str,
    profile: ConsumerProfile,
    requested_profiles: Sequence[ConsumerProfile],
    canonical: CanonicalClaimsFile,
    completeness: CanonicalCompletenessFile,
    summary: ConsumerProfileSummary,
) -> ConsumerManifest:
    counts = _aggregate_counts(canonical.claims)
    accepted = cast(list[CanonicalClaim], counts["accepted"])
    review = cast(list[CanonicalClaim], counts["review"])
    rejected = cast(list[CanonicalClaim], counts["rejected"])
    delivered = cast(list[CanonicalClaim], counts["delivered"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "format_version": CONSUMER_MANIFEST_VERSION,
        "consumer_surface_version": CONSUMER_SURFACE_VERSION,
        "repository": repository_name,
        "snapshot_digest": canonical.snapshot_digest,
        "snapshot_commit": canonical.snapshot_commit,
        "acquisition_profile_digest": canonical.profile_digest,
        "consumer_profile": profile,
        "canonical_claims_digest": canonical.artifact_digest,
        "completeness_digest": completeness.artifact_digest,
        "package_status": completeness.state.package_status,
        "max_catalog_file_bytes": MAX_CATALOG_FILE_BYTES,
        "max_rendered_compact_card_bytes": MAX_RENDERED_COMPACT_CARD_BYTES,
        "requested_consumer_profiles": list(requested_profiles),
        "emitted_consumer_profiles": list(requested_profiles),
        "canonical_accepted_count": len(accepted),
        "canonical_review_count": len(review),
        "canonical_rejected_count": len(rejected),
        "exposed_accepted_count": summary.exposed_accepted_count,
        "experimentally_exposed_review_count": (
            summary.experimentally_exposed_review_count
        ),
        "delivered_claim_count": len(delivered),
        "accepted_policy_counts": counts["accepted_policy_counts"],
        "accepted_risk_counts": counts["accepted_risk_counts"],
        "review_facet_counts": counts["review_facet_counts"],
        "review_facet_counts_are_overlapping": True,
        "catalog_card_count": summary.catalog_card_count,
        "catalog_shard_count": summary.catalog_shard_count,
        "detail_file_count": summary.detail_file_count,
        "path_view_count": summary.path_view_count,
        "path_card_occurrence_count": summary.path_card_occurrence_count,
        "path_unique_claim_count": summary.path_unique_claim_count,
        "markdown_file_count": summary.markdown_file_count,
        "task_tree_digest": summary.tree_digest,
    }
    return ConsumerManifest.model_validate(seal_payload(payload))


def render_consumer_view(
    *,
    bundle: UnifiedBundle,
    repository_name: str,
    profile: ConsumerProfile,
    canonical: CanonicalClaimsFile,
    completeness: CanonicalCompletenessFile,
    git_artifacts: Mapping[str, object] | None,
    requested_profiles: Sequence[ConsumerProfile] | None = None,
) -> ConsumerViewPlan:
    """Render one isolated surface-3.0 view and its colocated manifest."""

    selected_profile = _consumer_profile(profile)
    if not completeness.state.structurally_valid or (
        completeness.state.package_status == "invalid"
    ):
        raise ValueError("structurally invalid package is not consumer-renderable")
    if (
        canonical.snapshot_digest != completeness.snapshot_digest
        or canonical.snapshot_commit != completeness.snapshot_commit
        or canonical.profile_digest != completeness.profile_digest
    ):
        raise ValueError("canonical claim/completeness identities differ")
    if (
        canonical.snapshot_digest != bundle.index.snapshot_digest
        or canonical.snapshot_commit != bundle.index.snapshot_commit
        or canonical.profile_digest != bundle.index.profile_digest
    ):
        raise ValueError("consumer canonical identity differs from bundle")
    bound_profiles = tuple(
        sorted(set(requested_profiles or (selected_profile,)))
    )
    if any(item not in _PROFILES for item in bound_profiles):
        raise ValueError("consumer manifest binds an unknown requested profile")
    if selected_profile not in bound_profiles:
        raise ValueError("selected consumer profile is absent from emitted set")
    for claim in canonical.claims:
        for scope in claim.semantics.applies_to:
            # Routing safety is a canonical export precondition even when this
            # profile does not expose the claim or its delivery policy keeps it
            # out of path views.  A non-accepted audit item alone may preserve
            # exactly one truthful trailing directory marker.
            claim_path_anchor(claim, scope)
    projection = project_claims(canonical.claims, selected_profile)
    views = claims_by_anchor(projection.path_claims)
    current_items_by_case, current_by_case = _current_evidence_maps(
        bundle.current_evidence
    )
    findings_by_case = _findings_by_case(bundle.findings)
    git_items = _git_payload_by_id(git_artifacts)
    claim_by_id = {item.claim_id: item for item in canonical.claims}
    exposed_ids = {item.claim_id for item in projection.catalog_claims}

    files: dict[str, bytes] = {
        "START.md": _render_start(
            repository_name=repository_name,
            profile=selected_profile,
            snapshot_commit=canonical.snapshot_commit,
            package_status=completeness.state.package_status,
            projection=projection,
        ),
        "by-path/index.md": _render_path_index(views),
    }
    catalog_files, shard_count = _catalog_files(projection.catalog_claims)
    files.update(catalog_files)
    for anchor, claims in views.items():
        relative = path_view_relative(anchor)
        if relative in files:
            raise ValueError(f"consumer path-view collision: {relative}")
        files[relative] = _render_path_view(anchor, claims)
    for claim in projection.detail_claims:
        current_items = current_items_by_case.get(claim.case_id)
        current = current_by_case.get(claim.case_id)
        if current_items is None or current is None:
            raise ValueError("consumer claim has no case-local current evidence")
        relative = f"claims/{claim.claim_id}.md"
        files[relative] = _render_detail(
            claim,
            exposed_ids=exposed_ids,
            claim_by_id=claim_by_id,
            current_items=current_items,
            current=current,
            findings=findings_by_case.get(claim.case_id, ()),
            git_items=git_items,
        )
    files = dict(sorted(files.items()))
    tree_digest = consumer_tree_digest(files)
    path_occurrences = sum(len(items) for items in views.values())
    summary = ConsumerProfileSummary(
        profile=selected_profile,
        exposed_accepted_count=sum(
            item.epistemic_status == "accepted"
            for item in projection.catalog_claims
        ),
        experimentally_exposed_review_count=sum(
            item.epistemic_status == "review"
            for item in projection.catalog_claims
        ),
        catalog_card_count=len(projection.catalog_claims),
        catalog_shard_count=shard_count,
        detail_file_count=len(projection.detail_claims),
        path_view_count=len(views),
        path_card_occurrence_count=path_occurrences,
        path_unique_claim_count=len(
            {item.claim_id for item in projection.path_claims}
        ),
        markdown_file_count=len(
            [relative for relative in files if relative.endswith(".md")]
        ),
        tree_digest=tree_digest,
    )
    manifest = _make_manifest(
        repository_name=repository_name,
        profile=selected_profile,
        requested_profiles=cast(Sequence[ConsumerProfile], bound_profiles),
        canonical=canonical,
        completeness=completeness,
        summary=summary,
    )
    files["consumer-manifest.json"] = canonical_json_text(manifest).encode("utf-8")
    files = dict(sorted(files.items()))
    directories = _tree_directories(files)
    return ConsumerViewPlan(
        profile=selected_profile,
        files=files,
        directories=directories,
        summary=summary,
        manifest=manifest,
    )


def build_consumer_artifacts(
    *,
    bundle: UnifiedBundle,
    repository_name: str,
    profiles: Sequence[str],
    git_artifacts: Mapping[str, object] | None,
) -> ConsumerArtifacts:
    """Build canonical wrappers, isolated views, and the delivery report."""

    normalized = normalize_consumer_profiles(profiles)
    canonical = make_canonical_claims_file(bundle)
    completeness = make_canonical_completeness_file(bundle)
    plans = {
        profile: render_consumer_view(
            bundle=bundle,
            repository_name=repository_name,
            profile=profile,
            canonical=canonical,
            completeness=completeness,
            git_artifacts=git_artifacts,
            requested_profiles=normalized,
        )
        for profile in normalized
    }
    # Every published canonical package promises that either frozen profile
    # can be exported later without provider calls.  Exercise an unrequested
    # experimental projection in memory so unsafe review text or an oversized
    # review card cannot hide behind the conservative default view.
    for profile in _PROFILES:
        if profile not in plans:
            render_consumer_view(
                bundle=bundle,
                repository_name=repository_name,
                profile=profile,
                canonical=canonical,
                completeness=completeness,
                git_artifacts=git_artifacts,
                requested_profiles=(profile,),
            )
    counts = _aggregate_counts(canonical.claims)
    accepted = cast(list[CanonicalClaim], counts["accepted"])
    review = cast(list[CanonicalClaim], counts["review"])
    rejected = cast(list[CanonicalClaim], counts["rejected"])
    delivered = cast(list[CanonicalClaim], counts["delivered"])
    delivered_ids = {item.claim_id for item in delivered}
    exclusions: list[DeliveryExclusion] = []
    for claim in canonical.claims:
        if claim.claim_id in delivered_ids:
            continue
        reason = (
            "delivery_policy_reference_only"
            if claim.epistemic_status == "accepted"
            else f"epistemic_status_{claim.epistemic_status}"
        )
        exclusions.append(
            DeliveryExclusion(
                claim_id=claim.claim_id,
                epistemic_status=claim.epistemic_status,
                reason=cast(
                    Literal[
                        "delivery_policy_reference_only",
                        "epistemic_status_review",
                        "epistemic_status_rejected",
                    ],
                    reason,
                ),
            )
        )
    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "format_version": CANONICAL_DELIVERY_REPORT_FORMAT_VERSION,
        "consumer_manifest_version": CONSUMER_MANIFEST_VERSION,
        "consumer_surface_version": CONSUMER_SURFACE_VERSION,
        "max_catalog_file_bytes": MAX_CATALOG_FILE_BYTES,
        "max_rendered_compact_card_bytes": MAX_RENDERED_COMPACT_CARD_BYTES,
        "snapshot_digest": canonical.snapshot_digest,
        "snapshot_commit": canonical.snapshot_commit,
        "profile_digest": canonical.profile_digest,
        "canonical_claims_digest": canonical.artifact_digest,
        "completeness_digest": completeness.artifact_digest,
        "accepted_claim_count": len(accepted),
        "review_claim_count": len(review),
        "rejected_claim_count": len(rejected),
        "delivered_claim_count": len(delivered),
        "delivered_claim_ids": sorted(delivered_ids),
        "accepted_policy_counts": counts["accepted_policy_counts"],
        "accepted_risk_counts": counts["accepted_risk_counts"],
        "review_facet_counts": counts["review_facet_counts"],
        "review_facet_counts_are_overlapping": True,
        "exclusions": sorted(exclusions, key=lambda item: item.claim_id),
        "requested_consumer_profiles": list(normalized),
        "emitted_consumer_profiles": list(normalized),
        "consumer_profiles": [plans[item].summary for item in normalized],
        "consumer_manifest_digests": {
            item: plans[item].manifest.artifact_digest for item in normalized
        },
    }
    report = CanonicalDeliveryReport.model_validate(seal_payload(report_payload))
    return ConsumerArtifacts(
        canonical_claims=canonical,
        completeness=completeness,
        delivery_report=report,
        views=plans,
    )


@dataclass(frozen=True)
class _StandaloneRenderInputs:
    bundle: UnifiedBundle
    config: BuildConfig
    repository_name: str
    canonical: CanonicalClaimsFile
    completeness: CanonicalCompletenessFile
    delivery_report: CanonicalDeliveryReport
    git_artifacts: Mapping[str, object]
    authenticated_files: Mapping[str, tuple[str, int]]
    authenticated_directories: frozenset[str]


def _overlaps(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _resolve_input_directory(path: Path, *, label: str) -> Path:
    """Resolve one caller input while rejecting a symlink at its root."""

    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is unavailable: {error}") from error
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory")
    return resolved


def _load_standalone_inputs(source_run: Path) -> _StandaloneRenderInputs:
    """Load a bounded authenticated renderer subset from an immutable run."""

    # Import lazily: the full validator imports this projection module.
    from repo_constraint_memory_v3.unified.git_evidence import (
        GitChangeSlice,
        GitQueryManifest,
        GitTemporalBundle,
    )
    from repo_constraint_memory_v3.unified.models import (
        CurrentEvidenceBundle,
        UnifiedCaseFinding,
        UnifiedIndex,
    )
    from repo_constraint_memory_v3.unified.validation import (
        _snapshot_bytes,
        _snapshot_tree,
    )

    tree, errors = _snapshot_tree(source_run, max_artifact_bytes=1_000_000_000)
    if tree is None or errors:
        raise ValueError(
            "source run changed during authenticated loading: "
            + "; ".join(sorted(set(errors)))
        )

    def model_file(relative: str, model: type[StrictModel]) -> StrictModel:
        data = _snapshot_bytes(tree, relative, errors)
        if data is None:
            raise ValueError(f"required source-run artifact is missing: {relative}")
        try:
            return model.model_validate_json(data)
        except ValueError as error:
            raise ValueError(
                f"invalid source-run artifact {relative}: {error}"
            ) from error

    config = model_file("config.json", BuildConfig)
    canonical = model_file("canonical/claims.json", CanonicalClaimsFile)
    completeness = model_file(
        "canonical/completeness.json", CanonicalCompletenessFile
    )
    delivery_report = model_file(
        "canonical/delivery_report.json", CanonicalDeliveryReport
    )
    index = model_file("canonical/audit/index.json", UnifiedIndex)
    assert isinstance(config, BuildConfig)
    assert isinstance(canonical, CanonicalClaimsFile)
    assert isinstance(completeness, CanonicalCompletenessFile)
    assert isinstance(delivery_report, CanonicalDeliveryReport)
    assert isinstance(index, UnifiedIndex)
    if tree.total_bytes > config.unified.max_unified_artifact_bytes:
        raise ValueError(
            "source run exceeds configured unified artifact byte bound: "
            f"{tree.total_bytes} > "
            f"{config.unified.max_unified_artifact_bytes}"
        )
    if [item.claim_id for item in canonical.claims] != index.canonical_claim_ids:
        raise ValueError("standalone canonical claims differ from unified index")
    canonical_counts = Counter(
        item.epistemic_status for item in canonical.claims
    )
    if (
        canonical_counts["accepted"] != index.accepted_count
        or canonical_counts["review"] != index.review_count
        or canonical_counts["rejected"] != index.rejected_count
    ):
        raise ValueError("standalone canonical status counts differ from unified index")

    default_manifest = model_file(
        "consumers/accepted_only/consumer-manifest.json",
        ConsumerManifest,
    )
    assert isinstance(default_manifest, ConsumerManifest)
    repository_name = default_manifest.repository

    def collection(relative: str, model: type[StrictModel]) -> list[StrictModel]:
        prefix = relative + "/"
        values: list[StrictModel] = []
        for path in sorted(
            item
            for item in tree.files
            if item.startswith(prefix) and item.endswith(".json")
        ):
            values.append(model_file(path, model))
        return values

    current = collection(
        "canonical/audit/current-evidence", CurrentEvidenceBundle
    )
    findings = collection("canonical/audit/findings", UnifiedCaseFinding)
    current_paths = {
        item
        for item in tree.files
        if item.startswith("canonical/audit/current-evidence/")
        and item.endswith(".json")
    }
    expected_current_paths = {
        f"canonical/audit/current-evidence/{item.bundle_id}.json"
        for item in current
    }
    finding_paths = {
        item
        for item in tree.files
        if item.startswith("canonical/audit/findings/")
        and item.endswith(".json")
    }
    expected_finding_paths = {
        f"canonical/audit/findings/{item.finding_id}.json"
        for item in findings
    }
    if current_paths != expected_current_paths:
        raise ValueError(
            "current-evidence physical collection contains duplicate/foreign artifacts"
        )
    if finding_paths != expected_finding_paths:
        raise ValueError(
            "finding physical collection contains duplicate/foreign artifacts"
        )
    current_ids = [item.bundle_id for item in current]
    current_case_ids = [item.case_id for item in current]
    finding_ids = [item.finding_id for item in findings]
    if len(current_ids) != len(set(current_ids)) or len(current_case_ids) != len(
        set(current_case_ids)
    ):
        raise ValueError("current-evidence identities/cases must be unique")
    if len(finding_ids) != len(set(finding_ids)):
        raise ValueError("finding identities must be unique")
    if sorted(finding_ids) != index.finding_ids:
        raise ValueError("standalone finding collection differs from unified index")
    if any(
        item.snapshot_digest != index.snapshot_digest
        or item.profile_digest != index.profile_digest
        for item in [*current, *findings]
    ):
        raise ValueError("standalone audit artifact identity differs from unified index")
    current_case_counts = Counter(current_case_ids)
    if any(
        current_case_counts.get(claim.case_id, 0) != 1
        for claim in canonical.claims
    ):
        raise ValueError(
            "canonical claim case does not have exactly one current-evidence bundle"
        )

    refs_data = _snapshot_bytes(tree, "canonical/audit/git/refs.json", errors)
    if refs_data is None:
        raise ValueError("canonical Git reference index is missing")
    try:
        refs_raw = json.loads(refs_data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("canonical Git reference index is invalid") from error
    if (
        not isinstance(refs_raw, dict)
        or set(refs_raw) != {"schema_version", "snapshot_digest", "refs"}
        or refs_raw.get("schema_version") != SCHEMA_VERSION
        or refs_raw.get("snapshot_digest") != index.snapshot_digest
        or not isinstance(refs_raw.get("refs"), list)
    ):
        raise ValueError("canonical Git reference index has no refs list")
    model_by_kind: dict[str, type[StrictModel]] = {
        "git_temporal_bundle": GitTemporalBundle,
        "git_manifest": GitQueryManifest,
        "git_change_slice": GitChangeSlice,
    }
    git_artifacts: dict[str, object] = {}
    for raw_ref in refs_raw["refs"]:
        try:
            ref = ImmutableArtifactRef.model_validate(raw_ref)
        except ValueError as error:
            raise ValueError("canonical Git reference is malformed") from error
        stable_id = ref.stable_id
        kind = ref.kind
        relative = ref.artifact_path
        if kind not in model_by_kind:
            raise ValueError("canonical Git reference is malformed")
        if stable_id in git_artifacts:
            raise ValueError("canonical Git references duplicate a stable ID")
        artifact = model_file(relative, model_by_kind[kind])
        identity_field = {
            "git_temporal_bundle": "bundle_id",
            "git_manifest": "manifest_id",
            "git_change_slice": "item_id",
        }[kind]
        expected_directory = {
            "git_temporal_bundle": "bundles",
            "git_manifest": "manifests",
            "git_change_slice": "slices",
        }[kind]
        expected_path = (
            f"canonical/audit/git/{expected_directory}/{stable_id}.json"
        )
        if (
            relative != expected_path
            or getattr(artifact, identity_field) != stable_id
            or getattr(artifact, "snapshot_digest") != index.snapshot_digest
            or getattr(artifact, "snapshot_digest") != ref.snapshot_digest
            or getattr(artifact, "artifact_digest") != ref.artifact_digest
        ):
            raise ValueError("canonical Git reference differs from loaded artifact")
        git_artifacts[stable_id] = artifact
    if errors:
        raise ValueError(
            "source run changed during authenticated loading: "
            + "; ".join(sorted(set(errors)))
        )

    identity = (
        index.snapshot_digest,
        index.snapshot_commit,
        index.profile_digest,
    )
    if (
        (
            canonical.snapshot_digest,
            canonical.snapshot_commit,
            canonical.profile_digest,
        )
        != identity
        or (
            completeness.snapshot_digest,
            completeness.snapshot_commit,
            completeness.profile_digest,
        )
        != identity
        or (
            delivery_report.snapshot_digest,
            delivery_report.snapshot_commit,
            delivery_report.profile_digest,
        )
        != identity
    ):
        raise ValueError("standalone source identities do not agree")
    if (
        default_manifest.snapshot_digest != canonical.snapshot_digest
        or default_manifest.snapshot_commit != canonical.snapshot_commit
        or default_manifest.acquisition_profile_digest != canonical.profile_digest
        or default_manifest.canonical_claims_digest != canonical.artifact_digest
        or default_manifest.completeness_digest != completeness.artifact_digest
    ):
        raise ValueError("source consumer manifest identity differs from canonical data")
    if (
        delivery_report.canonical_claims_digest != canonical.artifact_digest
        or delivery_report.completeness_digest != completeness.artifact_digest
        or delivery_report.consumer_manifest_digests.get("accepted_only")
        != default_manifest.artifact_digest
    ):
        raise ValueError("source delivery report differs from canonical/view identity")

    bundle = UnifiedBundle.model_construct(
        index=index,
        current_evidence=current,
        findings=findings,
        canonical_claims=canonical.claims,
        completeness=completeness.state,
    )
    return _StandaloneRenderInputs(
        bundle=bundle,
        config=config,
        repository_name=repository_name,
        canonical=canonical,
        completeness=completeness,
        delivery_report=delivery_report,
        git_artifacts=git_artifacts,
        authenticated_files={
            relative: (metadata.sha256, metadata.size)
            for relative, metadata in sorted(tree.files.items())
        },
        authenticated_directories=tree.directories,
    )


def _source_files_unchanged(
    source_run: Path,
    authenticated: Mapping[str, tuple[str, int]],
    authenticated_directories: frozenset[str],
) -> list[str]:
    from repo_constraint_memory_v3.unified.validation import _snapshot_tree

    tree, errors = _snapshot_tree(source_run, max_artifact_bytes=1_000_000_000)
    if tree is None:
        return sorted(set(errors))
    expected_files = set(authenticated)
    actual_files = set(tree.files)
    if actual_files != expected_files:
        errors.append(
            "source-run file set changed after authenticated loading: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"foreign={sorted(actual_files - expected_files)}"
        )
    if tree.directories != authenticated_directories:
        errors.append(
            "source-run directory set changed after authenticated loading: "
            f"missing={sorted(authenticated_directories - tree.directories)}, "
            f"foreign={sorted(tree.directories - authenticated_directories)}"
        )
    for relative, expected in authenticated.items():
        actual = tree.files.get(relative)
        if actual is None or (actual.sha256, actual.size) != expected:
            errors.append(f"source-run artifact changed: {relative}")
    return sorted(set(errors))


def _repository_snapshot_errors(
    inputs: _StandaloneRenderInputs,
    repository_root: Path,
) -> list[str]:
    """Check the renderer's fixed-snapshot binding without artifact replay."""

    # Import lazily because the builder imports this projection module through
    # the unified exporter.
    from repo_constraint_memory_v3.discovery import (
        DiscoveryError,
        discover_repository,
    )

    try:
        current = discover_repository(
            repository_root.resolve(),
            inputs.config.discovery,
            repository_name=inputs.config.repository.name,
        )
    except (DiscoveryError, OSError, ValueError) as error:
        return [f"repository snapshot check failed: {error}"]
    expected = (
        inputs.canonical.snapshot_digest,
        inputs.canonical.snapshot_commit,
        inputs.repository_name,
    )
    actual = (
        current.repository.snapshot_digest,
        current.repository.git_head_for_reference,
        current.repository.name,
    )
    if actual != expected:
        return ["repository snapshot/HEAD identity differs from source run"]
    return []


def _compare_consumer_directory(
    view_directory: Path,
    plan: ConsumerViewPlan,
) -> list[str]:
    """Compare one physical view to its exact closed deterministic plan."""

    from repo_constraint_memory_v3.unified.validation import (
        _snapshot_bytes,
        _snapshot_tree,
    )

    tree, errors = _snapshot_tree(
        view_directory,
        max_artifact_bytes=max(
            1,
            sum(len(item) for item in plan.files.values()) + 1,
        ),
    )
    if tree is None:
        return sorted(set(errors))
    actual_files = set(tree.files)
    expected_files = set(plan.files)
    if actual_files != expected_files:
        errors.append(
            "consumer closed-tree file set differs from deterministic replay: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"foreign={sorted(actual_files - expected_files)}"
        )
    if set(tree.directories) != set(plan.directories):
        errors.append(
            "consumer closed-tree directory set differs from deterministic replay: "
            f"missing={sorted(set(plan.directories) - set(tree.directories))}, "
            f"foreign={sorted(set(tree.directories) - set(plan.directories))}"
        )
    for relative, expected in plan.files.items():
        if _snapshot_bytes(tree, relative, errors) != expected:
            errors.append(f"consumer generated bytes differ from replay: {relative}")
    return sorted(set(errors))


def _view_manifest(view_directory: Path) -> tuple[ConsumerManifest | None, list[str]]:
    from repo_constraint_memory_v3.unified.validation import (
        _snapshot_bytes,
        _snapshot_tree,
    )

    tree, errors = _snapshot_tree(
        view_directory,
        max_artifact_bytes=1_000_000_000,
    )
    if tree is None:
        return None, sorted(set(errors))
    data = _snapshot_bytes(tree, "consumer-manifest.json", errors)
    if data is None:
        errors.append("consumer manifest is missing")
        return None, sorted(set(errors))
    try:
        manifest = ConsumerManifest.model_validate_json(data)
    except ValueError as error:
        errors.append(f"consumer manifest failed validation: {error}")
        return None, sorted(set(errors))
    markdown_files = {
        relative: (metadata.sha256, metadata.size)
        for relative, metadata in tree.files.items()
        if relative.endswith(".md")
    }
    if (
        not errors
        and _consumer_tree_digest_from_metadata(markdown_files)
        != manifest.task_tree_digest
    ):
        errors.append("consumer manifest task-tree digest differs from Markdown tree")
    if len(markdown_files) != manifest.markdown_file_count:
        errors.append("consumer manifest Markdown count differs from physical tree")
    return manifest, sorted(set(errors))


def _render_standalone_plan(
    inputs: _StandaloneRenderInputs,
    *,
    profile: ConsumerProfile,
    requested_profiles: Sequence[ConsumerProfile],
) -> ConsumerViewPlan:
    return render_consumer_view(
        bundle=inputs.bundle,
        repository_name=inputs.repository_name,
        profile=profile,
        canonical=inputs.canonical,
        completeness=inputs.completeness,
        git_artifacts=inputs.git_artifacts,
        requested_profiles=requested_profiles,
    )


def render_standalone_consumer_view(
    source_run: Path,
    *,
    repository_root: Path,
    profile: str,
) -> ConsumerViewPlan:
    """Load one source run and replay a new isolated view in memory.

    This renderer authenticates exactly the bounded source subset it consumes
    and rechecks that subset afterwards.  It intentionally does not perform the
    independent full offline source replay.
    """

    selected = _consumer_profile(profile)
    source = _resolve_input_directory(source_run, label="source run")
    repository = _resolve_input_directory(
        repository_root,
        label="target repository",
    )
    if _overlaps(source, repository):
        raise ValueError("source run overlaps the target repository")
    inputs = _load_standalone_inputs(source)
    repository_errors = _repository_snapshot_errors(inputs, repository)
    if repository_errors:
        raise ValueError(
            "invalid repository binding: " + "; ".join(repository_errors)
        )
    plan = _render_standalone_plan(
        inputs,
        profile=selected,
        requested_profiles=(selected,),
    )
    changed = _source_files_unchanged(
        source,
        inputs.authenticated_files,
        inputs.authenticated_directories,
    )
    changed.extend(_repository_snapshot_errors(inputs, repository))
    if changed:
        raise ValueError(
            "source run changed after standalone rendering: "
            + "; ".join(sorted(set(changed)))
        )
    return plan


def validate_consumer_view(
    view_directory: Path,
    *,
    source_run: Path,
    repository_root: Path,
    require_complete: bool = False,
) -> list[str]:
    """Validate a view by manifest binding and exact offline regeneration."""

    from repo_constraint_memory_v3.unified.validation import (
        INCOMPLETE_PACKAGE_ERROR,
        validate_unified_directory,
    )

    try:
        source = _resolve_input_directory(source_run, label="source run")
        repository = _resolve_input_directory(
            repository_root,
            label="target repository",
        )
        initial_view_fingerprint = directory_tree_fingerprint(
            view_directory,
            max_bytes=1_000_000_000,
        )
        initial_source_fingerprint = directory_tree_fingerprint(
            source,
            max_bytes=1_000_000_000,
        )
    except ValueError as error:
        return [f"invalid consumer/source tree: {error}"]
    source_errors = validate_unified_directory(
        source,
        repository,
        require_complete=False,
    )
    if source_errors:
        return [f"invalid source run: {item}" for item in source_errors]
    try:
        inputs = _load_standalone_inputs(source)
    except ValueError as error:
        return [f"invalid source run: {error}"]
    manifest, manifest_errors = _view_manifest(view_directory)
    if manifest is None:
        return manifest_errors
    errors = list(manifest_errors)
    if manifest.repository != inputs.repository_name:
        errors.append("consumer manifest repository differs from source run")
    if (
        manifest.snapshot_digest != inputs.canonical.snapshot_digest
        or manifest.snapshot_commit != inputs.canonical.snapshot_commit
        or manifest.acquisition_profile_digest != inputs.canonical.profile_digest
        or manifest.canonical_claims_digest != inputs.canonical.artifact_digest
        or manifest.completeness_digest != inputs.completeness.artifact_digest
        or manifest.package_status != inputs.completeness.state.package_status
    ):
        errors.append("consumer manifest identity differs from source run")
    try:
        source_manifest_digest = (
            inputs.delivery_report.consumer_manifest_digests.get(
                manifest.consumer_profile
            )
        )
        requested_profiles = (
            tuple(inputs.delivery_report.requested_consumer_profiles)
            if source_manifest_digest == manifest.artifact_digest
            else (manifest.consumer_profile,)
        )
        plan = _render_standalone_plan(
            inputs,
            profile=manifest.consumer_profile,
            requested_profiles=requested_profiles,
        )
    except ValueError as error:
        errors.append(f"consumer deterministic replay failed: {error}")
    else:
        errors.extend(_compare_consumer_directory(view_directory, plan))
    errors.extend(_repository_snapshot_errors(inputs, repository))
    try:
        final_source_fingerprint = directory_tree_fingerprint(
            source,
            max_bytes=1_000_000_000,
        )
    except ValueError as error:
        errors.append(f"source run changed after validation: {error}")
    else:
        if final_source_fingerprint != initial_source_fingerprint:
            errors.append("source run changed during validation")
    try:
        final_view_fingerprint = directory_tree_fingerprint(
            view_directory,
            max_bytes=1_000_000_000,
        )
    except ValueError as error:
        errors.append(f"consumer view changed after validation: {error}")
    else:
        if final_view_fingerprint != initial_view_fingerprint:
            errors.append("consumer view changed during validation")
    if require_complete and inputs.completeness.state.package_status != "complete":
        errors.append(INCOMPLETE_PACKAGE_ERROR)
    return sorted(set(errors))


def write_consumer_view(
    plan: ConsumerViewPlan,
    destination: Path,
    *,
    read_only: bool = True,
) -> None:
    """Materialize one plan into an existing empty staging directory."""

    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("consumer staging destination is not a regular directory")
    if any(destination.iterdir()):
        raise ValueError("consumer staging destination is not empty")
    for relative in plan.directories:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError("consumer plan contains an unsafe directory")
        destination.joinpath(*pure.parts).mkdir(parents=True, exist_ok=True)
    for relative, data in plan.files.items():
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError("consumer plan contains an unsafe file")
        path = destination.joinpath(*pure.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if read_only:
            path.chmod(0o444)


def export_consumer_view(
    source_run: Path,
    *,
    repository_root: Path,
    profile: str,
    output_directory: Path,
) -> ConsumerViewPlan:
    """Load, stage, check, and atomically publish a standalone view."""

    selected = _consumer_profile(profile)
    if output_directory.is_symlink():
        raise ValueError("consumer output is a symlink")
    destination = output_directory.resolve(strict=False)
    repository = _resolve_input_directory(
        repository_root,
        label="target repository",
    )
    source = _resolve_input_directory(source_run, label="source run")
    if _overlaps(destination, repository):
        raise ValueError("consumer output overlaps the target repository")
    if _overlaps(destination, source):
        raise ValueError("consumer output overlaps the source run")
    if _overlaps(source, repository):
        raise ValueError("source run overlaps the target repository")
    consume_initial_empty_directory(destination)
    inputs = _load_standalone_inputs(source)
    repository_errors = _repository_snapshot_errors(inputs, repository)
    if repository_errors:
        raise ValueError(
            "invalid repository binding: " + "; ".join(repository_errors)
        )
    plan = _render_standalone_plan(
        inputs,
        profile=selected,
        requested_profiles=(selected,),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.publish.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError(
            f"another consumer export is publishing: {destination}"
        ) from error
    os.close(lock_fd)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    published = False
    try:
        write_consumer_view(plan, staging, read_only=True)
        fingerprint_bound = max(
            1,
            sum(len(item) for item in plan.files.values()) + 1,
        )
        staged_fingerprint = directory_tree_fingerprint(
            staging,
            max_bytes=fingerprint_bound,
        )
        staged_errors = _compare_consumer_directory(staging, plan)
        if staged_errors:
            raise ValueError(
                "refusing to publish invalid consumer view: "
                + "; ".join(staged_errors)
            )
        source_errors = _source_files_unchanged(
            source,
            inputs.authenticated_files,
            inputs.authenticated_directories,
        )
        source_errors.extend(_repository_snapshot_errors(inputs, repository))
        if _overlaps(destination, repository) or _overlaps(
            destination, source
        ):
            source_errors.append(
                "consumer destination overlaps a protected input"
            )
        if source_errors:
            raise ValueError(
                "source run changed before consumer publication: "
                + "; ".join(sorted(set(source_errors)))
            )
        if destination.exists() or destination.is_symlink():
            raise ValueError("consumer output appeared during publication")
        if directory_tree_fingerprint(
            staging,
            max_bytes=fingerprint_bound,
        ) != staged_fingerprint:
            raise ValueError("consumer staging tree changed before publication")
        rename_directory_noreplace(staging, destination)
        published = True
        return plan
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "CANONICAL_CLAIMS_FORMAT_VERSION",
    "CANONICAL_COMPLETENESS_FORMAT_VERSION",
    "CANONICAL_DELIVERY_REPORT_FORMAT_VERSION",
    "CONSUMER_MANIFEST_VERSION",
    "CONSUMER_SURFACE_VERSION",
    "MAX_CATALOG_FILE_BYTES",
    "MAX_RENDERED_COMPACT_CARD_BYTES",
    "CanonicalClaimsFile",
    "CanonicalCompletenessFile",
    "CanonicalDeliveryReport",
    "ConsumerArtifacts",
    "ConsumerManifest",
    "ConsumerProfile",
    "ConsumerProfileSummary",
    "ConsumerProjection",
    "ConsumerViewPlan",
    "build_consumer_artifacts",
    "canonical_json_text",
    "claim_path_anchor",
    "claims_by_anchor",
    "consumer_tree_digest",
    "export_consumer_view",
    "make_canonical_claims_file",
    "make_canonical_completeness_file",
    "normalize_consumer_profiles",
    "path_anchor",
    "path_view_relative",
    "project_claims",
    "render_claim_card",
    "render_consumer_view",
    "render_standalone_consumer_view",
    "validate_consumer_view",
    "write_consumer_view",
]
