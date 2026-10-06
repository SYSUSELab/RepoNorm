"""Pure assembly of mandatory current evidence for a ``CandidateCase``.

The assembler consumes only objects supplied by the two acquisition channels.
It performs no repository reads and no provider calls.  Population counts are
computed before physical-origin deduplication, while evidence origins are
deduplicated by the frozen ``path + location + content digest`` identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    canonical_bytes,
    make_current_evidence_bundle_id,
    make_evidence_id,
    make_evidence_origin_id,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CapabilityDependency,
    CurrentEvidenceBundle,
    CurrentEvidenceItem,
    EvidenceOrigin,
    ImmutableArtifactRef,
    MAX_EXPLICIT_EVIDENCE_DESCRIPTION_CHARS,
    MAX_OTHER_EVIDENCE_DESCRIPTION_CHARS,
    PopulationDenominator,
)

CURRENT_EVIDENCE_BUNDLE_VERSION = "current-evidence-bundle-1.3"

EvidenceRole = Literal[
    "support", "counterevidence", "unknown", "ineligible", "context"
]
PopulationLabel = Literal["support", "counterexample", "unknown", "ineligible"]

_ORIGIN_KIND_PRIORITY = {
    "explicit_span": 0,
    "configuration_atom": 1,
    "command_atom": 2,
    "workflow_atom": 3,
    "observation": 4,
    "population_member": 5,
    "git_change_slice": 6,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_ROLES = frozenset(
    {"support", "counterevidence", "unknown", "ineligible", "context"}
)
_POPULATION_LABELS = frozenset(
    {"support", "counterexample", "unknown", "ineligible"}
)


@dataclass(frozen=True)
class CurrentEvidenceLimits:
    max_items: int
    max_bytes: int

    def __post_init__(self) -> None:
        if self.max_items < 1 or self.max_bytes < 1:
            raise ValueError("current evidence limits must be positive")


@dataclass(frozen=True)
class EvidenceContribution:
    """One already-resolved explicit/current-snapshot evidence occurrence."""

    origin_kind: Literal[
        "explicit_span",
        "configuration_atom",
        "command_atom",
        "workflow_atom",
        "observation",
        "population_member",
    ]
    path: str
    line_start: int | None
    line_end: int | None
    source_content_digest: str
    independence_group: str
    role: EvidenceRole
    member_ids: tuple[str, ...]
    artifact_ref: ImmutableArtifactRef
    excerpt_digest: str
    description: str
    anchor_literals: tuple[str, ...] = ()
    anchor_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class PopulationEvidenceContribution:
    """A classified member occurrence; counts survive origin deduplication."""

    case_member_id: str
    population_id: str
    population_member_id: str
    label: PopulationLabel
    evidence: EvidenceContribution


@dataclass(frozen=True)
class CurrentEvidenceAssemblyResult:
    bundle: CurrentEvidenceBundle
    population_counts: Mapping[str, Mapping[str, int]]


def _safe_path(path: str) -> bool:
    return bool(path) and not (
        path.startswith(("/", "\\"))
        or "\\" in path
        or "://" in path
        or ".." in path.split("/")
    )


def _origin_key(contribution: EvidenceContribution) -> tuple[object, ...]:
    return (
        contribution.path,
        contribution.line_start,
        contribution.line_end,
        contribution.source_content_digest,
    )


def _ref_key(reference: ImmutableArtifactRef) -> tuple[str, ...]:
    return (
        reference.kind,
        reference.stable_id,
        reference.snapshot_digest,
        reference.artifact_path,
        reference.artifact_digest,
    )


def _canonical_refs(
    references: Iterable[ImmutableArtifactRef],
    *,
    snapshot_digest: str,
    allowed_kinds: set[str],
    invalid_reasons: set[str],
    reason_prefix: str,
) -> list[ImmutableArtifactRef]:
    canonical: dict[tuple[str, ...], ImmutableArtifactRef] = {}
    identities: dict[tuple[str, str], tuple[str, ...]] = {}
    for reference in references:
        if reference.snapshot_digest != snapshot_digest:
            invalid_reasons.add(f"{reason_prefix}_snapshot_mismatch")
            continue
        if reference.kind not in allowed_kinds:
            invalid_reasons.add(f"{reason_prefix}_kind_invalid")
            continue
        key = _ref_key(reference)
        identity = (reference.kind, reference.stable_id)
        previous = identities.get(identity)
        if previous is not None and previous != key:
            invalid_reasons.add(f"{reason_prefix}_identity_conflict")
            continue
        identities[identity] = key
        canonical[key] = reference
    return [canonical[key] for key in sorted(canonical)]


def _dependency(
    capability: str,
    *,
    requirement_mode: str,
    status: str,
    reason_codes: Iterable[str] = (),
) -> CapabilityDependency:
    return CapabilityDependency.model_validate(
        {
            "capability": capability,
            "requirement_mode": requirement_mode,
            "status": status,
            "reason_codes": sorted(set(reason_codes)),
        }
    )


def _merge_dependencies(
    supplied: Sequence[CapabilityDependency],
    *,
    actual: Mapping[str, tuple[str, tuple[str, ...]]],
    required_capabilities: Sequence[str],
    invalid_reasons: set[str],
) -> list[CapabilityDependency]:
    by_name: dict[str, CapabilityDependency] = {}
    for item in supplied:
        previous = by_name.get(item.capability)
        if previous is not None and previous != item:
            invalid_reasons.add("capability_dependency_conflict")
            continue
        by_name[item.capability] = item

    for capability, (status, reasons) in actual.items():
        existing = by_name.get(capability)
        mode = existing.requirement_mode if existing is not None else (
            "not_required" if status == "not_required" else "required"
        )
        if existing is None or status not in {
            "complete",
            "bounded_no_match",
            "not_required",
        }:
            by_name[capability] = _dependency(
                capability,
                requirement_mode=mode,
                status=status,
                reason_codes=reasons,
            )

    for capability in sorted(set(required_capabilities)):
        if capability == "current_evidence":
            # The bundle itself satisfies this case-level capability; recording
            # it as its own dependency would be circular.
            continue
        if capability not in by_name:
            by_name[capability] = _dependency(
                capability,
                requirement_mode="required",
                status="failed",
                reason_codes=("missing_capability_dependency",),
            )
    if not by_name:
        by_name["current_evidence_inputs"] = _dependency(
            "current_evidence_inputs",
            requirement_mode="required",
            status="failed",
            reason_codes=("missing_capability_dependency",),
        )
    return [by_name[key] for key in sorted(by_name)]


def _population_label_counts(
    contributions: Sequence[PopulationEvidenceContribution],
    *,
    implicit_member_ids: set[str],
    population_ids: set[str],
    invalid_reasons: set[str],
) -> tuple[
    dict[str, dict[str, int]],
    dict[tuple[str, str], dict[str, int]],
    list[EvidenceContribution],
]:
    labels: dict[tuple[str, str, str], PopulationLabel] = {}
    evidence_by_occurrence: dict[
        tuple[str, str, str], EvidenceContribution
    ] = {}
    for item in contributions:
        if item.case_member_id not in implicit_member_ids:
            invalid_reasons.add("population_member_case_member_invalid")
            continue
        if item.population_id not in population_ids:
            invalid_reasons.add("population_member_population_ref_invalid")
            continue
        if item.label not in _POPULATION_LABELS:
            invalid_reasons.add("population_member_label_invalid")
            continue
        if not item.population_member_id:
            invalid_reasons.add("population_member_id_invalid")
            continue
        if item.evidence.member_ids != (item.case_member_id,):
            invalid_reasons.add("population_evidence_member_binding_invalid")
            continue
        if item.evidence.role != (
            "counterevidence" if item.label == "counterexample" else item.label
        ):
            invalid_reasons.add("population_evidence_role_mismatch")
            continue
        key = (
            item.case_member_id,
            item.population_id,
            item.population_member_id,
        )
        previous = labels.get(key)
        if previous is not None and previous != item.label:
            invalid_reasons.add("population_member_label_conflict")
            continue
        labels[key] = item.label
        previous_evidence = evidence_by_occurrence.get(key)
        if previous_evidence is not None and previous_evidence != item.evidence:
            invalid_reasons.add("population_member_evidence_conflict")
            continue
        evidence_by_occurrence[key] = item.evidence

    counts = {
        member_id: {
            "support": 0,
            "counterexample": 0,
            "unknown": 0,
            "ineligible": 0,
        }
        for member_id in sorted(implicit_member_ids)
    }
    counts_by_population: dict[tuple[str, str], dict[str, int]] = {}
    for (member_id, population_id, _), label in labels.items():
        counts[member_id][label] += 1
        population_counts = counts_by_population.setdefault(
            (member_id, population_id),
            {
                "support": 0,
                "counterexample": 0,
                "unknown": 0,
                "ineligible": 0,
            },
        )
        population_counts[label] += 1
    evidence = [evidence_by_occurrence[key] for key in sorted(evidence_by_occurrence)]
    return counts, counts_by_population, evidence


def _validate_denominators(
    denominators: Sequence[PopulationDenominator],
    *,
    implicit_member_ids: set[str],
    population_refs: Sequence[ImmutableArtifactRef],
    counts_by_population: Mapping[tuple[str, str], Mapping[str, int]],
    invalid_reasons: set[str],
    incomplete_reasons: set[str],
) -> list[PopulationDenominator]:
    by_member: dict[str, PopulationDenominator] = {}
    population_ids = {item.stable_id for item in population_refs}
    for denominator in denominators:
        if denominator.member_id not in implicit_member_ids:
            invalid_reasons.add("population_denominator_member_invalid")
            continue
        if denominator.member_id in by_member:
            invalid_reasons.add("population_denominator_duplicate")
            continue
        by_member[denominator.member_id] = denominator
        if denominator.population_id not in population_ids:
            invalid_reasons.add("population_denominator_ref_missing")
        actual = counts_by_population.get(
            (denominator.member_id, denominator.population_id), {}
        )
        for field_name in ("support", "counterexample", "unknown", "ineligible"):
            if getattr(denominator, field_name) != actual.get(field_name, 0):
                incomplete_reasons.add("population_member_evidence_incomplete")
        if denominator.status != "complete":
            incomplete_reasons.add(f"population_denominator_{denominator.status}")

    missing = implicit_member_ids - set(by_member)
    denominator_keys = {
        (item.member_id, item.population_id) for item in by_member.values()
    }
    if missing or set(counts_by_population) - denominator_keys:
        incomplete_reasons.add("population_denominator_missing")
    return [by_member[key] for key in sorted(by_member)]


def _deduplicate_evidence(
    contributions: Sequence[EvidenceContribution],
    *,
    case_member_ids: set[str],
    snapshot_digest: str,
    invalid_reasons: set[str],
    incomplete_reasons: set[str],
    max_anchors_per_item: int,
) -> tuple[list[EvidenceOrigin], list[CurrentEvidenceItem]]:
    by_origin: dict[tuple[object, ...], list[EvidenceContribution]] = {}
    for contribution in contributions:
        if contribution.origin_kind not in _ORIGIN_KIND_PRIORITY:
            invalid_reasons.add("evidence_origin_kind_invalid")
            continue
        if contribution.role not in _EVIDENCE_ROLES:
            invalid_reasons.add("evidence_role_invalid")
            continue
        if not contribution.independence_group:
            invalid_reasons.add("evidence_independence_group_invalid")
            continue
        if not _SHA256_RE.fullmatch(contribution.source_content_digest):
            invalid_reasons.add("evidence_source_digest_invalid")
            continue
        if not _SHA256_RE.fullmatch(contribution.excerpt_digest):
            invalid_reasons.add("evidence_excerpt_digest_invalid")
            continue
        description_limit = (
            MAX_EXPLICIT_EVIDENCE_DESCRIPTION_CHARS
            if contribution.origin_kind
            in {
                "explicit_span",
                "configuration_atom",
                "command_atom",
                "workflow_atom",
            }
            else MAX_OTHER_EVIDENCE_DESCRIPTION_CHARS
        )
        if (
            not contribution.description
            or len(contribution.description) > description_limit
        ):
            invalid_reasons.add("evidence_description_invalid")
            continue
        if not _safe_path(contribution.path):
            invalid_reasons.add("evidence_origin_path_invalid")
            continue
        if (contribution.line_start is None) != (contribution.line_end is None):
            invalid_reasons.add("evidence_origin_location_incomplete")
            continue
        if (
            contribution.line_start is not None
            and contribution.line_end is not None
            and (
                contribution.line_start < 1
                or contribution.line_end < contribution.line_start
            )
        ):
            invalid_reasons.add("evidence_origin_location_invalid")
            continue
        if not contribution.member_ids or not set(contribution.member_ids) <= (
            case_member_ids
        ):
            invalid_reasons.add("evidence_member_binding_invalid")
            continue
        if contribution.artifact_ref.snapshot_digest != snapshot_digest:
            invalid_reasons.add("evidence_artifact_snapshot_mismatch")
            continue
        by_origin.setdefault(_origin_key(contribution), []).append(contribution)

    # Independence labels describe evidence provenance, but upstream channels
    # can assign different labels to the same immutable physical origin.  Such
    # aliases are not an artifact-integrity failure.  Canonicalize their
    # transitive equivalence instead, so a later origin carrying either alias
    # cannot be counted as an additional independent support source.
    independence_parent = {
        contribution.independence_group: contribution.independence_group
        for values in by_origin.values()
        for contribution in values
    }

    def find_independence(label: str) -> str:
        root = label
        while independence_parent[root] != root:
            root = independence_parent[root]
        while independence_parent[label] != label:
            parent = independence_parent[label]
            independence_parent[label] = root
            label = parent
        return root

    def union_independence(left: str, right: str) -> None:
        left_root = find_independence(left)
        right_root = find_independence(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        independence_parent[high] = low

    for values in by_origin.values():
        labels = sorted({item.independence_group for item in values})
        for label in labels[1:]:
            union_independence(labels[0], label)

    origins: list[EvidenceOrigin] = []
    origin_id_by_key: dict[tuple[object, ...], str] = {}
    for key, values in sorted(by_origin.items(), key=lambda item: repr(item[0])):
        independence_groups = {item.independence_group for item in values}
        canonical_independence_group = min(
            find_independence(label) for label in independence_groups
        )
        origin_kind = min(
            (item.origin_kind for item in values),
            key=lambda item: (_ORIGIN_KIND_PRIORITY[item], item),
        )
        representative = min(
            values,
            key=lambda item: (
                _ORIGIN_KIND_PRIORITY[item.origin_kind],
                item.artifact_ref.stable_id,
                item.role,
                item.member_ids,
            ),
        )
        span = (
            None
            if representative.line_start is None
            else [representative.line_start, representative.line_end]
        )
        origin_id = make_evidence_origin_id(
            snapshot_digest,
            origin_kind,
            representative.path,
            span,
            representative.source_content_digest,
        )
        origin_id_by_key[key] = origin_id
        origins.append(
            EvidenceOrigin(
                origin_id=origin_id,
                origin_kind=origin_kind,
                path=representative.path,
                line_start=representative.line_start,
                line_end=representative.line_end,
                source_content_digest=representative.source_content_digest,
                independence_group=canonical_independence_group,
            )
        )

    grouped_items: dict[
        tuple[str, str, tuple[str, ...]], list[EvidenceContribution]
    ] = {}
    for key, values in by_origin.items():
        origin_id = origin_id_by_key[key]
        for contribution in values:
            artifact_key = _ref_key(contribution.artifact_ref)
            grouped_items.setdefault(
                (origin_id, contribution.role, artifact_key), []
            ).append(contribution)

    items: list[CurrentEvidenceItem] = []
    for (origin_id, role, _), values in sorted(grouped_items.items()):
        member_ids = sorted(
            {member_id for item in values for member_id in item.member_ids}
        )
        excerpts = {item.excerpt_digest for item in values}
        descriptions = {item.description for item in values}
        if len(excerpts) != 1:
            invalid_reasons.add("evidence_excerpt_digest_conflict")
        if len(descriptions) != 1:
            invalid_reasons.add("evidence_description_conflict")
        raw_literals = {
            anchor for item in values for anchor in item.anchor_literals
        }
        raw_symbols = {
            anchor for item in values for anchor in item.anchor_symbols
        }
        anchor_literals = sorted(
            anchor
            for anchor in raw_literals
            if anchor
            and len(anchor) <= 512
            and not any(token in anchor for token in ("\n", "\r", "\x00"))
        )
        anchor_symbols = sorted(
            anchor
            for anchor in raw_symbols
            if anchor
            and len(anchor) <= 256
            and not any(token in anchor for token in ("\n", "\r", "\x00"))
        )
        if len(anchor_literals) != len(raw_literals) or len(anchor_symbols) != len(
            raw_symbols
        ):
            incomplete_reasons.add("current_evidence_anchor_invalid")
        if (
            len(anchor_literals) > max_anchors_per_item
            or len(anchor_symbols) > max_anchors_per_item
        ):
            incomplete_reasons.add("current_evidence_anchor_limit")
        anchor_literals = anchor_literals[:max_anchors_per_item]
        anchor_symbols = anchor_symbols[:max_anchors_per_item]
        representative = min(
            values,
            key=lambda item: (
                item.excerpt_digest,
                item.description,
                item.member_ids,
            ),
        )
        evidence_id = make_evidence_id(
            origin_id,
            role,
            member_ids,
            representative.artifact_ref.stable_id,
        )
        items.append(
            CurrentEvidenceItem(
                evidence_id=evidence_id,
                origin_id=origin_id,
                role=role,
                member_ids=member_ids,
                artifact_ref=representative.artifact_ref,
                excerpt_digest=representative.excerpt_digest,
                description=representative.description,
                anchor_literals=anchor_literals,
                anchor_symbols=anchor_symbols,
            )
        )
    origins.sort(key=lambda item: item.origin_id)
    items.sort(key=lambda item: item.evidence_id)
    return origins, items


def _apply_limits(
    origins: Sequence[EvidenceOrigin],
    items: Sequence[CurrentEvidenceItem],
    *,
    limits: CurrentEvidenceLimits,
    incomplete_reasons: set[str],
) -> tuple[list[EvidenceOrigin], list[CurrentEvidenceItem]]:
    retained = list(items[: limits.max_items])
    if len(retained) != len(items):
        incomplete_reasons.add("current_evidence_item_limit")

    origin_by_id = {item.origin_id: item for item in origins}

    def retained_origins() -> list[EvidenceOrigin]:
        ids = {item.origin_id for item in retained}
        return [origin_by_id[item] for item in sorted(ids)]

    while retained and len(
        canonical_bytes(
            {
                "origins": retained_origins(),
                "items": retained,
            }
        )
    ) > limits.max_bytes:
        retained.pop()
        incomplete_reasons.add("current_evidence_byte_limit")
    limited_origins = retained_origins()
    if not retained and items:
        incomplete_reasons.add("current_evidence_byte_limit")
    return limited_origins, retained


def assemble_current_evidence(
    case: CandidateCase,
    *,
    explicit_evidence: Sequence[EvidenceContribution],
    population_evidence: Sequence[PopulationEvidenceContribution],
    denominators: Sequence[PopulationDenominator],
    population_refs: Sequence[ImmutableArtifactRef],
    falsification_refs: Sequence[ImmutableArtifactRef],
    scope_condition_refs: Sequence[ImmutableArtifactRef],
    dependencies: Sequence[CapabilityDependency],
    limits: CurrentEvidenceLimits,
) -> CurrentEvidenceAssemblyResult:
    """Assemble and validate the mandatory current bundle for ``case``."""

    invalid_reasons: set[str] = set()
    incomplete_reasons: set[str] = set()
    member_ids = {item.member_id for item in case.members}
    explicit_member_ids = {
        item.member_id for item in case.members if item.channel == "explicit"
    }
    implicit_member_ids = {
        item.member_id
        for item in case.members
        if item.channel == "implicit_current"
    }

    canonical_population_refs = _canonical_refs(
        population_refs,
        snapshot_digest=case.snapshot_digest,
        allowed_kinds={"population"},
        invalid_reasons=invalid_reasons,
        reason_prefix="population_ref",
    )
    canonical_falsification_refs = _canonical_refs(
        falsification_refs,
        snapshot_digest=case.snapshot_digest,
        allowed_kinds={"falsification"},
        invalid_reasons=invalid_reasons,
        reason_prefix="falsification_ref",
    )
    canonical_scope_condition_refs = _canonical_refs(
        scope_condition_refs,
        snapshot_digest=case.snapshot_digest,
        allowed_kinds={"explicit_evidence", "falsification"},
        invalid_reasons=invalid_reasons,
        reason_prefix="scope_condition_ref",
    )

    (
        population_counts,
        counts_by_population,
        population_contributions,
    ) = _population_label_counts(
        population_evidence,
        implicit_member_ids=implicit_member_ids,
        population_ids={item.stable_id for item in canonical_population_refs},
        invalid_reasons=invalid_reasons,
    )
    canonical_denominators = _validate_denominators(
        denominators,
        implicit_member_ids=implicit_member_ids,
        population_refs=canonical_population_refs,
        counts_by_population=counts_by_population,
        invalid_reasons=invalid_reasons,
        incomplete_reasons=incomplete_reasons,
    )

    explicit_covered = {
        member_id
        for item in explicit_evidence
        for member_id in item.member_ids
        if item.origin_kind
        in {
            "explicit_span",
            "configuration_atom",
            "command_atom",
            "workflow_atom",
        }
        and item.line_start is not None
    }
    if explicit_member_ids - explicit_covered:
        incomplete_reasons.add("explicit_span_evidence_missing")
    if implicit_member_ids and not canonical_population_refs:
        incomplete_reasons.add("population_artifact_ref_missing")
    falsification_dependency = next(
        (item for item in dependencies if item.capability == "current_falsification"),
        None,
    )
    falsification_not_required = (
        falsification_dependency is not None
        and falsification_dependency.status == "not_required"
        and falsification_dependency.requirement_mode == "not_required"
    )
    if (
        implicit_member_ids
        and not canonical_falsification_refs
        and not falsification_not_required
    ):
        incomplete_reasons.add("falsification_artifact_ref_missing")
    if implicit_member_ids and falsification_dependency is None:
        incomplete_reasons.add("falsification_capability_state_missing")
    if not canonical_scope_condition_refs:
        incomplete_reasons.add("scope_condition_evidence_missing")

    origins, items = _deduplicate_evidence(
        [*explicit_evidence, *population_contributions],
        case_member_ids=member_ids,
        snapshot_digest=case.snapshot_digest,
        invalid_reasons=invalid_reasons,
        incomplete_reasons=incomplete_reasons,
        max_anchors_per_item=limits.max_items,
    )
    origins, items = _apply_limits(
        origins,
        items,
        limits=limits,
        incomplete_reasons=incomplete_reasons,
    )

    if not implicit_member_ids or falsification_not_required:
        actual_falsification = ("not_required", ())
    elif not canonical_falsification_refs:
        actual_falsification = ("unavailable", ("falsification_artifact_ref_missing",))
    elif falsification_dependency is None:
        actual_falsification = ("unavailable", ("falsification_capability_state_missing",))
    else:
        actual_falsification = (
            falsification_dependency.status,
            tuple(falsification_dependency.reason_codes),
        )

    actual_capabilities: dict[str, tuple[str, tuple[str, ...]]] = {
        "explicit_current_evidence": (
            (
                "not_required"
                if not explicit_member_ids
                else "complete"
                if not (explicit_member_ids - explicit_covered)
                else "partial"
            ),
            ()
            if not (explicit_member_ids - explicit_covered)
            else ("explicit_span_evidence_missing",),
        ),
        "current_population": (
            (
                "not_required"
                if not implicit_member_ids
                else "complete"
                if not {
                    reason
                    for reason in incomplete_reasons
                    if reason.startswith("population_")
                }
                else "partial"
            ),
            tuple(
                sorted(
                    reason
                    for reason in incomplete_reasons
                    if reason.startswith("population_")
                )
            ),
        ),
        "current_falsification": actual_falsification,
        "scope_condition_evidence": (
            "complete" if canonical_scope_condition_refs else "unavailable",
            ()
            if canonical_scope_condition_refs
            else ("scope_condition_evidence_missing",),
        ),
    }
    canonical_dependencies = _merge_dependencies(
        dependencies,
        actual=actual_capabilities,
        required_capabilities=case.required_capabilities,
        invalid_reasons=invalid_reasons,
    )
    for dependency in canonical_dependencies:
        if dependency.status not in {
            "complete",
            "bounded_no_match",
            "not_required",
        }:
            incomplete_reasons.update(
                f"capability:{dependency.capability}:{reason}"
                for reason in dependency.reason_codes
            )
            if not dependency.reason_codes:
                incomplete_reasons.add(
                    f"capability:{dependency.capability}:{dependency.status}"
                )

    if case.grouping_status == "incomplete":
        incomplete_reasons.add("candidate_grouping_incomplete")
    if invalid_reasons:
        incomplete_reasons.update(invalid_reasons)
        integrity = "invalid"
        status = "failed"
    elif incomplete_reasons:
        integrity = "valid"
        status = "partial"
    else:
        integrity = "valid"
        status = "complete"

    payload: dict[str, object] = {
        "bundle_id": make_current_evidence_bundle_id(
            case.snapshot_digest, case.profile_digest, case.case_id
        ),
        "case_id": case.case_id,
        "snapshot_digest": case.snapshot_digest,
        "profile_digest": case.profile_digest,
        "origins": origins,
        "items": items,
        "denominators": canonical_denominators,
        "population_refs": canonical_population_refs,
        "falsification_refs": canonical_falsification_refs,
        "scope_condition_refs": canonical_scope_condition_refs,
        "dependencies": canonical_dependencies,
        "integrity": integrity,
        "status": status,
        "incomplete_reasons": sorted(incomplete_reasons),
    }
    payload["artifact_digest"] = artifact_digest(payload)
    bundle = CurrentEvidenceBundle.model_validate(payload)
    return CurrentEvidenceAssemblyResult(
        bundle=bundle,
        population_counts={
            key: dict(value) for key, value in sorted(population_counts.items())
        },
    )


__all__ = [
    "CURRENT_EVIDENCE_BUNDLE_VERSION",
    "CurrentEvidenceAssemblyResult",
    "CurrentEvidenceLimits",
    "EvidenceContribution",
    "PopulationEvidenceContribution",
    "assemble_current_evidence",
]
