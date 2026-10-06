"""Strict validation and provenance for agent-proposed explicit sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from repo_constraint_memory_v3.discovery import (
    is_supported_explicit_path,
    valid_relative_posix_path,
)
from repo_constraint_memory_v3.models import (
    DiscoveryArtifact,
    DiscoveryOrigin,
    DiscoverySettings,
    SourceMetadata,
    SourceSupplement,
)


class SourcePlanError(ValueError):
    """A source supplement violates snapshot, path, or resource boundaries."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class ProposalDecision:
    path: str
    compared_baseline_paths: tuple[str, ...]
    reason: str
    outcome: str
    decision_reason: str
    overlap_with_deterministic: bool


@dataclass(frozen=True)
class SourcePlanResult:
    sources: tuple[tuple[SourceMetadata, DiscoveryOrigin], ...]
    decisions: tuple[ProposalDecision, ...]
    proposed_count: int
    accepted_count: int
    rejected_count: int
    overlap_count: int
    agent_only_source_count: int


def empty_supplement(snapshot_digest: str) -> SourceSupplement:
    return SourceSupplement(
        schema_version="1.2",
        snapshot_digest=snapshot_digest,
        search_status="complete",
        proposals=[],
    )


def validate_source_plan(
    discovery: DiscoveryArtifact,
    supplement: SourceSupplement,
    settings: DiscoverySettings,
) -> SourcePlanResult:
    """Validate all proposals before admitting any of them."""

    errors: list[str] = []
    decisions: list[ProposalDecision] = []
    if supplement.snapshot_digest != discovery.repository.snapshot_digest:
        errors.append("source supplement snapshot_digest does not match discovery")
    if len(supplement.proposals) > settings.max_supplement_proposals:
        errors.append(
            "source supplement exceeds max_supplement_proposals "
            f"({len(supplement.proposals)} > {settings.max_supplement_proposals})"
        )

    manifest_by_path = {item.path: item for item in discovery.manifest.files}
    baseline_by_path = {item.path: item for item in discovery.baseline_sources}
    candidates_by_path = {
        item.path: item for item in discovery.supplement_candidates
    }
    admitted_agent: list[SourceMetadata] = []
    for proposal in supplement.proposals:
        proposal_errors: list[str] = []
        unknown_comparisons = sorted(
            set(proposal.compared_baseline_paths) - set(baseline_by_path)
        )
        if unknown_comparisons:
            proposal_errors.append(
                "compared_baseline_paths contains paths outside "
                f"discovery.baseline_sources: {', '.join(unknown_comparisons)}"
            )
        if not valid_relative_posix_path(proposal.path):
            proposal_errors.append("unsafe repository-relative POSIX path")
        if PurePosixPath(proposal.path).suffix.casefold() == ".py":
            proposal_errors.append(
                "ordinary Python source is outside explicit source supplementation"
            )
        if not is_supported_explicit_path(proposal.path):
            proposal_errors.append("unsupported explicit source format")
        if proposal.path not in manifest_by_path:
            proposal_errors.append("path is not in the discovery manifest")
        overlap = proposal.path in baseline_by_path
        metadata = baseline_by_path.get(proposal.path) or candidates_by_path.get(
            proposal.path
        )
        if metadata is None and proposal.path in manifest_by_path:
            proposal_errors.append("path is not an admitted supplement candidate")
        if metadata is not None and metadata.size_bytes > settings.max_source_file_bytes:
            proposal_errors.append("source exceeds max_source_file_bytes")

        if proposal_errors:
            reason = ", ".join(proposal_errors)
            errors.append(f"{proposal.path}: {reason}")
            decisions.append(
                ProposalDecision(
                    path=proposal.path,
                    compared_baseline_paths=tuple(
                        proposal.compared_baseline_paths
                    ),
                    reason=proposal.reason,
                    outcome="rejected",
                    decision_reason=reason,
                    overlap_with_deterministic=overlap,
                )
            )
            continue
        assert metadata is not None
        decisions.append(
            ProposalDecision(
                path=proposal.path,
                compared_baseline_paths=tuple(proposal.compared_baseline_paths),
                reason=proposal.reason,
                outcome="accepted",
                decision_reason=(
                    "already selected deterministically"
                    if overlap
                    else "bounded manifest supplement candidate"
                ),
                overlap_with_deterministic=overlap,
            )
        )
        if not overlap:
            admitted_agent.append(metadata)

    baseline_bytes = sum(item.size_bytes for item in discovery.baseline_sources)
    agent_bytes = sum(item.size_bytes for item in admitted_agent)
    final_count = len(discovery.baseline_sources) + len(admitted_agent)
    if final_count > settings.max_source_files:
        errors.append(
            f"selected sources exceed max_source_files ({final_count} > "
            f"{settings.max_source_files})"
        )
    if baseline_bytes + agent_bytes > settings.max_total_source_bytes:
        errors.append(
            "selected sources exceed max_total_source_bytes "
            f"({baseline_bytes + agent_bytes} > {settings.max_total_source_bytes})"
        )
    if errors:
        raise SourcePlanError(errors)

    sources: list[tuple[SourceMetadata, DiscoveryOrigin]] = [
        (item, "deterministic") for item in discovery.baseline_sources
    ]
    sources.extend((item, "agent_supplement") for item in admitted_agent)
    sources.sort(key=lambda item: item[0].path)
    accepted = sum(item.outcome == "accepted" for item in decisions)
    rejected = sum(item.outcome == "rejected" for item in decisions)
    overlap_count = sum(item.overlap_with_deterministic for item in decisions)
    return SourcePlanResult(
        sources=tuple(sources),
        decisions=tuple(decisions),
        proposed_count=len(supplement.proposals),
        accepted_count=accepted,
        rejected_count=rejected,
        overlap_count=overlap_count,
        agent_only_source_count=len(admitted_agent),
    )
