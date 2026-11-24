"""Shared orchestration for full-build and build-unified entry points."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from repo_constraint_memory_v3.adjudication import CandidateDecision
from repo_constraint_memory_v3.discovery import matches_glob
from repo_constraint_memory_v3.explicit_verification import ExplicitVerificationResult
from repo_constraint_memory_v3.inference import (
    InferenceBudget,
    InferenceRecord,
    run_budgeted_stage_calls,
)
from repo_constraint_memory_v3.llm import LLMClient
from repo_constraint_memory_v3.temporal_need import git_need_required
from repo_constraint_memory_v3.models import LLMSettings, SCHEMA_VERSION, UnifiedSettings
from repo_constraint_memory_v3.observations.engine import ObservationBundle
from repo_constraint_memory_v3.semantic.validation import CurrentSourceBundle
from repo_constraint_memory_v3.unified.current_evidence import (
    CurrentEvidenceLimits,
    EvidenceContribution,
    PopulationEvidenceContribution,
    assemble_current_evidence,
)
from repo_constraint_memory_v3.unified.git_evidence import (
    BoundedGitRunner,
    RunLocalGitRunner,
    GitAnchor,
    GitLimits,
    GitPreflight,
    GitTemporalBundle,
    collect_git_evidence,
    preflight_git_repository,
)
from repo_constraint_memory_v3.unified.grouping import (
    CandidateGroupingMode,
    GroupingBatch,
    GroupingBatchPlan,
    GroupingLimits,
    GroupingPlan,
    MemberBlockingAnchors,
    assemble_candidate_cases,
    mark_grouping_pairs_incomplete,
    plan_candidate_grouping,
    plan_grouping_batches,
    resolve_direct_grouping,
)
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    canonical_bytes,
    content_digest,
    make_bundle_id,
    make_finding_id,
    make_member_id,
    seal_payload,
    stable_id,
)
from repo_constraint_memory_v3.unified.inputs import (
    CaseEvidenceInputs,
    ChannelAssessmentOmission,
    build_blocking_anchors,
    build_case_evidence_inputs,
    build_case_input_index,
    build_channel_assessment_result,
    build_observation_anchor_index,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CandidateCaseMember,
    CandidateMemberRelation,
    CanonicalClaim,
    CapabilityDependency,
    ChannelAssessment,
    CurrentEvidenceBundle,
    ImmutableArtifactRef,
    MemberDisposition,
    TemporalRelationMatrix,
    UnifiedBundle,
    UnifiedCaseFinding,
    UnifiedCompleteness,
    UnifiedFunnel,
    UnifiedIndex,
    UnifiedMemberFinding,
    UNIFIED_CONTRACT_VERSION,
    UNIFIED_POLICY_VERSION,
)
from repo_constraint_memory_v3.unified.policy import (
    CanonicalPolicyAccumulator,
    apply_case_policy,
    is_explicit_fast_path,
)
from repo_constraint_memory_v3.unified.refinement import (
    CurrentEvidenceReacquisition,
    GitReacquisition,
    RoundOneVerification,
    ValidatedRefinementProposal,
    execute_refinement_round,
)
from repo_constraint_memory_v3.unified.semantic import (
    candidate_semantic_group,
    temporal_evidence_classify_bounded,
    temporal_evidence_classify_cases_bounded,
    unified_case_verify,
)


UNIFIED_GIT_ROUTING_VERSION = "unified-git-routing-1.4"


@dataclass(frozen=True)
class CandidateGroupingExecution:
    """Run-local grouping denominator exposed in the build report."""

    mode: CandidateGroupingMode
    status: str
    reason: str | None
    planned_pairs: int
    planned_calls: int
    terminal_calls: int
    case_count: int

    def __post_init__(self) -> None:
        if self.status not in {"complete", "partial", "not_required"}:
            raise ValueError("invalid candidate grouping execution status")
        if min(
            self.planned_pairs,
            self.planned_calls,
            self.terminal_calls,
            self.case_count,
        ) < 0:
            raise ValueError("candidate grouping execution counts must be nonnegative")
        if self.terminal_calls > self.planned_calls:
            raise ValueError("candidate grouping terminal calls exceed plan")
        if (self.status == "not_required") != (self.reason is not None):
            raise ValueError(
                "candidate grouping not-required status and reason differ"
            )

    def report_metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "status": self.status,
            "reason": self.reason,
            "planned_pairs": self.planned_pairs,
            "planned_calls": self.planned_calls,
            "terminal_calls": self.terminal_calls,
            "case_count": self.case_count,
        }


@dataclass(frozen=True)
class UnifiedRunResult:
    bundle: UnifiedBundle
    git_bundles: tuple[GitTemporalBundle, ...]
    git_artifacts: Mapping[str, object]
    inference_records: tuple[InferenceRecord, ...]
    grouping_plan: GroupingPlan | None
    grouping_execution: CandidateGroupingExecution
    case_routes: Mapping[str, str]


_FUNNEL_COUNT_KEYS = (
    "raw_candidate_events",
    "unique_candidate_roots",
    "refinement_versions",
    "materialized_candidate_versions",
    "accepted",
    "review",
    "rejected",
    "invalid",
    "superseded",
)


def _empty_funnel_breakdown() -> dict[str, int]:
    return {key: 0 for key in _FUNNEL_COUNT_KEYS}


def _build_unified_funnel(
    *,
    assessments: Sequence[ChannelAssessment],
    claims: Sequence[CanonicalClaim],
    dispositions: Sequence[MemberDisposition],
    grouping_execution: CandidateGroupingExecution,
    raw_candidate_events_by_channel: Mapping[str, int],
    assessment_omissions: Sequence[ChannelAssessmentOmission] = (),
    cases: Sequence[CandidateCase] = (),
    current_evidence: Sequence[CurrentEvidenceBundle] = (),
    git_bundles: Sequence[GitTemporalBundle] = (),
) -> UnifiedFunnel:
    """Build candidate-denominator and final-disposition funnel counts.

    The top-level accepted/review/rejected fields retain their schema-4
    canonical-claim meaning.  Channel/route breakdowns instead account for
    every candidate version exactly once, which preserves the raw event ->
    unique root -> refinement version denominator independently of grouping.
    """

    assessment_by_member = {
        make_member_id(item.snapshot_digest, item.assessment_id): item
        for item in assessments
    }
    disposition_by_member = {item.member_id: item for item in dispositions}
    if len(disposition_by_member) != len(dispositions) or set(
        disposition_by_member
    ) != set(assessment_by_member):
        raise ValueError("funnel requires one disposition per assessment")
    claim_by_id = {item.claim_id: item for item in claims}
    if len(claim_by_id) != len(claims):
        raise ValueError("funnel canonical claim IDs must be unique")

    by_channel: dict[str, dict[str, int]] = {
        channel: _empty_funnel_breakdown()
        for channel in ("explicit", "implicit_current")
    }
    by_route: dict[str, dict[str, int]] = {}
    attrition: Counter[str] = Counter()

    omission_keys = [
        (item.channel, item.candidate_root_id, item.candidate_version)
        for item in assessment_omissions
    ]
    if len(omission_keys) != len(set(omission_keys)):
        raise ValueError("assessment omission candidate versions must be unique")
    assessment_keys = {
        (item.channel, item.candidate_root_id, item.candidate_version)
        for item in assessments
    }
    if assessment_keys & set(omission_keys):
        raise ValueError("assessment and omission candidate versions overlap")

    initial_assessments = [
        item
        for item in assessments
        if "unified_refinement_child" not in item.typed_reasons
    ]
    initial_roots_by_channel = Counter(
        item.channel for item in initial_assessments
    )
    initial_roots_by_channel.update(
        item.channel
        for item in assessment_omissions
        if item.candidate_version == 0
    )
    for channel in by_channel:
        raw_count = int(raw_candidate_events_by_channel.get(channel, 0))
        root_count = initial_roots_by_channel[channel]
        if raw_count < root_count:
            raise ValueError("channel raw candidate events are below unique roots")
        by_channel[channel]["raw_candidate_events"] = raw_count
        by_channel[channel]["unique_candidate_roots"] = root_count
        gap = raw_count - root_count
        if gap:
            reason = "candidate_event_not_materialized"
            by_channel[channel][f"attrition:{reason}"] = gap
            attrition[reason] += gap
            route_counts = by_route.setdefault(
                "pre_normalization_unattributed", _empty_funnel_breakdown()
            )
            route_counts["raw_candidate_events"] += gap
            route_counts[f"attrition:{reason}"] = (
                route_counts.get(f"attrition:{reason}", 0) + gap
            )

    for assessment in initial_assessments:
        route = by_route.setdefault(
            assessment.source_authority, _empty_funnel_breakdown()
        )
        route["raw_candidate_events"] += 1
        route["unique_candidate_roots"] += 1
    for omission in assessment_omissions:
        if omission.candidate_version != 0:
            continue
        route = by_route.setdefault(
            omission.source_authority, _empty_funnel_breakdown()
        )
        route["raw_candidate_events"] += 1
        route["unique_candidate_roots"] += 1

    for assessment in assessments:
        member_id = make_member_id(
            assessment.snapshot_digest, assessment.assessment_id
        )
        disposition = disposition_by_member[member_id]
        channel_counts = by_channel[assessment.channel]
        route_counts = by_route.setdefault(
            assessment.source_authority, _empty_funnel_breakdown()
        )
        channel_counts["materialized_candidate_versions"] += 1
        route_counts["materialized_candidate_versions"] += 1
        if assessment.candidate_version > 0:
            channel_counts["refinement_versions"] += 1
            route_counts["refinement_versions"] += 1

        if disposition.disposition == "invalid":
            terminal = "invalid"
        elif disposition.disposition == "superseded_by_refinement":
            terminal = "superseded"
        elif disposition.disposition == "review":
            terminal = "review"
        elif disposition.disposition == "rejected":
            terminal = "rejected"
        else:
            statuses = {
                claim_by_id[claim_id].epistemic_status
                for claim_id in disposition.canonical_claim_ids
                if claim_id in claim_by_id
            }
            if len(statuses) != 1:
                raise ValueError(
                    "final member disposition does not resolve to one status"
                )
            terminal = statuses.pop()
        channel_counts[terminal] += 1
        route_counts[terminal] += 1

        if terminal == "accepted":
            continue
        reasons = set(disposition.reason_codes)
        if terminal == "invalid":
            reasons.update(assessment.typed_reasons)
        for reason in sorted(reasons):
            key = f"attrition:{reason}"
            channel_counts[key] = channel_counts.get(key, 0) + 1
            route_counts[key] = route_counts.get(key, 0) + 1
            attrition[reason] += 1

    for omission in assessment_omissions:
        channel_counts = by_channel[omission.channel]
        route_counts = by_route.setdefault(
            omission.source_authority, _empty_funnel_breakdown()
        )
        channel_counts["materialized_candidate_versions"] += 1
        route_counts["materialized_candidate_versions"] += 1
        if omission.candidate_version > 0:
            channel_counts["refinement_versions"] += 1
            route_counts["refinement_versions"] += 1
        channel_counts["invalid"] += 1
        route_counts["invalid"] += 1
        key = f"attrition:{omission.reason_code}"
        channel_counts[key] = channel_counts.get(key, 0) + 1
        route_counts[key] = route_counts.get(key, 0) + 1
        attrition[omission.reason_code] += 1

    canonical_counts = Counter(item.epistemic_status for item in claims)
    need_counts = {
        status: sum(item.git_need.status == status for item in assessments)
        for status in ("affirmative", "no_established_need", "undetermined", "not_assessed", "invalid")
    }
    need_counts["candidate_versions"] = len(assessments)
    current_by_case = {item.case_id: item for item in current_evidence}
    assessment_by_id = {item.assessment_id: item for item in assessments}
    case_routes = {
        case.case_id: _route_case(case, assessment_by_id, current_by_case[case.case_id])
        for case in cases
    }
    git_counts = {
        "cases": len(case_routes),
        "eligible_cases": sum(route not in {"not_required_incomplete", "not_required_hard_invalid", "not_required_current_incomplete"} for route in case_routes.values()),
        "affirmative_cases": sum(route in {"augmentative", "ambiguity_resolving"} for route in case_routes.values()),
        "optional_cases": sum(route == "augmentative" for route in case_routes.values()),
        "required_cases": sum(route == "ambiguity_resolving" for route in case_routes.values()),
        "queried_cases": sum(bool(item.manifest.command_results) for item in git_bundles),
    }
    statuses = []
    for case_id, route in case_routes.items():
        dependencies = [item for item in current_by_case[case_id].dependencies if item.capability == "git_temporal_evidence"]
        if not dependencies and route.startswith("not_required"):
            # A child closed on required current evidence never reaches the
            # Git reroute stage; absence here is an actual not-required search.
            statuses.append("not_required")
            continue
        if len(dependencies) != 1:
            raise ValueError("Git funnel requires one terminal case dependency")
        dependency = dependencies[0]
        if route.startswith("not_required") and dependency.status != "not_required":
            raise ValueError("unqueried Git route cannot claim a completed search")
        statuses.append(dependency.status)
    for status in ("complete", "bounded_no_match", "partial", "unavailable", "failed", "not_required"):
        git_counts[f"result_{status}"] = statuses.count(status)
    # A shared preflight appears in each case bundle for provenance. Count its
    # physical work once, while every case keeps its own logical query budget.
    owned_preflights = {
        content_digest(item.preflight.model_dump(mode="json"))
        for item in git_bundles if item.raw_execution_scope == "collection_with_preflight"
    }
    shared_preflights = {
        content_digest(item.preflight.model_dump(mode="json")): item.preflight
        for item in git_bundles if item.raw_execution_scope == "collection_only"
    }
    for counter in ("logical_calls", "physical_calls", "cache_hits"):
        git_counts[f"raw_{counter}"] = sum(getattr(item.raw_execution, counter) for item in git_bundles)
    for digest, preflight in shared_preflights.items():
        if digest in owned_preflights:
            continue
        hits = sum(check.cache_hit for check in preflight.checks)
        git_counts["raw_logical_calls"] += len(preflight.checks)
        git_counts["raw_physical_calls"] += len(preflight.checks) - hits
        git_counts["raw_cache_hits"] += hits
    return UnifiedFunnel(
        git_need_counts=need_counts,
        git_case_counts=git_counts,
        candidate_grouping_mode=grouping_execution.mode,
        grouping_status=grouping_execution.status,
        grouping_reason=grouping_execution.reason,
        grouping_planned_pairs=grouping_execution.planned_pairs,
        grouping_planned_calls=grouping_execution.planned_calls,
        grouping_terminal_calls=grouping_execution.terminal_calls,
        grouping_case_count=grouping_execution.case_count,
        raw_candidate_events=sum(raw_candidate_events_by_channel.values()),
        unique_candidate_roots=(
            len(initial_assessments)
            + sum(
                item.candidate_version == 0
                for item in assessment_omissions
            )
        ),
        refinement_versions=sum(
            item.candidate_version > 0 for item in assessments
        )
        + sum(
            item.candidate_version > 0 for item in assessment_omissions
        ),
        invalid_members=sum(
            item.normalized_status == "invalid" for item in assessments
        )
        + len(assessment_omissions),
        accepted=canonical_counts["accepted"],
        review=canonical_counts["review"],
        rejected=canonical_counts["rejected"],
        by_channel={
            key: dict(sorted(value.items()))
            for key, value in sorted(by_channel.items())
        },  # type: ignore[arg-type]
        by_evidence_route={
            key: dict(sorted(value.items()))
            for key, value in sorted(by_route.items())
        },
        attrition_reasons=dict(sorted(attrition.items())),
    )


def _case_members(
    assessments: Sequence[ChannelAssessment],
) -> tuple[CandidateCaseMember, ...]:
    result: list[CandidateCaseMember] = []
    for assessment in assessments:
        member_id = make_member_id(
            assessment.snapshot_digest, assessment.assessment_id
        )
        result.append(
            CandidateCaseMember(
                member_id=member_id,
                assessment_id=assessment.assessment_id,
                snapshot_digest=assessment.snapshot_digest,
                channel=assessment.channel,
                semantics=assessment.semantics,
                assessment_ref=ImmutableArtifactRef(
                    kind="channel_assessment",
                    stable_id=assessment.assessment_id,
                    snapshot_digest=assessment.snapshot_digest,
                    artifact_path=(
                        "canonical/audit/channel-assessments/"
                        f"{assessment.assessment_id}.json"
                    ),
                    artifact_digest=assessment.artifact_digest,
                ),
            )
        )
    return tuple(sorted(result, key=lambda item: item.member_id))


def _grouping_limits(settings: UnifiedSettings) -> GroupingLimits:
    return GroupingLimits(
        max_candidate_members=settings.max_candidate_members,
        max_block_members=settings.max_block_members,
        max_group_pairs=settings.max_group_pairs,
        max_group_pairs_per_request=settings.max_group_pairs_per_request,
        max_group_members_per_request=settings.max_group_members_per_request,
        max_group_request_bytes=settings.max_group_request_bytes,
        max_case_members=settings.max_case_members,
        max_case_input_bytes=settings.max_case_input_bytes,
        max_anchors_per_member=settings.max_blocking_anchors_per_member,
        max_paths_per_member=settings.max_blocking_paths_per_member,
        max_tokens_per_member=settings.max_blocking_tokens_per_member,
        max_anchor_bytes=settings.max_blocking_anchor_bytes,
        lexical_min_shared_tokens=settings.lexical_min_shared_tokens,
    )


def _git_limits(settings: UnifiedSettings) -> GitLimits:
    return GitLimits(
        command_timeout_seconds=settings.git_command_timeout_seconds,
        object_scan_timeout_seconds=settings.git_object_scan_timeout_seconds,
        max_command_output_bytes=settings.git_max_bytes_per_command,
        max_object_scan_output_bytes=settings.git_max_object_scan_bytes,
        min_path_history_queries=settings.git_min_path_history,
        min_blame_queries=settings.git_min_blame,
        min_pickaxe_literal_queries=settings.git_min_pickaxe_literal,
        min_pickaxe_regex_queries=settings.git_min_pickaxe_regex,
        max_queries_per_family=settings.git_max_queries_per_family,
        max_total_commands=settings.git_max_commands_per_case,
        max_results_per_query=settings.git_max_results_per_query,
        max_results_per_family=(
            settings.git_max_results_per_query
            * settings.git_max_queries_per_family
        ),
        max_total_candidate_commits=settings.git_max_patches_per_case,
        max_anchor_chars=settings.max_blocking_anchor_bytes,
        max_blame_lines=200,
        rename_similarity_percent=settings.git_rename_similarity_percent,
        patch_context_lines=settings.git_patch_context_lines,
        max_changed_paths_per_commit=settings.git_max_changed_paths_per_commit,
        max_patch_bytes=settings.git_max_bytes_per_command,
        max_total_patch_bytes=settings.git_max_bytes_per_case,
        max_change_slices=min(
            settings.git_max_patches_per_case,
            settings.max_temporal_items_per_case,
        ),
        max_commit_message_bytes=settings.git_max_commit_message_bytes,
    )


def _route_case(
    case: CandidateCase,
    assessment_by_id: Mapping[str, ChannelAssessment],
    current_evidence: CurrentEvidenceBundle,
) -> str:
    if case.grouping_status == "incomplete":
        return "not_required_incomplete"
    values = [assessment_by_id[item.assessment_id] for item in case.members]
    if all(
        item.integrity == "invalid" or item.normalized_status == "invalid"
        for item in values
    ):
        # Invalid channel artifacts are typed attrition, not a semantic question
        # for Git or the unified verifier.  Preserve the case for audit and
        # member disposition, but do not expose invalid payloads to either stage.
        return "not_required_hard_invalid"
    # These dependencies are produced only after the case has already been
    # routed to Git/refinement.  They must not be allowed to retroactively
    # change that route during offline replay (or a later refinement check).
    # Unknown required capabilities remain fail-closed.
    post_route_capabilities = {
        "git_temporal_evidence",
        "temporal_evidence_classification",
        "refinement_git_reroute",
    }
    if any(
        item.requirement_mode == "required"
        and item.status not in {"complete", "bounded_no_match", "not_required"}
        and item.capability not in post_route_capabilities
        for item in getattr(current_evidence, "dependencies", ())
    ):
        # A verifier cannot turn a candidate with missing required current
        # substrate into an accepted or rejected claim.  Close this case as a
        # typed review before Git and semantic verification instead of paying
        # for evidence that the frozen policy is required to ignore.
        return "not_required_current_incomplete"
    if is_explicit_fast_path(
        case=case,
        assessments=assessment_by_id,
        current_evidence=current_evidence,
    ):
        return "not_required_fast_path"
    affirmative = [
        item.git_need for item in values
        if item.integrity == "valid" and item.git_need.status == "affirmative"
    ]
    if any(git_need_required(need) for need in affirmative):
        return "ambiguity_resolving"
    if affirmative:
        return "augmentative"
    return "not_required"


def _git_anchors(
    case: CandidateCase,
    evidence: CurrentEvidenceBundle,
    *,
    max_blame_lines: int = 200,
) -> tuple[GitAnchor, ...]:
    if max_blame_lines < 1:
        raise ValueError("Git blame line bound must be positive")
    origin_by_id = {item.origin_id: item for item in evidence.origins}
    anchor_origins: dict[tuple[object, ...], set[str]] = {}
    precise_origins: dict[tuple[object, ...], set[str]] = {}

    def bind(
        key: tuple[object, ...],
        origin_id: str,
        *,
        precise: bool = False,
    ) -> None:
        anchor_origins.setdefault(key, set()).add(origin_id)
        if precise:
            precise_origins.setdefault(key, set()).add(origin_id)

    for item in evidence.items:
        # Git discovery must be conditioned only by evidence that currently
        # supports or counters a case member.  Unknown, ineligible, and
        # context-only observations still remain in CurrentEvidenceBundle and
        # are shown to the case verifier, but they cannot widen history search.
        if item.role not in {"support", "counterevidence"}:
            continue
        origin = origin_by_id[item.origin_id]
        path_key = ("path", origin.path, None, None, None)
        bind(path_key, item.origin_id)
        if origin.line_start is not None and origin.line_end is not None:
            for line_start in range(
                origin.line_start,
                origin.line_end + 1,
                max_blame_lines,
            ):
                line_end = min(
                    origin.line_end,
                    line_start + max_blame_lines - 1,
                )
                line_key = (
                    "line",
                    origin.path,
                    None,
                    line_start,
                    line_end,
                )
                bind(line_key, item.origin_id)
        for kind, values in (
            ("literal", item.anchor_literals),
            ("symbol", item.anchor_symbols),
        ):
            for value in values:
                key = (kind, origin.path, value, None, None)
                # Explicit typed semantic keys are useful bounded query hints,
                # but are not guaranteed to be verbatim source. Exact quote
                # lines and observation/population symbols are source-derived
                # and may therefore satisfy the strong changed-line gate.
                is_precise = kind == "literal" or origin.origin_kind in {
                    "observation",
                    "population_member",
                }
                bind(
                    key,
                    item.origin_id,
                    precise=is_precise,
                )
    anchors: list[GitAnchor] = []
    for key in sorted(anchor_origins, key=repr):
        kind, path, value, start_line, end_line = key
        anchors.append(
            GitAnchor(
                anchor_id=stable_id("GITANCHOR", [case.case_id, *key]),
                kind=kind,  # type: ignore[arg-type]
                path=path,  # type: ignore[arg-type]
                value=value,  # type: ignore[arg-type]
                start_line=start_line,  # type: ignore[arg-type]
                end_line=end_line,  # type: ignore[arg-type]
                origin_ids=sorted(anchor_origins[key]),
                precise_origin_ids=sorted(precise_origins.get(key, set())),
            )
        )
    return tuple(sorted(anchors, key=lambda item: item.anchor_id))


def _with_dependencies(
    evidence: CurrentEvidenceBundle,
    additions: Sequence[CapabilityDependency],
) -> CurrentEvidenceBundle:
    by_capability = {item.capability: item for item in evidence.dependencies}
    for item in additions:
        by_capability[item.capability] = item
    dependencies = [by_capability[key] for key in sorted(by_capability)]
    payload = evidence.model_dump(mode="python", exclude={"artifact_digest"})
    # Evidence assembly quality and downstream capability availability are
    # separate audit axes.  Required Git/temporal gaps affect case policy and
    # package completeness through ``dependencies``; they must not rewrite a
    # complete current-snapshot evidence assembly into a synthetic partial.
    payload["dependencies"] = dependencies
    return CurrentEvidenceBundle.model_validate(seal_payload(payload))


def _with_case_marker(case: CandidateCase, marker: str) -> CandidateCase:
    payload = case.model_dump(mode="python", exclude={"artifact_digest"})
    payload["grouping_reason_codes"] = sorted(
        {*case.grouping_reason_codes, marker}
    )
    return CandidateCase.model_validate(seal_payload(payload))


_INFERENCE_BUDGET_REASONS = frozenset(
    {
        "max_total_calls",
        "max_total_input_tokens",
        "max_total_output_tokens",
    }
)


def _checked_budget_skip_reason(reason: str | None) -> str | None:
    if reason is not None and reason not in _INFERENCE_BUDGET_REASONS:
        raise ValueError("unknown inference-budget closure reason")
    return reason


def _dependency_for_git(
    route: str,
    bundle: GitTemporalBundle | None,
    *,
    host_skip_reason: str | None = None,
) -> CapabilityDependency:
    host_skip_reason = _checked_budget_skip_reason(host_skip_reason)
    if route.startswith("not_required"):
        return CapabilityDependency(
            capability="git_temporal_evidence",
            requirement_mode="not_required",
            status="not_required",
            reason_codes=[],
        )
    mode = "required" if route == "ambiguity_resolving" else "augmentative"
    if bundle is None:
        reasons = (
            [
                "git_not_run_after_inference_budget_exhausted",
                host_skip_reason,
            ]
            if host_skip_reason is not None
            else ["git_disabled_or_snapshot_commit_unavailable"]
        )
        return CapabilityDependency(
            capability="git_temporal_evidence",
            requirement_mode=mode,
            status="unavailable",
            reason_codes=sorted(reasons),
        )
    status = bundle.status
    return CapabilityDependency(
        capability="git_temporal_evidence",
        requirement_mode=mode,
        status=status,
        reason_codes=(
            []
            if status in {"complete", "bounded_no_match", "not_required"}
            else sorted(set(bundle.reason_codes))
        ),
    )


def _dependency_for_temporal(
    route: str,
    git_bundle: GitTemporalBundle | None,
    matrix: TemporalRelationMatrix | None,
    *,
    host_skip_reason: str | None = None,
) -> CapabilityDependency:
    host_skip_reason = _checked_budget_skip_reason(host_skip_reason)
    mode = "required" if route == "ambiguity_resolving" else "augmentative"
    if (
        route == "ambiguity_resolving"
        and git_bundle is not None
        and git_bundle.status == "bounded_no_match"
    ):
        return CapabilityDependency(
            capability="temporal_evidence_classification",
            requirement_mode="required",
            status="unavailable",
            reason_codes=["required_git_question_unanswered_no_match"],
        )
    if route.startswith("not_required") or (
        git_bundle is not None and git_bundle.status == "bounded_no_match"
    ):
        return CapabilityDependency(
            capability="temporal_evidence_classification",
            requirement_mode="not_required",
            status="not_required",
            reason_codes=[],
        )
    if git_bundle is None or not git_bundle.slices:
        if host_skip_reason is not None:
            return CapabilityDependency(
                capability="temporal_evidence_classification",
                requirement_mode=mode,
                status="unavailable",
                reason_codes=sorted(
                    [
                        "temporal_not_run_after_inference_budget_exhausted",
                        host_skip_reason,
                    ]
                ),
            )
        return CapabilityDependency(
            capability="temporal_evidence_classification",
            requirement_mode=mode,
            status=("not_required" if route == "augmentative" else "unavailable"),
            reason_codes=(
                []
                if route == "augmentative"
                else ["temporal_items_unavailable"]
            ),
        )
    if matrix is None:
        reasons = (
            [
                "temporal_not_run_after_inference_budget_exhausted",
                host_skip_reason,
            ]
            if host_skip_reason is not None
            else ["temporal_classifier_failed"]
        )
        return CapabilityDependency(
            capability="temporal_evidence_classification",
            requirement_mode=mode,
            status="failed",
            reason_codes=sorted(reasons),
        )
    return CapabilityDependency(
        capability="temporal_evidence_classification",
        requirement_mode=mode,
        status=matrix.status,
        reason_codes=(
            [] if matrix.status == "complete" else matrix.incomplete_reasons
        ),
    )


def _host_finding(
    case: CandidateCase,
    evidence: CurrentEvidenceBundle,
    *,
    profile_digest: str,
    status: str,
    reason_code: str,
    detail_reason_codes: Sequence[str] = (),
) -> UnifiedCaseFinding:
    hard_invalid = reason_code == "all_members_hard_invalid"
    reasons = sorted({reason_code, *detail_reason_codes})
    refs_by_member: dict[str, list[str]] = defaultdict(list)
    for item in evidence.items:
        if item.role == "support":
            for member_id in item.member_ids:
                refs_by_member[member_id].append(item.evidence_id)
    member_findings = [
        UnifiedMemberFinding(
            member_id=member.member_id,
            norm_relevance=(
                "fail"
                if hard_invalid
                else "pass" if status == "not_required" else "uncertain"
            ),
            rule_support=(
                "not_applicable"
                if hard_invalid
                else "pass" if status == "not_required" else "uncertain"
            ),
            scope_support=(
                "not_applicable"
                if hard_invalid
                else "pass" if status == "not_required" else "uncertain"
            ),
            condition_support=(
                "not_applicable"
                if hard_invalid
                else "pass" if status == "not_required" else "uncertain"
            ),
            temporal_status="not_required" if status == "not_required" else "uncertain",
            evidence_sufficiency=(
                "not_applicable"
                if hard_invalid
                else "pass"
                if status == "not_required" and refs_by_member[member.member_id]
                else "uncertain"
            ),
            evidence_refs=(
                []
                if hard_invalid
                else sorted(set(refs_by_member[member.member_id]))
            ),
            closed_local_reason_codes=[],
            reason_codes=reasons,
        )
        for member in case.members
    ]
    payload = {
        "finding_id": make_finding_id(
            case.snapshot_digest, profile_digest, case.case_id, 0
        ),
        "case_id": case.case_id,
        "snapshot_digest": case.snapshot_digest,
        "profile_digest": profile_digest,
        "verification_round": 0,
        "member_findings": member_findings,
        "member_relations": [],
        "unresolved_conflicts": [],
        "conclusions": [],
        "refinement_proposals": [],
        "inference_ref": None,
        "status": status,
        "failure_reasons": [] if status == "not_required" else reasons,
    }
    return UnifiedCaseFinding.model_validate(seal_payload(payload))


def _capability(
    name: str,
    statuses: Sequence[str],
    *,
    requirement_mode: str = "required",
    reasons: Sequence[str] = (),
) -> CapabilityDependency:
    if not statuses:
        return CapabilityDependency(
            capability=name,
            requirement_mode="not_required",
            status="not_required",
            reason_codes=[],
        )
    success = {"complete", "bounded_no_match", "not_required"}
    if all(item in success for item in statuses):
        status = "complete" if "complete" in statuses else "not_required"
        return CapabilityDependency(
            capability=name,
            requirement_mode=requirement_mode,
            status=status,
            reason_codes=[],
        )
    if "failed" in statuses:
        status = "failed"
    elif "unavailable" in statuses:
        status = "unavailable"
    else:
        status = "partial"
    return CapabilityDependency(
        capability=name,
        requirement_mode=requirement_mode,
        status=status,
        reason_codes=sorted(set(reasons)) or [f"{name}_{status}"],
    )


def _git_temporal_gap_reasons(
    bundles: Sequence[GitTemporalBundle],
) -> list[str]:
    """Project only actual Git gaps, excluding per-query closure telemetry."""

    closure = {"query_complete", "query_complete_no_match"}
    return sorted(
        {
            reason
            for item in bundles
            if item.status not in {"complete", "bounded_no_match", "not_required"}
            for reason in item.reason_codes
            if reason not in closure
        }
    )


def _role_for_population_label(label: str) -> str:
    return "counterevidence" if label == "counterexample" else label


def _refinement_scope_narrows(
    parent: Sequence[str], child: Sequence[str]
) -> bool:
    if set(child) < set(parent) or (
        set(parent) == {"**"} and set(child) != {"**"}
    ):
        return True
    if set(parent) == set(child):
        return False
    for child_pattern in child:
        child_prefix = child_pattern.split("*", 1)[0]
        if not any(
            parent_pattern == "**"
            or child_prefix.startswith(parent_pattern.split("*", 1)[0])
            for parent_pattern in parent
        ):
            return False
    return True


def _reacquire_refined_current_from_saved_inputs(
    *,
    parent_case: CandidateCase,
    refined_case: CandidateCase,
    parent: CandidateCaseMember,
    child_assessments: Sequence[ChannelAssessment],
    proposals: Sequence[ValidatedRefinementProposal],
    previous_evidence: CurrentEvidenceBundle,
    inputs: CaseEvidenceInputs,
    limits: CurrentEvidenceLimits,
) -> CurrentEvidenceReacquisition:
    """Rebind a complete saved population without inventing new evidence.

    The current-source superpopulation already serializes out-of-scope members
    as ineligible.  Expansion can therefore re-admit those occurrences, but an
    old ineligible or condition-bound judgment is conservatively changed to
    ``unknown`` unless the saved label remains valid by monotonic narrowing.
    """

    if previous_evidence.integrity != "valid" or previous_evidence.status != "complete":
        return CurrentEvidenceReacquisition(
            current_evidence=None,
            substrate_kind="unavailable",
            complete=False,
            reason_codes=("refinement_parent_current_evidence_incomplete",),
        )
    child_member_by_assessment = {
        item.assessment_id: item
        for item in refined_case.members
        if item.assessment_id in {
            assessment.assessment_id for assessment in child_assessments
        }
    }
    if len(child_member_by_assessment) != len(child_assessments):
        return CurrentEvidenceReacquisition(
            current_evidence=None,
            substrate_kind="unavailable",
            complete=False,
            reason_codes=("refinement_child_member_binding_invalid",),
        )
    # execute_refinement_round constructs children and proposals in the same
    # stable order.  Keep the association explicit instead of matching prose.
    proposal_values = list(proposals)
    if len(proposal_values) != len(child_assessments):
        return CurrentEvidenceReacquisition(
            current_evidence=None,
            substrate_kind="unavailable",
            complete=False,
            reason_codes=("refinement_child_proposal_alignment_invalid",),
        )

    parent_population = [
        item
        for item in inputs.population_evidence
        if item.case_member_id == parent.member_id
    ]
    parent_denominators = [
        item for item in inputs.denominators if item.member_id == parent.member_id
    ]
    if parent.channel == "implicit_current":
        if (
            len(parent_denominators) != 1
            or parent_denominators[0].status != "complete"
            or len(parent_population) != parent_denominators[0].serialized
        ):
            return CurrentEvidenceReacquisition(
                current_evidence=None,
                substrate_kind="unavailable",
                complete=False,
                reason_codes=("refinement_complete_population_unavailable",),
            )
    elif any(
        item.requires_current_reacquisition
        and set(item.proposal.change_kinds)
        & {"condition", "exception"}
        for item in proposal_values
    ):
        # A saved declaration can support a narrower split, but it cannot prove
        # a relaxed condition or changed exception over a newly admitted domain.
        return CurrentEvidenceReacquisition(
            current_evidence=None,
            substrate_kind="unavailable",
            complete=False,
            reason_codes=("refinement_explicit_semantic_reacquisition_unavailable",),
        )

    child_pairs = [
        (
            child_member_by_assessment[assessment.assessment_id],
            proposal,
        )
        for assessment, proposal in zip(child_assessments, proposal_values)
    ]
    explicit: list[EvidenceContribution] = []
    population: list[PopulationEvidenceContribution] = []

    for contribution in [
        *inputs.explicit_evidence,
        *inputs.observation_evidence,
    ]:
        if parent.member_id not in contribution.member_ids:
            explicit.append(contribution)
            continue
        for child, proposal in child_pairs:
            changes = set(proposal.proposal.change_kinds)
            unsafe = bool(
                changes & {"condition", "exception"}
                or (
                    "scope" in changes
                    and not _refinement_scope_narrows(
                        parent.semantics.applies_to,
                        child.semantics.applies_to,
                    )
                )
            )
            explicit.append(
                replace(
                    contribution,
                    role="context" if unsafe else contribution.role,
                    member_ids=(child.member_id,),
                    description=(
                        "Original explicit grounding retained as refinement context"
                        if unsafe
                        else contribution.description
                    ),
                )
            )

    parent_patterns = parent.semantics.applies_to
    for occurrence in inputs.population_evidence:
        if occurrence.case_member_id != parent.member_id:
            population.append(occurrence)
            continue
        for child, proposal in child_pairs:
            path = occurrence.evidence.path
            in_parent_scope = any(
                matches_glob(path, pattern) for pattern in parent_patterns
            )
            in_child_scope = any(
                matches_glob(path, pattern)
                for pattern in child.semantics.applies_to
            )
            changes = set(proposal.proposal.change_kinds)
            must_reclassify = (
                ("scope" in changes and in_child_scope and not in_parent_scope)
                or (
                    proposal.requires_current_reacquisition
                    and bool(changes & {"condition", "exception"})
                )
            )
            if not in_child_scope:
                label = "ineligible"
            elif must_reclassify:
                label = "unknown"
            else:
                label = occurrence.label
            description = occurrence.evidence.description
            if must_reclassify:
                description = (
                    "Complete saved substrate member requires refined semantic "
                    "classification"
                )
            rebound_evidence = replace(
                occurrence.evidence,
                role=_role_for_population_label(label),  # type: ignore[arg-type]
                member_ids=(child.member_id,),
                description=description,
            )
            population.append(
                PopulationEvidenceContribution(
                    case_member_id=child.member_id,
                    population_id=occurrence.population_id,
                    population_member_id=occurrence.population_member_id,
                    label=label,  # type: ignore[arg-type]
                    evidence=rebound_evidence,
                )
            )

    denominators = [
        item for item in inputs.denominators if item.member_id != parent.member_id
    ]
    if parent.channel == "implicit_current":
        parent_denominator = parent_denominators[0]
        for child, _ in child_pairs:
            child_occurrences = [
                item for item in population if item.case_member_id == child.member_id
            ]
            counts = Counter(item.label for item in child_occurrences)
            denominators.append(
                PopulationDenominator(
                    member_id=child.member_id,
                    population_id=parent_denominator.population_id,
                    status=(
                        "partial"
                        if parent_denominator.uninspected
                        else "complete"
                    ),
                    census_enumeration_status=(
                        "partial"
                        if parent_denominator.uninspected
                        else parent_denominator.census_enumeration_status
                    ),
                    sampling_plan_status=(
                        parent_denominator.sampling_plan_status
                    ),
                    sampling_mode=parent_denominator.sampling_mode,
                    scanned=parent_denominator.scanned,
                    serialized=(
                        len(child_occurrences)
                        + parent_denominator.uninspected
                    ),
                    matched=(
                        len(child_occurrences)
                        + parent_denominator.uninspected
                    ),
                    inspected=len(child_occurrences),
                    uninspected=parent_denominator.uninspected,
                    support=counts["support"],
                    counterexample=counts["counterexample"],
                    unknown=counts["unknown"],
                    ineligible=counts["ineligible"],
                )
            )
    result = assemble_current_evidence(
        refined_case,
        explicit_evidence=explicit,
        population_evidence=population,
        denominators=denominators,
        population_refs=inputs.population_refs,
        falsification_refs=inputs.falsification_refs,
        scope_condition_refs=inputs.scope_condition_refs,
        dependencies=inputs.dependencies,
        limits=limits,
    ).bundle
    complete = result.integrity == "valid" and result.status == "complete"
    return CurrentEvidenceReacquisition(
        current_evidence=result,
        substrate_kind=(
            "complete_population"
            if parent.channel == "implicit_current"
            else "complete_observation"
        ),
        complete=complete,
        reason_codes=(
            ()
            if complete
            else tuple(result.incomplete_reasons)
            or ("refinement_current_reassembly_incomplete",)
        ),
    )


def _refs_for_git_bundle(
    bundle: GitTemporalBundle,
) -> tuple[ImmutableArtifactRef, ...]:
    values = [
        (
            "git_temporal_bundle",
            bundle.bundle_id,
            bundle.artifact_digest,
            "bundles",
        ),
        (
            "git_manifest",
            bundle.manifest.manifest_id,
            bundle.manifest.artifact_digest,
            "manifests",
        ),
        *[
            (
                "git_change_slice",
                item.item_id,
                item.artifact_digest,
                "slices",
            )
            for item in bundle.slices
        ],
    ]
    return tuple(
        ImmutableArtifactRef(
            kind=kind,  # type: ignore[arg-type]
            stable_id=stable_id_value,
            snapshot_digest=bundle.snapshot_digest,
            artifact_path=(
                f"canonical/audit/git/{directory}/{stable_id_value}.json"
            ),
            artifact_digest=digest,
        )
        for kind, stable_id_value, digest, directory in values
    )


def _run_grouping_batch_plan(
    *,
    batch_plan: GroupingBatchPlan,
    members_by_id: Mapping[str, CandidateCaseMember],
    llm_settings: LLMSettings,
    client: LLMClient,
    budget: InferenceBudget,
    protocol_manifest_digest: str,
    max_request_bytes: int,
) -> tuple[
    list[CandidateMemberRelation],
    set[tuple[str, str]],
    list[InferenceRecord],
]:
    """Execute one grouping phase concurrently and merge in planned order."""

    def invoke(
        batch: GroupingBatch,
    ) -> tuple[list[CandidateMemberRelation] | None, InferenceRecord]:
        return candidate_semantic_group(
            [members_by_id[item] for item in batch.member_ids],
            llm_settings=llm_settings,
            client=client,
            budget=budget,
            protocol_manifest_digest=protocol_manifest_digest,
            max_request_bytes=max_request_bytes,
            required_pairs=batch.pairs,
            phase=batch.phase,  # type: ignore[arg-type]
            _stage_already_planned=True,
        )

    stage_run = run_budgeted_stage_calls(
        batch_plan.batches,
        invoke,
        stage="candidate_semantic_group",
        budget=budget,
        max_concurrency=llm_settings.max_concurrency,
        thread_name_prefix=f"repo-memory-v3-group-{batch_plan.phase}",
    )
    relations: list[CandidateMemberRelation] = []
    failed_pairs = set(batch_plan.omitted_pairs)
    records: list[InferenceRecord] = []
    for batch, (result, record) in zip(
        batch_plan.batches, stage_run.outcomes
    ):
        records.append(record)
        if result is None:
            failed_pairs.update(batch.pairs)
        else:
            relations.extend(result)
    for batch in batch_plan.batches[len(stage_run.outcomes) :]:
        failed_pairs.update(batch.pairs)
    return relations, failed_pairs, records


def run_unified_layer(
    *,
    repository_root: Path,
    snapshot_digest: str,
    snapshot_commit: str | None,
    profile_digest: str,
    settings: UnifiedSettings,
    llm_settings: LLMSettings,
    client: LLMClient,
    budget: InferenceBudget,
    protocol_manifest_digest: str,
    explicit_decisions: Sequence[CandidateDecision],
    explicit_verification: ExplicitVerificationResult,
    current_source: CurrentSourceBundle | None,
    observations: ObservationBundle | None,
    explicit_discovery_complete: bool,
    current_source_required: bool = False,
    raw_candidate_events: int | None = None,
) -> UnifiedRunResult:
    """Build the only canonical result from immutable two-channel inputs."""

    assessment_build = build_channel_assessment_result(
        explicit_decisions=explicit_decisions,
        explicit_verification=explicit_verification,
        current_source=current_source,
        snapshot_digest=snapshot_digest,
        profile_digest=profile_digest,
    )
    assessments = list(assessment_build.assessments)
    assessment_omissions = assessment_build.omissions
    initial_assessment_count = len(assessments)
    initial_candidate_roots = {
        (item.channel, item.candidate_root_id) for item in assessments
    }
    members = _case_members(assessments)
    assessment_by_id = {item.assessment_id: item for item in assessments}
    inference_records: list[InferenceRecord] = []
    plan: GroupingPlan | None = None
    cases: tuple[CandidateCase, ...] = ()
    grouping_planned_pairs = 0
    grouping_planned_calls = 0
    grouping_terminal_calls = 0
    anchors_by_member: dict[str, MemberBlockingAnchors] = {}
    input_index = build_case_input_index(
        assessments=assessments, explicit_decisions=explicit_decisions,
        observations=observations, current_source=current_source,
    )
    if members:
        anchors_by_member = build_blocking_anchors(
            input_index=input_index,
            members=members,
            assessments=assessments,
            explicit_decisions=explicit_decisions,
            current_source=current_source,
            observations=observations,
        )
        plan = plan_candidate_grouping(
            members,
            anchors_by_member=anchors_by_member,
            limits=_grouping_limits(settings),
            mode=settings.candidate_grouping_mode,
        )
        relations = []
        members_by_id = {item.member_id: item for item in plan.members}
        direct_batches = plan_grouping_batches(
            plan,
            plan.semantic_pairs,
            phase="direct",
        )
        grouping_planned_pairs += len(plan.semantic_pairs)
        grouping_planned_calls += len(direct_batches.batches)
        (
            direct_relations,
            failed_direct_pairs,
            direct_records,
        ) = _run_grouping_batch_plan(
            batch_plan=direct_batches,
            members_by_id=members_by_id,
            llm_settings=llm_settings,
            client=client,
            budget=budget,
            protocol_manifest_digest=protocol_manifest_digest,
            max_request_bytes=settings.max_group_request_bytes,
        )
        relations.extend(direct_relations)
        inference_records.extend(direct_records)
        grouping_terminal_calls += len(direct_records)
        plan, closure_pairs = resolve_direct_grouping(
            plan,
            relations,
            failed_pairs=sorted(failed_direct_pairs),
        )
        closure_batches = plan_grouping_batches(
            plan,
            closure_pairs,
            phase="closure",
        )
        grouping_planned_pairs += len(closure_pairs)
        grouping_planned_calls += len(closure_batches.batches)
        (
            closure_relations,
            failed_closure_pairs,
            closure_records,
        ) = _run_grouping_batch_plan(
            batch_plan=closure_batches,
            members_by_id=members_by_id,
            llm_settings=llm_settings,
            client=client,
            budget=budget,
            protocol_manifest_digest=protocol_manifest_digest,
            max_request_bytes=settings.max_group_request_bytes,
        )
        relations.extend(closure_relations)
        inference_records.extend(closure_records)
        grouping_terminal_calls += len(closure_records)
        if failed_closure_pairs:
            plan = mark_grouping_pairs_incomplete(
                plan,
                sorted(failed_closure_pairs),
                reason_code="candidate_semantic_group_failed",
            )
        grouping = assemble_candidate_cases(
            plan,
            relations,
            profile_digest=profile_digest,
            required_capabilities=("current_evidence",),
        )
        cases = grouping.cases
        if (
            settings.candidate_grouping_mode == "semantic"
            and len(cases) > settings.max_cases
        ):
            bounded_cases: list[CandidateCase] = []
            for ordinal, case in enumerate(cases):
                if ordinal < settings.max_cases:
                    bounded_cases.append(case)
                    continue
                payload = case.model_dump(
                    mode="python", exclude={"artifact_digest"}
                )
                payload.update(
                    {
                        "grouping_status": "incomplete",
                        "grouping_reason_codes": sorted(
                            {
                                *case.grouping_reason_codes,
                                "max_cases_exceeded",
                            }
                        ),
                    }
                )
                bounded_cases.append(
                    CandidateCase.model_validate(seal_payload(payload))
                )
            cases = tuple(bounded_cases)

    if settings.candidate_grouping_mode == "singleton":
        grouping_execution = CandidateGroupingExecution(
            mode="singleton",
            status="not_required",
            reason="candidate_grouping_mode_singleton",
            planned_pairs=grouping_planned_pairs,
            planned_calls=grouping_planned_calls,
            terminal_calls=grouping_terminal_calls,
            case_count=len(cases),
        )
    elif not members:
        grouping_execution = CandidateGroupingExecution(
            mode="semantic",
            status="not_required",
            reason="no_candidate_members",
            planned_pairs=0,
            planned_calls=0,
            terminal_calls=0,
            case_count=0,
        )
    else:
        grouping_execution = CandidateGroupingExecution(
            mode="semantic",
            status=(
                "complete"
                if grouping_terminal_calls == grouping_planned_calls
                and all(item.grouping_status == "complete" for item in cases)
                else "partial"
            ),
            reason=None,
            planned_pairs=grouping_planned_pairs,
            planned_calls=grouping_planned_calls,
            terminal_calls=grouping_terminal_calls,
            case_count=len(cases),
        )

    initial_evidence: dict[str, CurrentEvidenceBundle] = {}
    evidence_inputs_by_case: dict[str, CaseEvidenceInputs] = {}
    observation_anchor_index = build_observation_anchor_index(observations)
    for case in cases:
        inputs = build_case_evidence_inputs(
            input_index=input_index,
            case=case,
            assessments=assessments,
            explicit_decisions=explicit_decisions,
            current_source=current_source,
            observations=observations,
            explicit_discovery_complete=explicit_discovery_complete,
            anchors_by_member=anchors_by_member,
            observation_anchor_index=observation_anchor_index,
            max_observation_context_items=(
                settings.max_observation_context_items_per_case
            ),
        )
        evidence_inputs_by_case[case.case_id] = inputs
        initial_evidence[case.case_id] = assemble_current_evidence(
            case,
            explicit_evidence=[
                *inputs.explicit_evidence,
                *inputs.observation_evidence,
            ],
            population_evidence=inputs.population_evidence,
            denominators=inputs.denominators,
            population_refs=inputs.population_refs,
            falsification_refs=inputs.falsification_refs,
            scope_condition_refs=inputs.scope_condition_refs,
            dependencies=inputs.dependencies,
            limits=CurrentEvidenceLimits(
                max_items=settings.max_current_evidence_items,
                max_bytes=settings.max_current_evidence_bytes,
            ),
        ).bundle

    routes = {
        case.case_id: _route_case(
            case, assessment_by_id, initial_evidence[case.case_id]
        )
        for case in cases
    }
    git_limits = _git_limits(settings)
    needs_git = any(
        route in {"augmentative", "ambiguity_resolving"}
        for route in routes.values()
    )
    runner = RunLocalGitRunner(BoundedGitRunner(), snapshot_digest=snapshot_digest)
    shared_preflight: GitPreflight | None = None
    if (
        needs_git
        and settings.git_enabled
        and snapshot_commit is not None
        and budget.terminal_exhaustion_reason() is None
    ):
        shared_preflight = preflight_git_repository(
            repository_root,
            snapshot_commit,
            limits=git_limits,
            runner=runner,
        )
    git_bundles: list[GitTemporalBundle] = []
    git_by_case: dict[str, GitTemporalBundle] = {}
    temporal_by_case: dict[str, TemporalRelationMatrix] = {}
    temporal_status_by_case: dict[str, str] = {}
    git_budget_skip_by_case: dict[str, str | None] = {}
    for case in cases:
        route = routes[case.case_id]
        evidence = initial_evidence[case.case_id]
        git_bundle: GitTemporalBundle | None = None
        git_budget_skip_reason: str | None = None
        if snapshot_commit is not None:
            # A not-required route is a real absence of the temporal
            # dependency, not a zero-result Git search.  In particular the
            # The explicit local-positive fast path must not pay for preflight or
            # object traversal and must not manufacture a query manifest.
            if not route.startswith("not_required") and settings.git_enabled:
                git_budget_skip_reason = budget.terminal_exhaustion_reason()
                if git_budget_skip_reason is None:
                    git_bundle = collect_git_evidence(
                        repository_root,
                        case_id=case.case_id,
                        snapshot_digest=snapshot_digest,
                        cutoff_commit=snapshot_commit,
                        purpose=route,  # type: ignore[arg-type]
                        anchors=_git_anchors(
                            case,
                            evidence,
                            max_blame_lines=git_limits.max_blame_lines,
                        ),
                        required=True,
                        limits=git_limits,
                        runner=runner,
                        preflight=shared_preflight,
                    )
        if git_bundle is not None:
            git_bundles.append(git_bundle)
            git_by_case[case.case_id] = git_bundle
        git_budget_skip_by_case[case.case_id] = git_budget_skip_reason

    # Freeze every round-zero case/batch unit before starting this provider
    # stage.  The semantic executor flattens the resulting plan in stable
    # (case ordinal, batch ordinal) order so a wave can span small cases while
    # still folding records independently of provider completion order.
    temporal_inputs = [
        (case, git_by_case[case.case_id])
        for case in cases
        if case.case_id in git_by_case and git_by_case[case.case_id].slices
    ]
    temporal_results = temporal_evidence_classify_cases_bounded(
        temporal_inputs,
        max_batch_items=settings.max_temporal_batch_items,
        max_cells_per_request=settings.max_temporal_cells_per_request,
        max_total_items=settings.max_temporal_items_per_case,
        max_concurrency=llm_settings.max_concurrency,
        llm_settings=llm_settings,
        client=client,
        budget=budget,
        protocol_manifest_digest=protocol_manifest_digest,
        max_request_bytes=settings.max_temporal_request_bytes,
    )
    temporal_result_by_case = {
        case.case_id: result
        for (case, _), result in zip(
            temporal_inputs,
            temporal_results,
            strict=True,
        )
    }
    for case in cases:
        route = routes[case.case_id]
        evidence = initial_evidence[case.case_id]
        git_bundle = git_by_case.get(case.case_id)
        matrix: TemporalRelationMatrix | None = None
        temporal_result = temporal_result_by_case.get(case.case_id)
        if temporal_result is not None:
            matrix, records = temporal_result
            inference_records.extend(records)
            if matrix is not None:
                temporal_by_case[case.case_id] = matrix
        temporal_dependency = _dependency_for_temporal(
            route,
            git_bundle,
            matrix,
            host_skip_reason=git_budget_skip_by_case[case.case_id],
        )
        temporal_status_by_case[case.case_id] = temporal_dependency.status
        initial_evidence[case.case_id] = _with_dependencies(
            evidence,
            [
                _dependency_for_git(
                    route,
                    git_bundle,
                    host_skip_reason=git_budget_skip_by_case[case.case_id],
                ),
                temporal_dependency,
            ],
        )

    base_cases = tuple(cases)
    all_cases = list(cases)
    findings: list[UnifiedCaseFinding] = []
    policy_accumulator = CanonicalPolicyAccumulator()
    refinements = []
    verifier_status_by_case: dict[str, str] = {}
    refinement_statuses: list[str] = []
    refinement_reasons: set[str] = set()
    preflight_holder: list[GitPreflight | None] = [shared_preflight]

    def absorb_policy(
        policy_result: object,
        *,
        child_member_ids: set[str] | None = None,
        prefer_child_dispositions: bool = False,
        supersedes_case_id: str | None = None,
    ) -> None:
        policy_accumulator.absorb(
            policy_result,  # type: ignore[arg-type]
            include_claim_member_ids=child_member_ids,
            prefer_dispositions=prefer_child_dispositions,
            supersedes_case_id=supersedes_case_id,
        )

    # Round-zero verification is an independent phase.  Host-resolved cases
    # are closed without a call; every remaining case is executed in bounded
    # waves and merged back in the stable case plan before any refinement can
    # consume evidence or budget.
    round_zero_findings: dict[str, UnifiedCaseFinding] = {}
    round_zero_policy_findings: dict[str, UnifiedCaseFinding | None] = {}
    refinement_quarantine_cases: set[str] = set()
    verifier_work: list[CandidateCase] = []
    for case in base_cases:
        evidence = initial_evidence[case.case_id]
        route = routes[case.case_id]
        if route == "not_required_fast_path":
            finding = _host_finding(
                case,
                evidence,
                profile_digest=profile_digest,
                status="not_required",
                reason_code="explicit_local_positive_fast_path",
            )
            round_zero_findings[case.case_id] = finding
            round_zero_policy_findings[case.case_id] = None
            verifier_status_by_case[case.case_id] = "not_required"
        elif route == "not_required_hard_invalid":
            finding = _host_finding(
                case,
                evidence,
                profile_digest=profile_digest,
                status="not_required",
                reason_code="all_members_hard_invalid",
            )
            round_zero_findings[case.case_id] = finding
            round_zero_policy_findings[case.case_id] = None
            verifier_status_by_case[case.case_id] = "not_required"
        elif route == "not_required_current_incomplete":
            finding = _host_finding(
                case,
                evidence,
                profile_digest=profile_digest,
                status="not_required",
                reason_code="required_current_capability_incomplete",
                detail_reason_codes=tuple(evidence.incomplete_reasons),
            )
            round_zero_findings[case.case_id] = finding
            # The host finding is an audit closure, not a semantic substitute
            # for the missing required evidence.  Policy derives the typed
            # review directly from the dependency state.
            round_zero_policy_findings[case.case_id] = None
            verifier_status_by_case[case.case_id] = "not_required"
        elif case.grouping_status == "incomplete" or evidence.integrity == "invalid":
            reason = (
                "candidate_grouping_incomplete"
                if case.grouping_status == "incomplete"
                else "current_evidence_invalid"
            )
            finding = _host_finding(
                case,
                evidence,
                profile_digest=profile_digest,
                status="failed",
                reason_code=reason,
            )
            round_zero_findings[case.case_id] = finding
            round_zero_policy_findings[case.case_id] = finding
            verifier_status_by_case[case.case_id] = "failed"
        else:
            verifier_work.append(case)

    def verify_round_zero(
        case: CandidateCase,
    ) -> tuple[UnifiedCaseFinding | None, InferenceRecord]:
        return unified_case_verify(
            case,
            [assessment_by_id[item.assessment_id] for item in case.members],
            initial_evidence[case.case_id],
            temporal_by_case.get(case.case_id),
            verification_round=0,
            llm_settings=llm_settings,
            client=client,
            budget=budget,
            protocol_manifest_digest=protocol_manifest_digest,
            max_request_bytes=settings.max_case_input_bytes,
            _stage_already_planned=True,
        )

    verifier_stage = run_budgeted_stage_calls(
        verifier_work,
        verify_round_zero,
        stage="unified_case_verify",
        budget=budget,
        max_concurrency=llm_settings.max_concurrency,
        thread_name_prefix="repo-memory-v3-unified-verify",
    )
    for case, (finding_value, record) in zip(
        verifier_work, verifier_stage.outcomes
    ):
        inference_records.append(record)
        if (
            finding_value is not None
            and isinstance(record.structured_response, Mapping)
            and record.structured_response.get("refinement_quarantine")
            is not None
        ):
            refinement_quarantine_cases.add(case.case_id)
        if finding_value is None:
            finding = _host_finding(
                case,
                initial_evidence[case.case_id],
                profile_digest=profile_digest,
                status="failed",
                reason_code="unified_case_verifier_failed",
            )
            verifier_status_by_case[case.case_id] = "failed"
        else:
            finding = finding_value
            verifier_status_by_case[case.case_id] = finding.status
        round_zero_findings[case.case_id] = finding
        round_zero_policy_findings[case.case_id] = finding
    omitted_reason = verifier_stage.exhaustion_reason or "stage_execution_closed"
    for case in verifier_work[len(verifier_stage.outcomes) :]:
        finding = _host_finding(
            case,
            initial_evidence[case.case_id],
            profile_digest=profile_digest,
            status="failed",
            reason_code="unified_case_verifier_budget_exhausted",
            detail_reason_codes=(omitted_reason,),
        )
        round_zero_findings[case.case_id] = finding
        round_zero_policy_findings[case.case_id] = finding
        verifier_status_by_case[case.case_id] = "failed"

    for case in base_cases:
        evidence = initial_evidence[case.case_id]
        finding = round_zero_findings[case.case_id]
        verifier_for_policy = round_zero_policy_findings[case.case_id]
        findings.append(finding)
        policy_result = apply_case_policy(
            case=case,
            assessments=assessment_by_id,
            current_evidence=evidence,
            temporal_matrix=temporal_by_case.get(case.case_id),
            git_bundle=git_by_case.get(case.case_id),
            finding=verifier_for_policy,
        )
        absorb_policy(policy_result)

        if case.case_id in refinement_quarantine_cases:
            # The optional suggestion was audited and closed by the verifier;
            # the valid parent finding remains complete and no child is run.
            refinement_statuses.append("complete")
        if not finding.refinement_proposals:
            continue

        parent_inputs = evidence_inputs_by_case[case.case_id]

        class _CurrentReacquirer:
            def reacquire_current(self, **kwargs: object) -> CurrentEvidenceReacquisition:
                return _reacquire_refined_current_from_saved_inputs(
                    parent_case=kwargs["parent_case"],  # type: ignore[arg-type]
                    refined_case=kwargs["refined_case"],  # type: ignore[arg-type]
                    parent=kwargs["parent"],  # type: ignore[arg-type]
                    child_assessments=kwargs["child_assessments"],  # type: ignore[arg-type]
                    proposals=kwargs["proposals"],  # type: ignore[arg-type]
                    previous_evidence=kwargs["previous_evidence"],  # type: ignore[arg-type]
                    inputs=parent_inputs,
                    limits=CurrentEvidenceLimits(
                        max_items=settings.max_current_evidence_items,
                        max_bytes=settings.max_current_evidence_bytes,
                    ),
                )

        class _GitReacquirer:
            def reacquire_git(self, **kwargs: object) -> GitReacquisition:
                refined_case = kwargs["refined_case"]
                refined_assessments = kwargs["assessments"]
                refined_current = kwargs["current_evidence"]
                assert isinstance(refined_case, CandidateCase)
                assert isinstance(refined_current, CurrentEvidenceBundle)
                local_assessments = {
                    item.assessment_id: item for item in refined_assessments  # type: ignore[union-attr]
                }
                refined_route = _route_case(
                    refined_case, local_assessments, refined_current
                )
                routes[refined_case.case_id] = refined_route
                git_bundle: GitTemporalBundle | None = None
                git_budget_skip_reason: str | None = None
                if (
                    not refined_route.startswith("not_required")
                    and settings.git_enabled
                    and snapshot_commit is not None
                ):
                    git_budget_skip_reason = budget.terminal_exhaustion_reason()
                    if git_budget_skip_reason is None:
                        if preflight_holder[0] is None:
                            preflight_holder[0] = preflight_git_repository(
                                repository_root,
                                snapshot_commit,
                                limits=git_limits,
                                runner=runner,
                            )
                        git_bundle = collect_git_evidence(
                            repository_root,
                            case_id=refined_case.case_id,
                            snapshot_digest=snapshot_digest,
                            cutoff_commit=snapshot_commit,
                            purpose=refined_route,  # type: ignore[arg-type]
                            anchors=_git_anchors(
                                refined_case,
                                refined_current,
                                max_blame_lines=git_limits.max_blame_lines,
                            ),
                            required=True,
                            limits=git_limits,
                            runner=runner,
                            preflight=preflight_holder[0],
                        )
                        git_bundles.append(git_bundle)
                        git_by_case[refined_case.case_id] = git_bundle
                matrix: TemporalRelationMatrix | None = None
                records: tuple[InferenceRecord, ...] = ()
                if git_bundle is not None and git_bundle.slices:
                    matrix, matrix_records = temporal_evidence_classify_bounded(
                        refined_case,
                        git_bundle,
                        max_batch_items=settings.max_temporal_batch_items,
                        max_cells_per_request=(
                            settings.max_temporal_cells_per_request
                        ),
                        max_total_items=settings.max_temporal_items_per_case,
                        max_concurrency=llm_settings.max_concurrency,
                        llm_settings=llm_settings,
                        client=client,
                        budget=budget,
                        protocol_manifest_digest=protocol_manifest_digest,
                        max_request_bytes=settings.max_temporal_request_bytes,
                    )
                    records = tuple(matrix_records)
                    if matrix is not None:
                        temporal_by_case[refined_case.case_id] = matrix
                git_dependency = _dependency_for_git(
                    refined_route,
                    git_bundle,
                    host_skip_reason=git_budget_skip_reason,
                )
                temporal_dependency = _dependency_for_temporal(
                    refined_route,
                    git_bundle,
                    matrix,
                    host_skip_reason=git_budget_skip_reason,
                )
                temporal_status_by_case[refined_case.case_id] = (
                    temporal_dependency.status
                )
                dependencies = (git_dependency, temporal_dependency)
                gaps = [
                    item
                    for item in dependencies
                    if item.requirement_mode == "required"
                    and item.status
                    not in {"complete", "bounded_no_match", "not_required"}
                ]
                return GitReacquisition(
                    git_bundle_refs=(
                        ()
                        if git_bundle is None
                        else _refs_for_git_bundle(git_bundle)
                    ),
                    temporal_matrix=matrix,
                    complete=not gaps,
                    reason_codes=tuple(
                        sorted(
                            {
                                reason
                                for dependency in gaps
                                for reason in dependency.reason_codes
                            }
                        )
                    ),
                    dependencies=dependencies,
                    inference_records=records,
                )

        class _RoundOneVerifier:
            def verify_round_one_case(self, **kwargs: object) -> RoundOneVerification:
                exhausted = budget.terminal_exhaustion_reason()
                if exhausted is not None:
                    budget.plan_stage("unified_case_verify", 1)
                    budget.record_omitted(
                        "unified_case_verify", 1, reason=exhausted
                    )
                    return RoundOneVerification(
                        finding=None,
                        inference_record=None,
                        reason_codes=(
                            "round_one_verifier_budget_exhausted",
                            exhausted,
                        ),
                    )
                value, record = unified_case_verify(
                    kwargs["case"],  # type: ignore[arg-type]
                    kwargs["assessments"],  # type: ignore[arg-type]
                    kwargs["current_evidence"],  # type: ignore[arg-type]
                    kwargs["temporal_matrix"],  # type: ignore[arg-type]
                    verification_round=1,
                    llm_settings=llm_settings,
                    client=client,
                    budget=budget,
                    protocol_manifest_digest=protocol_manifest_digest,
                    max_request_bytes=settings.max_case_input_bytes,
                )
                return RoundOneVerification(
                    finding=value,
                    inference_record=record,
                    reason_codes=(
                        () if value is not None else ("round_one_verifier_failed",)
                    ),
                )

        execution = execute_refinement_round(
            case=case,
            assessments=assessment_by_id,
            current_evidence=evidence,
            proposals=finding.refinement_proposals,
            parent_finding=finding,
            max_children=settings.max_refinement_children,
            current_reacquirer=_CurrentReacquirer(),
            git_reacquirer=_GitReacquirer(),
            verifier=_RoundOneVerifier(),
        )
        assessments.extend(execution.child_assessments)
        assessment_by_id.update(
            {item.assessment_id: item for item in execution.child_assessments}
        )
        materialized_child_cases = (
            execution.child_cases
            if execution.status == "complete"
            else tuple(
                _with_case_marker(item, "refinement_review_branch")
                for item in execution.child_cases
            )
        )
        if execution.status == "complete":
            all_cases = [
                (
                    _with_case_marker(item, "refinement_parent_superseded")
                    if item.case_id == case.case_id
                    else item
                )
                for item in all_cases
            ]
        all_cases.extend(materialized_child_cases)
        for item in execution.current_evidence_bundles:
            initial_evidence[item.case_id] = item
        for item in materialized_child_cases:
            if item.case_id not in routes:
                routes[item.case_id] = _route_case(item, assessment_by_id, initial_evidence[item.case_id])
        for item in execution.temporal_matrices:
            temporal_by_case[item.case_id] = item
        findings.extend(execution.findings)
        refinements.extend(execution.refinement_records)
        inference_records.extend(execution.inference_records)
        child_member_ids = {
            item.member_id for item in execution.child_members
        }
        finding_by_case = {item.case_id: item for item in execution.findings}
        for child_case in materialized_child_cases:
            child_evidence = initial_evidence[child_case.case_id]
            child_finding = finding_by_case[child_case.case_id]
            verifier_status_by_case[child_case.case_id] = child_finding.status
            child_policy = apply_case_policy(
                case=child_case,
                assessments=assessment_by_id,
                current_evidence=child_evidence,
                temporal_matrix=temporal_by_case.get(child_case.case_id),
                git_bundle=git_by_case.get(child_case.case_id),
                finding=child_finding,
            )
            absorb_policy(
                child_policy,
                child_member_ids=(
                    None if execution.status == "complete" else set()
                ),
                prefer_child_dispositions=execution.status == "complete",
                supersedes_case_id=(
                    case.case_id if execution.status == "complete" else None
                ),
            )
        refinement_statuses.extend(
            "complete" if item.status == "complete" else "partial"
            for item in execution.refinement_records
        )
        if execution.status != "complete":
            refinement_reasons.update(execution.reason_codes)
            if not execution.refinement_records:
                refinement_statuses.append("partial")
        for record in execution.refinement_records:
            if record.status != "complete":
                continue
            policy_accumulator.supersede_member(
                parent_member_id=record.parent_member_id,
                child_member_ids=record.child_member_ids,
            )

    cases = tuple(sorted(all_cases, key=lambda item: item.case_id))
    policy_aggregation = policy_accumulator.finalize()
    claims = list(policy_aggregation.claims)
    dispositions = list(policy_aggregation.dispositions)
    findings.sort(key=lambda item: item.finding_id)
    git_bundles.sort(key=lambda item: item.case_id)

    git_refs: list[ImmutableArtifactRef] = []
    git_artifacts: dict[str, object] = {}
    for git_bundle in git_bundles:
        values: list[tuple[str, object, str, str]] = [
            (
                "git_temporal_bundle",
                git_bundle,
                git_bundle.bundle_id,
                git_bundle.artifact_digest,
            ),
            (
                "git_manifest",
                git_bundle.manifest,
                git_bundle.manifest.manifest_id,
                git_bundle.manifest.artifact_digest,
            ),
            *[
                (
                    "git_change_slice",
                    item,
                    item.item_id,
                    item.artifact_digest,
                )
                for item in git_bundle.slices
            ],
        ]
        for kind, value, stable_id_value, digest in values:
            git_refs.append(
                ImmutableArtifactRef(
                    kind=kind,  # type: ignore[arg-type]
                    stable_id=stable_id_value,
                    snapshot_digest=snapshot_digest,
                    artifact_path=(
                        "canonical/audit/git/"
                        + {
                            "git_temporal_bundle": "bundles",
                            "git_manifest": "manifests",
                            "git_change_slice": "slices",
                        }[kind]
                        + f"/{stable_id_value}.json"
                    ),
                    artifact_digest=digest,
                )
            )
            git_artifacts[stable_id_value] = value
    git_refs.sort(key=lambda item: (item.kind, item.stable_id))

    grouping_statuses = [item.grouping_status for item in cases]
    current_statuses = [item.status for item in initial_evidence.values()]
    git_dependencies = [
        dependency
        for evidence in initial_evidence.values()
        for dependency in evidence.dependencies
        if dependency.capability == "git_temporal_evidence"
        and dependency.requirement_mode != "not_required"
    ]
    git_statuses = [item.status for item in git_dependencies]
    git_requirement_mode = (
        "required"
        if any(item.requirement_mode == "required" for item in git_dependencies)
        else "augmentative"
    )
    temporal_dependencies = [
        dependency
        for evidence in initial_evidence.values()
        for dependency in evidence.dependencies
        if dependency.capability == "temporal_evidence_classification"
        and dependency.requirement_mode != "not_required"
    ]
    temporal_statuses = list(temporal_status_by_case.values())
    temporal_requirement_mode = (
        "required"
        if any(item.requirement_mode == "required" for item in temporal_dependencies)
        else "augmentative"
    )
    verifier_statuses = list(verifier_status_by_case.values())
    current_channel_status = (
        "complete"
        if current_source is not None and current_source.index.status == "complete"
        else "partial"
        if current_source_required or current_source is not None
        else "not_required"
    )
    capabilities = [
        CapabilityDependency(
            capability="channel_assessment_construction",
            requirement_mode="required",
            status=(
                "partial"
                if any(item.affects_completeness for item in assessment_omissions)
                else "complete"
            ),
            reason_codes=(
                ["channel_assessment_construction_incomplete"]
                if any(item.affects_completeness for item in assessment_omissions)
                else []
            ),
        ),
        CapabilityDependency(
            capability="explicit_candidate_discovery",
            requirement_mode="required",
            status="complete" if explicit_discovery_complete else "partial",
            reason_codes=(
                []
                if explicit_discovery_complete
                else ["explicit_candidate_discovery_incomplete"]
            ),
        ),
        CapabilityDependency(
            capability="explicit_semantic_verification",
            requirement_mode="required" if explicit_verification is not None else "not_required",
            status=(explicit_verification.coverage.status if explicit_verification is not None else "not_required"),
            reason_codes=(
                ["explicit_semantic_verification_incomplete"]
                if explicit_verification is not None and explicit_verification.coverage.status != "complete"
                else []
            ),
        ),
        CapabilityDependency(
            capability="current_candidate_discovery",
            requirement_mode=(
                "required" if current_source_required
                else "augmentative" if current_source is not None
                else "not_required"
            ),
            status=current_channel_status,
            reason_codes=(
                ["current_candidate_discovery_incomplete"]
                if current_channel_status == "partial"
                else []
            ),
        ),
        _capability(
            "candidate_grouping",
            ["complete" if item == "complete" else "partial" for item in grouping_statuses],
            reasons=[
                reason for case in cases for reason in case.grouping_reason_codes
            ],
        ),
        CapabilityDependency(
            capability="canonical_identity_resolution",
            requirement_mode="required",
            status=(
                "partial"
                if policy_aggregation.collision_claim_ids
                else "complete"
            ),
            reason_codes=list(policy_aggregation.reason_codes),
        ),
        _capability(
            "current_evidence",
            current_statuses,
            reasons=[
                reason
                for item in initial_evidence.values()
                for reason in item.incomplete_reasons
            ],
        ),
        _capability(
            "git_temporal_evidence",
            git_statuses,
            requirement_mode=git_requirement_mode,
            reasons=_git_temporal_gap_reasons(git_bundles),
        ),
        _capability(
            "temporal_evidence_classification",
            temporal_statuses,
            requirement_mode=temporal_requirement_mode,
        ),
        _capability(
            "unified_case_verification",
            verifier_statuses,
        ),
        _capability(
            "unified_refinement",
            refinement_statuses,
            reasons=sorted(refinement_reasons),
        ),
    ]
    capabilities.sort(key=lambda item: item.capability)
    gap_capabilities = [
        item
        for item in capabilities
        if item.status not in {"complete", "bounded_no_match", "not_required"}
    ]
    package_status = "partial" if gap_capabilities else "complete"
    completeness = UnifiedCompleteness(
        structurally_valid=True,
        package_status=package_status,
        capabilities=capabilities,
        omitted_possible=package_status == "partial",
        reasons=sorted(
            {
                reason
                for item in gap_capabilities
                for reason in item.reason_codes
            }
        ),
    )

    canonical_counts = Counter(item.epistemic_status for item in claims)
    initial_roots_by_channel = Counter(
        item.channel
        for item in assessments
        if "unified_refinement_child" not in item.typed_reasons
    )
    initial_roots_by_channel.update(
        item.channel
        for item in assessment_omissions
        if item.candidate_version == 0
    )
    recorded_current_raw_events = (
        current_source.index.candidate_events if current_source is not None else 0
    )
    if raw_candidate_events is None:
        current_raw_events = max(
            recorded_current_raw_events,
            initial_roots_by_channel["implicit_current"],
        )
        explicit_raw_events = max(
            sum(item.raw_candidate is not None for item in explicit_decisions),
            initial_roots_by_channel["explicit"],
        )
    else:
        current_raw_events = recorded_current_raw_events
        explicit_raw_events = raw_candidate_events - current_raw_events
    if explicit_raw_events < 0:
        raise ValueError("raw candidate event total is below current-source events")
    funnel = _build_unified_funnel(
        cases=all_cases,
        current_evidence=list(initial_evidence.values()),
        git_bundles=git_bundles,
        assessments=assessments,
        claims=claims,
        dispositions=dispositions,
        grouping_execution=grouping_execution,
        raw_candidate_events_by_channel={
            "explicit": explicit_raw_events,
            "implicit_current": current_raw_events,
        },
        assessment_omissions=assessment_omissions,
    )
    artifact_digests = sorted(
        [
            *(item.artifact_digest for item in assessments),
            *(item.artifact_digest for item in cases),
            *(item.artifact_digest for item in initial_evidence.values()),
            *(item.artifact_digest for item in temporal_by_case.values()),
            *(item.artifact_digest for item in findings),
            *(item.artifact_digest for item in refinements),
            *(item.artifact_digest for item in claims),
            *(item.artifact_digest for item in git_bundles),
            content_digest(funnel.model_dump(mode="json")),
            content_digest(completeness.model_dump(mode="json")),
        ]
    )
    bundle_id = make_bundle_id(snapshot_digest, profile_digest)
    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": UNIFIED_CONTRACT_VERSION,
        "policy_version": UNIFIED_POLICY_VERSION,
        "bundle_id": bundle_id,
        "snapshot_digest": snapshot_digest,
        "snapshot_commit": snapshot_commit,
        "profile_digest": profile_digest,
        "channel_assessment_ids": sorted(item.assessment_id for item in assessments),
        "case_ids": sorted(item.case_id for item in cases),
        "finding_ids": sorted(item.finding_id for item in findings),
        "canonical_claim_ids": sorted(item.claim_id for item in claims),
        "accepted_count": canonical_counts["accepted"],
        "review_count": canonical_counts["review"],
        "rejected_count": canonical_counts["rejected"],
        "artifact_set_digest": content_digest(artifact_digests),
    }
    index = UnifiedIndex.model_validate(seal_payload(index_payload))
    bundle_payload = {
        "index": index,
        "assessments": list(assessments),
        "cases": list(cases),
        "current_evidence": sorted(
            initial_evidence.values(), key=lambda item: item.case_id
        ),
        "git_bundle_refs": git_refs,
        "temporal_matrices": sorted(
            temporal_by_case.values(), key=lambda item: item.case_id
        ),
        "findings": findings,
        "refinements": sorted(
            refinements, key=lambda item: item.refinement_id
        ),
        "canonical_claims": claims,
        "member_dispositions": dispositions,
        "funnel": funnel,
        "completeness": completeness,
    }
    bundle = UnifiedBundle.model_validate(seal_payload(bundle_payload))
    if len(canonical_bytes(bundle)) > settings.max_unified_artifact_bytes:
        raise ValueError("unified bundle exceeds max_unified_artifact_bytes")
    return UnifiedRunResult(
        bundle=bundle,
        git_bundles=tuple(git_bundles),
        git_artifacts=git_artifacts,
        inference_records=tuple(inference_records),
        grouping_plan=plan,
        grouping_execution=grouping_execution,
        case_routes=dict(sorted(routes.items())),
    )


__all__ = [
    "UNIFIED_GIT_ROUTING_VERSION",
    "CandidateGroupingExecution",
    "UnifiedRunResult",
    "run_unified_layer",
]
