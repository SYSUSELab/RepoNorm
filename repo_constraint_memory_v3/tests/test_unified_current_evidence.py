from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repo_constraint_memory_v3.unified.current_evidence import (
    CurrentEvidenceLimits,
    EvidenceContribution,
    PopulationEvidenceContribution,
    assemble_current_evidence,
)
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    content_digest,
    make_assessment_id,
    make_case_id,
    make_member_id,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CandidateCaseMember,
    CandidateMemberRelation,
    CandidateSemantics,
    CapabilityDependency,
    ImmutableArtifactRef,
    PopulationDenominator,
)

SNAPSHOT = "sha256:" + "c" * 64
PROFILE = "d" * 64


def _ref(kind: str, stable_id: str, seed: str) -> ImmutableArtifactRef:
    return ImmutableArtifactRef(
        kind=kind,
        stable_id=stable_id,
        snapshot_digest=SNAPSHOT,
        artifact_path=f"upstream/{seed}.json",
        artifact_digest=content_digest([kind, stable_id, seed]),
    )


def _member(seed: str, channel: str) -> CandidateCaseMember:
    assessment_id = make_assessment_id(
        SNAPSHOT, PROFILE, channel, f"ROOT-{seed}", 0
    )
    return CandidateCaseMember(
        member_id=make_member_id(SNAPSHOT, assessment_id),
        assessment_id=assessment_id,
        snapshot_digest=SNAPSHOT,
        channel=channel,
        semantics=CandidateSemantics(
            rule="Register plugins through PluginManager",
            normalized_action="register-plugins-through-manager",
            polarity="positive",
            applies_to=["src/**"],
            condition="when adding a plugin",
            exception=None,
        ),
        assessment_ref=_ref(
            "explicit_candidate" if channel == "explicit" else "norm_hypothesis",
            assessment_id,
            seed,
        ),
    )


def _case(*members: CandidateCaseMember) -> CandidateCase:
    ordered = sorted(members, key=lambda item: item.member_id)
    relations = []
    if len(ordered) == 2:
        relations.append(
            CandidateMemberRelation(
                left_member_id=ordered[0].member_id,
                right_member_id=ordered[1].member_id,
                relation="related_variant",
                source="semantic_grouping",
                reason_code="same_plugin_decision",
            )
        )
    payload: dict[str, object] = {
        "case_id": make_case_id(SNAPSHOT, [item.member_id for item in ordered]),
        "snapshot_digest": SNAPSHOT,
        "profile_digest": PROFILE,
        "members": ordered,
        "relations": relations,
        "grouping_status": "complete",
        "grouping_reason_codes": [],
        "exact_identity_groups": [],
        "required_capabilities": ["current_evidence"],
    }
    payload["artifact_digest"] = artifact_digest(payload)
    return CandidateCase.model_validate(payload)


def _evidence(
    member_id: str,
    *,
    seed: str,
    role: str,
    origin_kind: str,
    path: str,
    line: int,
    source_digest: str | None = None,
    independence_group: str | None = None,
    artifact_kind: str = "population_member",
    artifact_id: str | None = None,
    anchor_literals: tuple[str, ...] = (),
    anchor_symbols: tuple[str, ...] = (),
) -> EvidenceContribution:
    digest = source_digest or content_digest(["source", seed])
    return EvidenceContribution(
        origin_kind=origin_kind,
        path=path,
        line_start=line,
        line_end=line,
        source_content_digest=digest,
        independence_group=independence_group or f"independent-{seed}",
        role=role,
        member_ids=(member_id,),
        artifact_ref=_ref(
            artifact_kind, artifact_id or f"ART-{seed}", f"evidence-{seed}"
        ),
        excerpt_digest=content_digest(["excerpt", seed]),
        description=f"Evidence occurrence {seed}",
        anchor_literals=anchor_literals,
        anchor_symbols=anchor_symbols,
    )


def _population(
    member_id: str,
    *,
    population_id: str,
    seed: str,
    label: str,
    path: str,
    line: int,
    source_digest: str | None = None,
    independence_group: str | None = None,
) -> PopulationEvidenceContribution:
    role = "counterevidence" if label == "counterexample" else label
    return PopulationEvidenceContribution(
        case_member_id=member_id,
        population_id=population_id,
        population_member_id=f"POPMEMBER-{seed}",
        label=label,
        evidence=_evidence(
            member_id,
            seed=seed,
            role=role,
            origin_kind="population_member",
            path=path,
            line=line,
            source_digest=source_digest,
            independence_group=independence_group,
            artifact_id=f"POPMEMBER-{seed}",
        ),
    )


def _denominator(
    member_id: str,
    population_id: str,
    *,
    support: int,
    counterexample: int = 0,
    unknown: int = 0,
    ineligible: int = 0,
    status: str = "complete",
) -> PopulationDenominator:
    serialized = support + counterexample + unknown + ineligible
    return PopulationDenominator(
        member_id=member_id,
        population_id=population_id,
        status=status,
        census_enumeration_status=status,
        sampling_plan_status="complete",
        sampling_mode="census",
        scanned=serialized,
        serialized=serialized,
        matched=serialized,
        inspected=serialized,
        uninspected=0,
        support=support,
        counterexample=counterexample,
        unknown=unknown,
        ineligible=ineligible,
    )


def _mandatory_refs(
    population_id: str,
) -> tuple[list[ImmutableArtifactRef], list[ImmutableArtifactRef], list[ImmutableArtifactRef]]:
    return (
        [_ref("population", population_id, "population")],
        [_ref("falsification", "FALSIFY-1", "falsification")],
        [_ref("explicit_evidence", "EXPLICIT-SCOPE-1", "scope")],
    )


@pytest.mark.parametrize("status", [None, "partial", "failed", "complete"])
def test_falsification_reference_cannot_replace_execution_capability(status) -> None:
    implicit = _member("falsification-gap", "implicit_current")
    case = _case(implicit)
    population_id = "POPULATION-gap"
    population_refs, falsification_refs, scope_refs = _mandatory_refs(population_id)
    dependencies = [] if status is None else [CapabilityDependency(
        capability="current_falsification", requirement_mode="required",
        status=status, reason_codes=[] if status == "complete" else ["saved_execution_gap"],
    )]
    result = assemble_current_evidence(
        case,
        explicit_evidence=[],
        population_evidence=[_population(
            implicit.member_id, population_id=population_id,
            seed="support", label="support", path="src/core.py", line=1,
        )],
        denominators=[_denominator(implicit.member_id, population_id, support=1)],
        population_refs=population_refs,
        falsification_refs=falsification_refs,
        scope_condition_refs=scope_refs,
        dependencies=dependencies,
        limits=CurrentEvidenceLimits(max_items=10, max_bytes=100_000),
    )
    dependency = next(item for item in result.bundle.dependencies
                      if item.capability == "current_falsification")
    assert dependency.status == (status or "unavailable")
    assert result.bundle.status == ("complete" if status == "complete" else "partial")
    assert result.bundle.integrity == "valid"
    if status is None:
        assert "falsification_capability_state_missing" in dependency.reason_codes


def test_assembly_keeps_four_way_denominator_and_deduplicates_physical_origin() -> None:
    explicit = _member("explicit", "explicit")
    implicit = _member("implicit", "implicit_current")
    case = _case(explicit, implicit)
    population_id = "POPULATION-1"
    shared_digest = content_digest(["shared physical source"])
    shared_group = "same-physical-origin"
    explicit_item = _evidence(
        explicit.member_id,
        seed="explicit",
        role="support",
        origin_kind="explicit_span",
        path="src/plugins.py",
        line=10,
        source_digest=shared_digest,
        independence_group=shared_group,
        artifact_kind="explicit_evidence",
        artifact_id="EXPLICIT-EVIDENCE-1",
        anchor_literals=("PluginManager", "strict_plugins"),
        anchor_symbols=("PluginManager.register",),
    )
    population = [
        _population(
            implicit.member_id,
            population_id=population_id,
            seed="support",
            label="support",
            path="src/plugins.py",
            line=10,
            source_digest=shared_digest,
            independence_group=shared_group,
        ),
        _population(
            implicit.member_id,
            population_id=population_id,
            seed="counter",
            label="counterexample",
            path="tests/test_plugins.py",
            line=20,
        ),
        _population(
            implicit.member_id,
            population_id=population_id,
            seed="unknown",
            label="unknown",
            path="src/unknown.py",
            line=30,
        ),
        _population(
            implicit.member_id,
            population_id=population_id,
            seed="ineligible",
            label="ineligible",
            path="examples/demo.py",
            line=40,
        ),
    ]
    population_refs, falsification_refs, scope_refs = _mandatory_refs(population_id)

    result = assemble_current_evidence(
        case,
        explicit_evidence=[explicit_item],
        population_evidence=population,
        denominators=[
            _denominator(
                implicit.member_id,
                population_id,
                support=1,
                counterexample=1,
                unknown=1,
                ineligible=1,
            )
        ],
        population_refs=population_refs,
        falsification_refs=falsification_refs,
        scope_condition_refs=scope_refs,
        dependencies=[CapabilityDependency(
            capability="current_falsification", requirement_mode="required",
            status="complete", reason_codes=[],
        )],
        limits=CurrentEvidenceLimits(max_items=20, max_bytes=100_000),
    )

    bundle = result.bundle
    assert bundle.status == "complete"
    assert bundle.integrity == "valid"
    assert result.population_counts[implicit.member_id] == {
        "support": 1,
        "counterexample": 1,
        "unknown": 1,
        "ineligible": 1,
    }
    assert bundle.denominators[0].serialized == 4
    assert len(bundle.items) == 5
    assert len(bundle.origins) == 4
    shared_origins = [
        item
        for item in bundle.origins
        if item.path == "src/plugins.py" and item.line_start == 10
    ]
    assert len(shared_origins) == 1
    assert shared_origins[0].origin_kind == "explicit_span"
    assert bundle.population_refs == population_refs
    assert bundle.falsification_refs == falsification_refs
    assert bundle.scope_condition_refs == scope_refs
    explicit_bundle_item = next(
        item for item in bundle.items if item.artifact_ref.stable_id == "EXPLICIT-EVIDENCE-1"
    )
    assert explicit_bundle_item.anchor_literals == ["PluginManager", "strict_plugins"]
    assert explicit_bundle_item.anchor_symbols == ["PluginManager.register"]


def test_explicit_literal_anchor_limit_is_canonical_and_typed_partial() -> None:
    explicit = _member("explicit-anchor-limit", "explicit")
    case = _case(explicit)
    evidence = _evidence(
        explicit.member_id,
        seed="explicit-anchor-limit",
        role="support",
        origin_kind="configuration_atom",
        path="pyproject.toml",
        line=10,
        artifact_kind="explicit_evidence",
        anchor_literals=("gamma = 3", "alpha = 1", "beta = 2", "alpha = 1"),
    )

    result = assemble_current_evidence(
        case,
        explicit_evidence=[evidence],
        population_evidence=[],
        denominators=[],
        population_refs=[],
        falsification_refs=[],
        scope_condition_refs=[
            _ref("explicit_evidence", "EXPLICIT-SCOPE-LIMIT", "scope-limit")
        ],
        dependencies=[],
        limits=CurrentEvidenceLimits(max_items=2, max_bytes=100_000),
    )

    assert result.bundle.items[0].anchor_literals == ["alpha = 1", "beta = 2"]
    assert result.bundle.status == "partial"
    assert "current_evidence_anchor_limit" in result.bundle.incomplete_reasons


def test_origin_dedup_does_not_reduce_population_occurrence_counts() -> None:
    implicit = _member("implicit", "implicit_current")
    case = _case(implicit)
    population_id = "POPULATION-2"
    digest = content_digest(["one clone origin"])
    population = [
        _population(
            implicit.member_id,
            population_id=population_id,
            seed=f"support-{index}",
            label="support",
            path="src/clone.py",
            line=12,
            source_digest=digest,
            independence_group="clone-group-1",
        )
        for index in range(2)
    ]
    population_refs, falsification_refs, scope_refs = _mandatory_refs(population_id)

    result = assemble_current_evidence(
        case,
        explicit_evidence=[],
        population_evidence=population,
        denominators=[
            _denominator(implicit.member_id, population_id, support=2)
        ],
        population_refs=population_refs,
        falsification_refs=falsification_refs,
        scope_condition_refs=scope_refs,
        dependencies=[CapabilityDependency(
            capability="current_falsification", requirement_mode="required",
            status="complete", reason_codes=[],
        )],
        limits=CurrentEvidenceLimits(max_items=10, max_bytes=100_000),
    )

    assert result.bundle.status == "complete"
    assert result.bundle.denominators[0].support == 2
    assert result.population_counts[implicit.member_id]["support"] == 2
    assert len(result.bundle.origins) == 1
    assert len(result.bundle.items) == 2


def test_missing_denominator_and_falsification_are_typed_partial_gaps() -> None:
    implicit = _member("implicit", "implicit_current")
    case = _case(implicit)
    population_id = "POPULATION-3"
    population_refs, _, scope_refs = _mandatory_refs(population_id)
    support = _population(
        implicit.member_id,
        population_id=population_id,
        seed="support",
        label="support",
        path="src/plugin.py",
        line=5,
    )

    result = assemble_current_evidence(
        case,
        explicit_evidence=[],
        population_evidence=[support],
        denominators=[],
        population_refs=population_refs,
        falsification_refs=[],
        scope_condition_refs=scope_refs,
        dependencies=[],
        limits=CurrentEvidenceLimits(max_items=10, max_bytes=100_000),
    )

    bundle = result.bundle
    assert bundle.integrity == "valid"
    assert bundle.status == "partial"
    assert "population_denominator_missing" in bundle.incomplete_reasons
    assert "falsification_artifact_ref_missing" in bundle.incomplete_reasons
    dependency = {item.capability: item for item in bundle.dependencies}
    assert dependency["current_population"].status == "partial"
    assert dependency["current_falsification"].status == "unavailable"


def test_same_origin_independence_aliases_are_conservatively_normalized() -> None:
    explicit = _member("explicit", "explicit")
    case = _case(explicit)
    digest = content_digest(["same origin"])
    first = _evidence(
        explicit.member_id,
        seed="first",
        role="support",
        origin_kind="explicit_span",
        path="AGENTS.md",
        line=8,
        source_digest=digest,
        independence_group="group-a",
        artifact_kind="explicit_evidence",
    )
    second = _evidence(
        explicit.member_id,
        seed="second",
        role="support",
        origin_kind="configuration_atom",
        path="AGENTS.md",
        line=8,
        source_digest=digest,
        independence_group="group-b",
        artifact_kind="explicit_evidence",
    )
    linked_origin = _evidence(
        explicit.member_id,
        seed="linked",
        role="support",
        origin_kind="explicit_span",
        path="AGENTS.md",
        line=9,
        source_digest=digest,
        independence_group="group-b",
        artifact_kind="explicit_evidence",
    )

    result = assemble_current_evidence(
        case,
        explicit_evidence=[first, second, linked_origin],
        population_evidence=[],
        denominators=[],
        population_refs=[],
        falsification_refs=[],
        scope_condition_refs=[_ref("explicit_evidence", "SCOPE-1", "scope")],
        dependencies=[],
        limits=CurrentEvidenceLimits(max_items=10, max_bytes=100_000),
    )

    assert result.bundle.integrity == "valid"
    assert result.bundle.status == "complete"
    assert result.bundle.incomplete_reasons == []
    assert len(result.bundle.origins) == 2
    assert {
        origin.independence_group for origin in result.bundle.origins
    } == {"group-a"}


def test_item_limit_preserves_full_denominator_and_marks_bundle_partial() -> None:
    implicit = _member("implicit", "implicit_current")
    case = _case(implicit)
    population_id = "POPULATION-4"
    population = [
        _population(
            implicit.member_id,
            population_id=population_id,
            seed=f"support-{index}",
            label="support",
            path=f"src/plugin_{index}.py",
            line=5,
        )
        for index in range(2)
    ]
    population_refs, falsification_refs, scope_refs = _mandatory_refs(population_id)

    result = assemble_current_evidence(
        case,
        explicit_evidence=[],
        population_evidence=population,
        denominators=[
            _denominator(implicit.member_id, population_id, support=2)
        ],
        population_refs=population_refs,
        falsification_refs=falsification_refs,
        scope_condition_refs=scope_refs,
        dependencies=[CapabilityDependency(
            capability="current_falsification", requirement_mode="required",
            status="complete", reason_codes=[],
        )],
        limits=CurrentEvidenceLimits(max_items=1, max_bytes=100_000),
    )

    assert result.bundle.status == "partial"
    assert result.bundle.integrity == "valid"
    assert result.bundle.denominators[0].support == 2
    assert result.population_counts[implicit.member_id]["support"] == 2
    assert len(result.bundle.items) == 1
    assert "current_evidence_item_limit" in result.bundle.incomplete_reasons


def test_assembly_uses_only_supplied_objects_without_repository_or_git_io(
    monkeypatch,
) -> None:
    explicit = _member("explicit", "explicit")
    case = _case(explicit)
    evidence = _evidence(
        explicit.member_id,
        seed="explicit",
        role="support",
        origin_kind="explicit_span",
        path="AGENTS.md",
        line=1,
        artifact_kind="explicit_evidence",
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("repository read is forbidden")
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Git/provider subprocess is forbidden")
        ),
    )

    result = assemble_current_evidence(
        case,
        explicit_evidence=[evidence],
        population_evidence=[],
        denominators=[],
        population_refs=[],
        falsification_refs=[],
        scope_condition_refs=[_ref("explicit_evidence", "SCOPE-2", "scope")],
        dependencies=[],
        limits=CurrentEvidenceLimits(max_items=10, max_bytes=100_000),
    )

    assert result.bundle.status == "complete"
