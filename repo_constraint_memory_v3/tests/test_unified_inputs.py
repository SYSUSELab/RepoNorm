from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from helpers import (
    config_for_tests,
    copied_observation_fixture,
    discover_fixture,
)
from repo_constraint_memory_v3.observations import run_observations
from repo_constraint_memory_v3.explicit_verification import (
    explicit_candidate_root_id,
)
from repo_constraint_memory_v3.unified import inputs as unified_inputs
from repo_constraint_memory_v3.unified.grouping import MemberBlockingAnchors
from repo_constraint_memory_v3.unified.ids import (
    artifact_digest,
    content_digest,
    make_assessment_id,
    make_case_id,
    make_member_id,
)
from repo_constraint_memory_v3.unified.inputs import (
    _explicit_quote_literal_anchors,
    _explicit_semantic_anchor_symbols,
    _explicit_typed_decision_slots,
    build_blocking_anchors,
    build_observation_anchor_index,
    build_observation_anchor_supplement,
)
from repo_constraint_memory_v3.unified.models import (
    CandidateCase,
    CandidateCaseMember,
    CandidateMemberRelation,
    CandidateSemantics,
    ImmutableArtifactRef,
)

PROFILE = "b" * 64


@pytest.mark.parametrize("companion", ["complete", "not_required", "explicit"])
def test_case_falsification_gap_is_not_covered_by_another_member(companion: str) -> None:
    """Exercise case aggregation independently of evidence rendering."""
    from repo_constraint_memory_v3.semantic.models import BoundedStageExecution

    members = []
    assessments = {}
    populations = {}
    falsifications = {}
    for ordinal, status in enumerate(("partial", companion)):
        hypothesis_id = f"HYP-gap-member-{ordinal}"
        assessment_id = f"ASSESS-gap-member-{ordinal}"
        channel = "explicit" if status == "explicit" else "implicit_current"
        members.append(SimpleNamespace(assessment_id=assessment_id, member_id=f"MEMBER-{ordinal}"))
        assessments[assessment_id] = SimpleNamespace(
            assessment_id=assessment_id, channel=channel,
            candidate_root_id=f"ROOT-{ordinal}",
            support_refs=(), counterevidence_refs=(), coverage_refs=(),
            extension_refs=(SimpleNamespace(kind="norm_hypothesis", stable_id=hypothesis_id),),
        )
        if status == "not_required":
            populations[hypothesis_id] = SimpleNamespace(
                status="partial", classification_plan=SimpleNamespace(execution=None),
            )
        elif status != "explicit":
            complete = status == "complete"
            falsifications[hypothesis_id] = SimpleNamespace(
                status=status, admission_estimate=None,
                incomplete_reasons=[] if complete else ["saved_partial_falsification"],
                inference_ids=["INFER-complete"] if complete else [],
                execution=BoundedStageExecution(
                    request_plan_complete=complete,
                    planned_calls=int(complete), terminal_calls=int(complete), omitted_calls=0,
                    planned_items=1, terminal_items=int(complete), omitted_items=int(not complete),
                    closure_reason=None if complete else "saved_partial_falsification",
                ),
            )
    index = unified_inputs.CaseInputIndex(
        assessments=assessments, decisions={}, locations={}, hypotheses={},
        populations=populations, falsifications=falsifications, packets={},
    )
    inputs = unified_inputs.build_case_evidence_inputs(
        input_index=index, case=SimpleNamespace(members=members),
        assessments=(), explicit_decisions=(), current_source=None,
        observations=None, explicit_discovery_complete=True,
    )
    dependency = next(item for item in inputs.dependencies
                      if item.capability == "current_falsification")
    assert dependency.requirement_mode == "required"
    assert dependency.status == "partial"
    assert "saved_partial_falsification" in dependency.reason_codes


def test_explicit_quote_literals_are_canonical_bounded_original_lines() -> None:
    overlong = "x" * 513
    quote = (
        "  max-line-length = 99  \r\n"
        "\n"
        "select = E,F\n"
        "  max-line-length = 99  \r"
        f"{overlong}\n"
        "\tunsafe-control-line\n"
        "unsafe-delete\x7f\n"
        "   \n"
    )

    assert _explicit_quote_literal_anchors(quote) == (
        "  max-line-length = 99  ",
        "select = E,F",
    )
    assert _explicit_quote_literal_anchors("") == ()
    assert _explicit_quote_literal_anchors("single = true\n") == (
        "single = true",
    )


def test_explicit_symbol_anchors_are_typed_and_bounded() -> None:
    prose_key = "declaration:" + "repository-specific prose " * 30

    assert _explicit_semantic_anchor_symbols(
        "direct_declaration", prose_key
    ) == ()
    assert _explicit_semantic_anchor_symbols(
        "configuration_entry", "ini_cfg:pytest.addopts=-ra"
    ) == ("ini_cfg:pytest.addopts=-ra",)
    assert _explicit_semantic_anchor_symbols(
        "command_definition", "make:lint"
    ) == ("make:lint",)
    assert _explicit_semantic_anchor_symbols(
        "workflow_step", "workflow:.github/workflows/ci.yml:run:pytest"
    ) == ("workflow:.github/workflows/ci.yml:run:pytest",)
    assert _explicit_semantic_anchor_symbols(
        "configuration_entry", "x" * 257
    ) == ()


def test_explicit_typed_decision_slots_remove_values_only_when_provable() -> None:
    assert _explicit_typed_decision_slots(
        "configuration_entry",
        "toml:tool.mypy.check_untyped_defs=true",
        semantic_flags=(
            "config_section:tool.mypy",
            "config_key:check_untyped_defs",
        ),
    ) == ("configuration:toml:tool.mypy:check_untyped_defs",)
    assert _explicit_typed_decision_slots(
        "command_definition",
        "command:lint:ruff check --fix",
        semantic_flags=("command_target:lint", "command:ruff check --fix"),
    ) == ("command-target:lint",)
    assert _explicit_typed_decision_slots(
        "workflow_step",
        "workflow:.github/workflows/ci.yml:literal:run:pytest",
        semantic_flags=("workflow_key:run", "command:pytest"),
        path=".github/workflows/ci.yml",
        line_start=20,
        line_end=20,
    ) == ("workflow-action:.github/workflows/ci.yml:20:20",)
    assert _explicit_typed_decision_slots(
        "workflow_step",
        "workflow:.github/workflows/ci.yml:literal:uses:actions/setup-python@v6",
        semantic_flags=("workflow_key:uses", "command:actions/setup-python@v6"),
        path=".github/workflows/ci.yml",
        line_start=20,
        line_end=20,
    ) == ("workflow-action:.github/workflows/ci.yml:20:20",)

    # An explicit empty section is the typed root section and remains stable.
    assert _explicit_typed_decision_slots(
        "configuration_entry",
        "toml:<root>.line-length=88",
        semantic_flags=("config_section:", "config_key:line-length"),
    ) == ("configuration:toml:<root>:line-length",)
    # Workflow evidence without an exact source slot is ambiguous.  The
    # negative gate must abstain rather than infer one.
    assert _explicit_typed_decision_slots(
        "workflow_step",
        "workflow:.github/workflows/ci.yml:literal:run:pytest",
        semantic_flags=("workflow_key:run", "command:pytest"),
    ) == ()
    assert _explicit_typed_decision_slots(
        "command_definition",
        "command:lint:ruff check",
        semantic_flags=("command_target:lint", "command_target:test"),
    ) == ()
    assert _explicit_typed_decision_slots(
        "configuration_entry",
        "toml:tool.example.long=true",
        semantic_flags=(
            "config_section:tool.example",
            "config_key:" + "x" * 300,
        ),
    ) == ()


def test_explicit_grouping_uses_typed_keys_but_not_prose_semantic_keys() -> None:
    fingerprint = "1" * 64
    prose_key = "declaration:" + "repository-specific prose " * 30
    config_key = "ini_cfg:flake8.max-line-length=99"
    command_key = "command:lint:ruff check"
    workflow_action = "workflow:.github/workflows/ci.yml:literal:run:pytest"
    assessment = SimpleNamespace(
        assessment_id="ASSESS-explicit",
        candidate_root_id=explicit_candidate_root_id(fingerprint),
        channel="explicit",
        extension_refs=(),
    )
    member = SimpleNamespace(
        assessment_id=assessment.assessment_id,
        member_id="MEMBER-explicit",
    )
    decision = SimpleNamespace(
        candidate_fingerprint=fingerprint,
        raw_candidate=object(),
        grounded=(
            SimpleNamespace(
                semantic_flags=frozenset(),
                evidence=SimpleNamespace(
                    path="CONTRIBUTING.md",
                    quote="Keep each repository rule independently scoped.",
                    semantic_key=prose_key,
                    type="direct_declaration",
                )
            ),
            SimpleNamespace(
                semantic_flags=frozenset(
                    {"config_section:flake8", "config_key:max-line-length"}
                ),
                evidence=SimpleNamespace(
                    path="tox.ini",
                    quote="max-line-length=99",
                    semantic_key=config_key,
                    type="configuration_entry",
                )
            ),
            SimpleNamespace(
                semantic_flags=frozenset(
                    {"command_target:lint", "command:ruff check"}
                ),
                evidence=SimpleNamespace(
                    path="Makefile",
                    quote="lint:\n\truff check",
                    semantic_key=command_key,
                    type="command_definition",
                )
            ),
            SimpleNamespace(
                semantic_flags=frozenset(
                    {"workflow_key:run", "command:pytest"}
                ),
                evidence=SimpleNamespace(
                    path=".github/workflows/ci.yml",
                    quote="run: pytest",
                    semantic_key=workflow_action,
                    type="workflow_step",
                    line_start=20,
                    line_end=20,
                )
            ),
        ),
    )

    anchors = build_blocking_anchors(
        members=[member],
        assessments=[assessment],
        explicit_decisions=[decision],
        current_source=None,
        observations=None,
    )[member.member_id]

    assert anchors.symbols == tuple(
        sorted((command_key, config_key, workflow_action))
    )
    assert anchors.config_keys == (config_key,)
    assert anchors.commands == (command_key,)
    assert anchors.workflow_actions == (workflow_action,)
    assert anchors.decision_slots == tuple(
        sorted(
            (
                "command-target:lint",
                "configuration:ini_cfg:flake8:max-line-length",
                "workflow-action:.github/workflows/ci.yml:20:20",
            )
        )
    )
    assert prose_key not in anchors.symbols
    assert anchors.paths == (
        ".github/workflows/ci.yml",
        "CONTRIBUTING.md",
        "Makefile",
        "tox.ini",
    )


def test_large_population_only_uses_semantically_contributing_blocking_anchors(
    monkeypatch,
) -> None:
    hypothesis_id = "HYP-large"
    assessment = SimpleNamespace(
        assessment_id="ASSESS-current",
        candidate_root_id="ROOT-current",
        channel="implicit_current",
        extension_refs=(
            SimpleNamespace(kind="norm_hypothesis", stable_id=hypothesis_id),
        ),
    )
    member = SimpleNamespace(
        assessment_id=assessment.assessment_id,
        member_id="MEMBER-current",
    )
    population_members = [
        SimpleNamespace(
            label="support",
            observation_ref=SimpleNamespace(
                observation_id="OBS-support",
                path="src/pkg/core.py",
            ),
        )
    ]
    population_members.extend(
        SimpleNamespace(
            label=label,
            observation_ref=SimpleNamespace(
                observation_id=f"OBS-{label}-{ordinal}",
                path=f"irrelevant/{label}/{ordinal}.py",
            ),
        )
        # PopulationMember admits only support/counterexample/ineligible/unknown;
        # observation context is added later to case evidence and never enters
        # this blocking-anchor loop.
        for label in ("ineligible", "unknown")
        for ordinal in range(128)
    )
    locations = {
        item.observation_ref.observation_id: SimpleNamespace(
            path=item.observation_ref.path,
            symbols=(
                "PluginManager.is_registered"
                if item.label == "support"
                else f"shared-noise-{item.label}",
            ),
        )
        for item in population_members
    }
    monkeypatch.setattr(
        unified_inputs,
        "_observation_locations",
        lambda observations, current_source: locations,
    )
    current_source = SimpleNamespace(
        falsifications=(), member_packets=(),
        hypotheses=(SimpleNamespace(hypothesis_id=hypothesis_id),),
        populations=(
            SimpleNamespace(
                hypothesis_id=hypothesis_id,
                members=tuple(population_members),
            ),
        ),
    )

    anchors = build_blocking_anchors(
        members=[member],
        assessments=[assessment],
        explicit_decisions=[],
        current_source=current_source,
        observations=None,
    )[member.member_id]

    assert anchors.symbols == ("PluginManager.is_registered",)
    assert anchors.paths == ("src/pkg/core.py",)


def test_counterexample_population_member_can_supply_blocking_anchors(
    monkeypatch,
) -> None:
    hypothesis_id = "HYP-counterexample"
    observation_id = "OBS-counterexample"
    assessment = SimpleNamespace(
        assessment_id="ASSESS-counterexample",
        candidate_root_id="ROOT-counterexample",
        channel="implicit_current",
        extension_refs=(
            SimpleNamespace(kind="norm_hypothesis", stable_id=hypothesis_id),
        ),
    )
    member = SimpleNamespace(
        assessment_id=assessment.assessment_id,
        member_id="MEMBER-counterexample",
    )
    population_member = SimpleNamespace(
        label="counterexample",
        observation_ref=SimpleNamespace(
            observation_id=observation_id,
            path="src/pkg/exception.py",
        ),
    )
    monkeypatch.setattr(
        unified_inputs,
        "_observation_locations",
        lambda observations, current_source: {
            observation_id: SimpleNamespace(
                path="src/pkg/exception.py",
                symbols=("PluginManager.exception",),
            )
        },
    )
    current_source = SimpleNamespace(
        falsifications=(), member_packets=(),
        hypotheses=(SimpleNamespace(hypothesis_id=hypothesis_id),),
        populations=(
            SimpleNamespace(
                hypothesis_id=hypothesis_id,
                members=(population_member,),
            ),
        ),
    )

    anchors = build_blocking_anchors(
        members=[member],
        assessments=[assessment],
        explicit_decisions=[],
        current_source=current_source,
        observations=None,
    )[member.member_id]

    assert anchors.symbols == ("PluginManager.exception",)
    assert anchors.paths == ("src/pkg/exception.py",)


def _observations(tmp_path: Path):
    repository = copied_observation_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    return run_observations(
        repository,
        discovery,
        config.observation,
    ).bundle


def _member(snapshot: str, seed: str) -> CandidateCaseMember:
    assessment_id = make_assessment_id(
        snapshot,
        PROFILE,
        "explicit",
        f"ROOT-{seed}",
        0,
    )
    return CandidateCaseMember(
        member_id=make_member_id(snapshot, assessment_id),
        assessment_id=assessment_id,
        snapshot_digest=snapshot,
        channel="explicit",
        semantics=CandidateSemantics(
            rule="Use the repository's established Python structure",
            normalized_action="use-established-python-structure",
            polarity="positive",
            applies_to=["src/**"],
            condition=None,
            exception=None,
        ),
        assessment_ref=ImmutableArtifactRef(
            kind="explicit_candidate",
            stable_id=assessment_id,
            snapshot_digest=snapshot,
            artifact_path="explicit_verification.json",
            artifact_digest=content_digest([seed]),
        ),
    )


def _case(snapshot: str, *members: CandidateCaseMember) -> CandidateCase:
    ordered = sorted(members, key=lambda item: item.member_id)
    relations = [
        CandidateMemberRelation(
            left_member_id=ordered[left].member_id,
            right_member_id=ordered[right].member_id,
            relation="related_variant",
            source="semantic_grouping",
            reason_code="test_relation",
        )
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
    ]
    payload: dict[str, object] = {
        "case_id": make_case_id(
            snapshot,
            [item.member_id for item in ordered],
        ),
        "snapshot_digest": snapshot,
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


def test_observation_anchor_supplement_exactly_matches_saved_sites(
    tmp_path: Path,
) -> None:
    observations = _observations(tmp_path)
    member = _member(observations.index.snapshot_digest, "one")
    case = _case(observations.index.snapshot_digest, member)
    index = build_observation_anchor_index(observations)
    assert index is not None

    supplement = build_observation_anchor_supplement(
        case=case,
        anchors_by_member={
            member.member_id: MemberBlockingAnchors(
                symbols=("Child.method", "slow", "os.path.join"),
                paths=("src/pkg/core.py",),
            )
        },
        observation_index=index,
        max_items=20,
    )

    assert supplement.dependency.status == "complete"
    assert supplement.omitted_origins == 0
    assert supplement.evidence
    matched = {
        symbol
        for item in supplement.evidence
        for symbol in item.anchor_symbols
    }
    assert {"Child.method", "slow", "os.path.join"} <= matched
    assert all(item.origin_kind == "observation" for item in supplement.evidence)
    assert all(item.role == "context" for item in supplement.evidence)
    assert all(item.artifact_ref.kind == "observation" for item in supplement.evidence)
    assert all(
        item.artifact_ref.artifact_path.startswith("observations/python-files/")
        for item in supplement.evidence
    )

    wrong_case = build_observation_anchor_supplement(
        case=case,
        anchors_by_member={
            member.member_id: MemberBlockingAnchors(symbols=("child.method",))
        },
        observation_index=index,
        max_items=20,
    )
    assert wrong_case.evidence == ()
    assert wrong_case.dependency.status == "bounded_no_match"


def test_observation_anchor_supplement_is_bounded_and_preserves_partiality(
    tmp_path: Path,
) -> None:
    observations = _observations(tmp_path)
    member = _member(observations.index.snapshot_digest, "cap")
    case = _case(observations.index.snapshot_digest, member)
    complete_index = build_observation_anchor_index(observations)
    assert complete_index is not None
    partial_index = replace(
        complete_index,
        status="partial",
        incomplete_reasons=("python_file_limit",),
    )

    supplement = build_observation_anchor_supplement(
        case=case,
        anchors_by_member={
            member.member_id: MemberBlockingAnchors(
                symbols=("Child", "Child.method", "slow", "os.path.join"),
                paths=("src/pkg/core.py",),
            )
        },
        observation_index=partial_index,
        max_items=1,
    )

    assert len(supplement.evidence) == 1
    assert supplement.matched_origins > 1
    assert supplement.omitted_origins == supplement.matched_origins - 1
    assert supplement.dependency.status == "partial"
    assert supplement.dependency.reason_codes == [
        "observation_anchor_supplement_limit",
    ]


def test_observation_anchor_supplement_excludes_only_matched_partial_shard(
    tmp_path: Path,
) -> None:
    observations = _observations(tmp_path)
    member = _member(observations.index.snapshot_digest, "local-partial")
    case = _case(observations.index.snapshot_digest, member)
    complete_index = build_observation_anchor_index(observations)
    assert complete_index is not None
    path = "src/pkg/core.py"
    partial_match = replace(
        complete_index.by_path[path],
        shard_status="skipped",
        shard_incomplete_reasons=("syntax_error",),
    )
    partial_index = replace(
        complete_index,
        status="partial",
        incomplete_reasons=("python_snapshot_partial",),
        by_path={**complete_index.by_path, path: partial_match},
    )

    supplement = build_observation_anchor_supplement(
        case=case,
        anchors_by_member={
            member.member_id: MemberBlockingAnchors(paths=(path,))
        },
        observation_index=partial_index,
        max_items=20,
    )

    assert supplement.evidence == ()
    assert supplement.matched_origins == 1
    assert supplement.omitted_origins == 1
    assert supplement.dependency.status == "partial"
    assert supplement.dependency.reason_codes == [
        "observation_anchor_dependency_partial"
    ]

    symbol = "Child.method"
    symbol_match = replace(
        complete_index.by_symbol[symbol][0],
        shard_status="partial",
        shard_incomplete_reasons=("max_ast_nodes_per_file",),
    )
    symbol_index = replace(
        complete_index,
        status="partial",
        by_symbol={**complete_index.by_symbol, symbol: (symbol_match,)},
    )
    symbol_supplement = build_observation_anchor_supplement(
        case=case,
        anchors_by_member={
            member.member_id: MemberBlockingAnchors(symbols=(symbol,))
        },
        observation_index=symbol_index,
        max_items=20,
    )
    assert symbol_supplement.evidence == ()
    assert symbol_supplement.matched_origins == 1
    assert symbol_supplement.dependency.status == "partial"
    assert symbol_supplement.dependency.reason_codes == [
        "observation_anchor_dependency_partial"
    ]


def test_observation_anchor_supplement_deduplicates_one_origin_across_members(
    tmp_path: Path,
) -> None:
    observations = _observations(tmp_path)
    snapshot = observations.index.snapshot_digest
    left = _member(snapshot, "left")
    right = _member(snapshot, "right")
    case = _case(snapshot, left, right)
    index = build_observation_anchor_index(observations)
    assert index is not None

    supplement = build_observation_anchor_supplement(
        case=case,
        anchors_by_member={
            left.member_id: MemberBlockingAnchors(symbols=("Child.method",)),
            right.member_id: MemberBlockingAnchors(symbols=("Child.method",)),
        },
        observation_index=index,
        max_items=20,
    )

    assert supplement.matched_origins == len(supplement.evidence)
    assert supplement.evidence
    assert all(
        item.member_ids == tuple(sorted((left.member_id, right.member_id)))
        for item in supplement.evidence
    )
    physical_origins = {
        (
            item.path,
            item.line_start,
            item.line_end,
            item.source_content_digest,
        )
        for item in supplement.evidence
    }
    assert len(physical_origins) == len(supplement.evidence)
