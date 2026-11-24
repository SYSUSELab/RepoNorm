from __future__ import annotations

import pytest

from repo_constraint_memory_v3.unified.grouping import (
    GroupingLimits,
    GroupingValidationError,
    MemberBlockingAnchors,
    _group_request_input_bytes,
    assemble_candidate_cases,
    exact_candidate_identity,
    mark_grouping_components_incomplete,
    mark_grouping_pairs_incomplete,
    plan_candidate_grouping,
    plan_grouping_batches,
    resolve_direct_grouping,
    validate_semantic_relation_matrix,
)
from repo_constraint_memory_v3.unified.ids import (
    content_digest,
    make_assessment_id,
    make_member_id,
)
from repo_constraint_memory_v3.unified.inputs import infer_polarity
from repo_constraint_memory_v3.unified.models import (
    CandidateCaseMember,
    CandidateMemberRelation,
    CandidateSemantics,
    ImmutableArtifactRef,
)

SNAPSHOT = "sha256:" + "a" * 64
PROFILE = "b" * 64


def test_polarity_extractor_does_not_treat_without_clause_as_prohibition() -> None:
    assert infer_polarity("Run the test suite without network access.") == "positive"
    assert infer_polarity("Do not run tests with network access.") == "negative"
    assert infer_polarity("Network access is not allowed in tests.") == "negative"


def _semantics(
    rule: str,
    *,
    scope: tuple[str, ...] = ("src/**",),
    condition: str | None = None,
    polarity: str = "positive",
    action: str = "run-tests",
    exception: str | None = None,
) -> CandidateSemantics:
    return CandidateSemantics(
        rule=rule,
        normalized_action=action,
        polarity=polarity,
        applies_to=sorted(scope),
        condition=condition,
        exception=exception,
    )


def _member(
    seed: str,
    semantics: CandidateSemantics,
    *,
    channel: str = "explicit",
) -> CandidateCaseMember:
    assessment_id = make_assessment_id(
        SNAPSHOT, PROFILE, channel, f"ROOT-{seed}", 0
    )
    member_id = make_member_id(SNAPSHOT, assessment_id)
    kind = "explicit_candidate" if channel == "explicit" else "norm_hypothesis"
    return CandidateCaseMember(
        member_id=member_id,
        assessment_id=assessment_id,
        snapshot_digest=SNAPSHOT,
        channel=channel,
        semantics=semantics,
        assessment_ref=ImmutableArtifactRef(
            kind=kind,
            stable_id=assessment_id,
            snapshot_digest=SNAPSHOT,
            artifact_path=f"upstream/{seed}.json",
            artifact_digest=content_digest([seed]),
        ),
    )


def _limits(**overrides: int) -> GroupingLimits:
    values = {
        "max_candidate_members": 20,
        "max_block_members": 8,
        "max_group_pairs": 30,
        "max_group_pairs_per_request": 8,
        "max_group_members_per_request": 8,
        "max_group_request_bytes": 20_000,
        "max_case_members": 8,
        "max_case_input_bytes": 20_000,
        "max_anchors_per_member": 8,
        "max_paths_per_member": 8,
        "max_tokens_per_member": 16,
        "max_anchor_bytes": 200,
        "lexical_min_shared_tokens": 2,
    }
    values.update(overrides)
    return GroupingLimits(**values)


def _relation(
    left: CandidateCaseMember,
    right: CandidateCaseMember,
    relation: str,
    *,
    source: str = "semantic_grouping",
) -> CandidateMemberRelation:
    low, high = sorted((left.member_id, right.member_id))
    return CandidateMemberRelation(
        left_member_id=low,
        right_member_id=high,
        relation=relation,
        source=source,
        reason_code=f"test_{relation}",
    )


def _relations_for(
    plan: object,
    members: list[CandidateCaseMember],
    values: dict[frozenset[str], str],
) -> list[CandidateMemberRelation]:
    by_id = {item.member_id: item for item in members}
    return [
        _relation(by_id[left], by_id[right], values[frozenset((left, right))])
        for left, right in plan.semantic_pairs
    ]


def test_exact_identity_normalizes_all_canonical_semantic_fields() -> None:
    left = _semantics(
        "ＲＵＮ   PyTest\nBefore Merge",
        scope=("./tests//unit/", "src/**"),
        condition="  WHEN   touching tests ",
        exception="except generated files",
    )
    right = _semantics(
        "run pytest before merge",
        scope=("src/**", "tests/unit"),
        condition="when touching TESTS",
        exception="  EXCEPT   generated files ",
    )

    assert exact_candidate_identity(left) == exact_candidate_identity(right)

    changed_exception = right.model_copy(
        update={"exception": "except vendored files"}
    )
    assert exact_candidate_identity(left) != exact_candidate_identity(
        changed_exception
    )


def test_exact_duplicates_keep_every_member_and_all_provenance() -> None:
    semantics = _semantics("Run pytest before merge")
    left = _member("left", semantics)
    right = _member("right", semantics, channel="implicit_current")
    plan = plan_candidate_grouping(
        [right, left], anchors_by_member={}, limits=_limits()
    )

    assert plan.semantic_pairs == ()
    assert plan.exact_identity_groups == (tuple(sorted((left.member_id, right.member_id))),)
    result = assemble_candidate_cases(plan, [], profile_digest=PROFILE)

    assert len(result.cases) == 1
    case = result.cases[0]
    assert [item.member_id for item in case.members] == sorted(
        (left.member_id, right.member_id)
    )
    assert {item.channel for item in case.members} == {
        "explicit",
        "implicit_current",
    }
    assert len(case.relations) == 1
    assert case.relations[0].source == "exact_identity"
    assert case.relations[0].relation == "equivalent"


def test_singleton_mode_bypasses_all_grouping_edges_after_entry_validation() -> None:
    duplicate_semantics = _semantics("Register shared plugins before tests")
    duplicate_left = _member("singleton-duplicate-left", duplicate_semantics)
    duplicate_right = _member(
        "singleton-duplicate-right",
        duplicate_semantics,
        channel="implicit_current",
    )
    lexical_peer = _member(
        "singleton-lexical-peer",
        _semantics(
            "Validate shared plugins before tests",
            action="validate-plugins",
        ),
    )
    members = [lexical_peer, duplicate_right, duplicate_left]
    anchors = {
        item.member_id: MemberBlockingAnchors(symbols=("SharedRegistry",))
        for item in members
    }

    plan = plan_candidate_grouping(
        members,
        anchors_by_member=anchors,
        limits=_limits(),
        mode="singleton",
    )

    ordered_ids = tuple(sorted(item.member_id for item in members))
    assert plan.blocks == ()
    assert plan.exact_identity_groups == ()
    assert plan.semantic_pairs == ()
    assert plan.incomplete_components == ()
    assert plan._raw_components == tuple((member_id,) for member_id in ordered_ids)
    assert plan_grouping_batches(plan, (), phase="direct").batches == ()
    resolved, closure_pairs = resolve_direct_grouping(plan, [])
    assert closure_pairs == ()

    result = assemble_candidate_cases(resolved, [], profile_digest=PROFILE)
    assert len(result.cases) == len(members)
    assert all(len(case.members) == 1 for case in result.cases)
    assert all(case.grouping_status == "complete" for case in result.cases)
    assert all(case.grouping_reason_codes == [] for case in result.cases)
    assert all(case.relations == [] for case in result.cases)


def test_singleton_mode_still_rejects_foreign_blocking_anchor_members() -> None:
    member = _member("singleton-entry-validation", _semantics("Run tests"))

    with pytest.raises(
        ValueError, match="blocking anchors reference unknown members"
    ):
        plan_candidate_grouping(
            [member],
            anchors_by_member={
                "MEMBER-" + "f" * 24: MemberBlockingAnchors(
                    symbols=("Foreign",)
                )
            },
            limits=_limits(),
            mode="singleton",
        )


def test_blocking_is_deterministic_and_uses_strong_anchors_and_tokens() -> None:
    first = _member("one", _semantics("Register frobnicator plugin safely"))
    second = _member("two", _semantics("Validate frobnicator plugin eagerly"))
    third = _member(
        "three", _semantics("Document release procedure", scope=("docs/release.md",))
    )
    anchors = {
        first.member_id: MemberBlockingAnchors(symbols=("PluginManager",)),
        second.member_id: MemberBlockingAnchors(symbols=("PluginManager",)),
        third.member_id: MemberBlockingAnchors(paths=("docs/release.md",)),
    }

    forward = plan_candidate_grouping(
        [first, second, third], anchors_by_member=anchors, limits=_limits()
    )
    reverse = plan_candidate_grouping(
        [third, second, first], anchors_by_member=dict(reversed(list(anchors.items()))), limits=_limits()
    )

    assert forward.blocks == reverse.blocks
    assert forward.semantic_pairs == reverse.semantic_pairs
    assert forward._raw_components == reverse._raw_components
    assert tuple(sorted((first.member_id, second.member_id))) in forward.semantic_pairs
    assert (third.member_id,) in forward._raw_components


@pytest.mark.parametrize(
    "field,left_value,right_value,left_slot,right_slot",
    [
        (
            "config_keys",
            "toml:tool.mypy.check_untyped_defs=true",
            "toml:tool.mypy.warn_unused_ignores=true",
            "configuration:toml:tool.mypy:check_untyped_defs",
            "configuration:toml:tool.mypy:warn_unused_ignores",
        ),
        (
            "commands",
            "command:lint:ruff check",
            "command:test:pytest",
            "command-target:lint",
            "command-target:test",
        ),
        (
            "workflow_actions",
            "workflow:.github/workflows/ci.yml:literal:run:pytest",
            "workflow:.github/workflows/ci.yml:literal:run:ruff check",
            "workflow-action:.github/workflows/ci.yml:20:20",
            "workflow-action:.github/workflows/ci.yml:30:30",
        ),
    ],
)
def test_lexical_overlap_does_not_pair_disjoint_explicit_typed_decisions(
    field: str,
    left_value: str,
    right_value: str,
    left_slot: str,
    right_slot: str,
) -> None:
    left = _member(
        "typed-left",
        _semantics("Configure the shared repository quality tool for alpha"),
    )
    right = _member(
        "typed-right",
        _semantics("Configure the shared repository quality tool for beta"),
    )
    anchors = {
        left.member_id: MemberBlockingAnchors(
            **{field: (left_value,)}, decision_slots=(left_slot,)
        ),
        right.member_id: MemberBlockingAnchors(
            **{field: (right_value,)}, decision_slots=(right_slot,)
        ),
    }

    plan = plan_candidate_grouping(
        [left, right], anchors_by_member=anchors, limits=_limits()
    )

    assert plan.semantic_pairs == ()
    assert plan._raw_components == tuple(
        (member_id,) for member_id in sorted((left.member_id, right.member_id))
    )
    assert plan.incomplete_components == ()


def test_shared_typed_decision_remains_a_strong_pair() -> None:
    left = _member(
        "shared-left",
        _semantics("Configure the shared repository quality tool for alpha"),
    )
    right = _member(
        "shared-right",
        _semantics("Configure the shared repository quality tool for beta"),
    )
    shared_key = "toml:tool.mypy.check_untyped_defs=true"

    plan = plan_candidate_grouping(
        [left, right],
        anchors_by_member={
            left.member_id: MemberBlockingAnchors(config_keys=(shared_key,)),
            right.member_id: MemberBlockingAnchors(config_keys=(shared_key,)),
        },
        limits=_limits(),
    )

    assert plan.semantic_pairs == (
        tuple(sorted((left.member_id, right.member_id))),
    )


@pytest.mark.parametrize(
    "field,left_value,right_value,shared_slot",
    [
        (
            "config_keys",
            "toml:tool.mypy.check_untyped_defs=true",
            "toml:tool.mypy.check_untyped_defs=false",
            "configuration:toml:tool.mypy:check_untyped_defs",
        ),
        (
            "commands",
            "command:lint:ruff check",
            "command:lint:ruff check --fix",
            "command-target:lint",
        ),
        (
            "workflow_actions",
            "workflow:.github/workflows/ci.yml:literal:run:pytest",
            "workflow:.github/workflows/ci.yml:literal:uses:actions/setup-python@v6",
            "workflow-action:.github/workflows/ci.yml:20:20",
        ),
    ],
)
def test_same_typed_slot_with_different_value_remains_a_candidate_pair(
    field: str,
    left_value: str,
    right_value: str,
    shared_slot: str,
) -> None:
    left = _member(
        "variant-left",
        _semantics("Configure the shared repository quality tool for alpha"),
    )
    right = _member(
        "variant-right",
        _semantics("Configure the shared repository quality tool for beta"),
    )

    plan = plan_candidate_grouping(
        [left, right],
        anchors_by_member={
            left.member_id: MemberBlockingAnchors(
                **{field: (left_value,)}, decision_slots=(shared_slot,)
            ),
            right.member_id: MemberBlockingAnchors(
                **{field: (right_value,)}, decision_slots=(shared_slot,)
            ),
        },
        limits=_limits(),
    )

    assert plan.semantic_pairs == (
        tuple(sorted((left.member_id, right.member_id))),
    )


@pytest.mark.parametrize(
    "left_slot,right_slot",
    [
        (
            "configuration:toml:tool.pytest.ini_options:addopts",
            "command-target:test",
        ),
        (
            "configuration:toml:tool.mypy:python_version",
            "configuration:ini_cfg:mypy:python_version",
        ),
        ("unparsed-left", "unparsed-right"),
    ],
)
def test_incomparable_or_unparsed_typed_slots_do_not_suppress_lexical_pairs(
    left_slot: str,
    right_slot: str,
) -> None:
    left = _member(
        "incomparable-left",
        _semantics("Configure the shared repository test behavior for alpha"),
    )
    right = _member(
        "incomparable-right",
        _semantics("Configure the shared repository test behavior for beta"),
    )

    plan = plan_candidate_grouping(
        [left, right],
        anchors_by_member={
            left.member_id: MemberBlockingAnchors(decision_slots=(left_slot,)),
            right.member_id: MemberBlockingAnchors(decision_slots=(right_slot,)),
        },
        limits=_limits(),
    )

    assert plan.semantic_pairs == (
        tuple(sorted((left.member_id, right.member_id))),
    )


def test_truncated_decision_slots_cannot_prove_lexical_pair_is_distinct() -> None:
    left = _member(
        "truncated-left",
        _semantics("Configure the shared repository quality tool for alpha"),
    )
    right = _member(
        "truncated-right",
        _semantics("Configure the shared repository quality tool for beta"),
    )

    plan = plan_candidate_grouping(
        [left, right],
        anchors_by_member={
            left.member_id: MemberBlockingAnchors(
                decision_slots=(
                    "configuration:toml:tool.example:alpha",
                    "configuration:toml:tool.example:shared",
                )
            ),
            right.member_id: MemberBlockingAnchors(
                decision_slots=(
                    "configuration:toml:tool.example:beta",
                    "configuration:toml:tool.example:shared",
                )
            ),
        },
        limits=_limits(max_anchors_per_member=1),
    )

    # The resource gap still fail-closes the component, but the lexical edge
    # joins both endpoints: the bounded slot prefix was not misused as proof
    # that they are distinct.
    assert plan.semantic_pairs == ()
    assert plan._raw_components == (
        tuple(sorted((left.member_id, right.member_id))),
    )
    assert plan.incomplete_components


def test_disjoint_typed_gate_does_not_reduce_cross_channel_recall() -> None:
    explicit = _member(
        "cross-explicit",
        _semantics("Configure the shared repository quality tool for alpha"),
    )
    current = _member(
        "cross-current",
        _semantics("Configure the shared repository quality tool for beta"),
        channel="implicit_current",
    )

    plan = plan_candidate_grouping(
        [explicit, current],
        anchors_by_member={
            explicit.member_id: MemberBlockingAnchors(
                config_keys=("toml:tool.mypy.alpha=true",),
                decision_slots=("configuration:toml:tool.mypy:alpha",),
            ),
            current.member_id: MemberBlockingAnchors(
                config_keys=("toml:tool.mypy.beta=true",),
                decision_slots=("configuration:toml:tool.mypy:beta",),
            ),
        },
        limits=_limits(),
    )

    assert plan.semantic_pairs == (
        tuple(sorted((explicit.member_id, current.member_id))),
    )


def test_disjoint_explicit_config_population_does_not_consume_pair_budget() -> None:
    members = [
        _member(
            f"config-{index}",
            _semantics(
                f"Configure the shared repository quality tool option {index}"
            ),
        )
        for index in range(64)
    ]
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            member.member_id: MemberBlockingAnchors(
                config_keys=(f"toml:tool.mypy.option-{index}=true",),
                decision_slots=(
                    f"configuration:toml:tool.mypy:option-{index}",
                ),
            )
            for index, member in enumerate(members)
        },
        limits=_limits(max_candidate_members=100, max_group_pairs=30),
    )

    assert plan.semantic_pairs == ()
    assert len(plan._raw_components) == len(members)
    assert all(len(component) == 1 for component in plan._raw_components)
    assert plan.incomplete_components == ()


def test_shared_evidence_file_does_not_collapse_many_unrelated_rules() -> None:
    members = [
        _member(str(index), _semantics(f"Concern{index} policy"))
        for index in range(58)
    ]
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            item.member_id: MemberBlockingAnchors(paths=("AGENTS.md",))
            for item in members
        },
        limits=_limits(
            max_candidate_members=64,
            max_block_members=32,
            max_case_members=32,
            max_group_pairs=2_000,
            lexical_min_shared_tokens=2,
        ),
    )

    assert len(plan._raw_components) == 58
    assert all(len(component) == 1 for component in plan._raw_components)
    assert plan.semantic_pairs == ()
    assert plan.incomplete_components == ()
    assert all(block.kind != "path" for block in plan.blocks)


def test_path_star_does_not_transitively_merge_but_lexical_pair_is_retained() -> None:
    center = _member(
        "center", _semantics("Validate plugin registration safely")
    )
    related = _member(
        "related", _semantics("Document plugin registration sequence")
    )
    guide_leaf = _member("guide", _semantics("Configure isolated environments"))
    config_leaf = _member("config", _semantics("Publish signed artifacts"))
    members = [center, related, guide_leaf, config_leaf]
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            center.member_id: MemberBlockingAnchors(
                paths=("AGENTS.md", "pyproject.toml")
            ),
            related.member_id: MemberBlockingAnchors(paths=("AGENTS.md",)),
            guide_leaf.member_id: MemberBlockingAnchors(paths=("AGENTS.md",)),
            config_leaf.member_id: MemberBlockingAnchors(
                paths=("pyproject.toml",)
            ),
        },
        limits=_limits(),
    )

    related_pair = tuple(sorted((center.member_id, related.member_id)))
    assert plan.semantic_pairs == (related_pair,)
    assert tuple(sorted(related_pair)) in plan._raw_components
    assert (guide_leaf.member_id,) in plan._raw_components
    assert (config_leaf.member_id,) in plan._raw_components


def test_direct_anchor_star_is_not_rejected_by_transitive_component_size() -> None:
    center = _member("center-star", _semantics("Central coordination rule"))
    leaves = [
        _member(f"leaf-{index}", _semantics(f"Leaf{index} topic{index}"))
        for index in range(3)
    ]
    plan = plan_candidate_grouping(
        [center, *leaves],
        anchors_by_member={
            center.member_id: MemberBlockingAnchors(
                symbols=tuple(f"edge-{index}" for index in range(3))
            ),
            **{
                leaf.member_id: MemberBlockingAnchors(symbols=(f"edge-{index}",))
                for index, leaf in enumerate(leaves)
            },
        },
        limits=_limits(
            max_block_members=2,
            max_case_members=2,
            max_group_request_bytes=100_000,
            max_case_input_bytes=100_000,
        ),
    )

    assert len(plan._raw_components) == 1
    assert len(plan._raw_components[0]) == 4
    assert len(plan.semantic_pairs) == 3
    assert plan.incomplete_components == ()
    by_id = {item.member_id: item for item in [center, *leaves]}
    result = assemble_candidate_cases(
        plan,
        [
            _relation(by_id[left], by_id[right], "distinct")
            for left, right in plan.semantic_pairs
        ],
        profile_digest=PROFILE,
    )

    assert len(result.cases) == 4
    assert all(case.grouping_status == "complete" for case in result.cases)


def test_high_frequency_typed_anchor_is_suppressed_as_incomplete_not_distinct() -> None:
    members = [
        _member(f"frequent-{index}", _semantics(f"Concern{index} topic{index}"))
        for index in range(4)
    ]
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            item.member_id: MemberBlockingAnchors(symbols=("generic.anchor",))
            for item in members
        },
        limits=_limits(max_block_members=3, max_case_members=3),
    )

    assert plan.semantic_pairs == ()
    assert len(plan._raw_components) == 4
    assert len(plan.incomplete_components) == 4
    assert all(
        "grouping_block_member_limit" in component.reason_codes
        for component in plan.incomplete_components
    )


def test_small_non_distinct_chain_gets_bounded_second_phase_closure() -> None:
    first = _member("chain-a", _semantics("Alpha mode"))
    middle = _member("chain-b", _semantics("Beta path"))
    last = _member("chain-c", _semantics("Gamma option"))
    members = [first, middle, last]
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            first.member_id: MemberBlockingAnchors(symbols=("edge-ab",)),
            middle.member_id: MemberBlockingAnchors(
                symbols=("edge-ab", "edge-bc")
            ),
            last.member_id: MemberBlockingAnchors(symbols=("edge-bc",)),
        },
        limits=_limits(),
    )
    by_id = {item.member_id: item for item in members}
    direct_relations = [
        _relation(by_id[left], by_id[right], "equivalent")
        for left, right in plan.semantic_pairs
    ]
    plan, closure_pairs = resolve_direct_grouping(plan, direct_relations)
    closure_relations = [
        _relation(by_id[left], by_id[right], "equivalent")
        for left, right in closure_pairs
    ]
    result = assemble_candidate_cases(
        plan,
        [*direct_relations, *closure_relations],
        profile_digest=PROFILE,
    )

    assert len(plan.semantic_pairs) == 3
    assert len(closure_pairs) == 1
    assert len(result.cases) == 1
    case = result.cases[0]
    assert len(case.members) == 3
    assert len(case.relations) == 3
    assert case.grouping_status == "complete"
    assert "grouping_relation_closure_missing" not in case.grouping_reason_codes


def test_small_four_member_direct_graph_gets_all_six_closure_cells() -> None:
    center = _member("closure-center", _semantics("Central closure rule"))
    leaves = [
        _member(f"closure-{index}", _semantics(f"Closure{index} topic{index}"))
        for index in range(3)
    ]
    plan = plan_candidate_grouping(
        [center, *leaves],
        anchors_by_member={
            center.member_id: MemberBlockingAnchors(
                symbols=tuple(f"closure-edge-{index}" for index in range(3))
            ),
            **{
                leaf.member_id: MemberBlockingAnchors(
                    symbols=(f"closure-edge-{index}",)
                )
                for index, leaf in enumerate(leaves)
            },
        },
        limits=_limits(),
    )

    assert len(plan._raw_components) == 1
    assert len(plan.semantic_pairs) == 3
    by_id = {item.member_id: item for item in [center, *leaves]}
    direct_relations = [
        _relation(by_id[left], by_id[right], "equivalent")
        for left, right in plan.semantic_pairs
    ]
    plan, closure_pairs = resolve_direct_grouping(plan, direct_relations)
    assert len(plan.semantic_pairs) == 6
    assert len(closure_pairs) == 3


def test_small_chain_omits_only_oversized_closure_batch() -> None:
    first = _member("byte-a", _semantics("Alpha " + "a" * 450))
    middle = _member("byte-b", _semantics("Beta byte path"))
    last = _member("byte-c", _semantics("Gamma " + "c" * 450))
    members = [first, middle, last]
    component = tuple(sorted(item.member_id for item in members))
    direct_pairs = tuple(
        sorted(
            (
                tuple(sorted((first.member_id, middle.member_id))),
                tuple(sorted((middle.member_id, last.member_id))),
            )
        )
    )
    members_by_id = {item.member_id: item for item in members}
    direct_request_bytes = max(
        _group_request_input_bytes(
            pair,
            [pair],
            members_by_id,
            phase="direct",
        )
        for pair in direct_pairs
    )
    closure_pair = tuple(
        sorted((first.member_id, last.member_id))
    )
    closure_request_bytes = _group_request_input_bytes(
        closure_pair,
        [closure_pair],
        members_by_id,
        phase="closure",
    )
    assert direct_request_bytes < closure_request_bytes
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            first.member_id: MemberBlockingAnchors(symbols=("byte-ab",)),
            middle.member_id: MemberBlockingAnchors(
                symbols=("byte-ab", "byte-bc")
            ),
            last.member_id: MemberBlockingAnchors(symbols=("byte-bc",)),
        },
        limits=_limits(max_group_request_bytes=direct_request_bytes),
    )
    direct_relations = [
        _relation(members_by_id[left], members_by_id[right], "equivalent")
        for left, right in plan.semantic_pairs
    ]
    plan, closure_pairs = resolve_direct_grouping(plan, direct_relations)
    batches = plan_grouping_batches(plan, closure_pairs, phase="closure")

    assert plan.semantic_pairs == tuple(sorted((*direct_pairs, closure_pair)))
    assert batches.batches == ()
    assert batches.omitted_pairs == (closure_pair,)


def test_small_chain_skips_closure_when_global_pair_budget_cannot_fit() -> None:
    first = _member("budget-a", _semantics("Alpha mode"))
    middle = _member("budget-b", _semantics("Beta path"))
    last = _member("budget-c", _semantics("Gamma option"))
    plan = plan_candidate_grouping(
        [first, middle, last],
        anchors_by_member={
            first.member_id: MemberBlockingAnchors(symbols=("budget-ab",)),
            middle.member_id: MemberBlockingAnchors(
                symbols=("budget-ab", "budget-bc")
            ),
            last.member_id: MemberBlockingAnchors(symbols=("budget-bc",)),
        },
        limits=_limits(max_group_pairs=2, max_group_pairs_per_request=2),
    )
    by_id = {item.member_id: item for item in [first, middle, last]}
    direct_relations = [
        _relation(by_id[left], by_id[right], "equivalent")
        for left, right in plan.semantic_pairs
    ]
    plan, closure_pairs = resolve_direct_grouping(plan, direct_relations)

    assert len(plan.semantic_pairs) == 2
    assert closure_pairs == ()
    assert plan.incomplete_components[0].reason_codes == (
        "grouping_incomplete",
        "grouping_pair_budget",
    )


def test_failed_direct_pair_still_consumes_global_closure_budget() -> None:
    first = _member("failed-budget-a", _semantics("Alpha mode"))
    middle = _member("failed-budget-b", _semantics("Beta path"))
    last = _member("failed-budget-c", _semantics("Gamma option"))
    failed_left = _member("failed-budget-d", _semantics("Delta command"))
    failed_right = _member("failed-budget-e", _semantics("Epsilon command"))
    members = [first, middle, last, failed_left, failed_right]
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            first.member_id: MemberBlockingAnchors(symbols=("budget-ab",)),
            middle.member_id: MemberBlockingAnchors(
                symbols=("budget-ab", "budget-bc")
            ),
            last.member_id: MemberBlockingAnchors(symbols=("budget-bc",)),
            failed_left.member_id: MemberBlockingAnchors(symbols=("budget-de",)),
            failed_right.member_id: MemberBlockingAnchors(symbols=("budget-de",)),
        },
        limits=_limits(max_group_pairs=3, max_group_pairs_per_request=3),
    )
    by_id = {item.member_id: item for item in members}
    failed_pair = next(
        pair
        for pair in plan.semantic_pairs
        if set(pair) == {failed_left.member_id, failed_right.member_id}
    )
    direct_relations = [
        _relation(by_id[left], by_id[right], "equivalent")
        for left, right in plan.semantic_pairs
        if (left, right) != failed_pair
    ]

    resolved, closure_pairs = resolve_direct_grouping(
        plan,
        direct_relations,
        failed_pairs=[failed_pair],
    )

    assert closure_pairs == ()
    reasons = {
        item.member_ids: item.reason_codes
        for item in resolved.incomplete_components
    }
    chain = tuple(sorted((first.member_id, middle.member_id, last.member_id)))
    failed_component = tuple(sorted(failed_pair))
    assert reasons[chain] == ("grouping_incomplete", "grouping_pair_budget")
    assert reasons[failed_component] == (
        "candidate_semantic_group_failed",
        "grouping_incomplete",
    )


def test_failed_sparse_component_filter_does_not_materialize_full_closure() -> None:
    center = _member("large-center", _semantics("Hub coordinator"))
    leaves = [
        _member(f"large-leaf-{index}", _semantics(f"Leaf{index} topic{index}"))
        for index in range(512)
    ]
    plan = plan_candidate_grouping(
        [center, *leaves],
        anchors_by_member={
            center.member_id: MemberBlockingAnchors(
                symbols=tuple(f"large-edge-{index}" for index in range(512))
            ),
            **{
                leaf.member_id: MemberBlockingAnchors(
                    symbols=(f"large-edge-{index}",)
                )
                for index, leaf in enumerate(leaves)
            },
        },
        limits=_limits(
            max_candidate_members=600,
            max_block_members=600,
            max_case_members=600,
            max_group_pairs=1_000,
            max_group_request_bytes=5_000_000,
            max_case_input_bytes=5_000_000,
            max_anchors_per_member=600,
        ),
    )

    assert len(plan.semantic_pairs) == 512
    failed = mark_grouping_components_incomplete(
        plan,
        [plan._raw_components[0]],
        reason_code="candidate_semantic_group_failed",
    )
    assert failed.semantic_pairs == ()
    assert len(failed.incomplete_components) == 1


def test_oversized_non_distinct_star_keeps_only_bounded_direct_relations() -> None:
    center = _member("oversized-center", _semantics("Central hub"))
    leaves = [
        _member(f"oversized-{index}", _semantics(f"Node{index} topic{index}"))
        for index in range(128)
    ]
    members = [center, *leaves]
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            center.member_id: MemberBlockingAnchors(
                symbols=tuple(f"oversized-edge-{index}" for index in range(128))
            ),
            **{
                leaf.member_id: MemberBlockingAnchors(
                    symbols=(f"oversized-edge-{index}",)
                )
                for index, leaf in enumerate(leaves)
            },
        },
        limits=_limits(
            max_candidate_members=150,
            max_block_members=150,
            max_case_members=32,
            max_group_pairs=500,
            max_group_request_bytes=1_000_000,
            max_case_input_bytes=1_000_000,
            max_anchors_per_member=150,
        ),
    )
    by_id = {item.member_id: item for item in members}
    result = assemble_candidate_cases(
        plan,
        [
            _relation(by_id[left], by_id[right], "related_variant")
            for left, right in plan.semantic_pairs
        ],
        profile_digest=PROFILE,
    )

    assert len(result.cases) == 1
    case = result.cases[0]
    assert case.grouping_status == "incomplete"
    assert len(case.members) == 129
    assert len(case.relations) == 128
    assert "grouping_case_member_limit" in case.grouping_reason_codes


def test_sparse_133_edge_star_is_split_into_bounded_direct_batches() -> None:
    center = _member("batch-center", _semantics("Qzxhub"))
    leaves = [
        _member(f"batch-leaf-{index}", _semantics(f"Qzxleaf{index}"))
        for index in range(133)
    ]
    members = [center, *leaves]
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            center.member_id: MemberBlockingAnchors(
                symbols=tuple(f"batch-edge-{index}" for index in range(133))
            ),
            **{
                leaf.member_id: MemberBlockingAnchors(
                    symbols=(f"batch-edge-{index}",)
                )
                for index, leaf in enumerate(leaves)
            },
        },
        limits=_limits(
            max_candidate_members=200,
            max_block_members=200,
            max_case_members=32,
            max_group_pairs=1_000,
            max_group_pairs_per_request=16,
            max_group_members_per_request=17,
            max_group_request_bytes=100_000,
            max_case_input_bytes=1_000_000,
            max_anchors_per_member=200,
        ),
    )
    batches = plan_grouping_batches(plan, plan.semantic_pairs, phase="direct")

    assert len(plan.semantic_pairs) == 133
    assert len(batches.batches) == 9
    assert batches.omitted_pairs == ()
    assert all(len(batch.pairs) <= 16 for batch in batches.batches)
    assert all(len(batch.member_ids) <= 17 for batch in batches.batches)
    assert {
        pair for batch in batches.batches for pair in batch.pairs
    } == set(plan.semantic_pairs)


def test_dense_pair_graph_is_packed_without_member_payload_fragmentation() -> None:
    members = [
        _member(f"dense-{index:02d}", _semantics(f"Dense rule {index}"))
        for index in range(64)
    ]
    anchors = {
        member.member_id: MemberBlockingAnchors(symbols=("shared-dense-anchor",))
        for member in members
    }
    limits = _limits(
        max_candidate_members=64,
        max_block_members=64,
        max_case_members=64,
        max_group_pairs=3_000,
        max_group_pairs_per_request=64,
        max_group_members_per_request=32,
        max_group_request_bytes=1_000_000,
    )
    forward = plan_candidate_grouping(
        members,
        anchors_by_member=anchors,
        limits=limits,
    )
    reverse = plan_candidate_grouping(
        list(reversed(members)),
        anchors_by_member=anchors,
        limits=limits,
    )

    forward_batches = plan_grouping_batches(
        forward, forward.semantic_pairs, phase="direct"
    )
    reverse_batches = plan_grouping_batches(
        reverse, reverse.semantic_pairs, phase="direct"
    )

    assert len(forward.semantic_pairs) == 2_016
    assert forward_batches == reverse_batches
    assert forward_batches.omitted_pairs == ()
    assert len(forward_batches.batches) <= 40
    assert sum(len(batch.pairs) for batch in forward_batches.batches) == 2_016
    assert all(len(batch.pairs) <= 64 for batch in forward_batches.batches)
    assert all(len(batch.member_ids) <= 32 for batch in forward_batches.batches)
    assert {
        pair for batch in forward_batches.batches for pair in batch.pairs
    } == set(forward.semantic_pairs)


def test_high_document_frequency_lexical_boilerplate_is_not_a_pair_key() -> None:
    members = [
        _member(
            f"high-df-{index:02d}",
            _semantics(f"Must modify unique{index} behavior"),
        )
        for index in range(40)
    ]
    anchored_pair = tuple(sorted((members[0].member_id, members[1].member_id)))
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            members[0].member_id: MemberBlockingAnchors(
                symbols=("shared-typed-symbol",)
            ),
            members[1].member_id: MemberBlockingAnchors(
                symbols=("shared-typed-symbol",)
            ),
        },
        limits=_limits(
            max_candidate_members=40,
            max_block_members=8,
            max_case_members=8,
            max_group_pairs=1_000,
        ),
    )

    assert plan.semantic_pairs == (anchored_pair,)
    assert {
        block.kind for block in plan.blocks
    } >= {"symbol", "lexical_high_document_frequency"}


def test_qualified_parent_plus_same_narrow_scope_survives_high_df_gate() -> None:
    related = [
        _member(
            f"narrow-fallback-{index:02d}",
            _semantics(
                rule,
                scope=("src/guardrails/response.py",),
            ),
        )
        for index, rule in enumerate(
            (
                "Must modify GuardrailsResponse and GuardrailsResponse.llm_response warning behavior",
                "Must modify GuardrailsResponse delegation behavior",
            )
        )
    ]
    unrelated = [
        _member(
            f"narrow-fallback-{index:02d}",
            _semantics(
                f"Must modify unrelated{index} behavior",
                scope=(f"src/guardrails/other_{index}.py",),
            ),
        )
        for index in range(2, 40)
    ]
    members = [*related, *unrelated]
    expected = tuple(sorted((members[0].member_id, members[1].member_id)))
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={},
        limits=_limits(
            max_candidate_members=40,
            max_block_members=8,
            max_case_members=8,
            max_group_pairs=1_000,
        ),
    )

    assert plan.semantic_pairs == (expected,)


def test_same_file_topic_word_does_not_trigger_high_df_fallback() -> None:
    related = [
        _member(
            f"topic-fallback-{index:02d}",
            _semantics(rule, scope=("src/checks/pii.py",)),
        )
        for index, rule in enumerate(
            (
                "Must modify PII entity definitions",
                "Must modify PII masking behavior",
            )
        )
    ]
    unrelated = [
        _member(
            f"topic-fallback-{index:02d}",
            _semantics(
                f"Must modify unrelated{index} behavior",
                scope=(f"src/checks/other_{index}.py",),
            ),
        )
        for index in range(2, 40)
    ]
    plan = plan_candidate_grouping(
        [*related, *unrelated],
        anchors_by_member={},
        limits=_limits(
            max_candidate_members=40,
            max_block_members=8,
            max_case_members=8,
            max_group_pairs=1_000,
        ),
    )

    assert plan.semantic_pairs == ()


def test_qualified_parent_fallback_does_not_lower_threshold_without_df_pruning() -> None:
    left = _member(
        "small-qualified-left",
        _semantics(
            "Widget.render alpha behavior",
            scope=("src/widget.py",),
        ),
    )
    right = _member(
        "small-qualified-right",
        _semantics(
            "Widget beta behavior",
            scope=("src/widget.py",),
        ),
    )
    plan = plan_candidate_grouping(
        [left, right],
        anchors_by_member={},
        limits=_limits(),
    )

    assert plan.semantic_pairs == ()


def test_failed_direct_star_edge_only_quarantines_its_potential_component() -> None:
    center = _member("local-center", _semantics("Local hub"))
    leaves = [
        _member(f"local-leaf-{index}", _semantics(f"Local leaf {index}"))
        for index in range(12)
    ]
    members = [center, *leaves]
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            center.member_id: MemberBlockingAnchors(
                symbols=tuple(f"local-edge-{index}" for index in range(12))
            ),
            **{
                leaf.member_id: MemberBlockingAnchors(
                    symbols=(f"local-edge-{index}",)
                )
                for index, leaf in enumerate(leaves)
            },
        },
        limits=_limits(
            max_candidate_members=20,
            max_block_members=20,
            max_case_members=20,
            max_group_pairs=100,
            max_group_pairs_per_request=4,
            max_group_members_per_request=5,
            max_anchors_per_member=20,
        ),
    )
    failed_pair = plan.semantic_pairs[0]
    by_id = {item.member_id: item for item in members}
    direct_relations = [
        _relation(by_id[left], by_id[right], "distinct")
        for left, right in plan.semantic_pairs
        if (left, right) != failed_pair
    ]
    plan, closure_pairs = resolve_direct_grouping(
        plan,
        direct_relations,
        failed_pairs=[failed_pair],
    )
    result = assemble_candidate_cases(
        plan,
        direct_relations,
        profile_digest=PROFILE,
    )

    assert closure_pairs == ()
    incomplete = [case for case in result.cases if case.grouping_status == "incomplete"]
    assert len(incomplete) == 1
    assert len(incomplete[0].members) == 2
    assert len(result.cases) == 12


def test_closure_batches_are_bounded_and_failure_is_case_local() -> None:
    center = _member("closure-batch-center", _semantics("Uvwhub"))
    leaves = [
        _member(f"closure-batch-{index}", _semantics(f"Uvwleaf{index}"))
        for index in range(7)
    ]
    members = [center, *leaves]
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={
            center.member_id: MemberBlockingAnchors(
                symbols=tuple(f"closure-batch-edge-{index}" for index in range(7))
            ),
            **{
                leaf.member_id: MemberBlockingAnchors(
                    symbols=(f"closure-batch-edge-{index}",)
                )
                for index, leaf in enumerate(leaves)
            },
        },
        limits=_limits(
            max_group_pairs=100,
            max_group_pairs_per_request=5,
            max_group_members_per_request=8,
        ),
    )
    by_id = {item.member_id: item for item in members}
    direct_relations = [
        _relation(by_id[left], by_id[right], "equivalent")
        for left, right in plan.semantic_pairs
    ]
    plan, closure_pairs = resolve_direct_grouping(plan, direct_relations)
    batches = plan_grouping_batches(plan, closure_pairs, phase="closure")
    failed = batches.batches[0].pairs
    successful_closure = [
        _relation(by_id[left], by_id[right], "equivalent")
        for batch in batches.batches[1:]
        for left, right in batch.pairs
    ]
    plan = mark_grouping_pairs_incomplete(
        plan,
        failed,
        reason_code="candidate_semantic_group_failed",
    )
    result = assemble_candidate_cases(
        plan,
        [*direct_relations, *successful_closure],
        profile_digest=PROFILE,
    )

    assert len(closure_pairs) == 21
    assert len(batches.batches) == 5
    assert result.cases[0].grouping_status == "incomplete"
    assert len(result.cases[0].relations) == 23


def test_lexical_threshold_is_monotonic() -> None:
    members = [
        _member("threshold-a", _semantics("alpha beta gamma delta left")),
        _member("threshold-b", _semantics("alpha beta gamma delta right")),
        _member("threshold-c", _semantics("alpha beta gamma other")),
    ]
    low = plan_candidate_grouping(
        members,
        anchors_by_member={},
        limits=_limits(lexical_min_shared_tokens=2),
    )
    high = plan_candidate_grouping(
        members,
        anchors_by_member={},
        limits=_limits(lexical_min_shared_tokens=4),
    )

    assert set(high.semantic_pairs) < set(low.semantic_pairs)
    assert high.incomplete_components == ()


def test_sixty_four_tokens_with_higher_threshold_has_constant_pair_work() -> None:
    shared = " ".join(f"s{index:02d}" for index in range(63))
    left = _member("wide-left", _semantics(f"{shared} uniqueleft"))
    right = _member("wide-right", _semantics(f"{shared} uniqueright"))
    plan = plan_candidate_grouping(
        [left, right],
        anchors_by_member={},
        limits=_limits(
            max_group_pairs=1,
            max_group_pairs_per_request=1,
            max_tokens_per_member=64,
            lexical_min_shared_tokens=6,
        ),
    )

    assert plan.semantic_pairs == (tuple(sorted((left.member_id, right.member_id))),)
    assert plan.incomplete_components == ()


def test_relation_matrix_rejects_missing_duplicate_extra_and_nonsemantic_cells() -> None:
    left = _member("left", _semantics("Validate plugin registration"))
    right = _member("right", _semantics("Check plugin registration"))
    plan = plan_candidate_grouping(
        [left, right],
        anchors_by_member={
            left.member_id: MemberBlockingAnchors(symbols=("PluginManager",)),
            right.member_id: MemberBlockingAnchors(symbols=("PluginManager",)),
        },
        limits=_limits(),
    )
    relation = _relation(left, right, "related_variant")

    with pytest.raises(GroupingValidationError, match="missing="):
        validate_semantic_relation_matrix(plan, [])
    with pytest.raises(GroupingValidationError, match="duplicate"):
        validate_semantic_relation_matrix(plan, [relation, relation])
    with pytest.raises(GroupingValidationError, match="invalid source"):
        validate_semantic_relation_matrix(
            plan, [relation.model_copy(update={"source": "host_distinct"})]
        )

    foreign = _member("foreign", _semantics("Unrelated foreign candidate"))
    with pytest.raises(GroupingValidationError, match="extra="):
        validate_semantic_relation_matrix(
            plan, [relation, _relation(left, foreign, "distinct")]
        )


def test_connected_component_keeps_uncertain_without_merging_members() -> None:
    first = _member("one", _semantics("Enable strict plugin loading"))
    second = _member("two", _semantics("Validate strict plugin loading"))
    third = _member("three", _semantics("Disable legacy plugin loading"))
    members = [first, second, third]
    common = MemberBlockingAnchors(symbols=("PluginManager",))
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={item.member_id: common for item in members},
        limits=_limits(),
    )
    values = {
        frozenset((first.member_id, second.member_id)): "related_variant",
        frozenset((second.member_id, third.member_id)): "uncertain",
        frozenset((first.member_id, third.member_id)): "distinct",
    }
    relations = _relations_for(plan, members, values)
    result = assemble_candidate_cases(plan, relations, profile_digest=PROFILE)

    assert len(result.cases) == 1
    case = result.cases[0]
    assert len(case.members) == 3
    assert len(case.relations) == 3
    assert case.grouping_status == "complete"
    assert case.grouping_reason_codes == ["uncertain_relation"]
    assert any(item.relation == "uncertain" for item in case.relations)


def test_distinct_relation_forms_two_cases_with_one_assignment_each() -> None:
    left = _member("left", _semantics("Require strict plugin loading"))
    right = _member("right", _semantics("Document optional plugin loading"))
    common = MemberBlockingAnchors(symbols=("PluginManager",))
    plan = plan_candidate_grouping(
        [left, right],
        anchors_by_member={left.member_id: common, right.member_id: common},
        limits=_limits(),
    )
    result = assemble_candidate_cases(
        plan, [_relation(left, right, "distinct")], profile_digest=PROFILE
    )

    assert len(result.cases) == 2
    assigned = [item.member_id for case in result.cases for item in case.members]
    assert sorted(assigned) == sorted((left.member_id, right.member_id))
    assert all(len(case.members) == 1 for case in result.cases)


def test_case_limit_is_applied_after_semantic_grouping() -> None:
    members = [
        _member(str(index), _semantics(f"Rule number {index}"))
        for index in range(3)
    ]
    common = MemberBlockingAnchors(symbols=("shared.anchor",))
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={item.member_id: common for item in members},
        limits=_limits(max_block_members=3, max_case_members=2),
    )

    assert len(plan.semantic_pairs) == 3
    assert plan.incomplete_components == ()
    relations = [
        _relation(
            next(item for item in members if item.member_id == left),
            next(item for item in members if item.member_id == right),
            "related_variant",
        )
        for left, right in plan.semantic_pairs
    ]
    result = assemble_candidate_cases(plan, relations, profile_digest=PROFILE)

    assert len(result.cases) == 1
    case = result.cases[0]
    assert case.grouping_status == "incomplete"
    assert len(case.members) == 3
    assert len(case.relations) == 3
    assert "grouping_incomplete" in case.grouping_reason_codes
    assert "grouping_case_member_limit" in case.grouping_reason_codes


def test_token_truncation_marks_only_affected_component_incomplete() -> None:
    bounded = _member("bounded", _semantics("alpha beta gamma delta epsilon"))
    survivor = _member("survivor", _semantics("short unique"), channel="implicit_current")
    plan = plan_candidate_grouping(
        [bounded, survivor],
        anchors_by_member={},
        limits=_limits(max_tokens_per_member=2),
    )
    result = assemble_candidate_cases(plan, [], profile_digest=PROFILE)

    by_member = {
        case.members[0].member_id: case
        for case in result.cases
        if len(case.members) == 1
    }
    assert by_member[bounded.member_id].grouping_status == "incomplete"
    assert "blocking_token_limit" in by_member[bounded.member_id].grouping_reason_codes
    assert by_member[survivor.member_id].grouping_status == "complete"


def test_lexical_pair_table_stops_at_budget_and_keeps_affected_set_together() -> None:
    members = [
        _member(str(index), _semantics(f"sharedtoken action{index}"))
        for index in range(3)
    ]
    plan = plan_candidate_grouping(
        members,
        anchors_by_member={},
        limits=_limits(
            max_group_pairs=1,
            max_group_pairs_per_request=1,
            lexical_min_shared_tokens=1,
        ),
    )

    assert len(plan.incomplete_components) == 1
    assert set(plan.incomplete_components[0].member_ids) == {
        item.member_id for item in members
    }
    assert "grouping_pair_budget" in plan.incomplete_components[0].reason_codes
    assert plan.semantic_pairs == ()

    result = assemble_candidate_cases(plan, [], profile_digest=PROFILE)
    assert len(result.cases) == 1
    assert result.cases[0].grouping_status == "incomplete"
