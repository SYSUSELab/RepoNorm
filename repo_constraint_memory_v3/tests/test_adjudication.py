from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3.adjudication import (
    adjudicate_candidates,
    classify_execution_risk,
    delivery_policy_for,
    finalize_candidate_decisions,
    prepare_candidate_decisions,
)
from repo_constraint_memory_v3.builder import _explicit_verification_inputs
from repo_constraint_memory_v3.chunking import chunk_sources
from repo_constraint_memory_v3.discovery import SourceFile
from repo_constraint_memory_v3.models import (
    BatchExtractionResponse,
    CandidateClaim,
    DeliveryPolicy,
    Evidence,
    ExecutionRisk,
)


def _source(path: str, content: str, kind: str = "instruction") -> SourceFile:
    encoded = content.encode("utf-8")
    return SourceFile(
        path=path,
        source_kind=kind,  # type: ignore[arg-type]
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        content=content,
    )


def _candidate(unit, *, strength: str, scope: list[str]) -> CandidateClaim:
    return CandidateClaim(
        rule=" ".join(unit.content.split()),
        claim_kind="normative_declaration",
        normative_strength=strength,  # type: ignore[arg-type]
        applies_to=scope,
        condition=None,
        evidence=[
            {
                "unit_id": unit.unit_id,
                "span_start_id": unit.spans[0].span_id,
                "span_end_id": unit.spans[-1].span_id,
            }
        ],
    )


def _line_candidate(
    unit,
    *,
    rule: str,
    strength: str,
    scope: list[str],
    span_index: int = -1,
) -> CandidateClaim:
    span = unit.spans[span_index]
    return CandidateClaim(
        rule=rule,
        claim_kind="normative_declaration",
        normative_strength=strength,  # type: ignore[arg-type]
        applies_to=scope,
        condition=None,
        evidence=[
            {
                "unit_id": unit.unit_id,
                "span_start_id": span.span_id,
                "span_end_id": span.span_id,
            }
        ],
    )


def _adjudicate_verified(candidates, **kwargs):
    prepared = prepare_candidate_decisions(candidates, **kwargs)
    for decision in prepared:
        if decision.host_eligible and not decision.errors:
            decision.verification_disposition = "eligible"
            decision.verification_reason_codes = ["verified_eligible"]
    return finalize_candidate_decisions(prepared)


def _workflow_candidate(unit, span_indexes: list[int]) -> CandidateClaim:
    return CandidateClaim(
        rule="The CI workflow runs pytest for pull requests.",
        claim_kind="workflow_fact",
        normative_strength="observed",
        applies_to=["**/*.py"],
        condition=None,
        evidence=[
            {
                "unit_id": unit.unit_id,
                "span_start_id": unit.spans[index].span_id,
                "span_end_id": unit.spans[index].span_id,
            }
            for index in span_indexes
        ],
    )


def _risk_evidence(
    quote: str,
    *,
    semantic_key: str = "workflow:.github/workflows/ci.yml:run:test",
) -> Evidence:
    digest = hashlib.sha256(quote.encode("utf-8")).hexdigest()
    return Evidence(
        type="workflow_step",
        adapter="workflow",
        semantic_key=semantic_key,
        path=".github/workflows/ci.yml",
        span_ids=["SPAN-" + digest[:24]],
        line_start=1,
        line_end=max(1, len(quote.splitlines())),
        quote=quote,
        quote_sha256=digest,
        source_content_sha256=digest,
        source_kind="workflow",
        discovery_origin="deterministic",
    )


@pytest.mark.parametrize(
    ("condition", "risk", "expected"),
    [
        (None, "read_only", "default_consider"),
        ("when editing Python files", "read_only", "condition_matched_only"),
        (None, "bounded_check", "default_consider"),
        ("before opening a pull request", "bounded_check", "condition_matched_only"),
        (None, "repository_wide_mutation", "reference_only"),
        ("when publishing", "external_side_effect", "reference_only"),
    ],
)
def test_delivery_policy_depends_on_condition_and_execution_risk_only(
    condition: str | None,
    risk: ExecutionRisk,
    expected: DeliveryPolicy,
) -> None:
    assert delivery_policy_for(condition, risk) == expected


def test_external_side_effect_overrides_operational_fact_kind() -> None:
    assert (
        classify_execution_risk(
            "operational_fact",
            "The release workflow publishes the package to PyPI.",
            [],
        )
        == "external_side_effect"
    )


def test_release_noun_does_not_imply_an_external_action() -> None:
    assert (
        classify_execution_risk(
            "normative_declaration",
            "Release changes must update the changelog.",
            [],
        )
        == "scoped_mutation"
    )
    assert (
        classify_execution_risk(
            "normative_declaration",
            "Contributors must publish the release artifact.",
            [],
        )
        == "external_side_effect"
    )
    assert (
        classify_execution_risk(
            "workflow_fact",
            "ignored sampled wording",
            [_risk_evidence("run: docker push example/image:latest\n")],
        )
        == "external_side_effect"
    )


def test_coverage_report_generation_is_a_bounded_check() -> None:
    assert (
        classify_execution_risk(
            "command_recipe",
            "Run pytest and generate the XML coverage report.",
            [],
        )
        == "bounded_check"
    )


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest --inline-snapshot=create",
        "uv run pytest --inline-snapshot=fix",
        "uv build",
        "ruff format .",
        "uv run pre-commit run -a",
        "pre-commit run --all-files",
        "Prefer running full pre-commit over individual tools (ruff/mypy).",
        "Run the complete pre-commit suite before submission.",
        "python scripts/release.py 1.2.3",
        "python scripts/regen.py",
    ],
)
def test_write_mode_and_generated_artifact_commands_are_repository_mutations(
    command: str,
) -> None:
    assert classify_execution_risk("command_recipe", command, []) == (
        "repository_wide_mutation"
    )


@pytest.mark.parametrize(
    "command",
    [
        "uv pip install -e .",
        "pip install -r requirements.txt",
        "python -m pip install hatch",
        "uv sync --frozen",
    ],
)
def test_dependency_install_and_sync_commands_are_external_side_effects(
    command: str,
) -> None:
    assert classify_execution_risk("command_recipe", command, []) == (
        "external_side_effect"
    )


@pytest.mark.parametrize(
    ("kind", "rule"),
    [
        ("operational_fact", "The hook runs autoflake --in-place."),
        ("operational_fact", "id: ruff-format"),
        ("enforced_configuration", "The hook invokes pyupgrade."),
        ("command_recipe", "ruff check --fix ."),
        ("command_recipe", "make format"),
        ("normative_declaration", "Always run uv build."),
        ("normative_declaration", "Run ruff format ."),
    ],
)
def test_mutating_config_and_precommit_actions_override_read_only_kind(
    kind: str,
    rule: str,
) -> None:
    assert classify_execution_risk(kind, rule, []) == "repository_wide_mutation"


@pytest.mark.parametrize(
    ("kind", "rule"),
    [
        ("command_recipe", "ruff format --check ."),
        ("command_recipe", "uv run pytest"),
        ("normative_declaration", "Always run pytest before submitting."),
        ("normative_declaration", "Type checks must pass."),
    ],
)
def test_non_writing_validation_actions_are_bounded(
    kind: str,
    rule: str,
) -> None:
    assert classify_execution_risk(kind, rule, []) == "bounded_check"


def test_adding_tests_is_not_misclassified_as_running_a_check() -> None:
    assert (
        classify_execution_risk(
            "normative_declaration",
            "Contributors must add tests for every change.",
            [],
        )
        == "scoped_mutation"
    )


def test_direct_reread_instruction_is_read_only() -> None:
    assert (
        classify_execution_risk(
            "normative_declaration",
            "Always re-read AGENTS.md before editing files.",
            [],
        )
        == "read_only"
    )


def test_workflow_risk_scans_action_not_predicate_or_sampled_rule() -> None:
    predicate = _risk_evidence(
        "if: github.ref == 'refs/heads/release'\n",
        semantic_key="workflow:.github/workflows/ci.yml:if:release",
    )
    action = _risk_evidence("run: pytest --cov=src\n")

    assert classify_execution_risk(
        "workflow_fact",
        "The release workflow publishes test results.",
        [predicate, action],
    ) == "bounded_check"
    assert classify_execution_risk(
        "workflow_fact",
        "The workflow checks a release predicate.",
        [predicate, _risk_evidence("run: uv publish\n")],
    ) == "external_side_effect"


def test_strength_conflict_is_not_silently_merged() -> None:
    hard_source = _source("AGENTS.md", "Contributors must add tests.\n")
    soft_source = _source(
        "CONTRIBUTING.md",
        "Contributors should add tests.\n",
        "documentation",
    )
    hard = chunk_sources([hard_source], max_unit_chars=1_000)[0]
    soft = chunk_sources([soft_source], max_unit_chars=1_000)[0]

    result = _adjudicate_verified(
        [
            (hard.unit_id, "direct_norm", _candidate(hard, strength="required", scope=["**"])),
            (
                soft.unit_id,
                "direct_norm",
                _candidate(soft, strength="recommended", scope=["**"]),
            ),
        ],
        units_by_id={hard.unit_id: hard, soft.unit_id: soft},
        sources_by_path={
            hard_source.path: hard_source,
            soft_source.path: soft_source,
        },
        manifest_paths={"AGENTS.md", "CONTRIBUTING.md", "src/pkg.py"},
    )

    assert not result.claims
    assert {item.outcome for item in result.decisions} == {"review"}
    assert all(item.outcome_stage == "merge" for item in result.decisions)


def test_rule_cannot_upgrade_recommended_evidence_to_required() -> None:
    source = _source("AGENTS.md", "Contributors should add tests.\n")
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _candidate(
        unit,
        strength="required",
        scope=["**"],
    ).model_copy(update={"rule": "Contributors must add tests."})

    result = _adjudicate_verified(
        [(unit.unit_id, "direct_norm", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py"},
    )

    assert not result.claims
    assert result.decisions[0].derived_strength == "recommended"
    assert any(
        "rule=required, evidence=recommended" in error
        for error in result.decisions[0].errors
    )


@pytest.mark.parametrize(
    ("evidence_text", "rule", "expected_strength"),
    [
        (
            "Contributors should add tests.\n",
            "Contributors should add tests.",
            "recommended",
        ),
        (
            "Contributors must add tests.\n",
            "Contributors should add tests.",
            "required",
        ),
    ],
)
def test_equal_or_weaker_rule_force_remains_eligible(
    evidence_text: str,
    rule: str,
    expected_strength: str,
) -> None:
    source = _source("AGENTS.md", evidence_text)
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _candidate(
        unit,
        strength="required",
        scope=["**"],
    ).model_copy(update={"rule": rule})

    result = _adjudicate_verified(
        [(unit.unit_id, "direct_norm", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py"},
    )

    assert len(result.claims) == 1
    assert result.claims[0].normative_strength == expected_strength


def test_observed_configuration_cannot_use_must_wording() -> None:
    source = _source(
        "pyproject.toml",
        "[tool.ruff]\nline-length = 88\n",
        "configuration",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = CandidateClaim(
        rule="Ruff must enforce a line length of 88.",
        claim_kind="operational_fact",
        normative_strength="observed",
        applies_to=["**/*.py"],
        condition=None,
        evidence=[
            {
                "unit_id": unit.unit_id,
                "span_start_id": unit.spans[1].span_id,
                "span_end_id": unit.spans[1].span_id,
            }
        ],
    )

    result = _adjudicate_verified(
        [(unit.unit_id, "configuration_fact", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"pyproject.toml", "src/pkg.py"},
    )

    assert not result.claims
    assert result.decisions[0].derived_strength == "observed"
    assert any(
        "rule=required, evidence=observed" in error
        for error in result.decisions[0].errors
    )


def test_observed_negative_fact_is_not_mistaken_for_normative_force() -> None:
    source = _source(
        "pyproject.toml",
        "[tool.coverage.run]\nbranch = false\n",
        "configuration",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = CandidateClaim(
        rule="Branch coverage is never enabled by this setting.",
        claim_kind="operational_fact",
        normative_strength="observed",
        applies_to=["**/*.py"],
        condition=None,
        evidence=[
            {
                "unit_id": unit.unit_id,
                "span_start_id": unit.spans[1].span_id,
                "span_end_id": unit.spans[1].span_id,
            }
        ],
    )

    result = _adjudicate_verified(
        [(unit.unit_id, "configuration_fact", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"pyproject.toml", "src/pkg.py"},
    )

    assert len(result.claims) == 1
    assert not result.decisions[0].errors


def test_positive_and_negative_directives_keep_distinct_identity() -> None:
    positive_source = _source(
        "AGENTS.md",
        "Contributors must use Ruff.\n",
    )
    negative_source = _source(
        "CONTRIBUTING.md",
        "Contributors do not use Ruff.\n",
        "documentation",
    )
    positive = chunk_sources([positive_source], max_unit_chars=1_000)[0]
    negative = chunk_sources([negative_source], max_unit_chars=1_000)[0]

    result = _adjudicate_verified(
        [
            (
                positive.unit_id,
                "direct_norm",
                _candidate(positive, strength="required", scope=["**"]),
            ),
            (
                negative.unit_id,
                "direct_norm",
                _candidate(negative, strength="required", scope=["**"]),
            ),
        ],
        units_by_id={positive.unit_id: positive, negative.unit_id: negative},
        sources_by_path={
            positive_source.path: positive_source,
            negative_source.path: negative_source,
        },
        manifest_paths={"AGENTS.md", "CONTRIBUTING.md", "src/pkg.py"},
    )

    assert len(result.claims) == 2
    assert {item.rule for item in result.claims} == {
        "Contributors must use Ruff.",
        "Contributors do not use Ruff.",
    }
    assert all(len(item.evidence) == 1 for item in result.claims)


@pytest.mark.parametrize(
    "token",
    [
        "yourpkg",
        "your_package",
        "your-project",
        "your module",
        "yourapp",
        "your_repo",
    ],
)
def test_unbound_template_package_tokens_cannot_be_accepted(token: str) -> None:
    source = _source(
        "AGENTS.md",
        f"Contributors must measure coverage for src/{token}.\n",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    result = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "direct_norm",
                _candidate(unit, strength="required", scope=["**/*.py"]),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py"},
    )

    assert not result.claims
    assert any(
        error.startswith("unbound template package token:")
        for error in result.decisions[0].errors
    )


def test_manifest_bound_package_name_is_not_treated_as_a_template() -> None:
    source = _source(
        "AGENTS.md",
        "Contributors must measure coverage for src/your_package.\n",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    result = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "direct_norm",
                _candidate(unit, strength="required", scope=["**/*.py"]),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={
            "AGENTS.md",
            "src/your_package/__init__.py",
        },
    )

    assert len(result.claims) == 1


def test_manifest_bound_single_file_module_stem_is_not_a_template() -> None:
    source = _source(
        "AGENTS.md",
        "Contributors must measure coverage for yourpkg.py.\n",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _candidate(unit, strength="required", scope=["yourpkg.py"])

    bound = _adjudicate_verified(
        [(unit.unit_id, "direct_norm", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "yourpkg.py"},
    )
    unbound = _adjudicate_verified(
        [(unit.unit_id, "direct_norm", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "otherpkg.py"},
    )

    assert len(bound.claims) == 1
    assert not unbound.claims
    assert any(
        error == "unbound template package token: yourpkg"
        for error in unbound.decisions[0].errors
    )


@pytest.mark.parametrize(
    "text",
    [
        "Contributors must name unit tests `test_<modulename>.py`.\n",
        (
            "Contributors must gate release work on "
            "`${{ github.repository }}`.\n"
        ),
    ],
)
def test_schematic_names_and_github_expressions_are_not_template_tokens(
    text: str,
) -> None:
    source = _source("AGENTS.md", text)
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    result = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "direct_norm",
                _candidate(unit, strength="required", scope=["**"]),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py", "tests/test_pkg.py"},
    )

    assert len(result.claims) == 1
    assert not any(
        "template package token" in error
        for error in result.decisions[0].errors
    )


def test_imperative_nested_example_item_cannot_be_accepted() -> None:
    source = _source(
        "AGENTS.md",
        "- Examples:\n  - Always use Hypothesis for every test.\n",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = CandidateClaim(
        rule="Always use Hypothesis for every test.",
        claim_kind="normative_declaration",
        normative_strength="required",
        applies_to=["tests/**"],
        condition=None,
        evidence=[
            {
                "unit_id": unit.unit_id,
                "span_start_id": unit.spans[1].span_id,
                "span_end_id": unit.spans[1].span_id,
            }
        ],
    )
    result = _adjudicate_verified(
        [(unit.unit_id, "direct_norm", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "tests/test_pkg.py"},
    )

    assert not result.claims
    assert "prose is nested under a typed example context" in (
        result.decisions[0].errors
    )


@pytest.mark.parametrize(
    "scope",
    [
        ["**"],
        ["**/*.py"],
        ["src/**"],
        ["test/**"],
    ],
)
def test_test_section_artifact_rule_rejects_scope_outside_manifest_test_roots(
    scope: list[str],
) -> None:
    source = _source(
        "AGENTS.md",
        "## Testing\n"
        "### Test naming\n"
        "- Each test function must start with `test_`.\n",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[-1]
    candidate = _line_candidate(
        unit,
        rule="Each test function must start with `test_`.",
        strength="required",
        scope=scope,
    )
    result = _adjudicate_verified(
        [(unit.unit_id, "direct_norm", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py", "tests/test_pkg.py"},
    )

    assert not result.claims
    assert "test_section_context" in (
        result.decisions[0].grounded[0].semantic_flags
    )
    assert any(
        "outside manifest test roots" in error
        for error in result.decisions[0].errors
    )


@pytest.mark.parametrize("scope", [["tests/**"], ["tests/unit/**"]])
def test_test_section_artifact_rule_accepts_manifest_test_scope(
    scope: list[str],
) -> None:
    source = _source(
        "AGENTS.md",
        "## Testing\n"
        "### Test style\n"
        "- Prefer one assertion per behavior.\n",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[-1]
    candidate = _line_candidate(
        unit,
        rule="Prefer one assertion per behavior.",
        strength="recommended",
        scope=scope,
    )
    result = _adjudicate_verified(
        [(unit.unit_id, "direct_norm", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={
            "AGENTS.md",
            "src/pkg.py",
            "tests/unit/test_pkg.py",
        },
    )

    assert len(result.claims) == 1
    assert result.claims[0].applies_to == scope


@pytest.mark.parametrize(
    ("rule", "strength"),
    [
        ("Each test function must start with `test_`.", "required"),
        ("Prefer one assertion per behavior.", "recommended"),
        (
            "Avoid logic and branching in test code except when asserting both outcomes.",
            "required",
        ),
    ],
)
def test_known_test_artifact_rules_cannot_keep_repository_python_scope(
    rule: str,
    strength: str,
) -> None:
    source = _source("AGENTS.md", f"## Test Style Guidelines\n- {rule}\n")
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    result = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "direct_norm",
                _line_candidate(
                    unit,
                    rule=rule,
                    strength=strength,
                    scope=["**/*.py"],
                ),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py", "tests/test_pkg.py"},
    )

    assert not result.claims
    assert any(
        error.endswith("outside manifest test roots: **/*.py")
        for error in result.decisions[0].errors
    )


@pytest.mark.parametrize(
    ("heading", "rule"),
    [
        (
            "Error Handling",
            "Code must differentiate programmer errors (`assert`) from user errors.",
        ),
        (
            "Testing",
            "All async code must be tested with `pytest.mark.asyncio`.",
        ),
        (
            "Testing",
            "All production code must be covered by pytest at 95%.",
        ),
        (
            "Testing",
            "Use hypothesis for all non-trivial data logic.",
        ),
    ],
)
def test_non_test_artifact_obligations_do_not_trigger_test_scope_gate(
    heading: str,
    rule: str,
) -> None:
    source = _source("AGENTS.md", f"## {heading}\n- {rule}\n")
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _line_candidate(
        unit,
        rule=rule,
        strength="required",
        scope=["**/*.py"],
    )
    result = _adjudicate_verified(
        [(unit.unit_id, "direct_norm", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py", "tests/test_pkg.py"},
    )

    assert not any(
        "manifest test roots" in error for error in result.decisions[0].errors
    )
    assert len(result.claims) == 1


def test_same_core_scope_relations_are_present_on_exported_claims() -> None:
    first_source = _source(
        "AGENTS.md",
        "Contributors must add tests.\n",
    )
    second_source = _source(
        "CONTRIBUTING.md",
        "Contributors must add tests.\n",
        "documentation",
    )
    first = chunk_sources([first_source], max_unit_chars=1_000)[0]
    second = chunk_sources([second_source], max_unit_chars=1_000)[0]
    result = _adjudicate_verified(
        [
            (
                first.unit_id,
                "direct_norm",
                _candidate(first, strength="required", scope=["src/**"]),
            ),
            (
                second.unit_id,
                "direct_norm",
                _candidate(second, strength="required", scope=["tests/**"]),
            ),
        ],
        units_by_id={first.unit_id: first, second.unit_id: second},
        sources_by_path={
            first_source.path: first_source,
            second_source.path: second_source,
        },
        manifest_paths={
            "AGENTS.md",
            "CONTRIBUTING.md",
            "src/pkg.py",
            "tests/test_pkg.py",
        },
    )

    assert len(result.claims) == 2
    assert all(
        [item.relation for item in claim.evidence_relations]
        == ["refines_scope"]
        for claim in result.claims
    )


def test_global_scope_is_valid_but_nested_instruction_escape_is_review() -> None:
    root_source = _source("AGENTS.md", "Contributors must add tests.\n")
    nested_source = _source(
        "src/pkg/AGENTS.md",
        "Package changes must preserve imports.\n",
    )
    root = chunk_sources([root_source], max_unit_chars=1_000)[0]
    nested = chunk_sources([nested_source], max_unit_chars=1_000)[0]

    result = _adjudicate_verified(
        [
            (root.unit_id, "direct_norm", _candidate(root, strength="required", scope=["**"])),
            (
                nested.unit_id,
                "direct_norm",
                _candidate(nested, strength="required", scope=["**"]),
            ),
        ],
        units_by_id={root.unit_id: root, nested.unit_id: nested},
        sources_by_path={
            root_source.path: root_source,
            nested_source.path: nested_source,
        },
        manifest_paths={"AGENTS.md", "src/pkg/AGENTS.md", "src/pkg/api.py"},
    )

    assert len(result.claims) == 1
    assert result.claims[0].applies_to == ["**"]
    nested_decision = next(
        item for item in result.decisions if item.unit_id == nested.unit_id
    )
    assert nested_decision.outcome == "review"
    assert "escapes" in nested_decision.errors[0]


def test_claim_id_is_deterministic() -> None:
    source = _source("AGENTS.md", "Contributors must add tests.\n")
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    inputs = [(unit.unit_id, "direct_norm", _candidate(unit, strength="required", scope=["**"]))]
    kwargs = {
        "units_by_id": {unit.unit_id: unit},
        "sources_by_path": {source.path: source},
        "manifest_paths": {"AGENTS.md", "src/pkg.py"},
    }
    first = _adjudicate_verified(inputs, **kwargs)
    second = _adjudicate_verified(inputs, **kwargs)
    assert first.claims[0].id == second.claims[0].id
    assert first.claims[0].proposition_key == second.claims[0].proposition_key
    assert first.claims[0].adjudication.policy_version == "1.9"


def test_host_finalize_cannot_accept_without_explicit_verification() -> None:
    source = _source("AGENTS.md", "Contributors must add tests.\n")
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    result = adjudicate_candidates(
        [
            (
                unit.unit_id,
                "direct_norm",
                _candidate(unit, strength="required", scope=["**"]),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py"},
    )

    assert not result.claims
    assert result.decisions[0].outcome == "review"
    assert result.decisions[0].outcome_stage == "explicit_semantic_verify"
    assert result.decisions[0].errors == ["explicit_verification:not_run"]


def test_verifier_input_preserves_raw_condition_for_host_ineligible_root() -> None:
    source = _source("AGENTS.md", "The test suite runs quickly.\n")
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _candidate(unit, strength="observed", scope=["**"]).model_copy(
        update={"condition": "when editing tests"}
    )
    prepared = prepare_candidate_decisions(
        [(unit.unit_id, "direct_norm", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py"},
    )

    assert not prepared[0].host_eligible
    assert len(prepared[0].grounded) == 1
    verifier_input = _explicit_verification_inputs(prepared)
    assert len(verifier_input) == 1
    assert verifier_input[0].condition == "when editing tests"


def test_verifier_input_uses_host_canonicalized_empty_condition() -> None:
    source = _source("AGENTS.md", "Contributors must add tests.\n")
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _candidate(
        unit,
        strength="required",
        scope=["**"],
    ).model_copy(update={"condition": "always"})
    prepared = prepare_candidate_decisions(
        [(unit.unit_id, "direct_norm", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py"},
    )

    assert prepared[0].host_eligible
    assert prepared[0].condition_source == "none"
    verifier_input = _explicit_verification_inputs(prepared)
    assert verifier_input[0].condition is None


@pytest.mark.parametrize(
    ("text", "expected_strength"),
    [
        ("Run all tests with `uv run pytest`.\n", "required"),
        ("Always reread files changed by pre-commit.\n", "required"),
        ("Consider backward compatibility.\n", "recommended"),
        ("All commands use `uv run`.\n", "required"),
    ],
)
def test_direct_instructions_do_not_require_modal_words(
    text: str,
    expected_strength: str,
) -> None:
    source = _source("AGENTS.md", text)
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    result = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "direct_norm",
                _candidate(unit, strength="observed", scope=["**"]),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py"},
    )

    assert len(result.claims) == 1
    assert result.claims[0].normative_strength == expected_strength
    assert "direct_instruction_text" in result.claims[0].acceptance_basis


def test_negated_requirement_cannot_be_promoted_to_required() -> None:
    source = _source(
        "AGENTS.md",
        "Contributors are not required to run slow tests.\n",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    result = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "direct_norm",
                _candidate(unit, strength="required", scope=["**"]),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py"},
    )

    assert not result.claims
    assert any(
        "no direct normative or directive language" in error
        for error in result.decisions[0].errors
    )


def test_documented_command_bullet_is_observed_recipe() -> None:
    source = _source(
        "CLAUDE.md",
        "- `uv run pytest testing/benchmark.py` - Run benchmark tests.\n",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    result = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "command_fact",
                _candidate(unit, strength="required", scope=["**"]),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"CLAUDE.md", "src/pkg.py"},
    )

    assert len(result.claims) == 1
    assert result.claims[0].claim_kind == "command_recipe"
    assert result.claims[0].normative_strength == "observed"


def test_descriptive_prose_is_not_treated_as_a_direct_instruction() -> None:
    source = _source("AGENTS.md", "The test suite runs quickly.\n")
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    result = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "direct_norm",
                _candidate(unit, strength="required", scope=["**"]),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"AGENTS.md", "src/pkg.py"},
    )

    assert not result.claims
    assert result.decisions[0].outcome == "review"
    assert any(
        "no direct normative or directive language" in error
        for error in result.decisions[0].errors
    )


def test_configuration_section_is_provenance_not_activation_condition() -> None:
    source = _source(
        "pyproject.toml",
        "[tool.ruff.lint]\nextend-select = [\"I\", \"F\"]\n",
        "configuration",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = CandidateClaim(
        rule="Ruff selects I and F lint rules.",
        claim_kind="enforced_configuration",
        normative_strength="observed",
        applies_to=["**/*.py"],
        condition=None,
        evidence=[
            {
                "unit_id": unit.unit_id,
                "span_start_id": unit.spans[0].span_id,
                "span_end_id": unit.spans[-1].span_id,
            }
        ],
    )
    result = _adjudicate_verified(
        [(unit.unit_id, "configuration_fact", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"pyproject.toml", "src/pkg.py"},
    )

    assert result.claims[0].condition is None
    assert result.claims[0].delivery_policy == "default_consider"
    assert result.decisions[0].condition_source == "typed_unconditional"
    assert result.decisions[0].proposition is not None
    assert result.decisions[0].proposition.condition is None
    assert "condition_derived_from_typed_context" not in (
        result.claims[0].adjudication.reasons
    )

    sampled_condition = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "configuration_fact",
                candidate.model_copy(
                    update={"condition": "when running Ruff lint"}
                ),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"pyproject.toml", "src/pkg.py"},
    )
    assert sampled_condition.claims[0].condition is None
    assert sampled_condition.claims[0].id == result.claims[0].id
    assert sampled_condition.decisions[0].condition_source == "typed_unconditional"


def test_command_target_is_the_host_canonical_condition() -> None:
    source = _source(
        "Makefile",
        "tests:\n\tuv run pytest\n",
        "command",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    base = _candidate(unit, strength="observed", scope=["**"])
    first = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "command_fact",
                base.model_copy(update={"condition": "when testing"}),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"Makefile", "src/pkg.py"},
    )
    second = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "command_fact",
                base.model_copy(update={"condition": "Make tests"}),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"Makefile", "src/pkg.py"},
    )

    assert first.claims[0].condition == "command target: tests"
    assert first.claims[0].delivery_policy == "condition_matched_only"
    assert first.claims[0].id == second.claims[0].id


def test_workflow_action_without_typed_predicate_stays_in_review() -> None:
    source = _source(
        ".github/workflows/ci.yml",
        "on: [pull_request]\nsteps:\n  - run: pytest\n",
        "workflow",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    result = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "workflow_fact",
                _workflow_candidate(unit, [2]),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={
            ".github/workflows/ci.yml",
            "src/pkg.py",
        },
    )

    assert not result.claims
    assert result.decisions[0].condition_source == "typed_action_only"
    assert "action-only workflow evidence cannot support trigger or activation wording" in (
        result.decisions[0].errors
    )


def test_workflow_action_only_can_support_an_unconditional_fact() -> None:
    source = _source(
        ".github/workflows/ci.yml",
        "on: [pull_request]\nsteps:\n  - run: pytest\n",
        "workflow",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _workflow_candidate(unit, [2]).model_copy(
        update={"rule": "The CI workflow runs pytest."}
    )
    result = _adjudicate_verified(
        [(unit.unit_id, "workflow_fact", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={".github/workflows/ci.yml", "src/pkg.py"},
    )

    assert len(result.claims) == 1
    assert result.claims[0].condition is None
    assert result.claims[0].execution_risk == "bounded_check"
    assert result.decisions[0].condition_source == "typed_action_only"


def test_workflow_action_only_cannot_keep_model_supplied_condition() -> None:
    source = _source(
        ".github/workflows/ci.yml",
        "on: [pull_request]\nsteps:\n  - run: pytest\n",
        "workflow",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _workflow_candidate(unit, [2]).model_copy(
        update={
            "rule": "The CI workflow runs pytest.",
            "condition": "conditioned on code changes",
        }
    )
    result = _adjudicate_verified(
        [(unit.unit_id, "workflow_fact", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={".github/workflows/ci.yml", "src/pkg.py"},
    )

    assert not result.claims
    assert any(
        "cannot support a model-supplied activation condition" in error
        for error in result.decisions[0].errors
    )


def test_workflow_action_only_cannot_assert_unevidenced_platform() -> None:
    source = _source(
        ".github/workflows/ci.yml",
        "on: [pull_request]\nsteps:\n  - run: pytest\n",
        "workflow",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _workflow_candidate(unit, [2]).model_copy(
        update={"rule": "The workflow runs pytest only on Windows."}
    )
    result = _adjudicate_verified(
        [(unit.unit_id, "workflow_fact", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={".github/workflows/ci.yml", "src/pkg.py"},
    )

    assert not result.claims
    assert any(
        "cannot support trigger or activation wording" in error
        for error in result.decisions[0].errors
    )


def test_observed_workflow_action_cannot_be_reworded_as_must() -> None:
    source = _source(
        ".github/workflows/ci.yml",
        "on: [pull_request]\nsteps:\n  - run: pytest\n",
        "workflow",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _workflow_candidate(unit, [2]).model_copy(
        update={"rule": "The CI workflow must run pytest."}
    )
    result = _adjudicate_verified(
        [(unit.unit_id, "workflow_fact", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={".github/workflows/ci.yml", "src/pkg.py"},
    )

    assert not result.claims
    assert any(
        "rule=required, evidence=observed" in error
        for error in result.decisions[0].errors
    )


def test_workflow_predicate_without_typed_action_stays_in_review() -> None:
    source = _source(
        ".github/workflows/publish.yml",
        "on: [release]\nsteps:\n  - uses: pypa/gh-action-pypi-publish@release/v1\n",
        "workflow",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _workflow_candidate(unit, [0]).model_copy(
        update={"rule": "Publishing to PyPI runs for a GitHub release."}
    )
    result = _adjudicate_verified(
        [(unit.unit_id, "workflow_fact", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={
            ".github/workflows/publish.yml",
            "src/pkg.py",
        },
    )

    assert not result.claims
    assert result.decisions[0].condition_source == "typed_context"
    assert "workflow evidence must identify an exact typed run/uses action" in (
        result.decisions[0].errors
    )


def test_workflow_predicate_is_the_exact_host_canonical_condition() -> None:
    source = _source(
        ".github/workflows/ci.yml",
        "on: [pull_request]\nsteps:\n  - run: pytest\n",
        "workflow",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    candidate = _workflow_candidate(unit, [0, 2]).model_copy(
        update={"condition": "whenever CI runs"}
    )
    result = _adjudicate_verified(
        [(unit.unit_id, "workflow_fact", candidate)],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={
            ".github/workflows/ci.yml",
            "src/pkg.py",
        },
    )

    assert len(result.claims) == 1
    assert result.claims[0].condition == "workflow predicate: [pull_request]"
    assert result.claims[0].execution_risk == "bounded_check"
    assert result.decisions[0].condition_source == "typed_context"


def test_uncertain_unit_cannot_smuggle_candidates_into_adjudication() -> None:
    with pytest.raises(ValidationError):
        BatchExtractionResponse.model_validate(
            {
                "results": [
                    {
                        "unit_id": "UNIT-" + "a" * 24,
                        "classification": "uncertain",
                        "reason_code": "scope_unclear",
                        "claims": [
                            {
                                "rule": "Run tests.",
                                "claim_kind": "normative_declaration",
                                "normative_strength": "required",
                                "applies_to": ["**"],
                                "condition": None,
                                "evidence": [
                                    {
                                        "unit_id": "UNIT-" + "a" * 24,
                                        "span_start_id": "SPAN-" + "b" * 24,
                                        "span_end_id": "SPAN-" + "b" * 24,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        )


def test_security_reporting_directive_is_reference_only() -> None:
    source = _source(
        "SECURITY.md",
        "Report security vulnerabilities through the private tracker.\n",
    )
    unit = chunk_sources([source], max_unit_chars=1_000)[0]
    result = _adjudicate_verified(
        [
            (
                unit.unit_id,
                "direct_norm",
                _candidate(unit, strength="required", scope=["**"]),
            )
        ],
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        manifest_paths={"SECURITY.md", "src/pkg.py"},
    )

    assert result.claims[0].execution_risk == "external_side_effect"
    assert result.claims[0].delivery_policy == "reference_only"
