from __future__ import annotations

from pathlib import Path

import pytest

from helpers import (
    build_fixture,
    config_for_tests,
    copied_fixture,
    discover_fixture,
    supplement_for,
)
from repo_constraint_memory_v3.chunking import chunk_source
from repo_constraint_memory_v3.discovery import SourceFile
from repo_constraint_memory_v3.models import CandidateEvidenceRange, EvidenceUnit
from repo_constraint_memory_v3.source_adapters import (
    AdapterError, GroundedEvidence, ground_evidence_range,
)


def _workflow_source(content: str) -> tuple[SourceFile, EvidenceUnit]:
    source = SourceFile(
        path=".github/workflows/ci.yml",
        source_kind="workflow",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="0" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_source(source, max_unit_chars=10_000)[0]
    source = SourceFile(
        **{
            **source.__dict__,
            "content_sha256": unit.source_content_sha256,
        }
    )
    return source, unit


def _ground_whole_workflow(content: str):
    source, unit = _workflow_source(content)
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[0].span_id,
        span_end_id=unit.spans[-1].span_id,
    )
    return ground_evidence_range(
        selection,
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        allowed_unit_ids={unit.unit_id},
    )


def _configuration_source(
    content: str,
    *,
    path: str,
) -> tuple[SourceFile, EvidenceUnit]:
    source = SourceFile(
        path=path,
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="0" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_source(source, max_unit_chars=10_000)[0]
    return (
        SourceFile(
            **{
                **source.__dict__,
                "content_sha256": unit.source_content_sha256,
            }
        ),
        unit,
    )


def _prose_source(
    content: str,
    *,
    path: str = "CONTRIBUTING.md",
) -> tuple[SourceFile, EvidenceUnit]:
    source = SourceFile(
        path=path,
        source_kind="instruction",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="0" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_source(source, max_unit_chars=10_000)[0]
    return (
        SourceFile(
            **{
                **source.__dict__,
                "content_sha256": unit.source_content_sha256,
            }
        ),
        unit,
    )


def _ground_whole_prose(content: str):
    source, unit = _prose_source(content)
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[0].span_id,
        span_end_id=unit.spans[-1].span_id,
    )
    return ground_evidence_range(
        selection,
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        allowed_unit_ids={unit.unit_id},
    )


def _ground_prose_line(
    content: str,
    line_index: int,
    *,
    path: str = "CONTRIBUTING.md",
):
    initial = SourceFile(
        path=path,
        source_kind="instruction",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="0" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = next(
        item
        for item in chunk_source(initial, max_unit_chars=10_000)
        if any(span.line_number == line_index + 1 for span in item.spans)
    )
    source = SourceFile(
        **{
            **initial.__dict__,
            "content_sha256": unit.source_content_sha256,
        }
    )
    span = next(
        item for item in unit.spans if item.line_number == line_index + 1
    )
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=span.span_id,
        span_end_id=span.span_id,
    )
    return ground_evidence_range(
        selection,
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        allowed_unit_ids={unit.unit_id},
    )


def _ini_source(
    content: str,
    *,
    unit_index: int = 0,
) -> tuple[SourceFile, EvidenceUnit]:
    source = SourceFile(
        path="tox.ini",
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="0" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_source(source, max_unit_chars=10_000)[unit_index]
    return (
        SourceFile(
            **{
                **source.__dict__,
                "content_sha256": unit.source_content_sha256,
            }
        ),
        unit,
    )


def test_ini_multiline_assignment_recovers_literal_continuation() -> None:
    source, unit = _ini_source("[pytest]\nfilterwarnings =\n  error\n")
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[1].span_id,
        span_end_id=unit.spans[2].span_id,
    )
    grounded = ground_evidence_range(
        selection,
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        allowed_unit_ids={unit.unit_id},
    )

    assert grounded.evidence.semantic_key == "ini_cfg:pytest.filterwarnings=error"


def test_ini_truncated_selection_expands_to_one_complete_assignment() -> None:
    source, unit = _ini_source(
        "[pytest]\n"
        "filterwarnings =\n"
        "  error\n"
        "  ignore:legacy warning\n"
        "addopts = -ra\n"
    )
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[1].span_id,
        span_end_id=unit.spans[1].span_id,
    )

    grounded = ground_evidence_range(
        selection,
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        allowed_unit_ids={unit.unit_id},
    )

    assert grounded.evidence.line_start == 2
    assert grounded.evidence.line_end == 4
    assert grounded.evidence.span_ids == [
        unit.spans[index].span_id for index in range(1, 4)
    ]
    assert grounded.evidence.quote == (
        "filterwarnings =\n  error\n  ignore:legacy warning\n"
    )
    assert grounded.evidence.semantic_key == (
        "ini_cfg:pytest.filterwarnings=error ignore:legacy warning"
    )


def test_long_complete_ini_assignment_uses_shared_logical_block_policy() -> None:
    content = "[report]\nexclude_lines =\n" + "".join(
        f"  pattern-{index}\n" for index in range(13)
    )
    source, unit = _ini_source(content)
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[1].span_id,
        span_end_id=unit.spans[-1].span_id,
    )

    grounded = ground_evidence_range(
        selection,
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        allowed_unit_ids={unit.unit_id},
    )

    assert grounded.evidence.line_end - grounded.evidence.line_start + 1 == 14
    assert grounded.evidence.semantic_key.startswith("ini_cfg:report.exclude_lines=")


def test_ini_evidence_cannot_merge_peer_assignments() -> None:
    source, unit = _ini_source(
        "[testenv:release]\n"
        "description = perform a release\n"
        "commands = python scripts/release.py {posargs}\n"
    )
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[1].span_id,
        span_end_id=unit.spans[2].span_id,
    )

    with pytest.raises(AdapterError, match="exactly one active assignment"):
        ground_evidence_range(
            selection,
            units_by_id={unit.unit_id: unit},
            sources_by_path={source.path: source},
            allowed_unit_ids={unit.unit_id},
        )


def test_yaml_sequence_mapping_is_atomic_and_never_inherits_sibling_args() -> None:
    content = (
        "repos:\n"
        "  - repo: local\n"
        "    hooks:\n"
        "      - id: ruff\n"
        "        args: [--fix]\n"
        "      - id: ruff-format\n"
    )
    source, unit = _configuration_source(
        content,
        path=".pre-commit-config.yaml",
    )
    selected = next(span for span in unit.spans if span.line_number == 6)
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=selected.span_id,
        span_end_id=selected.span_id,
    )

    grounded = ground_evidence_range(
        selection,
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        allowed_unit_ids={unit.unit_id},
    )

    assert grounded.evidence.semantic_key == (
        "yaml_json:sequence_item.id=ruff-format"
    )
    assert "config_key:id" in grounded.semantic_flags
    assert "--fix" not in grounded.evidence.semantic_key


def test_yaml_evidence_cannot_cross_sequence_siblings() -> None:
    content = (
        "hooks:\n"
        "  - id: ruff\n"
        "    args: [--fix]\n"
        "  - id: ruff-format\n"
    )
    source, unit = _configuration_source(
        content,
        path=".pre-commit-config.yaml",
    )
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[1].span_id,
        span_end_id=unit.spans[3].span_id,
    )

    with pytest.raises(AdapterError, match="sequence-item boundary"):
        ground_evidence_range(
            selection,
            units_by_id={unit.unit_id: unit},
            sources_by_path={source.path: source},
            allowed_unit_ids={unit.unit_id},
        )


@pytest.mark.parametrize("sequence_item", [False, True])
def test_yaml_cohesive_fields_require_assignment_ranges_or_a_sequence_root(
    sequence_item: bool,
) -> None:
    content = (
        "hooks:\n  - id: formatter\n    args: [--check]\n"
        if sequence_item else
        "version:\n  provider: mike\n  default: latest\n"
    )
    source, unit = _configuration_source(content, path="settings.yaml")

    def ground(start: int, end: int) -> GroundedEvidence:
        return ground_evidence_range(
            CandidateEvidenceRange(
                unit_id=unit.unit_id,
                span_start_id=unit.spans[start].span_id,
                span_end_id=unit.spans[end].span_id,
            ),
            units_by_id={unit.unit_id: unit},
            sources_by_path={source.path: source},
            allowed_unit_ids={unit.unit_id},
        )

    if sequence_item:
        assert ground(1, 2).evidence.semantic_key
    else:
        with pytest.raises(AdapterError, match="exactly one active assignment"):
            ground(0, 2)
        assert ground(1, 1).evidence.semantic_key
        assert ground(2, 2).evidence.semantic_key


def test_long_ordinary_prose_is_rejected_during_grounding() -> None:
    content = "".join(f"Line {index}\n" for index in range(13))
    source, unit = _prose_source(content)
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[0].span_id,
        span_end_id=unit.spans[-1].span_id,
    )

    with pytest.raises(AdapterError, match="ordinary evidence is limited to 12"):
        ground_evidence_range(
            selection,
            units_by_id={unit.unit_id: unit},
            sources_by_path={source.path: source},
            allowed_unit_ids={unit.unit_id},
        )


@pytest.mark.parametrize("path", [".editorconfig", ".flake8", ".coveragerc"])
def test_ini_style_dotfile_configuration_is_typed(path: str) -> None:
    content = "[tool]\nline-length = 99\n"
    source = SourceFile(
        path=path,
        source_kind="configuration",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="0" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_source(source, max_unit_chars=10_000)[0]
    source = SourceFile(
        **{**source.__dict__, "content_sha256": unit.source_content_sha256}
    )
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[1].span_id,
        span_end_id=unit.spans[1].span_id,
    )

    grounded = ground_evidence_range(
        selection,
        units_by_id={unit.unit_id: unit},
        sources_by_path={path: source},
        allowed_unit_ids={unit.unit_id},
    )

    assert grounded.evidence.adapter == "ini_cfg"
    assert grounded.evidence.semantic_key == "ini_cfg:tool.line-length=99"


def test_taskfile_command_uses_task_name_and_body() -> None:
    content = "tasks:\n  test:\n    cmds:\n      - uv run pytest\n"
    source = SourceFile(
        path="Taskfile.yml",
        source_kind="command",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="0" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_source(source, max_unit_chars=10_000)[0]
    source = SourceFile(
        **{**source.__dict__, "content_sha256": unit.source_content_sha256}
    )
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[1].span_id,
        span_end_id=unit.spans[3].span_id,
    )

    grounded = ground_evidence_range(
        selection,
        units_by_id={unit.unit_id: unit},
        sources_by_path={source.path: source},
        allowed_unit_ids={unit.unit_id},
    )

    assert grounded.evidence.semantic_key == "command:test:uv run pytest"
    assert "command_target:test" in grounded.semantic_flags


def test_taskfile_inventory_cannot_masquerade_as_command() -> None:
    content = "tasks:\n  test:\n    desc: Run tests\n"
    source = SourceFile(
        path="Taskfile.yml",
        source_kind="command",
        discovery_origin="deterministic",
        reason="test",
        referenced_by=(),
        content_sha256="0" * 64,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    unit = chunk_source(source, max_unit_chars=10_000)[0]
    source = SourceFile(
        **{**source.__dict__, "content_sha256": unit.source_content_sha256}
    )
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[0].span_id,
        span_end_id=unit.spans[-1].span_id,
    )

    with pytest.raises(AdapterError, match="active command body"):
        ground_evidence_range(
            selection,
            units_by_id={unit.unit_id: unit},
            sources_by_path={source.path: source},
            allowed_unit_ids={unit.unit_id},
        )


def test_ini_section_header_is_not_misgrounded_to_previous_assignment() -> None:
    source, unit = _ini_source(
        "[old]\ncommands =\n[pytest]\nminversion=8.0\n",
        unit_index=-1,
    )
    selection = CandidateEvidenceRange(
        unit_id=unit.unit_id,
        span_start_id=unit.spans[0].span_id,
        span_end_id=unit.spans[0].span_id,
    )

    with pytest.raises(AdapterError, match="no active assignment"):
        ground_evidence_range(
            selection,
            units_by_id={unit.unit_id: unit},
            sources_by_path={source.path: source},
            allowed_unit_ids={unit.unit_id},
        )


def test_workflow_trigger_span_is_typed_exact_evidence() -> None:
    grounded = _ground_whole_workflow(
        "name: checks\n"
        "on:\n"
        "  pull_request:\n"
        "    branches:\n"
        "      - main\n"
    )

    assert grounded.evidence.type == "workflow_step"
    assert "workflow_key:on" in grounded.semantic_flags
    assert "pull_request" in grounded.evidence.semantic_key


def test_workflow_multiline_run_is_typed_exact_evidence() -> None:
    grounded = _ground_whole_workflow(
        "steps:\n"
        "  - name: Check\n"
        "    run: |\n"
        "      set -euo pipefail\n"
        "      python -m pytest\n"
    )

    assert "workflow_key:run" in grounded.semantic_flags
    assert "set -euo pipefail" in grounded.evidence.semantic_key
    assert "python -m pytest" in grounded.evidence.semantic_key


def test_workflow_range_preserves_predicate_and_action_literals() -> None:
    grounded = _ground_whole_workflow(
        "if: github.event.pull_request.draft == false\n"
        "run: python -m pytest\n"
    )

    assert "workflow_key:if" in grounded.semantic_flags
    assert "workflow_key:run" in grounded.semantic_flags
    assert any(
        item.startswith("workflow_predicate:")
        for item in grounded.semantic_flags
    )
    assert "command:python -m pytest" in grounded.semantic_flags


def test_workflow_auxiliary_literal_is_typed_exact_evidence() -> None:
    grounded = _ground_whole_workflow(
        "with:\n"
        "  attestations: true\n"
    )

    assert "workflow_key:attestations" in grounded.semantic_flags
    assert "workflow_auxiliary_literal" in grounded.semantic_flags
    assert grounded.evidence.quote == "with:\n  attestations: true\n"


def test_workflow_sequence_literal_is_typed_exact_evidence() -> None:
    grounded = _ground_whole_workflow(
        "matrix:\n"
        "  recipe:\n"
        "    - conda\n"
        "    - pytest\n"
        "    - tox\n"
    )

    assert "workflow_key:recipe" in grounded.semantic_flags
    assert "workflow_sequence_literal" in grounded.semantic_flags
    assert grounded.evidence.semantic_key.endswith(":recipe:conda,pytest,tox")


def test_documented_command_bullet_is_a_typed_recipe_not_hard_directive() -> None:
    grounded = _ground_whole_prose(
        "## Development commands\n\n"
        "- `uv run pytest tests/benchmarks` - Run benchmark tests.\n"
    )

    assert grounded.hard_directive is False
    assert "documented_command_recipe" in grounded.semantic_flags


@pytest.mark.parametrize(
    "content",
    [
        "- Examples:\n  - Always use a fake package.\n",
        "* Example:\n  * Always use a fake package.\n",
        "Examples:\n- Always use a fake package.\n",
    ],
)
def test_nested_markdown_rst_example_items_get_typed_context_flag(
    content: str,
) -> None:
    grounded = _ground_prose_line(content, 1)

    assert grounded.example_context is True
    assert "example_context" in grounded.semantic_flags
    assert grounded.hard_directive is True


def test_example_parent_search_is_bounded() -> None:
    content = "Examples:\n" + "\n".join(
        f"context line {index}" for index in range(33)
    ) + "\n- Always use the real package.\n"
    grounded = _ground_prose_line(content, 34)

    assert grounded.example_context is False
    assert "example_context" not in grounded.semantic_flags


def test_nearer_non_example_list_parent_stops_example_context() -> None:
    grounded = _ground_prose_line(
        "- Examples:\n"
        "- Requirements:\n"
        "  - Always run the repository test suite.\n",
        2,
    )

    assert grounded.hard_directive is True
    assert grounded.example_context is False
    assert "example_context" not in grounded.semantic_flags


def test_same_level_generic_local_label_stops_example_context() -> None:
    grounded = _ground_prose_line(
        "Examples:\n"
        "- Never use a fake command.\n"
        "Validation commands:\n"
        "- Always run the repository test suite.\n",
        3,
    )

    assert grounded.hard_directive is True
    assert grounded.example_context is False
    assert "example_context" not in grounded.semantic_flags


def test_colon_inside_ordinary_sentence_does_not_end_example_context() -> None:
    grounded = _ground_prose_line(
        "Examples:\n"
        "This sentence contains a colon: it is not a local label.\n"
        "- Never use a fake command.\n",
        2,
    )

    assert grounded.example_context is True
    assert "example_context" in grounded.semantic_flags


def test_nested_markdown_test_heading_adds_typed_section_context() -> None:
    grounded = _ground_prose_line(
        "# Contributor Guide\n"
        "## Testing\n"
        "### Test naming\n"
        "- Each test function must start with `test_`.\n",
        3,
    )

    assert "test_section_context" in grounded.semantic_flags


def test_markdown_sibling_heading_ends_typed_test_section_context() -> None:
    grounded = _ground_prose_line(
        "# Contributor Guide\n"
        "## Testing\n"
        "### Test naming\n"
        "- Each test function must start with `test_`.\n"
        "## Error Handling\n"
        "### Assertions\n"
        "- Differentiate programmer errors (`assert`) from user errors.\n",
        6,
    )

    assert "test_section_context" not in grounded.semantic_flags


def test_nested_rst_test_heading_adds_typed_section_context() -> None:
    grounded = _ground_prose_line(
        "Contributor Guide\n"
        "=================\n"
        "Testing\n"
        "-------\n"
        "Test naming\n"
        "~~~~~~~~~~~\n"
        "Each test module must be named ``test_<module>.py``.\n",
        6,
        path="CONTRIBUTING.rst",
    )

    assert "test_section_context" in grounded.semantic_flags


def test_rst_sibling_heading_ends_typed_test_section_context() -> None:
    grounded = _ground_prose_line(
        "Contributor Guide\n"
        "=================\n"
        "Testing\n"
        "-------\n"
        "Each test must be deterministic.\n"
        "Error Handling\n"
        "--------------\n"
        "Differentiate programmer errors from user errors.\n",
        7,
        path="CONTRIBUTING.rst",
    )

    assert "test_section_context" not in grounded.semantic_flags


def test_typed_adapters_calibrate_kind_strength_and_context(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    supplement = supplement_for(
        discovery,
        "docs/team_practices.rst",
        "docs/product_behavior.md",
    )

    result = build_fixture(
        repository,
        tmp_path / "run",
        config=config,
        supplement=supplement,
    )

    claims = result.memory.claims
    marker = next(
        item
        for item in claims
        if "registers pytest marker" in item.rule
    )
    assert marker.claim_kind == "operational_fact"
    assert marker.normative_strength == "observed"
    assert marker.acceptance_basis == ["typed_repository_literal"]
    assert marker.condition is None
    assert marker.delivery_policy == "default_consider"
    assert marker.evidence[0].adapter == "toml"
    assert marker.evidence[0].type == "configuration_entry"

    ruff = next(item for item in claims if "line-length = 100" in item.rule)
    assert ruff.claim_kind == "enforced_configuration"
    assert ruff.normative_strength == "observed"
    assert ruff.acceptance_basis == ["executable_enforcement"]
    assert ruff.condition is None
    assert ruff.delivery_policy == "default_consider"

    command = next(item for item in claims if "`test` target" in item.rule)
    assert command.claim_kind == "command_recipe"
    assert command.acceptance_basis == ["typed_repository_literal"]
    assert command.evidence[0].adapter == "make_just_task"
    assert command.condition == "command target: test"
    assert command.delivery_policy == "condition_matched_only"

    workflow = next(
        item
        for item in claims
        if "workflow" in item.rule.casefold() and "make test" in item.rule
    )
    assert workflow.claim_kind == "workflow_fact"
    assert workflow.acceptance_basis == ["typed_repository_literal"]
    assert workflow.evidence[0].adapter == "workflow"
    assert workflow.condition == "workflow predicate: [push]"
    assert workflow.delivery_policy == "condition_matched_only"

    assert not any(
        evidence.path == "docs/product_behavior.md"
        for claim in claims
        for evidence in claim.evidence
    )


def test_soft_wrap_and_supplement_duplicate_merge_preserve_exact_evidence(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    result = build_fixture(
        repository,
        tmp_path / "run",
        config=config,
        supplement=supplement_for(discovery, "docs/team_practices.rst"),
    )

    merged = next(
        item
        for item in result.memory.claims
        if "public API changes documented" in item.rule
    )
    assert len(merged.evidence) == 2
    assert {item.discovery_origin for item in merged.evidence} == {
        "deterministic",
        "agent_supplement",
    }
    agent_quote = next(
        item.quote
        for item in merged.evidence
        if item.discovery_origin == "agent_supplement"
    )
    assert agent_quote == (
        "Developers must keep public API changes documented and covered by "
        "focused tests.\n"
    )
    deterministic_quote = next(
        item.quote
        for item in merged.evidence
        if item.discovery_origin == "deterministic"
    )
    assert deterministic_quote == (
        "Contributors must keep public API changes\n"
        "documented and covered by focused tests.\n"
    )


def test_pdm_style_news_soft_wrap_is_exactly_recovered(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    result = build_fixture(repository, tmp_path / "run")

    claim = next(
        item for item in result.memory.claims if "news fragment" in item.rule
    )
    assert claim.claim_kind == "normative_declaration"
    assert claim.normative_strength == "required"
    assert claim.evidence[0].quote == (
        "When you fix a bug or add a feature, you must add a news fragment "
        "describing\n"
        "your change.\n"
    )
    assert claim.execution_risk == "scoped_mutation"
