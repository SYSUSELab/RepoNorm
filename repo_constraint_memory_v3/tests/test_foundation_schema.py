from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3 import __version__
from repo_constraint_memory_v3.builder import load_build_config
from repo_constraint_memory_v3.models import (
    BuildConfig,
    DiscoveryArtifact,
    MemoryFile,
    SCHEMA_VERSION,
    UnifiedCapabilityCoverage,
)
from repo_constraint_memory_v3.run_validation import validate_run_directory

from helpers import (
    CONFIG_PATH,
    build_fixture,
    config_for_tests,
    copied_fixture,
)


def _rewrite_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def test_version_schema_and_strict_unified_coverage() -> None:
    assert __version__ == "0.18.0"
    assert SCHEMA_VERSION == "4.11-partition-only-required-evidence"
    complete = UnifiedCapabilityCoverage(
        status="complete", required=True, planned=1, completed=1
    )
    assert complete.status == "complete"
    with pytest.raises(ValidationError):
        UnifiedCapabilityCoverage.model_validate(
            {
                "status": "complete",
                "required": True,
                "planned": 1,
                "completed": 0,
                "reasons": ["missing_case"],
            }
        )
    with pytest.raises(ValidationError):
        UnifiedCapabilityCoverage.model_validate(
            {
                "status": "not_run",
                "required": True,
                "planned": 1,
                "completed": 0,
                "reasons": ["not_run"],
            }
        )


def test_target_content_admission_policy_has_no_hidden_toggle() -> None:
    payload = load_build_config(CONFIG_PATH).model_dump(mode="json")
    assert payload["discovery"]["target_content_admission_policy"] == (
        "controlled-research-unredacted-target-content-1.0"
    )
    payload["discovery"]["target_content_admission_policy"] = (
        "credential-quarantine"
    )

    with pytest.raises(ValidationError):
        BuildConfig.model_validate(payload)


@pytest.mark.parametrize("forbidden_key", ["implicit", "history"])
def test_old_and_future_config_sections_are_rejected(
    forbidden_key: str,
) -> None:
    payload = load_build_config(CONFIG_PATH).model_dump(mode="json")
    payload[forbidden_key] = {}
    with pytest.raises(ValidationError):
        BuildConfig.model_validate(payload)


def test_old_schema_and_deleted_accepted_claim_fields_are_rejected(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    result = build_fixture(repository, tmp_path / "run")

    discovery = json.loads(
        (tmp_path / "run" / "discovery.json").read_text(encoding="utf-8")
    )
    discovery["schema_version"] = "3.6-current-source-open-semantic-mining"
    with pytest.raises(ValidationError):
        DiscoveryArtifact.model_validate(discovery)

    memory = result.memory.model_dump(mode="json")
    assert memory["claims"]
    claim = memory["claims"][0]
    claim["origin_channels"] = ["implicit_current"]
    claim["evidence"] = [
        {
            "evidence_kind": "static_pattern",
            "family": "removed",
        }
    ]
    claim["adjudication"]["qualification_registry_digest"] = "0" * 64
    with pytest.raises(ValidationError):
        MemoryFile.model_validate(memory)


def test_pre_40_config_shape_is_rejected() -> None:
    payload = load_build_config(CONFIG_PATH).model_dump(mode="json")
    payload.pop("semantic")
    payload.pop("unified")
    for key in (
        "max_total_calls",
        "max_total_input_tokens",
        "max_total_output_tokens",
        "max_inference_artifact_bytes",
    ):
        payload["llm"].pop(key)
    with pytest.raises(ValidationError, match="schema 4.11"):
        BuildConfig.model_validate(payload)


def test_unified_profile_resource_contracts_are_strict() -> None:
    base = load_build_config(CONFIG_PATH).model_dump(mode="json")

    missing = json.loads(json.dumps(base))
    missing.pop("unified")
    with pytest.raises(ValidationError, match="schema 4.11"):
        BuildConfig.model_validate(missing)

    oversized_case = json.loads(json.dumps(base))
    oversized_case["unified"]["max_case_members"] = 33
    with pytest.raises(ValidationError, match="max_case_members"):
        BuildConfig.model_validate(oversized_case)

    insufficient_temporal_cells = json.loads(json.dumps(base))
    insufficient_temporal_cells["unified"]["max_temporal_cells_per_request"] = 1
    with pytest.raises(ValidationError, match="max_temporal_cells_per_request"):
        BuildConfig.model_validate(insufficient_temporal_cells)

    undersized_singleton_case_bound = json.loads(json.dumps(base))
    undersized_singleton_case_bound["unified"]["max_cases"] = (
        undersized_singleton_case_bound["unified"]["max_candidate_members"] - 1
    )
    with pytest.raises(ValidationError, match="singleton mode requires"):
        BuildConfig.model_validate(undersized_singleton_case_bound)

    insufficient_git_commands = json.loads(json.dumps(base))
    insufficient_git_commands["unified"]["git_max_commands_per_case"] = 3
    with pytest.raises(ValidationError, match="family minimums"):
        BuildConfig.model_validate(insufficient_git_commands)

    oversized_git_family = json.loads(json.dumps(base))
    oversized_git_family["unified"]["git_max_results_per_query"] = 2_000
    oversized_git_family["unified"]["git_max_queries_per_family"] = 64
    with pytest.raises(ValidationError, match="aggregate result bound"):
        BuildConfig.model_validate(oversized_git_family)

    insufficient_case_bytes = json.loads(json.dumps(base))
    insufficient_case_bytes["unified"]["git_max_bytes_per_case"] = 1_000_000
    with pytest.raises(ValidationError, match="per-command"):
        BuildConfig.model_validate(insufficient_case_bytes)

    oversized_refinement_split = json.loads(json.dumps(base))
    oversized_refinement_split["unified"]["max_refinement_children"] = 3
    with pytest.raises(ValidationError, match="max_refinement_children"):
        BuildConfig.model_validate(oversized_refinement_split)


def test_semantic_config_dependency_and_resource_contracts_are_strict() -> None:
    base = load_build_config(CONFIG_PATH).model_dump(mode="json")
    assert base["semantic"]["max_hypotheses_per_component"] == 64
    assert "max_reduce_merge_input_hypotheses" not in base["semantic"]

    agent_disabled = json.loads(json.dumps(base))
    agent_disabled["agent"]["required"] = False
    agent_disabled["agent"]["stages"]["source_discovery"]["enabled"] = False
    agent_disabled["agent"]["stages"]["source_discovery"]["required"] = False
    with pytest.raises(ValidationError):
        BuildConfig.model_validate(agent_disabled)

    disabled_required = json.loads(json.dumps(base))
    disabled_required["semantic"]["enabled"] = False
    with pytest.raises(ValidationError):
        BuildConfig.model_validate(disabled_required)

    undersized_packet = json.loads(json.dumps(base))
    undersized_packet["semantic"]["max_packet_bytes"] = (
        undersized_packet["semantic"]["max_unit_source_bytes"] - 1
    )
    with pytest.raises(ValidationError):
        BuildConfig.model_validate(undersized_packet)

    observations_disabled = json.loads(json.dumps(base))
    observations_disabled["observation"]["enabled"] = False
    observations_disabled["observation"]["queries"] = []
    with pytest.raises(ValidationError):
        BuildConfig.model_validate(observations_disabled)

    missing_clone_query = json.loads(json.dumps(base))
    missing_clone_query["observation"]["queries"] = [
        {"name": "python_snapshot", "required": True}
    ]
    with pytest.raises(ValidationError):
        BuildConfig.model_validate(missing_clone_query)

    invalid_numeric_limit = json.loads(json.dumps(base))
    invalid_numeric_limit["semantic"]["max_units"] = 0
    with pytest.raises(ValidationError):
        BuildConfig.model_validate(invalid_numeric_limit)

    removed_merge_limit = json.loads(json.dumps(base))
    removed_merge_limit["semantic"][
        "max_reduce_merge_input_hypotheses"
    ] = 128
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BuildConfig.model_validate(removed_merge_limit)

    oversized_hypothesis_limit = json.loads(json.dumps(base))
    oversized_hypothesis_limit["semantic"][
        "max_hypotheses_per_component"
    ] = 65
    with pytest.raises(ValidationError):
        BuildConfig.model_validate(oversized_hypothesis_limit)

    three_repairs = json.loads(json.dumps(base))
    three_repairs["llm"]["schema_repairs"] = 3
    assert BuildConfig.model_validate(three_repairs).llm.schema_repairs == 3

    too_many_repairs = json.loads(json.dumps(base))
    too_many_repairs["llm"]["schema_repairs"] = 4
    with pytest.raises(ValidationError):
        BuildConfig.model_validate(too_many_repairs)

    bounded_output_retry = json.loads(json.dumps(base))
    bounded_output_retry["llm"]["output_limit_retries"] = 2
    assert (
        BuildConfig.model_validate(bounded_output_retry).llm.output_limit_retries
        == 2
    )

    too_many_output_retries = json.loads(json.dumps(base))
    too_many_output_retries["llm"]["output_limit_retries"] = 3
    with pytest.raises(ValidationError):
        BuildConfig.model_validate(too_many_output_retries)

    invalid_thinking_mode = json.loads(json.dumps(base))
    invalid_thinking_mode["llm"]["thinking_mode"] = "reasoning.enabled=false"
    with pytest.raises(ValidationError):
        BuildConfig.model_validate(invalid_thinking_mode)


def test_explicit_only_build_runs_unified_and_emits_no_observation_tree(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    before = _tree_bytes(repository)
    config = config_for_tests(
        observation={"enabled": False, "queries": []},
    )
    run = tmp_path / "run"
    result = build_fixture(repository, run, config=config)

    # Related explicit variants alone no longer force a Git route.  With no
    # genuinely time-sensitive, uncertain, or mixed-polarity case, this
    # explicit-only fixture remains complete even though it is not a Git repo.
    assert result.build_report["complete"]
    assert result.memory.coverage.observation.mode == "not_run"
    assert (
        result.memory.coverage.current_source.semantic_units.status
        == "not_run"
    )
    assert result.memory.coverage.unified.candidate_grouping.status == "complete"
    assert (
        result.memory.coverage.unified.git_temporal_evidence.status
        == "not_required"
    )
    assert any(
        item.epistemic_status == "accepted"
        for item in result.unified.bundle.canonical_claims
    )
    assert not (run / "observations").exists()
    assert validate_run_directory(run, repository_root=repository) == []
    assert _tree_bytes(repository) == before


@pytest.mark.parametrize(
    "forbidden_directory",
    ["implicit", "current_source", "history", "adjudication", "by-trigger"],
)
def test_forbidden_future_or_old_directory_is_rejected(
    tmp_path: Path,
    forbidden_directory: str,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    injected = run / forbidden_directory
    injected.mkdir()
    (injected / "index.json").write_text("{}\n", encoding="utf-8")
    errors = validate_run_directory(run, repository_root=repository)
    assert errors
    assert any(
        "top-level" in error or "by-trigger must not exist" in error
        for error in errors
    )


@pytest.mark.parametrize(
    "query_name",
    ["python_companion_relations", "pytest_marker_population"],
)
def test_deleted_query_artifact_is_rejected(
    tmp_path: Path,
    query_name: str,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    (run / "observations" / "queries" / f"{query_name}.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    errors = validate_run_directory(run, repository_root=repository)
    assert any("unexpected/orphan" in error for error in errors)


def test_unified_coverage_tamper_cannot_claim_completion(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    completeness_path = run / "canonical" / "completeness.json"
    completeness = json.loads(completeness_path.read_text(encoding="utf-8"))
    capability = next(
        item
        for item in completeness["state"]["capabilities"]
        if item["capability"] == "git_temporal_evidence"
    )
    assert capability["status"] == "not_required"
    capability["status"] = "complete"
    _rewrite_json(completeness_path, completeness)
    errors = validate_run_directory(run, repository_root=repository)
    assert any(
        "completeness" in error or "reconstructed unified bundle" in error
        for error in errors
    )
