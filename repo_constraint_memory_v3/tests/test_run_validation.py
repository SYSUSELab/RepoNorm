from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from repo_constraint_memory_v3 import builder as builder_module
from repo_constraint_memory_v3.agent_runtime import (
    DeterministicFakeAgentRuntime,
    SubprocessAgentRuntime,
)
from repo_constraint_memory_v3.run_validation import (
    _inference_logical_record_bound,
    validate_run_directory,
)
from repo_constraint_memory_v3.llm import MockLLMClient
from repo_constraint_memory_v3.discovery import DiscoveryError
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.unified import validation as unified_validation

from helpers import (
    build_fixture,
    config_for_tests,
    copied_fixture,
    discover_fixture,
    supplement_for,
)


def test_inference_record_bound_covers_base_and_refined_temporal_bisection() -> None:
    config = config_for_tests()
    one_item = config.model_copy(
        update={
            "unified": config.unified.model_copy(
                update={"max_temporal_items_per_case": 1}
            )
        }
    )
    three_items = config.model_copy(
        update={
            "unified": config.unified.model_copy(
                update={"max_temporal_items_per_case": 3}
            )
        }
    )

    expected_delta = (
        config.unified.max_cases
        * (1 + config.unified.max_refinement_children)
        * 4
    )
    assert (
        _inference_logical_record_bound(three_items)
        - _inference_logical_record_bound(one_item)
        == expected_delta
    )


def test_inference_record_bound_covers_one_grouping_record_per_pair() -> None:
    config = config_for_tests()
    expanded = config.model_copy(
        update={
            "unified": config.unified.model_copy(
                update={"max_group_pairs": config.unified.max_group_pairs + 17}
            )
        }
    )

    assert (
        _inference_logical_record_bound(expanded)
        - _inference_logical_record_bound(config)
        == 17
    )


def test_complete_mock_round_trip_validates(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    assert {item.name for item in (run / "model").iterdir()} == {
        "profile.json",
        "execution-receipt.json",
    }
    assert not any(
        item.name == "model"
        for profile in (run / "consumers").iterdir()
        for item in profile.iterdir()
    )
    assert validate_run_directory(
        run,
        repository_root=repository,
    ) == []


def test_credential_shaped_target_content_round_trip_validates(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    agents_path = repository / "AGENTS.md"
    agents_path.write_text(
        agents_path.read_text(encoding="utf-8")
        + "\nControlled fixture: API_KEY = abcdefghijklmnopqrstuvwxyz\n",
        encoding="utf-8",
    )
    pyproject_path = repository / "pyproject.toml"
    pyproject_path.write_text(
        pyproject_path.read_text(encoding="utf-8")
        + '\n[tool.controlled_fixture]\napi_key = "abcdefghijklmnop"\n',
        encoding="utf-8",
    )
    credential_module = repository / "src/pkg/credential_fixture.py"
    credential_module.write_text(
        'API_KEY = "abcdefghijklmnopqrstuvwxyz"\n',
        encoding="utf-8",
    )

    result = build_fixture(repository, tmp_path / "credential-run")

    assert result.observation is not None
    shard = next(
        item
        for item in result.observation.shards
        if item.path == "src/pkg/credential_fixture.py"
    )
    assert shard.parse_status == "complete"
    assert shard.incomplete_reasons == []
    source = next(
        item
        for item in result.build_report["source_funnel"]["sources"]
        if item["path"] == "AGENTS.md"
    )
    assert source["source_plan_admitted"]
    assert source["unit_count"] > 0
    assert "deferred_sensitive" not in json.dumps(
        result.build_report,
        sort_keys=True,
    )
    assert validate_run_directory(
        tmp_path / "credential-run",
        repository_root=repository,
    ) == []


def test_execution_model_artifacts_and_config_are_closed(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "model-closure-run"
    build_fixture(repository, run)

    targets = (
        run / "model" / "profile.json",
        run / "model" / "execution-receipt.json",
    )
    for target in targets:
        original = target.read_bytes()
        target.chmod(0o600)
        target.write_bytes(original + b"\n")
        errors = validate_run_directory(run, repository_root=repository)
        assert any(
            "model" in item.lower() or "canonical" in item.lower()
            for item in errors
        ), errors
        target.write_bytes(original)

    config_path = run / "config.json"
    original_config = config_path.read_text(encoding="utf-8")
    config = json.loads(original_config)
    config["execution_model_profile_id"] = "tampered-model-profile"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    errors = validate_run_directory(run, repository_root=repository)
    assert any(
        "execution-model" in item and "differs" in item for item in errors
    ), errors

    config = json.loads(original_config)
    config["llm"]["model"] = "different-wire-model"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    errors = validate_run_directory(run, repository_root=repository)
    assert any("llm.model" in item for item in errors), errors


def test_consumer_view_rejects_execution_model_artifact_leak(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "consumer-model-leak-run"
    build_fixture(repository, run)
    leak = run / "consumers" / "accepted_only" / "model"
    leak.mkdir()
    (leak / "profile.json").write_text("{}\n", encoding="utf-8")

    errors = validate_run_directory(run, repository_root=repository)
    assert any("execution-model artifacts" in item for item in errors), errors


def test_final_closed_tree_recheck_catches_late_artifact_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "late-mutation-run"
    build_fixture(repository, run)
    real_closure = unified_validation._build_report_closure_errors
    mutated = False

    def close_report_then_mutate(*args, **kwargs):
        nonlocal mutated
        result = real_closure(*args, **kwargs)
        if not mutated:
            mutated = True
            path = run / "canonical" / "delivery_report.json"
            path.chmod(0o600)
            path.write_bytes(path.read_bytes() + b"\nRACED\n")
        return result

    monkeypatch.setattr(
        unified_validation,
        "_build_report_closure_errors",
        close_report_then_mutate,
    )
    errors = validate_run_directory(run, repository_root=repository)
    assert any("changed after snapshot" in item for item in errors)


def test_final_repository_snapshot_recheck_catches_dirty_worktree_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "worktree-race-run"
    build_fixture(repository, run)
    real_closure = unified_validation._build_report_closure_errors
    mutated = False

    def close_report_then_mutate_worktree(*args, **kwargs):
        nonlocal mutated
        result = real_closure(*args, **kwargs)
        if not mutated:
            mutated = True
            path = repository / "README.md"
            path.write_bytes(path.read_bytes() + b"\nworktree race\n")
        return result

    monkeypatch.setattr(
        unified_validation,
        "_build_report_closure_errors",
        close_report_then_mutate_worktree,
    )
    errors = validate_run_directory(run, repository_root=repository)

    assert mutated
    assert any(
        "repository changed during unified validation" in item for item in errors
    )


def test_repository_discovery_failure_is_typed_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "discovery-error-run"
    build_fixture(repository, run)

    def fail_discovery(*args, **kwargs):
        del args, kwargs
        raise DiscoveryError("repository changed into an unsupported state")

    monkeypatch.setattr(builder_module, "_verify_discovery", fail_discovery)
    errors = validate_run_directory(run, repository_root=repository)

    assert any("fixed-snapshot discovery replay failed" in item for item in errors)


def test_build_report_package_identity_and_completeness_are_closed(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "build-report-closure-run"
    build_fixture(repository, run)
    report_path = run / "build_report.json"
    original = json.loads(report_path.read_text(encoding="utf-8"))

    mutations = [
        (
            "schema version",
            lambda value: value.__setitem__("schema_version", "schema-tamper"),
        ),
        (
            "package version",
            lambda value: value.__setitem__("package_version", "0.0.0-tamper"),
        ),
        (
            "config",
            lambda value: value["config"]["repository"].__setitem__(
                "name", "tampered-repository-name"
            ),
        ),
        (
            "unified contract",
            lambda value: value["versions"].__setitem__(
                "unified_contract", "contract-tamper"
            ),
        ),
        (
            "unified policy",
            lambda value: value["versions"].__setitem__(
                "unified_policy", "policy-tamper"
            ),
        ),
        (
            "complete",
            lambda value: value.__setitem__(
                "complete", not bool(original["complete"])
            ),
        ),
        (
            "incomplete reasons",
            lambda value: value.__setitem__(
                "incomplete_reasons", ["tampered_reason"]
            ),
        ),
    ]
    for expected, mutate in mutations:
        value = json.loads(json.dumps(original))
        mutate(value)
        report_path.chmod(0o600)
        report_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        errors = validate_run_directory(run, repository_root=repository)
        assert any(expected in item for item in errors), (expected, errors)


def test_terminal_explicit_candidate_quarantine_replays_offline(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    (repository / "vercel.json").write_text(
        "{\n"
        '  "buildCommand": "pip install -r requirements-docs.txt && mkdocs build",\n'
        '  "outputDirectory": "site"\n'
        "}\n",
        encoding="utf-8",
    )
    config = config_for_tests(
        observation={"enabled": False, "required": False},
        unified={"git_enabled": False},
    )
    discovery = discover_fixture(repository, config)
    supplement = supplement_for(discovery, "vercel.json")

    class TerminalQuarantineClient:
        def __init__(self) -> None:
            self.mock = MockLLMClient()
            self.typed_invalid_calls = 0

        def complete(
            self,
            system_prompt: str,
            user_prompt: str,
        ) -> ProviderResponse:
            payload = json.loads(user_prompt)
            request = payload.get("original_request", payload)
            units = request.get("units", [])
            if units and units[0].get("path") == "vercel.json":
                self.typed_invalid_calls += 1
                unit = units[0]
                spans = unit["spans"]
                content = json.dumps(
                    {
                        "results": [
                            {
                                "unit_id": unit["unit_id"],
                                "classification": "contains_claims",
                                "reason_code": "configuration_fact",
                                "claims": [
                                    {
                                        "rule": (
                                            "Vercel builds the documentation "
                                            "and publishes the site directory."
                                        ),
                                        "claim_kind": "enforced_configuration",
                                        "normative_strength": "observed",
                                        "applies_to": ["docs/**"],
                                        "condition": None,
                                        "evidence": [
                                            {
                                                "unit_id": unit["unit_id"],
                                                "span_start_id": spans[1]["span_id"],
                                                "span_end_id": spans[2]["span_id"],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return ProviderResponse(
                    content=content,
                    model="terminal-quarantine-fake",
                    input_tokens=1,
                    output_tokens=1,
                    latency_ms=1,
                )
            return self.mock.complete(system_prompt, user_prompt)

    client = TerminalQuarantineClient()
    run = tmp_path / "run"
    build_fixture(
        repository,
        run,
        config=config,
        supplement=supplement,
        client=client,
    )

    assert client.typed_invalid_calls == config.llm.schema_repairs + 1
    upstream_text = (
        run / "canonical" / "audit" / "upstream-inputs.json"
    ).read_text(encoding="utf-8")
    assert "exactly one active assignment" in upstream_text
    assert validate_run_directory(run, repository_root=repository) == []


def test_validation_rejects_inference_calls_above_configured_hard_limit(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    config_path = run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inference_index = json.loads(
        (run / "inference" / "index.json").read_text(encoding="utf-8")
    )
    assert inference_index["total_calls"] > 0
    config["llm"]["max_total_calls"] = inference_index["total_calls"] - 1
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    errors = validate_run_directory(run, repository_root=repository)
    assert "inference index calls exceed configured hard limit" in errors


def test_explicit_relation_replay_uses_source_plan_not_request_id_order(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    for directory in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot"):
        root = repository / "src" / directory
        root.mkdir(parents=True)
        (root / "AGENTS.md").write_text(
            "Contributors must run pytest.\n",
            encoding="utf-8",
        )
        (root / "module.py").write_text(
            f'"""{directory} fixture."""\n',
            encoding="utf-8",
        )

    config = config_for_tests(
        observation={"enabled": False, "required": False},
        unified={"git_enabled": False},
    )
    run = tmp_path / "run"
    build_fixture(repository, run, config=config)

    upstream = json.loads(
        (
            run / "canonical" / "audit" / "upstream-inputs.json"
        ).read_text(encoding="utf-8")
    )
    source_order = [
        item["unit_id"] for item in upstream["explicit_decisions"]
    ]
    request_order: list[str] = []
    inference_index = json.loads(
        (run / "inference" / "index.json").read_text(encoding="utf-8")
    )
    for item in sorted(
        inference_index["requests"], key=lambda value: value["request_id"]
    ):
        record = json.loads(
            (
                run
                / "inference"
                / "requests"
                / f'{item["request_id"]}.json'
            ).read_text(encoding="utf-8")
        )
        if record["stage"] != "explicit_extract":
            continue
        request_order.extend(
            ref["stable_id"]
            for ref in record["input_manifest"]
            if ref["kind"] == "explicit_unit"
        )

    assert len(source_order) == 6
    assert request_order != source_order
    assert all(
        len(item["relations"]) == 5
        for item in upstream["explicit_decisions"]
    )
    assert validate_run_directory(run, repository_root=repository) == []


@pytest.mark.parametrize(
    "relative",
    [
        "canonical/claims.json",
        "consumers/accepted_only/consumer-manifest.json",
        "explicit_verification.json",
        "consumers/accepted_only/by-path/constraints.md",
        "observations/index.json",
        "observations/queries/python_snapshot.json",
        "observations/queries/python_clone_clusters.json",
    ],
)
def test_tampered_canonical_artifacts_are_detected(
    tmp_path: Path,
    relative: str,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    path = run / relative
    if path.suffix == ".json":
        path.chmod(0o644)
        value = json.loads(path.read_text(encoding="utf-8"))
        value["tampered"] = True
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
    else:
        path.chmod(0o644)
        path.write_text(
            path.read_text(encoding="utf-8") + "\nTAMPERED\n",
            encoding="utf-8",
        )
    assert validate_run_directory(run, repository_root=repository)


def test_source_mutation_breaks_snapshot_span_and_quote_replay(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    agents = repository / "AGENTS.md"
    agents.write_text(
        agents.read_text(encoding="utf-8") + "\nChanged after build.\n",
        encoding="utf-8",
    )

    errors = validate_run_directory(run, repository_root=repository)
    assert any("snapshot" in item or "hash" in item for item in errors)


def test_validator_never_scans_repository_root_supplied_by_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    discovery_path = run / "discovery.json"
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    discovery["repository"]["root"] = "/"
    discovery_path.write_text(
        json.dumps(discovery, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    def forbidden_discovery(*args, **kwargs):
        del args, kwargs
        raise AssertionError("validator scanned an artifact-selected repository")

    monkeypatch.setattr(
        "repo_constraint_memory_v3.builder.discover_repository",
        forbidden_discovery,
    )
    errors = validate_run_directory(run, repository_root=repository)
    assert any("source acquisition replay failed" in item for item in errors)


def test_incomplete_run_passes_integrity_validation_but_not_completeness_gate(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path, incomplete=True)
    run = tmp_path / "run"
    result = build_fixture(repository, run)
    assert result.memory.claims
    assert validate_run_directory(run, repository_root=repository) == []
    errors = validate_run_directory(
        run,
        repository_root=repository,
        require_complete=True,
    )
    assert "complete package required but unified package is partial" in errors
    start = (
        run / "consumers" / "accepted_only" / "START.md"
    ).read_text(encoding="utf-8")
    assert "Package status" not in start
    assert "This collection may omit repository conventions." in start
    manifest = json.loads(
        (run / "consumers" / "accepted_only" / "consumer-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["package_status"] == "partial"


def test_partial_run_still_fails_closed_on_delivery_view_tamper(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path, incomplete=True)
    run = tmp_path / "run"
    build_fixture(repository, run)
    view = run / "consumers/accepted_only/by-path/constraints.md"
    view.chmod(0o644)
    view.write_text(
        view.read_text(encoding="utf-8") + "\nTAMPERED\n",
        encoding="utf-8",
    )

    errors = validate_run_directory(run, repository_root=repository)

    assert any("by-path" in error for error in errors)


def test_snapshot_and_explicit_evidence_hash_tamper_are_detected(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)

    snapshot_run = tmp_path / "snapshot"
    build_fixture(repository, snapshot_run)
    discovery_path = snapshot_run / "discovery.json"
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    discovery["repository"]["snapshot_digest"] = "sha256:" + "0" * 64
    discovery_path.write_text(
        json.dumps(discovery, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    assert validate_run_directory(snapshot_run, repository_root=repository)

    evidence_run = tmp_path / "evidence"
    build_fixture(repository, evidence_run)
    memory_path = evidence_run / "canonical/claims.json"
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    assert memory["claims"]
    memory["claims"][0]["support"][0]["artifact_ref"]["artifact_digest"] = (
        "0" * 64
    )
    memory_path.write_text(
        json.dumps(memory, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    errors = validate_run_directory(evidence_run, repository_root=repository)
    assert any("canonical" in item and "claim" in item for item in errors)


def test_symlink_and_unexpected_top_level_artifacts_are_rejected(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)

    symlink_run = tmp_path / "symlink"
    build_fixture(repository, symlink_run)
    memory_path = symlink_run / "canonical/claims.json"
    memory_path.unlink()
    memory_path.symlink_to("completeness.json")
    assert any(
        "symlink" in error
        for error in validate_run_directory(
            symlink_run,
            repository_root=repository,
        )
    )

    extra_run = tmp_path / "extra"
    build_fixture(repository, extra_run)
    (extra_run / "unexpected.json").write_text("{}\n", encoding="utf-8")
    assert any(
        "unexpected top-level" in error
        for error in validate_run_directory(
            extra_run,
            repository_root=repository,
        )
    )


def test_validation_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "fifo"
    build_fixture(repository, run)
    memory_path = run / "canonical/claims.json"
    memory_path.unlink()
    os.mkfifo(memory_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "repo_constraint_memory_v3.cli",
            "validate",
            run.as_posix(),
            "--repository",
            repository.as_posix(),
        ],
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 2
    assert b"regular file" in completed.stderr


def test_validation_never_calls_a_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("validation attempted a provider call")

    monkeypatch.setattr(
        "repo_constraint_memory_v3.llm.OpenAICompatibleClient.complete",
        fail_if_called,
    )
    monkeypatch.setattr(
        "repo_constraint_memory_v3.llm.OpenAICompatibleClient.complete_with_settings",
        fail_if_called,
    )
    monkeypatch.setattr(SubprocessAgentRuntime, "run", fail_if_called)
    monkeypatch.setattr(DeterministicFakeAgentRuntime, "run", fail_if_called)
    assert validate_run_directory(run, repository_root=repository) == []


@pytest.mark.parametrize("field", ["protocol_manifest", "agent_profile"])
def test_build_report_agent_protocol_versions_are_replayed(
    tmp_path: Path,
    field: str,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    report_path = run / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["versions"][field] = "0" * 64
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    errors = validate_run_directory(run, repository_root=repository)
    assert any(field.replace("_", " ") in error for error in errors)


def test_build_report_source_funnel_is_replayed(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    report_path = run / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_funnel"] = {"tampered": True}
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    errors = validate_run_directory(run, repository_root=repository)
    assert "build_report source funnel differs from deterministic replay" in errors


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("thinking_mode", "enabled"),
        ("output_limit_retries", 0),
    ],
)
def test_offline_validation_rejects_inference_policy_tamper(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    config_path = run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["llm"][field] != value
    config["llm"][field] = value
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    errors = validate_run_directory(run, repository_root=repository)
    assert errors
    assert any(
        "protocol-manifest" in item
        or "model bounds" in item
        or "retry bound" in item
        or "config.json" in item
        for item in errors
    )


def test_empty_source_funnel_is_still_replayed(tmp_path: Path) -> None:
    repository = tmp_path / "empty-explicit-repository"
    repository.mkdir()
    (repository / "x.py").write_text("VALUE = 1\n", encoding="utf-8")
    run = tmp_path / "empty-explicit-run"
    config = config_for_tests(observation={"enabled": False})
    build_fixture(repository, run, config=config)
    assert validate_run_directory(run, repository_root=repository) == []

    report_path = run / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_funnel"] = {"tampered": True}
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    errors = validate_run_directory(run, repository_root=repository)
    assert "build_report source funnel differs from deterministic replay" in errors


def test_deep_raw_json_is_localized_instead_of_crashing(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "run"
    build_fixture(repository, run)
    (run / "build_report.json").write_text(
        "[" * 10_000 + "0" + "]" * 10_000,
        encoding="utf-8",
    )

    errors = validate_run_directory(run, repository_root=repository)
    assert any("build_report.json parse error" in error for error in errors)
