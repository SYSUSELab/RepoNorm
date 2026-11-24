from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import repo_constraint_memory_v3.builder as builder_module
from helpers import build_fixture, config_for_tests, copied_fixture
from repo_constraint_memory_v3.agent_runtime import stable_model_digest
from repo_constraint_memory_v3.builder import (
    load_build_config,
    protocol_manifest_for_config,
)
from repo_constraint_memory_v3.chunking import CHUNKER_VERSION
from repo_constraint_memory_v3.inference import (
    ProtocolManifest,
    endpoint_profile_digest,
)
from repo_constraint_memory_v3.run_validation import validate_run_directory
from repo_constraint_memory_v3.unified.current_evidence import (
    CURRENT_EVIDENCE_BUNDLE_VERSION,
)
from repo_constraint_memory_v3.unified.grouping import GROUPING_POLICY_VERSION
from repo_constraint_memory_v3.unified.git_evidence import GIT_EVIDENCE_VERSION
from repo_constraint_memory_v3.unified.models import (
    UNIFIED_CONTRACT_VERSION,
    UNIFIED_POLICY_VERSION,
)
from repo_constraint_memory_v3.unified.pipeline import UNIFIED_GIT_ROUTING_VERSION


def _rewrite(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_protocol_manifest_binds_current_source_stage_versions() -> None:
    config = config_for_tests()
    assert config.semantic.max_reduce_items_per_request == 16
    assert config.semantic.max_hypotheses_per_component == 64
    assert "max_reduce_merge_input_hypotheses" not in config.semantic.model_dump()
    assert config.semantic.target_population_members_per_hypothesis == 15
    assert (
        config.semantic.max_classified_population_members_per_hypothesis
        == 512
    )
    assert config.semantic.max_total_classified_population_members == 4_096
    assert config.semantic.max_classification_calls_per_hypothesis == 32
    assert (
        config.semantic.max_classification_request_bytes_per_hypothesis
        == 16_000_000
    )
    assert config.semantic.max_total_classification_calls == 256
    assert (
        config.semantic.max_total_classification_request_bytes == 64_000_000
    )
    assert config.semantic.max_falsification_calls_per_hypothesis == 16
    assert (
        config.semantic.max_falsification_request_bytes_per_hypothesis
        == 8_000_000
    )
    assert config.semantic.max_total_falsification_calls == 128
    assert (
        config.semantic.max_total_falsification_request_bytes == 32_000_000
    )
    assert config.unified.max_temporal_items_per_case == 64
    manifest = protocol_manifest_for_config(
        config,
        agent_profile_digest="0" * 64,
    )

    assert manifest.package_version == "0.18.0"
    assert manifest.target_content_admission_policy == (
        "controlled-research-unredacted-target-content-1.0"
    )
    assert manifest.agent_stage_protocol_versions[
        "source_discovery_prompt"
    ] == "source-discovery-agent-1.4"
    assert manifest.agent_stage_protocol_versions[
        "runner_protocol"
    ] == "norm4code-agent-runner-1.2"
    assert manifest.resource_policy_version == "bounded-inference-resource-1.12"
    assert manifest.parser_contract_version == "stdlib-ast-structural-1.2"
    assert manifest.observation_contract_versions == {
        "python_clone_clusters": "1.1",
        "python_snapshot": "1.4",
    }
    assert manifest.retry_repair_policy_version == (
        "schema-repair-provider-output-limit-retry-1.4"
    )
    assert manifest.explicit_prompt_version == (
        "explicit-coding-knowledge-extraction-3.20"
    )
    assert manifest.stage_protocol_versions[
        "explicit_adjudication"
    ] == "explicit-host-adjudication-1.12"
    assert manifest.stage_protocol_versions[
        "explicit_semantic_verify"
    ] == "explicit-semantic-verification-1.15"
    assert manifest.stage_protocol_versions[
        "current_semantic_map"
    ] == "current-semantic-map-1.5"
    assert manifest.stage_protocol_versions[
        "current_semantic_map_response"
    ] == "current-semantic-map-response-1.2"
    assert manifest.stage_protocol_versions[
        "current_hypothesis_reduce"
    ] == "current-hypothesis-reduce-1.31"
    assert manifest.stage_protocol_versions[
        "current_hypothesis_reduce_response"
    ] == "current-hypothesis-reduce-response-1.10"
    assert manifest.stage_protocol_versions[
        "current_hypothesis_reduction_index"
    ] == "current-hypothesis-reduction-index-1.13"
    assert "max_reduce_merge_input_hypotheses" not in manifest.resource_limits
    assert manifest.resource_limits["max_reduce_selection_candidates"] == 128
    assert manifest.resource_limits["max_reduce_selection_prompt_bytes"] == (
        131_072
    )
    assert manifest.resource_limits["max_reduce_items_per_request"] == 16
    assert manifest.resource_limits["max_hypotheses_per_component"] == 64
    assert manifest.resource_limits[
        "max_classified_population_members_per_hypothesis"
    ] == 512
    assert manifest.resource_limits[
        "max_total_classified_population_members"
    ] == 4_096
    assert manifest.resource_limits[
        "target_population_members_per_hypothesis"
    ] == 15
    assert manifest.resource_limits[
        "max_classification_calls_per_hypothesis"
    ] == 32
    assert manifest.resource_limits[
        "max_classification_request_bytes_per_hypothesis"
    ] == 16_000_000
    assert manifest.resource_limits[
        "max_total_classification_calls"
    ] == 256
    assert manifest.resource_limits[
        "max_total_classification_request_bytes"
    ] == 64_000_000
    assert manifest.resource_limits[
        "max_falsification_calls_per_hypothesis"
    ] == 16
    assert manifest.resource_limits[
        "max_falsification_request_bytes_per_hypothesis"
    ] == 8_000_000
    assert manifest.resource_limits[
        "max_total_falsification_calls"
    ] == 128
    assert manifest.resource_limits[
        "max_total_falsification_request_bytes"
    ] == 32_000_000
    assert manifest.stage_protocol_versions[
        "population_semantic_classify"
    ] == "semantic-member-classification-2.3"
    assert manifest.stage_protocol_versions[
        "population_semantic_classify_response"
    ] == "semantic-member-classification-response-1.5"
    assert manifest.stage_protocol_versions[
        "classification_plan"
    ] == "population-classification-plan-1.8"
    assert manifest.stage_protocol_versions[
        "population_sampling_plan"
    ] == "population-sampling-plan-1.3"
    assert manifest.stage_protocol_versions[
        "counterexample_verify"
    ] == "counterexample-falsification-1.11"
    assert manifest.stage_protocol_versions[
        "counterexample_verify_response"
    ] == "counterexample-falsification-response-1.5"
    assert manifest.stage_protocol_versions[
        "falsification_result"
    ] == "current-falsification-1.5"
    assert manifest.stage_protocol_versions[
        "population_member_packet"
    ] == "population-member-packet-bounded-source-1.5"
    assert manifest.stage_protocol_versions[
        "current_population"
    ] == "current-population-1.5"
    assert manifest.stage_protocol_versions[
        "current_source_index"
    ] == "current-source-index-1.10"
    assert manifest.stage_protocol_versions[
        "traceability"
    ] == "candidate-traceability-1.3"
    assert manifest.stage_protocol_versions[
        "unified_channel_adapter"
    ] == "unified-channel-assessment-adapter-1.5"
    assert manifest.stage_protocol_versions[
        "unified_evidence_contract"
    ] == UNIFIED_CONTRACT_VERSION
    assert manifest.stage_protocol_versions[
        "candidate_semantic_group"
    ] == "candidate-semantic-group-1.6"
    assert manifest.stage_protocol_versions[
        "candidate_semantic_group_response"
    ] == "candidate-semantic-group-response-1.2"
    assert manifest.stage_protocol_versions[
        "temporal_evidence_classify"
    ] == "temporal-evidence-classifier-1.2"
    assert manifest.stage_protocol_versions[
        "temporal_evidence_classify_response"
    ] == "temporal-evidence-classifier-response-1.0"
    assert manifest.stage_protocol_versions[
        "unified_git_routing"
    ] == UNIFIED_GIT_ROUTING_VERSION
    assert UNIFIED_GIT_ROUTING_VERSION == "unified-git-routing-1.4"
    assert manifest.stage_protocol_versions[
        "git_query_manifest"
    ] == GIT_EVIDENCE_VERSION
    assert manifest.stage_protocol_versions[
        "unified_case_verify"
    ] == "unified-case-verifier-2.3"
    assert manifest.stage_protocol_versions[
        "unified_case_verify_response"
    ] == "unified-case-verifier-response-1.4"
    assert manifest.stage_protocol_versions[
        "verifier_current_evidence_view"
    ] == "verifier-current-evidence-view-1.1"
    assert manifest.stage_protocol_versions[
        "unified_upstream_inputs"
    ] == "unified-upstream-inputs-1.1"
    assert manifest.stage_protocol_versions[
        "current_evidence_bundle"
    ] == CURRENT_EVIDENCE_BUNDLE_VERSION
    assert manifest.stage_protocol_versions[
        "unified_canonical_policy"
    ] == UNIFIED_POLICY_VERSION
    assert UNIFIED_POLICY_VERSION == "unified-canonical-policy-1.13"
    assert manifest.stage_protocol_versions[
        "structured_response_normalization"
    ] == "structured-response-normalization-1.0"
    assert "current_hypothesis_merge_child" not in manifest.stage_protocol_versions
    assert manifest.stage_protocol_versions["population_planning_context"] == (
        "population-planning-context-1.3"
    )
    assert manifest.stage_protocol_versions["classification_transport_plan"] == (
        "classification-transport-plan-1.0"
    )
    assert manifest.stage_protocol_versions["git_need_assessment"] == "git-need-assessment-1.0"
    assert manifest.stage_protocol_versions["git_raw_execution"] == "git-raw-execution-1.0"
    assert manifest.stage_protocol_versions["unified_refinement"] == "unified-refinement-round-1.1"
    assert "unified_export_format" not in manifest.stage_protocol_versions
    assert "unified_exporter" not in manifest.stage_protocol_versions
    assert "unified_validator" not in manifest.stage_protocol_versions
    assert manifest.resource_limits["max_temporal_cells_per_request"] == (
        config_for_tests().unified.max_temporal_cells_per_request
    )
    assert manifest.resource_limits["max_temporal_items_per_case"] == 64
    assert manifest.resource_limits["git_max_commands_per_case"] == 16
    assert manifest.resource_limits["git_max_queries_per_family"] == 4
    assert manifest.resource_limits["git_max_results_per_query"] == 24
    assert manifest.resource_limits["git_max_patches_per_case"] == 32
    assert manifest.resource_limits["git_max_bytes_per_command"] == 2_000_000
    assert manifest.resource_limits["git_max_object_scan_bytes"] == 64_000_000
    assert manifest.resource_limits["git_max_bytes_per_case"] == 8_000_000
    assert manifest.resource_limits["git_max_changed_paths_per_commit"] == 128
    assert manifest.stage_protocol_versions[
        "candidate_grouping"
    ] == GROUPING_POLICY_VERSION
    assert manifest.stage_protocol_versions[
        "current_evidence_bundle"
    ] == CURRENT_EVIDENCE_BUNDLE_VERSION
    assert manifest.stage_protocol_versions[
        "unified_canonical_policy"
    ] == UNIFIED_POLICY_VERSION


def test_protocol_manifest_digest_binds_response_and_normalization_versions(
    monkeypatch,
) -> None:
    config = config_for_tests()
    baseline = protocol_manifest_for_config(
        config,
        agent_profile_digest="0" * 64,
    )

    monkeypatch.setattr(
        builder_module,
        "MAP_RESPONSE_SCHEMA_VERSION",
        "current-semantic-map-response-test",
    )
    changed = protocol_manifest_for_config(
        config,
        agent_profile_digest="0" * 64,
    )

    assert changed.stage_protocol_versions[
        "current_semantic_map_response"
    ] == "current-semantic-map-response-test"
    assert changed.protocol_manifest_digest != baseline.protocol_manifest_digest


def test_protocol_manifest_binds_endpoint_without_persisting_secrets() -> None:
    config = config_for_tests()
    manifest = protocol_manifest_for_config(
        config,
        agent_profile_digest="0" * 64,
    )
    repeated = protocol_manifest_for_config(
        config,
        agent_profile_digest="0" * 64,
    )

    assert manifest.manifest_version == "1.11"
    assert manifest == repeated
    assert manifest.provider == config.llm.provider
    assert manifest.repository_settings_digest == stable_model_digest(
        config.repository
    )
    assert manifest.discovery_settings_digest == stable_model_digest(
        config.discovery
    )
    assert manifest.chunking_settings_digest == stable_model_digest(
        config.chunking
    )
    assert manifest.observation_settings_digest == stable_model_digest(
        config.observation
    )
    assert manifest.stage_protocol_versions["explicit_chunking"] == (
        CHUNKER_VERSION
    )
    assert manifest.resource_limits["max_concurrency"] == (
        config.llm.max_concurrency
    )
    assert manifest.endpoint_profile_digest == endpoint_profile_digest(
        provider=config.llm.provider,
        base_url=config.llm.base_url,
    )
    serialized = json.dumps(manifest.model_dump(mode="json"), sort_keys=True)
    assert config.llm.base_url not in serialized
    assert config.llm.api_key_env not in serialized

    other_endpoint = config.model_copy(
        update={
            "llm": config.llm.model_copy(
                update={"base_url": "https://other-endpoint.invalid/v1"}
            )
        }
    )
    assert protocol_manifest_for_config(
        other_endpoint,
        agent_profile_digest="0" * 64,
    ).protocol_manifest_digest != manifest.protocol_manifest_digest
    assert endpoint_profile_digest(
        provider="another-provider",
        base_url=config.llm.base_url,
    ) != manifest.endpoint_profile_digest

    renamed_secret = config.model_copy(
        update={
            "llm": config.llm.model_copy(
                update={"api_key_env": "ROTATED_API_KEY_ENV"}
            )
        }
    )
    assert protocol_manifest_for_config(
        renamed_secret,
        agent_profile_digest="0" * 64,
    ) == manifest

    diagnostics_only = config.model_copy(
        update={
            "output": config.output.model_copy(
                update={"save_diagnostics": not config.output.save_diagnostics}
            )
        }
    )
    assert protocol_manifest_for_config(
        diagnostics_only,
        agent_profile_digest="0" * 64,
    ) == manifest

    method_variants = [
        config.model_copy(
            update={
                "repository": config.repository.model_copy(
                    update={"name": "another-repository-profile"}
                )
            }
        ),
        config.model_copy(
            update={
                "discovery": config.discovery.model_copy(
                    update={
                        "max_manifest_files": (
                            config.discovery.max_manifest_files + 1
                        )
                    }
                )
            }
        ),
        config.model_copy(
            update={
                "chunking": config.chunking.model_copy(
                    update={"max_units": config.chunking.max_units + 1}
                )
            }
        ),
        config.model_copy(
            update={
                "observation": config.observation.model_copy(
                    update={
                        "max_python_files": (
                            config.observation.max_python_files + 1
                        )
                    }
                )
            }
        ),
        config.model_copy(
            update={
                "llm": config.llm.model_copy(
                    update={"max_concurrency": config.llm.max_concurrency + 1}
                )
            }
        ),
    ]
    assert all(
        protocol_manifest_for_config(
            variant,
            agent_profile_digest="0" * 64,
        ).protocol_manifest_digest
        != manifest.protocol_manifest_digest
        for variant in method_variants
    )

    tampered = manifest.model_dump(mode="json")
    tampered["endpoint_profile_digest"] = "f" * 64
    with pytest.raises(ValidationError, match="incorrect protocol-manifest digest"):
        ProtocolManifest.model_validate(tampered)

    legacy = manifest.model_dump(mode="json")
    legacy.pop("target_content_admission_policy")
    with pytest.raises(ValidationError, match="target_content_admission_policy"):
        ProtocolManifest.model_validate(legacy)


def test_active_full_run_uses_hardened_scale_and_git_resource_bounds() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "hardened.yaml"
    )
    config = load_build_config(path)
    assert (
        config.semantic.target_population_members_per_hypothesis,
        config.semantic.max_classified_population_members_per_hypothesis,
        config.semantic.max_total_classified_population_members,
        config.semantic.max_classification_calls_per_hypothesis,
        config.semantic.max_classification_request_bytes_per_hypothesis,
        config.semantic.max_total_classification_calls,
        config.semantic.max_total_classification_request_bytes,
        config.semantic.max_falsification_calls_per_hypothesis,
        config.semantic.max_falsification_request_bytes_per_hypothesis,
        config.semantic.max_total_falsification_calls,
        config.semantic.max_total_falsification_request_bytes,
    ) == (
        15,
        20,
        1_000,
        32,
        16_000_000,
        256,
        64_000_000,
        16,
        8_000_000,
        128,
        32_000_000,
    )
    unified = config.unified
    assert (
        unified.git_max_commands_per_case,
        unified.git_max_queries_per_family,
        unified.git_max_results_per_query,
        unified.git_max_patches_per_case,
        unified.git_max_bytes_per_command,
        unified.git_max_object_scan_bytes,
        unified.git_max_bytes_per_case,
        unified.git_max_changed_paths_per_commit,
        unified.max_temporal_items_per_case,
    ) == (16, 4, 24, 32, 2_000_000, 64_000_000, 8_000_000, 128, 64)


def test_complete_and_explicit_only_trees_validate(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    complete = tmp_path / "complete"
    result = build_fixture(repository, complete)
    assert result.inference_records
    assert (complete / "inference" / "index.json").is_file()
    assert (
        complete / "observations" / "semantic-units" / "index.json"
    ).is_file()
    assert result.inference_index.protocol_manifest == result.protocol_manifest
    assert result.protocol_manifest.package_version == "0.18.0"
    assert (
        result.protocol_manifest.semantic_unit_contract_version
        == "semantic-unit-1.0"
    )
    assert (
        result.protocol_manifest.population_spec_contract_version
        == "population-spec-1.2"
    )
    assert result.protocol_manifest.stage_protocol_versions[
        "current_hypothesis_reduce"
    ] == "current-hypothesis-reduce-1.31"
    assert result.protocol_manifest.stage_protocol_versions[
        "current_hypothesis_reduction_index"
    ] == "current-hypothesis-reduction-index-1.13"
    assert result.protocol_manifest.stage_protocol_versions[
        "population_semantic_classify"
    ] == "semantic-member-classification-2.3"
    assert result.protocol_manifest.stage_protocol_versions[
        "counterexample_verify"
    ] == "counterexample-falsification-1.11"
    assert not (complete / "calibration.json").exists()
    assert validate_run_directory(complete, repository_root=repository) == []

    explicit = tmp_path / "explicit"
    config = config_for_tests(
        observation={"enabled": False, "queries": []},
    )
    build_fixture(repository, explicit, config=config)
    assert (explicit / "inference" / "index.json").is_file()
    assert not (explicit / "observations").exists()
    assert validate_run_directory(explicit, repository_root=repository) == []

    observation_only = tmp_path / "observation-only"
    observation_config = config_for_tests(
        semantic={"enabled": False, "required": False},
    )
    build_fixture(
        repository,
        observation_only,
        config=observation_config,
    )
    assert (observation_only / "observations" / "index.json").is_file()
    assert not (
        observation_only / "observations" / "semantic-units"
    ).exists()
    assert validate_run_directory(
        observation_only,
        repository_root=repository,
    ) == []


def test_inference_and_semantic_tamper_are_detected(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    inference_run = tmp_path / "inference-run"
    build_fixture(repository, inference_run)
    request = next((inference_run / "inference" / "requests").glob("*.json"))
    value = json.loads(request.read_text(encoding="utf-8"))
    value["request_payload_digest"] = "0" * 64
    _rewrite(request, value)
    assert any(
        "inference" in item or "digest" in item
        for item in validate_run_directory(
            inference_run,
            repository_root=repository,
        )
    )

    semantic_run = tmp_path / "semantic-run"
    build_fixture(repository, semantic_run)
    unit = next(
        (
            semantic_run
            / "observations"
            / "semantic-units"
            / "units"
        ).glob("*.json")
    )
    value = json.loads(unit.read_text(encoding="utf-8"))
    value["artifact_digest"] = "0" * 64
    _rewrite(unit, value)
    assert any(
        "semantic-unit" in item
        for item in validate_run_directory(
            semantic_run,
            repository_root=repository,
        )
    )

    protocol_run = tmp_path / "protocol-run"
    build_fixture(repository, protocol_run)
    index_path = protocol_run / "inference" / "index.json"
    value = json.loads(index_path.read_text(encoding="utf-8"))
    value["protocol_manifest"]["explicit_prompt_version"] = "tampered"
    _rewrite(index_path, value)
    assert any(
        "protocol" in item or "digest" in item
        for item in validate_run_directory(
            protocol_run,
            repository_root=repository,
        )
    )


def test_validator_rejects_endpoint_profile_change(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "endpoint-profile-run"
    build_fixture(repository, run)
    config_path = run / "config.json"
    value = json.loads(config_path.read_text(encoding="utf-8"))
    value["llm"]["base_url"] = "https://other-endpoint.invalid/v1"
    _rewrite(config_path, value)

    errors = validate_run_directory(run, repository_root=repository)
    assert "inference protocol manifest differs from resolved config" in errors


def test_review_and_audit_must_match_inference_replay(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    # The ordinary smoke fixture can now close all cases after the stricter
    # grouping/routing fixes.  Exercise the review surface with an explicit
    # host-owned fail-closed boundary instead of relying on incidental model
    # grouping or a related_variant relation to require Git.
    review_config = config_for_tests(unified={"max_case_members": 1})
    review_run = tmp_path / "review-run"
    build_fixture(repository, review_run, config=review_config)
    review_path = review_run / "canonical" / "claims.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review_claim = next(
        item for item in review["claims"] if item["epistemic_status"] == "review"
    )
    assert review["format_version"] == "unified-canonical-claims-1.0"
    review_claim["reason_codes"].append("coordinated-but-false")
    review_claim["reason_codes"].sort()
    _rewrite(review_path, review)
    assert any(
        "canonical/claims.json" in item
        for item in validate_run_directory(
            review_run,
            repository_root=repository,
        )
    )

    audit_run = tmp_path / "audit-run"
    build_fixture(repository, audit_run, config=review_config)
    audit_path = audit_run / "audit.jsonl"
    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["reason_code"] = "coordinated-but-false"
    audit_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )
    assert any(
        "audit.jsonl differs" in item
        for item in validate_run_directory(
            audit_run,
            repository_root=repository,
        )
    )
