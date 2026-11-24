"""End-to-end schema-4 unified fixed-snapshot knowledge builder."""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import yaml

from repo_constraint_memory_v3 import __version__
from repo_constraint_memory_v3.adjudication import (
    EXPLICIT_ADJUDICATION_VERSION,
    AdjudicationResult,
    CandidateDecision,
    candidate_fingerprint,
    finalize_candidate_decisions,
    prepare_candidate_decisions,
)
from repo_constraint_memory_v3.agent_runtime import (
    AGENT_RUN_ARTIFACT_SCHEMA_VERSION,
    AGENT_RUNNER_PROTOCOL_VERSION,
    AGENT_RUNNER_REQUEST_SCHEMA_VERSION,
    AGENT_RUNNER_RESULT_SCHEMA_VERSION,
    SOURCE_ACQUISITION_SCHEMA_VERSION,
    SOURCE_DISCOVERY_PROMPT_VERSION,
    stable_model_digest,
)
from repo_constraint_memory_v3.chunking import (
    CHUNKER_VERSION,
    UnitBatch,
    batch_units,
    chunk_sources,
)
from repo_constraint_memory_v3.discovery import (
    SourceFile,
    discover_repository,
    load_source_file,
)
from repo_constraint_memory_v3.exporters import (
    ChannelArtifactWriter,
    execution_model_run_summary,
    stable_json_text,
    write_build_outputs,
)
from repo_constraint_memory_v3.explicit_verification import (
    EXPLICIT_VERIFICATION_PLAN_VERSION,
    EXPLICIT_VERIFICATION_RESULT_VERSION,
    EXPLICIT_VERIFICATION_POLICY_VERSION,
    EXPLICIT_VERIFICATION_PROTOCOL_VERSION,
    EXPLICIT_VERIFICATION_RESPONSE_SCHEMA_VERSION,
    MAX_CANDIDATE_BYTES as MAX_EXPLICIT_VERIFICATION_CANDIDATE_BYTES,
    MAX_CANDIDATES_PER_BATCH as MAX_EXPLICIT_VERIFICATION_CANDIDATES_PER_BATCH,
    MAX_OUTPUT_TOKENS as MAX_EXPLICIT_VERIFICATION_OUTPUT_TOKENS,
    MAX_REQUEST_BYTES as MAX_EXPLICIT_VERIFICATION_REQUEST_BYTES,
    MAX_RESPONSE_BYTES as MAX_EXPLICIT_VERIFICATION_RESPONSE_BYTES,
    MAX_SCHEMA_REPAIRS as MAX_EXPLICIT_VERIFICATION_SCHEMA_REPAIRS,
    ExplicitGroundedCandidate,
    ExplicitVerificationBundle,
    explicit_candidate_root_id,
    make_grounded_candidate_input,
    run_explicit_verification,
)
from repo_constraint_memory_v3.extraction import (
    MAX_OUTPUT_TOKENS as MAX_EXPLICIT_EXTRACTION_OUTPUT_TOKENS,
    MAX_RESPONSE_BYTES as MAX_EXPLICIT_EXTRACTION_RESPONSE_BYTES,
    MAX_TIMEOUT_SECONDS as MAX_EXPLICIT_EXTRACTION_TIMEOUT_SECONDS,
    PROMPT_VERSION,
    RESPONSE_SCHEMA_VERSION,
    BatchExtractionOutcome,
    extract_batch,
    order_results_by_source_plan,
    stable_batch_request_ordinal,
)
from repo_constraint_memory_v3.inference import (
    InferenceBudget,
    InferenceIndex,
    InferenceRecord,
    MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES,
    ProtocolManifest,
    STRUCTURED_RESPONSE_NORMALIZATION_VERSION,
    make_inference_index,
    make_protocol_manifest,
    run_call_permitted_stage_calls,
)
from repo_constraint_memory_v3.inference_cache import SuccessInferenceCache
from repo_constraint_memory_v3.llm import LLMClient
from repo_constraint_memory_v3.models import (
    AttemptAudit,
    AuditRecord,
    BuildConfig,
    CandidateAuditItem,
    Coverage,
    CurrentSourceCoverage,
    DeliveryCoverage,
    DiscoveryArtifact,
    EvidenceUnit,
    ExplicitCoverage,
    ExplicitReviewItem,
    MemoryFile,
    ObservationCoverage,
    ObservationQueryCoverage,
    RepositoryMetadata,
    ReviewFile,
    ReviewItem,
    SCHEMA_VERSION,
    UnifiedCapabilityCoverage,
    UnifiedCoverage,
    SemanticUnitCoverage,
    SemanticMapCoverage,
    HypothesisReductionCoverage,
    LLMSettings,
    PopulationVerificationCoverage,
    FalsificationCoverage,
)
from repo_constraint_memory_v3.model_profiles import (
    ExecutionModelProfile,
    load_execution_model_profile,
)
from repo_constraint_memory_v3.observations import (
    ObservationBundle,
    run_observations,
)
from repo_constraint_memory_v3.observations.canonical import sha256_hex
from repo_constraint_memory_v3.observations.models import (
    PYTHON_CLONE_QUERY_VERSION,
    PYTHON_SNAPSHOT_QUERY_VERSION,
)
from repo_constraint_memory_v3.observations.python_snapshot import (
    AST_CONTRACT_VERSION,
)
from repo_constraint_memory_v3.observations.semantic_units import (
    PACKET_CONTRACT_VERSION,
    SEMANTIC_UNIT_CONTRACT_VERSION,
    SEMANTIC_UNIT_INDEX_VERSION,
    SEMANTIC_UNIT_SELECTION_VERSION,
    SemanticUnitBundle,
    build_semantic_units,
)
from repo_constraint_memory_v3.propositions import PROPOSITION_CONTRACT_VERSION
from repo_constraint_memory_v3.semantic.classify import (
    CLASSIFY_PROTOCOL_VERSION,
    CLASSIFY_RESPONSE_SCHEMA_VERSION,
    CLASSIFICATION_PLAN_VERSION,
    MEMBER_PACKET_CONTRACT_VERSION,
    POPULATION_ARTIFACT_VERSION,
    POPULATION_SAMPLING_PLAN_VERSION,
)
from repo_constraint_memory_v3.semantic.classification_transport import (
    CLASSIFICATION_TRANSPORT_PLAN_VERSION,
)
from repo_constraint_memory_v3.semantic.falsification import (
    FALSIFICATION_PROTOCOL_VERSION,
    FALSIFICATION_RESPONSE_SCHEMA_VERSION,
)
from repo_constraint_memory_v3.semantic.models import (
    FALSIFICATION_RESULT_VERSION,
    POPULATION_SPEC_VERSION,
)
from repo_constraint_memory_v3.semantic.map import (
    MAP_INDEX_VERSION,
    MAP_PROTOCOL_VERSION,
    MAP_RESPONSE_SCHEMA_VERSION,
)
from repo_constraint_memory_v3.semantic.reduce import (
    POPULATION_CONTEXT_VERSION,
    MAX_REDUCE_SELECTION_PROMPT_BYTES,
    MAX_REDUCE_SELECTION_CANDIDATES,
    REDUCTION_INDEX_VERSION,
    REDUCE_PROTOCOL_VERSION,
    REDUCE_RESPONSE_SCHEMA_VERSION,
)
from repo_constraint_memory_v3.semantic.validation import (
    CURRENT_SOURCE_INDEX_VERSION,
    CurrentSourceBundle,
    run_current_source_pipeline,
)
from repo_constraint_memory_v3.unified.export import write_unified_outputs
from repo_constraint_memory_v3.unified.current_evidence import (
    CURRENT_EVIDENCE_BUNDLE_VERSION,
)
from repo_constraint_memory_v3.unified.grouping import GROUPING_POLICY_VERSION
from repo_constraint_memory_v3.unified.git_evidence import (
    GIT_EVIDENCE_VERSION,
    GIT_RAW_EXECUTION_VERSION,
)
from repo_constraint_memory_v3.temporal_need import GIT_NEED_PROTOCOL_VERSION
from repo_constraint_memory_v3.unified.refinement import UNIFIED_REFINEMENT_PROTOCOL_VERSION
from repo_constraint_memory_v3.unified.inputs import UNIFIED_CHANNEL_ADAPTER_VERSION
from repo_constraint_memory_v3.unified.models import (
    UNIFIED_CONTRACT_VERSION,
    UNIFIED_POLICY_VERSION,
)
from repo_constraint_memory_v3.unified.pipeline import (
    UNIFIED_GIT_ROUTING_VERSION,
    UnifiedRunResult,
    run_unified_layer,
)
from repo_constraint_memory_v3.unified.publication import (
    consume_initial_empty_directory,
)
from repo_constraint_memory_v3.unified.semantic import (
    CANDIDATE_GROUP_PROTOCOL_VERSION,
    CANDIDATE_GROUP_RESPONSE_VERSION,
    TEMPORAL_CLASSIFIER_PROTOCOL_VERSION,
    TEMPORAL_CLASSIFIER_RESPONSE_VERSION,
    UNIFIED_VERIFIER_PROTOCOL_VERSION,
    UNIFIED_VERIFIER_RESPONSE_VERSION,
    VERIFIER_CURRENT_EVIDENCE_VIEW_VERSION,
)
from repo_constraint_memory_v3.unified.upstream import (
    UNIFIED_UPSTREAM_INPUT_VERSION,
    UnifiedUpstreamInputs,
    make_unified_upstream_inputs,
)
from repo_constraint_memory_v3.traceability import (
    TraceabilityArtifact,
    build_explicit_channel,
    make_traceability,
    not_run_channel,
)
from repo_constraint_memory_v3.source_plan import (
    SourcePlanResult,
    validate_source_plan,
)
from repo_constraint_memory_v3.source_acquisition import (
    SourceAcquisitionBundle,
    validate_acquisition_for_build,
)


class SnapshotMismatchError(ValueError):
    """Discovery no longer describes the current repository snapshot."""


@dataclass(frozen=True)
class BuildResult:
    source_acquisition: SourceAcquisitionBundle
    memory: MemoryFile
    build_report: dict[str, object]
    audit_records: tuple[AuditRecord, ...]
    review: ReviewFile
    explicit_verification: ExplicitVerificationBundle
    observation: ObservationBundle | None
    semantic_units: SemanticUnitBundle | None
    current_source: CurrentSourceBundle | None
    upstream_inputs: UnifiedUpstreamInputs
    unified: UnifiedRunResult
    traceability: TraceabilityArtifact
    inference_index: InferenceIndex
    inference_records: tuple[InferenceRecord, ...]
    protocol_manifest: ProtocolManifest
    output_paths: dict[str, Path]


class _TrackingClient:
    """Measure actual simultaneous complete() calls around a shared client."""

    def __init__(self, delegate: LLMClient) -> None:
        self.delegate = delegate
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak_in_flight = 0

    def complete(self, system_prompt: str, user_prompt: str):
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            return self.delegate.complete(system_prompt, user_prompt)
        finally:
            with self._lock:
                self._in_flight -= 1

    def complete_with_settings(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        settings: LLMSettings,
    ):
        """Forward optional per-stage transport bounds while tracking concurrency."""

        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            bounded_complete = getattr(self.delegate, "complete_with_settings", None)
            if callable(bounded_complete):
                return bounded_complete(
                    system_prompt,
                    user_prompt,
                    settings=settings,
                )
            return self.delegate.complete(system_prompt, user_prompt)
        finally:
            with self._lock:
                self._in_flight -= 1


def load_build_config(
    path: Path,
    *,
    model_profile: ExecutionModelProfile | None = None,
) -> BuildConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    profile = model_profile or load_execution_model_profile()
    llm = raw.get("llm")
    if not isinstance(llm, dict):
        raise ValueError("configuration llm section must be a mapping")
    forbidden_identity = {
        "provider",
        "model",
        "require_response_model",
        "accepted_effective_model_ids",
        "api_key_env",
        "base_url",
        "temperature",
        "thinking_mode",
        "reasoning_effort",
        "reasoning_max_tokens",
        "reasoning_wire_policy",
    }
    overlap = forbidden_identity & set(llm)
    if overlap:
        raise ValueError(
            "build config cannot duplicate execution-model profile fields: "
            f"{sorted(overlap)}"
        )
    if profile.provider.protocol == "offline_mock":
        provider = "openai_compatible"
        api_key_env = "NORM4CODE_OFFLINE_UNUSED_KEY"
        base_url = "https://offline.invalid/v1"
    else:
        provider = "openai_compatible"
        assert profile.provider.api_key_env is not None
        assert profile.provider.base_url is not None
        api_key_env = profile.provider.api_key_env
        base_url = profile.provider.base_url
    raw = dict(raw)
    raw["execution_model_profile_id"] = profile.profile_id
    raw["execution_model_profile_digest"] = profile.profile_digest
    raw["llm"] = {
        **llm,
        "provider": provider,
        "model": profile.adapter_model_ids.core_openai_chat,
        "require_response_model": profile.effective_identity.require_response_model,
        "accepted_effective_model_ids": (
            profile.effective_identity.accepted_model_ids
        ),
        "api_key_env": api_key_env,
        "base_url": base_url,
        "temperature": profile.generation.temperature,
        "thinking_mode": profile.reasoning.mode,
        "reasoning_effort": profile.reasoning.effort,
        "reasoning_max_tokens": profile.reasoning.max_tokens,
        "reasoning_wire_policy": profile.reasoning.wire_policy,
    }
    return BuildConfig.model_validate(raw)


def protocol_manifest_for_config(
    config: BuildConfig,
    *,
    agent_profile_digest: str,
) -> ProtocolManifest:
    """Resolve the complete non-secret schema-4.11 acquisition protocol."""

    return make_protocol_manifest(
        settings=config.llm,
        execution_model_profile_id=config.execution_model_profile_id,
        execution_model_profile_digest=config.execution_model_profile_digest,
        agent_profile_digest=agent_profile_digest,
        repository_settings_digest=stable_model_digest(config.repository),
        discovery_settings_digest=stable_model_digest(config.discovery),
        agent_settings_digest=stable_model_digest(config.agent),
        chunking_settings_digest=stable_model_digest(config.chunking),
        observation_settings_digest=stable_model_digest(config.observation),
        semantic_settings_digest=stable_model_digest(config.semantic),
        unified_settings_digest=stable_model_digest(config.unified),
        target_content_admission_policy=(
            config.discovery.target_content_admission_policy
        ),
        agent_stage_protocol_versions={
            "source_discovery_prompt": SOURCE_DISCOVERY_PROMPT_VERSION,
            "runner_protocol": AGENT_RUNNER_PROTOCOL_VERSION,
            "runner_request": AGENT_RUNNER_REQUEST_SCHEMA_VERSION,
            "runner_result": AGENT_RUNNER_RESULT_SCHEMA_VERSION,
            "agent_run": AGENT_RUN_ARTIFACT_SCHEMA_VERSION,
            "source_acquisition": SOURCE_ACQUISITION_SCHEMA_VERSION,
        },
        explicit_prompt_version=PROMPT_VERSION,
        explicit_response_schema_version=RESPONSE_SCHEMA_VERSION,
        parser_contract_version=AST_CONTRACT_VERSION,
        observation_contract_versions={
            "python_snapshot": PYTHON_SNAPSHOT_QUERY_VERSION,
            "python_clone_clusters": PYTHON_CLONE_QUERY_VERSION,
        },
        semantic_unit_contract_version=SEMANTIC_UNIT_CONTRACT_VERSION,
        packet_contract_version=PACKET_CONTRACT_VERSION,
        population_spec_contract_version=POPULATION_SPEC_VERSION,
        proposition_contract_version=PROPOSITION_CONTRACT_VERSION,
        stage_protocol_versions={
            "explicit_chunking": CHUNKER_VERSION,
            "explicit_adjudication": EXPLICIT_ADJUDICATION_VERSION,
            "explicit_semantic_verify": EXPLICIT_VERIFICATION_PROTOCOL_VERSION,
            "explicit_semantic_verify_response": (
                EXPLICIT_VERIFICATION_RESPONSE_SCHEMA_VERSION
            ),
            "explicit_semantic_verify_policy": (
                EXPLICIT_VERIFICATION_POLICY_VERSION
            ),
            "explicit_semantic_verify_plan": EXPLICIT_VERIFICATION_PLAN_VERSION,
            "explicit_semantic_verify_result": EXPLICIT_VERIFICATION_RESULT_VERSION,
            "current_semantic_map": MAP_PROTOCOL_VERSION,
            "current_semantic_map_response": MAP_RESPONSE_SCHEMA_VERSION,
            "current_semantic_map_index": MAP_INDEX_VERSION,
            "semantic_unit_index": SEMANTIC_UNIT_INDEX_VERSION,
            "semantic_unit_selection": SEMANTIC_UNIT_SELECTION_VERSION,
            "current_hypothesis_reduce": REDUCE_PROTOCOL_VERSION,
            "current_hypothesis_reduce_response": (
                REDUCE_RESPONSE_SCHEMA_VERSION
            ),
            "current_hypothesis_reduction_index": REDUCTION_INDEX_VERSION,
            "population_planning_context": POPULATION_CONTEXT_VERSION,
            "population_semantic_classify": CLASSIFY_PROTOCOL_VERSION,
            "population_semantic_classify_response": (
                CLASSIFY_RESPONSE_SCHEMA_VERSION
            ),
            "counterexample_verify": FALSIFICATION_PROTOCOL_VERSION,
            "counterexample_verify_response": (
                FALSIFICATION_RESPONSE_SCHEMA_VERSION
            ),
            "classification_plan": CLASSIFICATION_PLAN_VERSION,
            "classification_transport_plan": CLASSIFICATION_TRANSPORT_PLAN_VERSION,
            "population_sampling_plan": POPULATION_SAMPLING_PLAN_VERSION,
            "falsification_result": FALSIFICATION_RESULT_VERSION,
            "population_member_packet": (
                MEMBER_PACKET_CONTRACT_VERSION
            ),
            "current_population": POPULATION_ARTIFACT_VERSION,
            "current_source_index": CURRENT_SOURCE_INDEX_VERSION,
            "traceability": "candidate-traceability-1.3",
            "unified_channel_adapter": UNIFIED_CHANNEL_ADAPTER_VERSION,
            "unified_evidence_contract": UNIFIED_CONTRACT_VERSION,
            "candidate_grouping": GROUPING_POLICY_VERSION,
            "candidate_semantic_group": CANDIDATE_GROUP_PROTOCOL_VERSION,
            "candidate_semantic_group_response": (
                CANDIDATE_GROUP_RESPONSE_VERSION
            ),
            "current_evidence_bundle": CURRENT_EVIDENCE_BUNDLE_VERSION,
            "unified_git_routing": UNIFIED_GIT_ROUTING_VERSION,
            "git_need_assessment": GIT_NEED_PROTOCOL_VERSION,
            "git_raw_execution": GIT_RAW_EXECUTION_VERSION,
            "git_query_manifest": GIT_EVIDENCE_VERSION,
            "temporal_evidence_classify": TEMPORAL_CLASSIFIER_PROTOCOL_VERSION,
            "temporal_evidence_classify_response": (
                TEMPORAL_CLASSIFIER_RESPONSE_VERSION
            ),
            "unified_case_verify": UNIFIED_VERIFIER_PROTOCOL_VERSION,
            "unified_case_verify_response": (
                UNIFIED_VERIFIER_RESPONSE_VERSION
            ),
            "verifier_current_evidence_view": (
                VERIFIER_CURRENT_EVIDENCE_VIEW_VERSION
            ),
            "unified_upstream_inputs": UNIFIED_UPSTREAM_INPUT_VERSION,
            "unified_refinement": UNIFIED_REFINEMENT_PROTOCOL_VERSION,
            "unified_canonical_policy": UNIFIED_POLICY_VERSION,
            "structured_response_normalization": (
                STRUCTURED_RESPONSE_NORMALIZATION_VERSION
            ),
        },
        additional_resource_limits={
            "max_explicit_extraction_units_per_batch": (
                config.chunking.max_units_per_batch
            ),
            "max_explicit_extraction_output_tokens": (
                MAX_EXPLICIT_EXTRACTION_OUTPUT_TOKENS
            ),
            "max_explicit_extraction_timeout_seconds": (
                MAX_EXPLICIT_EXTRACTION_TIMEOUT_SECONDS
            ),
            "max_explicit_extraction_response_bytes": (
                MAX_EXPLICIT_EXTRACTION_RESPONSE_BYTES
            ),
            "max_explicit_verification_candidates_per_batch": (
                MAX_EXPLICIT_VERIFICATION_CANDIDATES_PER_BATCH
            ),
            "max_explicit_verification_candidate_bytes": (
                MAX_EXPLICIT_VERIFICATION_CANDIDATE_BYTES
            ),
            "max_explicit_verification_request_bytes": (
                MAX_EXPLICIT_VERIFICATION_REQUEST_BYTES
            ),
            "max_explicit_verification_output_tokens": (
                MAX_EXPLICIT_VERIFICATION_OUTPUT_TOKENS
            ),
            "max_explicit_verification_response_bytes": (
                MAX_EXPLICIT_VERIFICATION_RESPONSE_BYTES
            ),
            "max_explicit_verification_schema_repairs": (
                MAX_EXPLICIT_VERIFICATION_SCHEMA_REPAIRS
            ),
            "max_semantic_units": config.semantic.max_units,
            "max_unit_source_bytes": config.semantic.max_unit_source_bytes,
            "max_packet_bytes": config.semantic.max_packet_bytes,
            "max_total_packet_bytes": config.semantic.max_total_packet_bytes,
            "max_context_refs_per_packet": (
                config.semantic.max_context_refs_per_packet
            ),
            "max_map_packets_per_request": (
                config.semantic.max_map_packets_per_request
            ),
            "max_map_request_bytes": config.semantic.max_map_request_bytes,
            "max_current_semantic_map_response_bytes": (
                MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES
            ),
            "max_observations_per_primary_unit": (
                config.semantic.max_observations_per_primary_unit
            ),
            "max_total_observations": (
                config.semantic.max_total_observations
            ),
            "max_reduce_items_per_request": (
                config.semantic.max_reduce_items_per_request
            ),
            "max_reduce_request_bytes": (
                config.semantic.max_reduce_request_bytes
            ),
            "max_current_hypothesis_reduce_response_bytes": (
                MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES
            ),
            "max_hypotheses_per_component": (
                config.semantic.max_hypotheses_per_component
            ),
            "max_reduce_selection_candidates": (
                MAX_REDUCE_SELECTION_CANDIDATES
            ),
            "max_reduce_selection_prompt_bytes": (
                MAX_REDUCE_SELECTION_PROMPT_BYTES
            ),
            "max_total_hypotheses": config.semantic.max_total_hypotheses,
            "max_population_query_nodes": (
                config.semantic.max_population_query_nodes
            ),
            "max_population_query_depth": (
                config.semantic.max_population_query_depth
            ),
            "max_population_literal_values": (
                config.semantic.max_population_literal_values
            ),
            "max_population_members": (
                config.semantic.max_population_members
            ),
            "max_population_members_per_hypothesis": (
                config.semantic.max_population_members_per_hypothesis
            ),
            "max_total_population_members": (
                config.semantic.max_total_population_members
            ),
            "target_population_members_per_hypothesis": (
                config.semantic.target_population_members_per_hypothesis
            ),
            "max_classified_population_members_per_hypothesis": (
                config.semantic
                .max_classified_population_members_per_hypothesis
            ),
            "max_total_classified_population_members": (
                config.semantic.max_total_classified_population_members
            ),
            "max_population_artifact_bytes": (
                config.semantic.max_population_artifact_bytes
            ),
            "max_classification_members_per_request": (
                config.semantic.max_classification_members_per_request
            ),
            "max_classification_request_bytes": (
                config.semantic.max_classification_request_bytes
            ),
            "max_population_semantic_classify_response_bytes": (
                MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES
            ),
            "max_classification_calls_per_hypothesis": (
                config.semantic.max_classification_calls_per_hypothesis
            ),
            "max_classification_request_bytes_per_hypothesis": (
                config.semantic
                .max_classification_request_bytes_per_hypothesis
            ),
            "max_total_classification_calls": (
                config.semantic.max_total_classification_calls
            ),
            "max_total_classification_request_bytes": (
                config.semantic.max_total_classification_request_bytes
            ),
            "max_member_packet_bytes": (
                config.semantic.max_member_packet_bytes
            ),
            "max_falsification_items_per_request": (
                config.semantic.max_falsification_items_per_request
            ),
            "max_falsification_request_bytes": (
                config.semantic.max_falsification_request_bytes
            ),
            "max_counterexample_verify_response_bytes": (
                MAX_CURRENT_SOURCE_STAGE_RESPONSE_BYTES
            ),
            "max_falsification_calls_per_hypothesis": (
                config.semantic.max_falsification_calls_per_hypothesis
            ),
            "max_falsification_request_bytes_per_hypothesis": (
                config.semantic
                .max_falsification_request_bytes_per_hypothesis
            ),
            "max_total_falsification_calls": (
                config.semantic.max_total_falsification_calls
            ),
            "max_total_falsification_request_bytes": (
                config.semantic.max_total_falsification_request_bytes
            ),
            "max_refinement_rounds": config.semantic.max_refinement_rounds,
            "max_current_source_artifact_bytes": (
                config.semantic.max_current_source_artifact_bytes
            ),
            "max_unified_candidate_members": (
                config.unified.max_candidate_members
            ),
            "max_unified_cases": config.unified.max_cases,
            "max_unified_block_members": config.unified.max_block_members,
            "max_unified_group_pairs": config.unified.max_group_pairs,
            "max_unified_group_pairs_per_request": (
                config.unified.max_group_pairs_per_request
            ),
            "max_unified_group_members_per_request": (
                config.unified.max_group_members_per_request
            ),
            "max_unified_group_request_bytes": (
                config.unified.max_group_request_bytes
            ),
            "max_unified_case_members": config.unified.max_case_members,
            "max_unified_case_input_bytes": (
                config.unified.max_case_input_bytes
            ),
            "max_current_evidence_items": (
                config.unified.max_current_evidence_items
            ),
            "max_current_evidence_bytes": (
                config.unified.max_current_evidence_bytes
            ),
            "git_command_timeout_seconds": (
                config.unified.git_command_timeout_seconds
            ),
            "git_max_commands_per_case": (
                config.unified.git_max_commands_per_case
            ),
            "git_max_queries_per_family": (
                config.unified.git_max_queries_per_family
            ),
            "git_max_results_per_query": (
                config.unified.git_max_results_per_query
            ),
            "git_max_patches_per_case": (
                config.unified.git_max_patches_per_case
            ),
            "git_max_bytes_per_command": (
                config.unified.git_max_bytes_per_command
            ),
            "git_max_object_scan_bytes": (
                config.unified.git_max_object_scan_bytes
            ),
            "git_max_bytes_per_case": (
                config.unified.git_max_bytes_per_case
            ),
            "git_max_changed_paths_per_commit": (
                config.unified.git_max_changed_paths_per_commit
            ),
            "max_temporal_items_per_case": (
                config.unified.max_temporal_items_per_case
            ),
            "max_temporal_batch_items": (
                config.unified.max_temporal_batch_items
            ),
            "max_temporal_cells_per_request": (
                config.unified.max_temporal_cells_per_request
            ),
            "max_temporal_request_bytes": (
                config.unified.max_temporal_request_bytes
            ),
            "max_unified_refinement_rounds": (
                config.unified.max_refinement_rounds
            ),
            "max_unified_refinement_children": (
                config.unified.max_refinement_children
            ),
            "max_unified_artifact_bytes": (
                config.unified.max_unified_artifact_bytes
            ),
            "agent_source_discovery_timeout_seconds": (
                config.agent.stages["source_discovery"].timeout_seconds
            ),
            "agent_source_discovery_max_turns": (
                config.agent.stages["source_discovery"].max_turns
            ),
            "agent_source_discovery_max_workspace_files": (
                config.agent.stages["source_discovery"].max_workspace_files
            ),
            "agent_source_discovery_max_workspace_bytes": (
                config.agent.stages["source_discovery"].max_workspace_bytes
            ),
            "agent_source_discovery_max_output_bytes": (
                config.agent.stages["source_discovery"].max_output_bytes
            ),
            "agent_source_discovery_max_trajectory_bytes": (
                config.agent.stages["source_discovery"].max_trajectory_bytes
            ),
        },
    )


def _verify_discovery(
    repository_root: Path,
    config: BuildConfig,
    discovery: DiscoveryArtifact,
) -> DiscoveryArtifact:
    resolved = repository_root.resolve()
    if Path(discovery.repository.root).resolve() != resolved:
        raise SnapshotMismatchError(
            "discovery repository root does not match the build repository"
        )
    if discovery.limits != config.discovery:
        raise SnapshotMismatchError(
            "discovery limits do not match the current build configuration"
        )
    current = discover_repository(
        resolved,
        config.discovery,
        repository_name=config.repository.name,
    )
    if current.repository.snapshot_digest != discovery.repository.snapshot_digest:
        raise SnapshotMismatchError(
            "repository snapshot changed after discover; rerun discover"
        )
    # Schema 4.1 binds temporal evidence to the exact Git cutoff as well as
    # the content snapshot.  A history-only HEAD move therefore invalidates
    # the acquisition even when tracked file bytes happen to be identical.
    comparable_current = current.model_dump(mode="json")
    comparable_supplied = discovery.model_dump(mode="json")
    if comparable_current != comparable_supplied:
        raise SnapshotMismatchError(
            "discovery artifact does not match deterministic rediscovery"
        )
    return current


def _extract_batches(
    batches: list[UnitBatch],
    *,
    sources_by_path: Mapping[str, SourceFile],
    client: LLMClient,
    config: BuildConfig,
    snapshot_digest: str,
    protocol_manifest: ProtocolManifest,
    budget: InferenceBudget,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[BatchExtractionOutcome], float, int]:
    tracking = (
        client if isinstance(client, _TrackingClient) else _TrackingClient(client)
    )
    progress_lock = threading.Lock()
    completed_count = 0
    batch_count = len(batches)
    budget.plan_stage("explicit_extract", batch_count)

    def notify(message: str) -> None:
        if progress_callback is None:
            return
        with progress_lock:
            progress_callback(message)

    def extract_one(work_item: tuple[int, UnitBatch]) -> BatchExtractionOutcome:
        nonlocal completed_count
        position, batch = work_item
        notify(
            f"batch {position}/{batch_count} {batch.batch_id} started "
            f"units={len(batch.units)} chars={batch.total_chars}"
        )
        started = time.monotonic()
        outcome = extract_batch(
            batch,
            tracking,
            sources_by_path=sources_by_path,
            model_name=config.llm.model,
            schema_repairs=config.llm.schema_repairs,
            provider_retries=config.llm.provider_retries,
            provider_retry_delay_seconds=float(
                config.llm.provider_retry_delay_seconds
            ),
            snapshot_digest=snapshot_digest,
            protocol_manifest_digest=(
                protocol_manifest.protocol_manifest_digest
            ),
            request_ordinal=stable_batch_request_ordinal(batch),
            budget=budget,
            llm_settings=config.llm,
        )
        elapsed = max(time.monotonic() - started, 0.0)
        with progress_lock:
            completed_count += 1
            if progress_callback is not None:
                progress_callback(
                    f"batch {position}/{batch_count} {batch.batch_id} "
                    f"{'completed' if outcome.succeeded else 'failed'} "
                    f"done={completed_count}/{batch_count} "
                    f"attempts={len(outcome.attempts)} "
                    f"repairs={sum(item.repair for item in outcome.attempts)} "
                    f"elapsed={elapsed:.1f}s"
                )
        return outcome

    started = time.monotonic()
    work_items = list(enumerate(batches, start=1))
    outcomes = run_call_permitted_stage_calls(
        work_items,
        extract_one,
        budget=budget,
        max_concurrency=config.llm.max_concurrency,
        thread_name_prefix="repo-memory-v3",
    )
    return (
        outcomes,
        max(time.monotonic() - started, 0.0),
        tracking.peak_in_flight,
    )


def _attempt_audit(outcome: BatchExtractionOutcome) -> list[AttemptAudit]:
    return [
        AttemptAudit(
            schema_attempt=item.schema_attempt,
            output_limit_retry=item.output_limit_retry,
            provider_try=item.provider_try,
            repair=item.repair,
            parsed=item.parsed,
            provider_error=item.provider_error,
            validation_errors=item.validation_errors,
        )
        for item in outcome.attempts
    ]


def _explicit_verification_inputs(
    decisions: Iterable[CandidateDecision],
) -> tuple[ExplicitGroundedCandidate, ...]:
    """Deduplicate every fully grounded candidate root for verifier coverage."""

    by_root: dict[str, ExplicitGroundedCandidate] = {}
    for decision in decisions:
        candidate = decision.raw_candidate
        if candidate is None or len(decision.grounded) != len(candidate.evidence):
            continue
        canonical_scope = (
            decision.canonical_scope
            if decision.canonical_scope
            else tuple(sorted(set(candidate.applies_to)))
        )
        condition = (
            (candidate.condition or "").strip() or None
            if decision.condition_source == "unprocessed"
            else decision.canonical_condition or None
        )
        item = make_grounded_candidate_input(
            unit_id=decision.unit_id,
            candidate_fingerprint=decision.candidate_fingerprint,
            candidate=candidate,
            grounded=decision.grounded,
            host_claim_kind=(
                decision.derived_kind
                if decision.derived_kind is not None
                and decision.derived_strength is not None
                else None
            ),
            host_normative_strength=(
                decision.derived_strength
                if decision.derived_kind is not None
                and decision.derived_strength is not None
                else None
            ),
            applies_to=canonical_scope,
            condition=condition,
        )
        existing = by_root.get(item.candidate_root_id)
        if existing is not None and existing != item:
            raise ValueError(
                "one explicit candidate root has non-identical grounded inputs"
            )
        by_root[item.candidate_root_id] = item
    return tuple(by_root[key] for key in sorted(by_root))


def _apply_explicit_verification(
    decisions: Iterable[CandidateDecision],
    verification: ExplicitVerificationBundle,
) -> None:
    """Attach verifier policy outcomes without allowing model-owned mutation."""

    policy_by_root = {
        item.candidate_root_id: item for item in verification.result.decisions
    }
    request_by_root = {
        item.stable_id: record.request_id
        for record in verification.inference_records
        for item in record.input_manifest
        if item.kind == "explicit_grounded_candidate"
    }
    for decision in decisions:
        candidate = decision.raw_candidate
        if candidate is None or len(decision.grounded) != len(candidate.evidence):
            continue
        root_id = explicit_candidate_root_id(decision.candidate_fingerprint)
        policy = policy_by_root.get(root_id)
        if policy is None:
            raise ValueError(f"explicit verifier omitted policy for {root_id}")
        decision.verification_disposition = policy.disposition
        decision.verification_request_id = request_by_root.get(root_id)
        decision.verification_reason_codes = list(policy.reason_codes)
        decision.verification_technical_reason_code = (
            policy.technical_reason_code
        )
        if policy.disposition == "eligible":
            continue
        decision.host_eligible = False
        decision.outcome = policy.disposition
        decision.outcome_stage = "explicit_semantic_verify"
        decision.errors.extend(
            f"explicit_verification:{reason}" for reason in policy.reason_codes
        )
        if policy.technical_reason_code is not None:
            decision.errors.append(
                "explicit_verification_technical:"
                f"{policy.technical_reason_code}"
            )


def _explicit_outcome_counts(
    decisions: Iterable[CandidateDecision],
) -> tuple[Counter[str], Counter[str]]:
    """Count raw events and unique roots without mixing their denominators."""

    raw_events: Counter[str] = Counter()
    outcome_by_root: dict[str, str] = {}
    for decision in decisions:
        raw_events[decision.outcome] += 1
        previous = outcome_by_root.setdefault(
            decision.candidate_fingerprint,
            decision.outcome,
        )
        if previous != decision.outcome:
            raise ValueError(
                "one explicit candidate root has conflicting host outcomes"
            )
    return raw_events, Counter(outcome_by_root.values())


def _review_item(
    decision: CandidateDecision,
    unit: EvidenceUnit,
    inference_request_id: str | None,
) -> ExplicitReviewItem:
    candidate = decision.raw_candidate
    status: str = "review" if decision.outcome != "rejected" else "rejected"
    return ExplicitReviewItem(
        candidate_fingerprint=decision.candidate_fingerprint,
        unit_id=unit.unit_id,
        path=unit.path,
        line_start=unit.line_start,
        line_end=unit.line_end,
        discovery_origin=unit.discovery_origin,
        inference_request_id=inference_request_id,
        verification_request_id=decision.verification_request_id,
        verification_reason_codes=decision.verification_reason_codes,
        verification_technical_reason_code=(
            decision.verification_technical_reason_code
        ),
        epistemic_status=status,  # type: ignore[arg-type]
        outcome_stage=decision.outcome_stage,  # type: ignore[arg-type]
        reason_code=decision.reason_code,
        raw_rule=candidate.rule if candidate else None,
        raw_claim_kind=candidate.claim_kind if candidate else None,
        raw_normative_strength=(
            candidate.normative_strength if candidate else None
        ),
        raw_applies_to=list(candidate.applies_to) if candidate else [],
        raw_condition=candidate.condition if candidate else None,
        recovered_evidence=[item.evidence for item in decision.grounded],
        derived_evidence_semantics=sorted(
            {
                flag
                for grounded in decision.grounded
                for flag in grounded.semantic_flags
            }
        ),
        evidence_relations=sorted(
            decision.relations,
            key=lambda item: (
                item.relation,
                item.route,
                item.target_fingerprint,
                item.reason,
            ),
        ),
        validation_errors=sorted(set(decision.errors)),
    )


def _source_funnel(
    plan: SourcePlanResult,
    units: list[EvidenceUnit],
    results_by_unit: Mapping[str, object],
    adjudication: AdjudicationResult,
    failed_unit_ids: set[str],
) -> dict[str, object]:
    unit_path = {unit.unit_id: unit.path for unit in units}
    source_rows: dict[str, dict[str, object]] = {}
    proposal_by_path = {item.path: item for item in plan.decisions}
    for metadata, origin in plan.sources:
        proposal = proposal_by_path.get(metadata.path)
        source_rows[metadata.path] = {
            "path": metadata.path,
            "origin": origin,
            "proposed": proposal is not None,
            "source_plan_admitted": True,
            "proposal_outcome": proposal.outcome if proposal else None,
            "proposal_reason": proposal.reason if proposal else None,
            "compared_baseline_paths": (
                list(proposal.compared_baseline_paths) if proposal else []
            ),
            "deterministic_overlap": (
                proposal.overlap_with_deterministic if proposal else False
            ),
            "loaded": True,
            "unit_count": 0,
            "raw_candidate_count": 0,
            "grounded_count": 0,
            "no_claim_unit_count": 0,
            "accepted_claim_ids": [],
            "review_count": 0,
            "rejected_count": 0,
            "failure_reasons": [],
        }
    for proposal in plan.decisions:
        if proposal.path in source_rows:
            continue
        source_rows[proposal.path] = {
            "path": proposal.path,
            "origin": "agent_supplement",
            "proposed": True,
            "source_plan_admitted": False,
            "proposal_outcome": proposal.outcome,
            "proposal_reason": proposal.reason,
            "compared_baseline_paths": list(
                proposal.compared_baseline_paths
            ),
            "deterministic_overlap": proposal.overlap_with_deterministic,
            "loaded": False,
            "unit_count": 0,
            "raw_candidate_count": 0,
            "grounded_count": 0,
            "no_claim_unit_count": 0,
            "accepted_claim_ids": [],
            "review_count": 0,
            "rejected_count": 0,
            "failure_reasons": [proposal.decision_reason],
        }
    for unit in units:
        row = source_rows.get(unit.path)
        if row is None:
            continue
        row["unit_count"] = int(row["unit_count"]) + 1
        if unit.unit_id in failed_unit_ids:
            cast_reasons = row["failure_reasons"]
            assert isinstance(cast_reasons, list)
            cast_reasons.append(f"unit_failed:{unit.unit_id}")
    for decision in adjudication.decisions:
        path = unit_path.get(decision.unit_id)
        row = source_rows.get(path or "")
        if row is None:
            continue
        row["raw_candidate_count"] = int(row["raw_candidate_count"]) + 1
        if (
            decision.raw_candidate is not None
            and len(decision.grounded) == len(decision.raw_candidate.evidence)
        ):
            row["grounded_count"] = int(row["grounded_count"]) + 1
        if decision.outcome == "accepted" and decision.claim_id:
            ids = row["accepted_claim_ids"]
            assert isinstance(ids, list)
            ids.append(decision.claim_id)
        elif decision.outcome == "rejected":
            row["rejected_count"] = int(row["rejected_count"]) + 1
        else:
            row["review_count"] = int(row["review_count"]) + 1
    for unit_id, result in results_by_unit.items():
        if getattr(result, "classification", None) != "no_claims":
            continue
        path = unit_path.get(unit_id)
        row = source_rows.get(path or "")
        if row is not None:
            row["no_claim_unit_count"] = int(row["no_claim_unit_count"]) + 1
    for row in source_rows.values():
        row["accepted_claim_ids"] = sorted(set(row["accepted_claim_ids"]))
        row["failure_reasons"] = sorted(set(row["failure_reasons"]))
    rows = sorted(source_rows.values(), key=lambda item: str(item["path"]))
    groups: dict[str, dict[str, object]] = {}
    for group_name, predicate in {
        "deterministic_only": lambda row: row["origin"] == "deterministic"
        and not row["deterministic_overlap"],
        "agent_only": lambda row: row["origin"] == "agent_supplement",
        "overlap": lambda row: bool(row["deterministic_overlap"]),
    }.items():
        selected = [row for row in rows if predicate(row)]
        groups[group_name] = {
            "source_count": len(selected),
            "accepted_claim_ids": sorted(
                {
                    claim_id
                    for row in selected
                    for claim_id in row["accepted_claim_ids"]
                }
            ),
        }
    return {
        "proposed_count": plan.proposed_count,
        "source_plan_admitted_count": plan.accepted_count,
        "rejected_count": plan.rejected_count,
        "deterministic_overlap_count": plan.overlap_count,
        "sources": rows,
        "groups": groups,
    }


def _unified_coverage(result: UnifiedRunResult) -> UnifiedCoverage:
    """Project the self-contained unified audit into the root coverage view."""

    bundle = result.bundle
    capability_by_name = {
        item.capability: item for item in bundle.completeness.capabilities
    }
    cases = bundle.cases
    evidence = bundle.current_evidence

    def capability(
        name: str,
        *,
        planned: int,
        completed: int,
        required_default: bool,
    ) -> UnifiedCapabilityCoverage:
        dependency = capability_by_name.get(name)
        if dependency is None:
            return UnifiedCapabilityCoverage(
                status="failed",
                required=required_default,
                planned=planned,
                completed=completed,
                reasons=[f"missing_unified_capability:{name}"],
            )
        return UnifiedCapabilityCoverage(
            status=dependency.status,
            required=dependency.requirement_mode == "required",
            planned=planned if dependency.status != "not_required" else 0,
            completed=(
                completed if dependency.status != "not_required" else 0
            ),
            reasons=list(dependency.reason_codes),
        )

    git_dependencies = [
        item
        for current in evidence
        for item in current.dependencies
        if item.capability == "git_temporal_evidence"
        and item.requirement_mode != "not_required"
    ]
    temporal_dependencies = [
        item
        for current in evidence
        for item in current.dependencies
        if item.capability == "temporal_evidence_classification"
        and item.requirement_mode != "not_required"
    ]
    completed_statuses = {"complete", "bounded_no_match", "not_required"}
    canonical_count = len(bundle.canonical_claims)
    return UnifiedCoverage(
        candidate_grouping=capability(
            "candidate_grouping",
            planned=len(cases),
            completed=sum(item.grouping_status == "complete" for item in cases),
            required_default=True,
        ),
        current_evidence=capability(
            "current_evidence",
            planned=len(evidence),
            completed=sum(item.status == "complete" for item in evidence),
            required_default=True,
        ),
        git_temporal_evidence=capability(
            "git_temporal_evidence",
            planned=len(git_dependencies),
            completed=sum(
                item.status in completed_statuses for item in git_dependencies
            ),
            required_default=False,
        ),
        temporal_classification=capability(
            "temporal_evidence_classification",
            planned=len(temporal_dependencies),
            completed=sum(
                item.status in completed_statuses
                for item in temporal_dependencies
            ),
            required_default=False,
        ),
        case_verification=capability(
            "unified_case_verification",
            planned=len(cases),
            completed=sum(
                item.status in {"complete", "not_required"}
                for item in bundle.findings
            ),
            required_default=True,
        ),
        refinement=capability(
            "unified_refinement",
            planned=len(bundle.refinements),
            completed=sum(item.status == "complete" for item in bundle.refinements),
            required_default=False,
        ),
        canonical_policy=UnifiedCapabilityCoverage(
            status="complete",
            required=True,
            planned=len(cases),
            completed=len(cases),
        ),
        case_count=len(cases),
        canonical_claim_count=canonical_count,
        accepted_count=bundle.index.accepted_count,
        review_count=bundle.index.review_count,
        rejected_count=bundle.index.rejected_count,
        invalid_member_count=bundle.funnel.invalid_members,
        incomplete_reasons=list(bundle.completeness.reasons),
    )


def _coverage(
    *,
    discovery_complete: bool,
    units_complete: bool,
    extraction_complete: bool,
    grounding_complete: bool,
    explicit_verification: ExplicitVerificationBundle,
    observation: ObservationBundle | None,
    observation_enabled: bool,
    semantic_units: SemanticUnitBundle | None,
    current_source: CurrentSourceBundle | None,
    semantic_enabled: bool,
    semantic_required: bool,
    unified: UnifiedRunResult,
) -> Coverage:
    explicit_status = "complete" if discovery_complete else "partial"
    return Coverage(
        explicit=ExplicitCoverage(
            manifest=explicit_status,
            deterministic_discovery=explicit_status,
            agent_source_discovery="complete",
            source_plan="complete",
            eligible_units="complete" if units_complete else "partial",
            extraction="complete" if extraction_complete else "partial",
            grounding="complete" if grounding_complete else "partial",
            semantic_verification=explicit_verification.result.coverage.status,
            claim_semantics=(
                "complete"
                if grounding_complete
                and explicit_verification.result.coverage.status == "complete"
                else "partial"
            ),
            proposition_merge=(
                "complete"
                if grounding_complete
                and explicit_verification.result.coverage.status == "complete"
                else "partial"
            ),
            source_recall="unknown",
        ),
        observation=ObservationCoverage(
            mode="foundational" if observation_enabled else "not_run",
            overall=(
                observation.index.overall
                if observation is not None
                else "not_run"
            ),
            queries=(
                [
                    ObservationQueryCoverage(
                        name=item.name,
                        version=item.version,
                        required=item.required,
                        status=item.status,
                        incomplete_reasons=item.incomplete_reasons,
                    )
                    for item in observation.index.query_plan
                ]
                if observation is not None
                else []
            ),
            observation_set_digest=(
                observation.index.observation_set_digest
                if observation is not None
                else None
            ),
        ),
        current_source=CurrentSourceCoverage(
            semantic_units=SemanticUnitCoverage(
                required=semantic_required,
                status=(
                    semantic_units.index.status
                    if semantic_units is not None
                    else "not_run"
                ),
                planned=(
                    semantic_units.index.planned_primary_count
                    if semantic_units is not None
                    else 0
                ),
                succeeded=(
                    semantic_units.index.ready_count
                    if semantic_units is not None
                    else 0
                ),
                no_input=(
                    semantic_units.index.no_input_count
                    if semantic_units is not None
                    else 0
                ),
                partial=(
                    semantic_units.index.partial_count
                    if semantic_units is not None
                    else 0
                ),
                failed=(
                    semantic_units.index.failed_count
                    if semantic_units is not None
                    else 0
                ),
                omitted=(
                    semantic_units.index.omitted_count
                    if semantic_units is not None
                    else 0
                ),
                reasons=(
                    semantic_units.index.incomplete_reasons
                    if semantic_units is not None
                    else []
                ),
            ),
            semantic_map=SemanticMapCoverage(
                required=semantic_required if semantic_enabled else False,
                status=(
                    current_source.semantic_map.index.status
                    if current_source is not None
                    else "not_run"
                ),
                planned=(
                    current_source.semantic_map.index.packet_count
                    if current_source is not None
                    else 0
                ),
                succeeded=(
                    current_source.semantic_map.index.packet_success
                    if current_source is not None
                    else 0
                ),
                no_observation=(
                    current_source.semantic_map.index.packet_no_observation
                    if current_source is not None
                    else 0
                ),
                uncertain=(
                    current_source.semantic_map.index.packet_uncertain
                    if current_source is not None
                    else 0
                ),
                failed=(
                    current_source.semantic_map.index.packet_failure
                    if current_source is not None
                    else 0
                ),
                omitted=0,
                reasons=(
                    current_source.semantic_map.index.incomplete_reasons
                    if current_source is not None
                    else []
                ),
            ),
            hypothesis_reduction=HypothesisReductionCoverage(
                required=semantic_required if semantic_enabled else False,
                status=(
                    current_source.reduction.index.status
                    if current_source is not None
                    else "not_run"
                ),
                planned=(
                    current_source.reduction.index.planned_calls
                    if current_source is not None
                    else 0
                ),
                succeeded=(
                    current_source.reduction.index.succeeded_calls
                    if current_source is not None
                    else 0
                ),
                no_hypothesis=(
                    current_source.reduction.index.no_hypothesis_calls
                    if current_source is not None
                    else 0
                ),
                uncertain=(
                    current_source.reduction.index.uncertain_calls
                    if current_source is not None
                    else 0
                ),
                failed=(
                    current_source.reduction.index.failed_calls
                    if current_source is not None
                    else 0
                ),
                reasons=(
                    current_source.reduction.index.incomplete_reasons
                    if current_source is not None
                    else []
                ),
            ),
            population_verification=PopulationVerificationCoverage(
                required=semantic_required if semantic_enabled else False,
                status=(
                    "complete"
                    if current_source is not None
                    and current_source.index.status == "complete"
                    else (
                        "partial"
                        if current_source is not None
                        else "not_run"
                    )
                ),
                planned_hypotheses=(
                    current_source.index.complete_populations
                    + current_source.index.partial_populations
                    + current_source.index.failed_populations
                    if current_source is not None
                    else 0
                ),
                complete_populations=(
                    current_source.index.complete_populations
                    if current_source is not None
                    else 0
                ),
                partial_populations=(
                    current_source.index.partial_populations
                    if current_source is not None
                    else 0
                ),
                failed_populations=(
                    current_source.index.failed_populations
                    if current_source is not None
                    else 0
                ),
                classified_members=(
                    current_source.index.classified_members
                    if current_source is not None
                    else 0
                ),
                matched_members=(
                    current_source.index.population_matched_members
                    if current_source is not None
                    else 0
                ),
                inspected_members=(
                    current_source.index.population_inspected_members
                    if current_source is not None
                    else 0
                ),
                uninspected_members=(
                    current_source.index.population_uninspected_members
                    if current_source is not None
                    else 0
                ),
                census_populations=(
                    current_source.index.census_populations
                    if current_source is not None
                    else 0
                ),
                sampled_populations=(
                    current_source.index.sampled_populations
                    if current_source is not None
                    else 0
                ),
                technical_unknown=(
                    current_source.index.technical_unknown_members
                    if current_source is not None
                    else 0
                ),
                reasons=(
                    current_source.index.incomplete_reasons
                    if current_source is not None
                    else []
                ),
            ),
            falsification=FalsificationCoverage(
                required=semantic_required if semantic_enabled else False,
                status=(
                    "complete"
                    if current_source is not None
                    and current_source.index.status == "complete"
                    else (
                        "partial"
                        if current_source is not None
                        else "not_run"
                    )
                ),
                planned_hypotheses=(
                    current_source.index.hypothesis_versions
                    if current_source is not None
                    else 0
                ),
                completed=(
                    sum(
                        item.status == "complete"
                        and item.verdict
                        not in {
                            "alternative_non_norm_explanation",
                            "reject_insufficient",
                            "uncertain",
                        }
                        and item.hypothesis_id
                        not in {
                            refinement.parent_hypothesis_id
                            for refinement in current_source.refinements
                        }
                        for item in current_source.falsifications
                    )
                    if current_source is not None
                    else 0
                ),
                refined=(
                    len(current_source.refinements)
                    if current_source is not None
                    else 0
                ),
                rejected=(
                    sum(
                        item.status == "complete"
                        and item.verdict
                        in {
                            "alternative_non_norm_explanation",
                            "reject_insufficient",
                        }
                        for item in current_source.falsifications
                    )
                    if current_source is not None
                    else 0
                ),
                uncertain=(
                    sum(
                        item.status == "complete"
                        and item.verdict == "uncertain"
                        for item in current_source.falsifications
                    )
                    if current_source is not None
                    else 0
                ),
                failed=(
                    sum(
                        item.status != "complete"
                        for item in current_source.falsifications
                    )
                    + (
                        current_source.index.hypothesis_versions
                        - len(current_source.falsifications)
                    )
                    if current_source is not None
                    else 0
                ),
                reasons=(
                    current_source.index.incomplete_reasons
                    if current_source is not None
                    else []
                ),
            ),
        ),
        unified=_unified_coverage(unified),
        delivery=DeliveryCoverage(
            scope_audit="complete",
            risk_audit="complete",
        ),
    )


def build_memory(
    repository_root: Path,
    config: BuildConfig,
    acquisition: SourceAcquisitionBundle,
    client: LLMClient,
    output_directory: Path,
    *,
    observation_cache: Path | None = None,
    inference_cache: Path | None = None,
    resume_from: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
    consumer_profiles: Sequence[str] = ("accepted_only",),
) -> BuildResult:
    """Build one immutable run without executing target-repository content."""

    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(
            f"output directory is not empty; refusing to overwrite: {output_directory}"
        )
    if output_directory.is_symlink():
        raise ValueError("generated run directory cannot be a symlink")
    resolved_output = output_directory.resolve()
    resolved_root = repository_root.resolve()
    if (
        resolved_output == resolved_root
        or resolved_output in resolved_root.parents
        or resolved_root in resolved_output.parents
    ):
        raise ValueError("generated run directory must be outside the target repository")
    resolved_acquisition = acquisition.directory.resolve()
    if (
        resolved_acquisition == resolved_root
        or resolved_acquisition in resolved_root.parents
        or resolved_root in resolved_acquisition.parents
    ):
        raise ValueError("source acquisition must be outside the target repository")
    if (
        resolved_output == resolved_acquisition
        or resolved_output in resolved_acquisition.parents
        or resolved_acquisition in resolved_output.parents
    ):
        raise ValueError("generated run directory must be outside source acquisition")
    for label, path in (
        ("observation cache", observation_cache),
        ("inference cache", inference_cache),
    ):
        if path is None:
            continue
        resolved_path = path.resolve()
        if (
            resolved_path == resolved_output
            or resolved_output in resolved_path.parents
            or resolved_path in resolved_output.parents
        ):
            raise ValueError(f"{label} must be outside the generated run directory")
        if (
            resolved_path == resolved_root
            or resolved_path in resolved_root.parents
            or resolved_root in resolved_path.parents
        ):
            raise ValueError(f"{label} must be outside the target repository")
        if (
            resolved_path == resolved_acquisition
            or resolved_path in resolved_acquisition.parents
            or resolved_acquisition in resolved_path.parents
        ):
            raise ValueError(f"{label} must be outside source acquisition")
    if resume_from is not None:
        resolved_resume = resume_from.resolve()
        if (
            resolved_resume == resolved_output
            or resolved_output in resolved_resume.parents
            or resolved_resume in resolved_output.parents
        ):
            raise ValueError(
                "resume run must be outside the generated run directory"
            )
        if (
            resolved_resume == resolved_root
            or resolved_resume in resolved_root.parents
            or resolved_root in resolved_resume.parents
        ):
            raise ValueError("resume run must be outside the target repository")
        if (
            resolved_resume == resolved_acquisition
            or resolved_resume in resolved_acquisition.parents
            or resolved_acquisition in resolved_resume.parents
        ):
            raise ValueError("resume run must be outside source acquisition")
    output_preflight = consume_initial_empty_directory(resolved_output)
    validate_acquisition_for_build(acquisition, config)
    discovery = acquisition.discovery
    supplement = acquisition.supplement
    current_discovery = _verify_discovery(resolved_root, config, discovery)
    plan = validate_source_plan(current_discovery, supplement, config.discovery)
    protocol_manifest = protocol_manifest_for_config(
        config,
        agent_profile_digest=acquisition.acquisition.agent_profile_sha256,
    )

    sources = [
        load_source_file(resolved_root, metadata, origin=origin)
        for metadata, origin in plan.sources
    ]
    sources_by_path = {source.path: source for source in sources}
    all_units = chunk_sources(
        sources,
        max_unit_chars=config.chunking.max_unit_chars,
    )
    units_by_id = {unit.unit_id: unit for unit in all_units}
    eligible_units: list[EvidenceUnit] = []
    dropped_unit_ids: set[str] = set()
    for unit in all_units:
        if len(unit.content) > config.chunking.max_batch_chars:
            dropped_unit_ids.add(unit.unit_id)
        else:
            eligible_units.append(unit)
    if len(eligible_units) > config.chunking.max_units:
        dropped_unit_ids.update(
            item.unit_id for item in eligible_units[config.chunking.max_units :]
        )
        eligible_units = eligible_units[: config.chunking.max_units]
    batches = batch_units(
        eligible_units,
        max_batch_chars=config.chunking.max_batch_chars,
        max_units_per_batch=config.chunking.max_units_per_batch,
    )
    if len(batches) > config.chunking.max_batches:
        for batch in batches[config.chunking.max_batches :]:
            dropped_unit_ids.update(unit.unit_id for unit in batch.units)
        batches = batches[: config.chunking.max_batches]

    success_cache = (
        SuccessInferenceCache(
            inference_cache,
            resume_from=resume_from,
            repository_root=resolved_root,
            max_entry_bytes=config.llm.max_inference_artifact_bytes,
            max_resume_entries=config.llm.max_total_calls,
        )
        if inference_cache is not None or resume_from is not None
        else None
    )
    inference_budget = InferenceBudget(
        config.llm,
        cache=success_cache,
        progress_callback=progress_callback,
    )
    tracking_client = _TrackingClient(client)
    outcomes, wall_seconds, _explicit_peak_concurrency = _extract_batches(
        batches,
        sources_by_path=sources_by_path,
        client=tracking_client,
        config=config,
        snapshot_digest=current_discovery.repository.snapshot_digest,
        protocol_manifest=protocol_manifest,
        budget=inference_budget,
        progress_callback=progress_callback,
    )
    failed_unit_ids = {
        unit_id
        for outcome in outcomes
        for unit_id in outcome.failed_unit_ids
    }
    explicit_budget_exhausted = set(inference_budget.exhausted_reasons)
    results_by_unit = order_results_by_source_plan(
        {
            result.unit_id: result
            for outcome in outcomes
            for result in outcome.results
        },
        batches,
    )
    outcome_by_unit = {
        unit.unit_id: outcome
        for batch, outcome in zip(batches, outcomes)
        for unit in batch.units
    }
    inference_request_by_unit = {
        unit.unit_id: outcome.inference_record.request_id
        for batch, outcome in zip(batches, outcomes)
        for unit in batch.units
    }

    observation: ObservationBundle | None = None
    observation_processing: dict[str, object] = {}
    if config.observation.enabled:
        observation_result = run_observations(
            resolved_root,
            current_discovery,
            config.observation,
            cache_directory=observation_cache,
        )
        observation = observation_result.bundle
        observation_processing = observation_result.processing_metrics

    semantic_units: SemanticUnitBundle | None = None
    if config.semantic.enabled:
        if observation is None:
            raise ValueError("semantic units require an observation bundle")
        semantic_units = build_semantic_units(
            resolved_root,
            current_discovery,
            observation,
            config.semantic,
        )

    raw_candidates = [
        (result.unit_id, result.reason_code, candidate)
        for result in results_by_unit.values()
        for candidate in result.claims
    ]
    prepared_decisions = prepare_candidate_decisions(
        raw_candidates,
        units_by_id=units_by_id,
        sources_by_path=sources_by_path,
        manifest_paths={
            item.path for item in current_discovery.manifest.files
        },
        snapshot_digest=current_discovery.repository.snapshot_digest,
        inference_request_by_unit=inference_request_by_unit,
        _grounding_memo={
            key: value
            for outcome in outcomes
            for key, value in (outcome.grounding_memo or {}).items()
        },
    )
    explicit_verification = run_explicit_verification(
        candidates=_explicit_verification_inputs(prepared_decisions),
        snapshot_digest=current_discovery.repository.snapshot_digest,
        llm_settings=config.llm,
        client=tracking_client,
        budget=inference_budget,
        protocol_manifest_digest=protocol_manifest.protocol_manifest_digest,
    )
    _apply_explicit_verification(prepared_decisions, explicit_verification)
    adjudication = finalize_candidate_decisions(prepared_decisions)
    decisions_by_unit: dict[str, list[CandidateDecision]] = defaultdict(list)
    for decision in adjudication.decisions:
        decisions_by_unit[decision.unit_id].append(decision)

    current_source: CurrentSourceBundle | None = None
    if config.semantic.enabled:
        assert observation is not None
        assert semantic_units is not None
        current_source = run_current_source_pipeline(
            repository_root=resolved_root,
            discovery=current_discovery,
            observations=observation,
            semantic_units=semantic_units,
            semantic_settings=config.semantic,
            llm_settings=config.llm,
            client=tracking_client,
            budget=inference_budget,
            protocol_manifest_digest=(
                protocol_manifest.protocol_manifest_digest
            ),
            # Candidate discovery channels remain independent in schema 4.11.
            # Cross-channel conflicts and corroboration are handled only by
            # CandidateCase assembly in the unified layer.
            explicit_context=(),
        )
    peak_concurrency = tracking_client.peak_in_flight

    units_complete = not dropped_unit_ids
    extraction_complete = (
        units_complete
        and not failed_unit_ids
        and not explicit_budget_exhausted
    )
    grounding_complete = all(
        decision.raw_candidate is not None
        and len(decision.grounded) == len(decision.raw_candidate.evidence)
        for decision in adjudication.decisions
    )
    # Unified cross-channel conflict discovery depends on whether every
    # planned explicit unit produced a complete structured result, not on
    # whether every *known* candidate later passed grounding or semantic
    # verification.  Those candidate-local failures remain visible in their
    # assessments and in root package coverage without poisoning unrelated
    # implicit CandidateCases.
    explicit_candidate_enumeration_complete = (
        current_discovery.complete
        and units_complete
        and not failed_unit_ids
    )
    unified_upstream_inputs = make_unified_upstream_inputs(
        snapshot_digest=current_discovery.repository.snapshot_digest,
        profile_digest=protocol_manifest.protocol_manifest_digest,
        snapshot_commit=current_discovery.repository.git_head_for_reference,
        explicit_decisions=adjudication.decisions,
        explicit_verification=explicit_verification.result,
        current_source=current_source,
        observations=observation,
        explicit_discovery_complete=(
            explicit_candidate_enumeration_complete
        ),
        raw_candidate_events=(
            len(raw_candidates)
            + (current_source.index.candidate_events if current_source else 0)
        ),
    )
    unified = run_unified_layer(
        repository_root=resolved_root,
        snapshot_digest=current_discovery.repository.snapshot_digest,
        snapshot_commit=current_discovery.repository.git_head_for_reference,
        profile_digest=protocol_manifest.protocol_manifest_digest,
        settings=config.unified,
        llm_settings=config.llm,
        client=tracking_client,
        budget=inference_budget,
        protocol_manifest_digest=protocol_manifest.protocol_manifest_digest,
        explicit_decisions=adjudication.decisions,
        explicit_verification=explicit_verification.result,
        current_source=current_source,
        observations=observation,
        explicit_discovery_complete=(
            explicit_candidate_enumeration_complete
        ),
        current_source_required=(
            config.semantic.enabled and config.semantic.required
        ),
        raw_candidate_events=unified_upstream_inputs.raw_candidate_events,
    )
    peak_concurrency = tracking_client.peak_in_flight

    reviews: list[ReviewItem] = []
    for decision in adjudication.decisions:
        if decision.outcome != "accepted":
            reviews.append(
                _review_item(
                    decision,
                    units_by_id[decision.unit_id],
                    inference_request_by_unit.get(decision.unit_id),
                )
            )
    for unit_id, result in results_by_unit.items():
        if result.classification != "uncertain" or result.claims:
            continue
        unit = units_by_id[unit_id]
        empty_decision = CandidateDecision(
            unit_id=unit_id,
            candidate_fingerprint=candidate_fingerprint(unit_id, None),
            outcome="review",
            outcome_stage="semantics",
            reason_code=result.reason_code,
            raw_candidate=None,
            errors=["model returned a bounded uncertain abstention"],
        )
        decisions_by_unit[unit_id].append(empty_decision)
        reviews.append(
            _review_item(
                empty_decision,
                unit,
                inference_request_by_unit.get(unit_id),
            )
        )
    if current_source is not None:
        reviews.extend(current_source.review_items)

    audit_records: list[AuditRecord] = []
    for unit in all_units:
        result = results_by_unit.get(unit.unit_id)
        decisions = decisions_by_unit.get(unit.unit_id, [])
        if unit.unit_id in dropped_unit_ids:
            classification = "not_sent"
            outcome = "failed"
            reason = "resource_or_safety_deferred"
            errors = ["unit was not sent because a bounded resource/safety limit applied"]
        elif unit.unit_id in failed_unit_ids or result is None:
            classification = "failed"
            outcome = "failed"
            reason = "extraction_failed"
            extraction_outcome = outcome_by_unit.get(unit.unit_id)
            errors = list(extraction_outcome.errors) if extraction_outcome else [
                "no extraction result"
            ]
        elif result.classification == "no_claims":
            classification = result.classification
            outcome = "rejected"
            reason = result.reason_code
            errors = []
        elif result.classification == "uncertain":
            classification = result.classification
            outcome = "review"
            reason = result.reason_code
            errors = [
                error for decision in decisions for error in decision.errors
            ]
        else:
            classification = result.classification
            outcome = (
                "review"
                if any(item.outcome != "accepted" for item in decisions)
                else "accepted"
            )
            reason = result.reason_code
            errors = [
                error
                for decision in decisions
                if decision.outcome != "accepted"
                for error in decision.errors
            ]
        extraction_outcome = outcome_by_unit.get(unit.unit_id)
        audit_records.append(
            AuditRecord(
                unit_id=unit.unit_id,
                path=unit.path,
                line_start=unit.line_start,
                line_end=unit.line_end,
                discovery_origin=unit.discovery_origin,
                inference_request_id=inference_request_by_unit.get(unit.unit_id),
                model_classification=classification,  # type: ignore[arg-type]
                local_outcome=outcome,  # type: ignore[arg-type]
                reason_code=reason,
                claim_ids=sorted(
                    {
                        decision.claim_id
                        for decision in decisions
                        if decision.claim_id
                    }
                ),
                validation_errors=sorted(set(errors)),
                attempts=(
                    _attempt_audit(extraction_outcome)
                    if extraction_outcome is not None
                    else []
                ),
                items=[
                    CandidateAuditItem(
                        candidate_fingerprint=decision.candidate_fingerprint,
                        verification_request_id=(
                            decision.verification_request_id
                        ),
                        verification_reason_codes=(
                            decision.verification_reason_codes
                        ),
                        verification_technical_reason_code=(
                            decision.verification_technical_reason_code
                        ),
                        local_outcome=decision.outcome,  # type: ignore[arg-type]
                        outcome_stage=decision.outcome_stage,  # type: ignore[arg-type]
                        reason_code=decision.reason_code,
                        rule=(
                            decision.raw_candidate.rule
                            if decision.raw_candidate
                            else None
                        ),
                        claim_id=decision.claim_id,
                        validation_errors=sorted(set(decision.errors)),
                    )
                    for decision in decisions
                ],
            )
        )

    coverage = _coverage(
        discovery_complete=current_discovery.complete,
        units_complete=units_complete,
        extraction_complete=extraction_complete,
        grounding_complete=grounding_complete,
        explicit_verification=explicit_verification,
        observation=observation,
        observation_enabled=config.observation.enabled,
        semantic_units=semantic_units,
        current_source=current_source,
        semantic_enabled=config.semantic.enabled,
        semantic_required=config.semantic.required,
        unified=unified,
    )
    memory = MemoryFile(
        schema_version=SCHEMA_VERSION,
        repository=RepositoryMetadata(
            **current_discovery.repository.model_dump(mode="python")
        ),
        language_scope="python",
        coverage=coverage,
        claims=list(adjudication.claims),
    )
    review = ReviewFile(
        schema_version=SCHEMA_VERSION,
        snapshot_digest=memory.repository.snapshot_digest,
        items=sorted(
            reviews,
            key=lambda item: (
                item.review_kind,
                getattr(item, "path", ""),
                getattr(item, "candidate_fingerprint", ""),
                getattr(item, "trace_id", ""),
            ),
        ),
    )
    # The canonical completeness artifact is the package-level truth exposed
    # to both embedded and standalone consumers.  Component-specific details
    # remain losslessly available in ``coverage`` and the channel/audit
    # artifacts; the build-report summary must not assign a second, wider set
    # of reason labels to the same package status.
    complete = unified.bundle.completeness.package_status == "complete"
    attempts = [
        attempt for outcome in outcomes for attempt in outcome.attempts
    ]
    inference_records = tuple(
        [
            *(outcome.inference_record for outcome in outcomes),
            *explicit_verification.inference_records,
            *(current_source.inference_records if current_source else ()),
            *unified.inference_records,
        ]
    )
    inference_budget.finalize_usage_overrun()
    inference_hash_by_id: dict[str, str] = {}
    inference_artifact_bytes = 0
    for item in inference_records:
        serialized = stable_json_text(item).encode("utf-8")
        inference_hash_by_id[item.request_id] = sha256_hex(serialized)
        inference_artifact_bytes += len(serialized)
    inference_index = make_inference_index(
        snapshot_digest=memory.repository.snapshot_digest,
        protocol_manifest=protocol_manifest,
        records=inference_records,
        artifact_sha256_by_id=inference_hash_by_id,
        total_artifact_bytes=inference_artifact_bytes,
        incomplete_reasons=sorted(inference_budget.exhausted_reasons),
        stage_omissions=inference_budget.stage_omissions,
    )
    source_funnel = _source_funnel(
        plan,
        all_units,
        results_by_unit,
        adjudication,
        failed_unit_ids | dropped_unit_ids,
    )
    raw_event_outcome_counts, unique_root_outcome_counts = (
        _explicit_outcome_counts(adjudication.decisions)
    )
    no_claim_count = sum(
        result.classification == "no_claims"
        for result in results_by_unit.values()
    )
    (
        explicit_traces,
        explicit_event_traces,
        explicit_trace_channel,
    ) = build_explicit_channel(
        snapshot_digest=memory.repository.snapshot_digest,
        decisions=adjudication.decisions,
        extraction_complete=(
            extraction_complete
            and grounding_complete
            and explicit_verification.result.coverage.status == "complete"
        ),
    )
    traceability = make_traceability(
        snapshot_digest=memory.repository.snapshot_digest,
        explicit_traces=explicit_traces,
        explicit_events=explicit_event_traces,
        explicit_channel=explicit_trace_channel,
        current_traces=current_source.traces if current_source else (),
        current_events=current_source.event_traces if current_source else (),
        current_channel=(
            current_source.trace_channel
            if current_source is not None
            else not_run_channel("disabled_by_config")
        ),
    )
    canonical_claims = list(unified.bundle.canonical_claims)
    accepted_claims = [
        item for item in canonical_claims if item.epistemic_status == "accepted"
    ]
    delivered_claims = [
        item
        for item in accepted_claims
        if item.delivery_policy != "reference_only"
    ]
    build_report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "entrypoint": "full",
        "snapshot_digest": memory.repository.snapshot_digest,
        "repository": memory.repository.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
        "versions": {
            "chunker": CHUNKER_VERSION,
            "prompt": PROMPT_VERSION,
            "explicit_response_schema": RESPONSE_SCHEMA_VERSION,
            "explicit_verification_protocol": (
                EXPLICIT_VERIFICATION_PROTOCOL_VERSION
            ),
            "explicit_verification_response_schema": (
                EXPLICIT_VERIFICATION_RESPONSE_SCHEMA_VERSION
            ),
            "explicit_verification_policy": (
                EXPLICIT_VERIFICATION_POLICY_VERSION
            ),
            "protocol_manifest": (
                protocol_manifest.protocol_manifest_digest
            ),
            "agent_profile": acquisition.acquisition.agent_profile_sha256,
            "observation_index": (
                observation.index.index_version if observation else None
            ),
            "unified_contract": unified.bundle.index.contract_version,
            "unified_policy": unified.bundle.index.policy_version,
        },
        "discovery": {
            "complete": current_discovery.complete,
            "incomplete_reasons": current_discovery.incomplete_reasons,
            "manifest_file_count": len(current_discovery.manifest.files),
            "baseline_source_count": len(
                current_discovery.baseline_sources
            ),
            "final_source_count": len(sources),
            "final_sources": [
                {
                    "path": source.path,
                    "source_kind": source.source_kind,
                    "discovery_origin": source.discovery_origin,
                    "content_sha256": source.content_sha256,
                }
                for source in sorted(sources, key=lambda item: item.path)
            ],
        },
        "agent_source_acquisition": {
            "complete": acquisition.acquisition.complete,
            "target_content_admission_policy": (
                acquisition.acquisition.target_content_admission_policy
            ),
            "profile_id": acquisition.profile.profile_id,
            "profile_digest": acquisition.acquisition.agent_profile_sha256,
            "framework_id": acquisition.profile.framework_id,
            "framework_version": acquisition.profile.framework_version,
            "model_id": acquisition.profile.model_id,
            "request_id": acquisition.request.request_id,
            "run_status": acquisition.agent_run.status,
            "exit_status": acquisition.agent_run.exit_status,
            "duration_ms": acquisition.agent_run.duration_ms,
            "usage": acquisition.agent_run.usage.model_dump(mode="json"),
            "proposed_count": len(acquisition.supplement.proposals),
        },
        "execution_model": execution_model_run_summary(
            config,
            acquisition,
            inference_records,
        ),
        "source_funnel": source_funnel,
        "processing": {
            "unit_count": len(all_units),
            "eligible_unit_count": len(eligible_units),
            "dropped_unit_count": len(dropped_unit_ids),
            "batch_count": len(batches),
            "failed_unit_count": len(failed_unit_ids),
            "explicit_verification": (
                explicit_verification.result.coverage.model_dump(mode="json")
            ),
            "observation": observation_processing,
            "semantic_units": (
                semantic_units.index.resource_use
                if semantic_units is not None
                else {}
            ),
            "current_source": (
                {
                    "index_version": current_source.index.index_version,
                    "status": current_source.index.status,
                    "matched_members": (
                        current_source.index.population_matched_members
                    ),
                    "inspected_members": (
                        current_source.index.population_inspected_members
                    ),
                    "uninspected_members": (
                        current_source.index.population_uninspected_members
                    ),
                    "census_populations": (
                        current_source.index.census_populations
                    ),
                    "sampled_populations": (
                        current_source.index.sampled_populations
                    ),
                    "reducer_cancelled_stages": (
                        current_source.reduction.index.cancelled_stages
                    ),
                    "resource_use": current_source.index.resource_use,
                }
                if current_source is not None
                else {}
            ),
        },
        "adjudication_funnel": {
            "raw_candidate_events": len(raw_candidates),
            "unique_candidate_roots": len(
                {
                    item.candidate_fingerprint
                    for item in adjudication.decisions
                }
            ),
            "grounded_unique_candidate_roots": (
                explicit_verification.result.coverage.grounded_roots
            ),
            "verified_unique_candidate_roots": (
                explicit_verification.result.coverage.verified_roots
            ),
            "verification_eligible_candidate_roots": (
                explicit_verification.result.coverage.policy_eligible
            ),
            "verification_review_candidate_roots": (
                explicit_verification.result.coverage.policy_review
            ),
            "verification_rejected_candidate_roots": (
                explicit_verification.result.coverage.policy_rejected
            ),
            "post_host_accepted_candidate_roots": (
                unique_root_outcome_counts["accepted"]
            ),
            "post_host_review_candidate_roots": (
                unique_root_outcome_counts["review"]
            ),
            "post_host_rejected_candidate_roots": (
                unique_root_outcome_counts["rejected"]
            ),
            "merged_accepted_propositions": len(accepted_claims),
            "no_claim_units": no_claim_count,
            "delivered_propositions": len(delivered_claims),
        },
        "channel_funnels": {
            "explicit": traceability.explicit.model_dump(mode="json"),
            "current_source": traceability.current_source.model_dump(mode="json"),
        },
        "outcomes": {
            "raw_candidate_events": {
                "accepted": raw_event_outcome_counts["accepted"],
                "review": raw_event_outcome_counts["review"],
                "rejected": raw_event_outcome_counts["rejected"],
                "total": sum(raw_event_outcome_counts.values()),
            },
            "candidate_roots": {
                "accepted": unique_root_outcome_counts["accepted"],
                "review": unique_root_outcome_counts["review"],
                "rejected": unique_root_outcome_counts["rejected"],
                "total": sum(unique_root_outcome_counts.values()),
            },
            "merged_propositions": {
                "accepted": unified.bundle.index.accepted_count,
                "review": unified.bundle.index.review_count,
                "rejected": unified.bundle.index.rejected_count,
                "delivered": len(delivered_claims),
            },
            "unit_classifications": {
                "no_claims": no_claim_count,
                "uncertain": sum(
                    result.classification == "uncertain"
                    for result in results_by_unit.values()
                ),
                "failed_or_not_sent": len(failed_unit_ids | dropped_unit_ids),
            },
            "exported_review_items": sum(
                item.epistemic_status in {"review", "rejected"}
                for item in canonical_claims
            ),
        },
        "scope_audit": {
            "global": sum(
                item.semantics.applies_to == ["**"] for item in canonical_claims
            ),
            "local": sum(
                item.semantics.applies_to != ["**"] for item in canonical_claims
            ),
            "origin_channels": sorted(
                {item.channel for item in unified.bundle.assessments}
            ),
        },
        "unified": {
            "bundle_id": unified.bundle.index.bundle_id,
            "profile_digest": unified.bundle.index.profile_digest,
            "case_count": len(unified.bundle.cases),
            "assessment_count": len(unified.bundle.assessments),
            "case_routes": dict(unified.case_routes),
            "candidate_grouping": (
                unified.grouping_execution.report_metadata()
            ),
            "git_case_count": sum(
                bool(item.manifest.commands) for item in unified.git_bundles
            ),
            "git_command_count": sum(
                len(item.manifest.command_results) for item in unified.git_bundles
            ),
            "temporal_matrix_count": len(unified.bundle.temporal_matrices),
            "finding_count": len(unified.bundle.findings),
            "refinement_count": len(unified.bundle.refinements),
            "canonical_counts": {
                "accepted": unified.bundle.index.accepted_count,
                "review": unified.bundle.index.review_count,
                "rejected": unified.bundle.index.rejected_count,
                "delivered": len(delivered_claims),
            },
            "completeness": unified.bundle.completeness.model_dump(mode="json"),
        },
        "llm": {
            "batch_wall_seconds": round(wall_seconds, 6),
            "observed_max_concurrency": peak_concurrency,
            "attempt_count": sum(len(item.attempts) for item in inference_records),
            "schema_repairs": sum(
                attempt.repair
                for record in inference_records
                for attempt in record.attempts
            ),
            "provider_retries": sum(
                attempt.provider_try > 0
                for record in inference_records
                for attempt in record.attempts
            ),
            "output_limit_retries": sum(
                attempt.output_limit_retry > 0
                for record in inference_records
                for attempt in record.attempts
            ),
            "input_tokens": sum(
                item.usage.input_tokens for item in inference_records
            ),
            "output_tokens": sum(
                item.usage.output_tokens for item in inference_records
            ),
            "reasoning_tokens": (
                sum(
                    item.usage.reasoning_tokens or 0
                    for item in inference_records
                )
                if all(
                    item.usage.reasoning_tokens is not None
                    for item in inference_records
                )
                else None
            ),
            "record_count": len(inference_records),
            "inference_set_digest": inference_index.inference_set_digest,
            "this_run_provider_calls": inference_budget.calls,
            "this_run_cache_hits": inference_budget.cache_hits,
            "stage_progress": inference_budget.stage_progress,
            "cache": (
                {
                    "hits": success_cache.metrics.hits,
                    "misses": success_cache.metrics.misses,
                    "rejected": success_cache.metrics.rejected,
                    "writes": success_cache.metrics.writes,
                    "resume_entries": success_cache.metrics.resume_entries,
                }
                if success_cache is not None
                else {
                    "hits": 0,
                    "misses": 0,
                    "rejected": 0,
                    "writes": 0,
                    "resume_entries": 0,
                }
            ),
        },
        "coverage": coverage.model_dump(mode="json"),
        "complete": complete,
        "incomplete_reasons": sorted(unified.bundle.completeness.reasons),
    }
    # A build can spend substantial time in provider and observation stages.
    # Refuse to publish if the target moved after the initial acquisition check.
    _verify_discovery(resolved_root, config, discovery)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    def write_channels(writer: ChannelArtifactWriter) -> None:
        write_build_outputs(
            writer.root,
            snapshot_digest=memory.repository.snapshot_digest,
            audit_records=audit_records,
            explicit_verification=explicit_verification,
            evidence_relations=adjudication.evidence_relations,
            observation=observation,
            semantic_units=semantic_units,
            current_source=current_source,
            traceability=traceability,
            inference_records=inference_records,
            source_acquisition=acquisition,
            config=config,
            writer=writer,
        )
    output_paths = write_unified_outputs(
        resolved_output,
        progress_callback=progress_callback,
        bundle=unified.bundle,
        git_artifacts=unified.git_artifacts,
        max_artifact_bytes=config.unified.max_unified_artifact_bytes,
        repository_root=resolved_root,
        repository_name=config.repository.name,
        consumer_profiles=consumer_profiles,
        channel_writer=write_channels,
        protocol_manifest=protocol_manifest,
        inference_records=inference_records,
        inference_incomplete_reasons=sorted(inference_budget.exhausted_reasons),
        inference_stage_omissions=inference_budget.stage_omissions,
        upstream_inputs=unified_upstream_inputs,
        build_report=build_report,
        _destination_preflight=output_preflight,
        _prepublication_check=lambda: _verify_discovery(
            resolved_root,
            config,
            discovery,
        ),
    )
    return BuildResult(
        source_acquisition=acquisition,
        memory=memory,
        build_report=build_report,
        audit_records=tuple(audit_records),
        review=review,
        explicit_verification=explicit_verification,
        observation=observation,
        semantic_units=semantic_units,
        current_source=current_source,
        upstream_inputs=unified_upstream_inputs,
        unified=unified,
        traceability=traceability,
        inference_index=inference_index,
        inference_records=inference_records,
        protocol_manifest=protocol_manifest,
        output_paths=output_paths,
    )
