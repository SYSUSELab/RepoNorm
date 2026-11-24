"""Strict schemas for agent-assisted fixed-snapshot norm acquisition."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from repo_constraint_memory_v3.evidence_validation import (
    MAX_LOGICAL_EVIDENCE_LINES,
)
from repo_constraint_memory_v3.safety import (
    TARGET_CONTENT_ADMISSION_POLICY_VERSION,
)

SCHEMA_VERSION = "4.11-partition-only-required-evidence"
SOURCE_PLAN_SCHEMA_VERSION = "1.0"
SPAN_VERSION = "physical-lines-1.0"
MAX_AGENT_OUTPUT_BYTES = 10_000_000
MAX_AGENT_TRAJECTORY_BYTES = 50_000_000

NonEmptyString = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(ge=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SnapshotDigest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
UnitId = Annotated[str, Field(pattern=r"^UNIT-[0-9a-f]{24}$")]
SpanId = Annotated[str, Field(pattern=r"^SPAN-[0-9a-f]{24}$")]
ClaimId = Annotated[str, Field(pattern=r"^RCM3-[0-9a-f]{24}$")]

SourceKind = Literal[
    "instruction",
    "readme",
    "documentation",
    "configuration",
    "command",
    "workflow",
]
DiscoveryOrigin = Literal["deterministic", "agent_supplement"]
AdapterKind = Literal[
    "prose",
    "toml",
    "ini_cfg",
    "yaml_json",
    "make_just_task",
    "workflow",
]
EvidenceType = Literal[
    "direct_declaration",
    "configuration_entry",
    "command_definition",
    "workflow_step",
]
ClaimKind = Literal[
    "normative_declaration",
    "enforced_configuration",
    "operational_fact",
    "command_recipe",
    "workflow_fact",
    "observed_convention",
]
NormativeStrength = Literal["required", "recommended", "observed"]
EpistemicStatus = Literal["accepted", "hypothesis", "review", "rejected"]
OriginChannel = Literal["explicit", "implicit_current"]
AcceptanceBasis = Literal[
    "direct_modal_text",
    "direct_instruction_text",
    "typed_repository_literal",
    "executable_enforcement",
]
DeliveryPolicy = Literal[
    "default_consider",
    "condition_matched_only",
    "reference_only",
]
ExecutionRisk = Literal[
    "read_only",
    "bounded_check",
    "scoped_mutation",
    "repository_wide_mutation",
    "external_side_effect",
]
EvidenceRelationKind = Literal[
    "corroborates",
    "contradicts",
    "refines_scope",
    "possible_duplicate",
]
CapabilityStatus = Literal["complete", "partial", "failed", "not_run"]
OutcomeStage = Literal[
    "parse",
    "grounding",
    "semantics",
    "explicit_semantic_verify",
    "merge",
    "delivery",
]
LocalOutcome = Literal["accepted", "review", "rejected", "failed"]


class StrictModel(BaseModel):
    """Reject unknown fields and type coercion at every trust boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)


class RepositorySettings(StrictModel):
    name: str | None = None

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("repository.name must be null or non-blank")
        return value


class DiscoverySettings(StrictModel):
    target_content_admission_policy: Literal[
        "controlled-research-unredacted-target-content-1.0"
    ] = TARGET_CONTENT_ADMISSION_POLICY_VERSION
    include_linked_docs_one_hop: bool = True
    max_manifest_files: Annotated[int, Field(ge=1, le=1_000_000)] = 25_000
    max_manifest_file_bytes: PositiveInt = 2_000_000
    max_manifest_total_bytes: PositiveInt = 50_000_000
    max_source_files: Annotated[int, Field(ge=1, le=10_000)] = 300
    max_source_file_bytes: PositiveInt = 1_000_000
    max_total_source_bytes: PositiveInt = 10_000_000
    max_supplement_proposals: Annotated[int, Field(ge=0, le=1_000)] = 64
    exclude: list[NonEmptyString] = Field(default_factory=list)


class AgentStageSettings(StrictModel):
    """Host-owned bounds for one interactive agent stage."""

    enabled: bool = True
    required: bool = True
    timeout_seconds: Annotated[int, Field(ge=1, le=86_400)] = 900
    max_turns: Annotated[int, Field(ge=1, le=1_000)] = 30
    max_workspace_files: Annotated[int, Field(ge=1, le=100_000)] = 10_000
    max_workspace_bytes: Annotated[
        int, Field(ge=1_000, le=2_000_000_000)
    ] = 50_000_000
    max_output_bytes: Annotated[
        int, Field(ge=100, le=MAX_AGENT_OUTPUT_BYTES)
    ] = 200_000
    max_trajectory_bytes: Annotated[
        int, Field(ge=100, le=MAX_AGENT_TRAJECTORY_BYTES)
    ] = 20_000_000
    max_input_tokens: Annotated[
        int, Field(ge=1, le=10_000_000_000)
    ] = 10_000_000
    max_output_tokens: Annotated[
        int, Field(ge=1, le=10_000_000_000)
    ] = 1_000_000
    cost_limit_usd: Annotated[float, Field(gt=0, le=10_000)] = 5.0

    @model_validator(mode="after")
    def enabled_required_contract(self) -> AgentStageSettings:
        if not self.enabled and self.required:
            raise ValueError("disabled agent stage cannot be required")
        return self


class AgentSettings(StrictModel):
    """Method settings shared by every agent call in one acquisition/build."""

    required: Literal[True] = True
    stages: dict[NonEmptyString, AgentStageSettings] = Field(
        default_factory=lambda: {
            "source_discovery": AgentStageSettings(),
        }
    )

    @model_validator(mode="after")
    def source_discovery_contract(self) -> AgentSettings:
        if set(self.stages) != {"source_discovery"}:
            raise ValueError(
                "schema 4 agent stages must be exactly source_discovery"
            )
        source = self.stages.get("source_discovery")
        if source is None:
            raise ValueError("agent stages require source_discovery")
        if not source.enabled or not source.required:
            raise ValueError(
                "required agent runtime requires enabled, required source_discovery"
            )
        return self


class ChunkingSettings(StrictModel):
    max_unit_chars: PositiveInt = 8_000
    max_units: Annotated[int, Field(ge=1, le=100_000)] = 5_000
    max_batch_chars: PositiveInt = 16_000
    max_units_per_batch: Annotated[int, Field(ge=1, le=100)] = 1
    max_batches: Annotated[int, Field(ge=1, le=10_000)] = 1_000

    @model_validator(mode="after")
    def batch_must_fit_units(self) -> ChunkingSettings:
        if self.max_batch_chars < self.max_unit_chars:
            raise ValueError("chunking.max_batch_chars must be >= max_unit_chars")
        return self


class LLMSettings(StrictModel):
    provider: Literal["openai_compatible"] = "openai_compatible"
    model: NonEmptyString
    require_response_model: Literal[True]
    accepted_effective_model_ids: Annotated[
        list[NonEmptyString], Field(min_length=1, max_length=16)
    ]
    api_key_env: NonEmptyString
    base_url: NonEmptyString
    temperature: int | float = 0
    thinking_mode: Literal["disabled", "enabled"] = "disabled"
    reasoning_effort: Literal[
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ] | None = None
    reasoning_max_tokens: Annotated[int, Field(ge=1, le=1_000_000)] | None = None
    reasoning_wire_policy: Literal[
        "deepseek_thinking_type",
        "qwen_enable_thinking",
        "openrouter_reasoning",
        "openai_responses_reasoning_effort",
        "offline_mock",
    ]
    max_output_tokens: PositiveInt = 16_000
    timeout_seconds: PositiveInt = 600
    # Transport-body guard applied while streaming, before JSON parsing.  The
    # default deliberately leaves wide headroom above a 64k-token completion;
    # stage-local parsed-content limits remain a separate semantic contract.
    max_provider_response_bytes: Annotated[
        int, Field(ge=1_000_000, le=1_000_000_000)
    ] = 32_000_000
    schema_repairs: Annotated[int, Field(ge=0, le=3)] = 1
    output_limit_retries: Annotated[int, Field(ge=0, le=2)] = 1
    max_concurrency: Annotated[int, Field(ge=1, le=32)] = 24
    provider_retries: Annotated[int, Field(ge=0, le=5)] = 2
    provider_retry_delay_seconds: Annotated[
        int | float,
        Field(ge=0, le=60),
    ] = 1.0
    provider_retry_backoff_multiplier: Annotated[
        int | float,
        Field(ge=1, le=10),
    ] = 2.0
    provider_retry_max_delay_seconds: Annotated[
        int | float,
        Field(ge=0, le=600),
    ] = 30.0
    provider_retry_jitter_seconds: Annotated[
        int | float,
        Field(ge=0, le=30),
    ] = 0.5
    max_total_calls: Annotated[int, Field(ge=1, le=1_000_000)] = 10_000
    max_total_input_tokens: Annotated[
        int, Field(ge=1, le=10_000_000_000)
    ] = 100_000_000
    max_total_output_tokens: Annotated[
        int, Field(ge=1, le=10_000_000_000)
    ] = 20_000_000
    max_inference_artifact_bytes: Annotated[
        int, Field(ge=1_000, le=1_000_000_000)
    ] = 20_000_000

    @field_validator("accepted_effective_model_ids")
    @classmethod
    def effective_model_ids_are_canonical(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("accepted effective model IDs must be unique")
        return sorted(value)

    @model_validator(mode="after")
    def explicit_reasoning_contract(self) -> LLMSettings:
        if self.thinking_mode == "disabled" and (
            self.reasoning_effort is not None
            or self.reasoning_max_tokens is not None
        ):
            raise ValueError("disabled thinking cannot carry a reasoning control")
        if (
            self.reasoning_effort is not None
            and self.reasoning_max_tokens is not None
        ):
            raise ValueError(
                "reasoning_effort and reasoning_max_tokens are mutually exclusive"
            )
        if (
            self.reasoning_max_tokens is not None
            and self.reasoning_wire_policy != "openrouter_reasoning"
        ):
            raise ValueError(
                "reasoning_max_tokens requires the OpenRouter reasoning wire"
            )
        if (
            self.thinking_mode == "enabled"
            and self.reasoning_wire_policy != "openrouter_reasoning"
            and self.reasoning_effort is None
        ):
            raise ValueError("enabled thinking requires reasoning_effort")
        return self

    @model_serializer(mode="wrap")
    def omit_unused_reasoning_max_tokens(self, handler):
        payload = handler(self)
        if self.reasoning_max_tokens is None:
            payload.pop("reasoning_max_tokens", None)
        return payload


class ObservationQuerySettings(StrictModel):
    name: NonEmptyString
    required: bool = True


class ObservationSettings(StrictModel):
    enabled: bool = True
    queries: list[ObservationQuerySettings] = Field(
        default_factory=lambda: [
            ObservationQuerySettings(name="python_snapshot", required=True),
            ObservationQuerySettings(
                name="python_clone_clusters",
                required=True,
            ),
        ]
    )
    max_python_files: Annotated[int, Field(ge=1, le=100_000)] = 5_000
    max_python_file_bytes: PositiveInt = 1_000_000
    max_total_python_bytes: PositiveInt = 20_000_000
    max_ast_nodes_per_file: Annotated[int, Field(ge=1, le=5_000_000)] = 200_000
    max_entities_per_file: Annotated[int, Field(ge=1, le=500_000)] = 10_000
    max_relations_per_file: Annotated[int, Field(ge=1, le=2_000_000)] = 50_000
    max_total_entities: Annotated[int, Field(ge=1, le=10_000_000)] = 500_000
    max_total_relations: Annotated[int, Field(ge=1, le=50_000_000)] = 2_000_000
    max_parse_failures: Annotated[int, Field(ge=1, le=100_000)] = 100
    max_symbol_chars: Annotated[int, Field(ge=16, le=4096)] = 512
    max_file_artifact_bytes: Annotated[
        int, Field(ge=1_000, le=1_000_000_000)
    ] = 5_000_000
    max_total_observation_bytes: Annotated[
        int, Field(ge=1_000, le=10_000_000_000)
    ] = 100_000_000
    max_cache_entry_bytes: Annotated[
        int, Field(ge=1_000, le=1_000_000_000)
    ] = 5_000_000
    max_workers: Annotated[int, Field(ge=1, le=32)] = 4

    @model_validator(mode="after")
    def enabled_query_contract(self) -> ObservationSettings:
        names = [item.name for item in self.queries]
        if self.enabled and not names:
            raise ValueError("enabled observation requires at least one query")
        if len(names) != len(set(names)):
            raise ValueError("observation queries must be unique")
        return self


class SemanticSettings(StrictModel):
    """Bounded resource contract for the complete current-source pipeline."""

    enabled: bool = True
    required: bool = True
    max_units: Annotated[int, Field(ge=1, le=5_000_000)] = 1_000
    max_unit_source_bytes: Annotated[
        int, Field(ge=1, le=100_000_000)
    ] = 200_000
    max_packet_bytes: Annotated[int, Field(ge=1, le=200_000_000)] = 300_000
    max_total_packet_bytes: Annotated[
        int, Field(ge=1, le=20_000_000_000)
    ] = 500_000_000
    max_context_refs_per_packet: Annotated[
        int, Field(ge=1, le=100_000)
    ] = 1_000
    max_map_packets_per_request: Annotated[int, Field(ge=1, le=10_000)] = 32
    max_map_request_bytes: Annotated[int, Field(ge=1, le=200_000_000)] = 2_000_000
    max_observations_per_primary_unit: Annotated[
        int, Field(ge=1, le=10_000)
    ] = 64
    max_total_observations: Annotated[int, Field(ge=1, le=5_000_000)] = 500
    max_reduce_items_per_request: Annotated[int, Field(ge=1, le=100_000)] = 64
    max_reduce_request_bytes: Annotated[int, Field(ge=1, le=200_000_000)] = 2_000_000
    max_hypotheses_per_component: Annotated[int, Field(ge=1, le=64)] = 64
    max_total_hypotheses: Annotated[int, Field(ge=1, le=1_000_000)] = 10_000
    max_population_query_nodes: Annotated[int, Field(ge=1, le=10_000)] = 256
    max_population_query_depth: Annotated[int, Field(ge=1, le=100)] = 16
    max_population_literal_values: Annotated[
        int, Field(ge=1, le=100_000)
    ] = 2_000
    max_population_members: Annotated[
        int, Field(ge=1, le=10_000_000)
    ] = 1_000_000
    max_population_members_per_hypothesis: Annotated[
        int, Field(ge=1, le=10_000_000)
    ] = 100_000
    max_total_population_members: Annotated[
        int, Field(ge=1, le=50_000_000)
    ] = 2_000_000
    # Reducers use the target as a soft selectivity objective.  Classification
    # freezes a deterministic sample when a complete semantic population is
    # wider than the hard inspected-member bounds; structural predicates stay
    # provider-free exact censuses.
    target_population_members_per_hypothesis: Annotated[
        int, Field(ge=1, le=10_000_000)
    ] = 15
    max_classified_population_members_per_hypothesis: Annotated[
        int, Field(ge=1, le=10_000_000)
    ] = 20
    max_total_classified_population_members: Annotated[
        int, Field(ge=1, le=50_000_000)
    ] = 1_000
    max_population_artifact_bytes: Annotated[
        int, Field(ge=1_000, le=1_000_000_000)
    ] = 50_000_000
    max_classification_members_per_request: Annotated[
        int, Field(ge=1, le=10_000)
    ] = 16
    max_classification_request_bytes: Annotated[
        int, Field(ge=1, le=200_000_000)
    ] = 2_000_000
    # Admission is based on the exact canonical requests produced after
    # packet construction.  These bounds are deliberately separate from the
    # global inference fuse: a single wide population must not consume the
    # repository-wide allowance before another candidate can run.
    max_classification_calls_per_hypothesis: Annotated[
        int, Field(ge=1, le=100_000)
    ] = 32
    max_classification_request_bytes_per_hypothesis: Annotated[
        int, Field(ge=1, le=2_000_000_000)
    ] = 16_000_000
    max_total_classification_calls: Annotated[
        int, Field(ge=1, le=1_000_000)
    ] = 256
    max_total_classification_request_bytes: Annotated[
        int, Field(ge=1, le=20_000_000_000)
    ] = 64_000_000
    max_member_packet_bytes: Annotated[
        int, Field(ge=1, le=100_000_000)
    ] = 250_000
    max_falsification_items_per_request: Annotated[
        int, Field(ge=1, le=10_000)
    ] = 16
    max_falsification_request_bytes: Annotated[
        int, Field(ge=1, le=200_000_000)
    ] = 2_000_000
    max_falsification_calls_per_hypothesis: Annotated[
        int, Field(ge=1, le=100_000)
    ] = 16
    max_falsification_request_bytes_per_hypothesis: Annotated[
        int, Field(ge=1, le=2_000_000_000)
    ] = 8_000_000
    max_total_falsification_calls: Annotated[
        int, Field(ge=1, le=1_000_000)
    ] = 128
    max_total_falsification_request_bytes: Annotated[
        int, Field(ge=1, le=20_000_000_000)
    ] = 32_000_000
    max_refinement_rounds: Annotated[int, Field(ge=0, le=2)] = 1
    max_current_source_artifact_bytes: Annotated[
        int, Field(ge=1_000, le=5_000_000_000)
    ] = 500_000_000

    @model_validator(mode="after")
    def resource_contract(self) -> SemanticSettings:
        if not self.enabled and self.required:
            raise ValueError("semantic.enabled false requires semantic.required false")
        if self.max_packet_bytes < self.max_unit_source_bytes:
            raise ValueError(
                "semantic.max_packet_bytes must be >= max_unit_source_bytes"
            )
        if self.max_map_request_bytes < self.max_packet_bytes:
            raise ValueError(
                "semantic.max_map_request_bytes must be >= max_packet_bytes"
            )
        if self.max_population_members_per_hypothesis > self.max_population_members:
            raise ValueError(
                "semantic.max_population_members_per_hypothesis must be "
                "<= max_population_members"
            )
        if self.target_population_members_per_hypothesis > (
            self.max_population_members_per_hypothesis
        ):
            raise ValueError(
                "semantic.target_population_members_per_hypothesis must be "
                "<= max_population_members_per_hypothesis"
            )
        if self.max_classified_population_members_per_hypothesis > (
            self.max_population_members_per_hypothesis
        ):
            raise ValueError(
                "semantic.max_classified_population_members_per_hypothesis "
                "must be <= max_population_members_per_hypothesis"
            )
        if self.max_total_classified_population_members < (
            self.max_classified_population_members_per_hypothesis
        ):
            raise ValueError(
                "semantic.max_total_classified_population_members must be >= "
                "max_classified_population_members_per_hypothesis"
            )
        if self.max_classification_request_bytes < self.max_member_packet_bytes:
            raise ValueError(
                "semantic.max_classification_request_bytes must be "
                ">= max_member_packet_bytes"
            )
        if (
            self.max_classification_request_bytes_per_hypothesis
            < self.max_classification_request_bytes
        ):
            raise ValueError(
                "semantic.max_classification_request_bytes_per_hypothesis "
                "must be >= max_classification_request_bytes"
            )
        if (
            self.max_total_classification_calls
            < self.max_classification_calls_per_hypothesis
        ):
            raise ValueError(
                "semantic.max_total_classification_calls must be >= "
                "max_classification_calls_per_hypothesis"
            )
        if (
            self.max_total_classification_request_bytes
            < self.max_classification_request_bytes_per_hypothesis
        ):
            raise ValueError(
                "semantic.max_total_classification_request_bytes must be >= "
                "max_classification_request_bytes_per_hypothesis"
            )
        if (
            self.max_falsification_request_bytes_per_hypothesis
            < self.max_falsification_request_bytes
        ):
            raise ValueError(
                "semantic.max_falsification_request_bytes_per_hypothesis "
                "must be >= max_falsification_request_bytes"
            )
        if (
            self.max_total_falsification_calls
            < self.max_falsification_calls_per_hypothesis
        ):
            raise ValueError(
                "semantic.max_total_falsification_calls must be >= "
                "max_falsification_calls_per_hypothesis"
            )
        if (
            self.max_total_falsification_request_bytes
            < self.max_falsification_request_bytes_per_hypothesis
        ):
            raise ValueError(
                "semantic.max_total_falsification_request_bytes must be >= "
                "max_falsification_request_bytes_per_hypothesis"
            )
        return self


class UnifiedSettings(StrictModel):
    """Frozen resource and policy profile for the canonical unified layer."""

    enabled: Literal[True] = True
    required: Literal[True] = True
    candidate_grouping_mode: Literal["semantic", "singleton"] = "semantic"
    max_candidate_members: Annotated[int, Field(ge=1, le=1_000_000)] = 20_000
    max_cases: Annotated[int, Field(ge=1, le=1_000_000)] = 10_000
    max_block_members: Annotated[int, Field(ge=2, le=1_000)] = 32
    max_group_pairs: Annotated[int, Field(ge=1, le=10_000_000)] = 100_000
    max_group_pairs_per_request: Annotated[
        int, Field(ge=1, le=100_000)
    ] = 64
    max_group_members_per_request: Annotated[
        int, Field(ge=2, le=10_000)
    ] = 32
    max_group_request_bytes: Annotated[
        int, Field(ge=1_000, le=200_000_000)
    ] = 1_000_000
    max_blocking_anchors_per_member: Annotated[int, Field(ge=1, le=10_000)] = 32
    max_blocking_paths_per_member: Annotated[int, Field(ge=1, le=10_000)] = 32
    max_blocking_tokens_per_member: Annotated[int, Field(ge=1, le=10_000)] = 64
    max_blocking_anchor_bytes: Annotated[int, Field(ge=16, le=4096)] = 512
    lexical_min_shared_tokens: Annotated[int, Field(ge=1, le=64)] = 2
    max_case_members: Annotated[int, Field(ge=1, le=1_000)] = 32
    max_case_input_bytes: Annotated[
        int, Field(ge=1_000, le=200_000_000)
    ] = 2_000_000
    max_current_evidence_items: Annotated[
        int, Field(ge=1, le=10_000_000)
    ] = 100_000
    max_current_evidence_bytes: Annotated[
        int, Field(ge=1_000, le=1_000_000_000)
    ] = 10_000_000
    max_observation_context_items_per_case: Annotated[
        int, Field(ge=1, le=1_000_000)
    ] = 64
    git_enabled: bool = True
    git_command_timeout_seconds: Annotated[int, Field(ge=1, le=600)] = 30
    git_object_scan_timeout_seconds: Annotated[int, Field(ge=1, le=3600)] = 120
    git_max_commands_per_case: Annotated[int, Field(ge=1, le=100)] = 16
    git_max_queries_per_family: Annotated[int, Field(ge=1, le=64)] = 4
    git_max_results_per_query: Annotated[int, Field(ge=1, le=10_000)] = 24
    git_max_patches_per_case: Annotated[int, Field(ge=1, le=10_000)] = 32
    git_max_bytes_per_command: Annotated[
        int, Field(ge=1_000, le=100_000_000)
    ] = 2_000_000
    git_max_object_scan_bytes: Annotated[
        int, Field(ge=1_000, le=1_000_000_000)
    ] = 64_000_000
    git_max_bytes_per_case: Annotated[
        int, Field(ge=1_000, le=1_000_000_000)
    ] = 8_000_000
    git_max_commit_message_bytes: Annotated[
        int, Field(ge=64, le=1_000_000)
    ] = 4096
    git_min_path_history: Annotated[int, Field(ge=0, le=100)] = 1
    git_min_blame: Annotated[int, Field(ge=0, le=100)] = 1
    git_min_pickaxe_literal: Annotated[int, Field(ge=0, le=100)] = 1
    git_min_pickaxe_regex: Annotated[int, Field(ge=0, le=100)] = 1
    git_rename_similarity_percent: Annotated[int, Field(ge=1, le=100)] = 50
    git_patch_context_lines: Annotated[int, Field(ge=0, le=100)] = 3
    git_max_changed_paths_per_commit: Annotated[int, Field(ge=1, le=10_000)] = 128
    max_temporal_items_per_case: Annotated[
        int, Field(ge=1, le=10_000)
    ] = 64
    max_temporal_batch_items: Annotated[int, Field(ge=1, le=1_000)] = 16
    max_temporal_cells_per_request: Annotated[
        int, Field(ge=1, le=10_000_000)
    ] = 128
    max_temporal_request_bytes: Annotated[
        int, Field(ge=1_000, le=200_000_000)
    ] = 2_000_000
    max_refinement_rounds: Literal[1] = 1
    max_refinement_children: Annotated[int, Field(ge=1, le=2)] = 2
    max_unified_artifact_bytes: Annotated[
        int, Field(ge=1_000, le=5_000_000_000)
    ] = 500_000_000

    @model_validator(mode="after")
    def resource_contract(self) -> UnifiedSettings:
        if (
            self.candidate_grouping_mode == "singleton"
            and self.max_candidate_members > self.max_cases
        ):
            raise ValueError(
                "unified singleton mode requires max_candidate_members <= max_cases"
            )
        if self.max_case_members > self.max_block_members:
            raise ValueError(
                "unified.max_case_members must be <= max_block_members"
            )
        if self.max_group_pairs_per_request > self.max_group_pairs:
            raise ValueError(
                "unified.max_group_pairs_per_request must be <= max_group_pairs"
            )
        if self.max_group_members_per_request > self.max_candidate_members:
            raise ValueError(
                "unified.max_group_members_per_request must be <= "
                "max_candidate_members"
            )
        if self.max_temporal_cells_per_request < self.max_case_members:
            raise ValueError(
                "unified.max_temporal_cells_per_request must be >= max_case_members"
            )
        minimum_commands = (
            self.git_min_path_history
            + self.git_min_blame
            + self.git_min_pickaxe_literal
            + self.git_min_pickaxe_regex
        )
        minimums = (
            self.git_min_path_history,
            self.git_min_blame,
            self.git_min_pickaxe_literal,
            self.git_min_pickaxe_regex,
        )
        if self.git_enabled and any(
            item > self.git_max_queries_per_family for item in minimums
        ):
            raise ValueError(
                "unified Git family minimum exceeds max_queries_per_family"
            )
        if self.git_enabled and minimum_commands > self.git_max_commands_per_case:
            raise ValueError(
                "unified Git family minimums exceed max_commands_per_case"
            )
        if (
            self.git_enabled
            and self.git_max_results_per_query
            * self.git_max_queries_per_family
            > 100_000
        ):
            raise ValueError(
                "unified Git per-family aggregate result bound exceeds 100000"
            )
        if self.git_max_bytes_per_case < self.git_max_bytes_per_command:
            raise ValueError(
                "unified.git_max_bytes_per_case must be >= per-command bytes"
            )
        return self


class OutputSettings(StrictModel):
    save_diagnostics: bool = False


class BuildConfig(StrictModel):
    execution_model_profile_id: NonEmptyString
    execution_model_profile_digest: Sha256Hex
    repository: RepositorySettings = Field(default_factory=RepositorySettings)
    discovery: DiscoverySettings = Field(default_factory=DiscoverySettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    llm: LLMSettings
    observation: ObservationSettings = Field(default_factory=ObservationSettings)
    semantic: SemanticSettings
    unified: UnifiedSettings
    output: OutputSettings = Field(default_factory=OutputSettings)

    @model_validator(mode="before")
    @classmethod
    def destructive_40_config_shape(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        if (
            "semantic" not in value
            or "agent" not in value
            or "unified" not in value
        ):
            raise ValueError(
                "schema 4.11 config requires agent, semantic, and unified sections"
            )
        llm = value.get("llm")
        required_llm = {
            "max_total_calls",
            "max_total_input_tokens",
            "max_total_output_tokens",
            "max_inference_artifact_bytes",
        }
        if not isinstance(llm, dict) or not required_llm <= set(llm):
            raise ValueError(
                "schema 4.11 config requires common total inference budgets"
            )
        return value

    @model_validator(mode="after")
    def semantic_dependency_contract(self) -> BuildConfig:
        if not self.observation.enabled and self.semantic.enabled:
            raise ValueError("semantic units require observation.enabled")
        if self.semantic.enabled:
            required = {
                item.name: item.required for item in self.observation.queries
            }
            for name in ("python_snapshot", "python_clone_clusters"):
                if required.get(name) is not True:
                    raise ValueError(
                        "semantic units require enabled, required "
                        f"observation query {name}"
                    )
        return self


class RepositoryMetadata(StrictModel):
    name: NonEmptyString
    root: NonEmptyString
    snapshot_digest: SnapshotDigest
    git_head_for_reference: (
        Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")] | None
    )


class ManifestFile(StrictModel):
    path: NonEmptyString
    content_sha256: Sha256Hex
    size_bytes: Annotated[int, Field(ge=0)]
    suffix: str
    role: Literal["python_source", "supported_text", "other"]


class SkippedPath(StrictModel):
    path: NonEmptyString
    reason: NonEmptyString


class SourceMetadata(StrictModel):
    path: NonEmptyString
    source_kind: SourceKind
    reason: NonEmptyString
    content_sha256: Sha256Hex
    size_bytes: Annotated[int, Field(ge=0)]
    referenced_by: list[NonEmptyString] = Field(default_factory=list)


class ManifestSection(StrictModel):
    mode: Literal["git", "walk", "provided"]
    file_list_sha256: Sha256Hex
    files: list[ManifestFile]
    skipped: list[SkippedPath]


class DiscoveryArtifact(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"]
    repository: RepositoryMetadata
    language_scope: Literal["python"]
    limits: DiscoverySettings
    manifest: ManifestSection
    baseline_sources: list[SourceMetadata]
    supplement_candidates: list[SourceMetadata]
    complete: bool
    incomplete_reasons: list[NonEmptyString]

    @model_validator(mode="after")
    def paths_and_sources_must_be_unique(self) -> DiscoveryArtifact:
        manifest_paths = [item.path for item in self.manifest.files]
        baseline_paths = [item.path for item in self.baseline_sources]
        candidate_paths = [item.path for item in self.supplement_candidates]
        if len(manifest_paths) != len(set(manifest_paths)):
            raise ValueError("manifest paths must be unique")
        if len(baseline_paths) != len(set(baseline_paths)):
            raise ValueError("baseline source paths must be unique")
        if len(candidate_paths) != len(set(candidate_paths)):
            raise ValueError("supplement candidate paths must be unique")
        if set(baseline_paths) & set(candidate_paths):
            raise ValueError(
                "baseline and supplement candidate paths must be disjoint"
            )
        return self


class SourceProposal(StrictModel):
    path: NonEmptyString
    compared_baseline_paths: list[NonEmptyString]
    reason: Annotated[str, Field(min_length=1, max_length=300)]

    @field_validator("compared_baseline_paths")
    @classmethod
    def compared_baseline_paths_must_be_unique(
        cls, value: list[str]
    ) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("compared baseline paths must be unique")
        return value

    @field_validator("reason")
    @classmethod
    def reason_must_be_short_single_line(cls, value: str) -> str:
        if not value.strip() or "\n" in value or "\r" in value:
            raise ValueError("proposal reason must be a non-blank single line")
        return value.strip()


class SourceSupplement(StrictModel):
    schema_version: Literal["1.2"]
    snapshot_digest: SnapshotDigest
    search_status: Literal["complete", "truncated"]
    proposals: list[SourceProposal]

    @model_validator(mode="after")
    def proposal_paths_must_be_unique(self) -> SourceSupplement:
        paths = [proposal.path for proposal in self.proposals]
        if len(paths) != len(set(paths)):
            raise ValueError("source supplement proposal paths must be unique")
        return self


class EvidenceSpan(StrictModel):
    span_id: SpanId
    line_number: PositiveInt
    text: str


class EvidenceUnit(StrictModel):
    unit_id: UnitId
    path: NonEmptyString
    source_kind: SourceKind
    discovery_origin: DiscoveryOrigin
    line_start: PositiveInt
    line_end: PositiveInt
    default_scope: Annotated[list[NonEmptyString], Field(min_length=1)]
    content: NonEmptyString
    content_sha256: Sha256Hex
    source_content_sha256: Sha256Hex
    spans: Annotated[list[EvidenceSpan], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_physical_lines(self) -> EvidenceUnit:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        numbers = [item.line_number for item in self.spans]
        if numbers != list(range(self.line_start, self.line_end + 1)):
            raise ValueError("unit spans must cover one contiguous physical-line range")
        if len({item.span_id for item in self.spans}) != len(self.spans):
            raise ValueError("unit span IDs must be unique")
        if "".join(item.text for item in self.spans) != self.content:
            raise ValueError("unit content must be the exact concatenation of spans")
        return self


class CandidateEvidenceRange(StrictModel):
    unit_id: UnitId
    span_start_id: SpanId
    span_end_id: SpanId


class CandidateClaim(StrictModel):
    rule: Annotated[str, Field(min_length=1, max_length=500)]
    claim_kind: ClaimKind
    normative_strength: NormativeStrength
    applies_to: Annotated[list[NonEmptyString], Field(min_length=1)]
    condition: str | None
    evidence: Annotated[list[CandidateEvidenceRange], Field(min_length=1)]

    @field_validator("condition")
    @classmethod
    def condition_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("condition must be null or non-blank")
        return value


ContainsReason = Literal[
    "direct_norm",
    "configuration_fact",
    "command_fact",
    "workflow_fact",
]
NoClaimsReason = Literal[
    "generic_advice",
    "single_example",
    "duplicate_content",
    "product_behavior",
    "no_actionable_repository_knowledge",
]
UncertainReason = Literal[
    "scope_unclear",
    "temporal_unclear",
    "conflicting_text",
    "unsupported_dynamic_config",
    "other_uncertain",
]


class ContainsClaimsResult(StrictModel):
    unit_id: UnitId
    classification: Literal["contains_claims"]
    reason_code: ContainsReason
    claims: Annotated[list[CandidateClaim], Field(min_length=1)]


class NoClaimsResult(StrictModel):
    unit_id: UnitId
    classification: Literal["no_claims"]
    reason_code: NoClaimsReason
    claims: Annotated[list[CandidateClaim], Field(max_length=0)] = Field(
        default_factory=list
    )


class UncertainResult(StrictModel):
    unit_id: UnitId
    classification: Literal["uncertain"]
    reason_code: UncertainReason
    claims: Annotated[list[CandidateClaim], Field(max_length=0)] = Field(
        default_factory=list
    )


UnitExtractionResult = Annotated[
    ContainsClaimsResult | NoClaimsResult | UncertainResult,
    Field(discriminator="classification"),
]


class BatchExtractionResponse(StrictModel):
    results: list[UnitExtractionResult]


class ProviderResponse(StrictModel):
    content: str
    model: NonEmptyString
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    response_content_bytes: Annotated[int, Field(ge=0)] | None = None
    reasoning_content_present: bool = False
    reasoning_content_bytes: Annotated[int, Field(ge=0)] = 0
    latency_ms: Annotated[int, Field(ge=0)]
    finish_reason: str | None = None

    @model_validator(mode="after")
    def validate_response_telemetry(self) -> ProviderResponse:
        content_bytes = len(self.content.encode("utf-8"))
        if self.response_content_bytes is None:
            self.response_content_bytes = content_bytes
        elif self.response_content_bytes != content_bytes:
            raise ValueError("response content byte telemetry differs from content")
        if self.reasoning_tokens is not None and self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning tokens exceed total completion tokens")
        if not self.reasoning_content_present and self.reasoning_content_bytes != 0:
            raise ValueError(
                "reasoning content bytes require reasoning content presence"
            )
        return self


class ProviderAttempt(StrictModel):
    schema_attempt: Annotated[int, Field(ge=0, le=3)]
    output_limit_retry: Annotated[int, Field(ge=0, le=2)] = 0
    provider_try: Annotated[int, Field(ge=0, le=5)]
    repair: bool
    parsed: bool
    provider_error: bool
    retryable: bool
    model: NonEmptyString
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    response_content_bytes: Annotated[int, Field(ge=0)] | None = None
    reasoning_content_present: bool = False
    reasoning_content_bytes: Annotated[int, Field(ge=0)] = 0
    latency_ms: Annotated[int, Field(ge=0)]
    finish_reason: str | None = None
    validation_errors: list[str]


class Evidence(StrictModel):
    evidence_kind: Literal["explicit_span"] = "explicit_span"
    type: EvidenceType
    adapter: AdapterKind
    semantic_key: NonEmptyString
    path: NonEmptyString
    span_ids: Annotated[
        list[SpanId], Field(min_length=1, max_length=MAX_LOGICAL_EVIDENCE_LINES)
    ]
    line_start: PositiveInt
    line_end: PositiveInt
    quote: NonEmptyString
    quote_sha256: Sha256Hex
    source_content_sha256: Sha256Hex
    source_kind: SourceKind
    discovery_origin: DiscoveryOrigin

    @model_validator(mode="after")
    def line_range_must_be_ordered(self) -> Evidence:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        if len(self.span_ids) != self.line_end - self.line_start + 1:
            raise ValueError("one span ID is required for each evidence line")
        return self


class EvidenceRelation(StrictModel):
    relation: EvidenceRelationKind
    route: NonEmptyString
    target_fingerprint: NonEmptyString
    reason: NonEmptyString
    effect: Literal["advisory", "blocking"] = "advisory"


class Adjudication(StrictModel):
    route: Literal["explicit_typed_evidence"]
    policy_version: Literal["1.9"]
    sufficiency: Literal["passed", "failed", "not_evaluated"]
    reasons: list[NonEmptyString]


class Constraint(StrictModel):
    id: ClaimId
    epistemic_status: Literal["accepted"]
    origin_channels: Annotated[list[OriginChannel], Field(min_length=1)]
    rule: NonEmptyString
    claim_kind: ClaimKind
    normative_strength: NormativeStrength
    acceptance_basis: Annotated[list[AcceptanceBasis], Field(min_length=1)]
    adjudication: Adjudication
    applies_to: Annotated[list[NonEmptyString], Field(min_length=1)]
    condition: str | None
    delivery_policy: DeliveryPolicy
    execution_risk: ExecutionRisk
    proposition_key: NonEmptyString
    evidence: Annotated[list[Evidence], Field(min_length=1)]
    evidence_relations: list[EvidenceRelation] = Field(default_factory=list)
    conflicts: list[NonEmptyString] = Field(default_factory=list)

    @field_validator("condition")
    @classmethod
    def condition_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("condition must be null or non-blank")
        return value

class ExplicitCoverage(StrictModel):
    manifest: Literal["complete", "partial"]
    deterministic_discovery: Literal["complete", "partial"]
    agent_source_discovery: Literal["complete", "partial", "failed"]
    source_plan: Literal["complete", "partial"]
    eligible_units: Literal["complete", "partial"]
    extraction: Literal["complete", "partial"]
    grounding: Literal["complete", "partial"]
    semantic_verification: Literal["complete", "partial", "failed"]
    claim_semantics: Literal["complete", "partial"]
    proposition_merge: Literal["complete", "partial"]
    source_recall: Literal["unknown"]


class ObservationQueryCoverage(StrictModel):
    name: NonEmptyString
    version: NonEmptyString
    required: bool
    status: CapabilityStatus
    incomplete_reasons: list[NonEmptyString]


class ObservationCoverage(StrictModel):
    mode: Literal["foundational", "not_run"]
    overall: CapabilityStatus
    queries: list[ObservationQueryCoverage]
    observation_set_digest: Sha256Hex | None

    @model_validator(mode="after")
    def queries_are_unique(self) -> ObservationCoverage:
        names = [item.name for item in self.queries]
        if len(names) != len(set(names)):
            raise ValueError("observation coverage query names must be unique")
        if self.mode == "not_run" and (
            self.overall != "not_run"
            or self.queries
            or self.observation_set_digest is not None
        ):
            raise ValueError("not-run observation coverage cannot contain queries")
        if self.mode == "foundational" and self.observation_set_digest is None:
            raise ValueError(
                "foundational observation coverage requires an observation digest"
            )
        return self


class UnifiedCapabilityCoverage(StrictModel):
    status: Literal[
        "complete",
        "bounded_no_match",
        "partial",
        "unavailable",
        "failed",
        "not_required",
        "not_run",
    ]
    required: bool
    planned: Annotated[int, Field(ge=0)] = 0
    completed: Annotated[int, Field(ge=0)] = 0
    reasons: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_shape(self) -> UnifiedCapabilityCoverage:
        if self.completed > self.planned:
            raise ValueError("unified capability completed count exceeds planned")
        if self.status in {"complete", "bounded_no_match"} and self.reasons:
            raise ValueError("complete unified capability cannot carry gaps")
        if self.status in {"not_required", "not_run"} and (
            self.planned or self.completed
        ):
            raise ValueError("unrun unified capability cannot carry work")
        return self


def _unified_not_run() -> UnifiedCapabilityCoverage:
    return UnifiedCapabilityCoverage(
        status="not_run",
        required=True,
        reasons=["unified_layer_not_run"],
    )


class UnifiedCoverage(StrictModel):
    candidate_grouping: UnifiedCapabilityCoverage = Field(
        default_factory=_unified_not_run
    )
    current_evidence: UnifiedCapabilityCoverage = Field(
        default_factory=_unified_not_run
    )
    git_temporal_evidence: UnifiedCapabilityCoverage = Field(
        default_factory=lambda: UnifiedCapabilityCoverage(
            status="not_run",
            required=False,
            reasons=["unified_layer_not_run"],
        )
    )
    temporal_classification: UnifiedCapabilityCoverage = Field(
        default_factory=lambda: UnifiedCapabilityCoverage(
            status="not_run",
            required=False,
            reasons=["unified_layer_not_run"],
        )
    )
    case_verification: UnifiedCapabilityCoverage = Field(
        default_factory=_unified_not_run
    )
    refinement: UnifiedCapabilityCoverage = Field(
        default_factory=lambda: UnifiedCapabilityCoverage(
            status="not_required", required=False
        )
    )
    canonical_policy: UnifiedCapabilityCoverage = Field(
        default_factory=_unified_not_run
    )
    case_count: Annotated[int, Field(ge=0)] = 0
    canonical_claim_count: Annotated[int, Field(ge=0)] = 0
    accepted_count: Annotated[int, Field(ge=0)] = 0
    review_count: Annotated[int, Field(ge=0)] = 0
    rejected_count: Annotated[int, Field(ge=0)] = 0
    invalid_member_count: Annotated[int, Field(ge=0)] = 0
    incomplete_reasons: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def disposition_algebra(self) -> UnifiedCoverage:
        if self.accepted_count + self.review_count + self.rejected_count != (
            self.canonical_claim_count
        ):
            raise ValueError("unified canonical disposition counts do not sum")
        return self


class SemanticUnitCoverage(StrictModel):
    required: bool
    status: CapabilityStatus
    planned: Annotated[int, Field(ge=0)]
    succeeded: Annotated[int, Field(ge=0)]
    no_input: Annotated[int, Field(ge=0)]
    partial: Annotated[int, Field(ge=0)] = 0
    failed: Annotated[int, Field(ge=0)]
    omitted: Annotated[int, Field(ge=0)]
    reasons: list[NonEmptyString]

    @model_validator(mode="after")
    def count_algebra(self) -> SemanticUnitCoverage:
        if (
            self.succeeded
            + self.no_input
            + self.partial
            + self.failed
            + self.omitted
            != self.planned
        ):
            raise ValueError("semantic-unit coverage counts do not sum to planned")
        if self.status == "complete" and (
            self.partial or self.failed or self.omitted or self.reasons
        ):
            raise ValueError("complete semantic-unit coverage has incomplete work")
        if self.status == "not_run" and any(
            (
                self.planned,
                self.succeeded,
                self.no_input,
                self.partial,
                self.failed,
                self.omitted,
            )
        ):
            raise ValueError("not-run semantic-unit coverage cannot contain work")
        return self


class SemanticMapCoverage(StrictModel):
    required: bool
    status: CapabilityStatus
    planned: Annotated[int, Field(ge=0)]
    succeeded: Annotated[int, Field(ge=0)]
    no_observation: Annotated[int, Field(ge=0)]
    uncertain: Annotated[int, Field(ge=0)]
    failed: Annotated[int, Field(ge=0)]
    omitted: Annotated[int, Field(ge=0)]
    reasons: list[NonEmptyString]

    @model_validator(mode="after")
    def count_algebra(self) -> SemanticMapCoverage:
        if (
            self.succeeded
            + self.no_observation
            + self.uncertain
            + self.failed
            + self.omitted
            != self.planned
        ):
            raise ValueError("semantic-map coverage counts do not sum to planned")
        _validate_stage_coverage(
            self.status,
            self.planned,
            self.failed + self.omitted,
            self.reasons,
            stage="semantic-map",
        )
        return self


class HypothesisReductionCoverage(StrictModel):
    required: bool
    status: CapabilityStatus
    planned: Annotated[int, Field(ge=0)]
    succeeded: Annotated[int, Field(ge=0)]
    no_hypothesis: Annotated[int, Field(ge=0)]
    uncertain: Annotated[int, Field(ge=0)]
    failed: Annotated[int, Field(ge=0)]
    reasons: list[NonEmptyString]

    @model_validator(mode="after")
    def count_algebra(self) -> HypothesisReductionCoverage:
        if (
            self.succeeded
            + self.no_hypothesis
            + self.uncertain
            + self.failed
            != self.planned
        ):
            raise ValueError("hypothesis-reduction counts do not sum to planned")
        _validate_stage_coverage(
            self.status,
            self.planned,
            self.failed,
            self.reasons,
            stage="hypothesis-reduction",
        )
        return self


class PopulationVerificationCoverage(StrictModel):
    required: bool
    status: CapabilityStatus
    planned_hypotheses: Annotated[int, Field(ge=0)]
    complete_populations: Annotated[int, Field(ge=0)]
    partial_populations: Annotated[int, Field(ge=0)]
    failed_populations: Annotated[int, Field(ge=0)]
    classified_members: Annotated[int, Field(ge=0)]
    matched_members: Annotated[int, Field(ge=0)]
    inspected_members: Annotated[int, Field(ge=0)]
    uninspected_members: Annotated[int, Field(ge=0)]
    census_populations: Annotated[int, Field(ge=0)]
    sampled_populations: Annotated[int, Field(ge=0)]
    technical_unknown: Annotated[int, Field(ge=0)]
    reasons: list[NonEmptyString]

    @model_validator(mode="after")
    def count_algebra(self) -> PopulationVerificationCoverage:
        if (
            self.complete_populations
            + self.partial_populations
            + self.failed_populations
            != self.planned_hypotheses
        ):
            raise ValueError("population coverage counts do not sum to planned")
        if self.inspected_members + self.uninspected_members != self.matched_members:
            raise ValueError("population coverage sampling counts do not sum")
        if self.classified_members != self.inspected_members:
            raise ValueError("classified population coverage differs from inspected")
        if self.census_populations + self.sampled_populations > (
            self.planned_hypotheses
        ):
            raise ValueError("population sampling modes exceed planned hypotheses")
        _validate_stage_coverage(
            self.status,
            self.planned_hypotheses,
            self.partial_populations + self.failed_populations,
            self.reasons,
            stage="population-verification",
        )
        if self.status == "complete" and self.technical_unknown:
            raise ValueError("complete population coverage has technical unknowns")
        return self


class FalsificationCoverage(StrictModel):
    required: bool
    status: CapabilityStatus
    planned_hypotheses: Annotated[int, Field(ge=0)]
    completed: Annotated[int, Field(ge=0)]
    refined: Annotated[int, Field(ge=0)]
    rejected: Annotated[int, Field(ge=0)]
    uncertain: Annotated[int, Field(ge=0)]
    failed: Annotated[int, Field(ge=0)]
    reasons: list[NonEmptyString]

    @model_validator(mode="after")
    def count_algebra(self) -> FalsificationCoverage:
        if (
            self.completed
            + self.refined
            + self.rejected
            + self.uncertain
            + self.failed
            != self.planned_hypotheses
        ):
            raise ValueError("falsification coverage counts do not sum to planned")
        _validate_stage_coverage(
            self.status,
            self.planned_hypotheses,
            self.failed,
            self.reasons,
            stage="falsification",
        )
        return self


def _not_run_semantic_map() -> SemanticMapCoverage:
    return SemanticMapCoverage(
        required=False,
        status="not_run",
        planned=0,
        succeeded=0,
        no_observation=0,
        uncertain=0,
        failed=0,
        omitted=0,
        reasons=[],
    )


def _not_run_reduction() -> HypothesisReductionCoverage:
    return HypothesisReductionCoverage(
        required=False,
        status="not_run",
        planned=0,
        succeeded=0,
        no_hypothesis=0,
        uncertain=0,
        failed=0,
        reasons=[],
    )


def _not_run_population() -> PopulationVerificationCoverage:
    return PopulationVerificationCoverage(
        required=False,
        status="not_run",
        planned_hypotheses=0,
        complete_populations=0,
        partial_populations=0,
        failed_populations=0,
        classified_members=0,
        matched_members=0,
        inspected_members=0,
        uninspected_members=0,
        census_populations=0,
        sampled_populations=0,
        technical_unknown=0,
        reasons=[],
    )


def _not_run_falsification() -> FalsificationCoverage:
    return FalsificationCoverage(
        required=False,
        status="not_run",
        planned_hypotheses=0,
        completed=0,
        refined=0,
        rejected=0,
        uncertain=0,
        failed=0,
        reasons=[],
    )


def _validate_stage_coverage(
    status: CapabilityStatus,
    planned: int,
    incomplete_count: int,
    reasons: list[str],
    *,
    stage: str,
) -> None:
    if status == "complete" and (incomplete_count or reasons):
        raise ValueError(f"complete {stage} coverage has incomplete work")
    if status == "not_run" and (planned or reasons):
        raise ValueError(f"not-run {stage} coverage cannot contain work")


class CurrentSourceCoverage(StrictModel):
    semantic_units: SemanticUnitCoverage
    semantic_map: SemanticMapCoverage = Field(default_factory=_not_run_semantic_map)
    hypothesis_reduction: HypothesisReductionCoverage = Field(
        default_factory=_not_run_reduction
    )
    population_verification: PopulationVerificationCoverage = Field(
        default_factory=_not_run_population
    )
    falsification: FalsificationCoverage = Field(
        default_factory=_not_run_falsification
    )


class DeliveryCoverage(StrictModel):
    scope_audit: Literal["complete", "partial"]
    risk_audit: Literal["complete", "partial"]


class Coverage(StrictModel):
    explicit: ExplicitCoverage
    observation: ObservationCoverage
    current_source: CurrentSourceCoverage
    unified: UnifiedCoverage = Field(default_factory=UnifiedCoverage)
    delivery: DeliveryCoverage


class MemoryFile(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"]
    repository: RepositoryMetadata
    language_scope: Literal["python"]
    coverage: Coverage
    claims: list[Constraint]

    @model_validator(mode="after")
    def claim_ids_must_be_unique(self) -> MemoryFile:
        ids = [claim.id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim IDs must be unique")
        return self


class CandidateAuditItem(StrictModel):
    candidate_fingerprint: str | None
    verification_request_id: str | None = Field(
        default=None, pattern=r"^INFER-[0-9a-f]{24}$"
    )
    verification_reason_codes: list[NonEmptyString] = Field(
        default_factory=list
    )
    verification_technical_reason_code: str | None = None
    local_outcome: LocalOutcome
    outcome_stage: OutcomeStage
    reason_code: NonEmptyString
    rule: str | None
    claim_id: str | None
    validation_errors: list[str]


class AttemptAudit(StrictModel):
    schema_attempt: Annotated[int, Field(ge=0, le=3)]
    output_limit_retry: Annotated[int, Field(ge=0, le=2)]
    provider_try: Annotated[int, Field(ge=0, le=5)]
    repair: bool
    parsed: bool
    provider_error: bool
    validation_errors: list[str]


class AuditRecord(StrictModel):
    unit_id: UnitId
    path: NonEmptyString
    line_start: PositiveInt
    line_end: PositiveInt
    discovery_origin: DiscoveryOrigin
    inference_request_id: str | None = Field(
        default=None, pattern=r"^INFER-[0-9a-f]{24}$"
    )
    model_classification: Literal[
        "contains_claims",
        "no_claims",
        "uncertain",
        "not_sent",
        "failed",
    ]
    local_outcome: LocalOutcome
    reason_code: NonEmptyString
    claim_ids: list[str]
    validation_errors: list[str]
    attempts: list[AttemptAudit]
    items: list[CandidateAuditItem]


class ExplicitReviewItem(StrictModel):
    review_kind: Literal["explicit"] = "explicit"
    candidate_fingerprint: NonEmptyString
    unit_id: UnitId
    path: NonEmptyString
    line_start: PositiveInt
    line_end: PositiveInt
    discovery_origin: DiscoveryOrigin
    inference_request_id: str | None = Field(
        default=None, pattern=r"^INFER-[0-9a-f]{24}$"
    )
    verification_request_id: str | None = Field(
        default=None, pattern=r"^INFER-[0-9a-f]{24}$"
    )
    verification_reason_codes: list[NonEmptyString] = Field(
        default_factory=list
    )
    verification_technical_reason_code: str | None = None
    epistemic_status: Literal["review", "rejected"]
    outcome_stage: OutcomeStage
    reason_code: NonEmptyString
    raw_rule: str | None
    raw_claim_kind: ClaimKind | None
    raw_normative_strength: NormativeStrength | None
    raw_applies_to: list[str]
    raw_condition: str | None
    recovered_evidence: list[Evidence]
    derived_evidence_semantics: list[str]
    evidence_relations: list[EvidenceRelation]
    validation_errors: list[str]


class CurrentSourceReviewItem(StrictModel):
    review_kind: Literal["current_source"] = "current_source"
    trace_id: Annotated[str, Field(pattern=r"^TRACE-[0-9a-f]{24}$")]
    candidate_root_id: NonEmptyString
    hypothesis_id: Annotated[str, Field(pattern=r"^HYP-[0-9a-f]{24}$")]
    proposition_id: Annotated[str, Field(pattern=r"^PROP-[0-9a-f]{24}$")] | None
    epistemic_status: Literal["hypothesis", "review", "rejected"]
    evidence_route: Literal[
        "current_structural_population",
        "current_semantic_population",
        "unoperationalized",
    ]
    category: Annotated[str, Field(min_length=1, max_length=120)] | None
    rule: Annotated[str, Field(min_length=1, max_length=500)]
    applies_to: Annotated[list[NonEmptyString], Field(min_length=1)]
    condition: str | None
    population_id: str | None = Field(
        default=None, pattern=r"^POP-[0-9a-f]{24}$"
    )
    falsification_ids: list[
        Annotated[str, Field(pattern=r"^FALSIFY-[0-9a-f]{24}$")]
    ]
    reason_codes: Annotated[list[NonEmptyString], Field(min_length=1)]
    limitations: list[NonEmptyString]


ReviewItem = Annotated[
    ExplicitReviewItem | CurrentSourceReviewItem,
    Field(discriminator="review_kind"),
]


class ReviewFile(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"]
    snapshot_digest: SnapshotDigest
    items: list[ReviewItem]


class CatalogClaim(StrictModel):
    id: ClaimId
    rule: NonEmptyString
    claim_kind: ClaimKind
    normative_strength: NormativeStrength
    applies_to: list[NonEmptyString]
    condition: str | None
    delivery_policy: DeliveryPolicy
    execution_risk: ExecutionRisk
    origin_channels: list[OriginChannel]
    adjudication_route: NonEmptyString
    adjudication_policy_version: NonEmptyString
    evidence_kinds: list[NonEmptyString]
    evidence_count: Annotated[int, Field(ge=1)]


class CatalogFile(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"]
    snapshot_digest: SnapshotDigest
    claims: list[CatalogClaim]


class DeliveryExclusion(StrictModel):
    claim_id: ClaimId
    reason: NonEmptyString


class DeliveryReport(StrictModel):
    schema_version: Literal["4.11-partition-only-required-evidence"]
    snapshot_digest: SnapshotDigest
    accepted_claim_count: Annotated[int, Field(ge=0)]
    delivered_claim_count: Annotated[int, Field(ge=0)]
    policy_counts: dict[str, Annotated[int, Field(ge=0)]]
    risk_counts: dict[str, Annotated[int, Field(ge=0)]]
    exclusions: list[DeliveryExclusion]
