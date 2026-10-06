from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from repo_constraint_memory_v3.builder import build_memory, load_build_config
from repo_constraint_memory_v3.agent_runtime import (
    DeterministicFakeAgentRuntime,
    fake_agent_profile,
)
from repo_constraint_memory_v3.discovery import discover_repository
from repo_constraint_memory_v3.llm import (
    LLMProviderError,
    MockLLMClient,
)
from repo_constraint_memory_v3.model_profiles import load_execution_model_profile
from repo_constraint_memory_v3.models import (
    BuildConfig,
    ProviderResponse,
    SourceSupplement,
)
from repo_constraint_memory_v3.source_acquisition import acquire_sources

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PACKAGE_ROOT / "tests" / "fixtures" / "explicit_hardening_repo"
INCOMPLETE_FIXTURE_ROOT = (
    PACKAGE_ROOT / "tests" / "fixtures" / "incomplete_observation_repo"
)
OBSERVATION_FIXTURE_ROOT = (
    PACKAGE_ROOT / "tests" / "fixtures" / "python_observation_repo"
)
SEMANTIC_FIXTURE_ROOT = (
    PACKAGE_ROOT / "tests" / "fixtures" / "semantic_substrate_repo"
)
CURRENT_SOURCE_FIXTURE_ROOT = (
    PACKAGE_ROOT / "tests" / "fixtures" / "current_source_open_repo"
)
CONFIG_PATH = PACKAGE_ROOT / "configs" / "hardened.yaml"


def copied_fixture(tmp_path: Path, *, incomplete: bool = False) -> Path:
    destination = tmp_path / ("incomplete-repository" if incomplete else "repository")
    shutil.copytree(
        INCOMPLETE_FIXTURE_ROOT if incomplete else FIXTURE_ROOT,
        destination,
    )
    return destination


def copied_observation_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "python-observation-repository"
    shutil.copytree(OBSERVATION_FIXTURE_ROOT, destination)
    return destination


def copied_semantic_fixture(
    tmp_path: Path,
    *,
    include_broken: bool = False,
) -> Path:
    destination = tmp_path / "semantic-substrate-repository"
    shutil.copytree(SEMANTIC_FIXTURE_ROOT, destination)
    if not include_broken:
        shutil.rmtree(destination / "broken")
    return destination


def copied_current_source_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "current-source-open-repository"
    shutil.copytree(CURRENT_SOURCE_FIXTURE_ROOT, destination)
    return destination


def config_for_tests(
    *,
    chunking: dict[str, int] | None = None,
    observation: dict[str, object] | None = None,
    semantic: dict[str, object] | None = None,
    unified: dict[str, object] | None = None,
) -> BuildConfig:
    model_profile = load_execution_model_profile(
        "offline-deterministic-test",
        allow_offline=True,
    )
    config = load_build_config(CONFIG_PATH, model_profile=model_profile)
    # Most tests isolate semantic correctness rather than the production cost
    # profile.  Keep their small fixtures on a census path; sampling-specific
    # tests opt into the active 20/1,000 bounds explicitly.
    config = config.model_copy(
        update={
            "semantic": config.semantic.model_copy(
                update={
                    "max_classified_population_members_per_hypothesis": 512,
                    "max_total_classified_population_members": 4_096,
                }
            ),
            # The shipped optimization profile intentionally selects singleton
            # grouping.  Most legacy grouping tests exercise the semantic
            # planner itself, so keep that behavior explicit in the test
            # harness; singleton-specific tests opt in below.
            "unified": config.unified.model_copy(
                update={"candidate_grouping_mode": "semantic"}
            ),
        }
    )
    if chunking:
        config = config.model_copy(
            update={
                "chunking": config.chunking.model_copy(update=chunking)
            }
        )
    if observation:
        semantic_update: dict[str, object] = {}
        if observation.get("enabled") is False:
            semantic_update = {"enabled": False, "required": False}
        queries = observation.get("queries")
        if isinstance(queries, list):
            required = {
                item.name: item.required
                for item in queries
                if hasattr(item, "name")
            }
            if not all(
                required.get(name) is True
                for name in ("python_snapshot", "python_clone_clusters")
            ):
                semantic_update = {"enabled": False, "required": False}
        config = config.model_copy(
            update={
                "observation": config.observation.model_copy(
                    update=observation
                ),
                "semantic": config.semantic.model_copy(
                    update=semantic_update
                ),
            }
        )
    if semantic:
        config = config.model_copy(
            update={"semantic": config.semantic.model_copy(update=semantic)}
        )
    if unified:
        config = config.model_copy(
            update={"unified": config.unified.model_copy(update=unified)}
        )
    return config


def discover_fixture(repository: Path, config: BuildConfig):
    return discover_repository(
        repository,
        config.discovery,
        repository_name=config.repository.name,
    )


def supplement_for(discovery, *paths: str) -> SourceSupplement:
    return SourceSupplement(
        schema_version="1.2",
        snapshot_digest=discovery.repository.snapshot_digest,
        search_status="complete",
        proposals=[
            {
                "path": path,
                "compared_baseline_paths": [],
                "reason": "Task-independent contributor source candidate.",
            }
            for path in paths
        ],
    )


def build_fixture(
    repository: Path,
    output: Path,
    *,
    config: BuildConfig | None = None,
    supplement: SourceSupplement | None = None,
    client=None,
    observation_cache: Path | None = None,
    inference_cache: Path | None = None,
    resume_from: Path | None = None,
    progress_callback=None,
):
    effective_config = config or config_for_tests()
    proposals = tuple(
        (item.path, item.reason)
        for item in (supplement.proposals if supplement is not None else [])
    )
    profile = fake_agent_profile()
    model_profile = load_execution_model_profile(
        "offline-deterministic-test",
        allow_offline=True,
    )
    acquisition = acquire_sources(
        repository,
        effective_config,
        profile,
        model_profile,
        DeterministicFakeAgentRuntime(profile, proposals=proposals),
        output.parent / f".{output.name}-acquisition-{uuid.uuid4().hex}",
    )
    return build_memory(
        repository,
        effective_config,
        acquisition,
        client or MockLLMClient(),
        output,
        observation_cache=observation_cache,
        inference_cache=inference_cache,
        resume_from=resume_from,
        progress_callback=progress_callback,
    )


class SequenceClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        del system_prompt, user_prompt
        self.calls += 1
        return ProviderResponse(
            content=self.responses.pop(0),
            model="sequence-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


class DelayedMockClient:
    def __init__(
        self,
        delay_seconds: float = 0.02,
        *,
        invalid_path: str | None = None,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.invalid_path = invalid_path
        self.mock = MockLLMClient()
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak_in_flight = 0
        self.in_flight_by_stage: dict[str, int] = {}
        self.peak_in_flight_by_stage: dict[str, int] = {}

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        payload = json.loads(user_prompt)
        request = payload.get("original_request", payload)
        stage = str(request.get("stage") or "explicit_extract")
        paths = {
            unit["path"]
            for unit in request.get("units", [])
            if isinstance(unit, dict) and "path" in unit
        }
        with self.lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            stage_in_flight = self.in_flight_by_stage.get(stage, 0) + 1
            self.in_flight_by_stage[stage] = stage_in_flight
            self.peak_in_flight_by_stage[stage] = max(
                self.peak_in_flight_by_stage.get(stage, 0),
                stage_in_flight,
            )
        try:
            time.sleep(self.delay_seconds)
            if self.invalid_path and self.invalid_path in paths:
                return ProviderResponse(
                    content="{not-json",
                    model="delayed-invalid",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=1,
                )
            return self.mock.complete(system_prompt, user_prompt)
        finally:
            with self.lock:
                self.in_flight -= 1
                self.in_flight_by_stage[stage] -= 1


class RetryThenMock:
    def __init__(self) -> None:
        self.calls = 0
        self.mock = MockLLMClient()

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1:
            raise LLMProviderError("temporary", retryable=True)
        return self.mock.complete(system_prompt, user_prompt)


class ParameterizedSemanticClient:
    """Data-driven fake that changes open semantic metadata, never production logic."""

    def __init__(self, *, category: str, rule: str) -> None:
        self.category = category
        self.rule = rule
        self.mock = MockLLMClient()

    def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        response = self.mock.complete(system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        stage = request.get("stage")
        value = json.loads(response.content)
        if stage == "current_semantic_map":
            for result in value["results"]:
                for observation in result["observations"]:
                    observation["category"] = self.category
                    observation["summary"] = (
                        f"Fixture evidence may express {self.category}."
                    )
        elif stage == "current_hypothesis_reduce":
            for hypothesis in value.get("hypotheses", []):
                hypothesis["category"] = self.category
                hypothesis["rule"] = self.rule
        return ProviderResponse(
            content=json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            model="parameterized-semantic-fake",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )
