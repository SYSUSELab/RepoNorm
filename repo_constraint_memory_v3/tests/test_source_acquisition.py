from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3.agent_runtime import (
    AgentProcessOutcome,
    AgentRunArtifact,
    AgentRunnerResult,
    DeterministicFakeAgentRuntime,
    SourceAcquisitionArtifact,
    SubprocessAgentRuntime,
    agent_model_binding,
    fake_agent_profile,
    sha256_bytes,
    stable_json_text,
    stable_model_digest,
)
from repo_constraint_memory_v3.integrations.agent_runners.mini_swe_agent import (
    FINALIZER_COMMAND,
    RunnerFailure,
    SUCCESS_SUBMISSION,
    _FINALIZER_SOURCE,
    _instance_task,
    _load_trajectory,
    _runner_options,
    _system_template,
)
from repo_constraint_memory_v3.model_profiles import load_execution_model_profile
from repo_constraint_memory_v3.source_acquisition import (
    SourceAcquisitionError,
    acquire_sources as _acquire_sources,
    load_source_acquisition,
    validate_acquisition_for_build,
)

from helpers import config_for_tests, copied_fixture, discover_fixture


def acquire_sources(repository, config, profile, runtime, output_directory):
    return _acquire_sources(
        repository,
        config,
        profile,
        load_execution_model_profile(
            "offline-deterministic-test",
            allow_offline=True,
        ),
        runtime,
        output_directory,
    )


class FailedRuntime:
    def __init__(self, profile) -> None:
        self.profile = profile
        self.calls = 0

    def run(self, *args, **kwargs) -> AgentProcessOutcome:
        del args, kwargs
        self.calls += 1
        return AgentProcessOutcome(
            result=None,
            exit_code=19,
            timed_out=False,
            stdout=b"",
            stderr=b"provider unavailable",
            errors=("agent_runner_nonzero_exit",),
        )


class IdentityMismatchRuntime:
    def __init__(self, profile) -> None:
        self.profile = profile
        self.delegate = DeterministicFakeAgentRuntime(profile)

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        outcome = self.delegate.run(
            request_path,
            result_path,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        assert outcome.result is not None
        mismatched = outcome.result.model_copy(update={"model_id": "other-model"})
        result_path.write_text(stable_json_text(mismatched), encoding="utf-8")
        return AgentProcessOutcome(
            result=mismatched,
            exit_code=outcome.exit_code,
            timed_out=outcome.timed_out,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            errors=outcome.errors,
        )


class ReportedFailureRuntime:
    def __init__(self, profile) -> None:
        self.profile = profile

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        del cwd, timeout_seconds, max_output_bytes
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = AgentRunnerResult(
            request_id=request["request_id"],
            stage="source_discovery",
            status="failed",
            framework_id=self.profile.framework_id,
            framework_version=self.profile.framework_version,
            execution_model_profile_id=(
                self.profile.execution_model_profile_id
            ),
            execution_model_profile_digest=(
                self.profile.execution_model_profile_digest
            ),
            model_id=self.profile.model_id,
            sandbox_profile_id=self.profile.sandbox_profile_id,
            sandbox_status="failed",
            exit_status="RunnerFailure",
            error_code="framework_failed",
            error_message="framework returned a typed failure",
        )
        result_path.write_text(stable_json_text(result), encoding="utf-8")
        return AgentProcessOutcome(
            result=result,
            exit_code=0,
            timed_out=False,
            stdout=b"",
            stderr=b"",
            errors=(),
        )


class FailedTrajectoryRuntime(ReportedFailureRuntime):
    def __init__(self, profile, *, diagnostic_secret: str) -> None:
        super().__init__(profile)
        self.diagnostic_secret = diagnostic_secret

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        outcome = super().run(
            request_path,
            result_path,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        request = json.loads(request_path.read_text(encoding="utf-8"))
        (cwd / request["trajectory_path"]).write_text(
            stable_json_text(
                {
                    "trajectory_format": "fake-agent-diagnostic-1.0",
                    "messages": [
                        {
                            "role": "exit",
                            "content": (
                                "LimitsExceeded " + self.diagnostic_secret
                            ),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return outcome


class SymlinkPayloadRuntime:
    def __init__(self, profile) -> None:
        self.profile = profile
        self.delegate = DeterministicFakeAgentRuntime(profile)

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        outcome = self.delegate.run(
            request_path,
            result_path,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        request = json.loads(request_path.read_text(encoding="utf-8"))
        payload = cwd / request["payload_path"]
        target = cwd / "payload-target.json"
        payload.replace(target)
        os.symlink(target, payload)
        return outcome


class ExtraWorkspaceFileRuntime:
    def __init__(
        self,
        profile,
        *,
        proposals: tuple[tuple[str, str], ...] = (),
        search_status: str = "complete",
    ) -> None:
        self.profile = profile
        self.delegate = DeterministicFakeAgentRuntime(
            profile,
            proposals=proposals,
            search_status=search_status,
        )

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        outcome = self.delegate.run(
            request_path,
            result_path,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        (cwd / "unexpected-agent-file.sh").write_text(
            "echo must-not-be-admitted\n",
            encoding="utf-8",
        )
        return outcome


class OversizedWorkspaceFileRuntime:
    def __init__(self, profile, size_bytes: int) -> None:
        self.profile = profile
        self.size_bytes = size_bytes
        self.delegate = DeterministicFakeAgentRuntime(profile)

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        outcome = self.delegate.run(
            request_path,
            result_path,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        (cwd / "oversized-agent-file.bin").write_bytes(b"x" * self.size_bytes)
        return outcome


class HostInputMutatingRuntime:
    def __init__(self, profile) -> None:
        self.profile = profile
        self.delegate = DeterministicFakeAgentRuntime(profile)

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        outcome = self.delegate.run(
            request_path,
            result_path,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        request_path.write_text("{}\n", encoding="utf-8")
        return outcome


class DeepTrajectoryRuntime:
    def __init__(self, profile) -> None:
        self.profile = profile
        self.delegate = DeterministicFakeAgentRuntime(profile)

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        outcome = self.delegate.run(
            request_path,
            result_path,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        request = json.loads(request_path.read_text(encoding="utf-8"))
        (cwd / request["trajectory_path"]).write_bytes(
            b"[" * 10_000 + b"0" + b"]" * 10_000
        )
        return outcome


class NonCanonicalTrajectoryRuntime:
    def __init__(self, profile) -> None:
        self.profile = profile
        self.delegate = DeterministicFakeAgentRuntime(profile)

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        outcome = self.delegate.run(
            request_path,
            result_path,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        request = json.loads(request_path.read_text(encoding="utf-8"))
        trajectory_path = cwd / request["trajectory_path"]
        parsed = json.loads(trajectory_path.read_bytes())
        trajectory_path.write_text(
            json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        return outcome


class SnapshotMutatingRuntime:
    def __init__(self, profile, repository: Path) -> None:
        self.profile = profile
        self.repository = repository
        self.delegate = DeterministicFakeAgentRuntime(profile)

    def run(
        self,
        request_path: Path,
        result_path: Path,
        *,
        cwd: Path,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> AgentProcessOutcome:
        outcome = self.delegate.run(
            request_path,
            result_path,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        (self.repository / "README.md").write_text(
            "# Mutated during source acquisition\n",
            encoding="utf-8",
        )
        return outcome


def _successful_acquisition(
    tmp_path: Path,
    *,
    proposals: tuple[tuple[str, str], ...] = (),
):
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    runtime = DeterministicFakeAgentRuntime(profile, proposals=proposals)
    output = tmp_path / "acquisition"
    bundle = acquire_sources(repository, config, profile, runtime, output)
    return repository, config, profile, output, bundle


def _load_saved_acquisition(output: Path):
    settings = config_for_tests().agent.stages["source_discovery"]
    return load_source_acquisition(
        output,
        expected_stage_settings=settings,
    )


def test_fake_agent_success_admits_one_nonstandard_source_and_replays(
    tmp_path: Path,
) -> None:
    _, _, profile, output, bundle = _successful_acquisition(
        tmp_path,
        proposals=(
            (
                "docs/team_practices.rst",
                "Contains task-independent contributor requirements.",
            ),
        ),
    )

    assert bundle.acquisition.complete
    assert bundle.acquisition.schema_version == "source-acquisition-1.2"
    assert bundle.acquisition.target_content_admission_policy == (
        "controlled-research-unredacted-target-content-1.0"
    )
    legacy = bundle.acquisition.model_dump(mode="json")
    legacy.pop("target_content_admission_policy")
    legacy["schema_version"] = "source-acquisition-1.1"
    with pytest.raises(ValidationError):
        SourceAcquisitionArtifact.model_validate(legacy)
    assert bundle.agent_run.status == "success"
    assert bundle.profile == profile
    assert [item.path for item in bundle.supplement.proposals] == [
        "docs/team_practices.rst"
    ]
    replayed = _load_saved_acquisition(output)
    assert replayed.acquisition == bundle.acquisition
    assert replayed.supplement == bundle.supplement
    assert replayed.request.profile_digest == bundle.request.profile_digest


def test_real_empty_supplement_requires_a_successful_agent_record(
    tmp_path: Path,
) -> None:
    _, _, _, output, bundle = _successful_acquisition(tmp_path)

    assert bundle.supplement.proposals == []
    assert bundle.agent_run.status == "success"
    assert bundle.agent_run.request_id == bundle.request.request_id
    assert bundle.acquisition.supplement_sha256 is not None
    assert (output / "source_supplement.json").is_file()
    assert _load_saved_acquisition(output).supplement.proposals == []


def test_live_acquisition_canonicalizes_untrusted_trajectory(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    output = tmp_path / "canonical-trajectory-acquisition"

    bundle = acquire_sources(
        repository,
        config,
        profile,
        NonCanonicalTrajectoryRuntime(profile),
        output,
    )

    expected = stable_json_text(
        json.loads((output / "agent" / "trajectory.json").read_bytes())
    ).encode("utf-8")
    assert bundle.trajectory == expected
    assert (output / "agent" / "trajectory.json").read_bytes() == expected
    assert bundle.agent_run.trajectory_sha256 == sha256_bytes(expected)
    assert bundle.acquisition.agent_run_sha256 == stable_model_digest(
        bundle.agent_run
    )
    assert _load_saved_acquisition(output) == bundle


def test_loader_normalizes_legacy_trajectory_without_mutating_acquisition(
    tmp_path: Path,
) -> None:
    _, config, _, output, bundle = _successful_acquisition(tmp_path)
    trajectory_path = output / "agent" / "trajectory.json"
    parsed = json.loads(trajectory_path.read_bytes())
    legacy_trajectory = json.dumps(
        parsed,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert legacy_trajectory != stable_json_text(parsed).encode("utf-8")
    trajectory_path.write_bytes(legacy_trajectory)
    legacy_run = bundle.agent_run.model_copy(
        update={"trajectory_sha256": sha256_bytes(legacy_trajectory)}
    )
    (output / "agent" / "run.json").write_text(
        stable_json_text(legacy_run),
        encoding="utf-8",
    )
    legacy_acquisition = bundle.acquisition.model_copy(
        update={"agent_run_sha256": stable_model_digest(legacy_run)}
    )
    (output / "source_acquisition.json").write_text(
        stable_json_text(legacy_acquisition),
        encoding="utf-8",
    )

    loaded = _load_saved_acquisition(output)
    canonical_trajectory = stable_json_text(parsed).encode("utf-8")

    assert trajectory_path.read_bytes() == legacy_trajectory
    assert SourceAcquisitionArtifact.model_validate_json(
        (output / "source_acquisition.json").read_bytes()
    ) == legacy_acquisition
    assert loaded.trajectory == canonical_trajectory
    assert loaded.agent_run.trajectory_sha256 == sha256_bytes(
        canonical_trajectory
    )
    assert loaded.acquisition.agent_run_sha256 == stable_model_digest(
        loaded.agent_run
    )
    validate_acquisition_for_build(loaded, config)


def test_task_and_response_schema_bind_quality_scope_and_dynamic_limit(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    base_config = config_for_tests()
    config = base_config.model_copy(
        update={
            "discovery": base_config.discovery.model_copy(
                update={"max_supplement_proposals": 7}
            )
        }
    )
    profile = fake_agent_profile()
    output = tmp_path / "dynamic-limit-acquisition"
    bundle = acquire_sources(
        repository,
        config,
        profile,
        DeterministicFakeAgentRuntime(profile),
        output,
    )
    schema = json.loads(bundle.response_schema_text)

    assert schema["properties"]["proposals"]["maxItems"] == 7
    assert schema["properties"]["schema_version"]["const"] == "1.2"
    assert set(schema["$defs"]["SourceProposal"]["properties"]) == {
        "path",
        "compared_baseline_paths",
        "reason",
    }
    assert "search_status" in schema["required"]
    assert "at most 7 candidates" in bundle.task_text
    assert "product, API, and CLI usage/reference docs" in bundle.task_text
    assert "repository-development command" in bundle.task_text
    assert "primary effect must be to build, check" in bundle.task_text
    assert 'prose labels the\nactivity "local development"' in bundle.task_text
    assert "not qualify merely because a test consumes it" in bundle.task_text
    assert "only repeat baseline sources" in bundle.task_text
    assert "counterfactual marginal value" in bundle.task_text
    assert "perform an explicit counterexample\nreview" in bundle.task_text
    assert "put every baseline source path\nactually used" in bundle.task_text
    assert 'means only a path listed in discovery.json\'s\nbaseline_sources' in (
        bundle.task_text
    )
    assert "Never list another supplement_candidate" in bundle.task_text
    assert "dismiss a high-potential file only" in bundle.task_text
    assert "VALID result automatically ends" in bundle.task_text
    assert "do not re-open the payload" in bundle.task_text
    assert bundle.request.prompt_version == "source-discovery-agent-1.4"
    assert bundle.agent_run.prompt_version == "source-discovery-agent-1.4"
    assert _load_saved_acquisition(output).request.prompt_version == (
        "source-discovery-agent-1.4"
    )
    assert _load_saved_acquisition(output).response_schema_text == (
        bundle.response_schema_text
    )


def test_truncated_agent_search_is_typed_incomplete(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    output = tmp_path / "truncated-search"

    with pytest.raises(
        SourceAcquisitionError,
        match="agent_source_search_truncated",
    ):
        acquire_sources(
            repository,
            config,
            profile,
            DeterministicFakeAgentRuntime(
                profile,
                search_status="truncated",
            ),
            output,
        )

    artifact = SourceAcquisitionArtifact.model_validate_json(
        (output / "source_acquisition.json").read_bytes()
    )
    assert artifact.incomplete_reasons == ["agent_source_search_truncated"]
    assert not artifact.complete
    assert not (output / "source_supplement.json").exists()


def test_truncated_search_does_not_hide_invalid_source_plan(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    output = tmp_path / "truncated-invalid-plan"

    with pytest.raises(SourceAcquisitionError) as raised:
        acquire_sources(
            repository,
            config,
            profile,
            DeterministicFakeAgentRuntime(
                profile,
                proposals=(("src/pkg/api.py", "Untrusted invalid proposal."),),
                search_status="truncated",
            ),
            output,
        )

    assert "agent_source_search_truncated" in raised.value.errors
    assert any(
        reason.startswith("agent_source_plan_invalid:")
        and "ordinary Python" in reason
        for reason in raised.value.errors
    )
    artifact = SourceAcquisitionArtifact.model_validate_json(
        (output / "source_acquisition.json").read_bytes()
    )
    assert artifact.incomplete_reasons == sorted(set(raised.value.errors))


def test_saturated_agent_search_cannot_claim_complete(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    base_config = config_for_tests()
    config = base_config.model_copy(
        update={
            "discovery": base_config.discovery.model_copy(
                update={"max_supplement_proposals": 1}
            )
        }
    )
    discovery = discover_fixture(repository, config)
    assert len(discovery.supplement_candidates) > 1
    profile = fake_agent_profile()
    output = tmp_path / "saturated-search"

    with pytest.raises(
        SourceAcquisitionError,
        match="agent_source_search_saturated",
    ):
        acquire_sources(
            repository,
            config,
            profile,
            DeterministicFakeAgentRuntime(
                profile,
                proposals=(("docs/team_practices.rst", "Inspected marginal source."),),
            ),
            output,
        )

    artifact = SourceAcquisitionArtifact.model_validate_json(
        (output / "source_acquisition.json").read_bytes()
    )
    assert artifact.incomplete_reasons == ["agent_source_search_saturated"]
    assert not artifact.complete


def test_saturated_search_does_not_hide_workspace_tampering(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    base_config = config_for_tests()
    config = base_config.model_copy(
        update={
            "discovery": base_config.discovery.model_copy(
                update={"max_supplement_proposals": 1}
            )
        }
    )
    profile = fake_agent_profile()
    output = tmp_path / "saturated-workspace-tamper"

    with pytest.raises(SourceAcquisitionError) as raised:
        acquire_sources(
            repository,
            config,
            profile,
            ExtraWorkspaceFileRuntime(
                profile,
                proposals=(("docs/team_practices.rst", "Inspected marginal source."),),
            ),
            output,
        )

    assert "agent_source_search_saturated" in raised.value.errors
    assert "agent_workspace_tree_invalid" in raised.value.errors
    artifact = SourceAcquisitionArtifact.model_validate_json(
        (output / "source_acquisition.json").read_bytes()
    )
    assert artifact.incomplete_reasons == sorted(set(raised.value.errors))


@pytest.mark.parametrize("search_status", ["complete", "truncated"])
def test_mini_finalizer_validates_v12_search_status_shape(
    tmp_path: Path,
    search_status: str,
) -> None:
    repository = copied_fixture(tmp_path)
    discovery = discover_fixture(repository, config_for_tests())
    script = tmp_path / "finalize_source_supplement.py"
    script.write_text(_FINALIZER_SOURCE, encoding="utf-8")
    draft = tmp_path / "draft.json"
    draft.write_text(
        stable_json_text(
            {
                "schema_version": "1.2",
                "snapshot_digest": discovery.repository.snapshot_digest,
                "search_status": search_status,
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    discovery_path = tmp_path / "discovery.json"
    discovery_path.write_text(stable_json_text(discovery), encoding="utf-8")
    final = tmp_path / "final.json"

    completed = subprocess.run(
        [
            sys.executable,
            script.as_posix(),
            draft.as_posix(),
            discovery_path.as_posix(),
            final.as_posix(),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.splitlines() == [
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        "VALID",
    ]
    terminal_lines = completed.stdout.lstrip().splitlines(keepends=True)
    assert "".join(terminal_lines[1:]) == SUCCESS_SUBMISSION
    assert json.loads(final.read_text(encoding="utf-8"))["search_status"] == (
        search_status
    )


def test_mini_runner_guides_compact_navigation_and_terminal_finalization() -> None:
    instance = _instance_task()
    system = _system_template()

    assert "discovery.repository.snapshot_digest" in instance
    assert "manifest-only path being absent is normal" in instance
    assert "./output in the current working directory" in instance
    assert "Do not print the whole manifest" in instance
    assert "actual baseline and candidate file" in instance
    assert "automatically ends the episode" in system
    assert "Never re-open or re-check" in system
    assert "never\nissue the mini completion echo yourself" in system


def test_mini_trajectory_rejects_manual_completion_without_valid_marker(
    tmp_path: Path,
) -> None:
    _, _, _, _, bundle = _successful_acquisition(tmp_path)
    environment = {
        "wrapper_args": ["--test-wrapper"],
        "cwd": "/test-agent-cwd",
        "executable": "/test-bwrap",
        "env": {"PAGER": "cat"},
    }
    trajectory = {
        "trajectory_format": "mini-swe-agent-1.1",
        "info": {
            "mini_version": bundle.request.framework_version,
            "exit_status": "Submitted",
            "submission": "",
            "model_stats": {"api_calls": 1, "instance_cost": 0},
            "config": {
                "model": {"model_name": bundle.request.model_id},
                "agent_type": "minisweagent.agents.default.DefaultAgent",
                "environment_type": (
                    "minisweagent.environments.extra.bubblewrap."
                    "BubblewrapEnvironment"
                ),
                "environment": environment,
            },
        },
        "messages": [],
    }
    path = tmp_path / "manual-submission-trajectory.json"
    path.write_text(stable_json_text(trajectory), encoding="utf-8")

    with pytest.raises(RunnerFailure, match="trusted finalizer") as raised:
        _load_trajectory(
            path,
            bundle.request,
            expected_wrapper_args=environment["wrapper_args"],
            expected_cwd=environment["cwd"],
            expected_executable=environment["executable"],
            expected_environment=environment["env"],
            expected_model_config=trajectory["info"]["config"]["model"],
        )

    assert raised.value.code == "terminal_submission_mismatch"


def test_mini_trajectory_accepts_only_safe_finalizer_terminal_actions(
    tmp_path: Path,
) -> None:
    _, _, _, _, bundle = _successful_acquisition(tmp_path)
    environment = {
        "wrapper_args": ["--test-wrapper"],
        "cwd": "/test-agent-cwd",
        "executable": "/test-bwrap",
        "env": {"PAGER": "cat"},
    }
    trajectory = {
        "trajectory_format": "mini-swe-agent-1.1",
        "info": {
            "mini_version": bundle.request.framework_version,
            "exit_status": "Submitted",
            "submission": SUCCESS_SUBMISSION,
            "model_stats": {"api_calls": 1, "instance_cost": 0},
            "config": {
                "model": {"model_name": bundle.request.model_id},
                "agent_type": "minisweagent.agents.default.DefaultAgent",
                "environment_type": (
                    "minisweagent.environments.extra.bubblewrap."
                    "BubblewrapEnvironment"
                ),
                "environment": environment,
            },
        },
        "messages": [
            {
                "role": "assistant",
                "extra": {
                    "actions": [{"command": FINALIZER_COMMAND}],
                    "response": {
                        "model": bundle.request.model.canonical_model_id,
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {"reasoning_content": None},
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "completion_tokens_details": {
                                "reasoning_tokens": 0
                            },
                        },
                    },
                },
            },
            {
                "role": "exit",
                "extra": {
                    "exit_status": "Submitted",
                    "submission": SUCCESS_SUBMISSION,
                },
            },
        ],
    }
    path = tmp_path / "trusted-finalizer-trajectory.json"
    path.write_text(stable_json_text(trajectory), encoding="utf-8")

    sanitized, exit_status, usage, receipt = _load_trajectory(
        path,
        bundle.request,
        expected_wrapper_args=environment["wrapper_args"],
        expected_cwd=environment["cwd"],
        expected_executable=environment["executable"],
        expected_environment=environment["env"],
        expected_model_config=trajectory["info"]["config"]["model"],
    )

    assert exit_status == "Submitted"
    assert usage.turns == 1
    assert receipt.status == "verified"
    assert "response" not in sanitized["messages"][0]["extra"]

    trajectory["messages"][0]["extra"]["actions"][0]["command"] = (
        f"cd {environment['cwd']} && {FINALIZER_COMMAND}"
    )
    path.write_text(stable_json_text(trajectory), encoding="utf-8")
    _, prefixed_exit_status, _, prefixed_receipt = _load_trajectory(
        path,
        bundle.request,
        expected_wrapper_args=environment["wrapper_args"],
        expected_cwd=environment["cwd"],
        expected_executable=environment["executable"],
        expected_environment=environment["env"],
        expected_model_config=trajectory["info"]["config"]["model"],
    )
    assert prefixed_exit_status == "Submitted"
    assert prefixed_receipt.status == "verified"

    invalid_commands = [
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        f"cd /wrong-agent-cwd && {FINALIZER_COMMAND}",
        f"echo before && {FINALIZER_COMMAND}",
        f"cd {environment['cwd']} && {FINALIZER_COMMAND} && echo after",
        f"cd {environment['cwd']}; {FINALIZER_COMMAND}",
        f"cd {environment['cwd']} || {FINALIZER_COMMAND}",
        f"cd . && {FINALIZER_COMMAND}",
        f"cd {environment['cwd']} && {FINALIZER_COMMAND} | tee output/result.txt",
        f"cd {environment['cwd']} && {FINALIZER_COMMAND} > output/result.txt",
        f"cd {environment['cwd']} && ( {FINALIZER_COMMAND} )",
        f"bash -c '{FINALIZER_COMMAND}'",
        f"env PYTHONPATH=/tmp {FINALIZER_COMMAND}",
    ]
    for invalid_command in invalid_commands:
        trajectory["messages"][0]["extra"]["actions"][0]["command"] = (
            invalid_command
        )
        path.write_text(stable_json_text(trajectory), encoding="utf-8")
        with pytest.raises(
            RunnerFailure,
            match="trusted finalizer action",
        ) as raised:
            _load_trajectory(
                path,
                bundle.request,
                expected_wrapper_args=environment["wrapper_args"],
                expected_cwd=environment["cwd"],
                expected_executable=environment["executable"],
                expected_environment=environment["env"],
                expected_model_config=trajectory["info"]["config"]["model"],
            )
        assert raised.value.code == "terminal_action_mismatch"


@pytest.mark.parametrize(
    ("model_profile_id", "expected_status"),
    [
        ("apiyi-deepseek-v4-flash-260425-nonthinking", "violated"),
        ("apiyi-deepseek-v4-flash-260425-thinking-high", "verified"),
    ],
)
def test_mini_reasoning_receipt_is_verified_before_hidden_text_is_removed(
    tmp_path: Path,
    model_profile_id: str,
    expected_status: str,
) -> None:
    _, _, _, _, bundle = _successful_acquisition(tmp_path)
    model_profile = load_execution_model_profile(model_profile_id)
    binding = agent_model_binding(model_profile)
    request = bundle.request.model_copy(
        update={
            "execution_model_profile_id": model_profile.profile_id,
            "execution_model_profile_digest": model_profile.profile_digest,
            "model_id": binding.model_id,
            "model": binding,
        }
    )
    options = _runner_options(request)
    model_config = {
        "model_name": request.model_id,
        "model_kwargs": options["model_kwargs"],
    }
    environment = {
        "wrapper_args": ["--test-wrapper"],
        "cwd": "/test-agent-cwd",
        "executable": "/test-bwrap",
        "env": {"PAGER": "cat"},
    }
    trajectory = {
        "trajectory_format": "mini-swe-agent-1.1",
        "info": {
            "mini_version": request.framework_version,
            "exit_status": "Submitted",
            "submission": SUCCESS_SUBMISSION,
            "model_stats": {"api_calls": 1, "instance_cost": 0},
            "config": {
                "model": model_config,
                "agent_type": "minisweagent.agents.default.DefaultAgent",
                "environment_type": (
                    "minisweagent.environments.extra.bubblewrap."
                    "BubblewrapEnvironment"
                ),
                "environment": environment,
            },
        },
        "messages": [
            {
                "role": "assistant",
                "reasoning_content": "private hidden trace",
                "extra": {
                    "actions": [{"command": FINALIZER_COMMAND}],
                    "response": {
                        "model": model_profile.canonical_model_id,
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "reasoning_content": "private hidden trace"
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 6,
                            "completion_tokens_details": {
                                "reasoning_tokens": 3
                            },
                        },
                    },
                },
            }
        ],
    }
    path = tmp_path / f"{model_profile_id}.json"
    path.write_text(stable_json_text(trajectory), encoding="utf-8")

    sanitized, _, _, receipt = _load_trajectory(
        path,
        request,
        expected_wrapper_args=environment["wrapper_args"],
        expected_cwd=environment["cwd"],
        expected_executable=environment["executable"],
        expected_environment=environment["env"],
        expected_model_config=model_config,
    )

    assert receipt.status == expected_status
    assert receipt.reasoning_tokens == 3
    assert receipt.reasoning_content_bytes == len("private hidden trace")
    published = stable_json_text(sanitized)
    assert "private hidden trace" not in published
    assert '"response"' not in published


def test_mini_finalizer_rejects_legacy_supplement_without_search_status(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    discovery = discover_fixture(repository, config_for_tests())
    script = tmp_path / "finalize_source_supplement.py"
    script.write_text(_FINALIZER_SOURCE, encoding="utf-8")
    draft = tmp_path / "legacy-draft.json"
    draft.write_text(
        stable_json_text(
            {
                "schema_version": "1.0",
                "snapshot_digest": discovery.repository.snapshot_digest,
                "proposals": [],
            }
        ),
        encoding="utf-8",
    )
    discovery_path = tmp_path / "discovery.json"
    discovery_path.write_text(stable_json_text(discovery), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            script.as_posix(),
            draft.as_posix(),
            discovery_path.as_posix(),
            (tmp_path / "final.json").as_posix(),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 2
    assert "supplement root keys are invalid" in completed.stderr


def test_mini_finalizer_rejects_nonbaseline_comparison_path(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    discovery = discover_fixture(repository, config_for_tests())
    script = tmp_path / "finalize_source_supplement.py"
    script.write_text(_FINALIZER_SOURCE, encoding="utf-8")
    draft = tmp_path / "draft.json"
    draft.write_text(
        stable_json_text(
            {
                "schema_version": "1.2",
                "snapshot_digest": discovery.repository.snapshot_digest,
                "search_status": "complete",
                "proposals": [
                    {
                        "path": "docs/team_practices.rst",
                        "compared_baseline_paths": ["docs/not-baseline.rst"],
                        "reason": "Adds a distinct contributor requirement.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    discovery_path = tmp_path / "discovery.json"
    discovery_path.write_text(stable_json_text(discovery), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            script.as_posix(),
            draft.as_posix(),
            discovery_path.as_posix(),
            (tmp_path / "final.json").as_posix(),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 2
    assert "unique baseline source paths" in completed.stderr


@pytest.mark.parametrize(
    "trajectory_bytes",
    [
        b"not-json",
        b"[" * 10_000 + b"0" + b"]" * 10_000,
        b'{"duplicate":1,"duplicate":2}',
        b'{"non_finite":NaN}',
    ],
    ids=["malformed", "deeply-nested", "duplicate-key", "non-finite"],
)
def test_offline_loader_rejects_digest_coordinated_invalid_trajectory(
    tmp_path: Path,
    trajectory_bytes: bytes,
) -> None:
    _, _, _, output, bundle = _successful_acquisition(tmp_path)
    trajectory = output / "agent" / "trajectory.json"
    trajectory.write_bytes(trajectory_bytes)
    agent_run = bundle.agent_run.model_copy(
        update={"trajectory_sha256": sha256_bytes(trajectory_bytes)}
    )
    (output / "agent" / "run.json").write_text(
        stable_json_text(agent_run),
        encoding="utf-8",
    )
    acquisition = bundle.acquisition.model_copy(
        update={"agent_run_sha256": stable_model_digest(agent_run)}
    )
    (output / "source_acquisition.json").write_text(
        stable_json_text(acquisition),
        encoding="utf-8",
    )

    with pytest.raises(SourceAcquisitionError, match="source_acquisition_invalid"):
        _load_saved_acquisition(output)


def test_live_acquisition_localizes_deep_trajectory_failure(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    output = tmp_path / "deep-trajectory"

    with pytest.raises(SourceAcquisitionError, match="agent_trajectory_json_invalid"):
        acquire_sources(
            repository,
            config,
            profile,
            DeepTrajectoryRuntime(profile),
            output,
        )

    assert not (output / "agent" / "workspace").exists()
    artifact = SourceAcquisitionArtifact.model_validate_json(
        (output / "source_acquisition.json").read_bytes()
    )
    assert "agent_trajectory_json_invalid" in artifact.incomplete_reasons


def test_offline_loader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    _, _, _, output, _ = _successful_acquisition(tmp_path)
    trajectory = output / "agent" / "trajectory.json"
    trajectory.unlink()
    os.mkfifo(trajectory)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from repo_constraint_memory_v3.models import AgentStageSettings; "
                "from repo_constraint_memory_v3.source_acquisition import "
                "load_source_acquisition; "
                "load_source_acquisition("
                "Path(__import__('sys').argv[1]), "
                "expected_stage_settings=AgentStageSettings())"
            ),
            output.as_posix(),
        ],
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode != 0


def test_loader_requires_trusted_stage_settings_before_large_artifacts(
    tmp_path: Path,
) -> None:
    _, config, _, output, _ = _successful_acquisition(tmp_path)
    expected = config.agent.stages["source_discovery"].model_copy(
        update={"max_trajectory_bytes": 100}
    )

    with pytest.raises(
        SourceAcquisitionError,
        match="source_acquisition_agent_settings_mismatch",
    ):
        load_source_acquisition(
            output,
            expected_stage_settings=expected,
        )


def test_failed_or_not_run_agent_cannot_become_empty_success(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    runtime = FailedRuntime(profile)
    output = tmp_path / "failed-acquisition"

    with pytest.raises(SourceAcquisitionError, match="agent_runner_nonzero_exit"):
        acquire_sources(repository, config, profile, runtime, output)

    assert runtime.calls == 1
    artifact = SourceAcquisitionArtifact.model_validate_json(
        (output / "source_acquisition.json").read_bytes()
    )
    assert not artifact.complete
    assert artifact.supplement_sha256 is None
    assert not (output / "source_supplement.json").exists()
    with pytest.raises(SourceAcquisitionError):
        _load_saved_acquisition(output)

    never_called = FailedRuntime(
        profile.model_copy(update={"profile_id": "different-profile"})
    )
    with pytest.raises(ValueError, match="runtime profile does not match"):
        acquire_sources(
            repository,
            config,
            profile,
            never_called,
            tmp_path / "profile-mismatch",
        )
    assert never_called.calls == 0


def test_runner_result_identity_mismatch_is_a_failed_acquisition(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    output = tmp_path / "identity-mismatch"

    with pytest.raises(SourceAcquisitionError, match="agent_result_schema_invalid"):
        acquire_sources(
            repository,
            config,
            profile,
            IdentityMismatchRuntime(profile),
            output,
        )

    assert not (output / "source_supplement.json").exists()
    artifact = SourceAcquisitionArtifact.model_validate_json(
        (output / "source_acquisition.json").read_bytes()
    )
    assert not artifact.complete


def test_typed_runner_failure_is_not_misclassified_as_profile_mismatch(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    output = tmp_path / "reported-failure"

    with pytest.raises(SourceAcquisitionError) as raised:
        acquire_sources(
            repository,
            config,
            profile,
            ReportedFailureRuntime(profile),
            output,
        )

    assert "agent_result_failed" in raised.value.errors
    assert "agent_result_profile_mismatch" not in raised.value.errors
    artifact = SourceAcquisitionArtifact.model_validate_json(
        (output / "source_acquisition.json").read_bytes()
    )
    assert artifact.incomplete_reasons == ["agent_result_failed"]


def test_failed_runner_preserves_bounded_redacted_diagnostic_trajectory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    environment_key = "NORM4CODE_TEST_AGENT_SECRET"
    secret = "diagnostic-secret-must-not-leak"
    monkeypatch.setenv(environment_key, secret)
    profile = fake_agent_profile().model_copy(
        update={"environment_keys": [environment_key]}
    )
    output = tmp_path / "failed-trajectory"

    with pytest.raises(SourceAcquisitionError, match="agent_result_failed"):
        acquire_sources(
            repository,
            config,
            profile,
            FailedTrajectoryRuntime(profile, diagnostic_secret=secret),
            output,
        )

    trajectory = (output / "agent" / "trajectory.json").read_bytes()
    assert secret.encode("utf-8") not in trajectory
    assert b"[REDACTED]" in trajectory
    agent_run = AgentRunArtifact.model_validate_json(
        (output / "agent" / "run.json").read_bytes()
    )
    assert agent_run.status == "failed"
    assert agent_run.trajectory_sha256 == sha256_bytes(trajectory)
    assert not (output / "agent" / "workspace").exists()


def test_payload_symlink_is_rejected_before_source_plan_admission(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    output = tmp_path / "symlink-payload"

    with pytest.raises(SourceAcquisitionError, match="agent_payload_not_regular_file"):
        acquire_sources(
            repository,
            config,
            profile,
            SymlinkPayloadRuntime(profile),
            output,
        )

    assert not (output / "source_supplement.json").exists()


def test_agent_cannot_add_files_outside_its_declared_output_tree(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()

    with pytest.raises(SourceAcquisitionError, match="workspace_tree_invalid"):
        acquire_sources(
            repository,
            config,
            profile,
            ExtraWorkspaceFileRuntime(profile),
            tmp_path / "extra-workspace-file",
        )


def test_workspace_scan_rejects_oversized_unexpected_file_before_reading_it(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    settings = config.agent.stages["source_discovery"]
    output = tmp_path / "oversized-workspace-file"

    with pytest.raises(SourceAcquisitionError, match="workspace_file_size_limit"):
        acquire_sources(
            repository,
            config,
            profile,
            OversizedWorkspaceFileRuntime(
                profile,
                settings.max_workspace_bytes + 1,
            ),
            output,
        )

    assert not (output / "agent" / "workspace").exists()


def test_runner_cannot_modify_host_owned_acquisition_inputs(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    output = tmp_path / "host-input-mutated"

    with pytest.raises(SourceAcquisitionError, match="host_agent_input_modified"):
        acquire_sources(
            repository,
            config,
            profile,
            HostInputMutatingRuntime(profile),
            output,
        )

    assert not (output / "source_supplement.json").exists()
    assert not (output / "agent" / "workspace").exists()


def test_snapshot_mutation_during_agent_run_is_incomplete(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    output = tmp_path / "snapshot-mutated"

    with pytest.raises(
        SourceAcquisitionError,
        match="target_snapshot_changed_during_agent_run",
    ):
        acquire_sources(
            repository,
            config,
            profile,
            SnapshotMutatingRuntime(profile, repository),
            output,
        )

    assert not (output / "source_supplement.json").exists()


def test_invalid_agent_proposal_fails_whole_acquisition(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    profile = fake_agent_profile()
    runtime = DeterministicFakeAgentRuntime(
        profile,
        proposals=(("src/pkg/api.py", "Ordinary Python must be rejected."),),
    )
    output = tmp_path / "invalid-proposal"

    with pytest.raises(SourceAcquisitionError, match="ordinary Python"):
        acquire_sources(repository, config, profile, runtime, output)

    assert not (output / "source_supplement.json").exists()


def test_offline_load_rejects_digest_tamper_without_calling_any_agent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, _, _, output, _ = _successful_acquisition(tmp_path)

    def forbidden_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("offline acquisition load called an external agent")

    monkeypatch.setattr(SubprocessAgentRuntime, "run", forbidden_run)
    assert _load_saved_acquisition(output).acquisition.complete

    supplement_path = output / "source_supplement.json"
    value = json.loads(supplement_path.read_text(encoding="utf-8"))
    value["proposals"] = [
        {
            "path": "docs/team_practices.rst",
            "compared_baseline_paths": [],
            "reason": "Tampered but schema-valid task-independent reason.",
        }
    ]
    supplement_path.write_text(stable_json_text(value), encoding="utf-8")
    with pytest.raises(SourceAcquisitionError, match="digest_mismatch"):
        _load_saved_acquisition(output)


def test_offline_load_rejects_symlinked_saved_trajectory(tmp_path: Path) -> None:
    _, _, _, output, _ = _successful_acquisition(tmp_path)
    trajectory = output / "agent" / "trajectory.json"
    target = tmp_path / "saved-trajectory-target.json"
    trajectory.replace(target)
    os.symlink(target, trajectory)

    with pytest.raises(SourceAcquisitionError, match="not_regular_file"):
        _load_saved_acquisition(output)


def test_offline_load_rejects_symlinked_canonical_json_input(
    tmp_path: Path,
) -> None:
    _, _, _, output, _ = _successful_acquisition(tmp_path)
    task = output / "agent" / "task.md"
    target = tmp_path / "saved-task-target.md"
    task.replace(target)
    os.symlink(target, task)

    with pytest.raises(SourceAcquisitionError, match="not_regular_file"):
        _load_saved_acquisition(output)


def test_offline_load_rejects_coordinated_request_identity_tamper(
    tmp_path: Path,
) -> None:
    _, _, _, output, bundle = _successful_acquisition(tmp_path)
    replacement_id = "AGENTRUN-" + "f" * 24
    assert replacement_id != bundle.request.request_id

    request_path = output / "agent" / "request.json"
    request_raw = json.loads(request_path.read_text(encoding="utf-8"))
    request_raw["request_id"] = replacement_id
    request_path.write_text(stable_json_text(request_raw), encoding="utf-8")

    with pytest.raises(
        SourceAcquisitionError,
        match="agent request identity",
    ):
        _load_saved_acquisition(output)


def test_agent_workspace_exposes_only_admitted_text_sources(
    tmp_path: Path,
) -> None:
    _, _, _, output, bundle = _successful_acquisition(tmp_path)
    assert bundle.request.workspace_inputs
    assert all(
        not item.path.endswith((".py", ".pyi"))
        for item in bundle.request.workspace_inputs
    )
    assert {item.role for item in bundle.request.workspace_inputs} == {
        "baseline_source",
        "supplement_candidate",
    }
    assert not (output / "agent" / "workspace").exists()
