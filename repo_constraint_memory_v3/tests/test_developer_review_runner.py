from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from repo_constraint_memory_v3.integrations import developer_review as runner
from repo_constraint_memory_v3.review.service import create_review, load_review
from test_consumer_cli import _mock_source_run
from repo_constraint_memory_v3.llm import MockLLMClient


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    original = MockLLMClient.complete
    def multiple_review_candidates(self, system_prompt, user_prompt):
        response = original(self, system_prompt, user_prompt)
        request = json.loads(user_prompt)
        request = request.get("original_request", request)
        if request.get("stage") == "explicit_semantic_verify":
            payload = json.loads(response.content)
            for result in payload["results"]:
                result["evidence_entailment"] = "uncertain"
            return response.model_copy(update={"content": json.dumps(payload)})
        if request.get("stage") == "unified_case_verify":
            payload = json.loads(response.content)
            for finding in payload["member_findings"]:
                finding["rule_support"] = "uncertain"
                finding["closed_local_reason_codes"] = []
            return response.model_copy(update={"content": json.dumps(payload)})
        return response
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MockLLMClient, "complete", multiple_review_candidates)
        result = _mock_source_run(tmp_path_factory.mktemp("automatic-source"), incomplete=True)
    assert sum(claim.epistemic_status == "review" for claim in result[2].claims) > 1
    return result


def test_subscription_preflight_no_api_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "operator-secret-never-forward")
    monkeypatch.setenv("CODEX_API_KEY", "operator-secret-never-forward")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://not-openai.invalid")
    monkeypatch.setenv("CODEX_THREAD_ID", "parent-thread")
    environment = runner._environment()
    assert not {"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_THREAD_ID"} & environment.keys()
    monkeypatch.setattr(runner.shutil, "which", lambda *args, **kwargs: "/mock/codex")
    seen = []
    def fake(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, "codex-cli 0.153.4\n" if "--version" in command else "", "" if "--version" in command else "Logged in using an API key\n")
    monkeypatch.setattr(runner.subprocess, "run", fake)
    with pytest.raises(ValueError, match="ChatGPT subscription"):
        runner._preflight("codex", environment, runner.ReviewExecutionProfile())
    assert all("exec" not in command for command in seen)


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(runner, "_preflight", lambda *args: "/mock/codex")
    monkeypatch.setattr(runner, "verify_sandbox", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "sandbox_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_runtime_environment", lambda auto, environment: environment)


def _round(source, directory):
    source_run, repository, _ = source
    create_review(source_run, repository_root=repository, output_directory=directory)
    return repository


def _telemetry(*, failed=False):
    return {
        "thread_id": "thread-review-001",
        "turn_failed": failed,
        "tool_policy_violation": False,
        "reported_models": [],
        "reported_reasoning_efforts": [],
        "usage": [{"input_tokens": 100, "output_tokens": 10, "reasoning_output_tokens": 5}],
        "identity_attestation": "unavailable",
        "exit_code": 1 if failed else 0,
        "stop_reason": "subscription_limit" if failed else None,
    }


def _fake_episode(calls, *, failed=False, limit=None, mutate=None):
    def fake(command, prompt, environment, event_path, stderr_path):
        work = Path(command[command.index("-C") + 1])
        inputs = work.parent / "inputs"
        snapshot = work.parent / "repository"
        candidates = json.loads((inputs / "claims.json").read_text())
        assert candidates
        for candidate in candidates:
            assert all(set(location) == {"path", "start_line", "end_line"} for location in candidate["repository_locations"])
            assert "support" not in candidate and "counterevidence" not in candidate
        calls.append((command, prompt, {item["claim_id"] for item in candidates}))
        assert "--ignore-user-config" in command and "--ignore-rules" in command
        assert 'features.multi_agent=true' in command
        assert 'features.code_mode_host=true' in command
        assert 'agents.default_subagent_model="gpt-5.6-sol"' in command
        assert 'agents.default_subagent_reasoning_effort="high"' in command
        assert 'features.shell_tool=false' not in command
        assert 'features.unified_exec=false' not in command
        assert 'forced_login_method="chatgpt"' in command
        assert 'model_provider="openai"' in command
        serialized = json.dumps(candidates)
        for forbidden in ("review_facets", "epistemic_status", "dependency_status", "source_run"):
            assert forbidden not in serialized
        path = next(path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*.py") if path.is_file())
        selected = candidates if limit is None else candidates[:limit]
        for index, candidate in enumerate(selected):
            result = {
                "claim_id": candidate["claim_id"], "decision": "accept",
                "rationale": "The repository implementation supports delivery of this rule.",
                "evidence": [{"path": path, "start_line": 1, "end_line": 1, "description": "Current implementation."}],
                "reviewer": f"worker-{index % 4}",
            }
            destination = work / "results" / result["reviewer"] / f'{result["claim_id"]}.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if mutate:
                mutate(index, result, destination, work)
            else:
                destination.write_text(json.dumps(result))
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_text('{"type":"thread.started","thread_id":"thread-review-001"}\n')
        stderr_path.write_text("")
        return _telemetry(failed=failed)
    return fake


def test_profile_has_concurrency_without_manual_quotas():
    profile = runner.ReviewExecutionProfile()
    assert profile.max_subagents == 4
    assert profile.protocol_version == "developer-review-2.0"
    assert profile.tool_policy == "isolated-shell-subagents-1.0"
    for removed in ("max_claims", "per_claim_seconds", "total_seconds", "max_attempts_per_claim", "max_tool_calls", "max_read_chars", "max_event_bytes"):
        with pytest.raises(ValueError):
            runner.ReviewExecutionProfile(**{removed: 1})
    with pytest.raises(ValueError):
        runner.ReviewExecutionProfile(protocol_version="developer-review-1.0")


def test_mock_is_independent_resumable_and_does_not_call_codex(source, tmp_path, monkeypatch):
    directory = tmp_path / "round"
    repository = _round(source, directory)
    monkeypatch.setattr(runner, "_preflight", lambda *args: pytest.fail("mock must not inspect auth"))
    monkeypatch.setattr(runner, "_execute", lambda *args: pytest.fail("mock must not execute Codex"))
    first = runner.run_codex_review(directory, repository, mock=True)
    assert first["status"] == "complete" and first["accepted"] == 0
    assert first["rejected"] == first["planned"] > 0
    second = runner.run_codex_review(directory, repository, mock=True)
    assert second["episodes"] == first["episodes"] == 1
    for profile, mock in ((runner.ReviewExecutionProfile(model="other-review-model"), True), (None, False)):
        with pytest.raises(ValueError, match="profile/mode"):
            runner.run_codex_review(directory, repository, profile=profile, mock=mock)


def test_one_episode_reviews_all_claims_and_preserves_shell_evidence(source, tmp_path, monkeypatch, fake_runtime):
    directory = tmp_path / "round"
    repository = _round(source, directory)
    calls = []
    monotonic = iter([100.0, 112.5])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(runner, "_execute", _fake_episode(calls))
    complete = runner.run_codex_review(directory, repository)
    assert complete["status"] == "complete"
    assert complete["timing"] == {"status": "complete", "elapsed_seconds": 12.5, "timed_episodes": 1, "episodes": 1}
    assert complete["accepted"] == complete["planned"] > 1
    assert len(calls) == complete["episodes"] == 1
    saved = load_review(directory, repository_root=repository).latest_decisions()
    assert all(item.evidence[0].content_digest for item in saved.values())
    for item in saved.values():
        receipt = json.loads((directory / item.execution_ref).read_text())
        assert receipt["identity_attestation"] == "unavailable"
    episode = json.loads(next((directory / "automatic").glob("episode-*.json")).read_text())
    assert episode["timing"]["clock"] == "monotonic"
    assert episode["timing"]["scope"] == "review_episode"
    assert episode["timing"]["elapsed_seconds"] == 12.5
    runner.run_codex_review(directory, repository)
    assert len(calls) == 1


def test_legacy_episode_without_timing_reports_unavailable(source, tmp_path, monkeypatch, fake_runtime):
    directory = tmp_path / "round"
    repository = _round(source, directory)
    monkeypatch.setattr(runner, "_execute", _fake_episode([]))
    runner.run_codex_review(directory, repository)
    receipt = next((directory / "automatic").glob("episode-*.json"))
    payload = json.loads(receipt.read_text())
    payload.pop("timing")
    receipt.write_text(json.dumps(payload))
    summary = runner.run_codex_review(directory, repository)
    assert summary["timing"] == {"status": "unavailable", "elapsed_seconds": None, "timed_episodes": 0, "episodes": 1}


def test_failed_episode_imports_partial_results_and_resumes_remaining(source, tmp_path, monkeypatch, fake_runtime):
    directory = tmp_path / "round"
    repository = _round(source, directory)
    calls = []
    monkeypatch.setattr(runner, "_execute", _fake_episode(calls, failed=True, limit=1))
    partial = runner.run_codex_review(directory, repository)
    assert partial["status"] == "partial" and partial["stop_reason"] == "subscription_limit"
    assert partial["accepted"] == 1 and partial["unprocessed"] > 0
    first = load_review(directory).latest_decisions()
    monkeypatch.setattr(runner, "_execute", _fake_episode(calls))
    complete = runner.run_codex_review(directory, repository)
    assert complete["status"] == "complete"
    assert complete["accepted"] == complete["planned"]
    assert complete["episodes"] == len(calls) == 2
    assert "resume" in calls[1][0]
    assert calls[1][2] == calls[0][2] - set(first)
    latest = load_review(directory).latest_decisions()
    for claim_id, decision in first.items():
        assert latest[claim_id] == decision


@pytest.mark.parametrize("failure", ["bad_path", "bad_lines", "unbound", "symlink", "conflict", "unprocessed"])
def test_invalid_or_incomplete_result_does_not_block_other_claims(source, tmp_path, monkeypatch, fake_runtime, failure):
    directory = tmp_path / "round"
    repository = _round(source, directory)
    def mutate(index, result, destination, work):
        if index != 0:
            destination.write_text(json.dumps(result))
            return
        if failure == "bad_path":
            result["evidence"][0]["path"] = "../private"
        elif failure == "bad_lines":
            result["evidence"][0]["end_line"] = 10000000
        elif failure == "unbound":
            result["claim_id"] = "CLAIM-unknown"
        elif failure == "symlink":
            outside = work.parent.parent / "outside.json"
            outside.write_text(json.dumps(result))
            destination.symlink_to(outside)
            return
        elif failure == "conflict":
            duplicate = work / "results" / "other-worker" / destination.name
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            duplicate.write_text(json.dumps({**result, "decision": "reject", "reviewer": "other-worker"}))
        elif failure == "unprocessed":
            result["decision"] = "unprocessed"
            result["evidence"] = []
        destination.write_text(json.dumps(result))
    monkeypatch.setattr(runner, "_execute", _fake_episode([], mutate=mutate))
    partial = runner.run_codex_review(directory, repository)
    assert partial["status"] == "partial"
    assert partial["accepted"] == partial["planned"] - 1
    assert partial["unprocessed"] == 1
    assert partial["rejected"] == 0


def test_model_cannot_overwrite_concurrent_developer_decision(source, tmp_path, monkeypatch, fake_runtime):
    directory = tmp_path / "round"
    repository = _round(source, directory)
    first_id = next(iter(load_review(directory).claim_digests))
    fake = _fake_episode([])
    def concurrent(*args):
        runner.service.save_decision(directory, repository_root=repository, claim_id=first_id, decision="reject", reviewer_id="repository-developer", rationale="Developer decided while the model was reading.")
        return fake(*args)
    monkeypatch.setattr(runner, "_execute", concurrent)
    result = runner.run_codex_review(directory, repository)
    assert result["status"] == "complete"
    decided = load_review(directory).latest_decisions()[first_id]
    assert decided.reviewer_id == "repository-developer" and decided.decision == "reject" and decided.version == 1
    assert result["accepted"] == result["planned"] - 1


def test_process_streaming_has_no_old_event_byte_limit(tmp_path):
    event_path = tmp_path / "events.jsonl"
    stderr_path = tmp_path / "stderr.log"
    script = "import json; print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'x' * 8100000}})); print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 1, 'output_tokens': 2}}))"
    telemetry = runner._execute([sys.executable, "-c", script], "", runner._environment(), event_path, stderr_path)
    assert telemetry["exit_code"] == 0 and telemetry["stop_reason"] is None
    assert event_path.stat().st_size > 8000000
    assert telemetry["usage"][0]["output_tokens"] == 2


def test_resume_rejects_tampered_state_before_model_execution(source, tmp_path, monkeypatch, fake_runtime):
    directory = tmp_path / "round"
    repository = _round(source, directory)
    monkeypatch.setattr(runner, "_execute", _fake_episode([], failed=True, limit=0))
    runner.run_codex_review(directory, repository)
    state_path = directory / "automatic" / "state.json"
    state = json.loads(state_path.read_text())
    state["episodes"][0]["thread_id"] = "another-private-thread"
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(runner, "_execute", lambda *args: pytest.fail("tampered state must not resume"))
    with pytest.raises(ValueError):
        runner.run_codex_review(directory, repository)


def test_telemetry_allows_shell_and_collaboration_but_not_external_tools():
    allowed = [
        {"type": "item.completed", "item": {"type": "command_execution", "command": "rg norm repository", "aggregated_output": "private content omitted"}},
        {"type": "item.completed", "item": {"type": "file_change", "changes": []}},
        {"type": "item.completed", "item": {"type": "collab_tool_call", "tool": "spawn_agent", "sender_thread_id": "main", "receiver_thread_ids": ["child"]}},
    ]
    telemetry = runner._telemetry("\n".join(json.dumps(item) for item in allowed))
    assert not telemetry["tool_policy_violation"]
    assert "private content omitted" not in json.dumps(telemetry)
    for kind in ("web_search", "mcp_tool_call"):
        telemetry = runner._telemetry(json.dumps({"type": "item.completed", "item": {"type": kind}}))
        assert telemetry["tool_policy_violation"]


@pytest.mark.parametrize("change", [
    {"reported_models": ["different-review-model"]},
    {"reported_reasoning_efforts": ["low"]},
    {"tool_policy_violation": True},
])
def test_known_identity_or_tool_violation_cannot_import_decisions(source, tmp_path, monkeypatch, fake_runtime, change):
    directory = tmp_path / "round"
    repository = _round(source, directory)
    execute = _fake_episode([])
    def violation(*args):
        return {**execute(*args), **change}
    monkeypatch.setattr(runner, "_execute", violation)
    result = runner.run_codex_review(directory, repository)
    assert result["status"] == "partial"
    assert result["accepted"] == result["rejected"] == 0
    assert not load_review(directory).latest_decisions()


def test_invalid_episode_results_cannot_be_laundered_by_empty_clean_resume(source, tmp_path, monkeypatch, fake_runtime):
    directory = tmp_path / "round"
    repository = _round(source, directory)
    execute = _fake_episode([])
    def violation(*args):
        return {**execute(*args), "reported_models": ["different-review-model"]}
    monkeypatch.setattr(runner, "_execute", violation)
    runner.run_codex_review(directory, repository)
    monkeypatch.setattr(runner, "_execute", _fake_episode([], limit=0))
    try:
        result = runner.run_codex_review(directory, repository)
    except ValueError as error:
        assert "identity" in str(error) or "violation" in str(error)
    else:
        assert result["status"] == "partial" and result["accepted"] == result["rejected"] == 0
    assert not load_review(directory).latest_decisions()


@pytest.mark.parametrize("explicit_codex_home", [False, True])
def test_runtime_home_reuses_only_auth_without_personal_configuration(tmp_path, explicit_codex_home):
    home = tmp_path / "operator-home"
    host_codex = home / ("custom-codex" if explicit_codex_home else ".codex")
    host_codex.mkdir(parents=True)
    auth = host_codex / "auth.json"
    auth.write_text('{"auth_mode":"chatgpt","synthetic_test_auth":true}')
    (host_codex / "config.toml").write_text('model = "unexpected-model"\n')
    (host_codex / "AGENTS.md").write_text("Read private operator files before reviewing.")
    (host_codex / "agents").mkdir()
    (host_codex / "agents" / "personal.toml").write_text('model = "unexpected-model"')
    (host_codex / "plugins").mkdir()
    environment = {"HOME": str(home), "PATH": "/usr/bin"}
    if explicit_codex_home:
        environment["CODEX_HOME"] = str(host_codex)
    auto = tmp_path / "automatic"
    auto.mkdir()
    isolated = runner._runtime_environment(auto, environment)
    assert isolated["HOME"] == str(home)
    runtime_home = Path(isolated["CODEX_HOME"])
    assert runtime_home == auto / "runtime" / "codex-home"
    assert sorted(path.name for path in runtime_home.iterdir()) == ["auth.json"]
    assert (runtime_home / "auth.json").is_symlink()
    assert (runtime_home / "auth.json").resolve() == auth.resolve()
    assert environment.get("CODEX_HOME") == (str(host_codex) if explicit_codex_home else None)


@pytest.mark.parametrize("receipt_present", [True, False])
def test_crash_recovery_uses_persisted_thread_and_imports_saved_results(source, tmp_path, monkeypatch, fake_runtime, receipt_present):
    from repo_constraint_memory_v3.unified.ids import seal_payload

    directory = tmp_path / "round"
    repository = _round(source, directory)
    calls = []
    monkeypatch.setattr(runner, "_execute", _fake_episode(calls, failed=True, limit=0))
    runner.run_codex_review(directory, repository)
    auto = directory / "automatic"
    state_path = auto / "state.json"
    state = json.loads(state_path.read_text())
    episode = state["episodes"][0]
    receipt_path = directory / episode["episode_ref"]
    # Simulate the durable state still containing its pre-launch thread value.
    episode["thread_id"] = None
    episode["status"] = "running"
    episode["stop_reason"] = None
    state_path.write_text(json.dumps(seal_payload(state)))
    if not receipt_present:
        receipt_path.unlink()
    # One worker persisted a result immediately before the controller died.
    candidates = json.loads((auto / "inputs" / "claims.json").read_text())
    claim_id = candidates[0]["claim_id"]
    evidence_path = next(path.relative_to(auto / "repository").as_posix() for path in (auto / "repository").rglob("*.py"))
    saved = auto / "work" / "results" / "worker-before-crash" / f"{claim_id}.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({"claim_id": claim_id, "decision": "accept", "reviewer": "worker-before-crash", "rationale": "Saved before process interruption.", "evidence": [{"path": evidence_path, "start_line": 1, "end_line": 1, "description": "Current source."}]}))
    monkeypatch.setattr(runner, "_execute", _fake_episode(calls))
    result = runner.run_codex_review(directory, repository)
    assert result["status"] == "complete"
    assert calls[1][0][calls[1][0].index("resume") + 1] == "thread-review-001"
    assert claim_id not in calls[1][2]
    decision = load_review(directory).latest_decisions()[claim_id]
    assert decision.rationale == "Saved before process interruption."
    if not receipt_present:
        assert json.loads(receipt_path.read_text())["recovered_after_interruption"]


def _model_batch(directory, repository):
    import hashlib
    from repo_constraint_memory_v3.review.models import ReviewDecision, ReviewEvidence
    session = load_review(directory, repository_root=repository)
    location = next(path for path in repository.rglob("*.py") if path.is_file())
    evidence = runner.service._check_evidence(repository, [ReviewEvidence(path=location.relative_to(repository).as_posix(), start_line=1, end_line=1)], bind=True)
    decisions = []
    for index, (claim_id, digest) in enumerate(session.claim_digests.items()):
        receipt = directory / "automatic" / f"batch-{index}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        runner._write(receipt, {"claim_id": claim_id, "mock": True})
        decisions.append(ReviewDecision(claim_id=claim_id, claim_digest=digest, decision="accept", reviewer_kind="model", reviewer_id="mock:batch-reviewer", rationale="Synthetic model batch test.", evidence=evidence, execution_ref=receipt.relative_to(directory).as_posix(), execution_digest=hashlib.sha256(receipt.read_bytes()).hexdigest(), decided_at="2026-09-09T00:00:00+00:00", version=1))
    return decisions


def test_batch_model_import_preserves_human_decision_and_binds_source_once(source, tmp_path, monkeypatch):
    directory = tmp_path / "round"
    repository = _round(source, directory)
    decisions = _model_batch(directory, repository)
    runner.service.save_decision(directory, repository_root=repository, claim_id=decisions[0].claim_id, decision="reject", reviewer_id="human", rationale="Decided before batch acquired the lock.")
    calls = {"bound": 0, "unchanged": 0, "write": 0}
    for name, key in (("_bound_inputs", "bound"), ("_check_unchanged", "unchanged"), ("_write_session", "write")):
        original = getattr(runner.service, name)
        def counted(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(runner.service, name, counted)
    updated, conflicts = runner.service.save_model_decisions(directory, repository_root=repository, decisions=decisions)
    assert conflicts == [decisions[0].claim_id]
    assert calls == {"bound": 1, "unchanged": 1, "write": 1}
    latest = updated.latest_decisions()
    assert latest[decisions[0].claim_id].reviewer_id == "human"
    assert latest[decisions[0].claim_id].decision == "reject"
    assert all(latest[item.claim_id] == item for item in decisions[1:])
    assert updated.progress()["unprocessed"] == 0


@pytest.mark.parametrize("invalid", ["receipt", "evidence", "claim", "reviewer", "version", "duplicate"])
def test_batch_model_import_is_atomic_when_input_binding_is_invalid(source, tmp_path, invalid):
    directory = tmp_path / "round"
    repository = _round(source, directory)
    decisions = _model_batch(directory, repository)
    target = decisions[-1]
    if invalid == "receipt":
        decisions[-1] = target.model_copy(update={"execution_digest": "0" * 64})
    elif invalid == "evidence":
        decisions[-1] = target.model_copy(update={"evidence": [target.evidence[0].model_copy(update={"content_digest": "0" * 64})]})
    elif invalid == "claim":
        decisions[-1] = target.model_copy(update={"claim_digest": "0" * 64})
    elif invalid == "reviewer":
        decisions[-1] = target.model_copy(update={"reviewer_kind": "developer"})
    elif invalid == "version":
        decisions[-1] = target.model_copy(update={"version": 2})
    else:
        decisions.append(target)
    original = (directory / "session.json").read_bytes()
    with pytest.raises(ValueError):
        runner.service.save_model_decisions(directory, repository_root=repository, decisions=decisions)
    assert (directory / "session.json").read_bytes() == original
