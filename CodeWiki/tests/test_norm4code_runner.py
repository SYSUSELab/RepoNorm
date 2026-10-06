"""Offline checks for fixed-snapshot preparation and probe success semantics."""

import asyncio
import json
import subprocess

import httpx
import pytest

from integration.run import backend_config, mock_response, probe, snapshot
from integration.runtime import AdapterError, Limits, ProfileRuntime
from repo_constraint_memory_v3.model_profiles import load_execution_model_profile


def test_generation_uses_native_cli_defaults(tmp_path):
    from codewiki.cli.models.config import Configuration
    runtime = ProfileRuntime(load_execution_model_profile(), "offline-key",
                             tmp_path / "calls", Limits())
    config = backend_config(runtime, tmp_path / "repo", tmp_path / "docs")
    expected = Configuration(base_url="unused", main_model="unused", cluster_model="unused")
    for name in ("max_tokens", "max_token_per_module", "max_token_per_leaf_module",
                 "max_depth", "use_gitignore", "prompt_caching"):
        assert getattr(config, name) == getattr(expected, name)
    assert all(value is None for value in runtime.limits.model_dump().values())
    from openai import OpenAI
    native = OpenAI(api_key="offline-key")
    try:
        assert runtime.create_client().timeout == native.timeout
        assert runtime.create_client().max_retries == native.max_retries
    finally:
        native.close()
        asyncio.run(runtime.close())


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "example.py").write_text("answer = 42\n", encoding="utf-8")
    (repo / ".gitignore").write_text("ignored-secret\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "example.py", ".gitignore"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "Fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return repo, commit


def test_snapshot_binds_exact_commit_and_content(repository):
    repo, commit = repository
    before = snapshot(repo, commit)
    assert before["tracked_files"] == 2
    assert snapshot(repo, commit) == before
    with pytest.raises(AdapterError, match="commit_mismatch"):
        snapshot(repo, "0" * 40)
    (repo / "example.py").write_text("answer = 41\n", encoding="utf-8")
    with pytest.raises(AdapterError, match="not_clean"):
        snapshot(repo, commit)


@pytest.mark.parametrize("name", ["untracked", "ignored-secret"])
def test_snapshot_rejects_extra_local_content(repository, name):
    repo, commit = repository
    (repo / name).write_text("must not enter model input", encoding="utf-8")
    with pytest.raises(AdapterError, match="not_clean"):
        snapshot(repo, commit)


def test_probe_checks_connectivity_and_tool_use_not_literal_model_wording(tmp_path):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        response = mock_response(request)
        payload = response.json()
        if payload["choices"][0]["message"]["content"]:
            payload["choices"][0]["message"]["content"] = "Ready."
        return httpx.Response(200, json=payload)
    runtime = ProfileRuntime(
        load_execution_model_profile(), "offline-key", tmp_path / "calls", Limits(),
        sync_transport=httpx.MockTransport(handler), async_transport=httpx.MockTransport(handler),
        execution_kind="offline_mock",
    )
    config = backend_config(runtime, tmp_path / "repo", tmp_path / "docs")
    assert config.max_token_per_leaf_module == 16000
    assert config.execution_profile_adapter is runtime
    assert config.llm_api_key == "host-adapter-owned"
    async def run():
        try:
            await probe(runtime, config)
        finally:
            await runtime.close()
    asyncio.run(run())
    assert len(requests) == 3
    assert all(c.status == "success" for c in runtime.calls)
    assert runtime.calls[-1].assistant_reasoning_replay


def test_probe_does_not_pass_when_model_skips_tool(tmp_path):
    def handler(request):
        payload = mock_response(request).json()
        payload["choices"][0].update(
            message={"role": "assistant", "content": "READY"}, finish_reason="stop",
        )
        return httpx.Response(200, json=payload)
    runtime = ProfileRuntime(
        load_execution_model_profile(), "offline-key", tmp_path / "calls", Limits(),
        sync_transport=httpx.MockTransport(handler), async_transport=httpx.MockTransport(handler),
    )
    async def run():
        try:
            await probe(runtime, backend_config(runtime, tmp_path / "repo", tmp_path / "docs"))
        finally:
            await runtime.close()
    with pytest.raises(AdapterError, match="tool_not_called"):
        asyncio.run(run())


def test_disabled_mermaid_is_explicit_and_never_calls_remote_parser(tmp_path, monkeypatch):
    from codewiki.src.be import utils
    monkeypatch.setenv("MERMAID_VALIDATE", "0")
    def forbidden(*args, **kwargs):
        pytest.fail("disabled validation must not reach a parser or network")
    monkeypatch.setattr(utils, "validate_single_diagram", forbidden)
    doc = tmp_path / "diagram.md"
    doc.write_text("```mermaid\ngraph TD\n A-->B\n```\n", encoding="utf-8")
    result = asyncio.run(utils.validate_mermaid_diagrams(str(doc), doc.name))
    assert "skipped" in result
    assert "unverified" in result
    assert "correct" not in result


def test_tokenizer_cache_stages_only_verified_local_bytes(tmp_path, monkeypatch):
    from integration import run
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(source))
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(AdapterError, match="tokenizer_cache_missing"):
        run.stage_tokenizer_cache(output)
    data = b"fixture tokenizer asset"
    (source / run.TOKENIZER_CACHE_NAME).write_bytes(data)
    with pytest.raises(AdapterError, match="hash_mismatch"):
        run.stage_tokenizer_cache(output)
    monkeypatch.setattr(run, "TOKENIZER_SHA256", run.digest(data))
    result = run.stage_tokenizer_cache(output)
    assert (result / run.TOKENIZER_CACHE_NAME).read_bytes() == data
