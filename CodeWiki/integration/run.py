"""Build CodeWiki knowledge for frozen CooperBench repository snapshots.

This operator entrypoint does not use or write ~/.codewiki credentials/settings.
The worker runs with a read-only filesystem except its new run directory and /tmp.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import dotenv_values

from integration.runtime import AdapterError, Limits, ProfileRuntime, digest, now
from repo_constraint_memory_v3.model_profiles import (
    load_execution_model_profile, load_profile_artifact, stable_profile_text,
)


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "outputs/codewiki"
# Frozen cl100k_base asset used by the installed tiktoken encoder. The worker
# has a private /tmp, so explicitly stage the verified host cache without HTTP.
TOKENIZER_CACHE_NAME = hashlib.sha1(
    b"https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
).hexdigest()
TOKENIZER_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
# One registry for all benchmark snapshots; generation sees no task artifacts.
_manifest = json.loads((ROOT / "benchmark/manifest.json").read_text())
SNAPSHOTS = {row["id"]: (row["repository_url"], row["commit"],
             sum(t["snapshot_id"] == row["id"] for t in _manifest["tasks"]))
             for row in _manifest["snapshots"]}
TARGET = "tik-4560a8896f5f"
ORIGIN, COMMIT, _ = SNAPSHOTS[TARGET]


def repository_for(target: str) -> Path:
    if target not in SNAPSHOTS:
        raise AdapterError("unregistered_target")
    return ROOT / "outputs/repositories" / target


def registered_snapshot(target: str) -> dict:
    repository = repository_for(target)
    if repository.is_symlink():
        raise AdapterError("target_repository_symlink")
    origin, commit, _ = SNAPSHOTS[target]
    state = snapshot(repository, commit)
    if git(repository, "remote", "get-url", "origin").strip().rstrip("/").removesuffix(".git") != origin.removesuffix(".git"):
        raise AdapterError("unexpected_target_origin")
    return state


def prepare_snapshot(target: str) -> dict:
    repository = repository_for(target)
    # Never repair, reset or fetch into an existing shared checkout.
    if repository.exists() or repository.is_symlink():
        return registered_snapshot(target)
    subprocess.run([sys.executable, str(ROOT / "runner/dataset.py"),
                    "clone", "--snapshot", target], check=True)
    return registered_snapshot(target)


def git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(repository), *args],
        capture_output=True, timeout=30, check=True,
    )
    return result.stdout.decode("utf-8")


def snapshot(repository: Path, expected: str = COMMIT) -> dict:
    if not (repository / ".git").is_dir():
        raise AdapterError("target_clone_missing")
    if git(repository, "rev-parse", "HEAD").strip() != expected:
        raise AdapterError("snapshot_commit_mismatch")
    if git(repository, "status", "--porcelain", "--untracked-files=all", "--ignored"):
        raise AdapterError("target_worktree_not_clean")
    files = git(repository, "ls-files", "-z").split("\0")[:-1]
    if len(files) > 10000:
        raise AdapterError("snapshot_file_limit")
    hasher = hashlib.sha256()
    total = 0
    for name in sorted(files):
        path = repository / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 50_000_000:
            raise AdapterError("unsupported_snapshot_file")
        content = path.read_bytes()
        total += len(content)
        if total > 100_000_000:
            raise AdapterError("snapshot_byte_limit")
        hasher.update(name.encode() + b"\0" + hashlib.sha256(content).digest())
    return {"commit": expected, "tracked_files": len(files), "tracked_bytes": total,
            "content_digest": hasher.hexdigest()}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stage_tokenizer_cache(run_dir: Path) -> Path:
    cache_root = (os.environ.get("TIKTOKEN_CACHE_DIR") or os.environ.get("DATA_GYM_CACHE_DIR")
                  or str(Path(tempfile.gettempdir()) / "data-gym-cache"))
    source = Path(cache_root) / TOKENIZER_CACHE_NAME
    if not source.is_file() or source.stat().st_size > 5_000_000:
        raise AdapterError("tokenizer_cache_missing_prepare_cl100k_base_in_codewiki_environment")
    data = source.read_bytes()
    if digest(data) != TOKENIZER_SHA256:
        raise AdapterError("tokenizer_cache_hash_mismatch")
    destination = run_dir / "tokenizer-cache"
    destination.mkdir()
    (destination / TOKENIZER_CACHE_NAME).write_bytes(data)
    return destination


def backend_config(runtime: ProfileRuntime, repository: Path, output: Path):
    from codewiki.cli.models.config import Configuration
    defaults = Configuration(
        base_url=runtime.profile.provider.base_url,
        main_model=runtime.wire["model"], cluster_model=runtime.wire["model"],
        fallback_model=runtime.wire["model"],
    )
    config = defaults.to_backend_config(str(repository), str(output), "host-adapter-owned")
    if runtime.limits.max_output_tokens is not None:
        config.max_tokens = runtime.limits.max_output_tokens  # Diagnostic probe only.
    config.execution_profile_adapter = runtime
    return config


async def probe(runtime: ProfileRuntime, config) -> None:
    from codewiki.src.be.llm_services import call_llm, create_fallback_models
    from pydantic_ai import Agent
    answer = call_llm("Reply with exactly READY.", config, model=config.cluster_model)
    if not answer or not answer.strip():
        raise AdapterError("sync_probe_output_invalid")

    def readiness_marker() -> str:
        """Return the local readiness marker. Call this once before replying."""
        return "READY"

    agent = Agent(create_fallback_models(config), tools=[readiness_marker])
    result = await agent.run(
        "Call readiness_marker exactly once, then reply with exactly its returned text."
    )
    if not result.output.strip():
        raise AdapterError("agent_probe_output_invalid")
    from pydantic_ai.messages import ToolReturnPart
    if not any(isinstance(part, ToolReturnPart) and part.tool_name == "readiness_marker"
               for message in result.all_messages() for part in message.parts):
        raise AdapterError("agent_probe_tool_not_called")


def mock_response(request: httpx.Request) -> httpx.Response:
    """Exercise the same sync and tool round-trip paths without network access."""
    payload = json.loads(request.content)
    thinking = (payload.get("enable_thinking") is True
                or payload.get("thinking", {}).get("type") == "enabled")
    message = {"role": "assistant", "content": "READY"}
    if thinking:
        message["reasoning_content"] = "Mock reasoning."
    reason = "stop"
    if payload.get("tools") and not any(m["role"] == "tool" for m in payload["messages"]):
        message.update(content=None, tool_calls=[{
            "id": "call_mock_marker", "type": "function",
            "function": {"name": "readiness_marker", "arguments": "{}"},
        }])
        reason = "tool_calls"
    return httpx.Response(200, json={
        "id": "mock-completion", "created": 0, "object": "chat.completion",
        "model": payload["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20,
                  "completion_tokens_details": {"reasoning_tokens": 2 if thinking else 0}},
    })


async def work(args, run_dir: Path, target: str) -> int:
    repository = repository_for(target)
    _, commit, _ = SNAPSHOTS[target]
    # Only the trusted host loads credentials. CodeWiki's implicit dotenv search
    # must not read target or ancestor .env files.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    # Installed mermaid-py renders through mermaid.ink, not a local Node parser.
    os.environ["MERMAID_VALIDATE"] = "0"
    logging.disable(logging.CRITICAL)
    profile = load_profile_artifact(run_dir / "execution-profile.json")
    if profile.profile_digest != args.expected_profile_digest:
        raise AdapterError("worker_profile_digest_mismatch")
    key = os.environ.pop(profile.provider.api_key_env, "")
    if not os.statvfs(repository).f_flag & os.ST_RDONLY:
        raise AdapterError("target_not_readonly_mounted")
    limits = (Limits(max_requests=6, max_output_tokens=2048,
                     max_reserved_output_tokens=12288, timeout_seconds=600)
              if args.mode == "probe" else Limits())
    before = registered_snapshot(target)
    runtime = ProfileRuntime(
        profile, key, run_dir / "calls", limits,
        sync_transport=httpx.MockTransport(mock_response) if args.mock else None,
        async_transport=httpx.MockTransport(mock_response) if args.mock else None,
        execution_kind="offline_mock" if args.mock else "real_provider",
    )
    started = time.monotonic()
    receipt = {
        "schema_version": "codewiki-run-receipt-1.0", "mode": args.mode,
        "execution_kind": runtime.execution_kind,
        "target": target, "repository": str(repository), "snapshot": before,
        "profile": profile.model_dump(mode="json"), "profile_digest": profile.profile_digest,
        "limits": limits.model_dump(), "status": "running", "started_at": now(),
        "sandbox": "bwrap-root-readonly-run-and-tmp-writable",
        "tokenizer_asset_sha256": TOKENIZER_SHA256,
        "codewiki_upstream_commit": "f5dac8c42096af4242f2fac283b4bde1e36ba065",
        "parent_commit": git(ROOT, "rev-parse", "HEAD").strip(),
        "parent_dirty": bool(git(ROOT, "status", "--porcelain")),
        "adapter_sha256": {p.name: digest(p.read_bytes()) for p in
                           (Path(__file__), Path(__file__).with_name("runtime.py"),
                            Path(__file__).with_name("consumer.py"))},
        "codewiki_source_sha256": {
            str(p.relative_to(ROOT / "CodeWiki")): digest(p.read_bytes())
            for p in sorted((ROOT / "CodeWiki/codewiki").rglob("*.py"))
        },
        "coding_agent_wrapper_vcs_commit": json.loads(
            importlib.metadata.distribution("coding-agent-wrapper").read_text("direct_url.json") or "{}"
        ).get("vcs_info", {}).get("commit_id"),
        "environment": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                        if d.metadata.get("Name")},
        "python": sys.version, "node": subprocess.run(
            ["node", "--version"], capture_output=True, text=True, check=True,
        ).stdout.strip(),
    }
    write_json(run_dir / "receipt.json", receipt)
    try:
        config = backend_config(runtime, repository, run_dir / "docs")
        receipt["generation_settings"] = {
            name: getattr(config, name) for name in (
                "max_depth", "max_tokens", "max_token_per_module",
                "max_token_per_leaf_module", "min_modules_for_super_grouping",
                "max_leaf_nodes_per_cluster", "prompt_caching", "use_gitignore",
            )
        }
        receipt["generation_settings"].update(
            defaults_source="codewiki.cli.models.config.Configuration",
            cross_model_fallback=False, provider_retries="sdk_default",
            mermaid_validation="disabled_external_service_not_authorized",
            new_run_no_resume=True,
        )
        write_json(run_dir / "receipt.json", receipt)
        if args.mode == "probe":
            from codewiki.src.be.dependency_analyzer import DependencyGraphBuilder
            components, leaves = DependencyGraphBuilder(config).build_dependency_graph()
            if not components:
                raise AdapterError("static_analysis_empty")
            from codewiki.src.be.cluster_modules import get_clustering_input_token_count
            receipt["static_analysis"] = {
                "components": len(components), "leaf_nodes": len(leaves),
                "clustering_input_tokens": get_clustering_input_token_count(leaves, components),
            }
            print("Checking sync completion and agent tool round-trip...", flush=True)
            await probe(runtime, config)
        else:
            from codewiki.src.be.documentation_generator import DocumentationGenerator
            print(f"Generating CodeWiki documents for {target} at {commit}...", flush=True)
            await DocumentationGenerator(config, commit_id=commit).run()
            overview = run_dir / "docs/overview.md"
            if not overview.is_file() or not overview.read_text(encoding="utf-8").strip():
                raise AdapterError("overview_missing_or_empty")
        if runtime.failure:
            raise AdapterError(runtime.failure)
        if not runtime.calls:
            raise AdapterError("no_provider_calls")
        receipt["status"] = "complete"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = runtime.failure or (str(error) if isinstance(error, AdapterError)
                                               else type(error).__name__)
        # Frame locations only: exception text, source lines and locals may
        # contain provider bodies or credentials and must not be copied here.
        receipt["error_frames"] = [
            {"file": frame.filename, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(error.__traceback__)
        ]
    finally:
        try:
            await runtime.close()
        except Exception:
            receipt.update(status="failed", error="client_close_failed")
        try:
            receipt["snapshot_unchanged"] = registered_snapshot(target) == before
        except Exception:
            receipt["snapshot_unchanged"] = False
        if not receipt["snapshot_unchanged"]:
            receipt.update(status="failed", error="snapshot_changed")
        receipt.update(ended_at=now(), wall_seconds=time.monotonic() - started,
                       peak_rss_kib=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
                       rss_scope="max_of_self_and_reaped_child_high_water_marks_not_sum",
                       provider=runtime.summary())
        write_json(run_dir / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "error": receipt.get("error"),
                      "requests": len(runtime.calls), "run_directory": str(run_dir)}), flush=True)
    return 0 if receipt["status"] == "complete" else 1


def run_target(args, target: str) -> int:
    repository = repository_for(target)
    if args.clone_only:
        print(json.dumps({"status": "prepared", "target": target,
                          "snapshot": prepare_snapshot(target)}, indent=2))
        return 0
    if args.worker_run_dir:
        return asyncio.run(work(args, args.worker_run_dir, target))
    profile = load_execution_model_profile(args.model_profile)
    ProfileRuntime.validate_profile(profile)
    state = (registered_snapshot(target) if args.mode == "check"
             else prepare_snapshot(target))
    if not shutil.which("bwrap") or not shutil.which("node"):
        raise AdapterError("bwrap_or_node_unavailable")
    # Check the actual host's mount capability before any provider call.
    subprocess.run(["bwrap", "--ro-bind", "/", "/", "--", sys.executable, "-c",
                    "import os,sys; sys.exit(0 if os.statvfs(sys.argv[1]).f_flag & os.ST_RDONLY else 1)",
                    str(repository)], capture_output=True, check=True, timeout=30)
    key = ("offline-mock-key" if args.mock else os.environ.get(profile.provider.api_key_env)
           or dotenv_values(ROOT / ".env").get(profile.provider.api_key_env))
    if args.mode == "check":
        print(json.dumps({"status": "ready" if key else "credential_missing",
                          "target": target, "snapshot": state,
                          "profile_id": profile.profile_id, "profile_digest": profile.profile_digest,
                          "credential_present": bool(key), "readonly_mount_check": "passed",
                          "provider_connectivity": "not_checked"}, indent=2))
        return 0 if key else 1
    if not key:
        raise AdapterError("credential_unavailable")
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d-%H%M%S-%f")
    run_dir = RUNS_ROOT / target / profile.profile_id / f"{stamp}-{args.mode}{'-mock' if args.mock else ''}"
    run_dir.mkdir(parents=True, exist_ok=False)
    profile_path = run_dir / "execution-profile.json"
    profile_path.write_text(stable_profile_text(profile), encoding="utf-8")
    tokenizer_cache = stage_tokenizer_cache(run_dir)
    # Keep this run independent of all existing caches and outputs.
    env = dict(os.environ)
    env.update({profile.provider.api_key_env: key, "PYTHON_DOTENV_DISABLED": "1",
                "PYTHONPATH": str(ROOT / "CodeWiki"),
                "TIKTOKEN_CACHE_DIR": str(tokenizer_cache),
                "PYTHONDONTWRITEBYTECODE": "1", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_OPTIONAL_LOCKS": "0"})
    command = ["bwrap", "--die-with-parent", "--ro-bind", "/", "/",
               "--bind", str(run_dir), str(run_dir), "--tmpfs", "/tmp",
               "--ro-bind", str(profile_path), str(profile_path),
               "--ro-bind", str(tokenizer_cache), str(tokenizer_cache),
               "--dev", "/dev", "--proc", "/proc", "--chdir", str(run_dir),
               "--", sys.executable, str(Path(__file__).resolve()), "--mode", args.mode,
               "--target", target, "--worker-run-dir", str(run_dir),
               "--expected-profile-digest", profile.profile_digest]
    if args.mock:
        command.insert(1, "--unshare-net")
        command.append("--mock")
    print(f"Run directory: {run_dir}", flush=True)
    command_receipt = {"schema_version": "codewiki-command-receipt-1.0", "status": "running",
                       "started_at": now(), "mode": args.mode, "mock": args.mock,
                       "target": target, "snapshot": state,
                       "profile_id": profile.profile_id, "profile_digest": profile.profile_digest,
                       "wall_limit_seconds": 3600 if args.mode == "probe" else None}
    write_json(run_dir / "command.json", command_receipt)
    command_start = time.monotonic()
    try:
        with (run_dir / "worker.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=command_receipt["wall_limit_seconds"])
        command_receipt.update(status="complete" if result.returncode == 0 else "failed",
                               exit_code=result.returncode)
    except subprocess.TimeoutExpired:
        command_receipt.update(status="timeout", exit_code=None)
        raise AdapterError("worker_wall_limit_exhausted") from None
    finally:
        command_receipt.update(ended_at=now(), wall_seconds=time.monotonic() - command_start)
        write_json(run_dir / "command.json", command_receipt)
    print(f"Worker exit: {result.returncode}; receipt: {run_dir / 'receipt.json'}", flush=True)
    if result.returncode == 0 and args.mode == "generate":
        from integration.consumer import export_consumer
        print(f"Consumer knowledge: {export_consumer(run_dir, repository, expected_commit=SNAPSHOTS[target][1])}", flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--target", action="append", choices=tuple(SNAPSHOTS),
                           help="registered snapshot ID; repeat for a sequential batch (default: Tiktoken)")
    selection.add_argument("--all", action="store_true", help="select all 14 snapshots")
    parser.add_argument("--list", action="store_true", help="list registered snapshots without model calls")
    parser.add_argument("--clone-only", action="store_true",
                        help="prepare missing shared clones and verify snapshots, without credentials or model calls")
    parser.add_argument("--mode", choices=("check", "probe", "generate"), default="check",
                        help="check is offline; probe and generate make paid API calls")
    parser.add_argument("--mock", action="store_true",
                        help="offline probe through fake transport; only with --mode probe")
    parser.add_argument("--model-profile", metavar="ID_OR_PATH",
                        help="shared Qwen/DeepSeek execution profile (default: Qwen thinking-medium)")
    parser.add_argument("--export-consumer", type=Path, metavar="RUN_DIR",
                        help="export an existing successful run; infer its registered target from its receipt")
    parser.add_argument("--allow-partial", action="store_true",
                        help="with --export-consumer only: admit IncompleteDocumentationError with existing Markdown")
    parser.add_argument("--worker-run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-profile-digest", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.allow_partial and args.export_consumer is None:
            raise AdapterError("allow_partial_requires_export_consumer")
        if args.list:
            for target, (origin, commit, count) in SNAPSHOTS.items():
                print(f"{target}\t{commit}\t{count} tasks\t{origin}")
            return 0
        if args.export_consumer is not None:
            if (args.mode != "check" or args.mock or args.model_profile or args.worker_run_dir
                    or args.clone_only or args.all):
                raise AdapterError("export_consumer_requires_standalone_invocation")
            from integration.consumer import export_consumer
            receipt = json.loads((args.export_consumer / "receipt.json").read_text(encoding="utf-8"))
            target = receipt.get("target")
            repository = repository_for(target)
            if args.target is not None and args.target != [target]:
                raise AdapterError("export_consumer_target_mismatch")
            registered_snapshot(target)
            knowledge = export_consumer(args.export_consumer, repository,
                                        expected_commit=SNAPSHOTS[target][1], allow_partial=args.allow_partial)
            print(f"Consumer knowledge: {knowledge}")
            return 0
        if args.mock and args.mode != "probe":
            raise AdapterError("mock_requires_probe_mode")
        if args.clone_only and (args.mode != "check" or args.mock or args.worker_run_dir):
            raise AdapterError("clone_only_requires_check_mode")
        targets = list(SNAPSHOTS) if args.all else list(dict.fromkeys(args.target or [TARGET]))
        if args.worker_run_dir and len(targets) != 1:
            raise AdapterError("worker_requires_one_target")
        failed = False
        for target in targets:
            try:
                failed = bool(run_target(args, target)) or failed
            except Exception as error:
                reason = str(error) if isinstance(error, AdapterError) else type(error).__name__
                print(f"CodeWiki preparation failed for {target}: {reason}", file=sys.stderr)
                failed = True
        return int(failed)
    except Exception as error:
        print("CodeWiki preparation failed: " + (str(error) if isinstance(error, AdapterError)
                                                else type(error).__name__), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
