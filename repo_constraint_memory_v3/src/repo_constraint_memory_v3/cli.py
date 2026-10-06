"""Agent-assisted discovery, fixed-snapshot build, and offline validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError

from repo_constraint_memory_v3.builder import (
    SnapshotMismatchError,
    build_memory,
    load_build_config,
)
from repo_constraint_memory_v3.agent_runtime import (
    DeterministicFakeAgentRuntime,
    SubprocessAgentRuntime,
    fake_agent_profile,
    load_agent_profile,
    stable_model_digest,
)
from repo_constraint_memory_v3.discovery import (
    DiscoveryError,
    UnsupportedRepositoryError,
)
from repo_constraint_memory_v3.current_source_run import (
    build_current_source_only,
    validate_current_source_only_run,
)
from repo_constraint_memory_v3.llm import MockLLMClient, OpenAICompatibleClient
from repo_constraint_memory_v3.model_profiles import (
    OFFLINE_EXECUTION_MODEL_PROFILE_ID,
    ExecutionModelProfile,
    builtin_profile_ids,
    load_execution_model_profile,
    load_profile_artifact,
)
from repo_constraint_memory_v3.models import BuildConfig, SCHEMA_VERSION
from repo_constraint_memory_v3.source_acquisition import (
    SourceAcquisitionError,
    acquire_sources,
    load_source_acquisition,
)
from repo_constraint_memory_v3.source_plan import SourcePlanError
from repo_constraint_memory_v3.run_receipt import (
    CachedInputTelemetry,
    CommandRunRecorder,
    cached_input_from_agent_receipt,
    unavailable_cached_input,
)
from repo_constraint_memory_v3.unified.consumer import (
    export_consumer_view,
    validate_consumer_view,
)
from repo_constraint_memory_v3.unified.rebuild import (
    UnifiedRebuildError,
    rebuild_unified,
)
from repo_constraint_memory_v3.unified.validation import (
    INCOMPLETE_PACKAGE_ERROR,
    validate_unified_directory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repo-memory-v3",
        description=(
            "Build and validate evidence-grounded Python repository norm "
            "knowledge with independent explicit/current-source discovery, "
            "candidate-conditioned Git evidence, unified policy, and offline replay."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover",
        help=(
            "Create a deterministic manifest and run the configured external "
            "agent to supplement explicit sources."
        ),
    )
    discover.add_argument("repository", type=Path)
    discover.add_argument("--config", required=True, type=Path)
    discover.add_argument("--agent-profile", required=True, type=Path)
    discover.add_argument(
        "--model-profile",
        help=(
            "Built-in execution-model profile ID or bounded YAML path. "
            "Omit for the frozen default; --mock-agent instead selects the "
            "offline test profile."
        ),
    )
    discover.add_argument(
        "--run-receipt",
        type=Path,
        help=(
            "Write an independent command timing/RSS/provider-cache receipt; "
            "the path must be outside the repository and output tree."
        ),
    )
    discover.add_argument("--out", required=True, type=Path)
    discover.add_argument(
        "--mock-agent",
        action="store_true",
        help=(
            "Exercise the complete acquisition protocol with a deterministic "
            "empty-proposal fake runner; for offline tests only."
        ),
    )

    build = subparsers.add_parser(
        "build",
        help=(
            "Run explicit extraction/verification plus configured current-source "
            "acquisition, then build the canonical unified package."
        ),
    )
    build.add_argument("repository", type=Path)
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--agent-profile", required=True, type=Path)
    build.add_argument("--acquisition", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    build.add_argument(
        "--observation-cache",
        type=Path,
        help=(
            "Optional external content-addressed Python parse cache; it must be "
            "outside the target repository and is never trusted by validate."
        ),
    )
    build.add_argument(
        "--run-receipt",
        type=Path,
        help=(
            "Write an independent command timing/RSS/provider-cache receipt; "
            "the path must be outside the repository and output tree."
        ),
    )
    build.add_argument(
        "--inference-cache",
        type=Path,
        help=(
            "Optional external success-only structured-inference cache. Entries "
            "are immutable, revalidated, and must be outside the target repository."
        ),
    )
    build.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Reuse strict terminal-success inference records from a prior run "
            "with the same snapshot and protocol; failures are always retried."
        ),
    )
    build.add_argument(
        "--max-concurrency",
        type=int,
        choices=range(1, 33),
        metavar="1..32",
        help=(
            "Override simultaneous explicit extraction/verifier and independent "
            "current-source LLM batch calls."
        ),
    )
    build.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic offline mock; do not read .env or use network.",
    )
    build.add_argument(
        "--emit-consumer-profile",
        choices=("accepted_plus_review",),
        help=(
            "Also emit the experimental accepted_plus_review static view; "
            "accepted_only is always emitted."
        ),
    )

    build_current_source = subparsers.add_parser(
        "build-current-source",
        help=(
            "Run only observation, semantic-unit, and current-source mining; "
            "write a non-delivery channel artifact with no explicit stages."
        ),
    )
    build_current_source.add_argument("repository", type=Path)
    build_current_source.add_argument("--config", required=True, type=Path)
    build_current_source.add_argument(
        "--agent-profile",
        required=True,
        type=Path,
    )
    build_current_source.add_argument(
        "--run-receipt",
        type=Path,
        help=(
            "Write an independent command timing/RSS/provider-cache receipt; "
            "the path must be outside the repository and output tree."
        ),
    )
    build_current_source.add_argument(
        "--acquisition",
        required=True,
        type=Path,
    )
    build_current_source.add_argument("--out", required=True, type=Path)
    build_current_source.add_argument(
        "--observation-cache",
        type=Path,
        help=(
            "Optional external content-addressed Python parse cache; it must be "
            "outside the target repository and is never trusted by validation."
        ),
    )
    build_current_source.add_argument(
        "--inference-cache",
        type=Path,
        help=(
            "Optional external success-only current-source inference cache. "
            "Entries are immutable and revalidated."
        ),
    )
    build_current_source.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Reuse strict terminal-success current-source inference records "
            "from a compatible prior run."
        ),
    )
    build_current_source.add_argument(
        "--max-concurrency",
        type=int,
        choices=range(1, 33),
        metavar="1..32",
        help="Override simultaneous current-source LLM batch calls.",
    )
    build_current_source.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic offline mock; do not read .env or use network.",
    )

    build_unified = subparsers.add_parser(
        "build-unified",
        help=(
            "Load a bound schema-4 channel run and rebuild the canonical unified "
            "package from its immutable upstream input envelope without offline replay."
        ),
    )
    build_unified.add_argument("repository", type=Path)
    build_unified.add_argument("--config", required=True, type=Path)
    build_unified.add_argument("--agent-profile", required=True, type=Path)
    build_unified.add_argument("--upstream-run", required=True, type=Path)
    build_unified.add_argument("--out", required=True, type=Path)
    build_unified.add_argument(
        "--inference-cache",
        type=Path,
        help="Optional external success-only cache for unified semantic stages.",
    )
    build_unified.add_argument(
        "--run-receipt",
        type=Path,
        help=(
            "Write an independent command timing/RSS/provider-cache receipt; "
            "the path must be outside the repository and output tree."
        ),
    )
    build_unified.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Reuse strict terminal-success unified-stage records from a "
            "compatible prior full or isolated run."
        ),
    )
    build_unified.add_argument(
        "--max-concurrency",
        type=int,
        choices=range(1, 33),
        metavar="1..32",
        help="Override concurrency; it must still match the upstream config.",
    )
    build_unified.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic offline mock; do not read .env or use network.",
    )
    build_unified.add_argument(
        "--emit-consumer-profile",
        choices=("accepted_plus_review",),
        help=(
            "Also emit the experimental accepted_plus_review static view; "
            "accepted_only is always emitted."
        ),
    )

    export_consumer = subparsers.add_parser(
        "export-consumer",
        help=(
            "Atomically export one checked static view from an immutable run "
            "without invoking full offline replay."
        ),
    )
    export_consumer.add_argument("source_run", type=Path)
    export_consumer.add_argument("--repository", required=True, type=Path)
    export_consumer.add_argument(
        "--profile",
        required=True,
        choices=("accepted_only", "accepted_plus_review"),
    )
    export_consumer.add_argument("--out", required=True, type=Path)

    validate_consumer = subparsers.add_parser(
        "validate-consumer",
        help="Offline-replay one manifest-selected static consumer view.",
    )
    validate_consumer.add_argument("view_directory", type=Path)
    validate_consumer.add_argument("--source-run", required=True, type=Path)
    validate_consumer.add_argument("--repository", required=True, type=Path)
    validate_consumer.add_argument(
        "--require-complete",
        action="store_true",
        help="Return 1 when the valid source package is truthfully partial.",
    )

    review = subparsers.add_parser(
        "review",
        help="Start an optional local developer review; no model is called.",
    )
    review_source = review.add_mutually_exclusive_group(required=True)
    review_source.add_argument("--run", type=Path, help="Immutable source extraction run.")
    review_source.add_argument("--session", type=Path, help="Continue this review session.")
    review.add_argument("--repository", required=True, type=Path)
    review.add_argument("--out", type=Path, help="Fresh session directory; required with --run.")
    review.add_argument("--port", type=int, default=0, help="Local port; 0 selects a free port.")
    review.add_argument("--reviewer", default="local-developer")
    review.add_argument("--no-open", action="store_true", help="Print the URL without opening a browser.")
    review.add_argument("--no-serve", action="store_true", help="Only create/check the session, without a web server.")

    review_publish = subparsers.add_parser(
        "review-publish",
        help="Publish original accepted plus this session's accepted reviews.",
    )
    review_publish.add_argument("session", type=Path)
    review_publish.add_argument("--repository", required=True, type=Path)
    review_publish.add_argument("--out", type=Path, help="Fresh publication directory; default is a numbered session publication.")

    review_auto = subparsers.add_parser(
        "review-auto",
        help="Review conventions in a persistent, isolated Codex session with native subagents.",
    )
    review_auto.add_argument("session", type=Path)
    review_auto.add_argument("--repository", required=True, type=Path)
    review_auto.add_argument("--profile", type=Path, help="Independent review profile JSON; default is gpt-5.6-sol/high with four subagents and no time quota.")
    review_auto.add_argument("--mock", action="store_true", help="Use deterministic offline decisions; no login, model or network.")

    for name, help_text in (
        ("check-review-consumer", "Check reviewed consumer binding and bytes without full source replay."),
        ("validate-review-consumer", "Independently audit a reviewed consumer with one full offline source replay."),
    ):
        review_check = subparsers.add_parser(name, help=help_text)
        review_check.add_argument("view_directory", type=Path)
        review_check.add_argument("--review", required=True, type=Path, help="Source review session directory.")
        review_check.add_argument("--repository", required=True, type=Path)
        if name == "validate-review-consumer":
            review_check.add_argument("--require-complete", action="store_true")

    validate = subparsers.add_parser(
        "validate",
        help=(
            "Alias of validate-unified for offline replay of a schema-4 package."
        ),
    )
    validate.add_argument("run_directory", type=Path)
    validate.add_argument(
        "--repository",
        required=True,
        type=Path,
        help=(
            "Expected target repository for bounded deterministic snapshot "
            "replay; never trust a repository path from the run artifact."
        ),
    )
    validate.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Also fail when any configured required capability has partial "
            "coverage. Without this flag, validation checks integrity, replay, "
            "and snapshot safety while allowing independently accepted claims from a "
            "truthfully incomplete run."
        ),
    )

    validate_current_source = subparsers.add_parser(
        "validate-current-source",
        help=(
            "Offline-replay a current-source-only channel artifact without "
            "provider calls."
        ),
    )
    validate_current_source.add_argument("run_directory", type=Path)
    validate_current_source.add_argument(
        "--repository",
        required=True,
        type=Path,
        help="Expected repository for deterministic fixed-snapshot replay.",
    )
    validate_current_source.add_argument(
        "--require-complete",
        action="store_true",
        help="Also fail when the current-source channel has partial coverage.",
    )

    validate_unified = subparsers.add_parser(
        "validate-unified",
        help="Offline-replay a schema-4 unified package without provider calls.",
    )
    validate_unified.add_argument("run_directory", type=Path)
    validate_unified.add_argument(
        "--repository",
        required=True,
        type=Path,
        help="Caller-authorized repository for exact snapshot and HEAD replay.",
    )
    validate_unified.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Return 1 for a valid partial package instead of allowing its "
            "independently accepted claims to be consumed."
        ),
    )

    model_profiles = subparsers.add_parser(
        "model-profiles",
        help="List the built-in execution-model profiles without calling a provider.",
    )
    model_profiles.add_argument(
        "--include-offline",
        action="store_true",
        help="Also show the deterministic test-only profile.",
    )
    model_profiles.add_argument(
        "--json",
        action="store_true",
        help="Emit stable JSON instead of a compact text table.",
    )
    return parser


def _print_progress(message: str) -> None:
    print(f"[repo-memory-v3] {message}", file=sys.stderr, flush=True)


def _remember_model_profile(
    args: argparse.Namespace,
    model_profile: ExecutionModelProfile,
) -> None:
    args._execution_model_profile = model_profile


def _remember_client_telemetry(
    args: argparse.Namespace,
    client: object | None,
) -> None:
    if getattr(args, "run_receipt", None) is None:
        return
    getter = getattr(client, "cached_input_telemetry", None)
    if getter is None:
        return
    args._provider_cached_input = CachedInputTelemetry.model_validate(getter())


def _consumer_profiles(extra_profile: str | None) -> tuple[str, ...]:
    profiles = ["accepted_only"]
    if extra_profile is not None:
        profiles.append(extra_profile)
    return tuple(profiles)


def _require_outside_repository(
    repository: Path,
    *,
    label: str,
    path: Path | None,
) -> None:
    if path is None:
        return
    resolved_repository = repository.resolve()
    resolved_path = path.resolve()
    if (
        resolved_path == resolved_repository
        or resolved_path in resolved_repository.parents
        or resolved_repository in resolved_path.parents
    ):
        raise ValueError(f"{label} must be outside the target repository")


def _load_runtime_environment(repository: Path) -> None:
    """Load operator credentials without overriding an exported environment."""

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        _require_outside_repository(
            repository,
            label="dotenv file",
            path=Path(dotenv_path),
        )
        load_dotenv(dotenv_path, override=False)


def _profile_selector_path(selector: str | None) -> Path | None:
    """Return a caller-supplied profile path, but never a built-in ID."""

    if selector is None or selector in builtin_profile_ids(include_offline=True):
        return None
    return Path(selector)


def _load_discovery_model_profile(args: argparse.Namespace) -> ExecutionModelProfile:
    selector = args.model_profile
    if args.mock_agent and selector is None:
        selector = OFFLINE_EXECUTION_MODEL_PROFILE_ID
    selector_path = _profile_selector_path(selector)
    _require_outside_repository(
        args.repository,
        label="execution-model profile",
        path=selector_path,
    )
    profile = load_execution_model_profile(
        selector,
        allow_offline=args.mock_agent,
    )
    if args.mock_agent and profile.profile_kind != "offline_test":
        raise ValueError("--mock-agent requires the offline execution-model profile")
    return profile


def _load_inherited_model_profile(run_directory: Path) -> ExecutionModelProfile:
    """Load the immutable profile selected at acquisition, without a fallback."""

    return load_profile_artifact(run_directory / "model" / "profile.json")


def _require_client_mode(
    model_profile: ExecutionModelProfile,
    *,
    mock: bool,
    command: str,
) -> None:
    if mock != (model_profile.profile_kind == "offline_test"):
        required = "--mock" if model_profile.profile_kind == "offline_test" else "real"
        raise ValueError(
            f"{command} must use {required} execution for inherited profile "
            f"{model_profile.profile_id}"
        )


def _client_for_profile(
    repository: Path,
    config: BuildConfig,
    model_profile: ExecutionModelProfile,
    *,
    mock: bool,
):
    if mock:
        return MockLLMClient()
    _load_runtime_environment(repository)
    llm = config.llm
    api_key = os.environ.get(llm.api_key_env)
    if not api_key:
        raise ValueError(
            "required API key environment variable is not set: "
            f"{llm.api_key_env}"
        )
    return OpenAICompatibleClient(
        llm,
        api_key,
        model_profile=model_profile,
    )


def _model_profiles(args: argparse.Namespace) -> int:
    rows: list[dict[str, object]] = []
    for profile_id in builtin_profile_ids(include_offline=args.include_offline):
        profile = load_execution_model_profile(
            profile_id,
            allow_offline=args.include_offline,
        )
        rows.append(
            {
                "profile_id": profile.profile_id,
                "default": profile.profile_id
                == load_execution_model_profile().profile_id,
                "kind": profile.profile_kind,
                "provider": profile.provider.provider_id,
                "model": profile.canonical_model_id,
                "thinking_mode": profile.reasoning.mode,
                "reasoning_effort": profile.reasoning.effort,
                "reasoning_max_tokens": profile.reasoning.max_tokens,
                "profile_digest": profile.profile_digest,
            }
        )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        for row in rows:
            marker = " (default)" if row["default"] else ""
            reasoning = str(row["thinking_mode"])
            if row["reasoning_effort"] is not None:
                reasoning += f"/{row['reasoning_effort']}"
            elif row["reasoning_max_tokens"] is not None:
                reasoning += f"/max_tokens={row['reasoning_max_tokens']}"
            print(
                f"{row['profile_id']}{marker}: model={row['model']} "
                f"provider={row['provider']} reasoning={reasoning}"
            )
    return 0


def _discover(args: argparse.Namespace) -> int:
    try:
        for label, path in (
            ("configuration", args.config),
            ("agent profile", args.agent_profile),
            ("source acquisition output", args.out),
        ):
            _require_outside_repository(
                args.repository,
                label=label,
                path=path,
            )
        model_profile = _load_discovery_model_profile(args)
        _remember_model_profile(args, model_profile)
        config = load_build_config(args.config, model_profile=model_profile)
        profile = load_agent_profile(
            args.agent_profile,
            model_profile=model_profile,
        )
        if args.mock_agent:
            if profile != fake_agent_profile():
                raise ValueError(
                    "--mock-agent requires the exact Norm4Code offline fake profile"
                )
            runtime = DeterministicFakeAgentRuntime(profile)
        else:
            _load_runtime_environment(args.repository)
            runtime = SubprocessAgentRuntime(profile)
        bundle = acquire_sources(
            args.repository,
            config,
            profile,
            model_profile,
            runtime,
            args.out,
        )
        if args.run_receipt is not None:
            args._provider_cached_input = cached_input_from_agent_receipt(
                bundle.model_execution
            )
    except UnsupportedRepositoryError as error:
        print(f"discover failed [unsupported_repository]: {error}", file=sys.stderr)
        return 2
    except SourceAcquisitionError as error:
        for detail in error.errors:
            print(f"discover failed [agent_acquisition]: {detail}", file=sys.stderr)
        return 1
    except (OSError, ValueError, ValidationError, yaml.YAMLError, DiscoveryError) as error:
        print(f"discover failed [input_or_snapshot]: {error}", file=sys.stderr)
        return 2
    print(
        "Acquisition complete=true "
        f"manifest_files={len(bundle.discovery.manifest.files)} "
        f"baseline_sources={len(bundle.discovery.baseline_sources)} "
        f"agent_proposals={len(bundle.supplement.proposals)} "
        f"agent={bundle.profile.framework_id} "
        f"model_profile={bundle.model_profile.profile_id}"
    )
    print(f"Snapshot: {bundle.discovery.repository.snapshot_digest}")
    print(f"Output: {args.out}")
    return 0


def _build(args: argparse.Namespace) -> int:
    client = None
    try:
        for label, path in (
            ("configuration", args.config),
            ("agent profile", args.agent_profile),
            ("source acquisition", args.acquisition),
            ("run output", args.out),
            ("observation cache", args.observation_cache),
            ("inference cache", args.inference_cache),
            ("resume run", args.resume_from),
        ):
            _require_outside_repository(
                args.repository,
                label=label,
                path=path,
            )
        model_profile = _load_inherited_model_profile(args.acquisition)
        _remember_model_profile(args, model_profile)
        config = load_build_config(args.config, model_profile=model_profile)
        if args.max_concurrency is not None:
            config = config.model_copy(
                update={
                    "llm": config.llm.model_copy(
                        update={"max_concurrency": args.max_concurrency}
                    )
                }
            )
        profile = load_agent_profile(
            args.agent_profile,
            model_profile=model_profile,
        )
        acquisition = load_source_acquisition(
            args.acquisition,
            expected_stage_settings=config.agent.stages["source_discovery"],
        )
        if stable_model_digest(profile) != acquisition.acquisition.agent_profile_sha256:
            raise ValueError(
                "build agent profile does not match the acquisition profile"
            )
        _require_client_mode(model_profile, mock=args.mock, command="build")
        client = _client_for_profile(
            args.repository,
            config,
            model_profile,
            mock=args.mock,
        )
        result = build_memory(
            args.repository,
            config,
            acquisition,
            client,
            args.out,
            observation_cache=args.observation_cache,
            inference_cache=args.inference_cache,
            resume_from=args.resume_from,
            progress_callback=_print_progress,
            consumer_profiles=_consumer_profiles(args.emit_consumer_profile),
        )
    except SourcePlanError as error:
        for detail in error.errors:
            print(f"build failed [source_plan]: {detail}", file=sys.stderr)
        return 2
    except SourceAcquisitionError as error:
        for detail in error.errors:
            print(f"build failed [source_acquisition]: {detail}", file=sys.stderr)
        return 2
    except SnapshotMismatchError as error:
        print(f"build failed [snapshot]: {error}", file=sys.stderr)
        return 2
    except (
        OSError,
        ValueError,
        ValidationError,
        yaml.YAMLError,
        DiscoveryError,
    ) as error:
        print(f"build failed [input_or_processing]: {error}", file=sys.stderr)
        return 2
    finally:
        _remember_client_telemetry(args, client)
        close = getattr(client, "close", None)
        if close is not None:
            close()
    report = result.build_report
    outcomes = report["outcomes"]
    candidate_roots = outcomes["candidate_roots"]
    propositions = outcomes["merged_propositions"]
    processing = report["processing"]
    llm = report["llm"]
    print(
        f"Build complete={str(report['complete']).lower()} "
        f"candidate_roots={candidate_roots['total']} "
        f"accepted_roots={candidate_roots['accepted']} "
        f"review_roots={candidate_roots['review']} "
        f"rejected_roots={candidate_roots['rejected']} "
        f"accepted_propositions={propositions['accepted']} "
        f"delivered_propositions={propositions['delivered']}"
    )
    print(
        f"Sources={report['discovery']['final_source_count']} "
        f"units={processing['unit_count']} batches={processing['batch_count']} "
        f"observed_concurrency={llm['observed_max_concurrency']}"
    )
    print(f"Outputs: {args.out}")
    return 0 if report["complete"] else 1


def _build_current_source(args: argparse.Namespace) -> int:
    client = None
    try:
        for label, path in (
            ("configuration", args.config),
            ("agent profile", args.agent_profile),
            ("source acquisition", args.acquisition),
            ("run output", args.out),
            ("observation cache", args.observation_cache),
            ("inference cache", args.inference_cache),
            ("resume run", args.resume_from),
        ):
            _require_outside_repository(
                args.repository,
                label=label,
                path=path,
            )
        model_profile = _load_inherited_model_profile(args.acquisition)
        _remember_model_profile(args, model_profile)
        config = load_build_config(args.config, model_profile=model_profile)
        if args.max_concurrency is not None:
            config = config.model_copy(
                update={
                    "llm": config.llm.model_copy(
                        update={"max_concurrency": args.max_concurrency}
                    )
                }
            )
        profile = load_agent_profile(
            args.agent_profile,
            model_profile=model_profile,
        )
        acquisition = load_source_acquisition(
            args.acquisition,
            expected_stage_settings=config.agent.stages["source_discovery"],
        )
        if stable_model_digest(profile) != acquisition.acquisition.agent_profile_sha256:
            raise ValueError(
                "build-current-source agent profile does not match the "
                "acquisition profile"
            )
        _require_client_mode(
            model_profile,
            mock=args.mock,
            command="build-current-source",
        )
        client = _client_for_profile(
            args.repository,
            config,
            model_profile,
            mock=args.mock,
        )
        result = build_current_source_only(
            args.repository,
            config,
            acquisition,
            client,
            args.out,
            observation_cache=args.observation_cache,
            inference_cache=args.inference_cache,
            resume_from=args.resume_from,
            progress_callback=_print_progress,
        )
    except SourcePlanError as error:
        for detail in error.errors:
            print(
                f"build-current-source failed [source_plan]: {detail}",
                file=sys.stderr,
            )
        return 2
    except SourceAcquisitionError as error:
        for detail in error.errors:
            print(
                f"build-current-source failed [source_acquisition]: {detail}",
                file=sys.stderr,
            )
        return 2
    except SnapshotMismatchError as error:
        print(
            f"build-current-source failed [snapshot]: {error}",
            file=sys.stderr,
        )
        return 2
    except (
        OSError,
        ValueError,
        ValidationError,
        yaml.YAMLError,
        DiscoveryError,
    ) as error:
        print(
            f"build-current-source failed [input_or_processing]: {error}",
            file=sys.stderr,
        )
        return 2
    finally:
        _remember_client_telemetry(args, client)
        close = getattr(client, "close", None)
        if close is not None:
            close()
    index = result.current_source.index
    print(
        f"Current-source build complete={str(result.run_manifest.complete).lower()} "
        f"candidate_roots={index.unique_candidate_roots} "
        f"hypothesis_roots={index.root_hypothesis} "
        f"review_roots={index.root_review} "
        f"rejected_roots={index.root_rejected} "
        f"experimental_accept_candidates="
        f"{index.experimental_accept_candidates} "
        f"propositions={index.propositions}"
    )
    print(
        f"Semantic_observations={index.semantic_observations} "
        f"classified_members={index.classified_members} "
        f"inference_calls={len(result.inference_records)}"
    )
    print(f"Outputs: {args.out}")
    return 0 if result.run_manifest.complete else 1


def _build_unified(args: argparse.Namespace) -> int:
    client = None
    try:
        for label, path in (
            ("configuration", args.config),
            ("agent profile", args.agent_profile),
            ("upstream run", args.upstream_run),
            ("run output", args.out),
            ("inference cache", args.inference_cache),
            ("resume run", args.resume_from),
        ):
            _require_outside_repository(
                args.repository,
                label=label,
                path=path,
            )
        model_profile = _load_inherited_model_profile(args.upstream_run)
        _remember_model_profile(args, model_profile)
        config = load_build_config(args.config, model_profile=model_profile)
        if args.max_concurrency is not None:
            config = config.model_copy(
                update={
                    "llm": config.llm.model_copy(
                        update={"max_concurrency": args.max_concurrency}
                    )
                }
            )
        profile = load_agent_profile(
            args.agent_profile,
            model_profile=model_profile,
        )
        _require_client_mode(
            model_profile,
            mock=args.mock,
            command="build-unified",
        )
        client = _client_for_profile(
            args.repository,
            config,
            model_profile,
            mock=args.mock,
        )
        result = rebuild_unified(
            args.repository,
            config,
            profile,
            args.upstream_run,
            client,
            args.out,
            inference_cache=args.inference_cache,
            resume_from=args.resume_from,
            progress_callback=_print_progress,
            consumer_profiles=_consumer_profiles(args.emit_consumer_profile),
        )
    except UnifiedRebuildError as error:
        for detail in error.errors:
            print(
                f"build-unified failed [upstream_or_binding]: {detail}",
                file=sys.stderr,
            )
        return 2
    except SnapshotMismatchError as error:
        print(f"build-unified failed [snapshot]: {error}", file=sys.stderr)
        return 2
    except (
        OSError,
        ValueError,
        ValidationError,
        yaml.YAMLError,
        DiscoveryError,
    ) as error:
        print(
            f"build-unified failed [input_or_processing]: {error}",
            file=sys.stderr,
        )
        return 2
    finally:
        _remember_client_telemetry(args, client)
        close = getattr(client, "close", None)
        if close is not None:
            close()
    bundle = result.bundle
    print(
        f"Unified build complete={str(result.complete).lower()} "
        f"cases={len(bundle.cases)} "
        f"accepted={bundle.index.accepted_count} "
        f"review={bundle.index.review_count} "
        f"rejected={bundle.index.rejected_count} "
        f"inference_records={len(result.inference_records)}"
    )
    print(f"Outputs: {args.out}")
    return 0 if result.complete else 1


def _validate_unified(args: argparse.Namespace) -> int:
    try:
        _require_outside_repository(
            args.repository,
            label="run directory",
            path=args.run_directory,
        )
        errors = validate_unified_directory(
            args.run_directory,
            args.repository,
            require_complete=True,
        )
    except (OSError, ValueError, ValidationError) as error:
        print(f"unified validation failed [artifact]: {error}", file=sys.stderr)
        return 2
    artifact_errors = [
        error for error in errors if error != INCOMPLETE_PACKAGE_ERROR
    ]
    if artifact_errors:
        for error in artifact_errors:
            print(f"unified validation failed: {error}", file=sys.stderr)
        return 2
    coverage = "partial" if INCOMPLETE_PACKAGE_ERROR in errors else "complete"
    suffix = (
        ""
        if coverage == "complete"
        else "; accepted claims are usable, but knowledge may be missing"
    )
    print(
        f"Valid unified {SCHEMA_VERSION} run: {args.run_directory} "
        f"(coverage={coverage}{suffix})"
    )
    if args.require_complete and coverage != "complete":
        print(
            "unified validation incomplete: complete package required",
            file=sys.stderr,
        )
        return 1
    return 0


def _validate(args: argparse.Namespace) -> int:
    """Compatibility alias for the only schema-4 unified validator."""

    return _validate_unified(args)


def _export_consumer(args: argparse.Namespace) -> int:
    try:
        for label, path in (
            ("source run", args.source_run),
            ("consumer output", args.out),
        ):
            _require_outside_repository(args.repository, label=label, path=path)
        plan = export_consumer_view(
            args.source_run,
            repository_root=args.repository,
            profile=args.profile,
            output_directory=args.out,
        )
    except (OSError, ValueError, ValidationError) as error:
        print(f"consumer export failed [artifact_or_binding]: {error}", file=sys.stderr)
        return 2
    print(
        f"Exported consumer profile={args.profile} files={len(plan.files)}: "
        f"{args.out}"
    )
    return 0


def _validate_consumer(args: argparse.Namespace) -> int:
    try:
        for label, path in (
            ("consumer view", args.view_directory),
            ("source run", args.source_run),
        ):
            _require_outside_repository(args.repository, label=label, path=path)
        errors = validate_consumer_view(
            args.view_directory,
            source_run=args.source_run,
            repository_root=args.repository,
            require_complete=True,
        )
    except (OSError, ValueError, ValidationError) as error:
        print(f"consumer validation failed [artifact]: {error}", file=sys.stderr)
        return 2
    artifact_errors = [
        error for error in errors if error != INCOMPLETE_PACKAGE_ERROR
    ]
    if artifact_errors:
        for error in artifact_errors:
            print(f"consumer validation failed: {error}", file=sys.stderr)
        return 2
    coverage = "partial" if INCOMPLETE_PACKAGE_ERROR in errors else "complete"
    print(
        f"Valid consumer view: {args.view_directory} "
        f"(coverage={coverage})"
    )
    if args.require_complete and coverage != "complete":
        print("consumer validation incomplete: complete package required", file=sys.stderr)
        return 1
    return 0


def _validate_current_source(args: argparse.Namespace) -> int:
    try:
        _require_outside_repository(
            args.repository,
            label="run directory",
            path=args.run_directory,
        )
        errors = validate_current_source_only_run(
            args.run_directory,
            repository_root=args.repository,
            require_complete=args.require_complete,
        )
    except (OSError, ValueError, ValidationError) as error:
        print(
            f"current-source validation failed [artifact]: {error}",
            file=sys.stderr,
        )
        return 1
    if errors:
        for error in errors:
            print(
                f"current-source validation failed: {error}",
                file=sys.stderr,
            )
        return 1
    manifest = json.loads(
        (args.run_directory / "channel_run.json").read_text(encoding="utf-8")
    )
    coverage = "complete" if manifest["complete"] else "partial"
    print(
        f"Valid {manifest['run_kind']} {SCHEMA_VERSION} run: "
        f"{args.run_directory} (coverage={coverage})"
    )
    return 0


def _review(args: argparse.Namespace) -> int:
    from repo_constraint_memory_v3.review.service import create_review, load_review
    from repo_constraint_memory_v3.review.ui import serve_review

    try:
        if not 0 <= args.port <= 65535:
            raise ValueError("review port must be between 0 and 65535")
        if not args.reviewer.strip() or len(args.reviewer) > 1024:
            raise ValueError("reviewer must contain 1 to 1024 characters")
        if args.run is not None:
            if args.out is None:
                raise ValueError("review --run requires --out for a fresh session")
            session = create_review(
                args.run, repository_root=args.repository, output_directory=args.out,
            )
            directory = args.out
        else:
            if args.out is not None:
                raise ValueError("review --session continues its own directory; omit --out")
            session = load_review(args.session, repository_root=args.repository)
            directory = args.session
        print(f"Review session: {directory.resolve()} ({json.dumps(session.progress(), sort_keys=True)})")
        if not args.no_serve:
            serve_review(
                directory, repository_root=args.repository, port=args.port,
                reviewer_id=args.reviewer, open_browser=not args.no_open,
            )
    except (OSError, ValueError, ValidationError) as error:
        print(f"review failed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Review stopped; saved decisions remain available.")
    return 0


def _review_publish(args: argparse.Namespace) -> int:
    from repo_constraint_memory_v3.review.service import publish_review

    try:
        plan = publish_review(
            args.session, repository_root=args.repository, output_directory=args.out,
        )
    except (OSError, ValueError, ValidationError) as error:
        print(f"review publication failed: {error}", file=sys.stderr)
        return 2
    print(
        f"Published accepted_after_review: {plan.publication_directory / 'consumer'} "
        f"(accepted={plan.manifest.exposed_accepted_count}, "
        f"coverage={plan.manifest.package_status})"
    )
    return 0


def _review_auto(args: argparse.Namespace) -> int:
    from repo_constraint_memory_v3.integrations.developer_review import (
        ReviewExecutionProfile, run_codex_review,
    )

    try:
        profile = None
        if args.profile is not None:
            with args.profile.open("rb") as handle:
                data = handle.read(65_537)
            if len(data) > 65_536:
                raise ValueError("review profile exceeds 65536 bytes")
            profile = ReviewExecutionProfile.model_validate_json(data)
        result = run_codex_review(
            args.session, args.repository, profile=profile, mock=args.mock,
        )
    except (OSError, ValueError, ValidationError) as error:
        print(f"automatic review failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") == "complete" else 1


def _check_review_consumer(args: argparse.Namespace) -> int:
    from repo_constraint_memory_v3.review.service import preflight_review_consumer

    try:
        manifest = preflight_review_consumer(
            args.view_directory, review_directory=args.review,
            repository_root=args.repository,
        )
    except (OSError, ValueError, ValidationError) as error:
        print(f"review consumer check failed: {error}", file=sys.stderr)
        return 2
    print(
        f"Checked reviewed consumer: {args.view_directory} "
        f"(coverage={manifest.package_status}; no full source replay)"
    )
    return 0


def _validate_review_consumer(args: argparse.Namespace) -> int:
    from repo_constraint_memory_v3.review.service import validate_review_consumer

    errors = validate_review_consumer(
        args.view_directory, review_directory=args.review,
        repository_root=args.repository, require_complete=True,
    )
    artifact_errors = [error for error in errors if error != INCOMPLETE_PACKAGE_ERROR]
    if artifact_errors:
        for error in artifact_errors:
            print(f"review consumer validation failed: {error}", file=sys.stderr)
        return 2
    coverage = "partial" if INCOMPLETE_PACKAGE_ERROR in errors else "complete"
    print(f"Valid reviewed consumer: {args.view_directory} (coverage={coverage})")
    return 1 if args.require_complete and coverage != "complete" else 0


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "discover":
        return _discover(args)
    if args.command == "build":
        return _build(args)
    if args.command == "build-current-source":
        return _build_current_source(args)
    if args.command == "build-unified":
        return _build_unified(args)
    if args.command == "export-consumer":
        return _export_consumer(args)
    if args.command == "validate-consumer":
        return _validate_consumer(args)
    if args.command == "review":
        return _review(args)
    if args.command == "review-publish":
        return _review_publish(args)
    if args.command == "review-auto":
        return _review_auto(args)
    if args.command == "check-review-consumer":
        return _check_review_consumer(args)
    if args.command == "validate-review-consumer":
        return _validate_review_consumer(args)
    if args.command == "validate":
        return _validate(args)
    if args.command == "validate-current-source":
        return _validate_current_source(args)
    if args.command == "validate-unified":
        return _validate_unified(args)
    if args.command == "model-profiles":
        return _model_profiles(args)
    raise AssertionError(f"unhandled command: {args.command}")


def _receipt_cached_input(args: argparse.Namespace) -> CachedInputTelemetry:
    observed = getattr(args, "_provider_cached_input", None)
    if isinstance(observed, CachedInputTelemetry):
        return observed
    if args.command == "discover":
        execution_receipt = args.out / "model" / "execution-receipt.json"
        if execution_receipt.is_file():
            try:
                from repo_constraint_memory_v3.agent_runtime import (
                    AgentModelExecutionReceipt,
                )

                receipt = AgentModelExecutionReceipt.model_validate_json(
                    execution_receipt.read_text(encoding="utf-8")
                )
                return cached_input_from_agent_receipt(receipt)
            except (OSError, ValidationError, ValueError):
                pass
        return unavailable_cached_input("source_agent_receipt")
    return unavailable_cached_input("core_openai_chat")


def _receipt_profile_identity(
    args: argparse.Namespace,
) -> tuple[str | None, str | None]:
    profile = getattr(args, "_execution_model_profile", None)
    if not isinstance(profile, ExecutionModelProfile):
        return None, None
    return profile.profile_id, profile.profile_digest


def _start_run_recorder(args: argparse.Namespace) -> CommandRunRecorder | None:
    receipt_path = getattr(args, "run_receipt", None)
    if receipt_path is None:
        return None
    _require_outside_repository(
        args.repository,
        label="run receipt",
        path=receipt_path,
    )
    resolved_receipt = receipt_path.resolve()
    resolved_output = args.out.resolve()
    if (
        resolved_receipt == resolved_output
        or resolved_output in resolved_receipt.parents
        or resolved_receipt in resolved_output.parents
    ):
        raise ValueError("run receipt must be outside the command output tree")
    args.run_receipt = resolved_receipt
    return CommandRunRecorder(
        resolved_receipt,
        command=args.command,
        repository=args.repository,
        output=args.out,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        recorder = _start_run_recorder(args)
    except (OSError, ValueError, ValidationError) as error:
        print(f"{args.command} failed [run_receipt]: {error}", file=sys.stderr)
        return 2
    if recorder is None:
        return _dispatch(args)

    try:
        exit_code = _dispatch(args)
    except BaseException:
        profile_id, profile_digest = _receipt_profile_identity(args)
        try:
            recorder.finish(
                exit_code=None,
                interrupted=True,
                execution_model_profile_id=profile_id,
                execution_model_profile_digest=profile_digest,
                provider_cached_input=_receipt_cached_input(args),
            )
        except (OSError, ValueError, ValidationError) as receipt_error:
            print(
                f"{args.command} failed [run_receipt_finalize]: {receipt_error}",
                file=sys.stderr,
            )
        raise

    profile_id, profile_digest = _receipt_profile_identity(args)
    try:
        recorder.finish(
            exit_code=exit_code,
            interrupted=False,
            execution_model_profile_id=profile_id,
            execution_model_profile_digest=profile_digest,
            provider_cached_input=_receipt_cached_input(args),
        )
    except (OSError, ValueError, ValidationError) as error:
        print(
            f"{args.command} failed [run_receipt_finalize]: {error}",
            file=sys.stderr,
        )
        return 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
