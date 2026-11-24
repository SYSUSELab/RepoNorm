from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from helpers import CONFIG_PATH, build_fixture, copied_fixture
from repo_constraint_memory_v3 import cli as cli_module
from repo_constraint_memory_v3.agent_runtime import (
    fake_agent_profile,
    stable_json_text,
)
from repo_constraint_memory_v3.cli import main
from repo_constraint_memory_v3.llm import MockLLMClient
from repo_constraint_memory_v3.models import ProviderResponse
from repo_constraint_memory_v3.unified import consumer as consumer_module
from repo_constraint_memory_v3.unified import validation as unified_validation
from repo_constraint_memory_v3.unified.consumer import (
    CanonicalClaimsFile,
    ConsumerManifest,
)
from repo_constraint_memory_v3.unified.ids import seal_payload


def _tree_fingerprint(
    root: Path,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    directories: list[str] = []
    files: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            files.append((relative, f"symlink:{os.readlink(path)}"))
        elif path.is_dir():
            directories.append(relative)
        elif path.is_file():
            files.append(
                (relative, hashlib.sha256(path.read_bytes()).hexdigest())
            )
    return tuple(directories), tuple(files)


def _export_args(
    run: Path,
    repository: Path,
    output: Path,
    *,
    profile: str = "accepted_plus_review",
) -> list[str]:
    return [
        "export-consumer",
        str(run),
        "--repository",
        str(repository),
        "--profile",
        profile,
        "--out",
        str(output),
    ]


def _validate_args(
    view: Path,
    run: Path,
    repository: Path,
    *,
    require_complete: bool = False,
) -> list[str]:
    result = [
        "validate-consumer",
        str(view),
        "--source-run",
        str(run),
        "--repository",
        str(repository),
    ]
    if require_complete:
        result.append("--require-complete")
    return result


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _mock_source_run(
    root: Path,
    *,
    incomplete: bool,
    emit_plus_review: bool = False,
) -> tuple[Path, Path, CanonicalClaimsFile]:
    root.mkdir(parents=True, exist_ok=True)
    repository = copied_fixture(root, incomplete=incomplete)
    if not incomplete:
        _git(repository, "init", "-q")
        _git(repository, "config", "user.name", "Norm4Code Test")
        _git(
            repository,
            "config",
            "user.email",
            "norm4code@example.invalid",
        )
        _git(repository, "add", ".")
        _git(repository, "commit", "-q", "-m", "consumer fixture")
    profile = root / "fake-agent-profile.json"
    runtime_profile = fake_agent_profile().model_dump(mode="json")
    for key in (
        "execution_model_profile_id",
        "execution_model_profile_digest",
        "model",
    ):
        runtime_profile.pop(key)
    profile.write_text(
        stable_json_text(runtime_profile),
        encoding="utf-8",
    )
    acquisition = root / "acquisition"
    run = root / "run"
    assert main(
        [
            "discover",
            str(repository),
            "--config",
            str(CONFIG_PATH),
            "--agent-profile",
            str(profile),
            "--out",
            str(acquisition),
            "--mock-agent",
        ]
    ) == 0
    build_args = [
        "build",
        str(repository),
        "--config",
        str(CONFIG_PATH),
        "--agent-profile",
        str(profile),
        "--acquisition",
        str(acquisition),
        "--out",
        str(run),
        "--observation-cache",
        str(root / "observation-cache"),
        "--mock",
    ]
    if emit_plus_review:
        build_args.extend(
            ["--emit-consumer-profile", "accepted_plus_review"]
        )
    status = main(build_args)
    assert status == (1 if incomplete else 0)
    assert main(["validate", str(run), "--repository", str(repository)]) == 0
    canonical = CanonicalClaimsFile.model_validate_json(
        (run / "canonical" / "claims.json").read_bytes()
    )
    return run, repository, canonical


@pytest.fixture(scope="module")
def complete_source(tmp_path_factory):
    return _mock_source_run(
        tmp_path_factory.mktemp("consumer-complete-source"),
        incomplete=False,
    )


@pytest.fixture(scope="module")
def partial_source(tmp_path_factory):
    return _mock_source_run(
        tmp_path_factory.mktemp("consumer-partial-source"),
        incomplete=True,
    )


def test_late_consumer_export_round_trip_isolated_manifest_selected_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_source,
) -> None:
    run, repository, canonical = complete_source
    assert not (run / "consumers" / "accepted_plus_review").exists()
    before = _tree_fingerprint(run)
    output = tmp_path / "late-plus-review"
    real_validate = unified_validation.validate_unified_directory
    source_replay_calls = 0

    def count_source_replays(*args, **kwargs):
        nonlocal source_replay_calls
        source_replay_calls += 1
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(
        unified_validation,
        "validate_unified_directory",
        count_source_replays,
    )

    assert main(_export_args(run, repository, output)) == 0
    assert source_replay_calls == 0
    assert _tree_fingerprint(run) == before
    assert (output / "START.md").is_file()
    assert (output / "catalog.md").is_file()
    assert not (output / "canonical").exists()
    assert not (output / "consumers").exists()
    assert not (output / "by-trigger").exists()

    manifest = ConsumerManifest.model_validate_json(
        (output / "consumer-manifest.json").read_bytes()
    )
    assert manifest.consumer_profile == "accepted_plus_review"
    assert manifest.requested_consumer_profiles == ["accepted_plus_review"]
    assert manifest.emitted_consumer_profiles == ["accepted_plus_review"]

    catalog = (output / "catalog.md").read_text(encoding="utf-8")
    exposed = {
        item.claim_id
        for item in canonical.claims
        if item.epistemic_status in {"accepted", "review"}
    }
    rejected = {
        item.claim_id
        for item in canonical.claims
        if item.epistemic_status == "rejected"
    }
    assert exposed
    assert all(claim_id in catalog for claim_id in exposed)
    assert all(claim_id not in catalog for claim_id in rejected)

    real_render = consumer_module._render_standalone_plan
    selected: list[tuple[str, tuple[str, ...]]] = []

    def record_selected_profile(inputs, *, profile, requested_profiles):
        selected.append((profile, tuple(requested_profiles)))
        return real_render(
            inputs,
            profile=profile,
            requested_profiles=requested_profiles,
        )

    monkeypatch.setattr(
        consumer_module,
        "_render_standalone_plan",
        record_selected_profile,
    )
    view_before_validation = _tree_fingerprint(output)
    assert main(_validate_args(output, run, repository)) == 0
    assert source_replay_calls == 1
    assert selected == [
        ("accepted_plus_review", ("accepted_plus_review",))
    ]
    assert _tree_fingerprint(run) == before
    assert _tree_fingerprint(output) == view_before_validation


def test_build_emit_consumer_profile_writes_both_isolated_views(
    tmp_path: Path,
) -> None:
    run, repository, _ = _mock_source_run(
        tmp_path / "build-plus-review",
        incomplete=False,
        emit_plus_review=True,
    )

    report = json.loads(
        (run / "canonical" / "delivery_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["requested_consumer_profiles"] == [
        "accepted_only",
        "accepted_plus_review",
    ]
    assert report["emitted_consumer_profiles"] == [
        "accepted_only",
        "accepted_plus_review",
    ]
    for profile in report["emitted_consumer_profiles"]:
        manifest = ConsumerManifest.model_validate_json(
            (
                run
                / "consumers"
                / profile
                / "consumer-manifest.json"
            ).read_bytes()
        )
        assert manifest.consumer_profile == profile
        assert manifest.emitted_consumer_profiles == [
            "accepted_only",
            "accepted_plus_review",
        ]
    assert main(["validate", str(run), "--repository", str(repository)]) == 0


def test_export_accepts_empty_destination_but_rejects_nonempty_overlap_and_symlink(
    tmp_path: Path,
    complete_source,
) -> None:
    run, repository, _ = complete_source

    empty = tmp_path / "empty-destination"
    empty.mkdir()
    assert main(
        _export_args(run, repository, empty, profile="accepted_only")
    ) == 0
    assert (empty / "START.md").is_file()

    nonempty = tmp_path / "nonempty-destination"
    nonempty.mkdir()
    marker = nonempty / "keep.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    assert main(_export_args(run, repository, nonempty)) == 2
    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert sorted(path.name for path in nonempty.iterdir()) == ["keep.txt"]

    inside_source = run / "nested-consumer"
    assert main(_export_args(run, repository, inside_source)) == 2
    assert not inside_source.exists()

    inside_repository = repository / "consumer"
    assert main(_export_args(run, repository, inside_repository)) == 2
    assert not inside_repository.exists()

    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    symlink_output = tmp_path / "symlink-output"
    symlink_output.symlink_to(symlink_target, target_is_directory=True)
    assert main(_export_args(run, repository, symlink_output)) == 2
    assert symlink_output.is_symlink()
    assert list(symlink_target.iterdir()) == []


def test_export_rejects_repository_snapshot_mismatch_without_full_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_source,
) -> None:
    run, repository, _ = complete_source
    output = tmp_path / "snapshot-mismatch"

    def forbidden_replay(*_: object, **__: object) -> list[str]:
        raise AssertionError("consumer export must not invoke full replay")

    monkeypatch.setattr(
        unified_validation,
        "validate_unified_directory",
        forbidden_replay,
    )
    source = repository / "README.md"
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\nchanged after build\n")
        assert main(
            _export_args(run, repository, output, profile="accepted_only")
        ) == 2
        assert not output.exists()
    finally:
        source.write_bytes(original)


def test_consumer_entrypoints_reject_symlinked_input_roots(
    tmp_path: Path,
    complete_source,
) -> None:
    run, repository, _ = complete_source
    view = tmp_path / "real-input-view"
    assert main(
        _export_args(run, repository, view, profile="accepted_only")
    ) == 0
    source_link = tmp_path / "source-link"
    source_link.symlink_to(run, target_is_directory=True)
    repository_link = tmp_path / "repository-link"
    repository_link.symlink_to(repository, target_is_directory=True)

    assert main(
        _export_args(
            source_link,
            repository,
            tmp_path / "source-link-output",
            profile="accepted_only",
        )
    ) == 2
    assert main(_validate_args(view, source_link, repository)) == 2
    with pytest.raises(ValueError, match="source run cannot be a symlink"):
        consumer_module.render_standalone_consumer_view(
            source_link,
            repository_root=repository,
            profile="accepted_only",
        )
    assert main(
        _export_args(
            run,
            repository_link,
            tmp_path / "repository-link-output",
            profile="accepted_only",
        )
    ) == 2


def test_export_enforces_source_configured_tree_byte_bound(
    tmp_path: Path,
    complete_source,
) -> None:
    source_run, repository, _ = complete_source
    run = tmp_path / "over-bound-source"
    shutil.copytree(source_run, run, copy_function=shutil.copy2)
    config_path = run / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["unified"]["max_unified_artifact_bytes"] = 1_000
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "over-bound-output"
    assert main(
        _export_args(run, repository, output, profile="accepted_only")
    ) == 2
    assert not output.exists()


def test_partial_consumer_default_succeeds_and_strict_returns_one(
    tmp_path: Path,
    partial_source,
) -> None:
    run, repository, canonical = partial_source
    output = tmp_path / "partial-consumer"
    before = _tree_fingerprint(run)

    assert main(
        _export_args(run, repository, output, profile="accepted_only")
    ) == 0
    assert main(_validate_args(output, run, repository)) == 0
    assert main(
        _validate_args(
            output,
            run,
            repository,
            require_complete=True,
        )
    ) == 1
    start = (output / "START.md").read_text(encoding="utf-8")
    assert "This collection may omit repository conventions." in start
    assert "validation" not in start
    catalog = (output / "catalog.md").read_text(encoding="utf-8")
    accepted = [
        item.claim_id
        for item in canonical.claims
        if item.epistemic_status == "accepted"
    ]
    assert accepted
    assert all(claim_id in catalog for claim_id in accepted)
    assert _tree_fingerprint(run) == before


@pytest.mark.parametrize("mutation", ["tamper", "extra", "missing", "symlink"])
def test_consumer_closed_tree_mutations_are_invalid(
    tmp_path: Path,
    complete_source,
    mutation: str,
) -> None:
    run, repository, _ = complete_source
    output = tmp_path / f"consumer-{mutation}"
    assert main(
        _export_args(run, repository, output, profile="accepted_only")
    ) == 0

    catalog = output / "catalog.md"
    if mutation == "tamper":
        catalog.chmod(0o600)
        catalog.write_bytes(catalog.read_bytes() + b"\nTAMPERED\n")
    elif mutation == "extra":
        (output / "foreign.md").write_text("foreign\n", encoding="utf-8")
    elif mutation == "missing":
        catalog.unlink()
    else:
        catalog.unlink()
        catalog.symlink_to("START.md")

    assert main(_validate_args(output, run, repository)) == 2


def test_invalid_source_run_returns_two_and_never_publishes(
    tmp_path: Path,
    complete_source,
) -> None:
    source_run, repository, _ = complete_source
    run = tmp_path / "invalid-source-copy"
    shutil.copytree(source_run, run, copy_function=shutil.copy2)
    canonical = run / "canonical" / "claims.json"
    canonical.chmod(0o600)
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    payload["artifact_digest"] = "0" * 64
    canonical.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "must-not-publish"

    assert main(_export_args(run, repository, output)) == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".must-not-publish.staging-*"))


@pytest.mark.parametrize(
    "mutation",
    ["foreign_current", "missing_finding", "canonical_index", "git_ref"],
)
def test_export_rejects_renderer_subset_identity_mismatches(
    tmp_path: Path,
    complete_source,
    mutation: str,
) -> None:
    source_run, repository, _ = complete_source
    if mutation == "git_ref":
        class HistoricalNeedClient(MockLLMClient):
            root_id: str | None = None

            def complete(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
                response = super().complete(system_prompt, user_prompt)
                request = json.loads(user_prompt)
                request = request.get("original_request", request)
                if request.get("stage") != "explicit_semantic_verify":
                    return response
                candidates = request["candidates"]
                if self.root_id is None:
                    self.root_id = candidates[0]["candidate_root_id"]
                value = json.loads(response.content)
                by_root = {item["candidate_root_id"]: item for item in candidates}
                for result in value["results"]:
                    if result["candidate_root_id"] != self.root_id:
                        continue
                    evidence = by_root[self.root_id]["evidence"][0]
                    result["git_need"] = {
                        "status": "affirmative",
                        "questions": [{
                            "question": "Which previous setting did the cited configuration replace?",
                            "current_gap": "The current setting does not identify its predecessor.",
                            "answer_use": "Record the setting's replacement relation for this candidate.",
                            "path": evidence["evidence"]["path"],
                            "evidence_refs": [evidence["evidence_digest"]],
                            "blocking_axis": None,
                        }],
                    }
                return response.model_copy(update={"content": json.dumps(value)})

        # A clean mock no longer queries Git automatically. Explicitly exercise
        # a legal historical-need route so the digest-tamper case stays covered.
        git_repository = tmp_path / "git-ref-repository"
        shutil.copytree(repository, git_repository)
        repository = git_repository
        source_run = tmp_path / "git-ref-source"
        build_fixture(repository, source_run, client=HistoricalNeedClient())
    run = tmp_path / f"renderer-identity-{mutation}"
    shutil.copytree(source_run, run, copy_function=shutil.copy2)
    if mutation == "foreign_current":
        current = next(
            (run / "canonical/audit/current-evidence").glob("*.json")
        )
        shutil.copy2(current, current.parent / "foreign.json")
    elif mutation == "missing_finding":
        next((run / "canonical/audit/findings").glob("*.json")).unlink()
    elif mutation == "canonical_index":
        canonical = json.loads(
            (run / "canonical/claims.json").read_text(encoding="utf-8")
        )
        index_path = run / "canonical/audit/index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        removed = index["canonical_claim_ids"].pop()
        status = next(
            item["epistemic_status"]
            for item in canonical["claims"]
            if item["claim_id"] == removed
        )
        index[f"{status}_count"] -= 1
        index.pop("artifact_digest")
        index_path.write_text(
            consumer_module.canonical_json_text(seal_payload(index)),
            encoding="utf-8",
        )
    else:
        refs_path = run / "canonical/audit/git/refs.json"
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
        assert refs["refs"]
        refs["refs"][0]["artifact_digest"] = "0" * 64
        refs_path.write_text(
            json.dumps(refs, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    output = tmp_path / f"renderer-output-{mutation}"
    assert main(
        _export_args(run, repository, output, profile="accepted_only")
    ) == 2
    assert not output.exists()


def test_validation_rejects_source_toctou_after_authenticated_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_source,
) -> None:
    source_run, repository, _ = complete_source
    run = tmp_path / "validation-racing-source-copy"
    shutil.copytree(source_run, run, copy_function=shutil.copy2)
    view = tmp_path / "validated-before-race"
    assert main(
        _export_args(run, repository, view, profile="accepted_only")
    ) == 0

    real_validate = unified_validation.validate_unified_directory
    calls = 0

    def validate_then_mutate(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = real_validate(*args, **kwargs)
        path = run / "source_acquisition.json"
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(
        unified_validation,
        "validate_unified_directory",
        validate_then_mutate,
    )
    assert main(_validate_args(view, run, repository)) == 2
    assert calls == 1


def test_export_rejects_source_toctou_before_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_source,
) -> None:
    source_run, repository, _ = complete_source
    run = tmp_path / "export-racing-source-copy"
    shutil.copytree(source_run, run, copy_function=shutil.copy2)
    output = tmp_path / "must-not-publish-after-race"
    real_unchanged = consumer_module._source_files_unchanged

    def mutate_then_check(*args, **kwargs):
        path = run / "source_acquisition.json"
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"\n")
        return real_unchanged(*args, **kwargs)

    monkeypatch.setattr(
        consumer_module,
        "_source_files_unchanged",
        mutate_then_check,
    )
    assert main(
        _export_args(run, repository, output, profile="accepted_only")
    ) == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".must-not-publish-after-race.staging-*"))


def test_export_rejects_staging_mutation_after_source_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_source,
) -> None:
    run, repository, _ = complete_source
    output = tmp_path / "staging-race"
    real_unchanged = consumer_module._source_files_unchanged

    def check_then_mutate_staging(*args, **kwargs):
        result = real_unchanged(*args, **kwargs)
        staging = next(tmp_path.glob(".staging-race.staging-*"))
        start = staging / "START.md"
        start.chmod(0o600)
        start.write_bytes(start.read_bytes() + b"\nRACED\n")
        return result

    monkeypatch.setattr(
        consumer_module,
        "_source_files_unchanged",
        check_then_mutate_staging,
    )
    assert main(
        _export_args(run, repository, output, profile="accepted_only")
    ) == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".staging-race.staging-*"))


def test_validation_rejects_view_mutation_after_source_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_source,
) -> None:
    run, repository, _ = complete_source
    view = tmp_path / "view-race"
    assert main(
        _export_args(run, repository, view, profile="accepted_only")
    ) == 0
    real_validate = unified_validation.validate_unified_directory
    calls = 0

    def validate_then_mutate_view(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = real_validate(*args, **kwargs)
        start = view / "START.md"
        start.chmod(0o600)
        start.write_bytes(start.read_bytes() + b"\nRACED\n")
        return result

    monkeypatch.setattr(
        unified_validation,
        "validate_unified_directory",
        validate_then_mutate_view,
    )
    assert main(_validate_args(view, run, repository)) == 2
    assert calls == 1


def test_validation_rechecks_repository_after_single_source_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_source,
) -> None:
    run, repository, _ = complete_source
    view = tmp_path / "repository-race-view"
    assert main(
        _export_args(run, repository, view, profile="accepted_only")
    ) == 0
    real_validate = unified_validation.validate_unified_directory
    source = repository / "README.md"
    original = source.read_bytes()
    calls = 0

    def validate_then_mutate_repository(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = real_validate(*args, **kwargs)
        source.write_bytes(original + b"\nlate repository race\n")
        return result

    monkeypatch.setattr(
        unified_validation,
        "validate_unified_directory",
        validate_then_mutate_repository,
    )
    try:
        assert main(_validate_args(view, run, repository)) == 2
        assert calls == 1
    finally:
        source.write_bytes(original)


def test_standalone_manifest_cannot_self_authorize_profile_set(
    tmp_path: Path,
    complete_source,
) -> None:
    run, repository, _ = complete_source
    view = tmp_path / "manifest-profile-tamper"
    assert main(
        _export_args(run, repository, view, profile="accepted_only")
    ) == 0
    manifest_path = view / "consumer-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("artifact_digest")
    payload["requested_consumer_profiles"] = [
        "accepted_only",
        "accepted_plus_review",
    ]
    payload["emitted_consumer_profiles"] = [
        "accepted_only",
        "accepted_plus_review",
    ]
    manifest_path.chmod(0o600)
    manifest_path.write_text(
        consumer_module.canonical_json_text(seal_payload(payload)),
        encoding="utf-8",
    )

    assert main(_validate_args(view, run, repository)) == 2


def test_export_preserves_destination_created_after_initial_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_source,
) -> None:
    run, repository, _ = complete_source
    output = tmp_path / "appears-after-preflight"
    real_load = consumer_module._load_standalone_inputs

    def load_then_create_destination(*args, **kwargs):
        result = real_load(*args, **kwargs)
        output.mkdir()
        return result

    monkeypatch.setattr(
        consumer_module,
        "_load_standalone_inputs",
        load_then_create_destination,
    )
    assert main(
        _export_args(run, repository, output, profile="accepted_only")
    ) == 2
    assert output.is_dir()
    assert list(output.iterdir()) == []


def _race_completeness_to_complete(run: Path) -> None:
    path = run / "canonical" / "completeness.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"]["package_status"] = "complete"
    payload["state"]["omitted_possible"] = False
    payload["state"]["reasons"] = []
    path.chmod(0o600)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_validate_strict_status_uses_authenticated_sentinel_not_late_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial_source,
) -> None:
    source_run, repository, _ = partial_source
    run = tmp_path / "strict-full-race"
    shutil.copytree(source_run, run, copy_function=shutil.copy2)
    real_validate = cli_module.validate_unified_directory

    def validate_then_change_status(*args, **kwargs):
        result = real_validate(*args, **kwargs)
        assert unified_validation.INCOMPLETE_PACKAGE_ERROR in result
        _race_completeness_to_complete(run)
        return result

    monkeypatch.setattr(
        cli_module,
        "validate_unified_directory",
        validate_then_change_status,
    )
    assert main(
        [
            "validate-unified",
            str(run),
            "--repository",
            str(repository),
            "--require-complete",
        ]
    ) == 1


def test_validate_consumer_strict_status_uses_authenticated_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial_source,
) -> None:
    source_run, repository, _ = partial_source
    run = tmp_path / "strict-consumer-source-race"
    shutil.copytree(source_run, run, copy_function=shutil.copy2)
    view = tmp_path / "strict-consumer-view-race"
    assert main(
        _export_args(run, repository, view, profile="accepted_only")
    ) == 0
    real_validate = cli_module.validate_consumer_view

    def validate_then_change_status(*args, **kwargs):
        result = real_validate(*args, **kwargs)
        assert unified_validation.INCOMPLETE_PACKAGE_ERROR in result
        _race_completeness_to_complete(run)
        return result

    monkeypatch.setattr(
        cli_module,
        "validate_consumer_view",
        validate_then_change_status,
    )
    assert main(
        _validate_args(view, run, repository, require_complete=True)
    ) == 1


def test_atomic_no_replace_race_preserves_competing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_source,
) -> None:
    run, repository, _ = complete_source
    output = tmp_path / "racing-empty-destination"
    real_rename = consumer_module.rename_directory_noreplace

    def insert_destination_then_rename(source: Path, destination: Path) -> None:
        destination.mkdir()
        real_rename(source, destination)

    monkeypatch.setattr(
        consumer_module,
        "rename_directory_noreplace",
        insert_destination_then_rename,
    )
    assert main(
        _export_args(run, repository, output, profile="accepted_only")
    ) == 2
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob(".racing-empty-destination.staging-*"))
    assert not (tmp_path / ".racing-empty-destination.publish.lock").exists()


def test_consumer_cli_help_returns_zero() -> None:
    for command in ("export-consumer", "validate-consumer"):
        with pytest.raises(SystemExit) as exit_signal:
            main([command, "--help"])
        assert exit_signal.value.code == 0
