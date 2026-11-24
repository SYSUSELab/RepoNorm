"""Read only 0.17.1 producers without sharing 0.18 inference identities."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import build_fixture, copied_fixture
from repo_constraint_memory_v3 import builder, current_source_run, inference
from repo_constraint_memory_v3.builder import protocol_manifest_for_config
from repo_constraint_memory_v3.inference import (
    ProtocolManifest,
    protocol_manifest_for_readonly_replay,
)
from repo_constraint_memory_v3.observations.canonical import payload_digest
from repo_constraint_memory_v3.run_validation import validate_run_directory
from test_current_source_only_run import _build_direct_run


def _legacy_producer(patch: pytest.MonkeyPatch) -> None:
    for module in (builder, inference, current_source_run):
        patch.setattr(module, "__version__", "0.17.1")


def test_legacy_full_run_validate_and_review_remain_readonly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from repo_constraint_memory_v3.review.service import (
        create_review, publish_review, validate_review_consumer,
    )
    from repo_constraint_memory_v3.unified.consumer import (
        export_consumer_view, validate_consumer_view,
    )

    repository = copied_fixture(tmp_path)
    run = tmp_path / "legacy-run"
    with monkeypatch.context() as patch:
        _legacy_producer(patch)
        result = build_fixture(repository, run)
    assert result.protocol_manifest.package_version == "0.17.1"
    before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    assert validate_run_directory(run, repository_root=repository) == []
    standalone = tmp_path / "standalone-consumer"
    export_consumer_view(
        run, repository_root=repository, profile="accepted_only",
        output_directory=standalone,
    )
    assert validate_consumer_view(
        standalone, source_run=run, repository_root=repository,
    ) == []
    review = tmp_path / "review"
    create_review(run, repository_root=repository, output_directory=review)
    output = tmp_path / "review-consumer"
    publish_review(review, repository_root=repository, output_directory=output)
    assert validate_review_consumer(
        output / "consumer", review_directory=review, repository_root=repository,
    ) == []
    after = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    assert after == before


def test_legacy_current_source_run_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        _legacy_producer(patch)
        repository, run, result = _build_direct_run(tmp_path)
    assert result.run_manifest.package_version == "0.17.1"
    assert current_source_run.validate_current_source_only_run(
        run, repository_root=repository,
    ) == []


@pytest.mark.parametrize("reuse", ["cache", "resume"])
def test_current_build_cannot_reuse_legacy_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reuse: str,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "legacy-run"
    cache = tmp_path / "inference-cache"
    with monkeypatch.context() as patch:
        _legacy_producer(patch)
        old = build_fixture(repository, run, inference_cache=cache)
    kwargs = {"inference_cache": cache} if reuse == "cache" else {"resume_from": run}
    new = build_fixture(repository, tmp_path / "new-run", **kwargs)
    assert new.protocol_manifest.package_version == "0.18.0"
    assert new.protocol_manifest.protocol_manifest_digest != old.protocol_manifest.protocol_manifest_digest
    assert new.build_report["llm"]["this_run_cache_hits"] == 0
    assert new.build_report["llm"]["this_run_provider_calls"] > 0
    assert validate_run_directory(tmp_path / "new-run", repository_root=repository) == []


def test_legacy_producer_does_not_allow_different_stage_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = copied_fixture(tmp_path)
    run = tmp_path / "different-contract-run"
    with monkeypatch.context() as patch:
        _legacy_producer(patch)
        patch.setattr(builder, "AST_CONTRACT_VERSION", "different-parser-contract")
        build_fixture(repository, run)
    errors = validate_run_directory(run, repository_root=repository)
    assert any("protocol manifest differs from resolved config" in error for error in errors)


@pytest.mark.parametrize("version", ["0.17.0", "0.16.0", "0.19.0"])
def test_other_producer_versions_are_not_supported(version: str) -> None:
    from helpers import config_for_tests

    current = protocol_manifest_for_config(config_for_tests(), agent_profile_digest="a" * 64)
    payload = current.model_dump(mode="json")
    payload["package_version"] = version
    del payload["protocol_manifest_digest"]
    payload["protocol_manifest_digest"] = payload_digest(payload)
    with pytest.raises(ValidationError):
        ProtocolManifest.model_validate(payload)
    with pytest.raises(ValueError, match="unsupported extraction producer"):
        protocol_manifest_for_readonly_replay(current, producer_package_version=version)
