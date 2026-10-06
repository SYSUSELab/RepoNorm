from __future__ import annotations

from pathlib import Path

import pytest

from repo_constraint_memory_v3.discovery import (
    UnsupportedRepositoryError,
    build_workspace_manifest,
    discover_repository,
    matches_glob,
    source_kind_for_path,
)

from helpers import config_for_tests, copied_fixture


def test_recursive_glob_matches_zero_or_more_directory_levels() -> None:
    pattern = "src/pkg/**/*.py"

    assert matches_glob("src/pkg/mod.py", pattern)
    assert matches_glob("src/pkg/nested/deep/mod.py", pattern)
    assert not matches_glob("src/other/mod.py", pattern)


def test_invalid_glob_character_range_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="invalid character class"):
        matches_glob("src/pkg/mod.py", "src/[z-a].py")


def test_discovery_is_stable_and_finds_expected_sources(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()

    first = discover_repository(repository, config.discovery)
    second = discover_repository(repository, config.discovery)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    baseline = [item.path for item in first.baseline_sources]
    candidates = [item.path for item in first.supplement_candidates]
    assert baseline == sorted(baseline)
    assert "AGENTS.md" in baseline
    assert "src/pkg/AGENTS.md" in baseline
    assert "pyproject.toml" in baseline
    assert "Makefile" in baseline
    assert ".github/workflows/tests.yml" in baseline
    assert "docs/team_practices.rst" not in baseline
    assert "docs/team_practices.rst" in candidates
    assert "docs/product_behavior.md" in candidates
    assert not any(path.endswith(".py") for path in baseline + candidates)
    assert any(item.role == "python_source" for item in first.manifest.files)


def test_non_python_repository_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "not-python"
    repository.mkdir()
    (repository / "README.md").write_text("# Docs\n", encoding="utf-8")
    config = config_for_tests()

    with pytest.raises(UnsupportedRepositoryError, match=r"no eligible \.py"):
        discover_repository(repository, config.discovery)


def test_manifest_records_symlink_secret_binary_and_unsafe_paths(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    (repository / "CLAUDE.md").unlink()
    (repository / "CLAUDE.md").symlink_to("AGENTS.md")
    (repository / ".env").write_text("SECRET=not-for-scanning\n", encoding="utf-8")
    (repository / "binary.md").write_bytes(b"hello\0world")
    config = config_for_tests()

    artifact = discover_repository(repository, config.discovery)
    skipped = {(item.path, item.reason) for item in artifact.manifest.skipped}
    assert ("CLAUDE.md", "symlink") in skipped
    assert (".env", "sensitive_filename") in skipped
    assert ("binary.md", "binary_content") in skipped
    assert artifact.complete

    manifest = build_workspace_manifest(
        repository,
        config.discovery,
        candidate_paths=["../escape", "/absolute", "AGENTS.md"],
    )
    assert {item.reason for item in manifest.section.skipped} == {"unsafe_path"}
    assert [item.path for item in manifest.section.files] == ["AGENTS.md"]


def test_oversized_baseline_is_explicitly_incomplete(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    (repository / "CONTRIBUTING.md").write_text("x" * 256, encoding="utf-8")
    config = config_for_tests()
    config = config.model_copy(
        update={
            "discovery": config.discovery.model_copy(
                update={"max_manifest_file_bytes": 128}
            )
        }
    )

    artifact = discover_repository(repository, config.discovery)

    assert not artifact.complete
    assert "max_manifest_file_bytes_exceeded" in artifact.incomplete_reasons


def test_credential_shaped_content_is_an_admitted_supplement_candidate(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    (repository / "docs" / "credentials.txt").write_text(
        "API_KEY = abcdefghijklmnopqrstuvwxyz\n",
        encoding="utf-8",
    )
    config = config_for_tests()

    artifact = discover_repository(repository, config.discovery)

    candidate = next(
        item
        for item in artifact.supplement_candidates
        if item.path == "docs/credentials.txt"
    )
    assert candidate.reason == "supported_text_candidate"
    assert not any(
        item.path == "docs/credentials.txt" for item in artifact.manifest.skipped
    )
    assert source_kind_for_path(".editorconfig") == "configuration"
    assert source_kind_for_path(".flake8") == "configuration"
    assert source_kind_for_path(".coveragerc") == "configuration"


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        (
            "pyproject.toml",
            '[tool.service]\nauthToken = """\nabcdefghijklmnop\n"""\n',
        ),
        (
            ".github/workflows/tests.yml",
            "name: tests\nservice: {api_key: abcdefghijklmnop}\n",
        ),
    ],
)
def test_valid_credential_shaped_structured_sources_are_admitted(
    tmp_path: Path,
    relative_path: str,
    content: str,
) -> None:
    repository = copied_fixture(tmp_path)
    target = repository / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    artifact = discover_repository(repository, config_for_tests().discovery)

    assert relative_path in {item.path for item in artifact.baseline_sources}
    assert not any(
        item.path == relative_path for item in artifact.manifest.skipped
    )


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        ("pyproject.toml", 'api_key = "first"\napi_key = "second"\n'),
        (
            ".github/workflows/tests.yml",
            "name: tests\nname: duplicate\n",
        ),
        ("pyproject.toml", 'api_key = "unterminated\n'),
    ],
)
def test_structurally_invalid_baseline_sources_remain_fail_closed(
    tmp_path: Path,
    relative_path: str,
    content: str,
) -> None:
    repository = copied_fixture(tmp_path)
    target = repository / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    artifact = discover_repository(repository, config_for_tests().discovery)

    assert relative_path not in {
        item.path for item in artifact.baseline_sources
    }
    assert any(
        item.path == relative_path
        and item.reason == "structured_content_unavailable"
        for item in artifact.manifest.skipped
    )
    assert not artifact.complete
    assert "structured_baseline_source_unavailable" in (
        artifact.incomplete_reasons
    )
