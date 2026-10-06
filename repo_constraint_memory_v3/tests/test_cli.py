from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import yaml

import repo_constraint_memory_v3.cli as cli_module
from repo_constraint_memory_v3.cli import main
from repo_constraint_memory_v3.agent_runtime import (
    fake_agent_profile,
    stable_json_text,
)
from repo_constraint_memory_v3.run_receipt import load_command_run_receipt

from helpers import CONFIG_PATH, copied_fixture


def _fake_profile(tmp_path: Path) -> Path:
    path = tmp_path / "fake-agent-profile.json"
    value = fake_agent_profile().model_dump(mode="json")
    for key in (
        "execution_model_profile_id",
        "execution_model_profile_digest",
        "model",
    ):
        value.pop(key)
    path.write_text(stable_json_text(value), encoding="utf-8")
    return path


def _real_profile() -> Path:
    return (
        CONFIG_PATH.parent
        / "agents"
        / "mini_swe_agent_2_4_6.example.yaml"
    )


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def test_root_and_subcommand_help_return_zero() -> None:
    for argv in (
        ["--help"],
        ["discover", "--help"],
        ["build", "--help"],
        ["validate", "--help"],
        ["model-profiles", "--help"],
    ):
        try:
            main(argv)
        except SystemExit as exit_signal:
            assert exit_signal.code == 0
        else:
            raise AssertionError("argparse help must exit cleanly")


def test_cli_mock_discover_build_validate_round_trip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_fixture(tmp_path)
    work = tmp_path / "work"
    acquisition = work / "acquisition"
    run = work / "run"
    discover_receipt = work / "receipts" / "discover.json"
    build_receipt = work / "receipts" / "build.json"
    profile = _fake_profile(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-read-in-mock")

    assert (
        main(
            [
                "discover",
                str(repository),
                "--config",
                str(CONFIG_PATH),
                "--agent-profile",
                str(profile),
                "--out",
                str(acquisition),
                "--run-receipt",
                str(discover_receipt),
                "--mock-agent",
            ]
        )
        == 0
    )
    assert (
        main(
            [
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
                str(tmp_path / "observation-cache"),
                "--run-receipt",
                str(build_receipt),
                "--mock",
            ]
        )
        # No evidence-bound historical need was established, so the non-Git
        # fixture can complete without a temporal query.
        == 0
    )
    assert main(
        ["validate", str(run), "--repository", str(repository)]
    ) == 0
    for receipt_path, command, output in (
        (discover_receipt, "discover", acquisition),
        (build_receipt, "build", run),
    ):
        receipt = load_command_run_receipt(receipt_path)
        assert receipt.command == command
        assert receipt.phase == "terminal"
        assert receipt.outcome == "complete"
        assert receipt.exit_code == 0
        assert receipt.wall_seconds is not None and receipt.wall_seconds >= 0
        assert receipt.peak_rss_kib is not None and receipt.peak_rss_kib > 0
        assert receipt.output == str(output.resolve())
        assert receipt.execution_model_profile_id == "offline-deterministic-test"
        assert receipt.provider_cached_input is not None
        assert receipt.provider_cached_input.status == "not_applicable"


def test_cli_snapshot_mismatch_is_input_error(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    acquisition = tmp_path / "acquisition"
    receipt_path = tmp_path / "failed-build-receipt.json"
    profile = _fake_profile(tmp_path)
    assert (
        main(
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
        )
        == 0
    )
    (repository / "README.md").write_text(
        "# Changed after discovery\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "build",
                str(repository),
                "--config",
                str(CONFIG_PATH),
                "--agent-profile",
                str(profile),
                "--acquisition",
                str(acquisition),
                "--out",
                str(tmp_path / "run"),
                "--run-receipt",
                str(receipt_path),
                "--mock",
            ]
        )
        == 2
    )
    receipt = load_command_run_receipt(receipt_path)
    assert receipt.outcome == "failed"
    assert receipt.exit_code == 2
    assert receipt.provider_cached_input is not None
    assert receipt.provider_cached_input.status == "not_applicable"


def test_cli_rejects_run_receipt_inside_command_output(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    acquisition = tmp_path / "acquisition"

    assert main(
        [
            "discover",
            str(repository),
            "--config",
            str(CONFIG_PATH),
            "--agent-profile",
            str(_fake_profile(tmp_path)),
            "--out",
            str(acquisition),
            "--run-receipt",
            str(acquisition / "receipt.json"),
            "--mock-agent",
        ]
    ) == 2
    assert not acquisition.exists()


def test_cli_rejects_head_only_snapshot_movement(tmp_path: Path) -> None:
    """The fixed snapshot identity is the file digest plus its Git commit."""

    repository = copied_fixture(tmp_path)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Norm4Code Test")
    _git(repository, "config", "user.email", "norm4code@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "initial snapshot")

    acquisition = tmp_path / "acquisition"
    profile = _fake_profile(tmp_path)
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

    # No repository byte changes, but the history cutoff is now different.
    _git(repository, "commit", "-q", "--allow-empty", "-m", "move HEAD")
    assert main(
        [
            "build",
            str(repository),
            "--config",
            str(CONFIG_PATH),
            "--agent-profile",
            str(profile),
            "--acquisition",
            str(acquisition),
            "--out",
            str(tmp_path / "run"),
            "--mock",
        ]
    ) == 2


def test_cli_partial_build_has_integrity_and_completeness_validation_modes(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path, incomplete=True)
    acquisition = tmp_path / "acquisition"
    run = tmp_path / "run"
    receipt_path = tmp_path / "build-receipt.json"
    profile = _fake_profile(tmp_path)
    assert (
        main(
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
        )
        == 0
    )
    assert (
        main(
            [
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
                "--run-receipt",
                str(receipt_path),
                "--mock",
            ]
        )
        == 1
    )
    receipt = load_command_run_receipt(receipt_path)
    assert receipt.outcome == "valid_partial"
    assert receipt.exit_code == 1
    assert main(
        ["validate", str(run), "--repository", str(repository)]
    ) == 0
    assert main(
        [
            "validate",
            str(run),
            "--repository",
            str(repository),
            "--require-complete",
        ]
    ) == 1


def test_cli_unknown_observation_query_is_input_error(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    config_value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config_value["observation"]["queries"] = [
        {"name": "unknown_query", "required": True}
    ]
    config_path = tmp_path / "unknown-query.yaml"
    config_path.write_text(
        yaml.safe_dump(config_value, sort_keys=False),
        encoding="utf-8",
    )
    acquisition = tmp_path / "acquisition"
    profile = _fake_profile(tmp_path)
    assert main(
        [
            "discover",
            str(repository),
            "--config",
            str(config_path),
            "--agent-profile",
            str(profile),
            "--out",
            str(acquisition),
            "--mock-agent",
        ]
    ) == 2
    assert not acquisition.exists()


def test_cli_mock_agent_cannot_impersonate_a_real_framework(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    profile_value = fake_agent_profile().model_dump(mode="json")
    for key in (
        "execution_model_profile_id",
        "execution_model_profile_digest",
        "model",
    ):
        profile_value.pop(key)
    profile_value["framework_id"] = "mini-swe-agent"
    profile_value["framework_version"] = "2.4.6"
    profile = tmp_path / "impersonating-profile.json"
    profile.write_text(stable_json_text(profile_value), encoding="utf-8")

    assert main(
        [
            "discover",
            str(repository),
            "--config",
            str(CONFIG_PATH),
            "--agent-profile",
            str(profile),
            "--out",
            str(tmp_path / "acquisition"),
            "--mock-agent",
        ]
    ) == 2


def test_cli_mock_agent_requires_the_exact_fake_profile(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    profile_value = fake_agent_profile().model_dump(mode="json")
    for key in (
        "execution_model_profile_id",
        "execution_model_profile_digest",
        "model",
    ):
        profile_value.pop(key)
    profile_value["profile_id"] = "modified-fake-profile"
    profile = tmp_path / "modified-fake-profile.json"
    profile.write_text(stable_json_text(profile_value), encoding="utf-8")

    assert main(
        [
            "discover",
            str(repository),
            "--config",
            str(CONFIG_PATH),
            "--agent-profile",
            str(profile),
            "--out",
            str(tmp_path / "acquisition"),
            "--mock-agent",
        ]
    ) == 2


def test_cli_real_discover_loads_dotenv_before_starting_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_fixture(tmp_path)
    profile = _real_profile()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        cli_module,
        "find_dotenv",
        lambda *, usecwd: calls.append(("find", usecwd)) or "/tmp/test.env",
    )
    monkeypatch.setattr(
        cli_module,
        "load_dotenv",
        lambda path, *, override: calls.append(("load", (path, override))) or True,
    )

    def fake_acquire(repository, config, agent_profile, model_profile, runtime, out):
        del repository, config, runtime, out
        calls.append(("profile", model_profile.profile_id))
        return SimpleNamespace(
            discovery=SimpleNamespace(
                manifest=SimpleNamespace(files=[]),
                baseline_sources=[],
                repository=SimpleNamespace(snapshot_digest="sha256:" + "0" * 64),
            ),
            supplement=SimpleNamespace(proposals=[]),
            profile=agent_profile,
            model_profile=model_profile,
        )

    monkeypatch.setattr(cli_module, "acquire_sources", fake_acquire)

    assert main(
        [
            "discover",
            str(repository),
            "--config",
            str(CONFIG_PATH),
            "--agent-profile",
            str(profile),
            "--out",
            str(tmp_path / "acquisition"),
        ]
    ) == 0
    assert calls == [
        ("find", True),
        ("load", ("/tmp/test.env", False)),
        ("profile", "apiyi-qwen3.8-flash-thinking-medium"),
    ]


def test_cli_discover_never_loads_dotenv_from_target_repository(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = copied_fixture(tmp_path)
    profile = _real_profile()
    target_dotenv = repository / ".env"
    calls: list[str] = []

    monkeypatch.setattr(
        cli_module,
        "find_dotenv",
        lambda *, usecwd: target_dotenv.as_posix(),
    )
    monkeypatch.setattr(
        cli_module,
        "load_dotenv",
        lambda *args, **kwargs: calls.append("loaded"),
    )
    monkeypatch.setattr(
        cli_module,
        "acquire_sources",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("acquisition must not run")
        ),
    )

    assert main(
        [
            "discover",
            str(repository),
            "--config",
            str(CONFIG_PATH),
            "--agent-profile",
            str(profile),
            "--out",
            str(tmp_path / "acquisition"),
        ]
    ) == 2
    assert calls == []


def test_cli_build_rejects_another_agent_profile(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    profile = _fake_profile(tmp_path)
    acquisition = tmp_path / "acquisition"
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

    other_value = fake_agent_profile().model_dump(mode="json")
    for key in (
        "execution_model_profile_id",
        "execution_model_profile_digest",
        "model",
    ):
        other_value.pop(key)
    other_value["profile_id"] = "different-profile"
    other_profile = tmp_path / "other-profile.json"
    other_profile.write_text(stable_json_text(other_value), encoding="utf-8")
    assert main(
        [
            "build",
            str(repository),
            "--config",
            str(CONFIG_PATH),
            "--agent-profile",
            str(other_profile),
            "--acquisition",
            str(acquisition),
            "--out",
            str(tmp_path / "run"),
            "--mock",
        ]
    ) == 2
