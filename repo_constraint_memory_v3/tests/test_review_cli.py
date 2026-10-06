from __future__ import annotations

from pathlib import Path

import pytest

from repo_constraint_memory_v3.cli import main
from repo_constraint_memory_v3.review.service import load_review, save_decision
from repo_constraint_memory_v3.unified.publication import directory_tree_fingerprint
from test_consumer_cli import _mock_source_run


@pytest.fixture(scope="module")
def review_cli_source(tmp_path_factory):
    return _mock_source_run(tmp_path_factory.mktemp("review-cli-source"), incomplete=True)


@pytest.mark.parametrize("command", [
    "review", "review-publish", "review-auto", "check-review-consumer",
    "validate-review-consumer",
])
def test_review_command_help(command, capsys):
    with pytest.raises(SystemExit) as result:
        main([command, "--help"])
    assert result.value.code == 0
    assert command in capsys.readouterr().out


def test_review_cli_init_resume_publish_and_independent_validation(
    review_cli_source, tmp_path, monkeypatch, capsys,
):
    source, repository, canonical = review_cli_source
    session = tmp_path / "review"
    repo_args = ["--repository", str(repository)]
    before = directory_tree_fingerprint(source, max_bytes=1_000_000_000)
    assert main(["review", "--run", str(source), "--out", str(session), *repo_args, "--no-serve"]) == 0
    assert main(["review", "--session", str(session), *repo_args, "--no-serve"]) == 0
    claim_id = next(iter(load_review(session, repository_root=repository).claim_digests))
    save_decision(session, repository_root=repository, claim_id=claim_id, decision="accept")
    assert main(["review-publish", str(session), *repo_args]) == 0
    view = session / "publications/000001/consumer"
    assert str(view) in capsys.readouterr().out

    def forbidden_replay(*args, **kwargs):
        pytest.fail("the consumer binding check must not run full offline replay")

    import repo_constraint_memory_v3.unified.validation as validator
    with monkeypatch.context() as context:
        context.setattr(validator, "validate_unified_directory", forbidden_replay)
        assert main(["check-review-consumer", str(view), "--review", str(session), *repo_args]) == 0
    assert main(["validate-review-consumer", str(view), "--review", str(session), *repo_args]) == 0
    assert main(["validate-review-consumer", str(view), "--review", str(session), *repo_args, "--require-complete"]) == 1
    assert directory_tree_fingerprint(source, max_bytes=1_000_000_000) == before
    assert {path.stem for path in (view / "claims").iterdir()} == {
        item.claim_id for item in canonical.claims if item.epistemic_status == "accepted"
    } | {claim_id}
    (view / "extra.md").write_text("unexpected", encoding="utf-8")
    assert main(["check-review-consumer", str(view), "--review", str(session), *repo_args]) == 2
    assert main(["validate-review-consumer", str(view), "--review", str(session), *repo_args]) == 2


def test_review_cli_checks_creation_arguments_without_mutation(review_cli_source, tmp_path):
    source, repository, _ = review_cli_source
    repo_args = ["--repository", str(repository)]
    assert main(["review", "--run", str(source), *repo_args, "--no-serve"]) == 2
    destination = tmp_path / "review"
    assert main(["review", "--run", str(source), "--out", str(destination), *repo_args, "--port", "-1", "--no-serve"]) == 2
    assert not destination.exists()
    assert main(["review", "--session", str(tmp_path / "missing"), "--out", str(destination), *repo_args, "--no-serve"]) == 2
    assert not destination.exists()


def test_review_cli_subscription_runner_mock_smoke(review_cli_source, tmp_path, monkeypatch, capsys):
    source, repository, _ = review_cli_source
    session = tmp_path / "auto"
    repo_args = ["--repository", str(repository)]
    assert main(["review", "--run", str(source), "--out", str(session), *repo_args, "--no-serve"]) == 0

    def forbidden_provider(*args, **kwargs):
        pytest.fail("mock review must not authenticate or invoke Codex")

    import subprocess
    # Repository identity checks can run Git, so block Codex specifically.
    original_run = subprocess.run

    def checked_run(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and Path(str(command[0])).name == "codex":
            forbidden_provider()
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", checked_run)
    assert main(["review-auto", str(session), *repo_args, "--mock"]) == 0
    assert load_review(session, repository_root=repository).progress()["unprocessed"] == 0
    assert main(["review-publish", str(session), *repo_args]) == 0
    assert "accepted_after_review" in capsys.readouterr().out
