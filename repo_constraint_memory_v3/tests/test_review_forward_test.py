"""Offline consumer forward test for the new reviewed handoff.

The existing skill forward test covers nonstandard source supplementation. This
fixture exercises the added partial-reviewed view through two native-file read
routes, with no canonical data or review ledger in the subject workspace.
"""
from __future__ import annotations

import posixpath
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

import pytest

from evaluation.consumer_handoff import prepare_consumer_handoff
from repo_constraint_memory_v3.review.service import (
    create_review, preflight_review_consumer, publish_review, save_decision,
    validate_review_consumer,
)
from repo_constraint_memory_v3.unified.publication import directory_tree_fingerprint
from test_consumer_cli import _mock_source_run


class NativeFileSubject:
    """Deterministic fake subject with only the selected view mounted."""

    def __init__(self, root: Path):
        self.root = root
        self.reads: list[str] = []

    def read(self, relative: str) -> str:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise PermissionError("outside selected consumer mount")
        text = path.read_text(encoding="utf-8")
        self.reads.append(relative)
        return text

    def follow(self, containing: str, destination: str) -> tuple[str, str]:
        relative = posixpath.normpath(posixpath.join(posixpath.dirname(containing), unquote(destination)))
        return relative, self.read(relative)


def _links(markdown: str) -> list[str]:
    return re.findall(r"\]\(([^)]+)\)", markdown)


@pytest.fixture(scope="module")
def reviewed_forward_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("reviewed-forward-controller")
    source, repository, canonical = _mock_source_run(root / "source", incomplete=True)
    review = root / "review-session"
    session = create_review(source, repository_root=repository, output_directory=review)
    candidate = next(
        claim for claim in canonical.claims
        if claim.claim_id in session.claim_digests
        and claim.execution_risk not in {"repository_wide_mutation", "external_side_effect"}
    )
    save_decision(review, repository_root=repository, claim_id=candidate.claim_id, decision="accept")
    plan = publish_review(review, repository_root=repository)
    return source, repository, review, plan, canonical, candidate.claim_id


@pytest.mark.parametrize("route", ["catalog", "path"])
def test_reviewed_partial_selected_view_forward_without_audit_gate(
    reviewed_forward_source, tmp_path, monkeypatch, route,
):
    source, repository, review, plan, canonical, accepted_id = reviewed_forward_source
    view = plan.publication_directory / "consumer"
    publication_before = directory_tree_fingerprint(view.parent, max_bytes=1_000_000_000)
    source_before = directory_tree_fingerprint(source, max_bytes=1_000_000_000)
    import repo_constraint_memory_v3.unified.validation as validator

    with monkeypatch.context() as context:
        context.setattr(validator, "validate_unified_directory", lambda *args, **kwargs: pytest.fail("no full audit before consumption"))
        assert preflight_review_consumer(view, review_directory=review, repository_root=repository).package_status == "partial"
        subject_workspace = tmp_path / "subject"
        receipt = prepare_consumer_handoff(
            view, source_run=source, repository_root=repository,
            output_directory=subject_workspace / "selected-consumer", review_directory=review,
        )
        assert receipt["task_tree_digest"]
        assert not (subject_workspace / "selected-consumer/consumer-manifest.json").exists()
        shutil.copytree(repository, subject_workspace / "target-repository")
        assert {path.name for path in subject_workspace.iterdir()} == {"selected-consumer", "target-repository"}
        subject = NativeFileSubject(subject_workspace / "selected-consumer")
        start = subject.read("START.md")
        assert "may omit repository conventions" in start
        assert all(word not in start.lower() for word in ("snapshot", "validation", "audit", "profile", "accepted_after_review"))
        if route == "catalog":
            containing = "catalog.md"
            text = subject.read(containing)
        else:
            index = subject.read("by-path/index.md")
            for destination in _links(index):
                containing, text = subject.follow("by-path/index.md", destination)
                if accepted_id in text:
                    break
            else:
                pytest.fail("review-accepted claim is not reachable through its path views")
        destination = next(link for link in _links(text) if link.endswith(accepted_id + ".md"))
        _, detail = subject.follow(containing, destination)
        assert "developer_confirmation" in detail
        assert subject.reads.count("START.md") == 1 and subject.reads[0] == "START.md"
        assert subject.reads[1] == ("catalog.md" if route == "catalog" else "by-path/index.md")
        assert len(subject.reads) < len(plan.files)
        for forbidden in ("consumer-manifest.json", "canonical/claims.json", "decisions.json", "audit/receipt.json", "by-trigger/index.md"):
            with pytest.raises(FileNotFoundError):
                subject.read(forbidden)
        with pytest.raises(PermissionError):
            subject.read("../decisions.json")
        exposed_ids = {path.stem for path in (subject.root / "claims").iterdir()}
        expected_ids = {claim.claim_id for claim in canonical.claims if claim.epistemic_status == "accepted"} | {accepted_id}
        assert exposed_ids == expected_ids

    # An independent audit after actual consumption leaves all source bytes intact.
    assert validate_review_consumer(view, review_directory=review, repository_root=repository) == []
    assert directory_tree_fingerprint(view.parent, max_bytes=1_000_000_000) == publication_before
    assert directory_tree_fingerprint(source, max_bytes=1_000_000_000) == source_before
