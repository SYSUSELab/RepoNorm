"""Provider-free fidelity, bounds, and path-safety tests for benchmark display."""
from __future__ import annotations

import hashlib
import posixpath
import re
from urllib.parse import unquote

import pytest

from evaluation import benchmark_projection as projection
from test_consumer_surface import _pair, _render


def _project(pairs, profile="accepted_only"):
    source = _render(pairs, profile=profile, partial=True)
    result = projection.render_benchmark_projection(
        [pair[0] for pair in pairs], source_files=source.files,
        consumer_profile=profile,
    )
    return source, result


def _assert_links(result):
    for path, data in result.files.items():
        for link in re.findall(r"\]\(([^)]+)\)", data.decode()):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(path), unquote(link)))
            assert target in result.files, (path, target)


def test_partial_projection_has_one_entry_complete_semantics_and_exact_details():
    pair = _pair("semantic", scopes=("src/core.py", "tests/**"),
                 condition="Only if `enabled` is true.", exception="Except generated files.",
                 risk="external_side_effect", delivery="condition_matched_only")
    source, result = _project([pair])
    start = result.files["START.md"].decode()
    assert "may omit" in source.files["START.md"].decode()
    for forbidden in ("omit", "incomplete", "partial", "SKILL", "by-path", "snapshot", "digest"):
        assert forbidden not in start
    assert "paths/src/core.py.md" in start
    assert "paths/tests.md" in start
    assert "observed" in start and "reference_only" in start
    assert not any(path.startswith("by-path/") or "SKILL" in path for path in result.files)
    claim = pair[0]
    for path in ("paths/src/core.py.md", "paths/tests.md", "catalog.md"):
        text = result.files[path].decode()
        assert "## Rule 001" in text
        for value in [claim.semantics.rule, *claim.semantics.applies_to,
                      claim.semantics.condition, claim.semantics.exception,
                      claim.normative_strength, claim.execution_risk]:
            assert value in text
        assert result.metadata["files"][path]["claim_ids"] == [claim.claim_id]
    detail = f"claims/{claim.claim_id}.md"
    assert result.files[detail] == source.files[detail]
    assert result.metadata["claims"][claim.claim_id]["source_claim_digest"] == claim.artifact_digest
    _assert_links(result)
    assert _project([pair])[1] == result


def test_reference_only_is_discoverable_in_full_catalog_not_promoted_to_path_action():
    pair = _pair("reference", delivery="reference_only", risk="repository_wide_mutation")
    _, result = _project([pair])
    assert not any(path.startswith("paths/") for path in result.files)
    assert "reference_only" in result.files["catalog.md"].decode()
    assert "repository_wide_mutation" in result.files["catalog.md"].decode()
    _assert_links(result)


def test_unconfirmed_rules_keep_usage_limitations_without_collection_warning():
    pair = _pair("pending", status="review", delivery="not_applicable",
                 facets=("factual_uncertainty", "technical_or_coverage_gap"),
                 dependency_status="incomplete")
    source, result = _project([pair], "accepted_plus_review")
    for path in ("catalog.md", "paths/src/pkg.py.md"):
        text = result.files[path].decode()
        assert "Unconfirmed (review)" in text
        assert "Proposed normative strength" in text
        assert "### Usage limitations" in text
        assert "Required dependency: incomplete" in text
    assert "may omit" not in result.files["START.md"].decode()
    detail = f"claims/{pair[0].claim_id}.md"
    assert result.files[detail] == source.files[detail]


def test_path_encoding_avoids_repository_file_directory_and_shard_collisions():
    scopes = ["*", "repository", "Repository", "a", "a.md/b", "a%2Emd/b",
              "a~part-0001", "a.part-0001", "目录/a.py", "x" * 255,
              "index.md", "constraints.md", "a[bc]/*.py"]
    pairs = [_pair(f"path-{ordinal}", scopes=(scope,)) for ordinal, scope in enumerate(scopes)]
    _, result = _project(pairs)
    assert "paths/repository.md" in result.files
    paths = list(result.files)
    assert len(paths) == len(set(paths))
    for path in paths:
        assert all(not other.startswith(path + "/") for other in paths if path != other)
        assert all(len(component.encode()) <= 255 for component in path.split("/"))
    assert len(result.metadata["path_anchors"]) == len(scopes) - 1  # both root globs
    _assert_links(result)


def test_cards_shard_at_complete_rule_boundaries_and_map_every_occurrence():
    pairs = [_pair(f"shard-{ordinal}", rule=(f"Rule {ordinal}. " + "x" * 400)) for ordinal in range(30)]
    _, result = _project(pairs)
    path_shards = [path for path in result.files if path.startswith("paths/")]
    assert len(path_shards) > 1
    seen = []
    for path in path_shards:
        assert len(result.files[path]) <= projection.TARGET_FILE_BYTES
        ids = result.metadata["files"][path]["claim_ids"]
        seen.extend(ids)
        for claim_id in ids:
            claim = next(pair[0] for pair in pairs if pair[0].claim_id == claim_id)
            assert claim.semantics.rule in result.files[path].decode()
    assert sorted(seen) == sorted(pair[0].claim_id for pair in pairs)
    for path, data in result.files.items():
        metadata = result.metadata["files"][path]
        assert metadata["size_bytes"] == len(data)
        assert metadata["sha256"] == hashlib.sha256(data).hexdigest()
    assert result.metadata["oversized_singleton_files"] == []
    _assert_links(result)


def test_single_long_rule_is_not_truncated_and_records_oversized_singleton(monkeypatch):
    monkeypatch.setattr(projection, "TARGET_FILE_BYTES", 500)
    pair = _pair("long", rule="Exact rule. " + "y" * 400)
    _, result = _project([pair])
    assert result.metadata["oversized_singleton_files"] == ["catalog.md", "paths/src/pkg.py.md"]
    assert pair[0].semantics.rule in result.files["catalog.md"].decode()


def test_singleton_above_frozen_limit_fails_instead_of_summarizing(monkeypatch):
    pair = _pair("long", rule="Exact rule. " + "y" * 400)
    source = _render([pair])
    monkeypatch.setattr(projection, "MAX_SINGLETON_FILE_BYTES", 500)
    with pytest.raises(ValueError, match="complete benchmark rule cannot fit"):
        projection.render_benchmark_projection([pair[0]], source_files=source.files, consumer_profile="accepted_only")


def test_navigation_only_shards_when_start_really_needs_them():
    pairs = [_pair(f"nav-{ordinal}", scopes=(f"src/module_{ordinal:04d}/core.py",)) for ordinal in range(110)]
    _, result = _project(pairs)
    assert any(path.startswith("navigation/") for path in result.files)
    assert len(result.files["START.md"]) <= projection.MAX_NAVIGATION_FILE_BYTES
    for path, data in result.files.items():
        if path.startswith("navigation/"):
            assert len(data) <= projection.MAX_NAVIGATION_FILE_BYTES
            assert result.metadata["files"][path]["claim_ids"] == []
    _assert_links(result)


@pytest.mark.parametrize("damage", ["missing", "foreign", "review_leak"])
def test_projection_does_not_relax_authenticated_membership(damage):
    accepted = _pair("accepted")
    review = _pair("review", status="review", delivery="not_applicable", facets=("factual_uncertainty",))
    source = _render([accepted, review], profile="accepted_plus_review")
    files = dict(source.files)
    if damage == "missing":
        files.pop(f"claims/{accepted[0].claim_id}.md")
        files.pop(f"claims/{review[0].claim_id}.md")
    elif damage == "foreign":
        files["claims/FOREIGN.md"] = b"foreign"
    with pytest.raises(ValueError):
        projection.render_benchmark_projection([accepted[0], review[0]], source_files=files, consumer_profile="accepted_only")


def test_empty_accepted_package_has_valid_searchable_catalog():
    _, result = _project([])
    assert set(result.files) == {"START.md", "catalog.md"}
    assert result.metadata["claims"] == {}
    _assert_links(result)
