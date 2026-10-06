from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_constraint_memory_v3.models import SourceSupplement
from repo_constraint_memory_v3.source_plan import (
    SourcePlanError,
    empty_supplement,
    validate_source_plan,
)

from helpers import (
    config_for_tests,
    copied_fixture,
    discover_fixture,
    supplement_for,
)


def test_empty_and_agent_supplements_are_admitted(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)

    empty = validate_source_plan(
        discovery,
        empty_supplement(discovery.repository.snapshot_digest),
        config.discovery,
    )
    agent = validate_source_plan(
        discovery,
        supplement_for(discovery, "docs/team_practices.rst"),
        config.discovery,
    )

    assert empty.proposed_count == 0
    assert empty_supplement(discovery.repository.snapshot_digest).search_status == (
        "complete"
    )
    assert empty.agent_only_source_count == 0
    assert agent.accepted_count == 1
    assert agent.agent_only_source_count == 1
    assert dict((item.path, origin) for item, origin in agent.sources)[
        "docs/team_practices.rst"
    ] == "agent_supplement"


def test_deterministic_overlap_is_recorded_without_changing_origin(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)

    result = validate_source_plan(
        discovery,
        supplement_for(discovery, "AGENTS.md"),
        config.discovery,
    )

    assert result.overlap_count == 1
    assert result.agent_only_source_count == 0
    assert dict((item.path, origin) for item, origin in result.sources)[
        "AGENTS.md"
    ] == "deterministic"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../escape.md", "unsafe"),
        ("/absolute.md", "unsafe"),
        ("src/pkg/first.py", "ordinary Python"),
        ("unknown.md", "not in the discovery manifest"),
    ],
)
def test_invalid_proposal_paths_fail_the_whole_plan(
    tmp_path: Path,
    path: str,
    message: str,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)

    with pytest.raises(SourcePlanError, match=message):
        validate_source_plan(
            discovery,
            supplement_for(discovery, path),
            config.discovery,
        )


def test_snapshot_mismatch_and_proposal_limit_fail(tmp_path: Path) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    mismatch = supplement_for(discovery, "docs/team_practices.rst").model_copy(
        update={"snapshot_digest": "sha256:" + "0" * 64}
    )
    with pytest.raises(SourcePlanError, match="snapshot_digest"):
        validate_source_plan(discovery, mismatch, config.discovery)

    limited = config.discovery.model_copy(
        update={"max_supplement_proposals": 0}
    )
    with pytest.raises(SourcePlanError, match="max_supplement_proposals"):
        validate_source_plan(
            discovery,
            supplement_for(discovery, "docs/team_practices.rst"),
            limited,
        )


def test_compared_baseline_paths_are_structured_and_host_validated(
    tmp_path: Path,
) -> None:
    repository = copied_fixture(tmp_path)
    config = config_for_tests()
    discovery = discover_fixture(repository, config)
    supplement = supplement_for(discovery, "docs/team_practices.rst")
    payload = supplement.model_dump(mode="json")
    payload["proposals"][0]["compared_baseline_paths"] = [
        "docs/not-a-baseline.md"
    ]

    with pytest.raises(SourcePlanError, match="outside discovery.baseline_sources"):
        validate_source_plan(
            discovery,
            SourceSupplement.model_validate(payload),
            config.discovery,
        )


def test_duplicate_paths_are_schema_invalid() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        SourceSupplement.model_validate(
            {
                "schema_version": "1.2",
                "snapshot_digest": "sha256:" + "0" * 64,
                "search_status": "complete",
                "proposals": [
                    {
                        "path": "a.md",
                        "compared_baseline_paths": [],
                        "reason": "first",
                    },
                    {
                        "path": "a.md",
                        "compared_baseline_paths": [],
                        "reason": "second",
                    },
                ],
            }
        )


def test_compared_baseline_paths_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="compared baseline paths"):
        SourceSupplement.model_validate(
            {
                "schema_version": "1.2",
                "snapshot_digest": "sha256:" + "0" * 64,
                "search_status": "complete",
                "proposals": [
                    {
                        "path": "a.md",
                        "compared_baseline_paths": ["AGENTS.md", "AGENTS.md"],
                        "reason": "duplicate comparison",
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "payload_update",
    [
        {"schema_version": "1.0"},
        {"search_status": "unknown"},
    ],
)
def test_supplement_requires_v12_search_status(payload_update: dict[str, str]) -> None:
    payload = {
        "schema_version": "1.2",
        "snapshot_digest": "sha256:" + "0" * 64,
        "search_status": "complete",
        "proposals": [],
    }
    payload.update(payload_update)

    with pytest.raises(ValidationError):
        SourceSupplement.model_validate(payload)
