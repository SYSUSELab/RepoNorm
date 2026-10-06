from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src" / "repo_constraint_memory_v3"
CONFIG_ROOT = PACKAGE_ROOT / "configs"


def _active_text(root: Path, suffixes: set[str]) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    )


def test_deleted_runtime_paths_and_family_contracts_do_not_exist() -> None:
    assert not (SOURCE_ROOT / "implicit").exists()
    assert not (SOURCE_ROOT / "observations" / "companions.py").exists()
    assert not (SOURCE_ROOT / "observations" / "pytest_markers.py").exists()

    active = _active_text(SOURCE_ROOT, {".py"}) + _active_text(
        CONFIG_ROOT,
        {".yaml"},
    )
    for forbidden in (
        "companion_artifact",
        "test_classification",
        "canonical_generated",
        "architecture_import_boundary",
        "ImplicitFamily",
        "QualificationRegistry",
        "StaticPatternEvidence",
        "ImplicitReviewItem",
        "python_companion_relations",
        "pytest_marker_population",
    ):
        assert forbidden not in active


def test_observation_registry_is_exactly_the_generic_pair() -> None:
    from repo_constraint_memory_v3.observations.registry import QUERY_REGISTRY

    assert list(QUERY_REGISTRY) == [
        "python_snapshot",
        "python_clone_clusters",
    ]
