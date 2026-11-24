from __future__ import annotations

import json
from zipfile import ZipFile

from repo_constraint_memory_v3.cli import main
from repo_constraint_memory_v3.explorer.package import load_package
from test_consumer_cli import _mock_source_run


def test_explorer_loads_actual_build_and_consumer_views(tmp_path):
    run, _, canonical = _mock_source_run(tmp_path, incomplete=True, emit_plus_review=True)
    package = load_package(run)
    assert package["coverage"] == "partial"
    assert {row["id"]: row["status"] for row in package["claims"]} == {
        claim.claim_id: claim.epistemic_status.title() for claim in canonical.claims
    }
    for profile in ("accepted_only", "accepted_plus_review"):
        view = run / "consumers" / profile
        directory = load_package(view)
        archive = tmp_path / (profile + ".zip")
        with ZipFile(archive, "w") as output:
            for file in view.rglob("*"):
                if file.is_file():
                    output.write(file, file.relative_to(view))
        assert load_package(archive) == directory
        by_id = {row["id"]: row for row in directory["claims"]}
        for claim in canonical.claims:
            if claim.claim_id in by_id:
                row = by_id[claim.claim_id]
                assert row["rule"] == claim.semantics.rule
                assert row["scopes"] == claim.semantics.applies_to
                assert row["status"] == claim.epistemic_status.title()
                assert row["strength"] == claim.normative_strength


def test_empty_and_missing_view_are_distinguishable(tmp_path):
    (tmp_path / "START.md").write_text("# Repository knowledge\n")
    assert main(["visualize", str(tmp_path), "--no-open"]) == 2
    manifest = tmp_path / "consumer-manifest.json"
    manifest.write_text(json.dumps({"detail_file_count": 0, "package_status": "partial"}))
    assert load_package(tmp_path)["claims"] == []
    manifest.write_text(json.dumps({"detail_file_count": 1}))
    assert main(["visualize", str(tmp_path), "--no-open"]) == 2


def test_review_card_preserves_rule_and_evidence_fences(tmp_path):
    (tmp_path / "START.md").write_text("# Repository knowledge\n")
    (tmp_path / "claims").mkdir()
    (tmp_path / "consumer-manifest.json").write_text(json.dumps({"consumer_profile": "accepted_plus_review"}))
    claim = tmp_path / "claims/CLAIM-0123456789abcdef01234567.md"
    claim.write_text(
        "# Rule details\n\n- **Status:** Unconfirmed (review)\n"
        "- **Rule:**\n  ````text\n  Preserve `value`.\n  Keep ``` intact.\n  ````\n"
        "- **Applies to:**\n  - `src/**`\n- **Exception:** Generated files\n"
        "- **Proposed normative strength:** `recommended`\n"
        "## Counterevidence\n\n````text\n## Literal heading\n````\n"
        "- **Sources:**\n  - `src/generated.py:2-5`\n"
    )
    row, = load_package(tmp_path)["claims"]
    assert row["rule"] == "Preserve `value`.\nKeep ``` intact."
    assert row["fields"]["Exception"] == ["Generated files"]
    assert (row["status"], row["strength"]) == ("Review", "recommended")
    section, = row["sections"]
    assert section["title"] == "Counterevidence"
    assert section["blocks"][0] == {"kind": "quote", "text": "## Literal heading"}
    assert section["blocks"][-1]["text"] == "src/generated.py:2-5"
