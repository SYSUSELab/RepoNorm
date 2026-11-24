"""Check sample independence, reviewer blinding, and judgment identity bindings."""
import csv
import shutil

import pytest

from reproduction import human_review


def test_sample_does_not_depend_on_model_verdicts(monkeypatch):
    expected, _ = human_review.select()
    samples = human_review.precision_samples(human_review.ROOT)
    for row in samples.values():
        row["verdict"] = "not available during sampling"
    monkeypatch.setattr(human_review, "precision_samples", lambda root: samples)
    assert human_review.select()[0] == expected


def test_review_inputs_omit_view_labels_and_judgments(tmp_path):
    output = tmp_path / "review"
    human_review.prepare(output)
    by_reviewer = [human_review.read(output / f"reviewer_{who}/items.json") for who in ("a", "b")]
    assert {r["review_id"] for r in by_reviewer[0]} == {r["review_id"] for r in by_reviewer[1]}
    assert [r["review_id"] for r in by_reviewer[0]] != [r["review_id"] for r in by_reviewer[1]]
    for rows in by_reviewer:
        assert len(rows) == 140
        for row in rows:
            assert set(row) == {"review_id", "snapshot_id", "snapshot_commit", "claim"}
            assert set(row["claim"]) == {"claim_kind", "normative_strength", "semantics", "support", "counterevidence"}
    with pytest.raises(FileExistsError):
        human_review.prepare(output)


@pytest.mark.parametrize("field,value", [
    ("review_id", "R000"),
    ("claim_id", "CLAIM-wrong"),
    ("snapshot_commit", "000000000000"),
    ("historical_model_verdict", "undetermined"),
    ("reviewer_a_verdict", "unknown"),
])
def test_validation_rejects_corrupted_review_rows(tmp_path, field, value):
    for name in ("benchmark/manifest.json", "data/expected.json",
                 "data/quality/precision_accepted_only.jsonl",
                 "data/quality/precision_accepted_after_review.jsonl",
                 "data/human_review/protocol.json", "data/human_review/selection.json",
                 "data/human_review/comparison.csv"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(human_review.ROOT / name, target)
    path = tmp_path / "data/human_review/comparison.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0][field] = value
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        human_review.validate(tmp_path)


def test_released_comparison_matches_reported_counts():
    total = human_review.validate()[0]
    assert (total["entries"], total["reviewer_agreement"],
            total["shared_judgment_matches_model"]) == (140, 126, 114)
