"""Reproduce the human-review sample and summarize the two reviewers' judgments."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("accepted_only", "accepted_after_review")
VERDICTS = {"correct", "incorrect", "undetermined"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def public_claim(claim):
    result = {key: claim[key] for key in ("claim_kind", "normative_strength", "semantics")}
    for key in ("support", "counterevidence"):
        result[key] = [{k: v for k, v in entry.items()
                        if k in {"path", "line_start", "line_end", "kind", "role"}}
                       for entry in claim.get(key, [])]
    return result


def precision_samples(root):
    samples = {}
    for view in VIEWS:
        path = root / f"data/quality/precision_{view}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (view, row["snapshot_id"], row["claim"]["claim_id"])
            if key in samples:
                raise ValueError(f"Duplicate precision sample: {key}")
            samples[key] = row
    return samples


def select(root=ROOT):
    """Select without consulting either human judgments or model verdicts."""
    protocol = read(root / "data/human_review/protocol.json")
    seed = protocol["selection_seed"]
    manifest = read(root / "benchmark/manifest.json")
    snapshots = {row["id"]: row for row in manifest["snapshots"]}
    groups = defaultdict(list)
    for (view, snapshot, _), row in precision_samples(root).items():
        groups[view, snapshot].append(row["claim"])
    if len(groups) != 28:
        raise ValueError("Expected 28 view/snapshot strata")
    selected, objects = [], {}
    for (view, snapshot_id), claims in sorted(groups.items()):
        snapshot = snapshots[snapshot_id]
        pool = protocol["snapshot_pools"][snapshot_id]
        if pool not in snapshot["upstream_pools"]:
            raise ValueError(f"Unknown pool for snapshot: {snapshot_id}")
        count = protocol["entries_per_view_snapshot"]
        if len(claims) < count:
            raise ValueError(f"Insufficient sample: {view}/{snapshot_id}")
        def rank(claim):
            return digest([seed, view, pool, snapshot["commit"], claim["claim_id"]])
        for claim in sorted(claims, key=rank)[:count]:
            public = public_claim(claim)
            identity = digest([snapshot["commit"], claim["claim_id"], public])
            selected.append(dict(view=view, snapshot_id=snapshot_id, pool=pool,
                                 snapshot_commit=snapshot["commit"], claim_id=claim["claim_id"],
                                 selection_hash=rank(claim), object_identity_sha256=identity))
            objects[identity] = dict(snapshot_id=snapshot_id,
                                     snapshot_commit=snapshot["commit"], claim=public)
    identities = sorted(objects, key=lambda key: digest([seed, "object-order", key]))
    ids = {identity: f"R{index:03d}" for index, identity in enumerate(identities, 1)}
    for row in selected:
        row["review_id"] = ids[row["object_identity_sha256"]]
    blind = [dict(review_id=ids[key], **objects[key]) for key in identities]
    return sorted(selected, key=lambda row: row["review_id"]), blind


def summarize(rows):
    summaries = []
    for view in ("all", *VIEWS):
        subset = [r for r in rows if view == "all" or r["view"] == view]
        agreed = [r for r in subset if r["reviewer_a_verdict"] == r["reviewer_b_verdict"]]
        matched = [r for r in agreed if r["reviewer_a_verdict"] == r["historical_model_verdict"]]
        summaries.append(dict(view=view, entries=len(subset), reviewer_agreement=len(agreed),
                              agreement_percent=100*len(agreed)/len(subset) if subset else None,
                              shared_judgment_matches_model=len(matched),
                              model_agreement_percent=100*len(matched)/len(agreed) if agreed else None))
    return summaries


def validate(root=ROOT):
    selected, blind = select(root)
    recorded = read(root / "data/human_review/selection.json")
    if selected != recorded:
        raise ValueError("Human-review selection differs from the seeded sample")
    with (root / "data/human_review/comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {r["review_id"]: r for r in rows}
    if len(rows) != 140 or len(by_id) != 140 or len(blind) != 140:
        raise ValueError("Expected 140 distinct human-review entries")
    if set(by_id) != {r["review_id"] for r in selected}:
        raise ValueError("Human-review IDs do not match the sample")
    samples = precision_samples(root)
    for item in selected:
        row = by_id[item["review_id"]]
        for key in ("view", "pool", "snapshot_commit", "claim_id"):
            if row[key] != item[key]:
                raise ValueError(f"Human-review binding differs: {item['review_id']}/{key}")
        for field in ("historical_model_verdict", "reviewer_a_verdict", "reviewer_b_verdict"):
            if row[field] not in VERDICTS:
                raise ValueError(f"Unknown verdict: {item['review_id']}/{field}")
        sample = samples[item["view"], item["snapshot_id"], item["claim_id"]]
        if row["historical_model_verdict"] != sample["verdict"]:
            raise ValueError(f"Model verdict differs from precision sample: {item['review_id']}")
    counts = Counter((r["view"], r["snapshot_commit"]) for r in rows)
    if len(counts) != 28 or set(counts.values()) != {5}:
        raise ValueError("Expected five entries per snapshot and view")
    summary = summarize(rows)
    expected = read(root / "data/expected.json")["human_review"]
    for key, value in expected.items():
        if summary[0][key] != value:
            raise ValueError(f"Human-review summary differs: {key}")
    return summary


def prepare(output, root=ROOT):
    selected, blind = select(root)
    output.mkdir(parents=True, exist_ok=False)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write(output / "selection.json", selected)
    write(output / "blind-items.json", blind)
    seed = read(root / "data/human_review/protocol.json")["selection_seed"]
    for reviewer in ("a", "b"):
        ordered = sorted(blind, key=lambda row: digest([seed, reviewer, row["review_id"]]))
        write(output / f"reviewer_{reviewer}/items.json", ordered)
    return dict(selected_entries=len(selected), review_objects=len(blind), output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write sample and blind reviewer inputs to a new directory")
    args = parser.parse_args()
    if args.output:
        print(json.dumps(prepare(args.output), indent=2))
    print(json.dumps(validate(), indent=2))


if __name__ == "__main__":
    main()
