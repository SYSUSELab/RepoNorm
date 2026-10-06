"""Validate release integrity without third-party Python dependencies or network access."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reproduction.human_review import validate as validate_human_review


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local(path):
    result = ROOT / path
    require(result.resolve().is_relative_to(ROOT), f"Path escapes release: {path}")
    require(result.is_file() and not result.is_symlink(), f"Missing or linked file: {path}")
    return result


def validate():
    manifest = read("benchmark/manifest.json")
    tasks = {row["id"]: row for row in manifest["tasks"]}
    snapshots = {row["id"]: row for row in manifest["snapshots"]}
    require(len(tasks) == len(manifest["tasks"]) == 121 and len(snapshots) == 14, "Dataset cardinality")
    require({r["statistical_order"] for r in tasks.values()} == set(range(121)), "Statistical order")
    for row in tasks.values():
        snapshot = snapshots[row["snapshot_id"]]
        require(row["snapshot"] == snapshot["commit"], f"Snapshot mismatch: {row['id']}")
        require(row["image_id"] in {i["image_id"] for i in snapshot["images"]}, "Image binding")
        for name, expected in row["artifacts"].items():
            require(digest(local(f"benchmark/{row['path']}/{name}")) == expected, f"Task hash: {row['id']}/{name}")
    runtime = read("benchmark/runtime.json")
    require(set(runtime["tasks"]) == set(tasks), "Runtime task coverage")
    for name, expected in runtime["shims"].items():
        require(digest(local("runner/shims/" + name)) == expected, "Frozen shim changed")
    for identifier, row in runtime["tasks"].items():
        require(bool(row["allowed_skips"]), f"Missing allowed skip count: {identifier}")
        require(all(row["checks"].values()) or row.get("preflight_policy") == "erratum_base_fails_gold_passes",
                f"Missing preflight reference: {identifier}")
    annotation = read("benchmark/norms.json")
    norms = {r["id"]: r for r in annotation["norms"]}
    associations = {(r["task_id"], r["norm_id"]): r for r in annotation["associations"]}
    require(len(norms) == 160 and len(associations) == len(annotation["associations"]) == 1024, "Annotation cardinality")
    require(len({n["oracle_kind"] for n in norms.values()}) == 11, "Norm category count")
    for (task, norm), row in associations.items():
        require(task in tasks and norm in norms and row["applicable"], "Invalid applicability link")
        require(tasks[task]["snapshot_id"] == norms[norm]["snapshot_id"], "Norm snapshot mismatch")
        require(row["applicability_condition_ref"] == norm, "Applicability condition binding")
    for backend in ("standard", "extended"):
        oracle = read(f"evaluation/checks/{backend}/norms.json")
        oracle_links = {(t["id"], n["norm_id"]) for t in oracle["tasks"] for n in t["norms"]}
        require(oracle_links == {k for k in associations if tasks[k[0]]["oracle_backend"] == backend}, "Oracle applicability mismatch")
    configurations = read("data/results/configurations.json")
    executions = lines("data/results/executions.jsonl")
    keys = {(r["model"], r["condition"], r["task_id"]) for r in executions}
    expected_keys = {(m, c, t) for m in ("deepseek", "qwen", "gemini") for c in ("raw", "codewiki", "reponorm") for t in tasks}
    require(keys == expected_keys and len(executions) == 1089, "Execution coverage")
    missing = 0
    for row in executions:
        require(row["configuration"] in configurations, "Missing model configuration")
        require(row["snapshot_id"] == tasks[row["task_id"]]["snapshot_id"], "Execution snapshot mismatch")
        if row["patch"]:
            require(digest(local(row["patch"])) == row["source_patch_sha256"], "Submission hash mismatch")
        else:
            missing += 1
        require(row["test_variant"] == "default" or (row["task_id"] == "JIN1559-F07" and row["model"] in {"deepseek", "qwen"}), "Unexpected test variant")
    for model in ("deepseek", "qwen", "gemini"):
        with local(f"data/results/norm_decisions_{model}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 3072, "Decision count")
        require({(r["condition"], r["task_id"], r["norm_id"]) for r in rows} ==
                {(c, t, n) for c in ("raw", "codewiki", "reponorm") for t, n in associations}, "Decision coverage")
    packages = read("data/knowledge/manifest.json")["packages"]
    views = (("reponorm", "accepted_only"), ("reponorm", "accepted_after_review"),
             ("codewiki", "documentation"))
    require(len(packages) == 42 and {(p["method"], p["view"], p["snapshot_id"]) for p in packages} ==
            {(m, v, s) for m, v in views for s in snapshots}, "Knowledge coverage")
    by_view = {(p["view"], p["snapshot_id"]): p for p in packages}
    for view, count in (("accepted_only", 1990), ("accepted_after_review", 6277)):
        require(sum(p["readable_entries"] for p in packages if p["view"] == view) == count,
                f"Knowledge entry count: {view}")
    exemptions = read("data/provenance/unicode_fixtures.json")
    claim_sets = {}
    for package in packages:
        path = local("data/knowledge/" + package["archive"])
        require(digest(path) == package["sha256"], "Archive hash mismatch")
        require(package["snapshot_commit"] == snapshots[package["snapshot_id"]]["commit"], "Knowledge snapshot mismatch")
        with zipfile.ZipFile(path) as archive:
            require(len(archive.namelist()) == len(package["files"]) and set(archive.namelist()) == set(package["files"]), "Archive member inventory")
            if package["method"] == "reponorm":
                claims = {PurePosixPath(name).stem for name in archive.namelist() if name.startswith("claims/")}
                claim_sets[package["view"], package["snapshot_id"]] = claims
                require(len(claims) == package["readable_entries"], "Readable claim count")
                if package["view"] == "accepted_after_review":
                    require(set(package["claim_artifact_digests"]) == claims, "Reviewed claim identity inventory")
            for name, expected in package["files"].items():
                require(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts and name.endswith(".md"), "Invalid archive path")
                raw = archive.read(name)
                require(hashlib.sha256(raw).hexdigest() == expected, "Archive member hash")
                if re.search(r"[\u3400-\u9fff]", raw.decode()):
                    fixture = f"data/knowledge/{package['archive']}!{name}"
                    require(exemptions.get(fixture) == expected, "Knowledge prose must be in English")
    for snapshot in snapshots:
        require(claim_sets["accepted_only", snapshot] <= claim_sets["accepted_after_review", snapshot],
                "Reviewed view omits an accepted claim")
    patches = read("data/calibration/patch_inventory.json")
    require(len(patches) == 455, "Calibration patch count")
    for row in patches:
        require(row["status"] == "included" and digest(local("data/calibration/" + row["path"])) == row["sha256"], "Calibration patch hash")
    for view, count in (("accepted_only", 300), ("accepted_after_review", 600)):
        precision = lines(f"data/quality/precision_{view}.jsonl")
        coverage = lines(f"data/quality/coverage_{view}.jsonl")
        require(len(precision) == count and len(coverage) == 160, "Assessment count")
        require(len({(r["snapshot_id"], r["claim"]["claim_id"]) for r in precision}) == count, "Duplicate precision sample")
        require(all(r["snapshot_id"] in snapshots for r in precision), "Precision snapshot binding")
        for row in precision:
            package = by_view[view, row["snapshot_id"]]
            claim = row["claim"]
            require(claim["claim_id"] in claim_sets[view, row["snapshot_id"]], "Sampled claim missing from knowledge")
            if view == "accepted_after_review":
                require(package["claim_artifact_digests"][claim["claim_id"]] == claim["artifact_digest"],
                        "Reviewed sample identity differs from knowledge")
        for row in coverage:
            require(set(row["matched_claim_ids"]) <= claim_sets[view, row["snapshot_id"]],
                    "Coverage match missing from knowledge")
        for row in read(f"data/quality/protocol_{view}.json")["packages"]:
            package = by_view[view, row["snapshot_id"]]
            require(row["population"] == package["readable_entries"], "Quality population differs from knowledge")
            if view == "accepted_after_review":
                require(all(row[key] == package[key] for key in ("canonical_claims_digest", "consumer_manifest_digest")),
                        "Quality assessment package binding")
    require(len(lines("data/reference_audit/norms.jsonl")) == 42 and
            len(lines("data/reference_audit/associations.jsonl")) == 84, "Reference audit count")
    reference_records = {r["review_id"]: r for name in ("norms", "associations")
                         for r in lines(f"data/reference_audit/{name}.jsonl")}
    with local("data/reference_audit/human_review.csv").open(newline="") as stream:
        confirmations = list(csv.DictReader(stream))
    require(len(confirmations) == 126 and {r["review_id"] for r in confirmations} == set(reference_records),
            "Reference human-review coverage")
    require(len({r["human_reviewer"] for r in confirmations}) == 1, "Reference reviewer count")
    for row in confirmations:
        reference = reference_records[row["review_id"]]
        require(bool(row["human_reviewer"] and row["human_date"]), "Missing author confirmation identity")
        for field in ("norm_id", "snapshot_id", "task_id"):
            require(row[field] == reference.get(field, ""), "Reference confirmation binding")
        for field in ("validity", "evidence_support", "applicability", "prompt_explicitness"):
            require(row["human_" + field] == reference.get(field, ""), "Reference confirmation judgment")
    human_review = validate_human_review(ROOT)
    checksums = read("data/provenance/checksums.json")
    observed = {str(p.relative_to(ROOT)) for folder in ("data", "benchmark") for p in (ROOT / folder).rglob("*")
                if p.is_file() and p.name != "checksums.json"}
    require(observed == set(checksums), "Export checksum inventory differs")
    for name, expected in checksums.items():
        require(digest(local(name)) == expected, f"Export hash mismatch: {name}")
    for name, expected in exemptions.items():
        if "!" in name:
            archive_name, member = name.split("!", 1)
            with zipfile.ZipFile(local(archive_name)) as archive:
                observed_digest = hashlib.sha256(archive.read(member)).hexdigest()
        else:
            observed_digest = digest(local(name))
        require(observed_digest == expected, f"Unicode fixture changed: {name}")
    for folder in ("benchmark", "data", "evaluation", "repo_constraint_memory_v3", "runner", "CodeWiki", "CooperBench", "docs", "skills"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or any(part in {"__pycache__", ".pytest_cache"} or part.endswith(".egg-info") for part in path.parts):
                continue
            if path.suffix in {".py", ".md", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".patch", ".sh", ".toml"}:
                if re.search(r"[\u3400-\u9fff]", path.read_text(encoding="utf-8")):
                    require(str(path.relative_to(ROOT)) in exemptions, f"Non-English prose: {path.relative_to(ROOT)}")
    report = dict(status="passed", tasks=len(tasks), snapshots=len(snapshots), norms=len(norms),
                  associations=len(associations), executions=len(executions), unavailable_patches=missing,
                  human_review_entries=human_review[0]["entries"],
                  reference_author_confirmations=len(confirmations),
                  accepted_only_entries=1990, accepted_after_review_entries=6277,
                  knowledge_archives=len(packages), calibration_patches=len(patches), checked_exports=len(checksums))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    validate()
