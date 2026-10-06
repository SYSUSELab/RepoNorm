"""Recompute the reported results without API keys, Docker, or model calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from math import comb, fsum, isclose
from pathlib import Path
from statistics import mean, median

from .human_review import validate as validate_human_review

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = {"deepseek": "DeepSeek", "qwen": "Qwen", "gemini": "Gemini"}
CONDITIONS = ("raw", "codewiki", "reponorm")
CONTRIBUTION = {"authored_tests", "docs", "changelog", "inline_change", "style", "typing"}
SCOPES = ("overall", "six_categories", "remaining_five", "prompt_omitted")
SCOPE_LABELS = {"overall": "Overall", "six_categories": "Contribution",
                "remaining_five": "Behavioral", "prompt_omitted": "Prompt-omitted"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def exact_mcnemar(gain, loss):
    n = gain + loss
    return min(Fraction(1), Fraction(2 * sum(comb(n, k) for k in range(min(gain, loss) + 1)), 2**n)) if n else Fraction(1)


def intervals(base, treatment, seed_key):
    import numpy as np
    from scipy.stats import bootstrap

    seed = hashlib.sha256(f"20260930:{seed_key}".encode()).digest()
    generator = np.random.Generator(np.random.PCG64(int.from_bytes(seed[:16], "big")))
    sample = (np.array([float(x) for x in base]), np.array([float(x) for x in treatment]))
    result = bootstrap(sample, lambda x, y, axis=-1: np.mean(y - x, axis=axis),
                       paired=True, vectorized=True, n_resamples=100000, batch=2000,
                       confidence_level=0.95, method="BCa", alternative="two-sided", rng=generator)
    return [float(x * 100) for x in result.confidence_interval]


def plot_intervals(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True, layout="constrained")
    labels = list(SCOPE_LABELS.values())
    for axis, model in zip(axes, MODELS.values()):
        axis.axvline(0, color="#777777", linewidth=0.8)
        for baseline, offset, color in (("raw", -0.12, "#2463a6"), ("codewiki", 0.12, "#bc601b")):
            selected = [next(r for r in rows if r["model"] == model and r["baseline"] == baseline
                             and r["selection"] == "full" and r["scope"] == scope) for scope in labels]
            y = [i + offset for i in range(len(labels))]
            x = [r["difference_pp"] for r in selected]
            errors = [[r["difference_pp"] - r["ci_low_pp"] for r in selected],
                      [r["ci_high_pp"] - r["difference_pp"] for r in selected]]
            axis.errorbar(x, y, xerr=errors, fmt="o", markersize=4, capsize=3,
                          color=color, label="vs " + ("Raw" if baseline == "raw" else "CodeWiki"))
        axis.set_title(model)
        axis.set_xlabel("RepoNorm NCR difference (pp)")
        axis.grid(axis="x", alpha=0.15)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[-1].legend(frameon=False, fontsize=9)
    figure.savefig(output / "paired_ncr.pdf")
    figure.savefig(output / "paired_ncr.png", dpi=200)
    plt.close(figure)


def reproduce(output: Path, *, bootstrap: bool = True):
    output.mkdir(parents=True, exist_ok=True)
    manifest = read(ROOT / "benchmark/manifest.json")
    tasks = {t["id"]: t for t in manifest["tasks"]}
    annotation = read(ROOT / "benchmark/norms.json")
    links = {(r["task_id"], r["norm_id"]): r for r in annotation["associations"]}
    assert (len(tasks), len(manifest["snapshots"]), len(annotation["norms"]), len(links)) == (121, 14, 160, 1024)
    executions = read_lines(DATA / "results/executions.jsonl")
    execution = {(r["model"], r["condition"], r["task_id"]): r for r in executions}
    assert len(execution) == len(executions) == 1089
    expected = read(DATA / "expected.json")
    revisions = read(DATA / "sensitivity/prompt_revisions.json")
    originally_absent = {(r["task_id"], r["norm_id"]) for r in revisions if r["group"] == "already_absent"}
    removed_tasks = {r["task_id"] for r in revisions if r["group"] == "reminder_removed"}
    excluded_docs = {(r["task_id"], r["norm_id"]) for r in read(DATA / "sensitivity/documentation_exclusions.json")["associations"]}
    assert (len(revisions), len(originally_absent), len(removed_tasks), len(excluded_docs)) == (670, 651, 14, 59)
    selections = [("full", scope) for scope in SCOPES] + [
        ("already_absent", "prompt_omitted"), ("excluding_removal_tasks", "overall"),
        ("excluding_docs", "overall"), ("excluding_docs", "six_categories")]
    ncr, binary, comparisons, ci_records = [], [], [], []
    for model, label in MODELS.items():
        rows = read_csv(DATA / f"results/norm_decisions_{model}.csv")
        assert len(rows) == 3072
        by_condition = {arm: [r for r in rows if r["condition"] == arm] for arm in CONDITIONS}
        for arm, records in by_condition.items():
            assert {(r["task_id"], r["norm_id"]) for r in records} == set(links)
            for row in records:
                assert row["functional_outcome"] == execution[model, arm, row["task_id"]]["outcome"]
        for selection, scope in selections:
            rates = {}
            for arm, records in by_condition.items():
                counts = defaultdict(Counter)
                for row in records:
                    key = row["task_id"], row["norm_id"]
                    if scope == "six_categories" and row["norm_kind"] not in CONTRIBUTION:
                        continue
                    if scope == "remaining_five" and row["norm_kind"] in CONTRIBUTION:
                        continue
                    if scope == "prompt_omitted" and links[key]["prompt_explicitness"] != "omitted":
                        continue
                    if selection == "already_absent" and key not in originally_absent:
                        continue
                    if selection == "excluding_removal_tasks" and row["task_id"] in removed_tasks:
                        continue
                    if selection == "excluding_docs" and key in excluded_docs:
                        continue
                    counts[row["task_id"]][row["status"]] += 1
                values = {task: Fraction(c["pass"], c["pass"] + c["fail"]) for task, c in counts.items() if c["pass"] + c["fail"]}
                rates[arm] = values
                average = sum(values.values(), Fraction()) / len(values)
                original_arm = "norm4code" if arm == "reponorm" else arm
                exp = expected["ncr"][label][selection + "/" + scope][original_arm]
                assert average == Fraction(*exp["task_mean_fraction"]) and len(values) == exp["valid_tasks"]
                ncr.append(dict(model=label, condition=arm, selection=selection, scope=SCOPE_LABELS[scope],
                                tasks=len(values), ncr_percent=float(average * 100)))
            for baseline in ("raw", "codewiki"):
                paired = sorted(set(rates[baseline]) & set(rates["reponorm"]), key=lambda t: tasks[t]["statistical_order"])
                base = [rates[baseline][t] for t in paired]
                treatment = [rates["reponorm"][t] for t in paired]
                delta = sum((b - a for a, b in zip(base, treatment)), Fraction()) / len(paired)
                exp = next(r for r in expected["confidence_intervals"] if (r["model"], r["baseline"], r["selection"], r["metric"]) == (label, baseline, selection, scope))
                assert len(paired) == exp["n"] and delta == Fraction(*exp["delta_fraction"])
                record = dict(model=label, baseline=baseline, selection=selection, scope=SCOPE_LABELS[scope], tasks=len(paired), difference_pp=float(delta * 100))
                if bootstrap:
                    ci = intervals(base, treatment, exp["seed_key"])
                    assert all(isclose(a, b, abs_tol=1e-10, rel_tol=0) for a, b in zip(ci, exp["bca_95_ci_pp"])), (record, ci, exp["bca_95_ci_pp"])
                    record.update(ci_low_pp=ci[0], ci_high_pp=ci[1])
                ci_records.append(record)
        outcomes = {}
        for arm, records in by_condition.items():
            grouped = defaultdict(list)
            for r in records:
                grouped[r["task_id"]].append(r)
            outcomes[arm] = {}
            for task, rs in grouped.items():
                applicable = [r for r in rs if r["status"] != "not_applicable"]
                functional = execution[model, arm, task]["functional_success"]
                outcomes[arm][task] = dict(functional_success=functional,
                    joint_success=functional and bool(applicable) and all(r["status"] == "pass" for r in applicable))
            for metric in ("functional_success", "joint_success"):
                passed = sum(r[metric] for r in outcomes[arm].values())
                original_arm = "norm4code" if arm == "reponorm" else arm
                assert passed == expected["binary_counts"][label][metric][original_arm]
                binary.append(dict(model=label, condition=arm, metric=metric, passed=passed, tasks=121, percent=100 * passed / 121))
        for baseline in ("raw", "codewiki"):
            for metric in ("functional_success", "joint_success"):
                gain = sum(outcomes["reponorm"][t][metric] and not outcomes[baseline][t][metric] for t in tasks)
                loss = sum(outcomes[baseline][t][metric] and not outcomes["reponorm"][t][metric] for t in tasks)
                comparisons.append(dict(model=label, baseline=baseline, metric=metric, gain=gain, loss=loss, p=exact_mcnemar(gain, loss)))
    running = Fraction()
    for rank, r in enumerate(sorted(comparisons, key=lambda r: r["p"])):
        running = max(running, (len(comparisons) - rank) * r["p"])
        r["p_holm"] = min(Fraction(1), running)
        exp = next(e for e in expected["binary_tests"] if (e["model"], e["baseline"], e["metric"]) == (r["model"], r["baseline"], r["metric"]))
        assert r["p"] == Fraction(*exp["p_fraction"]) and r["p_holm"] == Fraction(*exp["p_holm_12_fraction"])
        r["p"], r["p_holm"] = float(r["p"]), float(r["p_holm"])
    quality = []
    for view in ("accepted_only", "accepted_after_review"):
        precision = Counter(r["verdict"] for r in read_lines(DATA / f"quality/precision_{view}.jsonl"))
        coverage = Counter(r["verdict"] for r in read_lines(DATA / f"quality/coverage_{view}.jsonl"))
        expected_counts = (258, 300, 35, 63) if view == "accepted_only" else (451, 600, 84, 125)
        assert (precision["correct"], sum(precision.values()), coverage["exact"], coverage["exact"] + coverage["usable_partial"]) == expected_counts
        quality.append(dict(view=view,correct=precision["correct"],sample=sum(precision.values()),precision=100*precision["correct"]/sum(precision.values()),fully_covered=coverage["exact"],full_or_partial=coverage["exact"]+coverage["usable_partial"],reference_norms=sum(coverage.values())))
    human_review = validate_human_review(ROOT)
    calibration = read_csv(DATA / "calibration/decisions.csv")
    matrix = Counter((r["expected_label"], r["status"]) for r in calibration)
    assert matrix == {("compliant", "pass"): 1010, ("violating", "fail"): 302, ("violating", "pass"): 39}
    assert len({r["case_id"] for r in calibration}) == 455
    norm_review = read_lines(DATA / "reference_audit/norms.jsonl")
    association_review = read_lines(DATA / "reference_audit/associations.jsonl")
    assert len(norm_review) == 42 and Counter(r["evidence_support"] for r in norm_review) == {"full": 41, "partial": 1}
    assert len(association_review) == 84 and Counter(r["prompt_explicitness"] for r in association_review) == {"explicit": 11, "partial": 20, "omitted": 53}
    construction = read_csv(DATA / "benchmark/test_changes.csv")
    assert len(construction) == 121
    changes = {k:sum(int(r[k]) for r in construction) for k in ("added_identities", "removed_identities", "retained_ast_modified")}
    assert list(changes.values()) == [501, 503, 78]
    acquisition = read(DATA / "cost/acquisition.json")
    seconds = [r["discover_seconds"] + r["build_seconds"] for r in acquisition]
    assert len(seconds) == 14 and round(mean(seconds)/60, 2) == 9.18 and round(median(seconds)/60, 2) == 8.86
    cost = [dict(stage="acquisition",model="shared",executions=14,
        input_tokens=sum(r["input_tokens"] for r in acquisition),output_tokens=sum(r["output_tokens"] for r in acquisition),hours=fsum(seconds)/3600)]
    for model in (*MODELS, "all_models"):
        for arm in CONDITIONS:
            rs = [r for r in executions if r["condition"] == arm and (model=="all_models" or r["model"]==model)]
            cost.append(dict(stage=arm,model=model,executions=len(rs),input_tokens=sum(r["tokens"]["input_tokens"]["value"] for r in rs),output_tokens=sum(r["tokens"]["output_tokens"]["value"] for r in rs),hours=fsum(r["agent_execution"]["elapsed_seconds"] for r in rs)/3600))
    expected_totals = {"acquisition": (134.52, 7.67, 2.14), "raw": (346.45, 10.10, 53.56),
                       "codewiki": (391.73, 10.89, 63.72), "reponorm": (463.83, 12.70, 67.12)}
    for row in cost:
        if row["model"] in {"shared", "all_models"}:
            actual = (round(row["input_tokens"]/1e6, 2), round(row["output_tokens"]/1e6, 2), round(row["hours"], 2))
            assert actual == expected_totals[row["stage"]], (row["stage"], actual)
    if bootstrap:
        plot_intervals(ci_records, output)
    for table, rows in [("ncr",ncr),("paired_ncr",ci_records),("functional",binary),("paired_binary",comparisons),("quality",quality),("human_review",human_review),("cost",cost)]:
        write_csv(output/(table+".csv"),rows)
    report = dict(tasks=121,snapshots=14,reference_norms=160,associations=1024,executions=1089,
        descriptive_ncr_cells=len(ncr),paired_ncr_comparisons=len(ci_records),bootstrap_recomputed=bootstrap,
        binary_comparisons=len(comparisons),calibration_cases=455,calibration_checks=1351,
        reference_audit_norms=42,reference_audit_associations=84,test_changes=changes,
        human_review_entries=human_review[0]["entries"],
        human_reviewer_agreement=human_review[0]["reviewer_agreement"],
        human_shared_judgment_matches_model=human_review[0]["shared_judgment_matches_model"],
        acquisition_mean_minutes=mean(seconds)/60,acquisition_median_minutes=median(seconds)/60)
    (output/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    print(f"Results written to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/reproduction")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Check point estimates without the 100,000-resample confidence intervals")
    args = parser.parse_args()
    reproduce(args.output, bootstrap=not args.skip_bootstrap)


if __name__ == "__main__":
    main()
