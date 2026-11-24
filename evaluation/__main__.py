"""Prepare norm-checker resources or score saved patches from a unified task selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from . import execution, proxies
from .delivery import assess_authored_test_delivery
from .patch_facts import git, index_files

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def read(path: Path):
    return json.loads(path.read_text())


def load_backend(name: str) -> dict:
    return read(HERE / "checks" / name / "norms.json")


def selected_tasks(ids: list[str] | None) -> list[dict]:
    rows = read(ROOT / "benchmark/manifest.json")["tasks"]
    if ids:
        unknown = set(ids) - {row["id"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
        rows = [row for row in rows if row["id"] in ids]
    return rows


def prepare(selected: list[dict], destination: Path, registry: str, base: str) -> None:
    from .standard import prepare as standard
    from .extended import prepare as extended
    for backend in ("standard", "extended"):
        ids = {t["id"] for t in selected if t["oracle_backend"] == backend}
        if not ids:
            continue
        tasks = [t for t in load_backend(backend)["tasks"] if t["id"] in ids]
        folder = destination / backend
        folder.mkdir(parents=True, exist_ok=True)
        record = read(folder / "prepared.json") if (folder / "prepared.json").exists() else {
            "runtime_images": {}, "snapshots": {}, "snapshot_sha256": {}, "static_images": {}, "tool_receipts": {}}
        for task in tasks:
            if "image_ref" not in task:
                snapshot = next(s for s in read(ROOT / "benchmark/manifest.json")["snapshots"] if s["commit"] == task["snapshot"])
                task = dict(task, image_ref=next(i["image_ref"] for i in snapshot["images"] if i["image_id"] == task["image_id"]))
            image, bundle = extended.prepare_snapshot(task, folder, registry)
            key = task["code"] if backend == "standard" else task["id"]
            snapshot_key = task["code"] if backend == "standard" else task["snapshot"]
            record["runtime_images"][key] = image
            record["snapshots"][snapshot_key] = bundle
            record["snapshot_sha256"][snapshot_key] = extended.digest(Path(bundle))
        execution.atomic(folder / "prepared.json", record)
        if backend == "standard":
            record.update(standard.prepare_assets(folder))
            execution.atomic(folder / "prepared.json", record)
            standard.build_tools(folder, base)
            updated = read(folder / "prepared.json")
            record.update(static_images=updated["static_images"], tool_receipts=updated["tool_receipts"])
            execution.atomic(folder / "prepared.json", record)
            standard.verify_resources(record, sorted({t["code"] for t in tasks}), checks=HERE / "checks/standard")
        else:
            for profile in sorted({t["tool_profile"] for t in tasks}):
                try:
                    extended.verify_resources(record, [t for t in tasks if t["tool_profile"] == profile])
                except (KeyError, ValueError, OSError, subprocess.SubprocessError):
                    image, receipt = extended.build_tools(profile, folder, base)
                    record["static_images"][profile] = image
                    record["tool_receipts"][profile] = receipt
                execution.atomic(folder / "prepared.json", record)
            extended.verify_resources(record, tasks)
        print(f"Prepared {backend} resources for {len(tasks)} selected tasks.")


def patch_facts(bundle: Path, snapshot: str, patch: Path, changed: list[str]) -> dict:
    with tempfile.TemporaryDirectory(prefix="reponorm-index-") as temporary:
        directory = Path(temporary)
        repo, index = directory / "repo.git", directory / "index"
        subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "clone", "--bare", "--quiet", str(bundle), str(repo)], check=True)
        git(repo, index, "read-tree", snapshot)
        before = index_files(repo, index, set())
        git(repo, index, "apply", "--cached", "--whitespace=nowarn", str(patch))
        after = index_files(repo, index, set(changed))
    text = patch.read_text()
    additions = proxies.added_lines_by_path(text)
    return {"patch": text, "additions": additions, "changed": changed, "before_files": before, "patched_files": after,
            "docs_changed": any(proxies.is_user_documentation(p) and any(line.strip() for line in lines)
                                for p, lines in additions.items()) or any(
                proxies.has_docstring_change(before.get(p, ""), after.get(p, "")) for p in changed if p.endswith(".py"))}


def standard_verdict(task: dict, data: dict, patch: Path, resources: dict, dynamic: dict, static: dict) -> dict:
    record, strict = execution.execution_verdict(task["id"], execution.sha(patch.read_bytes()), dynamic, static,
                                                  data, HERE / "checks/standard")
    values = {key: dict(value) for key, value in strict["norms"].items()}
    observed = dynamic.get("value") or {}
    if "changed" not in observed:
        for row in values.values():
            if row["kind"] not in {"runtime", "public_export", "style", "typing"}:
                row.update(status="error", reason="dynamic_execution_not_completed")
        return values
    facts = patch_facts(Path(resources["snapshots"][task["code"]]), task["snapshot"], patch, observed["changed"])
    delivery = assess_authored_test_delivery(observed)
    norms = {n["id"]: n for n in data["norms"]}
    for identifier, row in values.items():
        if row["kind"] in {"runtime", "public_export", "style", "typing"}:
            continue
        if row["kind"] == "authored_tests":
            row.update(status=delivery["status"], reason=delivery["reason"])
        else:
            status, reason = proxies.coarse_status({"dynamic": observed, "static": record["static"], "repository": task["code"]}, norms[identifier], facts, None)
            row.update(status=status, reason=reason)
    return values


def score_one(task: dict, condition: str, patch: Path | None, output: Path, resources: dict, *, mock: bool = False) -> dict:
    backend = task["oracle_backend"]
    data = load_backend(backend)
    selected = next(t for t in data["tasks"] if t["id"] == task["id"])
    checks = HERE / "checks" / backend
    adapter = HERE / backend
    output.mkdir(parents=True, exist_ok=False)
    applicable = {link["norm_id"] for link in selected["norms"]}
    result = {"task_id": task["id"], "condition": condition, "profile": "coarse_auto", "executions": {}}
    if mock or patch is None or not patch.read_bytes().strip():
        status = "not_run" if mock else "unavailable"
        result["norms"] = {n["id"]: {"status": status, "kind": n["oracle_kind"], "reason": "mock" if mock else "no_nonempty_submission"}
                           for n in data["norms"] if n["id"] in applicable}
    else:
        target = output / "submission.patch"
        target.write_bytes(patch.read_bytes())
        result["patch_sha256"] = execution.sha(target.read_bytes())
        for mode in ("dynamic", "static"):
            code = selected["code"]
            if backend == "standard" and mode == "static" and code == "TIK":
                result["executions"][mode] = {"status": "not_applicable", "value": {"style": [], "typing": []}}
                continue
            mounts = [(str(checks), "/checks"), (str(adapter), "/adapter"), (str(target), "/submission.patch")]
            env = {}
            if backend == "standard":
                key = code
                if mode == "dynamic":
                    image = resources["runtime_images"][key]
                    if code == "TIK":
                        mounts += [(resources["tiktoken_cache"], "/tiktoken-cache"), (resources["tiktoken_wheels"], "/runtime-wheels")]
                        env.update(TIKTOKEN_CACHE_DIR="/tiktoken-cache", N4C_RUNTIME_WHEELS="/runtime-wheels")
                    elif code == "OUT":
                        mounts += [(resources["outlines_wheels"], "/runtime-wheels")]
                        env.update(N4C_RUNTIME_WHEELS="/runtime-wheels", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0",
                                   JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
                                   XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
                else:
                    image = resources["static_images"][key]
                    mounts += [(resources["snapshots"][key], "/snapshot.bundle")]
            else:
                env["ENGINEERING_77_TASK_ID"] = selected["id"]
                if mode == "dynamic":
                    image = resources["runtime_images"][selected["id"]]
                else:
                    image = resources["static_images"][selected["tool_profile"]]
                    mounts += [(resources["snapshots"][selected["snapshot"]], "/snapshot.bundle"), (str(HERE / "standard"), "/standard-adapter")]
            name = "reponorm-score-" + uuid.uuid4().hex
            prefix = "STATIC_RESULT=" if mode == "static" else "CONTRIBUTION_RESULT=" if backend == "standard" else "ENGINEERING_77_RESULT="
            command = execution.docker_command(image, name, mounts, "/adapter/container_" + mode + ".py", selected["id"], env)
            receipt = execution.execute(command, name, output / (mode + ".log"), 1200 if backend == "standard" and mode == "dynamic" else 900, prefix)
            if receipt["value"] is not None:
                value = receipt["value"]
                # Standard static receipts have no patch/snapshot fields; their command and mounted patch are frozen here.
                for key, expected in (("submission_sha256", result["patch_sha256"]), ("snapshot", selected["snapshot"]), ("task", selected["id"])):
                    if (backend == "extended" or key in value) and value.get(key) != expected:
                        raise ValueError(f"Checker execution binding mismatch: {key}")
            result["executions"][mode] = receipt
        dynamic, static = (result["executions"][key] for key in ("dynamic", "static"))
        if backend == "standard":
            result["norms"] = standard_verdict(selected, data, target, resources, dynamic, static)
        else:
            spec = importlib.util.spec_from_file_location("extended_assess", checks / "assess.py")
            assess = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(assess)
            record = {"task": selected["id"], "state": "agent", "patch_sha256": result["patch_sha256"],
                      "container_exit_code": dynamic["exit_code"], "dynamic": dynamic["value"] or {},
                      "static": static["value"] or {"style": [], "typing": []}}
            result["norms"] = assess.assess(record, "coarse_auto", data, {})["norms"]
            delivery = assess_authored_test_delivery(record["dynamic"])
            for row in result["norms"].values():
                if row["kind"] == "authored_tests":
                    row.update(status=delivery["status"], reason=delivery["reason"])
    execution.atomic(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    parser.add_argument("--task", action="append")
    parser.add_argument("--resources", type=Path, default=ROOT / "outputs/evaluation-resources")
    parser.add_argument("--registry", default="")
    parser.add_argument("--tool-base", default="python:3.10.21-slim")
    parser.add_argument("--model", choices=("deepseek", "qwen", "gemini"))
    parser.add_argument("--condition", choices=("raw", "codewiki", "reponorm"), default="reponorm")
    parser.add_argument("--patch", type=Path, help="Score one new patch; requires one --task")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mock", action="store_true", help="Offline wiring check; every verdict remains not_run")
    args = parser.parse_args()
    selected = selected_tasks(args.task)
    if args.action == "prepare":
        prepare(selected, args.resources.resolve(), args.registry, args.tool_base)
        return
    if args.output is None or (args.patch is not None and len(selected) != 1) or (args.patch is None and not args.model and not args.mock):
        parser.error("score requires --output and either --model or --patch with one task")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipts = {(r["model"], r["condition"], r["task_id"]): r for r in
                (json.loads(line) for line in (ROOT / "data/results/executions.jsonl").read_text().splitlines())}
    resources_by_backend = {}
    if not args.mock:
        from .standard.prepare import verify_resources as verify_standard
        from .extended.prepare import verify_resources as verify_extended
        for backend in {t["oracle_backend"] for t in selected}:
            resources = read(args.resources / backend / "prepared.json")
            ids = {t["id"] for t in selected if t["oracle_backend"] == backend}
            bindings = [t for t in load_backend(backend)["tasks"] if t["id"] in ids]
            if backend == "standard":
                verify_standard(resources, sorted({t["code"] for t in bindings}), checks=HERE / "checks/standard")
            else:
                verify_extended(resources, bindings)
            resources_by_backend[backend] = resources
    results = []
    for task in selected:
        patch = args.patch.resolve() if args.patch else None
        if args.model and patch is None:
            path = receipts[args.model, args.condition, task["id"]]["patch"]
            patch = ROOT / path if path else None
        result = score_one(task, args.condition, patch, output / task["id"], resources_by_backend.get(task["oracle_backend"], {}), mock=args.mock)
        results.append(result)
        print(f"Scored {task['id']}", flush=True)
    with (output / "norms.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "task_id", "norm_id", "kind", "status", "reason"])
        for result in results:
            for identifier, row in result["norms"].items():
                writer.writerow([result["condition"], result["task_id"], identifier, row["kind"], row["status"], row.get("reason")])
    execution.atomic(output / "run.json", {"schema": "reponorm-scoring-run-1.0", "mock": args.mock,
        "profile": "coarse_auto", "tasks": [t["id"] for t in selected], "resources": resources_by_backend,
        "implementation_sha256": {str(p.relative_to(ROOT)): execution.sha(p.read_bytes()) for p in HERE.rglob("*.py")}})
    if any(row["status"] == "error" for result in results for row in result["norms"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
