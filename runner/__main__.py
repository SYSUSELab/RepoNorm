"""Plan or run new coding jobs using the released experiment settings."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ROOT, REPOSITORIES, task_pool, tasks, verify
from four_arm_common import atomic_json, digest, repository_identity, sha, utc
from task_knowledge import CODEWIKI_DESTINATION, DELIVERY, SCHEMA, verify_staged


CONDITIONS = {"raw": "raw", "codewiki": "codewiki", "reponorm": "accepted_only"}


def run_worker(path: Path) -> dict:
    with (path.parent / "worker.log").open("w") as log:
        process = subprocess.run([sys.executable, str(Path(__file__).with_name("four_arm_worker.py")), str(path)],
                                 stdout=log, stderr=subprocess.STDOUT)
    receipt = path.parent / "result.json"
    result = json.loads(receipt.read_text()) if receipt.exists() else {}
    outcome = result.get("outcome", "missing_result")
    return {"specification": str(path), "worker_exit_code": process.returncode, "outcome": outcome,
            "execution_completed": process.returncode == 0 and outcome in {"resolved", "functional_failure", "budget_exhausted"}}


def stage_knowledge(condition: str, task: dict, destination: Path) -> dict:
    arm = CONDITIONS[condition]
    if condition == "raw":
        return {"arm": arm, "snapshot_commit": task["snapshot"]}
    root = ROOT / "data/knowledge"
    packages = json.loads((root / "manifest.json").read_text())["packages"]
    view = "accepted_only" if condition == "reponorm" else "documentation"
    package = next(p for p in packages if p["method"] == condition and p["view"] == view
                   and p["snapshot_id"] == task["snapshot_id"])
    if package["snapshot_commit"] != task["snapshot"]:
        raise ValueError("Knowledge snapshot mismatch")
    archive = root / package["archive"]
    if sha(archive) != package["sha256"]:
        raise ValueError("Knowledge archive changed")
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != set(package["files"]):
            raise ValueError("Knowledge archive file inventory mismatch")
        for name, expected in package["files"].items():
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()) or not name.endswith(".md"):
                raise ValueError("Invalid knowledge path")
            content = source.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Knowledge file changed: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    receipt = {"arm": arm, "schema_version": SCHEMA, "delivery_version": DELIVERY,
               "snapshot_commit": task["snapshot"], "knowledge_directory": str(destination.resolve()),
               "knowledge_tree_sha256": digest(package["files"]),
               "container_destination": CODEWIKI_DESTINATION if condition == "codewiki" else "/knowledge",
               "host_receipt": {"original_markdown_sha256": package["files"]}}
    verify_staged(receipt)
    return receipt


def plan(model: str, conditions: list[str], selected: list[dict], output: Path, *, live: bool = False,
         max_calls: int | None = None, workers: int = 1) -> list[Path]:
    verify()
    executions = [json.loads(line) for line in (ROOT / "data/results/executions.jsonl").read_text().splitlines()]
    configurations = json.loads((ROOT / "data/results/configurations.json").read_text())
    by_key = {(r["model"], r["condition"], r["task_id"]): r for r in executions}
    output.mkdir(parents=True, exist_ok=False)
    run_identity = digest({"path": str(output.resolve()), "created": utc()})
    specs = []
    for condition in conditions:
        for task in selected:
            previous = by_key[model, condition, task["id"]]
            config = dict(configurations[previous["configuration"]])
            overrides = {}
            if max_calls is not None:
                config["max_logical_llm_calls"] = max_calls
                overrides["max_logical_llm_calls"] = max_calls
            config["global_concurrency"] = min(workers, 20)
            directory = output / condition / task["id"]
            directory.mkdir(parents=True)
            handoff = stage_knowledge(condition, task, directory / "knowledge")
            pool = task_pool(task)
            if previous["test_variant"] != "default":
                pool["tests_variant"] = "variants/message-literal-tests.patch"
            repository = REPOSITORIES / task["snapshot_id"]
            identity = repository_identity(repository) if repository.exists() else None
            if live and identity is None:
                raise ValueError(f"Missing snapshot checkout; run python runner/dataset.py clone --snapshot {task['snapshot_id']}")
            if identity and identity["head"] != task["snapshot"]:
                raise ValueError("Host repository identity mismatch")
            spec = {"schema": "reponorm-coding-spec-1.0", "task_key": f"{condition}-{task['id']}",
                    "task_identifier": task["id"], "model": model, "arm": CONDITIONS[condition],
                    "feature": pool["features"][0], "pool": pool, "config": config,
                    "source_configuration": previous["configuration"], "overrides": overrides,
                    "run_identity": run_identity, "handoff": handoff,
                    "repository_identity": identity, "queued_at": utc(),
                    "queued_monotonic": time.monotonic()}
            atomic_json(directory / "spec.json", spec)
            specs.append(directory / "spec.json")
    atomic_json(output / "run.json", {"schema": "reponorm-coding-run-1.0", "identity": run_identity,
                "model": model, "conditions": conditions, "tasks": [t["id"] for t in selected],
                "workers": workers, "live": live, "specifications": [str(p.relative_to(output)) for p in specs]})
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"), help="plan is offline; run makes paid model calls")
    parser.add_argument("--model", choices=("deepseek", "qwen", "gemini"), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), action="append")
    parser.add_argument("--task", action="append", help="Task ID; repeat for a subset; default: all 121")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-calls", type=int, help="Override the recorded budget for a new pilot")
    args = parser.parse_args()
    if not 1 <= args.workers <= 20 or (args.max_calls is not None and args.max_calls < 1):
        parser.error("workers must be 1..20 and max-calls must be positive")
    selected = tasks(args.task)
    conditions = list(dict.fromkeys(args.condition or CONDITIONS))
    specs = plan(args.model, conditions, selected, args.output.resolve(), live=args.action == "run",
                 max_calls=args.max_calls, workers=args.workers)
    if args.action == "run":
        from environment import container_options
        for task in selected:
            container_options(task_pool(task))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            statuses = list(executor.map(run_worker, specs))
        atomic_json(args.output / "summary.json", statuses)
        if any(not row["execution_completed"] for row in statuses):
            raise SystemExit("Some workers failed; inspect their result.json and worker.log files.")
    print(json.dumps({"action": args.action, "jobs": len(specs), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
