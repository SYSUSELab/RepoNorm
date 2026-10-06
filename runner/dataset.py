"""Unified task and repository snapshot registry for RepoNormBench."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"
REPOSITORIES = ROOT / "outputs/repositories"


def manifest() -> dict:
    return json.loads((BENCHMARK / "manifest.json").read_text())


def tasks(ids: list[str] | None = None) -> list[dict]:
    rows = manifest()["tasks"]
    if ids:
        unknown = set(ids) - {row["id"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
        rows = [row for row in rows if row["id"] in ids]
    return rows


def snapshot(snapshot_id: str) -> dict:
    return next(row for row in manifest()["snapshots"] if row["id"] == snapshot_id)


def task_pool(task: dict) -> dict:
    repository, number, feature = task["upstream_task_path"].split("/")
    return {"repository": repository, "task_id": int(number.removeprefix("task")),
            "base_commit": task["snapshot"], "image": task["image_id"],
            "snapshot_id": task["snapshot_id"], "oracle_backend": task["oracle_backend"],
            "features": [int(feature.removeprefix("feature"))]}


def task_for_pool(pool: dict, feature: int) -> dict:
    path = f"{pool['repository']}/task{pool['task_id']}/feature{feature}"
    return next(row for row in tasks() if row["upstream_task_path"] == path)


def verify() -> dict:
    data = manifest()
    assert len(data["tasks"]) == len({t["id"] for t in data["tasks"]}) == 121
    assert len(data["snapshots"]) == 14
    for task in data["tasks"]:
        assert task["snapshot"] == snapshot(task["snapshot_id"])["commit"]
        folder = BENCHMARK / task["path"]
        assert folder.resolve().is_relative_to(BENCHMARK.resolve())
        for name, digest in task["artifacts"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Artifact checksum mismatch: {task['id']}/{name}")
    return data


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", "-C", str(path), *args], text=True).strip()


def clone(snapshot_id: str) -> Path:
    row = snapshot(snapshot_id)
    destination = REPOSITORIES / snapshot_id
    if destination.is_symlink():
        raise ValueError("Snapshot checkout cannot be a symlink")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "clone", "--no-checkout",
                        row["repository_url"], str(destination)], check=True)
        git(destination, "checkout", "--detach", row["commit"])
    if git(destination, "rev-parse", "HEAD") != row["commit"] or git(destination, "status", "--porcelain"):
        raise ValueError(f"Existing checkout is not the clean selected snapshot: {destination}")
    if git(destination, "remote", "get-url", "origin").removesuffix(".git") != row["repository_url"].removesuffix(".git"):
        raise ValueError("Snapshot origin mismatch")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "verify", "clone"))
    parser.add_argument("--task", action="append", help="Stable task ID; repeat for a subset")
    parser.add_argument("--snapshot", action="append", help="Snapshot ID; repeat for a subset")
    args = parser.parse_args()
    rows = tasks(args.task)
    selected = args.snapshot or sorted({row["snapshot_id"] for row in rows})
    unknown = set(selected) - {row["id"] for row in manifest()["snapshots"]}
    if unknown:
        parser.error(f"Unknown snapshots: {sorted(unknown)}")
    if args.action == "verify":
        verify()
        print("Verified 121 tasks and 14 snapshots.")
    elif args.action == "clone":
        for identifier in selected:
            print(clone(identifier))
    else:
        for row in rows:
            if row["snapshot_id"] in selected:
                print(f"{row['id']}\t{row['snapshot_id']}\t{row['path']}")


if __name__ == "__main__":
    main()
