"""Prepare and evaluate the frozen RepoNormBench Docker environments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid

from dataset import BENCHMARK, ROOT, manifest, snapshot, task_for_pool, task_pool, tasks, verify
from four_arm_common import atomic_json, sha

CACHE = ROOT / "outputs/runtime"
ENVIRONMENT = CACHE / "environment.json"
SHIMS = Path(__file__).resolve().parent / "shims"
WHEEL = "numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"


def command(args: list[str], timeout: int = 600):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {result.stderr.strip()}")
    return result


def load_environment() -> dict:
    value = json.loads(ENVIRONMENT.read_text())
    if (value["manifest_sha256"] != sha(BENCHMARK / "manifest.json")
            or value["runtime_sha256"] != sha(BENCHMARK / "runtime.json")
            or value["implementation_sha256"] != sha(Path(__file__))):
        raise ValueError("Environment contract changed; run prepare again")
    for name, expected in value["shims"].items():
        if sha(SHIMS / name) != expected:
            raise ValueError(f"Shim changed: {name}")
    for name, expected in value["cache_files"].items():
        if sha(CACHE / name) != expected:
            raise ValueError(f"Runtime cache changed: {name}")
    return value


def container_options(pool: dict, environment: dict | None = None) -> dict:
    value = environment or load_environment()
    task = task_for_pool(pool, pool["features"][0])
    row = value["images"][task["image_id"]]
    if row["base_commit"] != pool["base_commit"] or row["image_id"] != task["image_id"]:
        raise ValueError("Runtime snapshot mismatch")
    env = {"PATH": "/n4c-preflight-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
           "UV_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    run_args = ["--network", "none", "--pull", "never", "--mount",
                f"type=bind,src={SHIMS},dst=/n4c-preflight-bin,readonly"]
    if task["oracle_backend"] == "extended":
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        run_args += ["--tmpfs", "/tmp:rw,nosuid,nodev,size=2g,mode=1777"]
        if pool["repository"] == "dspy_task":
            env["DSPY_CACHEDIR"] = "/tmp/n4c-dspy-cache"
    else:
        env.update(TIKTOKEN_CACHE_DIR="/n4c-runtime-cache/tiktoken", TRANSFORMERS_NO_TF="1", TRANSFORMERS_NO_FLAX="1")
        for source, target in (("TIK-cache", "tiktoken"), ("TIK-wheels", "wheels")):
            run_args += ["--mount", f"type=bind,src={CACHE / source},dst=/n4c-runtime-cache/{target},readonly"]
    return {"image": row["image_id"], "env": env, "run_args": run_args}


def prepare_container(container: str, pool: dict) -> None:
    task = task_for_pool(pool, pool["features"][0])
    runtime = json.loads((BENCHMARK / "runtime.json").read_text())["tasks"][task["id"]]
    script = "set -eu\ncd /workspace/repo\n"
    script += f'test "$(git rev-parse HEAD)" = {pool["base_commit"]}\n'
    script += 'test -z "$(git status --porcelain --untracked-files=no)"\n'
    if runtime["runner_sha256"]:
        script += f"echo '{runtime['runner_sha256']}  /usr/local/bin/runner.sh' | sha256sum -c -\n"
    if pool["repository"] == "openai_tiktoken_task":
        script += f"python -m pip install --no-index --no-deps /n4c-runtime-cache/wheels/{WHEEL} >/dev/null\n"
    script += "if test -f outlines/_version.py; then mkdir -p /tmp/n4c-preflight-runtime/outlines; cp outlines/_version.py /tmp/n4c-preflight-runtime/outlines/_version.py; fi\n"
    command(["docker", "exec", container, "bash", "-c", script])


def prepare(selected: list[dict], registry: str = "") -> dict:
    verify()
    CACHE.mkdir(parents=True, exist_ok=True)
    from sys import path
    path.insert(0, str(ROOT))
    from evaluation.standard.prepare import prepare_assets
    # The frozen standard environments mount these caches for all repositories.
    if any(t["oracle_backend"] == "standard" for t in selected):
        prepare_assets(CACHE)
    runtime = json.loads((BENCHMARK / "runtime.json").read_text())
    for name, expected in runtime["shims"].items():
        if sha(SHIMS / name) != expected:
            raise ValueError(f"Frozen shim changed: {name}")
    prior = json.loads(ENVIRONMENT.read_text()) if ENVIRONMENT.exists() else {}
    rows = prior.get("images", {})
    for task in {t["image_id"]: t for t in selected}.values():
        identifier = task["id"]
        row = snapshot(task["snapshot_id"])
        image = task["image_id"]
        present = subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
        if not present:
            reference = (registry.rstrip("/") + "/" if registry else "") + next(i["image_ref"] for i in row["images"] if i["image_id"] == image)
            subprocess.run(["docker", "pull", reference], check=True)
        actual = command(["docker", "image", "inspect", image, "--format", "{{.Id}}"] ).stdout.strip()
        head = command(["docker", "run", "--rm", "--network", "none", "--entrypoint", "", image,
                        "git", "-C", "/workspace/repo", "rev-parse", "HEAD"]).stdout.strip()
        if actual != image or head != row["commit"]:
            raise ValueError(f"Image identity mismatch: {identifier}")
        rows[image] = {"image_id": image, "base_commit": head}
    result = {"schema": "reponorm-environment-1.0", "manifest_sha256": sha(BENCHMARK / "manifest.json"),
              "runtime_sha256": sha(BENCHMARK / "runtime.json"),
              "implementation_sha256": sha(Path(__file__)), "shims": runtime["shims"], "images": rows,
              "cache_files": {str(p.relative_to(CACHE)): sha(p) for folder in ("TIK-cache", "TIK-wheels", "OUT-wheels")
                              for p in (CACHE / folder).glob("*") if p.is_file()}}
    atomic_json(ENVIRONMENT, result)
    return result


def parse_counts(output: str) -> dict:
    summaries = [line for line in output.splitlines() if re.search(r"\d+ (?:passed|failed|skipped|error)", line)
                 and re.search(r"in [\d.]+s", line)]
    summary = summaries[-1] if summaries else ""
    return {kind: sum(int(n) for n in re.findall(rf"(\d+) {('errors?' if kind == 'errors' else kind)}\b", summary))
            for kind in ("passed", "failed", "skipped", "errors", "xfailed", "xpassed")}


def evaluate(pool: dict, feature: int, patch: str | None, *, timeout: int = 900) -> dict:
    started = time.monotonic()
    task = task_for_pool(pool, feature)
    options = container_options(pool)
    contract = json.loads((BENCHMARK / "runtime.json").read_text())["tasks"][task["id"]]
    container = "reponorm-eval-" + uuid.uuid4().hex
    output, code, timed_out = "", None, False
    tests = BENCHMARK / task["path"] / pool.get("tests_variant", "tests.patch")
    if not tests.resolve().is_relative_to((BENCHMARK / task["path"]).resolve()):
        raise ValueError("Test variant escapes task directory")
    with tempfile.TemporaryDirectory(prefix="patches-", dir=CACHE) as temporary:
        directory = Path(temporary)
        (directory / "tests.patch").write_bytes(tests.read_bytes())
        if patch is not None:
            (directory / "agent.patch").write_text(patch)
        args = ["docker", "run", "-d", "--name", container, "--entrypoint", "", *options["run_args"],
                "--mount", f"type=bind,src={directory},dst=/patches,readonly"]
        if task["oracle_backend"] == "extended":
            args += ["--cpus", "2", "--memory", "6g"]
        for key, value in options["env"].items():
            args += ["-e", f"{key}={value}"]
        try:
            command([*args, options["image"], "sleep", "infinity"])
            prepare_container(container, pool)
            result = subprocess.run(["docker", "exec", container, "bash", "/usr/local/bin/runner.sh", "tests.patch",
                                     *(["agent.patch"] if patch is not None else [])],
                                    capture_output=True, text=True, timeout=timeout)
            output, code = result.stdout + result.stderr, result.returncode
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or b"").decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            timed_out = True
        finally:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)
    counts = parse_counts(output)
    passed = code == 0 and counts["passed"] > 0 and not counts["failed"] and not counts["errors"] and counts["skipped"] in contract["allowed_skips"]
    return {"passed": passed, "exit_code": code, "timed_out": timed_out, "duration_seconds": time.monotonic() - started,
            "network": "none", "pytest": counts, "tests_sha256": sha(tests),
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(), "output": output}


def preflight(selected: list[dict]) -> dict:
    results, failures = {}, []
    contract = json.loads((BENCHMARK / "runtime.json").read_text())["tasks"]
    for task in selected:
        pool = task_pool(task)
        feature = pool["features"][0]
        rows = {"base": evaluate(pool, feature, None),
                "gold": evaluate(pool, feature, (BENCHMARK / task["path"] / "feature.patch").read_text())}
        for role, row in rows.items():
            output_text = row.pop("output")
            (CACHE / f"{task['id']}-{role}.log").write_text(output_text)
            expected = contract[task["id"]]["checks"][role]
            matches = any(row["exit_code"] == ref["exit_code"] and (ref["counts"] is None or all(
                row["pytest"].get(k, 0) == ref["counts"].get(k, ref["counts"].get("error", 0) if k == "errors" else 0)
                for k in row["pytest"])) for ref in expected)
            row["reference_counts_match"] = matches
            if contract[task["id"]].get("preflight_policy") == "base_failure_gold_exact" and role == "base":
                markers = ("cannot connect to the docker daemon", "no such image", "no space left on device",
                           "network is unreachable", "temporary failure in name resolution")
                matches = (row["exit_code"] == expected[0]["exit_code"]
                           and row["pytest"]["failed"] + row["pytest"]["errors"] > 0
                           and row["pytest"]["skipped"] == 0
                           and not any(marker in output_text.lower() for marker in markers))
            if contract[task["id"]].get("preflight_policy") == "erratum_base_fails_gold_passes":
                matches = row["exit_code"] not in (None, 0) if role == "base" else row["passed"]
            if row["timed_out"] or not matches or (role == "gold" and not row["passed"]):
                failures.append({"task": task["id"], "role": role})
        results[task["id"]] = rows
        print(f"{task['id']}: base={rows['base']['exit_code']} gold={rows['gold']['exit_code']}", flush=True)
    result = {"status": "failed" if failures else "passed", "tasks": results, "failures": failures}
    atomic_json(CACHE / "preflight.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "preflight"))
    parser.add_argument("--task", action="append")
    parser.add_argument("--registry", default="", help="Optional Docker Hub mirror prefix")
    args = parser.parse_args()
    selected = tasks(args.task)
    if args.action == "prepare":
        prepare(selected, args.registry)
    elif args.action == "preflight":
        result = preflight(selected)
        print(json.dumps({k: v for k, v in result.items() if k != "tasks"}))
        raise SystemExit(result["status"] != "passed")
    else:
        load_environment()
        for task in selected:
            container_options(task_pool(task))
    print(ENVIRONMENT)


if __name__ == "__main__":
    main()
