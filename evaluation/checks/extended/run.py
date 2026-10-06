import argparse

import ast

import hashlib

import json

import re

import shutil

import subprocess

import tempfile

import time

import uuid

from pathlib import Path



def command(args, cwd, timeout=180):
    start = time.monotonic()
    try:
        process = subprocess.run(
            [str(arg) for arg in args],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "exit_code": process.returncode,
            "output": process.stdout.decode(errors="replace"),
            "seconds": round(time.monotonic() - start, 3),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "exit_code": None,
            "status": "error",
            "reason": "subprocess_timeout",
            "output": (error.stdout or b"").decode(errors="replace"),
            "seconds": round(time.monotonic() - start, 3),
        }

def basic_style(repo, changed):
    violations = []
    for name in changed:
        path = repo / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            violations.append(f"{name}: no newline at EOF")
        for line_number, line in enumerate(raw.decode(errors="replace").splitlines(), 1):
            if line.rstrip() != line:
                violations.append(f"{name}:{line_number}: trailing whitespace")
            if line.startswith(("<<<<<<< ", ">>>>>>> ")) or line == "=======":
                violations.append(f"{name}:{line_number}: merge marker")
        if path.suffix == ".py":
            try:
                tree = ast.parse(raw.decode())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "breakpoint":
                        violations.append(f"{name}:{node.lineno}: debug statement")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "set_trace":
                        violations.append(f"{name}:{node.lineno}: debug statement")
    return {"exit_code": 1 if violations else 0, "output": "\n".join(violations)}

def static_checks(repo, changed, task):
    python_files = [name for name in changed if name.endswith(".py") and (repo / name).is_file()]
    profile = "jinja-2021" if task["id"].startswith("JIN") else task["tool_profile"]
    tools = TOOLS_BY_PROFILE[profile]
    if task["id"].startswith("JIN"):
        style = [
            command([tools / "black", "--check", "--workers", "1", *python_files], repo),
            command([tools / "flake8", *python_files], repo),
            command(["git", "diff", "--check"], repo),
            basic_style(repo, changed),
        ]
        with tempfile.TemporaryDirectory(dir=LOCAL, prefix="jinja-style-") as temporary:
            copied = []
            for name in python_files:
                destination = Path(temporary) / Path(name).name
                shutil.copyfile(repo / name, destination)
                copied.append(destination)
            reorder = command(
                [
                    tools / "reorder-python-imports",
                    "--application-directories",
                    "src",
                    "--diff-only",
                    *python_files,
                ],
                repo,
            )
            style.append(reorder)
            upgrade_tool = LOCAL / "tools-OUT1371/bin/pyupgrade"
            upgrade = command([upgrade_tool, "--py37-plus", *copied], repo)
            upgrade["exit_code"] = (
                0
                if all(
                    destination.read_bytes() == (repo / name).read_bytes()
                    for destination, name in zip(copied, python_files)
                )
                else 1
            )
            style.append(upgrade)
    elif task["style_profile"] == "black-isort-flake8-pyupgrade":
        style = [
            command([tools / "black", "--check", "--workers", "1", *python_files], repo),
            command([tools / "isort", "--profile", "black", "--check-only", *python_files], repo),
            command([tools / "flake8", *python_files], repo),
            command(["git", "diff", "--check"], repo),
            basic_style(repo, changed),
        ]
        with tempfile.TemporaryDirectory(dir=LOCAL, prefix="pyupgrade-") as temporary:
            copied = []
            for name in python_files:
                destination = Path(temporary) / Path(name).name
                shutil.copyfile(repo / name, destination)
                copied.append(destination)
            upgrade = command([tools / "pyupgrade", "--py37-plus", *copied], repo)
            upgrade["exit_code"] = (
                0
                if all(
                    destination.read_bytes() == (repo / name).read_bytes()
                    for destination, name in zip(copied, python_files)
                )
                else 1
            )
            style.append(upgrade)
    elif task["style_profile"] in {"ruff-basic", "ruff-check-format"}:
        style = [
            command([tools / "ruff", "check", "--no-fix", "--config=pyproject.toml", *python_files], repo),
            command(["git", "diff", "--check"], repo),
            basic_style(repo, changed),
        ]
        if task["style_profile"] == "ruff-check-format":
            style.insert(
                1,
                command([tools / "ruff", "format", "--check", "--config=pyproject.toml", *python_files], repo),
            )
    else:
        raise ValueError(f"unknown style profile: {task['style_profile']}")
    typing = []
    if task.get("typing_profile", "mypy") == "mypy":
        typing_files = [name for name in python_files if name.startswith("src/")]
        commands = [[tools / "mypy", "--no-incremental", "--allow-redefinition", *typing_files]]
        if task["tool_profile"] in {"click-2024", "click-2025"} and not task["id"].startswith("JIN"):
            commands.append([tools / "pyright", "--pythonpath", tools / "python", *typing_files])
        typing = []
        with tempfile.TemporaryDirectory(dir=LOCAL, prefix="typing-base-") as temporary:
            baseline = Path(temporary) / "repo"
            subprocess.run(["git", "clone", "--quiet", "--shared", str(repo), str(baseline)], check=True)
            for args in commands:
                current = command(args, repo)
                base = command(args, baseline)
                current_normalized = current["output"].replace(str(repo), "<repo>")
                base_normalized = base["output"].replace(str(baseline), "<repo>")
                current_normalized = re.sub(r":\d+:\d+", ":<line>:<col>", current_normalized)
                base_normalized = re.sub(r":\d+:\d+", ":<line>:<col>", base_normalized)
                current["baseline_exit_code"] = base["exit_code"]
                if base["exit_code"] not in {None, 0} and current_normalized == base_normalized:
                    current["exit_code"] = 0
                    current["baseline_unchanged"] = True
                typing.append(current)
    versions = command([tools / "python", "-m", "pip", "freeze", "--all"], repo)
    return {"style": style, "typing": typing, "tool_versions": versions}

LOCAL = Path("/tmp/static")
TOOLS_BY_PROFILE = {name: Path("/opt/tools/bin") for name in ("jinja-2021", "outlines-2023", "outlines-2025", "click-2024", "click-2025", "dspy-2025")}
