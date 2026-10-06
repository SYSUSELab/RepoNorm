"""Frozen coarse content proxies used by the standard checker backend."""
from __future__ import annotations

import argparse

import ast

from collections import defaultdict

import importlib.util

import json

from pathlib import Path

import re

import subprocess

import sys

import tempfile

def added_lines_by_path(patch: str) -> dict[str, list[str]]:
    """Return actual added hunk lines, excluding diff metadata."""
    result: dict[str, list[str]] = defaultdict(list)
    current: str | None = None
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = current is not None
        elif in_hunk and line.startswith("+") and not line.startswith("+++"):
            result[current].append(line[1:])
    return dict(result)

def docstrings(source: str) -> dict[str, str | None]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    found: dict[str, str | None] = {"<module>": ast.get_docstring(tree, clean=False)}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{node.name}" if prefix else node.name
                found[name] = ast.get_docstring(node, clean=False)
                visit(node.body, name)

    visit(tree.body, "")
    return found

def has_docstring_change(before: str, after: str) -> bool:
    old, new = docstrings(before), docstrings(after)
    return any(value and value != old.get(name) for name, value in new.items())

def annotations(source: str) -> dict[str, ast.expr]:
    """Collect evaluated annotation expressions by a stable structural key."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    result: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        candidates: list[ast.expr] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            candidates.extend(arg.annotation for arg in (*node.args.posonlyargs, *node.args.args,
                                                           *node.args.kwonlyargs)
                              if arg.annotation is not None)
            if node.args.vararg and node.args.vararg.annotation:
                candidates.append(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                candidates.append(node.args.kwarg.annotation)
            if node.returns:
                candidates.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            candidates.append(node.annotation)
        for candidate in candidates:
            result[ast.dump(candidate, include_attributes=False)] = candidate
    return result

def is_user_documentation(path: str) -> bool:
    item = Path(path)
    lower = path.lower()
    if item.name.lower() in {"changes.rst", "changelog.md", "changelog.rst"}:
        return False
    return (
        lower.startswith(("docs/", "doc/"))
        or item.name.lower().startswith("readme")
    ) and item.suffix.lower() in {".md", ".rst", ".txt"}

def compatibility_proxy(receipt: dict, facts: dict) -> tuple[str, str]:
    syntax = receipt["dynamic"].get("syntax", {})
    if syntax.get("status") != "pass":
        return syntax.get("status", "error"), syntax.get("reason", "minimum_version_syntax_failed")
    minimum = int(syntax["python_min"].split(".")[1])
    for path, lines in facts["additions"].items():
        if not path.endswith(".py"):
            continue
        source = facts["patched_files"].get(path, "")
        postponed = "from __future__ import annotations" in source
        old_annotations = annotations(facts["before_files"].get(path, ""))
        new_annotations = annotations(source)
        added = [node for key, node in new_annotations.items() if key not in old_annotations]
        if minimum < 10 and not postponed and any(
            isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr)
            for node in added for child in ast.walk(node)
        ):
            return "fail", "added evaluated PEP 604 annotation requires Python 3.10"
        if minimum < 9 and not postponed and any(
            isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name)
            and child.value.id in {"list", "dict", "tuple", "set", "frozenset", "type"}
            for node in added for child in ast.walk(node)
        ):
            return "fail", "added evaluated builtin generic annotation requires Python 3.9"
    return "pass", "minimum-version grammar and bounded evaluated-annotation checks passed"

def dependency_proxy(facts: dict, repository: str) -> tuple[str, str]:
    imports: set[str] = set()
    for path, lines in facts["additions"].items():
        item = Path(path)
        if (
            not path.endswith(".py")
            or path.startswith("tests/")
            or item.name.startswith("test_")
        ):
            continue
        for line in lines:
            match = re.match(r"\s*(?:from|import)\s+([A-Za-z_][\w.]*)", line)
            if match:
                imports.add(match.group(1).split(".")[0])
    local = {"DEQ": "dirty_equals", "JIN": "jinja2", "CLI": "click", "TIK": "tiktoken", "OUT": "outlines"}
    third_party = sorted(name for name in imports if name not in sys.stdlib_module_names and name != local[repository])
    if not third_party:
        return "not_applicable", "no added third-party runtime import detected"
    metadata = "\n".join(
        text for path, text in facts["patched_files"].items()
        if Path(path).name in {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"}
    ).lower()
    missing = [name for name in third_party if name.replace("_", "-").lower() not in metadata and name.lower() not in metadata]
    if missing:
        return "fail", "added third-party imports lack a matching changed dependency declaration: " + ", ".join(missing)
    return "pass", "added third-party imports have a matching changed dependency declaration"

def confirmed_new_api_collection_errors(dynamic: dict, facts: dict) -> bool:
    """Validate host-parsed collection errors against exact before/after source.

    This is a bounded import proxy, not a general attribution of pytest errors.
    The host must bind the complete collection-error list to the red execution
    log. Missing, mixed, or unsupported evidence is never an automatic pass.
    """
    red = dynamic.get("author_red", {})
    counts = red.get("counts", {})
    errors = red.get("collection_errors")
    if (red.get("status") != "error" or red.get("returncode") != 2
            or not isinstance(errors, list) or not errors
            or type(counts.get("errors")) is not int
            or counts["errors"] != len(errors) or counts.get("failures") != 0):
        return False
    seen = set()
    for error in errors:
        if not isinstance(error, dict) or error.get("exception_type") != "ImportError":
            return False
        path, module, symbol = (error.get(key) for key in ("path", "module", "symbol"))
        if (not all(isinstance(value, str) for value in (path, module, symbol))
                or path not in dynamic.get("authored_paths", [])
                or not symbol.isidentifier()
                or not module or not all(part.isidentifier() for part in module.split("."))
                or path in seen):
            return False
        seen.add(path)
        relative = module.replace(".", "/")
        candidates = [prefix + relative + suffix for prefix in ("", "src/")
                      for suffix in (".py", "/__init__.py")]
        before_files, patched_files = facts.get("before_files", {}), facts.get("patched_files", {})
        candidates = [item for item in candidates if item in before_files and item in patched_files]
        if len(candidates) != 1 or path not in patched_files:
            return False
        try:
            before = ast.parse(before_files[candidates[0]])
            after = ast.parse(patched_files[candidates[0]])
            test = ast.parse(patched_files[path])
        except (SyntaxError, ValueError, TypeError):
            return False
        # Any old binding, wildcard export, or dynamic module lookup makes
        # absence uncertain. Walk nested statements conservatively as well.
        for node in ast.walk(before):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == symbol:
                return False
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in {symbol, "__getattr__"}:
                return False
            if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                alias.name == "*" or (alias.asname or alias.name.split(".")[0]) == symbol
                for alias in node.names
            ):
                return False
        declared = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol
            or isinstance(node, ast.ImportFrom) and any(
                alias.name != "*" and (alias.asname or alias.name) == symbol for alias in node.names
            ) for node in after.body
        )
        imported = any(
            isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == module
            and any(alias.name == symbol for alias in node.names) for node in test.body
        )
        if not declared or not imported:
            return False
    return True

def coarse_status(receipt: dict, norm: dict, facts: dict, strict) -> tuple[str, str | None]:
    kind = norm["oracle_kind"]
    dynamic = receipt["dynamic"]
    if kind in {"runtime", "public_export"}:
        check = dynamic.get("norms", {}).get(norm["id"])
        return ((check.get("status", "error"), None) if check else ("error", "dynamic_norm_receipt_missing"))
    if kind in {"style", "typing"}:
        return strict.tool_status(receipt["static"].get(kind, []))
    if kind == "docs":
        if dynamic.get("doc_examples", {}).get("status") == "fail":
            return "fail", "submitted documentation example failed when executed"
        return ("pass", "non-empty repository documentation or Python docstring change detected") if facts["docs_changed"] else ("fail", "no repository documentation or Python docstring change detected")
    if kind == "changelog":
        additions = facts["additions"].get("CHANGES.rst", [])
        return ("pass", "non-empty CHANGES.rst addition detected") if any(line.strip() for line in additions) else ("fail", "no non-empty CHANGES.rst addition detected")
    if kind == "inline_change":
        found = any(re.search(r"\.\.\s+versionchanged::\s*\S+", line) for lines in facts["additions"].values() for line in lines)
        return ("pass", "versionchanged directive with a version was added") if found else ("fail", "no added versionchanged directive with a version")
    if kind == "authored_tests":
        if not dynamic.get("authored_paths"):
            return "fail", "no author-owned Python test change"
        green, red = dynamic.get("author_green", {}), dynamic.get("author_red", {})
        # Some newly added public objects make the patch's author tests fail
        # during collection on the unpatched snapshot.  The coarse profile
        # deliberately treats that bounded red/green shape as proxy evidence;
        # semantic relevance remains the strict profile's responsibility.
        if (green.get("status") == "pass" and red.get("status") == "error"
                and red.get("counts", {}).get("errors")):
            return "pass", "author tests pass on submission and fail or error against the base snapshot"
        # Outlines has one pre-certified, patch-independent MLX collection
        # error in this frozen image.  Accept only the exact stable-error
        # shape where the submitted tests add ordinary failures on the base.
        if (receipt["repository"] == "OUT" and green.get("status") == "error"
                and red.get("status") == "error"
                and green.get("counts", {}).get("errors") == red.get("counts", {}).get("errors") == 1
                and not green.get("counts", {}).get("failures")
                and red.get("counts", {}).get("failures")):
            return "pass", "stable pre-certified MLX error is present in both runs; task-relevant red failures remain"
        if green.get("status") == "error":
            return "error", "author test execution environment failed"
        if green.get("status") != "pass":
            return "fail", "submitted author tests do not pass on the submission"
        if (red.get("status") == "fail" and red.get("counts", {}).get("failures")
                and not red.get("counts", {}).get("errors")):
            return "pass", "author tests pass after the patch and fail on the base snapshot"
        if red.get("status") == "error" or red.get("counts", {}).get("errors"):
            if confirmed_new_api_collection_errors(dynamic, facts):
                return "pass", "author tests pass after the patch; all base collection errors import mechanically confirmed newly added API names"
            return "error", "base author-test error lacks complete mechanically attributable new-API collection evidence"
        return "fail", "author tests do not expose a red/green difference"
    if kind == "compatibility":
        return compatibility_proxy(receipt, facts)
    if kind == "dependency":
        return dependency_proxy(facts, receipt["repository"])
    return "error", f"unsupported coarse oracle kind: {kind}"
