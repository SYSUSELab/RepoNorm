"""Saved-patch checks: no reference patch, review cards, or hidden tests."""

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path("/checks")
TASK = Path("/task")
REPO = Path("/workspace/repo")
BASE = Path("/tmp/unpatched")


def check(name, paths, selector=None, isolated=False, timeout=180):
    xml = Path("/tmp") / f"engineering-77-{name}.xml"
    command = [sys.executable, "-m", "pytest", *paths, "-q", f"--junitxml={xml}"]
    if isolated:
        command += ["--confcutdir=/checks/tests", "-c", "/dev/null"]
    if selector:
        command += ["-k", selector]
    start = time.monotonic()
    try:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        print((error.stdout or b"").decode(errors="replace"))
        return {
            "status": "error",
            "reason": "subprocess_timeout",
            "exit_code": 2,
            "seconds": round(time.monotonic() - start, 3),
        }
    print(process.stdout.decode(errors="replace"))
    cases = list(ET.parse(xml).iter("testcase")) if xml.exists() else []
    counts = {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }
    if process.returncode == 0 and counts["tests"] > counts["skipped"]:
        status, exit_code = "pass", 0
    elif process.returncode == 1 and counts["failures"] and not counts["errors"]:
        status, exit_code = "fail", 1
    else:
        status, exit_code = "error", 2
    if isolated and counts["skipped"]:
        status, exit_code = "error", 2
    return {
        "status": status,
        "exit_code": exit_code,
        "pytest_exit_code": process.returncode,
        "counts": counts,
        "seconds": round(time.monotonic() - start, 3),
        "command": command,
    }


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def public_api_docstrings(source):
    """Return module-function and class-method docstrings for public APIs."""
    tree = ast.parse(source)
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            result[node.name] = ast.get_docstring(node)
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            result[node.name] = ast.get_docstring(node)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                    result[f"{node.name}.{child.name}"] = ast.get_docstring(child)
    return result


def minimum_python_annotation_error(tree, minimum_python):
    """Return the first annotation unsupported by the configured minimum Python."""
    minimum = tuple(map(int, minimum_python.split(".")[:2]))
    if minimum >= (3, 11):
        return None
    future_annotations = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )
    bound_names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bound_names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if not (isinstance(node, ast.ImportFrom) and node.module == "builtins"):
                    bound_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound_names.update(target.id for target in targets if isinstance(target, ast.Name))
    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations.extend(arg.annotation for arg in node.args.args if arg.annotation)
            annotations.extend(arg.annotation for arg in node.args.kwonlyargs if arg.annotation)
            if node.args.vararg and node.args.vararg.annotation:
                annotations.append(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                annotations.append(node.args.kwarg.annotation)
            if node.returns:
                annotations.append(node.returns)
    for annotation in annotations:
        if minimum < (3, 10) and not future_annotations and any(
            isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr)
            for child in ast.walk(annotation)
        ):
            return f"line {annotation.lineno}: PEP 604 union requires Python 3.10"
        if not future_annotations and any(
            isinstance(child, ast.Name)
            and child.id in {"ExceptionGroup", "BaseExceptionGroup"} - bound_names
            for child in ast.walk(annotation)
        ):
            return f"line {annotation.lineno}: ExceptionGroup requires Python 3.11"
    return None


def changed_test_paths(changed):
    return [name for name in changed if name.startswith('tests/')
            and name.endswith('.py') and Path(name).is_file()
            and (Path(name).name.startswith('test_') or Path(name).name.endswith('_test.py'))]


def pytest_nodes(source, path):
    """Return collectable pytest function/method fingerprints for one module."""
    tree = ast.parse(source, filename=path)
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test'):
            result[f'{path}::{node.name}'] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            class_context = '|'.join([
                *(ast.dump(item, include_attributes=False) for item in node.decorator_list),
                *(ast.dump(item, include_attributes=False) for item in node.bases),
                *(ast.dump(item, include_attributes=False) for item in node.keywords),
            ])
            for item in node.body:
                if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name.startswith('test')):
                    result[f'{path}::{node.name}::{item.name}'] = (
                        class_context + '|' + ast.dump(item, include_attributes=False))
    return result


def changed_test_nodes(path, before, after):
    before_nodes = pytest_nodes(before, path) if before is not None else {}
    after_nodes = pytest_nodes(after, path)
    return sorted(node_id for node_id, fingerprint in after_nodes.items()
                  if before_nodes.get(node_id) != fingerprint)


def authored_selection(changed):
    paths = changed_test_paths(changed)
    nodeids = []
    for path in paths:
        original = subprocess.run(
            ['git', 'show', f'HEAD:{path}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        before = original.stdout if original.returncode == 0 else None
        nodeids.extend(changed_test_nodes(path, before, Path(path).read_bytes()))
    selected_paths = sorted({node_id.split('::', 1)[0] for node_id in nodeids})
    return {'status': 'pass', 'changed_test_paths': paths,
            'paths': selected_paths, 'nodeids': sorted(nodeids)}


def copy_support(changed):
    """Overlay submitted test assets onto base without submitted implementation."""
    selected = [name for name in changed if name.startswith('tests/') or name == 'conftest.py']
    for name in selected:
        source, target = REPO / name, BASE / name
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        if source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(source))
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return selected


def guarded(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        return {"status": "error", "reason": type(error).__name__, "detail": str(error)}


def module_origin(task, root):
    module = task.get("import_module", "outlines")
    origin = subprocess.check_output([sys.executable, "-c", f"import {module}; print({module}.__file__)"], text=True).strip()
    if not origin.startswith(str(root / task.get("pythonpath", "")) + "/"):
        raise RuntimeError("wrong_module_origin")
    return {"status": "pass", "path": origin}


def red_check(task, changed, nodeids):
    subprocess.run(["git", "worktree", "add", "--detach", str(BASE), task["snapshot"]], check=True)
    support = copy_support(changed)
    try:
        os.chdir(BASE)
        os.environ["PYTHONPATH"] = str(BASE / task.get("pythonpath", ""))
        origin = module_origin(task, BASE)
        result = check("author-red", nodeids)
        result.update(module_origin=origin, support_paths=support)
        return result
    finally:
        os.chdir(REPO)
        os.environ["PYTHONPATH"] = str(REPO / task.get("pythonpath", ""))


def nonempty_added_content(name):
    path = Path(name)
    if not path.is_file() or not path.read_text().strip():
        return False
    patch = subprocess.check_output(['git', 'diff', '--no-ext-diff', '--unified=0', '--', name], text=True)
    return any(line.startswith('+') and not line.startswith('+++') and line[1:].strip()
               for line in patch.splitlines())


def main():
    data = json.loads((HERE / "norms.json").read_text())
    task_id = os.environ["ENGINEERING_77_TASK_ID"]
    task = next(item for item in data["tasks"] if item["id"] == task_id)
    os.chdir(REPO)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != task["snapshot"]:
        raise SystemExit(f"snapshot mismatch: {head}")
    if subprocess.check_output(["git", "status", "--porcelain"]):
        raise SystemExit("target image repository is dirty")

    subprocess.run(["git", "apply", "/submission.patch"], check=True)
    subprocess.run(["git", "add", "-N", "."], check=True)
    changed = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    python_root = REPO / task.get("pythonpath", "")
    os.environ["PYTHONPATH"] = str(python_root)
    import_module = task.get("import_module", "outlines")
    origin_receipt = guarded(module_origin, task, REPO)
    origin = origin_receipt.get("path")

    result = {
        "task": task["id"],
        "snapshot": head,
        "module_origin": origin,
        "module_origin_check": origin_receipt,
        "changed": changed,
        "norms": {},
    }
    selected = {link["norm_id"] for link in task["norms"]}
    runtime_ids = {
        norm["id"]
        for norm in data["norms"]
        if norm["id"] in selected and norm["oracle_kind"] in {"runtime", "public_interface"}
    }
    test_file = "/checks/tests/test_" + task_id.lower().replace("-", "_") + ".py"
    for norm_id in sorted(runtime_ids):
        selector = norm_id.replace("-", "_")
        result["norms"][norm_id] = guarded(check,
            norm_id,
            [test_file],
            selector,
            isolated=True,
        ) if origin_receipt.get("status") == "pass" else {"status": "error", "reason": "module_origin_not_verified"}

    selection = guarded(authored_selection, changed)
    result["authored_selection"] = selection
    if selection.get("status") == "pass":
        result["authored_paths"] = selection["paths"]
        nodeids = selection["nodeids"]
        if nodeids:
            result["author_green"] = (guarded(check, "author-green", nodeids) if origin_receipt.get("status") == "pass" else {"status": "error", "reason": "module_origin_not_verified"})
            result["author_red"] = guarded(red_check, task, changed, nodeids)
        else:
            result["author_green"] = result["author_red"] = {"status": "not_run"}
    else:
        result["author_green"] = result["author_red"] = selection

    result["regression"] = (guarded(check, "regression", task["regression_paths"]) if origin_receipt.get("status") == "pass" else {"status": "error", "reason": "module_origin_not_verified"})

    syntax = {"status": "pass", "python_min": task["minimum_python"]}
    try:
        for name in changed:
            path = Path(name)
            if path.suffix == ".py" and path.is_file():
                minimum = tuple(map(int, task["minimum_python"].split(".")[:2]))
                tree = ast.parse(path.read_text(), filename=name, feature_version=minimum)
                unsupported = minimum_python_annotation_error(tree, task["minimum_python"])
                if unsupported:
                    raise SyntaxError(unsupported)
    except (OSError, SyntaxError, ValueError, UnicodeError) as error:
        syntax = {
            "status": "fail" if isinstance(error, SyntaxError) else "error",
            "python_min": task["minimum_python"],
            "reason": str(error),
        }
    result["syntax"] = syntax

    try:
        docs_page_changed = any(name in changed and nonempty_added_content(name) for name in task["docs_paths"])
        docstrings_changed = False
        inline_change_added = False
        for runtime_name in task.get("doc_runtime_paths", [task["runtime_path"]]):
            base_source = subprocess.check_output(["git", "show", f"{head}:{runtime_name}"], text=True)
            base_docstrings = public_api_docstrings(base_source)
            current_docstrings = public_api_docstrings(Path(runtime_name).read_text())
            docstrings_changed = docstrings_changed or any(
                value and value != base_docstrings.get(name) for name, value in current_docstrings.items()
            )
            marker = f".. versionchanged:: {task.get('development_version', '')}"
            inline_change_added = inline_change_added or any(
                value
                and value != base_docstrings.get(name)
                and marker in value
                and marker not in (base_docstrings.get(name) or "")
                for name, value in current_docstrings.items()
            )
        docs_changed = docs_page_changed or docstrings_changed
        result["docs_signal"] = {
            "status": "pass" if docs_changed else "fail",
            "reference_page_changed": docs_page_changed,
            "public_docstrings_changed": docstrings_changed,
        }
        changelog_path = task.get("changelog_path")
        result["changelog_signal"] = {
            "status": "pass" if changelog_path in changed and nonempty_added_content(changelog_path) else "fail",
            "path": changelog_path,
        }
        result["inline_change_signal"] = {
            "status": "pass" if inline_change_added else "fail",
            "version": task.get("development_version"),
        }
    
    except (OSError, SyntaxError, ValueError, subprocess.SubprocessError) as error:
        for key in ("docs_signal", "changelog_signal", "inline_change_signal"):
            result[key] = {"status": "error", "reason": type(error).__name__}

    result["functional"] = {"status": "not_run", "reason": "original_coding_receipt_only"}
    result["submission_sha256"] = sha256(Path("/submission.patch").read_bytes())
    print("ENGINEERING_77_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
