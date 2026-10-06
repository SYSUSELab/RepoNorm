"""Conservative, replay-only evidence for the authored-test coarse proxy.

No source is executed. ``before_files`` must contain the complete Python source
snapshot (not only changed files); ``patched_files`` is the corresponding patched
snapshot. Unknown bindings, dynamic exports and unmatched errors remain errors.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Mapping


Files = Mapping[str, str]
IMPORT_ERROR = re.compile(
    r"^E\s+ImportError: cannot import name '([A-Za-z]\w*)' from "
    r"'([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)' \(/tmp/unpatched/([^\s)]+)\)$",
    re.MULTILINE,
)
COLLECTION = re.compile(r'^_+ ERROR collecting ([^\n]+?) _+\s*$', re.MULTILINE)


def _module_path(module: str, files: Files) -> str | None:
    stem = module.replace('.', '/')
    candidates = [p for p in (f'{stem}.py', f'{stem}/__init__.py',
                              f'src/{stem}.py', f'src/{stem}/__init__.py') if p in files]
    return candidates[0] if len(candidates) == 1 else None


def _tree(files: Files, path: str) -> ast.Module:
    return ast.parse(files[path], filename=path)


def _absent(tree: ast.Module, name: str) -> bool:
    # A nested/conditional mention deliberately prevents proof; no inference from
    # __all__ or unavailable dynamic import behavior is allowed.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in (name, '__getattr__'):
                return False
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name:
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == '*' or (alias.asname or alias.name.split('.')[0]) == name:
                    return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ('exec', 'eval', 'globals', 'locals', 'vars', 'setattr'):
                return False
    return True


def _definition(files: Files, module: str, name: str,
                seen: frozenset[tuple[str, str]] = frozenset()) -> list[dict[str, Any]] | None:
    key = (module, name)
    if key in seen or len(seen) >= 20:
        return None
    path = _module_path(module, files)
    if path is None:
        return None
    tree = _tree(files, path)
    bindings = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            bindings.append((node, None))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    package = module.split('.') if path.endswith('/__init__.py') else module.split('.')[:-1]
                    if node.level:
                        if node.level > len(package):
                            return None
                        parts = package[:len(package) - node.level + 1]
                        target = '.'.join(parts + ([node.module] if node.module else []))
                    else:
                        target = node.module or ''
                    bindings.append((node, (target, alias.name)))
    if not bindings:
        return None
    node, target = bindings[0]
    # Identical repeated re-exports have the same statically resolved target.
    # Distinct targets or repeated definitions still cannot establish one chain.
    if len(bindings) > 1 and (target is None or any(other != target for _, other in bindings)):
        return None
    bound_nodes = {id(item) for item, _ in bindings}
    remaining = ast.Module(body=[item for item in tree.body if id(item) not in bound_nodes], type_ignores=[])
    if not _absent(remaining, name):
        return None
    witness = {'module': module, 'symbol': name, 'path': path, 'line': node.lineno}
    if target is None:
        return [witness]
    rest = _definition(files, *target, seen | {key})
    return [witness, *rest] if rest else None


def assess_authored_tests(dynamic: Mapping[str, Any], before_files: Files,
                          patched_files: Files, dynamic_log: str) -> dict[str, Any]:
    """Return status/reason and AST/log witnesses; never promote arbitrary error."""
    def result(status: str, reason: str, **evidence: Any) -> dict[str, Any]:
        return {'status': status, 'reason': reason, 'protocol': 'authored-test-evidence-1.0', **evidence}

    if 'authored_paths' not in dynamic:
        return result('error', 'authored_test_discovery_not_completed')
    authored = dynamic.get('authored_paths')
    if not authored:
        return result('fail', 'no_author_owned_python_test_change')
    green, red = dynamic.get('author_green', {}), dynamic.get('author_red', {})
    if green.get('status') == 'fail':
        return result('fail', 'author_tests_fail_on_patch')
    if green.get('status') != 'pass' or green.get('counts', {}).get('tests', 0) <= 0:
        return result('error', 'green_execution_not_proven')
    if red.get('status') == 'fail' and red.get('counts', {}).get('failures', 0) > 0:
        if red.get('counts', {}).get('errors', 0):
            return result('error', 'red_failures_mixed_with_execution_errors')
        return result('pass', 'green_pass_and_base_test_failure')
    if red.get('status') == 'pass':
        return result('fail', 'no_red_green_difference')
    if red.get('status') != 'error' or red.get('pytest_exit_code') != 2:
        return result('error', 'red_execution_not_supported')
    origin = red.get('module_origin_check', {})
    if origin.get('status') != 'pass' or not origin.get('path', '').startswith('/tmp/unpatched/'):
        return result('error', 'base_module_origin_not_proven')
    log = re.sub(r'\x1b\[[0-9;]*m', '', dynamic_log)
    # Match only pytest collection blocks tied to base test-module imports.
    # Multiple test invocations in the historical log are not interchangeable.
    headers = list(COLLECTION.finditer(log))
    blocks = []
    for i, header in enumerate(headers):
        block = log[header.end():headers[i + 1].start() if i + 1 < len(headers) else len(log)]
        block = block.split('short test summary info', 1)[0]
        if "ImportError while importing test module '/tmp/unpatched/" in block:
            blocks.append((header.group(1).strip(), block))
    if len(blocks) != red.get('counts', {}).get('errors') or not blocks:
        return result('error', 'base_collection_error_count_mismatch')
    witnesses = []
    try:
        for test_path, block in blocks:
            matches = list(IMPORT_ERROR.finditer(block))
            error_lines = re.findall(r'^E\s+.*(?:Error|Exception):.*$', block, re.MULTILINE)
            if len(matches) != 1 or len(error_lines) != 1:
                return result('error', 'collection_error_not_exact_missing_import')
            match = matches[0]
            name, module, path = match.groups()
            if test_path not in authored or test_path not in patched_files:
                return result('error', 'collection_error_outside_authored_tests')
            if _module_path(module, before_files) != path or _module_path(module, patched_files) != path:
                return result('error', 'error_module_source_binding_not_proven')
            root = module.split('.')[0]
            root_path = _module_path(root, before_files)
            if origin['path'] != '/tmp/unpatched/' + (root_path or ''):
                return result('error', 'error_module_outside_verified_package')
            imports = [node for node in ast.walk(_tree(patched_files, test_path))
                       if isinstance(node, ast.ImportFrom) and not node.level
                       and node.module == module and any(a.name == name for a in node.names)]
            chain = _definition(patched_files, module, name)
            if not imports or not chain or not _absent(_tree(before_files, path), name):
                return result('error', 'new_public_api_not_proven')
            witnesses.append({'test_path': test_path, 'test_import_line': imports[0].lineno,
                              'missing_symbol': name, 'module': module,
                              'base_path': path, 'definition_chain': chain,
                              'error_line': match.group(0)})
    except (SyntaxError, ValueError, KeyError):
        return result('error', 'source_parse_or_evidence_missing')
    return result('pass', 'green_pass_and_proven_new_public_api_collection_error', witnesses=witnesses)
