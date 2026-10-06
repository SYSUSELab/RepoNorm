import argparse

import ast

from concurrent.futures import ThreadPoolExecutor

import hashlib

import json

import os

from pathlib import Path

import subprocess

import tempfile

import time

import uuid

def command(args, cwd, env=None, timeout=120):
    start = time.monotonic()
    try:
        r = subprocess.run([str(x) for x in args], cwd=cwd, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return {'exit_code': r.returncode, 'output': r.stdout.decode(errors='replace'),
                'seconds': round(time.monotonic() - start, 3)}
    except subprocess.TimeoutExpired as e:
        return {'exit_code': None, 'status': 'error', 'reason': 'timeout',
                'output': (e.stdout or b'').decode(errors='replace')}

def static_checks(task, repo, changed):
    if task['code'] == 'TIK':
        # This snapshot declares no Python formatter, linter, or type-checker
        # gate. Do not manufacture one from another repository's policy.
        return {'style': [], 'typing': [],
                'tool_versions': command(['python', '--version'], repo)}
    tool = (Path('/opt/tools/bin')
            if task['code'] == 'OUT' else LOCAL / f'tools-{task["code"]}' / 'bin')
    py = [p for p in changed if p.endswith('.py') and (repo / p).exists()]
    cmds = []
    if task['code'] == 'OUT':
        cmds = [[tool / 'ruff', 'check', '--config', 'pyproject.toml', *py]]
    elif task['code'] == 'CLI':
        cmds = [[tool / 'ruff', 'format', '--check', *py],
                [tool / 'ruff', 'check', '--no-fix', *py]]
    else:
        # One file per black command avoids old black's multiprocessing
        # event-loop requirements in restrictive development shells.
        cmds = [[tool / 'black', '--check', p] for p in py]
        if task['code'] == 'DEQ':
            cmds += [[tool / 'isort', '--check-only', *py],
                     [tool / 'flake8', '--max-complexity', '10', '--max-line-length', '120',
                      '--ignore', 'E203,W503', *py]]
        else:
            cmds += [[tool / 'flake8', *py]]
            # Import reordering is a rewriting tool, so run only on copies.
            with tempfile.TemporaryDirectory(dir=LOCAL, prefix='imports-') as tmp:
                for name in py:
                    target = Path(tmp) / Path(name).name
                    target.write_bytes((repo / name).read_bytes())
                    result = command([tool / 'reorder-python-imports', '--application-directories', 'src', target], repo)
                    if target.read_bytes() != (repo / name).read_bytes():
                        result['exit_code'] = 1
                    else:
                        result['exit_code'] = 0 if result['exit_code'] == 0 else result['exit_code']
                    cmds.append(None)
                    # Keep individual inspection receipt alongside other tools.
                    cmds[-1] = result
    style = [x if isinstance(x, dict) else command(x, repo) for x in cmds]
    style.append(command(['git', 'diff', '--check'], repo))
    if task['code'] == 'OUT':
        violations = []
        for name in changed:
            path = repo / name
            if not path.is_file():
                continue
            raw = path.read_bytes()
            if raw and not raw.endswith(b'\n'):
                violations.append(f'{name}: no newline at end of file')
            text = raw.decode(errors='replace')
            if any(line.startswith(('<<<<<<<', '=======', '>>>>>>>'))
                   for line in text.splitlines()):
                violations.append(f'{name}: merge-conflict marker')
            if name.endswith('.py'):
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name)
                            and node.func.id == 'breakpoint'):
                        violations.append(f'{name}:{node.lineno}: debug statement')
                    if (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == 'set_trace'):
                        violations.append(f'{name}:{node.lineno}: debug statement')
        style.append({'exit_code': 1 if violations else 0,
                      'output': '\n'.join(violations),
                      'check': 'pre-commit-hooks'})
    type_cmd = [tool / 'mypy', '--no-incremental']
    if task['code'] == 'DEQ':
        type_cmd += ['dirty_equals', 'tests/mypy_checks.py']
    elif task['code'] == 'OUT':
        type_cmd += ['--allow-redefinition', *py]
    typing = [command(type_cmd, repo)]
    if task['code'] == 'CLI':
        # This tool environment is private to the CLI worker. Rebind editable
        # installation so verifytypes cannot silently inspect another checkout.
        install = command([tool / 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', repo], repo)
        typing.append(install)
        env = dict(os.environ, XDG_CACHE_HOME=str(LOCAL / 'cache'), PYTHONNOUSERSITE='1',
                   PATH=str(tool) + os.pathsep + os.environ['PATH'])
        for args in (['tests/typing'], ['--verifytypes', 'click', '--ignoreexternal']):
            typing.append(command([tool / 'pyright', '--pythonpath', tool / 'python', *args], repo, env))
        last = typing[-1]
        if str(repo / 'src/click') not in last['output']:
            last.update(exit_code=None, status='error', reason='wrong_pyright_package_origin')
    return {'style': style, 'typing': typing,
            'tool_versions': command([tool / 'pip', 'freeze', '--all'], repo)}

LOCAL = Path("/opt/static")
N4C = Path("/opt/n4c")
