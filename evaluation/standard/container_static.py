"""Run frozen static gates in a disposable, network-disabled tool container."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def changed_source_syntax(result: dict, executable: str, changed: list[str]) -> bool:
    if executable != 'mypy' or result.get('exit_code') != 2:
        return False
    changed_paths = {str(Path(path)) for path in changed}
    pattern = re.compile(r'^(.*?):\d+(?::\d+)?: error: .*\[syntax\]$', re.MULTILINE)
    return any(str(Path(match.group(1))) in changed_paths
               for match in pattern.finditer(result.get('output', '')))


def classify_tool_result(result: dict, args: list[object], changed: list[str]) -> dict:
    executable = Path(str(args[0])).name
    code = result.get('exit_code')
    allowed = {'black': {0, 1}, 'ruff': {0, 1}, 'mypy': {0, 1},
               'isort': {0, 1}, 'pyright': {0, 1}, 'flake8': {0, 1},
               'reorder-python-imports': {0, 1}, 'git': {0, 2}}
    if changed_source_syntax(result, executable, changed):
        result.update(status='fail', reason='source_syntax_diagnostic')
    elif code is None or code not in allowed.get(executable, {0}):
        result.update(status='error', reason='static_tool_execution_error')
    # Some wrappers return 1 for bootstrap/plugin failure rather than a finding.
    if any(marker in result.get('output', '') for marker in
           ('Traceback (most recent call last)', 'FailedToLoadPlugin',
            'There was a critical error')):
        result.update(status='error', reason='static_tool_execution_error')
    return result


def main() -> None:
    task_id = sys.argv[1]
    data = json.loads(Path('/checks/norms.json').read_text())
    task = next(t for t in data['tasks'] if t['id'] == task_id)
    subprocess.run(['git', 'clone', '--quiet', '/snapshot.bundle', '/workspace/repo'], check=True)
    repo = Path('/workspace/repo')
    subprocess.run(['git', '-C', str(repo), 'checkout', '--quiet', '--detach', task['snapshot']], check=True)
    subprocess.run(['git', '-C', str(repo), 'apply', '/submission.patch'], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '-N', '.'], check=True)
    changed = subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only'], text=True).splitlines()
    # Preserve original tool commands while moving all target execution off host.
    spec = importlib.util.spec_from_file_location(
        'frozen_static', '/checks/run.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    frozen.LOCAL = Path('/opt/static')
    frozen.N4C = Path('/opt/n4c')
    frozen.LOCAL.mkdir(parents=True, exist_ok=True)
    for target in (frozen.LOCAL / f'tools-{task["code"]}',
                   frozen.N4C / 'outputs/2026-09-11-task-norm-full/tools-OUT'):
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to('/opt/tools')
    os.environ.update(PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
                      PYTHONNOUSERSITE='1', PATH='/opt/tools/bin:' + os.environ['PATH'])
    actual_command = frozen.command

    def command(args, cwd, env=None, timeout=120):
        result = actual_command(args, cwd, env, timeout)
        result['command'] = [str(arg) for arg in args]
        # A tool invocation/configuration failure is not a source violation.
        return classify_tool_result(result, args, changed)

    frozen.command = command
    py = [p for p in changed if p.endswith('.py') and (repo / p).is_file()]
    if not py:
        result = {'style': [{'status': 'error', 'exit_code': None, 'reason': 'no_python_targets'}],
                  'typing': [{'status': 'error', 'exit_code': None, 'reason': 'no_python_targets'}]}
    else:
        result = frozen.static_checks(task, repo, changed)
    print('STATIC_RESULT=' + json.dumps(result))


if __name__ == '__main__':
    main()
