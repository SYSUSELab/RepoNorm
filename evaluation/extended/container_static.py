"""Execute calibrated tool gates only inside an isolated disposable container."""
import importlib.util
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

def classify_result(value, args, changed):
    if Path(str(args[0])).name == 'pyupgrade' and value.get('exit_code') in (0, 1):
        if not any(marker in value.get('output', '') for marker in ('Traceback (most recent call last)', 'FailedToLoadPlugin', 'There was a critical error')):
            return value
    sys.path.insert(0, '/standard-adapter')
    from container_static import classify_tool_result
    return classify_tool_result(value, args, changed)


def typing_command(args, task, changed_python):
    # Both frozen Outlines hooks pass changed Python files, excluding examples;
    # their package is outlines/, not src/. Preserve all original tool flags.
    if Path(str(args[0])).name == 'mypy' and task['tool_profile'].startswith('outlines-'):
        return [args[0], '--no-incremental', '--allow-redefinition',
                *(name for name in changed_python if not name.startswith('examples/'))]
    return args


def main():
    task = next(t for t in json.loads(Path('/checks/norms.json').read_text())['tasks'] if t['id'] == sys.argv[1])
    repo = Path('/workspace/repo')
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--quiet', '/snapshot.bundle', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '--quiet', '--detach', task['snapshot']], check=True)
    subprocess.run(['git', '-C', str(repo), 'apply', '/submission.patch'], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '-N', '.'], check=True)
    changed = subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only'], text=True).splitlines()
    sys.path.insert(0, '/checks')
    frozen_path = Path('/tmp/frozen/CooperBench/checks/run.py')
    frozen_path.parent.mkdir(parents=True)
    shutil.copyfile('/checks/run.py', frozen_path)
    spec = importlib.util.spec_from_file_location('frozen_tools', frozen_path)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    frozen.LOCAL = Path('/tmp/static')
    frozen.LOCAL.mkdir()
    frozen.TOOLS_BY_PROFILE = {key: Path('/opt/tools/bin') for key in frozen.TOOLS_BY_PROFILE}
    (frozen.LOCAL / 'tools-OUT1371').symlink_to('/opt/tools')
    python_files = [name for name in changed if name.endswith('.py') and (repo / name).is_file()]
    original = frozen.command
    def command(args, cwd, timeout=180):
        args = typing_command(args, task, python_files)
        if Path(str(args[0])).name == 'mypy' and len(args) == 3:
            return {'status': 'error', 'exit_code': None, 'reason': 'no_python_typing_targets', 'output': '', 'command': [str(a) for a in args]}
        value = original(args, cwd, timeout)
        value['command'] = [str(arg) for arg in args]
        return classify_result(value, args, changed)
    frozen.command = command
    os.environ.update(PIP_NO_INDEX='1', PYTHONNOUSERSITE='1', PATH='/opt/tools/bin:' + os.environ['PATH'])
    python_files = [name for name in changed if name.endswith('.py') and (repo / name).is_file()]
    if python_files:
        result = frozen.static_checks(repo, changed, task)
    else:
        result = {key: [{'status': 'error', 'exit_code': None, 'reason': 'no_python_targets'}] for key in ('style', 'typing')}
    result.update(task=task['id'], snapshot=task['snapshot'], submission_sha256=hashlib.sha256(Path('/submission.patch').read_bytes()).hexdigest())
    print('STATIC_RESULT=' + json.dumps(result))


if __name__ == '__main__':
    main()
