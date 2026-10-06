"""Patch-bound deterministic checks; execute only in the isolated target image.

The host verifies the frozen image, patch and resource identities. This adapter
does not inject functional tests or make semantic judgments. Rust source changes
use the frozen Tiktoken image's cached offline build, with a pristine red baseline.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from pytest_check import check


REPO = Path('/workspace/repo')
BASE = Path('/tmp/unpatched')
NATIVE_BASE = Path('/tmp/norm-native-base')
MODULES = {'DEQ': 'dirty_equals', 'JIN': 'jinja2', 'CLI': 'click',
           'TIK': 'tiktoken', 'OUT': 'outlines'}


def git(*args):
    return subprocess.check_output(['git', *args])


def guarded(function, *args):
    try:
        return function(*args)
    except Exception as error:
        return {'status': 'error', 'reason': type(error).__name__, 'detail': str(error)}


def oracle_check(norm_id, test_file):
    previous = os.environ.get('PYTEST_ADDOPTS')
    os.environ['PYTEST_ADDOPTS'] = (previous or '') + ' --import-mode=importlib'
    try:
        return check(norm_id, [test_file], norm_id.replace('-', '_'), isolated=True)
    finally:
        if previous is None:
            os.environ.pop('PYTEST_ADDOPTS', None)
        else:
            os.environ['PYTEST_ADDOPTS'] = previous


def runtime_setup(task):
    prefix = ''
    wheels = os.environ.get('N4C_RUNTIME_WHEELS')
    if wheels:
        paths = sorted(str(path) for path in Path(wheels).glob('*.whl'))
        if not paths:
            raise RuntimeError('runtime_wheel_mount_empty')
        if task['code'] == 'OUT':
            subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages',
                            '/tmp/reg-venv'], check=True)
            python = '/tmp/reg-venv/bin/python'
            subprocess.run([python, '-m', 'pip', 'install', '--no-index', '--no-deps',
                            '--ignore-installed', *paths], check=True)
            subprocess.run([python, '-m', 'pip', 'check'], check=True)
            os.environ['PATH'] = '/tmp/reg-venv/bin:' + os.environ['PATH']
            site = subprocess.check_output(
                [python, '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
                text=True).strip()
            (Path(site) / 'sitecustomize.py').write_text('import torch._dynamo\n')
            prefix = site + ':'
        else:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                            '--no-deps', '--target', '/tmp/runtime-packages', *paths],
                           check=True)
            prefix = '/tmp/runtime-packages:'
    set_source(REPO, prefix)
    if task['code'] == 'TIK':
        os.environ['PYTEST_ADDOPTS'] = f'--hypothesis-seed={task["hypothesis_seed"]}'
    return {'status': 'pass', 'prefix': prefix}


def set_source(root, prefix):
    os.environ['PYTHONPATH'] = f'{prefix}{root}/src:{root}'


def module_origin(task, root):
    name = MODULES[task['code']]
    origin = subprocess.check_output(
        ['python', '-c', f'import {name}; print({name}.__file__)'], text=True).strip()
    if not origin.startswith(str(root) + '/'):
        raise RuntimeError(f'wrong_module_origin: {origin}')
    result = {'status': 'pass', 'path': origin}
    if task['code'] == 'TIK':
        native = subprocess.check_output(['python', '-c',
            'import tiktoken._tiktoken; print(tiktoken._tiktoken.__file__)'], text=True).strip()
        if not native.startswith(str(root) + '/'):
            raise RuntimeError(f'wrong_native_module_origin: {native}')
        result.update(native_path=native, native_sha256=hashlib.sha256(Path(native).read_bytes()).hexdigest())
    return result


def native_rebuild(changed):
    """Compile only Rust source edits using the frozen image's offline cache."""
    unsupported = [name for name in changed if name.endswith('.so') or name in (
        'Cargo.toml', 'Cargo.lock', 'setup.py', 'setup.cfg', 'pyproject.toml', 'build.rs')
        or name.startswith('.cargo/') or (name.endswith('.rs') and not name.startswith('src/'))]
    if unsupported:
        return {'status': 'error', 'reason': 'native_build_input_unsupported', 'paths': unsupported}
    rust = [name for name in changed if name.endswith('.rs')]
    if not rust:
        return {'status': 'not_applicable', 'reason': 'native_inputs_unchanged'}
    originals = sorted((REPO / 'tiktoken').glob('_tiktoken*.so'))
    if len(originals) != 1:
        return {'status': 'error', 'reason': 'native_extension_identity_ambiguous'}
    original = originals[0]
    NATIVE_BASE.mkdir()
    saved = NATIVE_BASE / original.name
    shutil.copyfile(original, saved)
    command = ['cargo', 'build', '--release', '--features', 'python', '--offline', '--locked']
    receipt = {'status': 'error', 'paths': rust, 'command': command,
               'extension_path': str(original),
               'base_sha256': hashlib.sha256(saved.read_bytes()).hexdigest()}
    start = time.monotonic()
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=180)
        print(process.stdout.decode(errors='replace'), flush=True)
        receipt['exit_code'] = process.returncode
        receipt['output_sha256'] = hashlib.sha256(process.stdout).hexdigest()
        if process.returncode:
            receipt['reason'] = 'native_build_failed'
            return receipt
        built = REPO / 'target/release/libtiktoken.so'
        if not built.is_file():
            receipt['reason'] = 'native_build_output_missing'
            return receipt
        shutil.copyfile(built, original)
        receipt.update(status='pass', built_sha256=hashlib.sha256(original.read_bytes()).hexdigest())
        return receipt
    except (OSError, subprocess.TimeoutExpired) as error:
        receipt.update(reason='native_build_execution_error', detail=str(error))
        return receipt
    finally:
        receipt['seconds'] = round(time.monotonic() - start, 3)


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


def red_check(task, changed, paths, prefix):
    subprocess.run(['git', 'worktree', 'add', '--detach', str(BASE), task['snapshot']],
                   check=True, stdout=subprocess.DEVNULL)
    support = copy_support(changed)
    if task['code'] == 'TIK':
        # The red baseline always uses the image's original native extension,
        # including when the submitted Rust sources were rebuilt for green.
        native_source = NATIVE_BASE if NATIVE_BASE.exists() else REPO / 'tiktoken'
        for source in native_source.glob('*.so'):
            shutil.copyfile(source, BASE / 'tiktoken' / source.name)
    try:
        os.chdir(BASE)
        set_source(BASE, prefix)
        origin = module_origin(task, BASE)
        receipt = check('author_red', paths)
        receipt.update(module_origin=origin['path'], module_origin_check=origin, support_paths=support)
        return receipt
    finally:
        os.chdir(REPO)
        set_source(REPO, prefix)


def syntax_check(task, changed):
    checked, deleted = [], []
    for name in changed:
        if not name.endswith('.py'):
            continue
        if not Path(name).exists():
            deleted.append(name)
            continue
        try:
            ast.parse(Path(name).read_bytes(), feature_version=(3, task['minimum_python']))
        except (SyntaxError, ValueError) as error:
            return {'status': 'fail', 'reason': str(error), 'path': name,
                    'python_min': f'3.{task["minimum_python"]}'}
        checked.append(name)
    return {'status': 'pass', 'python_min': f'3.{task["minimum_python"]}',
            'checked_paths': checked, 'deleted_paths': deleted}


def doc_examples(task):
    # A fresh interpreter uses the same submitted-module environment as tests.
    script = '''import json, re, textwrap, dirty_equals
try:
    obj = getattr(dirty_equals, DOC_OBJECT, None)
    doc = getattr(getattr(obj, '__init__', None), '__doc__', '') or ''
    examples = re.findall(r'```py[^\\n]*\\n(.*?)```', doc, flags=re.S)
    for example in examples:
        exec(textwrap.dedent(example), {})
    result = {'status': 'pass' if examples else 'review_required', 'blocks': len(examples)}
except Exception as error:
    result = {'status': 'fail', 'reason': repr(error)}
print('DOC_RESULT=' + json.dumps(result))
'''
    proc = subprocess.run(['python', '-c', 'DOC_OBJECT = ' + repr(task['doc_object']) + '\n' + script],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    print(proc.stdout)
    lines = [line for line in proc.stdout.splitlines() if line.startswith('DOC_RESULT=')]
    if proc.returncode or not lines:
        return {'status': 'error', 'reason': 'doc_example_process_failed', 'exit_code': proc.returncode}
    return json.loads(lines[-1].split('=', 1)[1])


def evaluate(task, norms):
    result = {'task': task['id'], 'snapshot': task['snapshot'], 'norms': {},
              'functional': {'status': 'not_run', 'reason': 'reuse_original_formal_run'},
              'regression': {'status': 'not_run', 'reason': 'outside_norm_check_scope'}}
    os.chdir(REPO)
    head = git('rev-parse', 'HEAD').decode().strip()
    if head != task['snapshot']:
        raise RuntimeError(f'snapshot_mismatch: {head}')
    if git('diff', 'HEAD', '--name-only'):
        raise RuntimeError('target_checkout_not_clean')
    patch = Path('/submission.patch')
    if patch.stat().st_size:
        subprocess.run(['git', 'apply', '--check', str(patch)], check=True)
        subprocess.run(['git', 'apply', str(patch)], check=True)
    subprocess.run(['git', 'add', '-N', '.'], check=True)
    changed = [name.decode() for name in git('diff', '--name-only', '-z').split(b'\0') if name]
    result['changed'] = changed
    candidate_paths = changed_test_paths(changed)
    selection = guarded(authored_selection, changed)
    result['authored_selection'] = selection
    paths = selection.get('paths', candidate_paths)
    nodeids = selection.get('nodeids', [])
    result['authored_paths'] = paths
    result['authored_nodeids'] = nodeids
    result['syntax'] = guarded(syntax_check, task, changed)
    runtime = guarded(runtime_setup, task)
    result['runtime_setup'] = runtime
    native = guarded(native_rebuild, changed) if task['code'] == 'TIK' else None
    if native is not None:
        result['native_build'] = native
    origin = guarded(module_origin, task, REPO) if runtime['status'] == 'pass' else runtime
    result['module_origin_check'] = origin
    result['module_origin'] = origin.get('path')
    blocked = runtime if runtime['status'] != 'pass' else origin if origin['status'] != 'pass' else None
    if task['code'] == 'TIK':
        result['hypothesis_seed'] = task['hypothesis_seed']
        if native['status'] == 'error':
            blocked = native
    for norm in norms:
        if norm['oracle_kind'] in ('runtime', 'public_export'):
            result['norms'][norm['id']] = blocked or guarded(
                oracle_check, norm['id'], '/checks/' + task['norm_test_files'][norm['id']])
    if selection['status'] != 'pass':
        result['author_green'] = result['author_red'] = selection
    elif not nodeids:
        result['author_green'] = result['author_red'] = {
            'status': 'not_run', 'reason': 'no_authored_tests'}
    elif blocked:
        result['author_green'] = result['author_red'] = blocked
    else:
        result['author_green'] = guarded(check, 'author_green', nodeids)
        result['author_red'] = guarded(red_check, task, changed, nodeids, runtime['prefix'])
    if task['code'] == 'DEQ':
        result['doc_examples'] = blocked or guarded(doc_examples, task)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    args = parser.parse_args()
    data = json.loads(Path('/checks/norms.json').read_text())
    task = next(task for task in data['tasks'] if task['id'] == args.task)
    selected = {item['norm_id'] for item in task['norms']}
    result = guarded(evaluate, task, [norm for norm in data['norms'] if norm['id'] in selected])
    result.setdefault('task', args.task)
    print('CONTRIBUTION_RESULT=' + json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
