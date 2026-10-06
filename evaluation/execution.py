"""Authenticated isolated checker execution shared by both backends."""
from __future__ import annotations

import argparse

from collections import Counter, defaultdict

from concurrent.futures import ThreadPoolExecutor, as_completed

import csv

from datetime import datetime, timezone

import fcntl

import hashlib

import importlib.util

import json

from pathlib import Path

import subprocess

import time

import uuid

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def digest(value: object) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())

def read(path: Path) -> dict:
    return json.loads(path.read_text())

def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def docker_command(image: str, name: str, mounts: list[tuple[str, str]], script: str,
                   task_id: str, env: dict | None = None) -> list[str]:
    command = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--pids-limit', '512', '--memory', '10g', '--cpus', '2', '--entrypoint', '',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PIP_NO_INDEX=1',
               '-e', 'PIP_DISABLE_PIP_VERSION_CHECK=1']
    for source, destination in mounts:
        command += ['--mount', f'type=bind,src={source},dst={destination},readonly']
    for key, value in (env or {}).items():
        command += ['-e', f'{key}={value}']
    return command + [image, 'python', script, task_id]

def execute(command: list[str], name: str, log: Path, timeout: int, prefix: str) -> dict:
    start = time.monotonic()
    timed_out = False
    execution_error = None
    with log.open('wb') as output:
        try:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out, code = True, None
        except OSError as error:
            execution_error, code = str(error), None
        finally:
            try:
                subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=60, check=True)
            except (OSError, subprocess.SubprocessError):
                # --rm normally removed it already; verify no named container remains.
                check = subprocess.run(['docker', 'ps', '-aq', '--filter', f'name=^/{name}$'],
                                       capture_output=True, text=True, timeout=60, check=True)
                if check.stdout.strip():
                    raise RuntimeError(f'Container cleanup failed: {name}')
    raw = log.read_bytes()
    lines = [line for line in raw.decode(errors='replace').splitlines() if line.startswith(prefix)]
    value = None
    if code == 0 and len(lines) == 1:
        try:
            value = json.loads(lines[0][len(prefix):])
        except ValueError:
            pass
    return {'exit_code': code, 'execution_error': execution_error, 'timeout': timed_out, 'seconds': time.monotonic() - start,
            'log_sha256': sha(raw), 'log': str(log), 'command': command, 'value': value,
            'status': 'complete' if value is not None else 'error'}

def assess(record: dict, data: dict, checks: Path) -> dict:
    spec = importlib.util.spec_from_file_location('frozen_assess', checks / 'assess.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.assess(record, data, {})
    for norm_id, row in result['norms'].items():
        if row['kind'] in ('runtime', 'public_export'):
            evidence = record['dynamic'].get('norms', {}).get(norm_id, {})
            row['reason'] = evidence.get('reason')
        elif row['kind'] in ('style', 'typing') and row['status'] == 'error':
            row['reason'] = 'static_check_not_validly_completed'
    # Absence of a completed check is not evidence of absent authored tests.
    if 'authored_paths' not in record['dynamic']:
        for row in result['norms'].values():
            if row['kind'] == 'authored_tests':
                row.update(status='error', reason='authored_test_discovery_not_completed')
    return result

def execution_verdict(task_id: str, patch_sha256: str, dynamic: dict, static: dict,
                      data: dict, checks: Path) -> tuple[dict, dict]:
    """Derive the strict verdict from authenticated completed or partial evidence."""
    record = {'task': task_id, 'state': 'candidate', 'patch_sha256': patch_sha256,
              'dynamic': dynamic['value'] or {},
              'static': static['value'] or {'style': [], 'typing': []},
              'changed': (dynamic['value'] or {}).get('changed', [])}
    verdict = assess(record, data, checks)
    if dynamic['value'] is None or 'changed' not in dynamic['value']:
        for row in verdict['norms'].values():
            if row['kind'] == 'changelog':
                row.update(status='error', reason='changed_files_not_observed')
    return record, verdict

CHECKS = Path(__file__).resolve().parent / "checks/standard"
