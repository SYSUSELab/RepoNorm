"""Host-owned task container cleanup, scoped by exact frozen run/task labels."""
from __future__ import annotations

import subprocess
from contextlib import contextmanager

from four_arm_common import utc


def container_labels(identity: str, task: str) -> list[str]:
    return [f'n4c.run={identity}', f'n4c.task={task}', 'n4c.role=solver']


def cleanup_solver(identity: str, task: str) -> dict:
    # Include both the solver and its evaluator; never touch other run/tasks.
    labels = container_labels(identity, task)[:2]
    query = ['docker', 'ps', '-aq', '--no-trunc']
    for label in labels:
        query.extend(['--filter', 'label=' + label])
    receipt = {'labels': labels, 'status': 'unconfirmed', 'containers': [], 'finished_at': None}
    try:
        found = subprocess.run(query, capture_output=True, text=True, timeout=180, check=True)
        receipt['containers'] = found.stdout.split()
        removal_errors = []
        for container in receipt['containers']:
            try:
                subprocess.run(['docker', 'rm', '-f', container], capture_output=True,
                               text=True, timeout=180, check=True)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                removal_errors.append({'container': container, 'error_type': type(exc).__name__})
        if removal_errors:
            receipt['removal_errors'] = removal_errors
        # A timed-out CLI may still have removed the container; --rm can also
        # race this command. The final query, not the rm exit code, decides.
        remaining = subprocess.run(query, capture_output=True, text=True, timeout=180, check=True)
        if remaining.stdout.strip():
            raise RuntimeError('Task containers remain after removal')
        receipt['status'] = 'complete'
    except Exception as exc:
        # Keep terminal task output writable even when Docker is unavailable.
        receipt['error_type'] = type(exc).__name__
    receipt['finished_at'] = utc()
    return receipt


@contextmanager
def evaluator_lifecycle(identity: str, task: str):
    """Bound host cleanup without changing the frozen test runner or timeout.

    Used only inside the dedicated single-task worker process. Restore the
    subprocess hook before the worker's final, label-scoped absence check.
    """
    original_run = subprocess.run
    cleanup_errors = []

    def run(args, **kwargs):
        removal = args[:3] == ['docker', 'rm', '-f']
        if args[:2] == ['docker', 'run']:
            labels = [*container_labels(identity, task)[:2], 'n4c.role=evaluator']
            args = [*args[:2], *[part for label in labels for part in ('--label', label)], *args[2:]]
        if removal:
            kwargs['timeout'] = 180
        try:
            completed = original_run(args, **kwargs)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            if removal:
                cleanup_errors.append({'container': args[3], 'error_type': type(exc).__name__})
                # Preserve the evaluation result; final cleanup checks absence.
                return subprocess.CompletedProcess(args, 124, b'', b'')
            if isinstance(exc, subprocess.TimeoutExpired) and '/usr/local/bin/runner.sh' not in args:
                # Frozen runtimes catch test timeouts. Setup timeouts instead
                # need infrastructure recovery of the same saved patch.
                raise RuntimeError('Evaluator environment setup timed out') from exc
            raise
        if removal and completed.returncode:
            cleanup_errors.append({'container': args[3], 'returncode': completed.returncode})
        return completed

    subprocess.run = run
    try:
        yield cleanup_errors
    finally:
        subprocess.run = original_run
