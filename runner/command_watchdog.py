"""Container-side command deadline and process-group cleanup; stdlib only."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def live_group(group: int) -> bool:
    # Zombies have stopped executing and cannot keep a command alive.
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def run(command: str, interpreter: list[str], timeout: float, grace: float) -> dict:
    started = time.monotonic()
    try:
        process = subprocess.Popen([*interpreter, command], start_new_session=True,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
    except (OSError, ValueError) as exc:
        # Invalid command arguments fail before a child process exists. Report
        # them as command feedback so the agent can correct the tool call.
        return {
            'output': f'Command was not started: {type(exc).__name__}: {exc}\n',
            'returncode': 126,
            'timed_out': False,
            'cleanup_status': 'not_required',
            'signals': [],
            'elapsed_seconds': time.monotonic() - started,
            'process_started': False,
            'error_type': type(exc).__name__,
        }
    timed_out = False
    signals = []

    def send(sig):
        try:
            os.killpg(process.pid, sig)
            signals.append(signal.Signals(sig).name)
        except ProcessLookupError:
            pass

    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        send(signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=grace / 2)
        except subprocess.TimeoutExpired:
            send(signal.SIGKILL)
            output, _ = process.communicate(timeout=grace / 4)
    # A normal return does not terminate intentionally backgrounded work.
    if not timed_out:
        return {'output': output.decode('utf-8', errors='replace'),
                'returncode': process.returncode, 'timed_out': False,
                'cleanup_status': 'not_required', 'signals': [],
                'elapsed_seconds': time.monotonic() - started,
                'process_started': True, 'error_type': None}
    if live_group(process.pid):
        send(signal.SIGKILL)
    deadline = time.monotonic() + grace / 4
    while live_group(process.pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    cleaned = not live_group(process.pid)
    return {'output': output.decode('utf-8', errors='replace'),
            'returncode': 124 if timed_out else process.returncode,
            'timed_out': timed_out, 'cleanup_status': 'complete' if cleaned else 'failed',
            'signals': signals, 'elapsed_seconds': time.monotonic() - started,
            'process_started': True, 'error_type': None}


if __name__ == '__main__':
    request = json.loads(sys.argv[1])
    print(json.dumps(run(**request)))
