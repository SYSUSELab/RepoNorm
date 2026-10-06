"""Host-side bounded commands for the four-arm integration, not extraction."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from four_arm_common import atomic_json, digest, utc

WATCHDOG = Path(__file__).with_name('command_watchdog.py').read_text()


class CommandInfrastructureError(RuntimeError):
    pass


class BoundedCommands:
    """Mixin placed before the vendored Docker environment."""
    command_timeout: int
    command_cleanup_grace: int
    command_audit_directory: Path

    def execute(self, action: dict, cwd: str = '', *, timeout: int | None = None) -> dict:
        if getattr(self, 'command_infrastructure_failed', False):
            raise CommandInfrastructureError('Task command supervisor is unavailable')
        limit = min(timeout or self.command_timeout, self.command_timeout)
        command = action.get('command', '')
        request = {'command': command, 'interpreter': self.config.interpreter,
                   'timeout': limit, 'grace': self.command_cleanup_grace}
        argv = [self.config.executable, 'exec', '-w', cwd or self.config.cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                argv.extend(['-e', f'{key}={value}'])
        for key, value in self.config.env.items():
            argv.extend(['-e', f'{key}={value}'])
        argv.extend([self.container_id, 'python', '-c', WATCHDOG, json.dumps(request)])
        index = getattr(self, '_command_index', 0) + 1
        self._command_index = index
        receipt_path = self.command_audit_directory / f'{index:05}.json'
        receipt = {'index': index, 'command_digest': digest(command),
                   'started_at': utc(), 'ended_at': None, 'elapsed_seconds': None,
                   'timeout_seconds': limit, 'cleanup_grace_seconds': self.command_cleanup_grace,
                   'status': 'running', 'cleanup_status': 'not_run'}
        atomic_json(receipt_path, receipt)
        started = time.monotonic()
        try:
            completed = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8',
                                       errors='replace', stdin=subprocess.DEVNULL,
                                       timeout=limit + self.command_cleanup_grace)
            if completed.returncode:
                raise CommandInfrastructureError('Container command supervisor did not return successfully')
            result = json.loads(completed.stdout)
            if result['cleanup_status'] != ('complete' if result['timed_out'] else 'not_required'):
                raise CommandInfrastructureError('Container command process-group cleanup failed')
            receipt.update({key: result[key] for key in ('returncode', 'timed_out', 'cleanup_status', 'signals')})
            receipt['process_started'] = result.get('process_started', True)
            receipt['command_error_type'] = result.get('error_type')
            receipt['status'] = ('not_started' if not receipt['process_started'] else
                                 'timed_out' if result['timed_out'] else 'complete')
            output = {'output': result['output'], 'returncode': result['returncode'], 'exception_info': ''}
            if result['timed_out']:
                note = (f'Command exceeded {limit} seconds and was terminated with its process group. '
                        'Check for blocking operations or split the command before retrying.')
                output['output'] += '\n' + note
                output['exception_info'] = note
        except Exception as exc:
            self.command_infrastructure_failed = True
            receipt.update(status='infrastructure_error', cleanup_status='unconfirmed',
                           error_type=type(exc).__name__)
            raise CommandInfrastructureError('Command supervision failed; stop this task and clean up its container') from exc
        finally:
            receipt.update(ended_at=utc(), elapsed_seconds=time.monotonic() - started)
            atomic_json(receipt_path, receipt)
        self._check_finished(output)
        return output
