"""Offline deadline, process-tree cleanup, receipts and agent feedback tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runner'))
from command_watchdog import run, live_group
from task_commands import BoundedCommands, CommandInfrastructureError


def python_command(code):
    return shlex.join([sys.executable, '-c', code])


class WatchdogTests(unittest.TestCase):
    def test_normal_and_nonzero_output(self):
        for code in (0, 7):
            result = run(f'echo retained; exit {code}', ['bash', '-c'], 2, 1)
            self.assertEqual(result['returncode'], code)
            self.assertEqual(result['output'], 'retained\n')
            self.assertFalse(result['timed_out'])
            self.assertEqual(result['cleanup_status'], 'not_required')
            self.assertTrue(result['process_started'])

    def test_invalid_prelaunch_argument_is_recoverable(self):
        result = run('printf before\0after', ['bash', '-c'], 2, 1)
        self.assertEqual(result['returncode'], 126)
        self.assertIn('embedded null byte', result['output'])
        self.assertFalse(result['timed_out'])
        self.assertFalse(result['process_started'])
        self.assertEqual(result['cleanup_status'], 'not_required')
        self.assertEqual(result['error_type'], 'ValueError')

    def test_deadlock_terminated_and_output_retained(self):
        result = run(python_command("import threading; print('before', flush=True); threading.Barrier(3).wait()"),
                     ['bash', '-c'], 0.2, 1)
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['returncode'], 124)
        self.assertIn('before', result['output'])
        self.assertEqual(result['cleanup_status'], 'complete')

    def test_term_ignoring_descendants_are_killed(self):
        code = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); os.fork(); print(os.getpgrp(),flush=True); time.sleep(30)"
        result = run(python_command(code), ['bash', '-c'], 0.2, 1)
        self.assertIn('SIGKILL', result['signals'])
        self.assertEqual(result['cleanup_status'], 'complete')
        for group in result['output'].splitlines():
            self.assertFalse(live_group(int(group)))
        self.assertLess(result['elapsed_seconds'], 2)

    def test_normal_background_work_is_preserved(self):
        result = run('sleep 30 >/dev/null 2>&1 & echo $!', ['bash', '-c'], 2, 1)
        child = int(result['output'].strip())
        try:
            os.kill(child, 0)
            self.assertFalse(result['timed_out'])
            self.assertEqual(result['cleanup_status'], 'not_required')
        finally:
            os.kill(child, 9)


class FakeEnvironment(BoundedCommands):
    def __init__(self, path):
        self.command_timeout = 1
        self.command_cleanup_grace = 10
        self.command_audit_directory = path
        self.container_id = 'container'
        self.config = SimpleNamespace(executable='docker', cwd='/', forward_env=[], env={}, interpreter=['bash', '-c'])
        self.finished = []

    def _check_finished(self, output):
        self.finished.append(output)


class HostTests(unittest.TestCase):
    def test_timeout_feedback_receipt_and_following_command(self):
        actual_run = subprocess.run

        def local_supervisor(argv, **kwargs):
            i = argv.index('container')
            return actual_run([sys.executable, *argv[i + 2:]], **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch('task_commands.subprocess.run', side_effect=local_supervisor):
            env = FakeEnvironment(Path(tmp))
            result = env.execute({'command': 'echo before; sleep 30'}, timeout=0.1)
            self.assertEqual(result['returncode'], 124)
            self.assertIn('Check for blocking operations', result['output'])
            self.assertIn('before', result['output'])
            receipt = json.loads((Path(tmp) / '00001.json').read_text())
            self.assertEqual(receipt['status'], 'timed_out')
            self.assertEqual(receipt['cleanup_status'], 'complete')
            self.assertEqual(env.execute({'command': 'echo recovered'})['returncode'], 0)
            self.assertEqual(len(env.finished), 2)

    def test_prelaunch_failure_is_agent_feedback_and_does_not_poison_environment(self):
        actual_run = subprocess.run

        def local_supervisor(argv, **kwargs):
            i = argv.index('container')
            return actual_run([sys.executable, *argv[i + 2:]], **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch(
            'task_commands.subprocess.run', side_effect=local_supervisor
        ):
            env = FakeEnvironment(Path(tmp))
            result = env.execute({'command': 'printf before\0after'})
            self.assertEqual(result['returncode'], 126)
            self.assertIn('embedded null byte', result['output'])
            self.assertFalse(getattr(env, 'command_infrastructure_failed', False))
            receipt = json.loads((Path(tmp) / '00001.json').read_text())
            self.assertEqual(receipt['status'], 'not_started')
            self.assertFalse(receipt['process_started'])
            self.assertEqual(receipt['cleanup_status'], 'not_required')
            self.assertEqual(receipt['command_error_type'], 'ValueError')
            self.assertEqual(env.execute({'command': 'echo recovered'})['returncode'], 0)
            self.assertEqual(len(env.finished), 2)

    def test_missing_supervisor_receipt_stops_future_commands(self):
        with tempfile.TemporaryDirectory() as tmp, patch('task_commands.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='invalid')) as call:
            env = FakeEnvironment(Path(tmp))
            for _ in range(2):
                with self.assertRaises(CommandInfrastructureError):
                    env.execute({'command': 'echo test'})
            self.assertEqual(call.call_count, 1)
            self.assertEqual(json.loads((Path(tmp) / '00001.json').read_text())['status'], 'infrastructure_error')

    def test_host_transport_timeout_is_not_recoverable_command_timeout(self):
        with tempfile.TemporaryDirectory() as tmp, patch('task_commands.subprocess.run', side_effect=subprocess.TimeoutExpired('docker', 2)):
            env = FakeEnvironment(Path(tmp))
            with self.assertRaises(CommandInfrastructureError):
                env.execute({'command': 'sleep 30'})
            self.assertEqual(json.loads((Path(tmp) / '00001.json').read_text())['cleanup_status'], 'unconfirmed')


if __name__ == '__main__':
    unittest.main()
