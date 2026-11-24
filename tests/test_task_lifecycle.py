"""Offline cleanup/recovery checks; Docker is mocked."""
import builtins
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runner'))
from four_arm_common import atomic_json
from task_lifecycle import cleanup_solver
from four_arm_worker import execute


class LifecycleTests(unittest.TestCase):
    @patch('task_lifecycle.subprocess.run')
    def test_rm_timeout_uses_final_absence_query(self, run):
        run.side_effect = [subprocess.CompletedProcess([], 0, 'abc\n'),
                           subprocess.TimeoutExpired('docker rm', 180),
                           subprocess.CompletedProcess([], 0, '')]
        receipt = cleanup_solver('frozen', 'task1')
        self.assertEqual(receipt['status'], 'complete')
        self.assertEqual(receipt['removal_errors'][0]['error_type'], 'TimeoutExpired')

    @patch('task_lifecycle.subprocess.run')
    def test_removes_only_label_matched_ids_and_confirms_absence(self, run):
        run.side_effect = [subprocess.CompletedProcess([], 0, 'abc\ndef\n'),
                           subprocess.CompletedProcess([], 0, ''),
                           subprocess.CompletedProcess([], 0, ''),
                           subprocess.CompletedProcess([], 0, '')]
        receipt = cleanup_solver('frozen', 'task1')
        self.assertEqual(receipt['status'], 'complete')
        query = run.call_args_list[0].args[0]
        self.assertIn('label=n4c.run=frozen', query)
        self.assertIn('label=n4c.task=task1', query)
        self.assertNotIn('label=n4c.role=solver', query)  # solver and evaluator
        self.assertEqual(run.call_args_list[1].args[0], ['docker', 'rm', '-f', 'abc'])
        self.assertEqual(run.call_args_list[2].args[0], ['docker', 'rm', '-f', 'def'])
        self.assertEqual(run.call_args_list[3].args[0], query)

    def test_failures_are_receipted_without_raising(self):
        for error in (subprocess.TimeoutExpired('docker', 180),
                      subprocess.CalledProcessError(1, 'docker')):
            with self.subTest(error=type(error).__name__), patch('task_lifecycle.subprocess.run', side_effect=error):
                self.assertEqual(cleanup_solver('frozen', 'task1')['status'], 'unconfirmed')

    @patch('task_lifecycle.subprocess.run')
    def test_remaining_container_is_not_success(self, run):
        run.side_effect = [subprocess.CompletedProcess([], 0, ''),
                           subprocess.CompletedProcess([], 0, 'still-present')]
        self.assertEqual(cleanup_solver('frozen', 'task1')['status'], 'unconfirmed')

    def test_worker_interrupt_writes_terminal_result_after_cleanup(self):
        original_import = builtins.__import__

        def interrupted_import(name, *args, **kwargs):
            if name == 'environment':
                raise KeyboardInterrupt('Task worker received SIGTERM')
            return original_import(name, *args, **kwargs)

        spec = {'task_key': 'task1', 'run_identity': 'frozen', 'arm': 'raw',
                'config': {'dataset': 'engineering-v5', 'global_concurrency': 20},
                'feature': 1, 'pool': {'repository': 'repo', 'task_id': 1}}
        with tempfile.TemporaryDirectory() as temporary, \
                patch('four_arm_worker.global_slot', return_value=nullcontext(0)), \
                patch('four_arm_worker.signal.signal'), \
                patch('task_lifecycle.cleanup_solver', return_value={'status': 'complete'}) as cleanup, \
                patch('builtins.__import__', side_effect=interrupted_import):
            directory = Path(temporary)
            result = execute(spec, directory, mock=True)
            self.assertEqual(result['outcome'], 'interrupted')
            self.assertIn('KeyboardInterrupt', result['error'])
            self.assertEqual(result['container_cleanup']['status'], 'complete')
            self.assertTrue((directory / 'result.json').is_file())
            cleanup.assert_called_once_with('frozen', 'task1')



if __name__ == '__main__':
    unittest.main()
