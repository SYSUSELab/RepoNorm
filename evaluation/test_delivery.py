"""Tests for the total coarse authored-test delivery decision."""
import unittest

from evaluation.delivery import assess_authored_test_delivery


class AuthoredTestDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.dynamic = {
            'authored_paths': ['tests/test_new.py'],
            'authored_selection': {
                'status': 'pass',
                'changed_test_paths': ['tests/test_new.py'],
                'nodeids': ['tests/test_new.py::test_new'],
            },
            'author_green': {'status': 'pass', 'counts': {'tests': 1}},
            'author_red': {'status': 'error', 'reason': 'collection_failed'},
        }

    def test_green_delivery_passes_without_red_gate(self):
        result = assess_authored_test_delivery(self.dynamic)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['protocol'], 'authored-test-delivery-1.0')

    def test_missing_or_unrecognized_tests_fail(self):
        for change in (
            {'authored_paths': []},
            {'authored_selection': {'status': 'fail'}},
            {'authored_selection': {'status': 'pass', 'changed_test_paths': [], 'nodeids': []}},
        ):
            with self.subTest(change=change):
                dynamic = dict(self.dynamic)
                dynamic.update(change)
                self.assertEqual(assess_authored_test_delivery(dynamic)['status'], 'fail')

    def test_green_failure_error_timeout_or_zero_tests_fail(self):
        for green in (
            {'status': 'fail', 'counts': {'tests': 1}},
            {'status': 'error', 'reason': 'subprocess_timeout'},
            {'status': 'not_run'},
            {'status': 'pass', 'counts': {'tests': 0}},
        ):
            with self.subTest(green=green):
                dynamic = dict(self.dynamic, author_green=green)
                self.assertEqual(assess_authored_test_delivery(dynamic)['status'], 'fail')

    def test_selection_must_be_bound_to_changed_test(self):
        self.dynamic['authored_selection']['changed_test_paths'] = ['tests/test_other.py']
        self.assertEqual(assess_authored_test_delivery(self.dynamic)['status'], 'fail')


if __name__ == '__main__':
    unittest.main()
