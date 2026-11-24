"""Offline tests distinguish submitted API collection errors from infra errors."""
import unittest

from evaluation.author_evidence import assess_authored_tests


class AuthorEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.before = {'pkg/__init__.py': 'from .core import Old\n',
                       'pkg/core.py': 'class Old: pass\n'}
        self.after = dict(self.before, **{
            'pkg/__init__.py': 'from .core import Old, New\n',
            'pkg/core.py': 'class Old: pass\nclass New: pass\n',
            'tests/test_new.py': 'from pkg import New\ndef test_new(): assert New()\n'})
        self.dynamic = {'authored_paths': ['tests/test_new.py'],
                        'author_green': {'status': 'pass', 'counts': {'tests': 1}},
                        'author_red': {'status': 'error', 'pytest_exit_code': 2,
                                       'counts': {'errors': 1, 'failures': 0},
                                       'module_origin_check': {'status': 'pass',
                                                              'path': '/tmp/unpatched/pkg/__init__.py'}}}
        self.log = """_____________________ ERROR collecting tests/test_new.py _____________________
ImportError while importing test module '/tmp/unpatched/tests/test_new.py'.
E   ImportError: cannot import name 'New' from 'pkg' (/tmp/unpatched/pkg/__init__.py)
=========================== short test summary info ============================
ERROR tests/test_new.py
"""

    def assess(self):
        return assess_authored_tests(self.dynamic, self.before, self.after, self.log)

    def test_reexported_new_api_proves_collection_error(self):
        result = self.assess()
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(len(result['witnesses'][0]['definition_chain']), 2)

    def test_identical_reexports_are_one_provable_binding(self):
        self.after['pkg/__init__.py'] += 'from .core import New\n'
        self.assertEqual(self.assess()['status'], 'pass')

    def test_conflicting_reexports_are_not_provable(self):
        self.after['pkg/__init__.py'] += 'from .other import New\n'
        self.after['pkg/other.py'] = 'class New: pass\n'
        self.assertEqual(self.assess()['status'], 'error')

    def test_reexport_and_definition_are_not_deduplicated(self):
        self.after['pkg/__init__.py'] += 'class New: pass\n'
        self.assertEqual(self.assess()['status'], 'error')

    def test_missing_dependency_is_not_new_api(self):
        self.log = self.log.replace("ImportError: cannot import name 'New' from 'pkg' (/tmp/unpatched/pkg/__init__.py)",
                                    "ModuleNotFoundError: No module named 'dependency'")
        self.assertEqual(self.assess()['status'], 'error')

    def test_existing_api_is_not_new(self):
        self.before['pkg/__init__.py'] += 'from .core import New\n'
        self.assertEqual(self.assess()['status'], 'error')

    def test_dynamic_and_star_exports_prevent_absence_proof(self):
        for source in ('from .core import *\n', 'def __getattr__(name): return 1\n',
                       "exec('New = 1')\n", "if False:\n    New = 1\n"):
            with self.subTest(source=source):
                self.before['pkg/__init__.py'] = source
                self.assertEqual(self.assess()['status'], 'error')

    def test_rebound_public_name_is_not_a_proven_definition(self):
        self.after['pkg/__init__.py'] += 'New = None\n'
        self.assertEqual(self.assess()['status'], 'error')

    def test_missing_definition_cannot_be_proven_by_import_alone(self):
        self.after['pkg/core.py'] = 'class Old: pass\n'
        self.assertEqual(self.assess()['status'], 'error')

    def test_error_must_match_authored_test_import(self):
        self.after['tests/test_new.py'] = 'from pkg import Old\n'
        self.assertEqual(self.assess()['status'], 'error')

    def test_source_origin_must_match(self):
        self.dynamic['author_red']['module_origin_check']['path'] = '/usr/lib/pkg/__init__.py'
        self.assertEqual(self.assess()['status'], 'error')

    def test_all_collection_errors_must_be_accounted_for(self):
        self.dynamic['author_red']['counts']['errors'] = 2
        self.assertEqual(self.assess()['status'], 'error')

    def test_timeout_never_passes(self):
        self.dynamic['author_red'] = {'status': 'error', 'reason': 'timeout'}
        self.assertEqual(self.assess()['status'], 'error')

    def test_ordinary_assertion_red_passes_but_mixed_error_does_not(self):
        self.dynamic['author_red'] = {'status': 'fail', 'counts': {'failures': 1, 'errors': 0}}
        self.assertEqual(self.assess()['status'], 'pass')
        self.dynamic['author_red']['counts']['errors'] = 1
        self.assertEqual(self.assess()['status'], 'error')

    def test_failed_or_unexecuted_green_cannot_pass(self):
        for green, expected in (({'status': 'fail'}, 'fail'), ({'status': 'error'}, 'error'),
                                ({'status': 'not_run'}, 'error'),
                                ({'status': 'pass', 'counts': {'tests': 0}}, 'error')):
            with self.subTest(green=green):
                self.dynamic['author_green'] = green
                self.assertEqual(self.assess()['status'], expected)

    def test_private_import_not_accepted(self):
        self.log = self.log.replace("'New'", "'_New'")
        self.assertEqual(self.assess()['status'], 'error')

    def test_src_layout_and_alias_are_supported(self):
        self.before = {'src/' + k: v for k, v in self.before.items()}
        self.after = {('src/' + k if k.startswith('pkg/') else k): v for k, v in self.after.items()}
        self.after['tests/test_new.py'] = 'from pkg import New as Tested\n'
        self.log = self.log.replace('/tmp/unpatched/pkg/', '/tmp/unpatched/src/pkg/')
        self.dynamic['author_red']['module_origin_check']['path'] = '/tmp/unpatched/src/pkg/__init__.py'
        self.assertEqual(self.assess()['status'], 'pass')

    def test_cycle_and_parse_failure_are_not_pass(self):
        self.after['pkg/core.py'] = 'from . import New\n'
        self.assertEqual(self.assess()['status'], 'error')
        self.after['pkg/core.py'] = 'bad syntax!'
        self.assertEqual(self.assess()['status'], 'error')

    def test_no_tests_and_no_red_difference_fail(self):
        self.dynamic['author_red'] = {'status': 'pass'}
        self.assertEqual(self.assess()['status'], 'fail')
        self.dynamic['authored_paths'] = []
        self.assertEqual(self.assess()['status'], 'fail')


if __name__ == '__main__':
    unittest.main()
