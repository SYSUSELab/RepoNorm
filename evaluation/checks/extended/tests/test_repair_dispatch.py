"""Offline checks for evaluator-owned, hash-bound alternative dispatch."""
import importlib.util
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
spec = importlib.util.spec_from_file_location('repair_run',HERE/'run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
sys.path.pop(0)


def setup_dispatch(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'HERE',tmp_path)
    patch = tmp_path/'alternative.patch'
    patch.write_bytes(b'delta\n')
    task = {'id':'test','reference':'gold.patch','path':'example',
            'alternative_reference':'alternative.patch','alternative_sha256':run.sha256(patch.read_bytes())}
    calls = []
    monkeypatch.setattr(run.subprocess,'run',lambda args,**kwargs:calls.append(args))
    return task,calls


def test_alternative_applies_gold_then_verified_delta(tmp_path,monkeypatch):
    task,calls = setup_dispatch(tmp_path,monkeypatch)
    run.apply_state(tmp_path/'repo',task,'alternative')
    assert len(calls) == 2
    assert calls[0][-1] == str(tmp_path/'gold.patch')
    assert calls[1][-1] == str(tmp_path/'alternative.patch')


def test_alternative_rejects_mismatched_hash_before_delta_apply(tmp_path,monkeypatch):
    task,calls = setup_dispatch(tmp_path,monkeypatch)
    task['alternative_sha256'] = 'invalid'
    with pytest.raises(ValueError,match='alternative patch hash mismatch'):
        run.apply_state(tmp_path/'repo',task,'alternative')
    assert len(calls) == 1


def test_gold_does_not_require_alternative_asset(tmp_path,monkeypatch):
    task,calls = setup_dispatch(tmp_path,monkeypatch)
    (tmp_path/'alternative.patch').unlink()
    run.apply_state(tmp_path/'repo',task,'gold')
    assert len(calls) == 1


def test_nonzero_exit_probe_changes_only_old_error_guard(tmp_path,monkeypatch):
    task,calls = setup_dispatch(tmp_path,monkeypatch)
    source = tmp_path/'runtime.py'
    source.write_text('if exit_code != 0:\n    raise RuntimeError()\n')
    task.update(runtime_path='runtime.py',runtime_probe='editor_nonzero_exit')
    run.apply_state(tmp_path,task,'bad-runtime-contract')
    assert source.read_text() == 'if False and exit_code != 0:\n    raise RuntimeError()\n'
    assert len(calls) == 1


def test_nonzero_exit_probe_fails_on_ambiguous_source(tmp_path,monkeypatch):
    task,calls = setup_dispatch(tmp_path,monkeypatch)
    source = tmp_path/'runtime.py'
    source.write_text('pass\n')
    task.update(runtime_path='runtime.py',runtime_probe='editor_nonzero_exit')
    with pytest.raises(ValueError,match='source shape changed'):
        run.apply_state(tmp_path,task,'bad-runtime-contract')
    assert source.read_text() == 'pass\n'


def test_context_probe_changes_only_existing_gate(tmp_path,monkeypatch):
    task,calls = setup_dispatch(tmp_path,monkeypatch)
    source = tmp_path/'runtime.py'
    source.write_text('use_dataset_summary=self.use_dataset_summary,\nother=True,\n')
    task.update(runtime_path='runtime.py',runtime_probe='proposer_context_switch')
    run.apply_state(tmp_path,task,'bad-runtime-contract')
    assert source.read_text() == 'use_dataset_summary=False,\nother=True,\n'
    assert len(calls) == 1


def test_context_probe_rejects_ambiguous_source(tmp_path,monkeypatch):
    task,calls = setup_dispatch(tmp_path,monkeypatch)
    source = tmp_path/'runtime.py'
    source.write_text('use_dataset_summary=self.use_dataset_summary,\n'*2)
    original = source.read_text()
    task.update(runtime_path='runtime.py',runtime_probe='proposer_context_switch')
    with pytest.raises(ValueError,match='context gate source changed'):
        run.apply_state(tmp_path,task,'bad-runtime-contract')
    assert source.read_text() == original
