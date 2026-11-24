"""Offline regression: real vendored retry loops, fake synchronous SDK calls."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'runner'),
                str(Path(__file__).resolve().parents[1] / 'CooperBench/src')]
from api_request_policy import APIAttemptDeadline, configure_request_policy
from dashscope_profile import BOUNDED_PROFILE, BoundedDashScopeCodingProfile, bounded_config, frozen_config
from four_arm_common import model_kwargs
from task_telemetry import install_hooks, task_telemetry




@pytest.mark.parametrize('kind', ['solver', 'compaction'])
def test_actual_agent_retry_stops_after_two_and_records_failures(monkeypatch, kind):
    from cooperbench.agents.mini_swe_agent_v2.models import litellm_model
    attempts = []
    def failing(**kwargs):
        attempts.append(kwargs)
        raise TimeoutError('offline SDK timeout')
    module = SimpleNamespace(completion=failing)
    monkeypatch.setenv('MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT', '99')
    configure_request_policy(bounded_config(), module)
    assert os.environ['MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT'] == '2'
    # Use the real query and summarize_context retry loops without loading a model.
    class Model(litellm_model.LitellmModel):
        def __init__(self):
            self.abort_exceptions = []
            self.config = SimpleNamespace(model_name="offline", model_kwargs=model_kwargs(bounded_config()))
        def _prepare_messages_for_api(self, messages):
            return messages
        def _serialize_transcript_for_summary(self, messages):
            return 'offline transcript'
        def _query(self, messages, **kwargs):
            return module.completion(**model_kwargs(bounded_config()))
    install_hooks(Model, module)
    original_retry = litellm_model.retry
    def retry_no_wait(**kwargs):
        value = original_retry(**kwargs)
        value.sleep = lambda seconds: None
        return value
    monkeypatch.setattr(litellm_model, 'retry', retry_no_wait)
    monkeypatch.setattr(litellm_model, 'litellm', module)
    with task_telemetry() as state, pytest.raises(TimeoutError):
        if kind == 'solver':
            Model().query([])
        else:
            Model().summarize_context([], 'summarize')
    assert len(attempts) == 2
    receipt = state.receipt()
    assert receipt['sdk_completion_attempts'] == 2
    assert receipt['sdk_completion_retries'] == 1
    assert receipt['failed'] == 1
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_wall_deadline_interrupts_blocking_attempt_and_restores_handler(monkeypatch):
    monkeypatch.setenv('MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT', '10')
    module = SimpleNamespace(completion=lambda **kw: time.sleep(2))
    previous = signal.getsignal(signal.SIGALRM)
    configure_request_policy(bounded_config() | {'api_attempt_deadline_seconds': 0.02}, module)
    started = time.monotonic()
    with pytest.raises(APIAttemptDeadline):
        module.completion()
    assert time.monotonic() - started < 1
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_request_receipt_rejects_removed_sdk_retry_bound():
    from task_telemetry import _wire_record
    wire = {'model': bounded_config()['wire_model'], **model_kwargs(bounded_config())}
    assert _wire_record(wire, wire)['status'] == 'matched'
    for key in ('timeout', 'max_retries', 'num_retries'):
        altered = dict(wire)
        del altered[key]
        assert f'sdk_request_{key}_mismatch' in _wire_record(altered, wire)['violations']
