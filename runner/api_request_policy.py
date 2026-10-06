"""Bound API attempts in the isolated Linux task worker, including compaction."""
from functools import wraps
import os
import signal
import threading


class APIAttemptDeadline(TimeoutError):
    """The entire synchronous completion exceeded its wall-clock allowance."""


def configure_request_policy(config: dict, completion_module=None) -> None:
    if 'api_max_attempts' not in config:
        return
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('API deadline requires the isolated worker main thread')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('API deadline cannot replace an active timer')
    os.environ['MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT'] = str(config['api_max_attempts'])
    if completion_module is None:
        import litellm as completion_module
    if getattr(completion_module, '_n4c_api_deadline', None) is not None:
        raise RuntimeError('API deadline already configured for this worker')
    original = completion_module.completion
    seconds = config['api_attempt_deadline_seconds']

    @wraps(original)
    def completion(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('API completion left the worker main thread')
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError('API deadline cannot replace an active timer')
        previous = signal.getsignal(signal.SIGALRM)

        def expire(signum, frame):
            raise APIAttemptDeadline(f'API completion exceeded {seconds}s wall deadline')

        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return original(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    completion_module.completion = completion
    completion_module._n4c_api_deadline = seconds
