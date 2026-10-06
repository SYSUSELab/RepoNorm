**Title**: Retry transient failures of Context cleanup callbacks

Add `retry_cleanup` to Click Contexts. It is the number of additional attempts
after a cleanup callback's first attempt. A root Context defaults to zero. A
child inherits an omitted value; explicit zero disables retries for that child.

Each callback registered with `call_on_close()` has its own retry budget. Retry
ordinary exceptions with exponential backoff and stop retrying after success.
Expose `retry_cleanup_backoff`, the initial delay in seconds: it defaults to
0.1 at the root and inherits from the parent when omitted or `None`. An explicit
value overrides inheritance and must be a finite positive integer or float
(not a boolean); reject invalid values with `ValueError`. Before the first retry
wait this delay, then double it for each subsequent retry of that callback.
Each callback starts its own delay sequence. Do not delay before the first
attempt, after success, or after the final exhausted attempt. Timing naturally
allows ordinary operating-system scheduling overhead; no specific wait API
or internal implementation is required. After exhaustion propagate
the last failure exception, retaining its identity and details. The remaining
callbacks must still receive their own cleanup attempts through the normal
cleanup lifecycle, and callbacks that already succeeded must not run again.

Keep `call_on_close()` usable as a decorator with its existing return behavior.
Do not retry `KeyboardInterrupt` or `SystemExit`. Resources registered through
`with_resource()` retain their existing context-manager exit semantics; this
feature must not try to re-enter or repeatedly exit a generator context manager.
Existing command and shell-completion behavior must remain compatible.
