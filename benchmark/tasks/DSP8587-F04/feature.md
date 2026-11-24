**Title**: Automatically End an Idle Stream

`StreamListener` accepts
`idle_timeout_s: float | None = None` and
`clock: Callable[[], float] | None = None`.

`None` disables idle termination. A configured timeout must be a positive,
finite `int` or `float` and must not be a boolean; otherwise construction raises
`ValueError("idle_timeout_s must be a positive finite number or None")`.
The optional clock supplies monotonic seconds and defaults to
`time.monotonic`. A supplied clock and `tick(now_ts=...)` use the same time
domain.

After the configured field has started, every non-empty received model chunk
refreshes its last-activity time. `tick(now_ts: float | None = None)` checks
that state in the clock's monotonic time domain. Before the timeout it returns
`None`. At or after the timeout it returns one `StreamResponse` with an empty
`chunk` and clears the complete per-stream boundary/cache state, so subsequent
ticks do not emit duplicates.

`streamify()` automatically pumps these checks while upstream is stalled;
callers do not need to call `tick()`. The timeout response is yielded, the
stalled upstream program is cancelled, and that async stream invocation
finishes without requiring a normal final `Prediction`. If multiple configured
listeners expire together, the first listener in the supplied list produces
the single timeout response and terminates the invocation.

This standalone feature does not add `is_last_chunk` or `on_chunk`. When those
separate features are combined, their final-flag/callback contracts apply to
the timeout response.
