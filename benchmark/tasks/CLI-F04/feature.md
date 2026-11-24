**Title**: Bound cleanup waiting while supporting safe Context reuse

Add optional `cleanup_timeout` to Click Contexts. A root's default `None` preserves
synchronous cleanup; an omitted or `None` child timeout inherits its parent's setting. An explicit
numeric timeout must be positive, otherwise construction raises `ValueError`.

When a timeout is configured, `close()` waits at most that duration for cleanup
(allowing normal scheduling overhead). Cleanup completed before the deadline
keeps its usual exception behavior. If cleanup is still running, return control
and issue a warning; do not claim to forcibly terminate Python cleanup code.
An error raised after the caller has stopped waiting must be reported by a
warning identifying the error, not silently lost or raised by a later unrelated `close()` call.
Warnings may use Python's warning mechanism or a diagnostic on stderr.

Each timed close owns the resources and callbacks already registered when it
starts. Registrations made after it starts, including from a cleanup callback,
belong to the next close. A timed-out cleanup must neither consume nor discard
these later registrations. Repeated close must not restart an earlier cleanup.
Preserve normal cleanup order within each batch, and clean each registration at
most once. New batches can be closed while an old timed-out batch finishes.
This does not require inheriting thread-local current-context state in workers.
