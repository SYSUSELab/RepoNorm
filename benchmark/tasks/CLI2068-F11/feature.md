# Monitor the editor process lifecycle

Add an optional `monitor_process` callback to `click.edit`. The callback receives `(event_type, process_id, editor_command, timestamp)` where the event is `start`, `suspend`, `resume`, or `terminate`, the PID and command identify the launched editor session, and timestamp is a float from the wall clock. Reject a non-callable non-`None` value with `TypeError`.

Emit `start` exactly once after successful process creation and before every other event. On the fixed POSIX scoring platform, observe real stopped/running transitions and emit `suspend` and `resume` once per transition. Emit `terminate` exactly once after the process exits and after any final state event. A quickly exiting process still gets `start` then `terminate`. Callback exceptions produce `UserWarning` and do not change editor execution, later monitoring, return values, or existing nonzero-exit behavior.

Platforms without a reliable stopped-state interface may emit only `start` and `terminate`. The task requires observable event ordering and isolation, not a callback signature inspection strategy, polling interval, thread structure, or private helper.
