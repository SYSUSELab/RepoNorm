# Editor process priority

Add `process_priority` to `click.edit`, accepting `None` or one of `low`, `below_normal`, `normal`, `above_normal`, and `high`. Reject every other value with `ValueError` before process launch. `None` preserves existing process creation.

When a level is requested, apply the closest supported CPU scheduling priority to the launched editor process. The five names are ordered from least to most favored; exact numeric values are platform-specific. If the platform lacks priority support or permission is denied, continue the edit at the platform default rather than failing. Preserve configured editor arguments, filename boundaries, custom environment, and exit behavior. The task does not require `renice`, `wmic`, a post-launch PID update, or any private adapter.
