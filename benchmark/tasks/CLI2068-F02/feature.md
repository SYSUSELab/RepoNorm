**Title**: Add an Editor Session Timeout to `click.edit`

Add an optional `timeout` parameter to `click.edit` so callers can bound how long an editor session may run.

## Public API

```python
click.edit(
    text=None,
    editor=None,
    env=None,
    require_save=True,
    extension=".txt",
    filename=None,
    timeout: int | None = None,
)
```

`timeout=None` preserves the existing behavior and waits indefinitely. Otherwise, `timeout` must be a positive integer number of seconds. Zero, negative integers, booleans, floats, strings, and other values raise `ValueError` before an editor is started.

The option applies equally when editing temporary text and when `filename` is supplied directly. All existing return-value and save-detection behavior remains unchanged.

## Timeout behavior

The editor is allowed to complete normally for at most `timeout` seconds. If it does not:

1. Click requests termination of the editor session.
2. Click waits briefly for graceful shutdown.
3. If the session is still running, Click forcefully stops it and reaps the process before returning control.
4. Click raises `ClickException` with this message:

```
{editor}: Editing timed out after {timeout} seconds
```

The count is interpolated without pluralization, so a one-second timeout ends in `after 1 seconds`.

On platforms where the configured editor is launched through a shell, timeout cleanup must also prevent child processes belonging to that launched editor session from continuing after `click.edit` returns. Equivalent platform-appropriate process management strategies are accepted; no particular `subprocess` option or private helper is required.

A nonzero editor exit before the timeout continues to use Click's existing `{editor}: Editing failed` error. Operating-system launch failures continue to use the existing editing-failed error path.
