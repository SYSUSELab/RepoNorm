from contextlib import contextmanager

import click


def test_CLI_N03_state_restore_leaves_unified_cleanup_stack_untouched():
    events = []

    @contextmanager
    def resource():
        events.append("resource-enter")
        try:
            yield
        finally:
            events.append("resource-exit")

    ctx = click.Context(click.Command("cmd"))
    snapshot = ctx.save_state()
    ctx.call_on_close(lambda: events.append("registered-first"))
    ctx.with_resource(resource())
    ctx.call_on_close(lambda: events.append("registered-last"))
    ctx.restore_state(snapshot)
    ctx.close()

    assert events == [
        "resource-enter",
        "registered-last",
        "resource-exit",
        "registered-first",
    ]
