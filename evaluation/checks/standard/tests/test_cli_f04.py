from contextlib import contextmanager

import click


def test_CLI_N03_timed_cleanup_keeps_unified_lifo_order():
    events = []

    @contextmanager
    def resource():
        events.append("resource-enter")
        try:
            yield
        finally:
            events.append("resource-exit")

    ctx = click.Context(click.Command("cmd"), cleanup_timeout=1)
    ctx.call_on_close(lambda: events.append("registered-first"))
    ctx.with_resource(resource())
    ctx.call_on_close(lambda: events.append("registered-last"))
    ctx.close()

    assert events == [
        "resource-enter",
        "registered-last",
        "resource-exit",
        "registered-first",
    ]
