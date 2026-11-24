from contextlib import contextmanager

import click


def test_CLI_N03_retry_keeps_resource_and_callback_unified_lifo():
    events = []

    @contextmanager
    def resource():
        events.append("resource-enter")
        try:
            yield
        finally:
            events.append("resource-exit")

    ctx = click.Context(click.Command("cmd"), retry_cleanup=1)
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
