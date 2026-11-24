from contextlib import contextmanager

import click


def test_CLI_N03_resource_limit_keeps_unified_lifo_cleanup():
    events = []

    @contextmanager
    def resource():
        events.append("resource-enter")
        try:
            yield
        finally:
            events.append("resource-exit")

    ctx = click.Context(click.Command("cmd"), max_resources=1)
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
