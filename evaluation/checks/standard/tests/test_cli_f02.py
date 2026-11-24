import click


def test_CLI_N03_on_exit_preserves_registered_resource_lifo_order():
    events = []

    class Resource:
        def __enter__(self):
            events.append("resource-enter")
            return self

        def __exit__(self, *exc_info):
            events.append("resource-exit")

    ctx = click.Context(
        click.Command("cmd"), on_exit=lambda current: events.append("on-exit")
    )
    ctx.call_on_close(lambda: events.append("registered-first"))
    ctx.with_resource(Resource())
    ctx.call_on_close(lambda: events.append("registered-last"))

    with ctx:
        pass

    assert [event for event in events if event != "on-exit"] == [
        "resource-enter",
        "registered-last",
        "resource-exit",
        "registered-first",
    ]
    assert events.count("on-exit") == 1
