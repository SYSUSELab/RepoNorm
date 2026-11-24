from contextlib import contextmanager

import pytest

import click
from click.shell_completion import ShellComplete


@pytest.mark.parametrize("chain", [False, True])
def test_CLI_N01_validated_completion_does_not_execute_commands(chain):
    calls = []
    group = click.Group("root", chain=chain, callback=lambda **kw: calls.append("root"))
    group.add_command(click.Command("sub", callback=lambda **kw: calls.append("sub")))
    complete = ShellComplete(group, {"validate_nesting": True}, "root", "_COMPLETE")
    complete.get_completions(["sub"], "")
    assert calls == []


def test_CLI_N02_validated_completion_does_not_prompt(monkeypatch):
    prompts = []
    monkeypatch.setattr(click.core, "prompt", lambda *a, **kw: prompts.append((a, kw)))
    cmd = click.Command(
        "root", params=[click.Option(["--name"], prompt=True, required=True)]
    )
    complete = ShellComplete(cmd, {"validate_nesting": True}, "root", "_COMPLETE")
    complete.get_completions(["--name"], "")
    assert prompts == []


def test_CLI_N03_rejected_parent_close_keeps_unified_lifo_cleanup():
    events = []

    @contextmanager
    def resource():
        events.append("enter")
        try:
            yield
        finally:
            events.append("resource-exit")

    parent = click.Context(click.Command("parent"), validate_nesting=True)
    parent.call_on_close(lambda: events.append("first"))
    parent.with_resource(resource())
    parent.call_on_close(lambda: events.append("last"))
    child = click.Context(click.Command("child"), parent=parent)
    with pytest.raises(RuntimeError):
        parent.close()
    assert events == ["enter"]
    child.close()
    parent.close()
    assert events == ["enter", "last", "resource-exit", "first"]
