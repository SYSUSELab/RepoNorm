from contextlib import contextmanager
import pytest
import click
from click.shell_completion import ShellComplete, CompletionItem


@pytest.mark.parametrize('chain', [False, True])
def test_CLI_N01_no_command_execution(chain):
    calls = []
    group = click.Group('root', chain=chain, callback=lambda **kw: calls.append('root'))
    group.add_command(click.Command('sub', callback=lambda **kw: calls.append('sub')))
    complete = ShellComplete(group, {}, 'root', '_TEST_COMPLETE')
    complete.get_completions(['sub'], '')
    assert calls == []


def test_CLI_N02_no_interactive_prompt(monkeypatch):
    prompts = []

    def prompt(*args, **kwargs):
        prompts.append((args, kwargs))
        return 'fallback'

    monkeypatch.setattr(click.core, 'prompt', prompt)
    cmd = click.Command('root', params=[click.Option(['--name'], prompt=True, required=True)])
    complete = ShellComplete(cmd, {}, 'root', '_TEST_COMPLETE')
    complete.get_completions([], '--n')
    complete.get_completions(['--name'], '')
    assert prompts == []


@pytest.mark.parametrize('raise_in_completion', [False, True])
def test_CLI_N03_registered_cleanup_order(raise_in_completion):
    events = []

    @contextmanager
    def resource():
        events.append('enter')
        try:
            yield object()
        finally:
            events.append('resource-exit')

    def configure(ctx, param, value):
        # Parameter callbacks DO run during resilient parsing: they are not
        # command callbacks. Do not encode the upstream doc's broad wording.
        ctx.call_on_close(lambda: events.append('first'))
        ctx.with_resource(resource())
        ctx.call_on_close(lambda: events.append('last'))
        return value

    def candidates(ctx, param, incomplete):
        assert events == ['enter']
        if raise_in_completion:
            raise LookupError('completion failed')
        return [CompletionItem('ok')]

    cmd = click.Command('root', params=[
        click.Option(['--setup'], callback=configure),
        click.Argument(['value'], shell_complete=candidates),
    ])
    complete = ShellComplete(cmd, {}, 'root', '_TEST_COMPLETE')
    if raise_in_completion:
        with pytest.raises(LookupError, match='completion failed'):
            complete.get_completions([], '')
    else:
        assert [x.value for x in complete.get_completions([], '')] == ['ok']
    assert events == ['enter', 'last', 'resource-exit', 'first']
