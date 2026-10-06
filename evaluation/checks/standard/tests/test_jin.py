import asyncio
import pytest
from jinja2 import Environment, pass_context, pass_environment, pass_eval_context
from jinja2.sandbox import SandboxedEnvironment


def render(env, source, **values):
    template = env.from_string(source)
    if env.is_async:
        return asyncio.run(template.render_async(**values))
    return template.render(**values)


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('mode', ['map', 'groupby'])
def test_JIN_N01_environment_lookup(asynchronous, mode):
    class VirtualEnvironment(Environment):
        def getitem(self, obj, key):
            if key == 'virtual':
                return {'label': 'visible'}
            return super().getitem(obj, key)

    env = VirtualEnvironment(enable_async=asynchronous)
    if mode == 'map':
        source = "{{ xs|map(attribute='virtual__label', separator='__')|join(',') }}"
    else:
        source = "{% for g in xs|groupby('virtual__label', separator='__') %}{{ g.grouper }}{% endfor %}"
    assert render(env, source, xs=[object()]) == 'visible'


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('mode', ['map', 'groupby'])
def test_JIN_N01_sandbox_policy(asynchronous, mode):
    class Value:
        secret = 'should-not-leak'
        public = 'allowed'

    class Restricted(SandboxedEnvironment):
        def is_safe_attribute(self, obj, attr, value):
            return attr != 'secret' and super().is_safe_attribute(obj, attr, value)

    env = Restricted(enable_async=asynchronous)
    if mode == 'map':
        source = "{{ xs|map(attribute='box__secret', separator='__', default='blocked')|join(',') }}"
    else:
        source = "{% for g in xs|groupby('box__secret', separator='__', default='blocked') %}{{ g.grouper }}{% endfor %}"
    assert render(env, source, xs=[{'box': Value()}]) == 'blocked'


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('kind', ['context', 'environment', 'eval_context'])
def test_JIN_N02_filter_dispatch(asynchronous, kind):
    env = Environment(enable_async=asynchronous, autoescape=False)
    if kind == 'context':
        @pass_context
        def check(ctx, value, separator):
            return ctx['prefix'] + separator + value
    elif kind == 'environment':
        @pass_environment
        def check(actual, value, separator):
            assert actual is env
            return 'E' + separator + value
    else:
        @pass_eval_context
        def check(ctx, value, separator):
            assert ctx.environment is env
            return 'V' + separator + value
    env.filters['custom'] = check
    prefix = {'context': 'C', 'environment': 'E', 'eval_context': 'V'}[kind]
    assert render(env, "{{ xs|map('custom', separator=':')|join(',') }}",
                  xs=['a', 'b'], prefix='C') == f'{prefix}:a,{prefix}:b'


@pytest.mark.parametrize('async_input', [False, True])
def test_JIN_N03_await_filter_result(async_input):
    env = Environment(enable_async=True)

    async def deferred(value, separator):
        await asyncio.sleep(0)
        return value + separator

    async def stream():
        for value in ['a', 'b']:
            yield value

    env.filters['deferred'] = deferred
    values = stream() if async_input else iter(['a', 'b'])
    assert render(env, "{{ xs|map('deferred', separator='!')|join(',') }}", xs=values) == 'a!,b!'
