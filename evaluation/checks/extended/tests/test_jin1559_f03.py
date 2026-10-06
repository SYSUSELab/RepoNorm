import pytest

from jinja2 import Environment
from jinja2 import pass_context


@pytest.mark.parametrize("newstyle", [False, True])
def test_jin_896a62135bcc_n031_snapshot_witness(newstyle):
    env = Environment(autoescape=True, extensions=["jinja2.ext.i18n"])
    seen = []

    @pass_context
    def gettext(context, message):
        seen.append(context.environment)
        return "<b>%(name)s</b>"

    env.install_gettext_callables(gettext, lambda one, many, n: many, newstyle=newstyle)
    assert (
        env.from_string("{% trans name=name %}Hi {{ name }}{% endtrans %}").render(
            name="<x>"
        )
        == "<b>&lt;x&gt;</b>"
    )
    assert seen == [env]


@pytest.mark.parametrize("newstyle", [False, True])
def test_jin_896a62135bcc_n029_snapshot_witness(newstyle):
    env = Environment(extensions=["jinja2.ext.i18n"])
    template = env.from_string("{% trans %}hello{% endtrans %}")
    for prefix in ("old:", "new:"):
        env.install_gettext_callables(
            lambda message, prefix=prefix: prefix + message,
            lambda one, many, n: many,
            newstyle=newstyle,
            pgettext=lambda context, message: message,
            npgettext=lambda context, one, many, n: many,
        )
        assert template.render() == prefix + "hello"
    env.uninstall_gettext_translations(None)
    for key in ("gettext", "ngettext", "pgettext", "npgettext"):
        assert key not in env.globals


@pytest.mark.parametrize("newstyle", [False, True])
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_jin_896a62135bcc_n031_priority_inherits_existing_providers(
    newstyle, escape, count
):
    env = Environment(extensions=["jinja2.ext.i18n"])
    seen = []

    @pass_context
    def gettext(context, message):
        seen.append(context.environment)
        return "<b>%(name)s</b>"

    @pass_context
    def ngettext(context, singular, plural, n):
        seen.append(context.environment)
        return "<b>%(name)s</b>"

    env.install_gettext_callables(gettext, ngettext, newstyle=newstyle)
    source = (
        '{% autoescape escape %}{% trans priority=2, n=count, '
        "name=name, trimmed %}\n"
        "one{% pluralize %}\nmany{% endtrans %}{% endautoescape %}"
    )
    assert env.from_string(source).render(
        name="<x>", escape=escape, count=count
    ) == env.from_string(source.replace('priority=2, ', "")).render(
        name="<x>", escape=escape, count=count
    )
    assert seen == [env, env]


@pytest.mark.parametrize("resolver", [False, True])
def test_jin_896a62135bcc_n031_priority_keeps_newstyle_percent_format(resolver):
    env = Environment(extensions=["jinja2.ext.i18n"])
    kwargs = {"priority_gettext": lambda message, level: message} if resolver else {}
    env.install_gettext_callables(
        lambda message: message,
        lambda singular, plural, count: plural,
        newstyle=True,
        **kwargs,
    )
    source = "{% trans priority=2 %}100% ready{% endtrans %}"
    assert env.from_string(source).render() == (
        env.from_string(source.replace("priority=2", "")).render()
    )
