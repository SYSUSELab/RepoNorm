import pytest

from jinja2 import Environment
from jinja2 import pass_context


@pytest.mark.parametrize("newstyle", [False, True])
def test_jin_896a62135bcc_n031_context_call_and_existing_render_semantics(newstyle):
    actual = Environment(autoescape=True, extensions=["jinja2.ext.i18n"])
    baseline = Environment(autoescape=True, extensions=["jinja2.ext.i18n"])
    seen = []

    @pass_context
    def primary(context, message):
        seen.append(("primary", context.environment))
        return message

    @pass_context
    def selected(context, message):
        seen.append(("selected", context.environment))
        return "<b>%(name)s</b>"

    actual.install_gettext_callables_with_fallback(
        primary,
        lambda one, many, n: many,
        fallback_gettext_funcs=[selected],
        newstyle=newstyle,
    )
    baseline.install_gettext_callables(
        lambda message: "<b>%(name)s</b>",
        lambda one, many, n: many,
        newstyle=newstyle,
    )
    direct = (
        '{{ gettext("x", name=value) }}'
        if newstyle
        else '{{ gettext("x")|format(name=value) }}'
    )
    for source in [direct, "{% trans name=value %}Hi {{ name }}{% endtrans %}"]:
        assert actual.from_string(source).render(value="<x>") == (
            baseline.from_string(source).render(value="<x>")
        )
    assert seen and all(bound is actual for _, bound in seen)


def test_jin_896a62135bcc_n029_normal_reinstall_and_uninstall():
    env = Environment(extensions=["jinja2.ext.i18n"])
    template = env.from_string('{{ gettext("x") }}')
    env.install_gettext_callables_with_fallback(
        lambda message: message,
        lambda one, many, n: many,
        fallback_gettext_funcs=[lambda message: "fallback"],
    )
    assert template.render() == "fallback"
    env.install_gettext_callables(lambda message: "normal", lambda one, many, n: many)
    assert template.render() == "normal"
    env.uninstall_gettext_translations(None)
    assert not {"gettext", "ngettext", "pgettext", "npgettext"}.intersection(
        env.globals
    )
