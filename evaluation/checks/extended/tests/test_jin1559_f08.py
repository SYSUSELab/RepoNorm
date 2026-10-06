import pytest

from jinja2 import Environment
from jinja2 import pass_context
from jinja2.ext import extract_from_ast


@pytest.mark.parametrize("newstyle", [False, True])
@pytest.mark.parametrize("escape", [False, True])
def test_jin_896a62135bcc_n031_conditional_matches_existing_trans(newstyle, escape):
    env = Environment(extensions=["jinja2.ext.i18n", "jinja2.ext.conditional_i18n"])
    seen = []

    @pass_context
    def gettext(context, message):
        seen.append(context.environment)
        return "<b>%(name)s</b>"

    env.install_gettext_callables(gettext, lambda one, many, n: many, newstyle=newstyle)
    ordinary = (
        "{% autoescape escape %}{% trans name=name %}Hi {{ name }}"
        "{% endtrans %}{% endautoescape %}"
    )
    conditional = ordinary.replace("trans name=", "ctrans enabled, name=").replace(
        "{% endtrans %}", "{% fallback %}off{% endctrans %}"
    )
    assert env.from_string(conditional).render(
        enabled=True, name="<x>", escape=escape
    ) == env.from_string(ordinary).render(name="<x>", escape=escape)
    assert seen == [env, env]


@pytest.mark.parametrize("newstyle", [False, True])
def test_jin_896a62135bcc_n031_percent_and_existing_plural_formatting(newstyle):
    env = Environment(extensions=["jinja2.ext.i18n", "jinja2.ext.conditional_i18n"])
    env.install_null_translations(newstyle=newstyle)
    for body in [
        "100%",
        "{{ n }} item{% pluralize %}{{ n }} {{ label }}",
    ]:
        plain = "{% trans %}" + body + "{% endtrans %}"
        conditional = "{% ctrans True %}" + body + "{% fallback %}off{% endctrans %}"
        assert env.from_string(conditional).render(n=2, label="items") == (
            env.from_string(plain).render(n=2, label="items")
        )


def test_jin_896a62135bcc_n032_existing_ast_and_babel_protocol():
    env = Environment(extensions=["jinja2.ext.i18n", "jinja2.ext.conditional_i18n"])
    env.install_null_translations(newstyle=True)
    source = (
        "{% ctrans enabled, n=n %}one{% pluralize %}many"
        '{% fallback %}{{ gettext("fallback", name=name) }}{% endctrans %}'
    )
    expected = [
        (1, "ngettext", ("one", "many", None, None)),
        (1, "gettext", ("fallback", None)),
    ]
    assert list(env.extract_translations(source)) == expected
    assert list(env.extract_translations(env.parse(source))) == expected
    assert list(extract_from_ast(env.parse(source), babel_style=False)) == [
        (1, "ngettext", ("one", "many")),
        (1, "gettext", ("fallback",)),
    ]


@pytest.mark.parametrize("newstyle", [False, True])
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("trim", ["trimmed", "notrimmed"])
def test_jin_896a62135bcc_n031_plural_context_and_trim(newstyle, count, trim):
    env = Environment(extensions=["jinja2.ext.i18n", "jinja2.ext.conditional_i18n"])
    seen = []

    @pass_context
    def npgettext(context, domain, one, many, n):
        seen.append((context.environment, domain, one, many, n))
        return one if n == 1 else many

    env.install_gettext_callables(
        lambda message: message,
        lambda one, many, n: one if n == 1 else many,
        npgettext=npgettext,
        newstyle=newstyle,
    )
    plain = (
        "{% trans n=n, " + trim + " %}\n  {{ n }} item\n"
        "{% pluralize %}\n  {{ n }} items\n{% endtrans %}"
    )
    conditional = plain.replace("trans n=n", 'ctrans True, context="menu", n=n')
    conditional = conditional.replace(
        "{% endtrans %}", "{% fallback %}off{% endctrans %}"
    )
    assert env.from_string(conditional).render(n=count) == env.from_string(
        plain
    ).render(n=count)
    assert len(seen) == 1
    assert seen[0][0:2] == (env, "menu")
    assert seen[0][-1] == count


@pytest.mark.parametrize("newstyle", [False, True])
def test_jin_896a62135bcc_n031_assigned_trim_name_is_a_variable(newstyle):
    env = Environment(extensions=["jinja2.ext.i18n", "jinja2.ext.conditional_i18n"])
    env.install_null_translations(newstyle=newstyle)
    plain = "{% trans trimmed=value %}{{ trimmed }}{% endtrans %}"
    conditional = plain.replace("trans trimmed=", "ctrans True, trimmed=").replace(
        "{% endtrans %}", "{% fallback %}off{% endctrans %}"
    )
    assert env.from_string(conditional).render(value="visible") == (
        env.from_string(plain).render(value="visible")
    )
