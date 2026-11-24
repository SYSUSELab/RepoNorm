import pytest

from jinja2 import Environment
from jinja2 import pass_context
from jinja2.ext import extract_from_ast


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


def test_jin_896a62135bcc_n032_snapshot_witness():
    env = Environment(extensions=["jinja2.ext.i18n"])
    source = (
        '{{ pgettext("menu", "Open") }}\n' '{{ npgettext("files", "one", "many", n) }}'
    )
    assert list(env.extract_translations(source)) == [
        (1, "pgettext", ("menu", "Open")),
        (2, "npgettext", ("files", "one", "many", None)),
    ]
    dynamic = (
        '{{ gettext("Hi", name=value) }}\n'
        '{{ gettext("Dynamic", *args, **kwargs) }}\n'
        '{{ unknown("ignored") }}'
    )
    assert list(env.extract_translations(dynamic)) == [
        (1, "gettext", ("Hi", None)),
        (2, "gettext", ("Dynamic", None, None)),
    ]
    assert list(extract_from_ast(env.parse(dynamic), babel_style=False)) == [
        (1, "gettext", ("Hi",)),
        (2, "gettext", ("Dynamic",)),
    ]




@pytest.mark.parametrize("newstyle", [False, True])
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_jin_896a62135bcc_n031_metadata_preserves_current_rendering(
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
        '{% autoescape escape %}{% trans note="compile", n=count, '
        'name=name, trimmed %}\n'
        "one{% pluralize %}\nmany{% endtrans %}{% endautoescape %}"
    )
    plain = source.replace('note="compile", ', "")
    assert env.from_string(source).render(
        name="<x>", escape=escape, count=count
    ) == env.from_string(plain).render(
        name="<x>", escape=escape, count=count
    )
    assert seen == [env, env]


@pytest.mark.parametrize("newstyle", [False, True])
def test_jin_896a62135bcc_n032_metadata_preserves_ordinary_extraction(newstyle):
    env = Environment(extensions=["jinja2.ext.i18n"])
    env.install_null_translations(newstyle=newstyle)
    source = (
        '{% trans note="compile", n=count, name=name, trimmed %}\n'
        "one {{ name }}{% pluralize %}\nmany {{ name }}{% endtrans %}"
    )
    plain = source.replace('note="compile", ', "")
    expected = list(env.extract_translations(plain))
    assert expected
    assert list(env.extract_translations(source)) == expected
    assert list(env.extract_translations(env.parse(source))) == expected
