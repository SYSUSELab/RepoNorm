import pytest

from jinja2 import Environment
from jinja2 import pass_context
from jinja2 import TemplateAssertionError


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
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_jin_896a62135bcc_n031_snapshot_witness_valid_count_call(
    newstyle, escape, count
):
    env = Environment(extensions=["jinja2.ext.i18n"])
    seen = []
    calls = []

    @pass_context
    def ngettext(context, singular, plural, n):
        seen.append((context.environment, n))
        return "<b>%(name)s</b>"

    def make():
        calls.append(count)
        return count

    env.install_gettext_callables(lambda message: message, ngettext, newstyle=newstyle)
    template = env.from_string(
        "{% autoescape escape %}{% trans n=make(), name=name, trimmed %}\n"
        "one{% pluralize %}\nmany{% endtrans %}{% endautoescape %}"
    )
    assert template.render(make=make, name="<x>", escape=escape) == (
        "<b>&lt;x&gt;</b>" if escape else "<b><x></b>"
    )
    assert seen == [(env, count)]
    assert calls == [count]


@pytest.mark.parametrize(
    "name,source",
    [
        ("value", "{% trans value=1, value=2 %}{{ value }}{% endtrans %}"),
        ("数量", "{% trans 数量=1, 数量=2 %}{{ 数量 }}{% endtrans %}"),
        (
            "n",
            "{% trans n=1, n=2 %}one{% pluralize %}many{% endtrans %}",
        ),
    ],
)
def test_jin_896a62135bcc_n033_snapshot_witness_duplicate(name, source):
    env = Environment(extensions=["jinja2.ext.i18n"])
    with pytest.raises(TemplateAssertionError, match="defined twice") as error:
        env.from_string(source)
    assert name in str(error.value)
