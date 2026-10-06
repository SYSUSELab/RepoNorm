import pytest

from jinja2 import Environment
from jinja2 import nodes
from jinja2 import pass_context
from jinja2.ext import extract_from_ast


@pytest.mark.parametrize("newstyle", [False, True])
def test_jin_896a62135bcc_n031_snapshot_witness(newstyle):
    env = Environment(autoescape=True, extensions=["jinja2.ext.i18n"])

    @pass_context
    def translated(context, domain, message):
        assert context.environment is env
        return "<b>%(name)s</b>"

    env.install_gettext_callables(
        lambda message: "<b>%(name)s</b>",
        lambda one, many, n: many,
        newstyle=newstyle,
        pgettext=translated,
    )
    assert (
        env.from_string("{% trans name=value %}Hi {{ name }}{% endtrans %}").render(
            value="<x>"
        )
        == "<b>&lt;x&gt;</b>"
    )
    source = (
        '{{ pgettext("html", "x", name=value) }}'
        if newstyle
        else '{{ pgettext("html", "x")|format(name=value) }}'
    )
    expected = "<b>&lt;x&gt;</b>" if newstyle else "&lt;b&gt;&lt;x&gt;&lt;/b&gt;"
    assert env.from_string(source).render(value="<x>") == expected


@pytest.mark.parametrize("newstyle", [False, True])
@pytest.mark.parametrize("escape", [False, True])
def test_jin_896a62135bcc_n031_context_inherits_current_rendering(newstyle, escape):
    env = Environment(extensions=["jinja2.ext.i18n"])
    seen = []

    @pass_context
    def contextual(context, domain, message):
        seen.append(context.environment)
        return "<b>%(name)s</b>"

    env.install_gettext_callables(
        lambda message: "<b>%(name)s</b>",
        lambda one, many, n: many,
        newstyle=newstyle,
        pgettext=contextual,
    )
    source = (
        '{% autoescape escape %}{% trans "html" name=value %}'
        "Hi {{ name }}{% endtrans %}{% endautoescape %}"
    )
    assert env.from_string(source).render(value="<x>", escape=escape) == (
        env.from_string(source.replace('trans "html"', "trans")).render(
            value="<x>", escape=escape
        )
    )
    assert seen == [env]


@pytest.mark.parametrize("newstyle", [False, True])
def test_jin_896a62135bcc_n031_context_preserves_plural_and_whitespace(newstyle):
    env = Environment(extensions=["jinja2.ext.i18n"])
    env.install_gettext_callables(
        lambda message: message,
        lambda one, many, n: one if n == 1 else many,
        newstyle=newstyle,
        pgettext=lambda domain, message: message,
        npgettext=lambda domain, one, many, n: one if n == 1 else many,
    )
    for modifier in ("trimmed", "notrimmed"):
        source = (
            '{% trans "files" count=n, ' + modifier + " %}\n"
            "{{ count }} file\n{% pluralize %}\n{{ count }} files\n{% endtrans %}"
        )
        for count in (1, 2):
            assert env.from_string(source).render(n=count) == (
                env.from_string(source.replace('trans "files"', "trans")).render(
                    n=count
                )
            )


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


def test_jin_896a62135bcc_n032_generated_calls_use_existing_extraction_protocol():
    env = Environment(extensions=["jinja2.ext.i18n"])
    source = (
        '{% trans "files" count=n, trimmed %}\n'
        "{{ count }} file\n{% pluralize %}\n{{ count }} files\n{% endtrans %}"
    )
    tree = env.parse(source)
    recognized = {"_", "gettext", "ngettext", "pgettext", "npgettext"}
    expected = []
    for call in tree.find_all(nodes.Call):
        if not isinstance(call.node, nodes.Name) or call.node.name not in recognized:
            continue
        args = [
            arg.value
            if isinstance(arg, nodes.Const) and isinstance(arg.value, str)
            else None
            for arg in call.args
        ]
        args.extend([None] * len(call.kwargs))
        if call.dyn_args is not None:
            args.append(None)
        if call.dyn_kwargs is not None:
            args.append(None)
        message = args[0] if len(args) == 1 else tuple(args)
        expected.append((call.lineno, call.node.name, message))
    assert expected
    assert list(env.extract_translations(source)) == expected
    assert list(env.extract_translations(tree)) == expected
