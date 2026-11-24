from jinja2 import Environment
from jinja2 import pass_context
from jinja2.ext import InternationalizationExtension


def test_jin_896a62135bcc_n031_context_and_current_format_escape():
    env = Environment(extensions=["jinja2.ext.i18n"])
    seen = []

    @pass_context
    def provider(context, message):
        seen.append(context.environment)
        return "<b>%(name)s</b>"

    env.install_gettext_callables(
        provider, lambda one, many, n: many, cache_size=2, newstyle=True
    )
    template = env.from_string(
        '{% autoescape escape %}{{ gettext("x", name=name) }}{% endautoescape %}'
    )
    assert template.render(name="<a>", escape=True) == "<b>&lt;a&gt;</b>"
    assert template.render(name="<b>", escape=False) == "<b><b></b>"
    assert seen and all(bound is env for bound in seen)


def test_jin_896a62135bcc_n037_overlay_preserves_extension_binding():
    env = Environment(extensions=["jinja2.ext.i18n"])
    env.install_gettext_callables(
        lambda value: value,
        lambda one, many, n: many,
        cache_size=2,
    )
    overlay = env.overlay()
    key = InternationalizationExtension.identifier
    assert overlay.extensions[key] is not env.extensions[key]
    assert overlay.extensions[key].environment is overlay
    assert env.extensions[key].environment is env
    child = overlay.overlay()
    assert child.extensions[key].environment is child
    assert overlay.extensions[key].environment is overlay


def test_jin_896a62135bcc_n029_provider_reinstall_and_uninstall():
    env = Environment(extensions=["jinja2.ext.i18n"])
    template = env.from_string('{{ gettext("x") }}')
    for value in ("first", "second"):
        env.install_gettext_callables(
            lambda message: value, lambda one, many, n: many, cache_size=2
        )
        assert template.render() == value
    env.uninstall_gettext_translations(None)
    assert not {"gettext", "ngettext", "pgettext", "npgettext"}.intersection(
        env.globals
    )
