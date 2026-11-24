from io import BytesIO

import pytest

from jinja2 import BaseLoader
from jinja2 import Environment
from jinja2 import FunctionLoader
from jinja2 import TemplateNotFound
from jinja2 import TemplateSyntaxError
from jinja2.ext import babel_extract


def environment(loader=None):
    return Environment(
        loader=loader, extensions=["jinja2.ext.i18n", "jinja2.ext.extractor"]
    )


def test_jin_896a62135bcc_n032_marker_uses_current_extracted_message_protocol():
    env = environment()
    source = (
        '{% extract note="x" %}{{ ngettext("one", "many", n) }}\n'
        '{{ gettext("Hi", name=name) }}\n'
        '{{ gettext("Dynamic", *args, **kwargs) }}{% endextract %}'
    )
    expected = list(env.extract_translations(source))
    messages = list(env.extract_marked_translations(env.parse(source)))[0]["messages"]
    assert [(m["lineno"], m["function"], m["message"]) for m in messages] == expected
    assert expected == [
        (1, "ngettext", ("one", "many", None)),
        (2, "gettext", ("Hi", None)),
        (3, "gettext", ("Dynamic", None, None)),
    ]


def test_jin_896a62135bcc_n035_babel_comments_lines_and_keywords():
    source = (
        '{% extract note="x" %}\n{# trans Caption #}\n'
        '{{ gettext("Open") }}\n{{ ignored("omit") }}\n{% endextract %}'
    )
    opts = {"extensions": "jinja2.ext.extractor", "silent": "false"}
    assert list(
        babel_extract(BytesIO(source.encode()), ["gettext"], ["trans"], opts)
    ) == [(3, "gettext", "Open", ["Caption"])]


@pytest.mark.parametrize("silent", [True, False])
def test_jin_896a62135bcc_n035_babel_syntax_error_option(silent):
    opts = {"extensions": "jinja2.ext.extractor", "silent": str(silent).lower()}
    source = b"{% extract note=dynamic %}x{% endextract %}"
    if silent:
        assert list(babel_extract(BytesIO(source), ["gettext"], [], opts)) == []
    else:
        with pytest.raises(TemplateSyntaxError):
            list(babel_extract(BytesIO(source), ["gettext"], [], opts))


@pytest.mark.parametrize("filename", [None, "/not-a-physical-file"])
def test_jin_896a62135bcc_n036_loader_optional_filename_and_error(filename):
    seen = []
    failure = TemplateNotFound("missing")

    class Loader(BaseLoader):
        def get_source(self, env, name):
            seen.append((env, name))
            if name == "missing":
                raise failure
            return (
                '{% extract note="x" %}{{ gettext("Open") }}{% endextract %}',
                filename,
                None,
            )

    env = environment(Loader())
    assert (
        list(env.extract_marked_translations_from_templates(["logical"]))[0][
            "messages"
        ][0]["message"]
        == "Open"
    )
    assert seen and all(item == (env, "logical") for item in seen)
    with pytest.raises(TemplateNotFound) as caught:
        list(env.extract_marked_translations_from_templates(["missing"]))
    assert caught.value.name == failure.name


def test_jin_896a62135bcc_n036_loader_unsupported_listing_not_empty_success():
    env = environment(
        FunctionLoader(lambda name: '{% extract note="x" %}x{% endextract %}')
    )
    assert len(list(env.extract_marked_translations_from_templates(["logical"]))) == 1
    with pytest.raises(TypeError):
        list(env.extract_marked_translations_from_templates())


def test_jin_896a62135bcc_n035_snapshot_witness():
    source = b'{# trans Caption #}\n{{ gettext("Open") }}'
    assert list(babel_extract(BytesIO(source), ["gettext"], ["trans"], {})) == [
        (2, "gettext", "Open", ["Caption"])
    ]
    invalid = b"{% if %}x{% endif %}"
    assert list(babel_extract(BytesIO(invalid), ["gettext"], [], {})) == []
    with pytest.raises(TemplateSyntaxError):
        list(babel_extract(BytesIO(invalid), ["gettext"], [], {"silent": "false"}))


def test_jin_896a62135bcc_n036_snapshot_witness():
    failure = TemplateNotFound("missing")

    class Loader(BaseLoader):
        def get_source(self, env, name):
            if name == "missing":
                raise failure
            return "{{ value }}", None, None

    env = Environment(loader=Loader())
    assert env.get_template("logical").render(value="works") == "works"
    with pytest.raises(TemplateNotFound) as caught:
        env.get_template("missing")
    assert caught.value.name == failure.name
    with pytest.raises(TypeError):
        env.list_templates()
