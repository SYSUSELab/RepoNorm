import jinja2
import outlines
import pytest
from outlines.prompts import Prompt


def test_OUT_7149a5e613b9_N001_custom_paths_keep_strict_undefined(tmp_path):
    filters = {"tag": lambda value: f"<{value}>"}

    @outlines.prompt(filters=filters)
    def decorated(value):
        """{{ value | tag }} / {{ missing }}"""

    file_prompt = tmp_path / "prompt.txt"
    file_prompt.write_text("{{ value | tag }} / {{ missing }}", encoding="utf-8")
    prompts = [
        decorated,
        Prompt.from_str("{{ value | tag }} / {{ missing }}", filters=filters),
        Prompt.from_file(file_prompt, filters=filters),
        Prompt.from_str("{{ value | tag }} / {{ missing }}", filters=filters).with_filters(
            {"tag": lambda value: f"[{value}]"}
        ),
    ]

    for prompt in prompts:
        with pytest.raises(jinja2.UndefinedError):
            prompt(value="snow")


def test_OUT_7149a5e613b9_N002_derivation_keeps_prompt_formatting():
    original = Prompt.from_str(
        """
        Heading
            {{ value | upper }}

        """
    )

    derived = original.with_filters({"upper": lambda value: f"<{value}>"})

    assert original(value="snow") == "Heading\n    SNOW\n"
    assert derived(value="snow") == "Heading\n    <snow>\n"


def test_OUT_7149a5e613b9_N003_derivation_keeps_signature_and_binding():
    @outlines.prompt
    def original(value="snow", suffix="!"):
        """{{ value | upper }}{{ suffix }}"""

    derived = original.with_filters({"upper": lambda value: f"<{value}>"})

    assert list(derived.signature.parameters) == ["value", "suffix"]
    assert derived() == "<snow>!"
    assert derived("rain", suffix="?") == "<rain>?"
    with pytest.raises(TypeError):
        derived(unexpected="x")


def test_OUT_7149a5e613b9_N004_file_derivation_keeps_relative_loading(tmp_path):
    (tmp_path / "base.txt").write_text("{% block body %}{% endblock %}", encoding="utf-8")
    (tmp_path / "body.txt").write_text("{{ value | decorate }}", encoding="utf-8")
    child = tmp_path / "child.txt"
    child.write_text(
        '{% extends "base.txt" %}{% block body %}{% include "body.txt" %}{% endblock %}',
        encoding="utf-8",
    )

    original = Prompt.from_file(child, filters={"decorate": str.upper})
    derived = original.with_filters({"decorate": lambda value: f"<{value}>"})

    assert original(value="snow") == "SNOW"
    assert derived(value="snow") == "<snow>"
