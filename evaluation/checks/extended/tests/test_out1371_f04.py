from pathlib import Path

import jinja2
import outlines
import pytest
from outlines.prompts import Prompt


def test_OUT_7149a5e613b9_N001_builtin_paths_keep_strict_undefined():
    for prompt in (
        Prompt.from_str("{{ missing }}", ["string_utils"]),
        Prompt.from_str("{{ missing }}", ["math_utils"]),
        Prompt.from_str("{{ missing }}", ["format_utils"]),
    ):
        with pytest.raises(jinja2.UndefinedError):
            prompt()


def test_OUT_7149a5e613b9_N002_builtin_paths_keep_formatting():
    prompt = Prompt.from_str(
        """
        Heading
            {{ value | upper }}

        """,
        ["string_utils"],
    )
    assert prompt(value="snow") == "Heading\n    SNOW\n"


def test_OUT_7149a5e613b9_N003_configured_decorator_keeps_binding():
    @outlines.prompt(builtin_filters=["string_utils"])
    def decorated(value="snow"):
        """{{ value | upper }}"""

    assert list(decorated.signature.parameters) == ["value"]
    assert decorated() == "SNOW"
    with pytest.raises(TypeError):
        decorated(unexpected="x")


def test_OUT_7149a5e613b9_N004_builtin_file_keeps_relative_loading(tmp_path: Path):
    (tmp_path / "base.txt").write_text(
        "{% block body %}{% endblock %}", encoding="utf-8"
    )
    (tmp_path / "part.txt").write_text(
        "{{ value | title }}", encoding="utf-8"
    )
    child = tmp_path / "child.txt"
    child.write_text(
        '{% extends "base.txt" %}{% block body %}'
        '{% include "part.txt" %}{% endblock %}',
        encoding="utf-8",
    )
    assert Prompt.from_file(child, ["string_utils"])(value="winter wind") == "Winter Wind"
