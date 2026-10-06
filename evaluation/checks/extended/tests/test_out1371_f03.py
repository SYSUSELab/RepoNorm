from pathlib import Path

import jinja2
import outlines
import pytest
from outlines.prompts import Prompt


@pytest.fixture(autouse=True)
def clean_condition_environment(monkeypatch):
    for name in (
        "OUTLINES_DEBUG",
        "ENVIRONMENT",
        "ENV",
        "OUTLINES_TESTING",
        "TESTING_MODE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_OUT_7149a5e613b9_N001_conditional_paths_keep_strict_undefined(monkeypatch):
    monkeypatch.setenv("OUTLINES_TESTING", "TRUE")
    filters = {"testing": {"identity": lambda value: value}}
    prompt = Prompt.from_str("{{ value | identity }} / {{ missing }}", conditional_filters=filters)
    with pytest.raises(jinja2.UndefinedError):
        prompt(value="snow")


def test_OUT_7149a5e613b9_N002_conditional_paths_keep_formatting(monkeypatch):
    monkeypatch.setenv("TESTING_MODE", "on")
    filters = {"testing": {"identity": lambda value: value}}
    prompt = Prompt.from_str(
        """
        Heading
            {{ value | identity }}

        """,
        conditional_filters=filters,
    )
    assert prompt(value="snow") == "Heading\n    snow\n"


def test_OUT_7149a5e613b9_N003_bare_and_configured_decorators_keep_binding(
    monkeypatch,
):
    monkeypatch.setenv("ENV", "testing")

    @outlines.prompt
    def bare(value="snow"):
        """{{ value }}"""

    @outlines.prompt(conditional_filters={"testing": {"decorate": lambda value: f"<{value}>"}})
    def configured(value="snow"):
        """{{ value | decorate }}"""

    assert list(bare.signature.parameters) == ["value"]
    assert list(configured.signature.parameters) == ["value"]
    assert bare() == "snow"
    assert configured() == "<snow>"
    with pytest.raises(TypeError):
        configured(unexpected="x")


def test_OUT_7149a5e613b9_N004_conditional_file_keeps_relative_loading(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ENVIRONMENT", "production")
    filters = {"production": {"decorate": lambda value: f"<{value}>"}}
    (tmp_path / "base.txt").write_text("{% block body %}{% endblock %}", encoding="utf-8")
    (tmp_path / "body.txt").write_text("{{ value | decorate }}", encoding="utf-8")
    child = tmp_path / "child.txt"
    child.write_text(
        '{% extends "base.txt" %}{% block body %}{% include "body.txt" %}{% endblock %}',
        encoding="utf-8",
    )
    assert Prompt.from_file(child, conditional_filters=filters)(value="x") == "<x>"
