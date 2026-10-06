from pathlib import Path

import jinja2
import pytest
from outlines.prompts import Prompt, register_filter, unregister_filter


def test_OUT_7149a5e613b9_N001_registry_keeps_strict_undefined():
    name = "n4c_norm_identity"
    register_filter(name, lambda value: value)
    try:
        prompt = Prompt.from_str("{{ value | n4c_norm_identity }} / {{ missing }}")
        with pytest.raises(jinja2.UndefinedError):
            prompt(value="snow")
    finally:
        unregister_filter(name)


def test_OUT_7149a5e613b9_N002_registry_keeps_prompt_formatting():
    name = "n4c_norm_identity"
    register_filter(name, lambda value: value)
    try:
        prompt = Prompt.from_str(
            """
            Heading
                {{ value | n4c_norm_identity }}

            """
        )
        assert prompt(value="snow") == "Heading\n    snow\n"
    finally:
        unregister_filter(name)


def test_OUT_7149a5e613b9_N004_registry_keeps_relative_file_loading(tmp_path: Path):
    name = "n4c_norm_decorate"
    register_filter(name, lambda value: f"<{value}>")
    try:
        (tmp_path / "base.txt").write_text("{% block body %}{% endblock %}", encoding="utf-8")
        (tmp_path / "body.txt").write_text("{{ value | n4c_norm_decorate }}", encoding="utf-8")
        child = tmp_path / "child.txt"
        child.write_text(
            '{% extends "base.txt" %}{% block body %}{% include "body.txt" %}{% endblock %}',
            encoding="utf-8",
        )
        assert Prompt.from_file(child)(value="snow") == "<snow>"
    finally:
        unregister_filter(name)
