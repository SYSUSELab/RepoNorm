"""Offline checks for the model-facing editor's repository/docs boundary."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def editor(tmp_path, monkeypatch):
    module = importlib.import_module("codewiki.src.be.agent_tools.str_replace_editor")
    repo = tmp_path / "repo"
    docs = tmp_path / "docs"
    repo.mkdir()
    docs.mkdir()
    (repo / "source.py").write_text("source content", encoding="utf-8")
    (docs / "guide.md").write_text("documentation", encoding="utf-8")
    validation = AsyncMock(return_value="Mermaid valid")
    monkeypatch.setattr(module, "validate_mermaid_diagrams", validation)
    ctx = SimpleNamespace(deps=SimpleNamespace(
        absolute_repo_path=str(repo), absolute_docs_path=str(docs), registry={},
    ))

    def call(working_dir, command, path, **kwargs):
        return asyncio.run(module.str_replace_editor(
            ctx, working_dir, command, path=path, **kwargs,
        ))

    return SimpleNamespace(
        call=call, repo=repo, docs=docs, module=module, ctx=ctx,
        validation=validation,
    )


@pytest.mark.parametrize("working_dir", ["repo", "docs"])
@pytest.mark.parametrize("absolute", [False, True])
def test_view_rejects_path_escape(editor, tmp_path, working_dir, absolute):
    outside = tmp_path / "outside.txt"
    outside.write_text("private outside content", encoding="utf-8")
    path = str(outside) if absolute else "../outside.txt"
    result = editor.call(working_dir, "view", path)
    assert "outside the selected working directory" in result
    assert "private outside content" not in result


@pytest.mark.parametrize("command,kwargs", [
    ("create", {"file_text": "changed"}),
    ("str_replace", {"old_str": "source content", "new_str": "changed"}),
    ("insert", {"insert_line": 0, "new_str": "changed"}),
    ("undo_edit", {}),
])
def test_docs_cannot_write_repository(editor, command, kwargs):
    result = editor.call("docs", command, "../repo/source.py", **kwargs)
    assert "outside the selected working directory" in result
    assert (editor.repo / "source.py").read_text() == "source content"
    result = editor.call("repo", command, "source.py", **kwargs)
    assert "only allowed command" in result


def test_create_rejects_absolute_escape(editor, tmp_path):
    destination = tmp_path / "outside.md"
    result = editor.call("docs", "create", str(destination), file_text="changed")
    assert "outside the selected working directory" in result
    assert not destination.exists()
    editor.validation.assert_not_called()


@pytest.mark.parametrize("working_dir", ["repo", "docs"])
def test_view_rejects_symlink_escape(editor, tmp_path, working_dir):
    outside = tmp_path / "outside.txt"
    outside.write_text("private outside content", encoding="utf-8")
    root = getattr(editor, working_dir)
    (root / "linked.txt").symlink_to(outside)
    result = editor.call(working_dir, "view", "linked.txt")
    assert "outside the selected working directory" in result
    assert "private outside content" not in result


def test_create_rejects_symlink_parent_escape(editor):
    (editor.docs / "linked").symlink_to(editor.repo, target_is_directory=True)
    result = editor.call("docs", "create", "linked/new.md", file_text="changed")
    assert "outside the selected working directory" in result
    assert not (editor.repo / "new.md").exists()
    editor.validation.assert_not_called()


def test_overlapping_docs_root_does_not_allow_repository_writes(editor):
    editor.ctx.deps.absolute_docs_path = str(editor.repo)
    result = editor.call("docs", "create", "new.md", file_text="changed")
    assert "Repository files are read-only" in result
    assert not (editor.repo / "new.md").exists()


def test_normal_view_create_and_edit_preserve_mermaid_validation(editor):
    assert "source content" in editor.call("repo", "view", "./source.py")
    assert "documentation" in editor.call("docs", "view", str(editor.docs / "guide.md"))
    assert "created successfully" in editor.call("docs", "create", "new.md", file_text="first")
    editor.call("docs", "str_replace", "new.md", old_str="first", new_str="second")
    editor.call("docs", "insert", "new.md", insert_line=1, new_str="third")
    assert (editor.docs / "new.md").read_text() == "second\nthird"
    assert editor.validation.await_count == 3


def test_internal_symlink_is_allowed(editor):
    (editor.repo / "linked.py").symlink_to(editor.repo / "source.py")
    assert "source content" in editor.call("repo", "view", "linked.py")


def test_directory_names_are_literal_not_shell_commands(editor, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dirname = "directory; touch injected; #"
    directory = editor.repo / dirname
    directory.mkdir()
    (directory / "visible.txt").write_text("visible", encoding="utf-8")
    (directory / ".hidden").write_text("hidden", encoding="utf-8")
    result = editor.call("repo", "view", dirname)
    assert "visible.txt" in result
    assert ".hidden" not in result
    assert not (tmp_path / "injected").exists()
