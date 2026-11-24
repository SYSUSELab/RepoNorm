import os
import shlex
from pathlib import Path

import click
import pytest


class _Process:
    # Accommodate equivalent subprocess waiting strategies. Never execute
    # preexec_fn or OS scheduling operations on the host test process.
    pid = 987654321

    def __init__(self, code=0):
        self.returncode = code

    def wait(self, timeout=None):
        return self.returncode

    def communicate(self, input=None, timeout=None):
        return (b"", b"")

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass


def _argv(command):
    # For these plain trusted editor commands, equivalent quoted shell and
    # argv forms have the same observable argument boundaries.
    return shlex.split(command) if isinstance(command, str) else list(command)


def _install(monkeypatch, *, code=0, payload=None, error=None):
    calls = []

    def launch(command, *args, **kwargs):
        argv = _argv(command)
        calls.append((argv, kwargs))
        if error is not None:
            raise error
        if payload is not None:
            name = Path(argv[-1])
            name.write_bytes(payload)
            stamp = name.stat().st_mtime
            os.utime(name, (stamp + 4, stamp + 4))
        return _Process(code)

    monkeypatch.setattr("subprocess.Popen", launch)
    return calls


@pytest.mark.parametrize("text,expected", [("source", "changed\n"), (b"source", b"changed\n")])
def test_cli_d79153731794_n001_public_text_mode(monkeypatch, text, expected):
    calls = _install(monkeypatch, payload=b"changed\n")
    assert click.edit(text, editor="editor", **_feature_options()) == expected
    assert len(calls) == 1
    assert not Path(calls[0][0][-1]).exists()


def test_cli_d79153731794_n001_direct_file_mode(tmp_path, monkeypatch):
    name = tmp_path / "two words.txt"
    name.write_text("source", encoding="utf-8")
    calls = _install(monkeypatch)
    assert click.edit(filename=str(name), editor="editor", require_save=True,
                      extension=".ignored", **_feature_options()) is None
    assert len(calls) == 1
    assert calls[0][0][-1] == str(name)
    assert name.read_text(encoding="utf-8") == "source"


@pytest.mark.parametrize("selection", ["explicit", "visual", "editor"])
def test_cli_d79153731794_n002_command_selection_and_arguments(tmp_path, monkeypatch, selection):
    monkeypatch.setenv("VISUAL", "visual-editor --visual")
    monkeypatch.setenv("EDITOR", "fallback-editor --fallback")
    options = _feature_options()
    if selection == "explicit":
        options["editor"] = "chosen-editor --wait"
        prefix = ["chosen-editor", "--wait"]
    elif selection == "visual":
        prefix = ["visual-editor", "--visual"]
    else:
        monkeypatch.delenv("VISUAL")
        prefix = ["fallback-editor", "--fallback"]
    name = tmp_path / "two words.txt"
    name.touch()
    calls = _install(monkeypatch)
    click.edit(filename=str(name), **options)
    assert len(calls) == 1
    assert calls[0][0] == prefix + _extra_editor_args() + [str(name)]


@pytest.mark.parametrize("overlay", [None, {}, {"N4C_PARENT": "overridden", "N4C_CHILD": "added"}])
def test_cli_d79153731794_n003_environment_overlay(tmp_path, monkeypatch, overlay):
    monkeypatch.setenv("N4C_PARENT", "kept")
    monkeypatch.setenv("N4C_UNTOUCHED", "retained")
    before = dict(os.environ)
    name = tmp_path / "one"
    name.touch()
    calls = _install(monkeypatch)
    click.edit(filename=str(name), editor="editor", env=overlay, **_feature_options())
    assert len(calls) == 1
    child = calls[0][1].get("env")
    effective = before if child is None else dict(child)
    assert effective == {**before, **(overlay or {})}
    assert dict(os.environ) == before


@pytest.mark.parametrize("mode", ["direct", "text"])
@pytest.mark.parametrize("failure", ["nonzero", "oserror"])
def test_cli_d79153731794_n004_process_errors_and_cleanup(tmp_path, monkeypatch, mode, failure):
    original = OSError("editor launch unavailable") if failure == "oserror" else None
    calls = _install(monkeypatch, code=7, error=original)
    name = tmp_path / "one"
    name.touch()
    options = {"editor": "editor", **_feature_options()}
    if mode == "direct":
        options["filename"] = str(name)
        args = ()
    else:
        args = ("source",)
    with pytest.raises(click.ClickException, match="Editing failed") as caught:
        click.edit(*args, **options)
    assert len(calls) == 1
    if original is not None:
        assert caught.value.__cause__ is original


@pytest.mark.parametrize("text,expected", [("source", "changed\n"), (b"source", b"\xef\xbb\xbfchanged\r\n")])
def test_cli_d79153731794_n005_type_newline_conversion_and_cleanup(monkeypatch, text, expected):
    calls = _install(monkeypatch, payload=b"\xef\xbb\xbfchanged\r\n")
    assert click.edit(text, editor="editor", **_feature_options()) == expected
    assert len(calls) == 1
    assert not Path(calls[0][0][-1]).exists()


@pytest.mark.parametrize("require_save", [True, False])
def test_cli_d79153731794_n005_unsaved_detection_and_cleanup(monkeypatch, require_save):
    calls = _install(monkeypatch)
    result = click.edit("source", editor="editor", require_save=require_save, **_feature_options())
    assert result == (None if require_save else "source\n")
    assert len(calls) == 1
    assert not Path(calls[0][0][-1]).exists()


@pytest.mark.parametrize("text", ["source", b"source"])
@pytest.mark.parametrize("failure", ["nonzero", "oserror"])
def test_cli_d79153731794_n005_failed_session_cleans_tempfile(monkeypatch, text, failure):
    original = OSError("editor launch unavailable") if failure == "oserror" else None
    calls = _install(monkeypatch, code=7, error=original)
    with pytest.raises(click.ClickException):
        click.edit(text, editor="editor", **_feature_options())
    assert len(calls) == 1
    assert not Path(calls[0][0][-1]).exists()


def _feature_options():
    return {"process_priority": "normal"}


def _extra_editor_args():
    return []
