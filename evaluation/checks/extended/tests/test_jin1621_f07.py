import inspect
import os
from pathlib import Path

import pytest

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import TemplateNotFound


@pytest.fixture
def make_loader(monkeypatch):
    def make(roots, expanded, **kwargs):
        roots = [roots] if isinstance(roots, (str, Path)) else roots
        if (
            expanded
            and "expand_searchpath" in inspect.signature(FileSystemLoader).parameters
        ):
            markers = []
            for index, root in enumerate(roots):
                variable = "N4C_NORM_ROOT_" + str(index)
                monkeypatch.setenv(variable, os.fspath(root))
                markers.append("$" + variable)
            return FileSystemLoader(markers, expand_searchpath=True, **kwargs)
        return FileSystemLoader(roots, **kwargs)

    return make


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("name", ["../secret", "sub/../../secret"])
def test_jin_a2920752fd11_n001_snapshot_witness(tmp_path, make_loader, expanded, name):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret").write_text("outside")
    with pytest.raises(TemplateNotFound) as caught:
        make_loader(root, expanded).get_source(Environment(), name)
    assert caught.value.name == name


@pytest.mark.parametrize("expanded", [False, True])
def test_jin_a2920752fd11_n002_snapshot_witness_source(tmp_path, make_loader, expanded):
    path = tmp_path / "page"
    path.write_bytes("café".encode("latin-1"))
    source, filename, fresh = make_loader(
        tmp_path, expanded, encoding="latin-1"
    ).get_source(Environment(), "page")
    assert source == "café" and os.path.samefile(filename, path) and fresh()
    path.unlink()
    assert fresh() is False


@pytest.mark.parametrize("expanded", [False, True])
def test_jin_a2920752fd11_n002_snapshot_witness_release(
    tmp_path, make_loader, expanded
):
    (tmp_path / "page").write_bytes(b"\xff")
    loader = make_loader(tmp_path, expanded)
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(12):
        with pytest.raises(UnicodeDecodeError):
            loader.get_source(Environment(), "page")
    assert len(os.listdir("/proc/self/fd")) <= before


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("follow", [False, True])
def test_jin_a2920752fd11_n003_snapshot_witness_listing(
    tmp_path, make_loader, expanded, follow
):
    root, linked = tmp_path / "root", tmp_path / "linked"
    root.mkdir()
    linked.mkdir()
    (root / "z").write_text("z")
    (root / "a").write_text("a")
    (linked / "child").write_text("child")
    (root / "nested").symlink_to(linked, target_is_directory=True)
    (root / "folder").mkdir()
    (root / "folder/page").write_text("page")
    expected = ["a", "folder/page", "z"]
    if follow:
        expected.insert(2, "nested/child")
    assert (
        make_loader([root, root], expanded, followlinks=follow).list_templates()
        == expected
    )


@pytest.mark.parametrize("kind", ["string", "pathlike", "mixed"])
def test_jin_a2920752fd11_n018_snapshot_witness_input_order(tmp_path, kind):
    first, second = tmp_path / "z-first", tmp_path / "a-second"
    for root, text in [(first, "first"), (second, "second")]:
        root.mkdir()
        (root / "page").write_text(text)
    roots = str(first) if kind == "string" else first
    if kind == "mixed":
        roots = [first, str(second)]
    kwargs = (
        {"expand_searchpath": True}
        if "expand_searchpath" in inspect.signature(FileSystemLoader).parameters
        else {}
    )
    assert (
        Environment(loader=FileSystemLoader(roots, **kwargs))
        .get_template("page")
        .render()
        == "first"
    )


def test_jin_a2920752fd11_n018_snapshot_witness_relative_cwd(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    for root, text in [(first, "first"), (second, "second")]:
        (root / "templates").mkdir(parents=True)
        (root / "templates/page").write_text(text)
    monkeypatch.chdir(first)
    kwargs = (
        {"expand_searchpath": True}
        if "expand_searchpath" in inspect.signature(FileSystemLoader).parameters
        else {}
    )
    loader = FileSystemLoader("templates", **kwargs)
    monkeypatch.chdir(second)
    assert Environment(loader=loader).get_template("page").render() == "second"
