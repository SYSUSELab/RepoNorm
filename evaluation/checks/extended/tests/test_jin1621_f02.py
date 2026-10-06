import os
import sys
import zipfile

import pytest

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import PackageLoader
from jinja2 import TemplateNotFound


@pytest.fixture
def package_factory(tmp_path, monkeypatch):
    names = []

    def make(archive, label):
        name = "n4c_norm_paths_" + label
        names.append(name)
        if archive:
            root = tmp_path / (name + ".zip")
            with zipfile.ZipFile(root, "w") as z:
                z.writestr(name + "/__init__.py", "")
                z.writestr(name + "/secret", "outside")
                z.writestr(name + "/templates/page", "café".encode("latin-1"))
            physical = None
        else:
            root = tmp_path
            physical = root / name / "templates" / "page"
            physical.parent.mkdir(parents=True)
            (physical.parent.parent / "__init__.py").write_text("")
            (physical.parent.parent / "secret").write_text("outside")
            physical.write_bytes("café".encode("latin-1"))
        monkeypatch.syspath_prepend(str(root))
        return PackageLoader(name, encoding="latin-1"), physical

    yield make
    for name in names:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("kind", ["filesystem", "directory", "archive"])
def test_jin_a2920752fd11_n011_snapshot_witness_logical_parent_root_guard(
    tmp_path, package_factory, kind
):
    if kind == "filesystem":
        root = tmp_path / "root"
        root.mkdir()
        (tmp_path / "secret").write_text("outside")
        loader = FileSystemLoader(root)
    else:
        loader, _ = package_factory(kind == "archive", kind)
    for name in ("../secret", "child/../../secret"):
        with pytest.raises(TemplateNotFound) as caught:
            loader.get_source(Environment(), name)
        assert caught.value.name == name


def test_jin_a2920752fd11_n002_snapshot_witness_filesystem_source(tmp_path):
    path = tmp_path / "page"
    path.write_bytes("café".encode("latin-1"))
    text, filename, current = FileSystemLoader(
        tmp_path, encoding="latin-1"
    ).get_source(Environment(), "page")
    assert text == "café" and os.path.samefile(filename, path)
    assert current() is True
    path.unlink()
    assert current() is False


def test_jin_a2920752fd11_n002_snapshot_witness_files_release_on_decode_error(tmp_path):
    (tmp_path / "page").write_bytes(b"\xff")
    loader = FileSystemLoader(tmp_path)
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(12):
        with pytest.raises(UnicodeDecodeError):
            loader.get_source(Environment(), "page")
    assert len(os.listdir("/proc/self/fd")) <= before


@pytest.mark.parametrize("archive", [False, True])
def test_jin_a2920752fd11_n012_snapshot_witness_package_source_protocol(
    package_factory, archive
):
    loader, path = package_factory(archive, "source")
    text, filename, current = loader.get_source(Environment(), "page")
    assert text == "café"
    assert filename.endswith("templates/page")
    if archive:
        assert current is None
    else:
        assert os.path.samefile(filename, path)
        assert current() is True
        path.unlink()
        assert current() is False


def test_jin_a2920752fd11_n013_snapshot_witness_public_name_cache_identity(tmp_path):
    (tmp_path / "page").write_text("hello")
    env = Environment(loader=FileSystemLoader(tmp_path))
    plain, dotted = env.get_template("page"), env.get_template("./page")
    assert plain.render() == dotted.render() == "hello"
    assert dotted is not plain
    assert env.get_template("page") is plain
    assert env.get_template("./page") is dotted
    env.loader = FileSystemLoader(tmp_path)
    assert env.get_template("page") is not plain
