import os

import pytest

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import TemplateNotFound


@pytest.mark.parametrize("weighted", [False, True], ids=["snapshot", "priority"])
def test_jin_a2920752fd11_n001_snapshot_witness_parent_guard(tmp_path, weighted):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret").write_text("outside")
    loader = FileSystemLoader([(root, 1)] if weighted else root)
    for name in ("../secret", "sub/../../secret"):
        with pytest.raises(TemplateNotFound) as caught:
            loader.get_source(Environment(), name)
        assert caught.value.name == name


@pytest.mark.parametrize("weighted", [False, True], ids=["snapshot", "priority"])
def test_jin_a2920752fd11_n002_snapshot_witness_source_protocol(tmp_path, weighted):
    path = tmp_path / "page"
    path.write_bytes("café".encode("latin-1"))
    loader = FileSystemLoader(
        [(tmp_path, 2)] if weighted else tmp_path, encoding="latin-1"
    )
    text, filename, current = loader.get_source(Environment(), "page")
    assert text == "café" and os.path.samefile(filename, path)
    assert current() is True
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2000000000))
    assert current() is False
    path.unlink()
    assert current() is False


def test_jin_a2920752fd11_n002_snapshot_witness_decode_releases_files(tmp_path):
    (tmp_path / "page").write_bytes(b"\xff")
    loader = FileSystemLoader(tmp_path)
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(12):
        with pytest.raises(UnicodeDecodeError):
            loader.get_source(Environment(), "page")
    assert len(os.listdir("/proc/self/fd")) <= before


@pytest.mark.parametrize("weighted", [False, True], ids=["snapshot", "priority"])
@pytest.mark.parametrize("follow", [False, True])
def test_jin_a2920752fd11_n003_snapshot_witness_listing(tmp_path, weighted, follow):
    root, linked = tmp_path / "root", tmp_path / "linked"
    root.mkdir()
    linked.mkdir()
    (root / "z").write_text("z")
    (root / "a").write_text("a")
    (linked / "child").write_text("child")
    (root / "nested").symlink_to(linked, target_is_directory=True)
    paths = [(root, 1), (root, -1)] if weighted else [root, root]
    loader = FileSystemLoader(paths, followlinks=follow)
    expected = ["a", "nested/child", "z"] if follow else ["a", "z"]
    assert loader.list_templates() == expected
