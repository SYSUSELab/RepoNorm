import os

import pytest

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import TemplateNotFound


def make_loader(path, size, **kwargs):
    if size:
        kwargs["path_cache_size"] = size
    return FileSystemLoader(path, **kwargs)


@pytest.mark.parametrize("size", [0, 2], ids=["snapshot", "cache"])
def test_jin_a2920752fd11_n001_snapshot_witness_parent_guard(tmp_path, size):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret").write_text("outside")
    loader = make_loader(root, size)
    for name in ("../secret", "sub/../../secret"):
        with pytest.raises(TemplateNotFound) as caught:
            loader.get_source(Environment(), name)
        assert caught.value.name == name


@pytest.mark.parametrize("size", [0, 2], ids=["snapshot", "cache"])
def test_jin_a2920752fd11_n002_snapshot_witness_source_protocol(tmp_path, size):
    path = tmp_path / "page"
    path.write_bytes("café".encode("latin-1"))
    loader = make_loader(tmp_path, size, encoding="latin-1")
    for _ in range(2):
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
