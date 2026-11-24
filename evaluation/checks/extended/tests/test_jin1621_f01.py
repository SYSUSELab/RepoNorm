import os

import pytest

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import TemplateNotFound


@pytest.mark.parametrize("name", ["../secret", "sub/../../secret"])
def test_jin_a2920752fd11_n001_fallback_cannot_bypass_path_guard(tmp_path, name):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret").write_text("outside")
    loader = FileSystemLoader([], fallback_searchpath=root)
    with pytest.raises(TemplateNotFound) as caught:
        loader.get_source(Environment(), name)
    assert caught.value.name == name


def test_jin_a2920752fd11_n001_snapshot_witness(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret").write_text("outside")
    with pytest.raises(TemplateNotFound):
        FileSystemLoader(root).get_source(Environment(), "../secret")


def test_jin_a2920752fd11_n002_source_encoding_filename_and_freshness(tmp_path):
    source = tmp_path / "page"
    source.write_bytes("café".encode("latin-1"))
    loader = FileSystemLoader([], encoding="latin-1", fallback_searchpath=tmp_path)
    text, filename, current = loader.get_source(Environment(), "page")
    assert text == "café"
    assert os.path.samefile(filename, source)
    assert current() is True
    source.unlink()
    assert current() is False


def test_jin_a2920752fd11_n002_reads_release_files_even_on_decode_error(tmp_path):
    (tmp_path / "page").write_bytes(b"\xff")
    loader = FileSystemLoader([], fallback_searchpath=tmp_path)
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(12):
        with pytest.raises(UnicodeDecodeError):
            loader.get_source(Environment(), "page")
    assert len(os.listdir("/proc/self/fd")) <= before


def test_jin_a2920752fd11_n002_snapshot_witness(tmp_path):
    path = tmp_path / "page"
    path.write_bytes("café".encode("latin-1"))
    text, filename, current = FileSystemLoader(
        tmp_path, encoding="latin-1"
    ).get_source(Environment(), "page")
    assert text == "café" and os.path.samefile(filename, path)
    assert current() is True
    path.unlink()
    assert current() is False


@pytest.mark.parametrize("follow", [False, True])
def test_jin_a2920752fd11_n003_listing_keeps_logical_order_and_links(tmp_path, follow):
    root, linked = tmp_path / "root", tmp_path / "linked"
    root.mkdir()
    linked.mkdir()
    (root / "z").write_text("z")
    (root / "a").write_text("a")
    (linked / "child").write_text("child")
    (root / "nested").symlink_to(linked, target_is_directory=True)
    loader = FileSystemLoader(
        [root], followlinks=follow, fallback_searchpath=[root]
    )
    expected = ["a", "nested/child", "z"] if follow else ["a", "z"]
    assert loader.list_templates() == expected


def test_jin_a2920752fd11_n003_snapshot_witness(tmp_path):
    (tmp_path / "z").write_text("z")
    (tmp_path / "a").write_text("a")
    assert FileSystemLoader([tmp_path, tmp_path]).list_templates() == ["a", "z"]
