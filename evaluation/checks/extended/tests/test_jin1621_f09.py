import inspect
import os

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import TemplateNotFound


@pytest.fixture
def make_loader():
    loaders = []

    def make(roots, enabled=False, **kwargs):
        if "auto_reload" in inspect.signature(FileSystemLoader).parameters:
            kwargs["auto_reload"] = enabled
        loader = FileSystemLoader(roots, **kwargs)
        loaders.append(loader)
        return loader

    yield make
    for loader in loaders:
        if hasattr(loader, "close"):
            loader.close()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("name", ["../secret", "sub/../../secret"])
def test_jin_a2920752fd11_n001_guard(tmp_path, make_loader, enabled, name):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret").write_text("outside")
    with pytest.raises(TemplateNotFound) as caught:
        make_loader(root, enabled).get_source(Environment(), name)
    assert caught.value.name == name


@pytest.mark.parametrize("enabled", [False, True])
def test_jin_a2920752fd11_n002_source(tmp_path, make_loader, enabled):
    path = tmp_path / "page"
    path.write_bytes("café".encode("latin-1"))
    loader = make_loader(tmp_path, enabled, encoding="latin-1")
    source, filename, fresh = loader.get_source(Environment(), "page")
    assert source == "café" and os.path.samefile(filename, path) and fresh()
    path.unlink()
    assert fresh() is False


@pytest.mark.parametrize("enabled", [False, True])
def test_jin_a2920752fd11_n002_release(tmp_path, make_loader, enabled):
    (tmp_path / "page").write_bytes(b"\xff")
    loader = make_loader(tmp_path, enabled)
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(12):
        with pytest.raises(UnicodeDecodeError):
            loader.get_source(Environment(), "page")
    assert len(os.listdir("/proc/self/fd")) <= before


@pytest.mark.parametrize("kind", ["string", "pathlike", "mixed"])
def test_jin_a2920752fd11_n018_input_order(tmp_path, make_loader, kind):
    first, second = tmp_path / "z-first", tmp_path / "a-second"
    for root, text in [(first, "first"), (second, "second")]:
        root.mkdir()
        (root / "page").write_text(text)
    roots = str(first) if kind == "string" else first
    if kind == "mixed":
        roots = [first, str(second)]
    assert (
        Environment(loader=make_loader(roots)).get_template("page").render() == "first"
    )


def test_jin_a2920752fd11_n018_relative_cwd(tmp_path, monkeypatch, make_loader):
    first, second = tmp_path / "first", tmp_path / "second"
    for root, text in [(first, "first"), (second, "second")]:
        (root / "templates").mkdir(parents=True)
        (root / "templates/page").write_text(text)
    monkeypatch.chdir(first)
    loader = make_loader("templates")
    monkeypatch.chdir(second)
    assert Environment(loader=loader).get_template("page").render() == "second"


@pytest.mark.parametrize("size", [0, -1, 2])
def test_jin_a2920752fd11_n025_cache_modes(tmp_path, make_loader, size):
    (tmp_path / "page").write_text("page")
    loader = make_loader(tmp_path, True)
    env = Environment(loader=loader, cache_size=size, auto_reload=False)
    first = env.get_template("page")
    assert (env.get_template("page") is first) == (size != 0)
    if hasattr(loader, "invalidate_changed"):
        loader.invalidate_changed([str(tmp_path / "missing")])
    assert (env.get_template("page") is first) == (size != 0)
    if size == 0:
        assert env.cache is None


def test_jin_a2920752fd11_n025_target_isolation(tmp_path, make_loader):
    for name in ["page", "untouched"]:
        (tmp_path / name).write_text(name)
    loader = make_loader(tmp_path, True)
    other = DictLoader({"page": "foreign"})
    env = Environment(loader=loader, cache_size=-1, auto_reload=False)
    env.get_template("page")
    untouched = env.get_template("untouched")
    env.loader = other
    foreign = env.get_template("page")
    env.loader = loader
    if hasattr(loader, "invalidate_changed"):
        loader.invalidate_changed([str(tmp_path / "page")])
    assert env.get_template("untouched") is untouched
    env.loader = other
    assert env.get_template("page") is foreign


def test_jin_a2920752fd11_n025_hit_globals(tmp_path, make_loader):
    (tmp_path / "page").write_text("{{ value }}")
    env = Environment(loader=make_loader(tmp_path, True))
    env.globals["value"] = "env"
    first = env.get_template("page", globals={"value": "first"})
    assert first.render() == "first"
    assert env.get_template("page", globals={"value": "second"}) is first
    assert first.render() == "second"
    assert env.globals["value"] == "env"


def test_jin_a2920752fd11_n025_lru_capacity(tmp_path, make_loader):
    for name in ["one", "two", "three"]:
        (tmp_path / name).write_text(name)
    loader = make_loader(tmp_path, True)
    env = Environment(loader=loader, cache_size=2, auto_reload=False)
    first = env.get_template("one")
    second = env.get_template("two")
    assert env.get_template("one") is first
    if hasattr(loader, "invalidate_changed"):
        loader.invalidate_changed([str(tmp_path / "missing")])
    env.get_template("three")
    assert env.get_template("one") is first
    assert env.get_template("two") is not second
