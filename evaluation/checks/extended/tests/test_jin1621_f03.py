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
    text, filename, current = FileSystemLoader(tmp_path, encoding="latin-1").get_source(
        Environment(), "page"
    )
    assert text == "café" and os.path.samefile(filename, path)
    assert current() is True
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2000000000))
    assert current() is False
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
        stamp = path.stat()
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2000000000))
        assert current() is False
        path.unlink()
        assert current() is False


@pytest.mark.parametrize("archive", [False, True])
@pytest.mark.parametrize("kind", ["module", "prefix", "choice"])
def test_jin_a2920752fd11_n014_snapshot_witness_compiled_without_source(
    tmp_path, archive, kind
):
    from jinja2 import loaders

    path = tmp_path / "compiled.zip" if archive else tmp_path / "compiled"
    source = Environment(loader=loaders.DictLoader({"page": "Hi {{ who }}"}))
    source.compile_templates(str(path), zip="deflated" if archive else None)
    child = loaders.ModuleLoader(path)
    with pytest.raises(RuntimeError):
        child.get_source(Environment(), "page")
    loader, name = child, "page"
    if kind == "prefix":
        loader, name = loaders.PrefixLoader({"site": child}), "site/page"
    elif kind == "choice":
        loader = loaders.ChoiceLoader([loaders.DictLoader({}), child])
    assert (
        Environment(loader=loader).get_template(name).render(who="world") == "Hi world"
    )


@pytest.mark.parametrize(
    "name,pieces",
    [("foo/bar", ["foo", "bar"]), ("./foo//bar", ["foo", "bar"]), ("./", [])],
)
def test_jin_a2920752fd11_n015_snapshot_witness_public_path_split(name, pieces):
    from jinja2.loaders import split_template_path

    assert split_template_path(name) == pieces


def test_jin_a2920752fd11_n015_snapshot_witness_additional_parent_check():
    from jinja2.loaders import split_template_path

    for name in ("../page", "foo/../page"):
        with pytest.raises(TemplateNotFound) as caught:
            split_template_path(name)
        assert caught.value.name == name


@pytest.mark.parametrize(
    "mode", ["source", "tuple-none", "tuple-callback", "tuple-file"]
)
def test_jin_a2920752fd11_n016_snapshot_witness_function_source_protocol(mode):
    from jinja2 import FunctionLoader

    seen = []

    def provider(name):
        seen.append(name)
        if mode != "source":
            return (
                "hello",
                "provider.tpl" if mode == "tuple-file" else None,
                (lambda: False) if mode in {"tuple-callback", "tuple-file"} else None,
            )
        return "hello"

    text, filename, current = FunctionLoader(provider).get_source(Environment(), "page")
    assert text == "hello"
    assert filename == ("provider.tpl" if mode == "tuple-file" else None)
    assert seen and all(n == "page" for n in seen)
    if mode in {"tuple-callback", "tuple-file"}:
        assert current() is False
    else:
        assert current is None


def test_jin_a2920752fd11_n016_snapshot_witness_provider_missing_and_error():
    from jinja2 import FunctionLoader

    with pytest.raises(TemplateNotFound) as caught:
        FunctionLoader(lambda name: None).get_source(Environment(), "page")
    assert caught.value.name == "page"

    def provider(name):
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        FunctionLoader(provider).get_source(Environment(), "page")
