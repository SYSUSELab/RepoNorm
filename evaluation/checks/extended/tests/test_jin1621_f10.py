import os
import sys
import zipfile

import pytest

from jinja2 import BaseLoader
from jinja2 import BytecodeCache
from jinja2 import ChoiceLoader
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import FunctionLoader
from jinja2 import ModuleLoader
from jinja2 import PackageLoader
from jinja2 import PrefixLoader
from jinja2 import TemplateNotFound


def alias_request(loader, target, alias="public"):
    if hasattr(loader, "add_alias"):
        loader.add_alias(alias, target)
        return alias
    return target


@pytest.fixture
def package_factory(tmp_path, monkeypatch):
    names = []

    def make(archive):
        name = "n4c_norm_alias_zip" if archive else "n4c_norm_alias_directory"
        names.append(name)
        members = {
            "__init__.py": b"",
            "secret": b"outside",
            "templates/page": "café".encode("latin-1"),
            "templates/nested/a": b"a",
            "templates/z": b"z",
        }
        if archive:
            root = tmp_path / (name + ".zip")
            with zipfile.ZipFile(root, "w") as output:
                for path, content in members.items():
                    output.writestr(name + "/" + path, content)
                output.writestr(name + "/templates/empty/", b"")
            physical = None
        else:
            root = tmp_path
            for path, content in members.items():
                member = root / name / path
                member.parent.mkdir(parents=True, exist_ok=True)
                member.write_bytes(content)
            (root / name / "templates/empty").mkdir()
            physical = root / name / "templates/page"
        monkeypatch.syspath_prepend(str(root))
        return PackageLoader(name, encoding="latin-1"), physical

    yield make
    for name in names:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("kind", ["filesystem", "directory", "zip"])
def test_jin_a2920752fd11_n011_resolved_target_guard(tmp_path, package_factory, kind):
    if kind == "filesystem":
        root = tmp_path / "root"
        root.mkdir()
        (tmp_path / "secret").write_text("outside")
        loader = FileSystemLoader(root)
    else:
        loader, _ = package_factory(kind == "zip")
    request = alias_request(loader, "../secret")
    with pytest.raises(TemplateNotFound):
        Environment(loader=loader).get_template(request)


def test_jin_a2920752fd11_n002_source_and_freshness(tmp_path):
    path = tmp_path / "page"
    path.write_bytes("café".encode("latin-1"))
    loader = FileSystemLoader(tmp_path, encoding="latin-1")
    env = Environment(loader=loader)
    template = env.get_template(alias_request(loader, "page"))
    assert template.render() == "café" and os.path.samefile(template.filename, path)
    source, filename, fresh = loader.get_source(env, "page")
    assert source == "café" and os.path.samefile(filename, path) and fresh()
    path.unlink()
    assert fresh() is False


def test_jin_a2920752fd11_n002_release_on_decode_failure(tmp_path):
    (tmp_path / "page").write_bytes(b"\xff")
    loader = FileSystemLoader(tmp_path)
    env = Environment(loader=loader)
    request = alias_request(loader, "page")
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(12):
        with pytest.raises(UnicodeDecodeError):
            env.get_template(request)
    assert len(os.listdir("/proc/self/fd")) <= before


@pytest.mark.parametrize("follow", [False, True])
def test_jin_a2920752fd11_n003_fs_listing(tmp_path, follow):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "z").write_text("z")
    (root / "nested").mkdir()
    (root / "nested/a").write_text("a")
    (outside / "child").write_text("child")
    (root / "link").symlink_to(outside, target_is_directory=True)
    loader = FileSystemLoader([root, root], followlinks=follow)
    request = alias_request(loader, "nested/a")
    expected = {"nested/a", "z"}
    if follow:
        expected.add("link/child")
    expected.add(request)
    assert loader.list_templates() == sorted(expected)


@pytest.mark.parametrize("archive", [False, True])
def test_jin_a2920752fd11_n012_package_source(package_factory, archive):
    loader, physical = package_factory(archive)
    env = Environment(loader=loader)
    template = env.get_template(alias_request(loader, "page"))
    assert template.render() == "café"
    source, filename, fresh = loader.get_source(env, "page")
    assert source == "café" and template.filename == filename
    if archive:
        assert fresh is None
    else:
        assert os.path.samefile(filename, physical) and fresh()
        physical.unlink()
        assert fresh() is False


def test_jin_a2920752fd11_n013_original_cache_identity():
    loader = DictLoader({"one": "same", "two": "same", "stored": "same"})
    if hasattr(loader, "add_alias"):
        loader.add_alias("one", "stored")
        loader.add_alias("two", "stored")
    env = Environment(loader=loader)
    one, two = env.get_template("one"), env.get_template("two")
    assert one is not two
    assert env.get_template("one") is one
    assert env.get_template("two") is two
    env.loader = DictLoader({"one": "foreign"})
    assert env.get_template("one").render() == "foreign"


@pytest.mark.parametrize("kind", ["direct", "prefix", "choice"])
def test_jin_a2920752fd11_n014_sourceless_precompiled(tmp_path, kind):
    Environment(loader=DictLoader({"real": "{{ who }}"})).compile_templates(
        str(tmp_path), zip=None
    )
    child = ModuleLoader(tmp_path)
    name = alias_request(child, "real", "local")
    loader = child
    if kind == "prefix":
        loader = PrefixLoader({"pkg": child})
        name = "pkg/" + name
    elif kind == "choice":
        loader = ChoiceLoader([child])
    if kind != "direct":
        name = alias_request(loader, name, "top")
    assert (
        Environment(loader=loader).get_template(name, globals={"who": "ok"}).render()
        == "ok"
    )
    with pytest.raises(RuntimeError):
        child.get_source(Environment(), "real")


@pytest.mark.parametrize("kind", ["string", "tuple", "missing", "error"])
def test_jin_a2920752fd11_n016_function_provider(kind):
    seen = []

    def provider(name):
        seen.append(name)
        if kind == "error":
            raise OSError("provider failure")
        if kind == "missing":
            return None
        if kind == "tuple":
            return "{{ who }}", None, None
        return "{{ who }}"

    loader = FunctionLoader(provider)
    env = Environment(loader=loader)
    request = alias_request(loader, "real")
    if kind in {"missing", "error"}:
        with pytest.raises(TemplateNotFound if kind == "missing" else OSError):
            env.get_template(request)
    else:
        assert env.get_template(request, globals={"who": "ok"}).render() == "ok"
    assert seen == ["real"]


@pytest.mark.parametrize("kind", ["prefix", "choice"])
@pytest.mark.parametrize("reload", [False, True])
def test_jin_a2920752fd11_n017_custom_composite_protocol(kind, reload):
    state = {"source": "first {{ who }}"}

    class Source(BaseLoader):
        def __init__(self):
            self.state = state

        def get_source(self, environment, name):
            assert name == "page"
            value = self.state["source"]
            return value, None, lambda: value == self.state["source"]

    child = Source()
    loader, target = ChoiceLoader([child]), "page"
    if kind == "prefix":
        loader, target = PrefixLoader({"pkg": child}, delimiter=":"), "pkg:page"
    request = alias_request(loader, target)
    env = Environment(loader=loader, auto_reload=reload)
    first = env.get_template(request, globals={"who": "ok"})
    assert first.render() == "first ok"
    state["source"] = "second {{ who }}"
    latest = env.get_template(request, globals={"who": "ok"})
    assert latest.render() == ("second ok" if reload else "first ok")


def test_jin_a2920752fd11_n019_prefix_listing_delimiter_and_extension():
    class Child(BaseLoader):
        def __init__(self):
            self.names = ["z", "nested/a"]

        def list_templates(self):
            return list(self.names)

    child = Child()
    loader = PrefixLoader({"pkg": child}, delimiter=":")
    request = alias_request(loader, "pkg:nested/a")
    assert set(loader.list_templates()) == {"pkg:z", "pkg:nested/a", request}
    child.names.append("new")
    assert "pkg:new" in loader.list_templates()


@pytest.mark.parametrize("kind", ["base", "function", "module", "choice", "prefix"])
def test_jin_a2920752fd11_n020_unsupported_listing(tmp_path, kind):
    child = FunctionLoader(lambda name: "source")
    loader = child
    if kind == "base":
        loader = BaseLoader()
    elif kind == "module":
        loader = ModuleLoader(tmp_path)
    elif kind == "choice":
        loader = ChoiceLoader([child])
    elif kind == "prefix":
        loader = PrefixLoader({"pkg": child})
    alias_request(loader, "target")
    with pytest.raises(TypeError):
        loader.list_templates()


def test_jin_a2920752fd11_n022_live_dict_names():
    mapping = {"z": "z", "a": "a"}
    loader = DictLoader(mapping)
    request = alias_request(loader, "a")
    assert loader.list_templates() == sorted({"z", "a", request})
    mapping["new"] = "new"
    assert loader.list_templates() == sorted({"z", "a", "new", request})


@pytest.mark.parametrize("archive", [False, True])
def test_jin_a2920752fd11_n023_package_listing(package_factory, archive):
    loader, _ = package_factory(archive)
    request = alias_request(loader, "page")
    assert loader.list_templates() == sorted({"nested/a", "page", "z", request})


def test_jin_a2920752fd11_n024_choice_listing_union():
    first, second = {"z": "z", "a": "first"}, {"a": "second", "b": "b"}
    loader = ChoiceLoader([DictLoader(first), DictLoader(second)])
    request = alias_request(loader, "a")
    assert loader.list_templates() == sorted({"z", "a", "b", request})
    first["new"] = "new"
    assert loader.list_templates() == sorted({"z", "a", "b", "new", request})


def test_jin_a2920752fd11_n026_base_extension_and_bytecode_cache():
    class MemoryBytecode(BytecodeCache):
        def __init__(self):
            self.buckets = {}
            self.loads = self.dumps = 0

        def load_bytecode(self, bucket):
            self.loads += 1
            if bucket.key in self.buckets:
                bucket.bytecode_from_string(self.buckets[bucket.key])

        def dump_bytecode(self, bucket):
            self.dumps += 1
            self.buckets[bucket.key] = bucket.bytecode_to_string()

    class Source(BaseLoader):
        def __init__(self):
            self.value = "{{ who }}"

        def get_source(self, environment, name):
            assert name == "real"
            return self.value, None, None

    source, bytecode = Source(), MemoryBytecode()
    request = alias_request(source, "real")
    env = Environment(loader=source, cache_size=0, bytecode_cache=bytecode)
    assert env.get_template(request, globals={"who": "one"}).render() == "one"
    assert env.get_template(request, globals={"who": "two"}).render() == "two"
    assert bytecode.loads == 2 and bytecode.dumps == 1
