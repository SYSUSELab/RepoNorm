import sys
import zipfile

import pytest

from jinja2 import Environment
from jinja2 import loaders


@pytest.mark.parametrize("follow", [False, True])
def test_jin_a2920752fd11_n003_snapshot_witness_fs_listing(tmp_path, follow):
    root, linked = tmp_path / "root", tmp_path / "linked"
    root.mkdir()
    linked.mkdir()
    (root / "z").write_text("z")
    (root / "a").write_text("a")
    (linked / "child").write_text("child")
    (root / "nested").symlink_to(linked, target_is_directory=True)
    expected = ["a", "nested/child", "z"] if follow else ["a", "z"]
    assert (
        loaders.FileSystemLoader([root, root], followlinks=follow).list_templates()
        == expected
    )


@pytest.fixture
def custom_children():
    class Child(loaders.BaseLoader):
        def __init__(self, names):
            self.names = names

        def list_templates(self):
            return self.names[:]

    return Child(["z", "a"]), Child(["m", "a"])


def test_jin_a2920752fd11_n019_snapshot_witness_prefix(custom_children):
    z, a = custom_children
    parent = loaders.PrefixLoader({"z": z, "a": a}, delimiter=":")
    assert set(parent.list_templates()) == {"z:z", "z:a", "a:m", "a:a"}
    z.names.append("new")
    assert set(parent.list_templates()) == {"z:z", "z:a", "z:new", "a:m", "a:a"}


def test_jin_a2920752fd11_n024_snapshot_witness_choice(custom_children):
    z, a = custom_children
    parent = loaders.ChoiceLoader([z, a])
    assert parent.list_templates() == ["a", "m", "z"]
    z.names.append("new")
    assert parent.list_templates() == ["a", "m", "new", "z"]


@pytest.mark.parametrize("kind", ["base", "function", "module"])
@pytest.mark.parametrize("entry", ["loader", "environment", "choice", "prefix"])
def test_jin_a2920752fd11_n020_snapshot_witness_unsupported(tmp_path, kind, entry):
    child = loaders.BaseLoader()
    if kind == "function":
        child = loaders.FunctionLoader(lambda name: "ok")
    elif kind == "module":
        child = loaders.ModuleLoader(tmp_path)
    parent = child
    if entry == "choice":
        parent = loaders.ChoiceLoader([loaders.DictLoader({"a": "a"}), child])
    elif entry == "prefix":
        parent = loaders.PrefixLoader({"site": child})
    with pytest.raises(TypeError):
        if entry == "environment":
            Environment(loader=parent).list_templates()
        else:
            parent.list_templates()


def test_jin_a2920752fd11_n021_snapshot_witness_environment():
    loader = loaders.DictLoader({"z.html": "Z", "a.txt": "A", "b.html": "B"})
    env = Environment(loader=loader)
    seen = []
    assert env.list_templates(
        filter_func=lambda name: seen.append(name) or name == "b.html"
    ) == ["b.html"]
    assert seen == ["a.txt", "b.html", "z.html"]
    assert env.list_templates(extensions=["html"]) == ["b.html", "z.html"]
    assert env.list_templates() == ["a.txt", "b.html", "z.html"]
    with pytest.raises(TypeError):
        env.list_templates(extensions=[], filter_func=lambda name: True)
    error = RuntimeError("one-shot")

    def bad(name):
        raise error

    with pytest.raises(RuntimeError) as caught:
        env.list_templates(filter_func=bad)
    assert caught.value is error


def test_jin_a2920752fd11_n022_snapshot_witness_dict():
    mapping = {"z": "Z", "folder/a": "A"}
    loader = loaders.DictLoader(mapping)
    assert loader.list_templates() == ["folder/a", "z"]
    mapping["m"] = "M"
    mapping.pop("z")
    assert loader.list_templates() == ["folder/a", "m"]


@pytest.mark.parametrize("archive", [False, True])
def test_jin_a2920752fd11_n023_snapshot_witness_package(tmp_path, monkeypatch, archive):
    name = "n4c_listing_package_" + str(int(archive))
    if archive:
        root = tmp_path / (name + ".zip")
        with zipfile.ZipFile(root, "w") as z:
            z.writestr(name + "/__init__.py", "")
            z.writestr(name + "/outside", "ignore")
            z.writestr(name + "/templates/empty/", "")
            z.writestr(name + "/templates/z", "z")
            z.writestr(name + "/templates/folder/a", "a")
    else:
        root = tmp_path
        package = root / name
        (package / "templates/folder").mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "outside").write_text("ignore")
        (package / "templates/z").write_text("z")
        (package / "templates/folder/a").write_text("a")
        (package / "templates/empty").mkdir()
    monkeypatch.syspath_prepend(str(root))
    try:
        assert loaders.PackageLoader(name).list_templates() == ["folder/a", "z"]
    finally:
        sys.modules.pop(name, None)
