from jinja2 import Environment
from jinja2.ext import InternationalizationExtension


def test_jin_896a62135bcc_n037_rebinding_preserves_extension_protocol():
    parent = Environment(extensions=["jinja2.ext.i18n"])
    parent.set_translation_version("message", "version")
    child = parent.overlay()
    key = InternationalizationExtension.identifier
    assert child.extensions[key] is not parent.extensions[key]
    assert child.extensions[key].environment is child
    assert parent.extensions[key].environment is parent
    grandchild = child.overlay()
    assert grandchild.extensions[key].environment is grandchild
    assert child.extensions[key].environment is child


def test_jin_896a62135bcc_n037_snapshot_witness():
    parent = Environment(extensions=["jinja2.ext.i18n"])
    key = InternationalizationExtension.identifier
    child = parent.overlay()
    assert parent.extensions[key].environment is parent
    assert child.extensions[key].environment is child
    assert child.extensions[key] is not parent.extensions[key]
