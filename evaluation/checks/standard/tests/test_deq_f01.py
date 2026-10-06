"""Task-specific checks for shared Dirty Equals validator contracts."""
from ipaddress import IPv4Address

import pytest

import dirty_equals as de


def test_DEQ_N01_comparison_repr():
    validator = de.IsIP(4)
    value = IPv4Address('192.0.2.1')
    assert validator == value
    assert repr(validator) == repr(value)
    assert not (validator == 'not an address')
    assert repr(validator) == repr(de.IsIP(4))


@pytest.mark.parametrize('value', ['192.0.2.1', '2001:db8::1', 'invalid', None])
def test_DEQ_N02_boolean_composition(value):
    ipv4 = value == de.IsIP(4)
    ipv6 = value == de.IsIP(6)
    assert (value == (de.IsIP(4) | de.IsIP(6))) == (ipv4 or ipv6)
    assert (value == (de.IsIP() & de.IsStr())) == (ipv4 or ipv6)
    assert (value == ~de.IsIP(4)) == (not ipv4)


def test_DEQ_N03_public_export():
    namespace = {}
    exec('from dirty_equals import *', namespace)
    assert namespace['IsIP'] is de.IsIP


@pytest.mark.parametrize('value', ['bad address', object(), [], None])
def test_DEQ_N08_invalid_values(value):
    validator = de.IsIP()
    assert (validator == value) is False
    assert (validator != value) is True


def test_DEQ_N09_value_lifecycle():
    validator = de.IsIP()
    with pytest.raises(AttributeError):
        _ = validator.value
    value = IPv4Address('192.0.2.1')
    assert validator == value
    assert validator.value is value
    assert not (validator == 'bad address')
    with pytest.raises(AttributeError):
        _ = validator.value
