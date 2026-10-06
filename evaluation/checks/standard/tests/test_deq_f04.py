"""Task-specific checks for shared Dirty Equals validator contracts."""
import pytest

import dirty_equals as de


def test_DEQ_N01_comparison_repr():
    validator = de.IsURL('https')
    value = 'https://example.com'
    assert validator == value
    assert repr(validator) == repr(value)
    assert not (validator == 'invalid')
    assert repr(validator) == repr(de.IsURL('https'))


@pytest.mark.parametrize('value', ['https://example.com', 'ftp://example.com', 'invalid', None])
def test_DEQ_N02_boolean_composition(value):
    https = value == de.IsURL('https')
    ftp = value == de.IsURL('ftp')
    assert (value == (de.IsURL('https') | de.IsURL('ftp'))) == (https or ftp)
    assert (value == (de.IsURL() & de.IsStr())) == (https or ftp)
    assert (value == ~de.IsURL('https')) == (not https)


def test_DEQ_N03_public_export():
    namespace = {}
    exec('from dirty_equals import *', namespace)
    assert namespace['IsURL'] is de.IsURL


@pytest.mark.parametrize('value', ['invalid', 'https:///path', object(), None])
def test_DEQ_N08_invalid_values(value):
    validator = de.IsURL()
    assert (validator == value) is False
    assert (validator != value) is True


def test_DEQ_N09_value_lifecycle():
    validator = de.IsURL()
    with pytest.raises(AttributeError):
        _ = validator.value
    value = 'https://example.com'
    assert validator == value
    assert validator.value is value
    assert not (validator == 'invalid')
    with pytest.raises(AttributeError):
        _ = validator.value
