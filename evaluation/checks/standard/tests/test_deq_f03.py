"""Task-specific checks for shared Dirty Equals validator contracts."""
import pytest

import dirty_equals as de


def test_DEQ_N01_comparison_repr():
    validator = de.IsEmail('example.com')
    value = 'a@example.com'
    assert validator == value
    assert repr(validator) == repr(value)
    assert not (validator == 'invalid')
    assert repr(validator) == repr(de.IsEmail('example.com'))


@pytest.mark.parametrize('value', ['a@example.com', 'a@other.org', 'invalid', None])
def test_DEQ_N02_boolean_composition(value):
    example = value == de.IsEmail('example.com')
    other = value == de.IsEmail('other.org')
    assert (value == (de.IsEmail('example.com') | de.IsEmail('other.org'))) == (example or other)
    assert (value == (de.IsEmail() & de.IsStr())) == (example or other)
    assert (value == ~de.IsEmail('example.com')) == (not example)


def test_DEQ_N03_public_export():
    namespace = {}
    exec('from dirty_equals import *', namespace)
    assert namespace['IsEmail'] is de.IsEmail


@pytest.mark.parametrize('value', ['invalid', 'a..b@example.com', object(), None])
def test_DEQ_N08_invalid_values(value):
    validator = de.IsEmail()
    assert (validator == value) is False
    assert (validator != value) is True


def test_DEQ_N09_value_lifecycle():
    validator = de.IsEmail()
    with pytest.raises(AttributeError):
        _ = validator.value
    value = 'a@example.com'
    assert validator == value
    assert validator.value is value
    assert not (validator == 'invalid')
    with pytest.raises(AttributeError):
        _ = validator.value
