"""Task-specific checks for shared Dirty Equals validator contracts."""
import pytest

import dirty_equals as de


def test_DEQ_N01_comparison_repr():
    validator = de.IsMac('colon')
    value = '00:1B:44:11:3A:B7'
    assert validator == value
    assert repr(validator) == repr(value)
    assert not (validator == 'invalid')
    assert repr(validator) == repr(de.IsMac('colon'))


@pytest.mark.parametrize('value', ['00:1B:44:11:3A:B7', '001b44113ab7', 'invalid', None])
def test_DEQ_N02_boolean_composition(value):
    colon = value == de.IsMac('colon')
    compact = value == de.IsMac('compact')
    assert (value == (de.IsMac('colon') | de.IsMac('compact'))) == (colon or compact)
    assert (value == (de.IsMac() & de.IsStr())) == (colon or compact)
    assert (value == ~de.IsMac('colon')) == (not colon)


def test_DEQ_N03_public_export():
    namespace = {}
    exec('from dirty_equals import *', namespace)
    assert namespace['IsMac'] is de.IsMac


@pytest.mark.parametrize('value', ['not-mac', '00:11-22:33-44:55', object(), None])
def test_DEQ_N08_invalid_values(value):
    validator = de.IsMac()
    assert (validator == value) is False
    assert (validator != value) is True


def test_DEQ_N09_value_lifecycle():
    validator = de.IsMac()
    with pytest.raises(AttributeError):
        _ = validator.value
    value = '001b44113ab7'
    assert validator == value
    assert validator.value is value
    assert not (validator == 'invalid')
    with pytest.raises(AttributeError):
        _ = validator.value
