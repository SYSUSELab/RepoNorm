import pytest

import dirty_equals as de


def test_DEQ_N01_comparison_repr():
    validator = de.IsRegex('abc', mode='search')
    original = repr(validator)
    value = 'xabc'
    assert validator == value
    assert repr(validator) == repr(value)
    assert not (validator == 'xyz')
    assert repr(validator) == original


@pytest.mark.parametrize('value', ['abc', 'xabc', 'xyz', None])
def test_DEQ_N02_boolean_composition(value):
    begins = value == de.IsRegex('abc')
    contains = value == de.IsRegex('abc', mode='search')
    assert (value == (de.IsRegex('abc') & de.IsStr())) == (begins and isinstance(value, str))
    assert (value == (de.IsRegex('abc') | de.IsRegex('xyz'))) == (begins or value == 'xyz')
    assert (value == ~de.IsRegex('abc', mode='search')) == (not contains)


def test_DEQ_N03_public_export():
    namespace = {}
    exec('from dirty_equals import *', namespace)
    assert namespace['IsRegex'] is de.IsRegex


@pytest.mark.parametrize('value', [123, b'abc', object(), None])
def test_DEQ_N08_invalid_values(value):
    validator = de.IsRegex('abc')
    assert (validator == value) is False
    assert (validator != value) is True


def test_DEQ_N09_value_lifecycle():
    validator = de.IsRegex('abc', mode='search')
    with pytest.raises(AttributeError):
        _ = validator.value
    value = 'xabc'
    assert validator == value
    assert validator.value is value
    assert not (validator == 'xyz')
    with pytest.raises(AttributeError):
        _ = validator.value
