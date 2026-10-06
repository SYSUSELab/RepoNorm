import pytest

import dirty_equals as de


def test_DEQ_N01_comparison_repr():
    validator = de.IsColor('hex')
    original = repr(validator)
    value = '#abc'
    assert validator == value
    assert repr(validator) == repr(value)
    assert not (validator == 'not-a-color')
    assert repr(validator) == original


@pytest.mark.parametrize('value', ['#abc', 'rgb(0,0,0)', 'not-a-color', None])
def test_DEQ_N02_boolean_composition(value):
    is_hex = value == de.IsColor('hex')
    is_rgb = value == de.IsColor('rgb')
    assert (value == (de.IsColor() & de.IsStr())) == ((value == de.IsColor()) and isinstance(value, str))
    assert (value == (de.IsColor('hex') | de.IsColor('rgb'))) == (is_hex or is_rgb)
    assert (value == ~de.IsColor()) == (not (value == de.IsColor()))


def test_DEQ_N03_public_export():
    namespace = {}
    exec('from dirty_equals import *', namespace)
    assert namespace['IsColor'] is de.IsColor


@pytest.mark.parametrize('value', [123, b'#abc', object(), None])
def test_DEQ_N08_invalid_values(value):
    validator = de.IsColor()
    assert (validator == value) is False
    assert (validator != value) is True


def test_DEQ_N09_value_lifecycle():
    validator = de.IsColor('rgb')
    with pytest.raises(AttributeError):
        _ = validator.value
    value = 'rgb(0,0,0)'
    assert validator == value
    assert validator.value is value
    assert not (validator == 'not-a-color')
    with pytest.raises(AttributeError):
        _ = validator.value
