import pytest

import dirty_equals as de


def test_DEQ_N01_comparison_repr():
    validator = de.IsPhoneNumber('US')
    value = '555-123-4567'
    assert validator == value
    assert repr(validator) == repr(value)
    assert not (validator == 'invalid')
    assert repr(validator) == repr(de.IsPhoneNumber('US'))


@pytest.mark.parametrize('value', ['555-123-4567', '+44 20 7946 0958', 'invalid', None])
def test_DEQ_N02_boolean_composition(value):
    us = value == de.IsPhoneNumber('US')
    gb = value == de.IsPhoneNumber('GB')
    assert (value == (de.IsPhoneNumber('US') | de.IsPhoneNumber('GB'))) == (us or gb)
    assert (value == (de.IsPhoneNumber() & de.IsStr())) == (us or gb)
    assert (value == ~de.IsPhoneNumber('US')) == (not us)


def test_DEQ_N03_public_export():
    namespace = {}
    exec('from dirty_equals import *', namespace)
    assert namespace['IsPhoneNumber'] is de.IsPhoneNumber


@pytest.mark.parametrize('value', ['invalid', '+1\n5551234567', object(), None])
def test_DEQ_N08_invalid_values(value):
    validator = de.IsPhoneNumber()
    assert (validator == value) is False
    assert (validator != value) is True


def test_DEQ_N09_value_lifecycle():
    validator = de.IsPhoneNumber()
    with pytest.raises(AttributeError):
        _ = validator.value
    value = '+1 555 123 4567'
    assert validator == value
    assert validator.value is value
    assert not (validator == 'invalid')
    with pytest.raises(AttributeError):
        _ = validator.value
