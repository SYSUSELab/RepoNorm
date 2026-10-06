import pytest

import dirty_equals as de


def test_DEQ_N01_comparison_repr():
    validator = de.IsCreditCard('Visa')
    value = '4111111111111111'
    assert validator == value
    assert repr(validator) == repr(value)
    assert not (validator == 'invalid')
    assert repr(validator) == repr(de.IsCreditCard('Visa'))


@pytest.mark.parametrize('value', ['4111111111111111', '5555555555554444', 'invalid', None])
def test_DEQ_N02_boolean_composition(value):
    visa = value == de.IsCreditCard('Visa')
    mastercard = value == de.IsCreditCard('Mastercard')
    assert (value == (de.IsCreditCard() & de.IsStr())) == (
        (value == de.IsCreditCard()) and (value == de.IsStr())
    )
    assert (value == (de.IsCreditCard('Visa') | de.IsCreditCard('Mastercard'))) == (visa or mastercard)
    assert (value == ~de.IsCreditCard('Visa')) == (not visa)


def test_DEQ_N03_public_export():
    namespace = {}
    exec('from dirty_equals import *', namespace)
    assert namespace['IsCreditCard'] is de.IsCreditCard


@pytest.mark.parametrize('value', ['invalid', '４１１１１１１１１１１１１１１１', object(), None])
def test_DEQ_N08_invalid_values(value):
    validator = de.IsCreditCard()
    assert (validator == value) is False
    assert (validator != value) is True


def test_DEQ_N09_value_lifecycle():
    validator = de.IsCreditCard()
    with pytest.raises(AttributeError):
        _ = validator.value
    value = 4111111111111111
    assert validator == value
    assert validator.value is value
    assert not (validator == 'invalid')
    with pytest.raises(AttributeError):
        _ = validator.value
