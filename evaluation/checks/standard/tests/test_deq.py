"""Public behavior only; no reference implementation layout assertions."""
import pytest
from dirty_equals import IsHash, IsStr


@pytest.mark.parametrize('algorithm,size', [('md5', 32), ('SHA256', 64)])
def test_DEQ_N01_comparison_repr(algorithm, size):
    value = 'A1' * (size // 2)
    validator = IsHash(algorithm)
    assert value == validator
    assert repr(validator) == repr(value)
    assert not (validator == 'not a hash')
    assert repr(validator) == f'IsHash({algorithm!r})'
    assert validator == value.lower()
    assert repr(validator) == repr(value.lower())


@pytest.mark.parametrize('value', ['A1' * 16, 'b2' * 32, 'invalid', None, 17])
def test_DEQ_N02_boolean_composition(value):
    left = value == IsHash('md5')
    right = value == IsHash('sha256')
    is_string = value == IsStr()
    assert (value == (IsHash('md5') | IsHash('sha256'))) == (left or right)
    assert (value == (IsHash('md5') & IsStr())) == (left and is_string)
    assert (value == ~IsHash('md5')) == (not left)
    # Composition also works with the new validator on the right.
    assert (value == (IsStr() & IsHash('md5'))) == (is_string and left)
