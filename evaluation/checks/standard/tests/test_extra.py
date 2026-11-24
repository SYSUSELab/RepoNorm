"""Additional public extension contracts for the IsHash task."""
import pytest
import dirty_equals as de


def test_DEQ_N03_public_export():
    namespace = {}
    exec('from dirty_equals import *', namespace)
    assert namespace['IsHash'] is de.IsHash


@pytest.mark.parametrize('value', [None, 15, object(), [], {}, b'a' * 32])
def test_DEQ_N08_invalid_values(value):
    validator = de.IsHash('md5')
    assert (validator == value) is False
    assert (validator != value) is True


def test_DEQ_N09_value_lifecycle():
    validator = de.IsHash('md5')
    with pytest.raises(AttributeError):
        _ = validator.value
    value = 'Ab' * 16
    assert validator == value
    assert validator.value is value
    assert not (validator == 'bad')
    with pytest.raises(AttributeError):
        _ = validator.value


@pytest.mark.parametrize('error', [TypeError, ValueError])
def test_DEQ_N08_comparison_exceptions(error):
    class BrokenLength(str):
        def __len__(self):
            raise error('not a comparable digest')

    value = BrokenLength('a' * 32)
    assert (de.IsHash('md5') == value) is False
    assert (de.IsHash('md5') != value) is True
