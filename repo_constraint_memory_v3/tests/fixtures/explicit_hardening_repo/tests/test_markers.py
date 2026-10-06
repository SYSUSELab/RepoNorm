import pytest


def test_deprecated_behavior_remains_supported() -> None:
    assert True


@pytest.mark.slow
def test_slow_path() -> None:
    assert True


@pytest.mark.integration(reason="call form")
def test_integration_path() -> None:
    assert True
