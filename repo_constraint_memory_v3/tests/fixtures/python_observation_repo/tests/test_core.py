import pytest as pt
from pytest import mark as pm


@pt.mark.slow
def test_slow():
    assert True


@pt.mark.integration()
async def test_integration():
    assert True


@pm.custom
def test_custom():
    assert True


@getattr(pt.mark, "dynamic")
def test_dynamic_marker():
    assert True
