"""Test that setuptools-scm is working correctly"""
import casanovo

def test_version():
    """Check that the version is not None"""
    assert casanovo.__version__ is not None
