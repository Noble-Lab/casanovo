"""Test that setuptools-scm is working correctly"""
import casanovo
from ..denovo.model import Spec2Pep

def test_version():
    """Check that the version is not None"""
    assert casanovo.__version__ is not None
    
def test_tensorboard():
    """Check that the version is not None"""
    model = Spec2Pep(
        tb_summarywriter="test_path",
    )
    assert model.tb_summarywriter is not None
    
    model = Spec2Pep()
    assert model.tb_summarywriter is None
