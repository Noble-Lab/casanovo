"""Unit tests specifically for the model_runner module."""
from pathlib import Path

import pytest

from casanovo.config import Config
from casanovo.denovo.model_runner import ModelRunner


def test_initialize_model(tmp_path):
    """Test that"""
    config = Config()
    config.train_from_scratch = False
    ModelRunner(config=config).initialize_model(train=True)

    with pytest.raises(ValueError):
        ModelRunner(config=config).initialize_model(train=False)

    with pytest.raises(FileNotFoundError):
        runner = ModelRunner(config=config, model_filename="blah")
        runner.initialize_model(train=True)

    with pytest.raises(FileNotFoundError):
        runner = ModelRunner(config=config, model_filename="blah")
        runner.initialize_model(train=False)

    # This should work now:
    config.train_from_scratch = True
    runner = ModelRunner(config=config, model_filename="blah")
    runner.initialize_model(train=True)

    # But this should still fail:
    with pytest.raises(FileNotFoundError):
        runner = ModelRunner(config=config, model_filename="blah")
        runner.initialize_model(train=False)

    # If the model initialization throws and EOFError, then the Spec2Pep model
    # has tried to load the weights:
    weights = tmp_path / "blah"
    weights.touch()
    with pytest.raises(EOFError):
        runner = ModelRunner(config=config, model_filename=str(weights))
        runner.initialize_model(train=False)
