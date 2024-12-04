"""Unit tests specifically for the model_runner module."""

import shutil
import unittest.mock
from pathlib import Path

import depthcharge.tokenizers.peptides
import pytest
import torch

from casanovo.config import Config
from casanovo.data.psm import PepSpecMatch
from casanovo.denovo.model_runner import ModelRunner


def test_initialize_model(tmp_path, mgf_small):
    """Test initializing a new or existing model."""
    print("Initializing test configuration")
    config = Config()
    config.model_save_folder_path = tmp_path

    # Test: Initializing model without initializing tokenizer raises an error
    print("Testing initialization without tokenizer (expecting RuntimeError)")
    with pytest.raises(RuntimeError):
        ModelRunner(config=config).initialize_model(train=True)

    # Test: No model filename given, so train from scratch
    print("Initializing tokenizer and model for training (train from scratch)")
    runner = ModelRunner(config=config)
    runner.initialize_tokenizer()
    runner.initialize_model(train=True)

    # Test: No model filename given during inference = error
    print("Testing inference with no model filename (expecting ValueError)")
    with pytest.raises(ValueError):
        runner = ModelRunner(config=config)
        runner.initialize_tokenizer()
        runner.initialize_model(train=False)

    # Test: Non-existing model filename during inference = error
    print(
        "Testing inference with non-existing model filename (expecting FileNotFoundError)"
    )
    with pytest.raises(FileNotFoundError):
        runner = ModelRunner(config=config, model_filename="blah")
        runner.initialize_tokenizer()
        runner.initialize_model(train=False)

    # Train a quick model
    print("Training a quick model with minimal configuration")
    config.max_epochs = 1
    config.n_layers = 1
    ckpt = tmp_path / "existing.ckpt"
    with ModelRunner(config=config, output_dir=tmp_path) as runner:
        runner.train([mgf_small], [mgf_small])
        runner.trainer.save_checkpoint(ckpt)
    print(f"Quick model trained and checkpoint saved at {ckpt}")

    # Test: Resume training from previous model
    print(f"Resuming training from checkpoint {ckpt}")
    runner = ModelRunner(config=config, model_filename=str(ckpt))
    runner.initialize_tokenizer()
    runner.initialize_model(train=True)

    # Test: Inference with previous model
    print(f"Initializing model for inference with checkpoint {ckpt}")
    runner = ModelRunner(config=config, model_filename=str(ckpt))
    runner.initialize_tokenizer()
    runner.initialize_model(train=False)

    # Test: Spec2Pep model tries to load weights and throws EOFError
    print("Testing Spec2Pep model weight loading (expecting EOFError)")
    weights = tmp_path / "blah"
    weights.touch()
    with pytest.raises(EOFError):
        runner = ModelRunner(config=config, model_filename=str(weights))
        runner.initialize_tokenizer()
        runner.initialize_model(train=False)

    print("All tests for model initialization completed successfully")
