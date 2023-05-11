"""Unit tests specifically for the model_runner module."""
import pytest
import torch

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


def test_save_and_load_weights(tmp_path, mgf_small, tiny_config):
    """Test saving aloading weights"""
    config = Config(tiny_config)
    config.max_epochs = 1
    config.n_layers = 1
    ckpt = tmp_path / "test.ckpt"

    with ModelRunner(config=config) as runner:
        runner.train([mgf_small], [mgf_small])
        runner.trainer.save_checkpoint(ckpt)

    # Try changing model arch:
    other_config = Config(tiny_config)
    other_config.n_layers = 50  # lol
    with torch.device("meta"):
        # Now load the weights into a new model
        # The device should be meta for all the weights.
        runner = ModelRunner(config=other_config, model_filename=str(ckpt))
        runner.initialize_model(train=False)

    obs_layers = runner.model.encoder.transformer_encoder.num_layers
    assert obs_layers == 1  # Match the original arch.
    assert next(runner.model.parameters()).device == torch.device("meta")

    # If the Trainer correctly moves the weights to the accelerator,
    # then it should fail if the weights are on the "meta" device.
    with torch.device("meta"):
        with ModelRunner(other_config, model_filename=str(ckpt)) as runner:
            with pytest.raises(NotImplementedError) as err:
                runner.evaluate([mgf_small])

    assert "meta tensor; no data!" in str(err.value)

    # Try without arch:
    ckpt_data = torch.load(ckpt)
    del ckpt_data["hyper_parameters"]
    torch.save(ckpt_data, ckpt)

    # Shouldn't work:
    with ModelRunner(other_config, model_filename=str(ckpt)) as runner:
        with pytest.raises(RuntimeError):
            runner.evaluate([mgf_small])

    # Should work:
    with ModelRunner(config=config, model_filename=str(ckpt)) as runner:
        runner.evaluate([mgf_small])
