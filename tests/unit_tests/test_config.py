"""Test configuration loading"""

import pytest
import yaml

from casanovo.config import Config


def test_default():
    """Test that loading the default works"""
    config = Config()
    assert config.random_seed == 454
    assert config["random_seed"] == 454
    assert config.accelerator == "auto"
    assert config.file == "default"


def test_override(tmp_path, tiny_config):
    # Test expected config option is missing.
    filename = str(tmp_path / "config_missing.yml")
    with open(tiny_config, "r") as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        # Remove config option.
        del cfg["random_seed"]
        yaml.safe_dump(cfg, f_out)

    with pytest.raises(KeyError):
        Config(filename)

    # Test invalid config option is present.
    filename = str(tmp_path / "config_invalid.yml")
    with open(tiny_config, "r") as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        # Insert invalid config option.
        cfg["random_seed_"] = 354
        yaml.safe_dump(cfg, f_out)

    with pytest.raises(KeyError):
        Config(filename)


def test_deprecated(tmp_path, tiny_config):
    filename = str(tmp_path / "config_deprecated.yml")
    with open(tiny_config, "r") as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        # Insert deprecated config option.
        cfg["max_iters"] = 1
        yaml.safe_dump(cfg, f_out)

    with pytest.warns(DeprecationWarning):
        Config(filename)
