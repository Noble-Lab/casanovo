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
    for content in ("", "null", "~"):
        filename = tmp_path / "config_empty.yml"
        filename.write_text(content, encoding="utf-8")

        with pytest.raises(KeyError, match="Missing expected config option"):
            Config(str(filename))

    for content in ("[]", "false", "0", "''"):
        filename = tmp_path / "config_list.yml"
        filename.write_text(content, encoding="utf-8")

        with pytest.raises(TypeError, match="must define a mapping"):
            Config(str(filename))

    # Test expected config option is missing.
    filename = str(tmp_path / "config_missing.yml")
    with (
        open(tiny_config, "r", encoding="utf-8") as f_in,
        open(filename, "w", encoding="utf-8") as f_out,
    ):
        cfg = yaml.safe_load(f_in)
        # Remove config option.
        del cfg["random_seed"]
        yaml.safe_dump(cfg, f_out)

    with pytest.raises(KeyError):
        Config(filename)

    # Test invalid config option is present.
    filename = str(tmp_path / "config_invalid.yml")
    with (
        open(tiny_config, "r", encoding="utf-8") as f_in,
        open(filename, "w", encoding="utf-8") as f_out,
    ):
        cfg = yaml.safe_load(f_in)
        # Insert invalid config option.
        cfg["random_seed_"] = 354
        yaml.safe_dump(cfg, f_out)

    with pytest.raises(KeyError):
        Config(filename)


def test_deprecated(tmp_path, tiny_config):
    filename = str(tmp_path / "config_deprecated.yml")
    with (
        open(tiny_config, "r", encoding="utf-8") as f_in,
        open(filename, "w", encoding="utf-8") as f_out,
    ):
        cfg = yaml.safe_load(f_in)
        # Insert remapped deprecated config option.
        cfg["max_iters"] = 1
        yaml.safe_dump(cfg, f_out)

    with pytest.warns(DeprecationWarning):
        Config(filename)

    with (
        open(tiny_config, "r", encoding="utf-8") as f_in,
        open(filename, "w", encoding="utf-8") as f_out,
    ):
        cfg = yaml.safe_load(f_in)
        # Insert non-remapped deprecated config option.
        cfg["save_top_k"] = 5
        yaml.safe_dump(cfg, f_out)

    with pytest.warns(DeprecationWarning):
        Config(filename)


def test_val_check_interval(tmp_path, tiny_config):
    """val_check_interval accepts an int or a float in [0, 1]."""

    def _write(value):
        filename = str(tmp_path / "config_vci.yml")
        with (
            open(tiny_config, "r", encoding="utf-8") as f_in,
            open(filename, "w", encoding="utf-8") as f_out,
        ):
            cfg = yaml.safe_load(f_in)
            cfg["val_check_interval"] = value
            yaml.safe_dump(cfg, f_out)
        return filename

    config = Config(_write(50))
    assert config.val_check_interval == 50
    assert isinstance(config.val_check_interval, int)

    config = Config(_write(0.25))
    assert config.val_check_interval == 0.25
    assert isinstance(config.val_check_interval, float)

    with pytest.raises(TypeError, match="val_check_interval"):
        Config(_write(1.5))
