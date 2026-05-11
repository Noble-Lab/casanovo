"""Test configuration loading"""

import logging
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
        # Insert remapped deprecated config option.
        cfg["max_iters"] = 1
        yaml.safe_dump(cfg, f_out)

    with pytest.warns(DeprecationWarning):
        Config(filename)

    with open(tiny_config, "r") as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        # Insert non-remapped deprecated config option.
        cfg["save_top_k"] = 5
        yaml.safe_dump(cfg, f_out)

    with pytest.warns(DeprecationWarning):
        Config(filename)


def test_override_mps(monkeypatch, tiny_config, tmp_path, caplog):
    filename = str(tmp_path / "config_auto.yml")
    with open(tiny_config, "r") as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        cfg["accelerator"] = "auto"
        yaml.safe_dump(cfg, f_out)

    with monkeypatch.context() as ctx, caplog.at_level(logging.WARNING):
        # Overwrite on Apple Silicon
        ctx.setattr("platform.system", lambda: "Darwin")
        ctx.setattr("platform.machine", lambda: "arm64")
        cfg = Config(filename)

        assert cfg["accelerator"] == "cpu"
        assert cfg.accelerator == "cpu"
        assert any(
            "overwritten to 'cpu' on Apple Silicon" in rec.getMessage()
            for rec in caplog.records
        )

        # Don't overwrite on x86-64
        caplog.clear()
        ctx.setattr("platform.machine", lambda: "x86-64")
        cfg = Config(filename)

        assert cfg["accelerator"] == "auto"
        assert cfg.accelerator == "auto"
        assert not any(
            "overwritten to 'cpu' on Apple Silicon" in rec.getMessage()
            for rec in caplog.records
        )


# Regression tests for https://github.com/Noble-Lab/casanovo/issues/627.
# val_check_interval must accept both int (training-step count) and
# float in [0.0, 1.0] (fraction of epoch), matching PyTorch Lightning's
# Trainer signature.

def test_val_check_interval_accepts_int(tmp_path, tiny_config):
    filename = str(tmp_path / "vci_int.yml")
    with open(tiny_config, "r") as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        cfg["val_check_interval"] = 50_000
        yaml.safe_dump(cfg, f_out)

    cfg = Config(filename)
    assert cfg.val_check_interval == 50_000
    assert isinstance(cfg.val_check_interval, int)


def test_val_check_interval_accepts_float(tmp_path, tiny_config):
    filename = str(tmp_path / "vci_float.yml")
    with open(tiny_config, "r") as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        cfg["val_check_interval"] = 0.5
        yaml.safe_dump(cfg, f_out)

    cfg = Config(filename)
    assert cfg.val_check_interval == 0.5
    assert isinstance(cfg.val_check_interval, float)


def test_val_check_interval_rejects_float_above_one(tmp_path, tiny_config):
    filename = str(tmp_path / "vci_too_big.yml")
    with open(tiny_config, "r") as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        cfg["val_check_interval"] = 1.5
        yaml.safe_dump(cfg, f_out)

    with pytest.raises(TypeError, match="val_check_interval"):
        Config(filename)


def test_val_check_interval_rejects_float_below_zero(tmp_path, tiny_config):
    filename = str(tmp_path / "vci_neg.yml")
    with open(tiny_config, "r") as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        cfg["val_check_interval"] = -0.1
        yaml.safe_dump(cfg, f_out)

    with pytest.raises(TypeError, match="val_check_interval"):
        Config(filename)


def test_val_check_interval_rejects_bool(tmp_path, tiny_config):
    # bool is a subclass of int in Python, must not silently coerce
    # `true` / `false` into 1 / 0.
    filename = str(tmp_path / "vci_bool.yml")
    with open(tiny_config, "r") as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        cfg["val_check_interval"] = True
        yaml.safe_dump(cfg, f_out)

    with pytest.raises(TypeError, match="val_check_interval"):
        Config(filename)


def test_val_check_interval_default_is_one_point_oh():
    """The default config now ships `val_check_interval: 1.0` (validate
    once per epoch end), not the old `50_000`-step value."""
    cfg = Config()
    assert cfg.val_check_interval == 1.0
