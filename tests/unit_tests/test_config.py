"""Test configuration loading"""

import logging

import pytest
import yaml

from casanovo.config import Config


def test_default(monkeypatch):
    """Test that loading the default works"""
    with monkeypatch.context() as ctx:
        ctx.setattr("platform.machine", lambda: "x86-64")
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
