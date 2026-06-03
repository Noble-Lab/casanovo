"""Test configuration loading."""

import logging
import math

import pytest
import yaml

from casanovo.config import Config


def test_default(monkeypatch):
    """Test that loading the default works."""
    with monkeypatch.context() as ctx:
        ctx.setattr("platform.machine", lambda: "x86-64")
        config = Config()

    assert config.random_seed == 454
    assert config["random_seed"] == 454
    assert config.accelerator == "auto"
    assert config.file == "default"


def test_override(tmp_path, tiny_config):
    # Test that a partial config is accepted; missing keys fall back to
    # defaults from config.yaml.
    filename = str(tmp_path / "config_partial.yml")
    with open(filename, "w") as f:
        yaml.dump({"learning_rate": 1e-3}, f)
    config = Config(filename)
    assert config.learning_rate == pytest.approx(1e-3)
    assert config.random_seed == 454  # default

    # Test that an unrecognized config option raises KeyError.
    filename = str(tmp_path / "config_invalid.yml")
    with (
        open(tiny_config, encoding="utf-8") as f_in,
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
        open(tiny_config, encoding="utf-8") as f_in,
        open(filename, "w", encoding="utf-8") as f_out,
    ):
        cfg = yaml.safe_load(f_in)
        # Insert remapped deprecated config option.
        cfg["max_iters"] = 1
        yaml.safe_dump(cfg, f_out)

    with pytest.warns(DeprecationWarning):
        Config(filename)

    with (
        open(tiny_config, encoding="utf-8") as f_in,
        open(filename, "w", encoding="utf-8") as f_out,
    ):
        cfg = yaml.safe_load(f_in)
        # Insert non-remapped deprecated config option.
        cfg["save_top_k"] = 5
        yaml.safe_dump(cfg, f_out)

    with pytest.warns(DeprecationWarning):
        Config(filename)


def test_override_mps(monkeypatch, tiny_config, tmp_path, caplog):
    filename = str(tmp_path / "config_auto.yml")
    with (
        open(tiny_config, encoding="utf-8") as f_in,
        open(filename, "w", encoding="utf-8") as f_out,
    ):
        cfg = yaml.safe_load(f_in)
        cfg["accelerator"] = "auto"
        yaml.safe_dump(cfg, f_out)

    with monkeypatch.context() as ctx, caplog.at_level(logging.WARNING):
        # Overwrite on Apple Silicon.
        ctx.setattr("platform.system", lambda: "Darwin")
        ctx.setattr("platform.machine", lambda: "arm64")
        cfg = Config(filename)

        assert cfg["accelerator"] == "cpu"
        assert cfg.accelerator == "cpu"
        assert any(
            "overwritten to 'cpu' on Apple Silicon" in rec.getMessage()
            for rec in caplog.records
        )

        # Don't overwrite on x86-64.
        caplog.clear()
        ctx.setattr("platform.machine", lambda: "x86-64")
        cfg = Config(filename)

        assert cfg["accelerator"] == "auto"
        assert cfg.accelerator == "auto"
        assert not any(
            "overwritten to 'cpu' on Apple Silicon" in rec.getMessage()
            for rec in caplog.records
        )


def test_invalid_yaml_type(tmp_path):
    """Test that a non-mapping YAML file raises TypeError."""
    filename = str(tmp_path / "config_list.yml")
    with open(filename, "w") as f:
        f.write("- item1\n- item2\n")
    with pytest.raises(TypeError, match="must contain a YAML mapping"):
        Config(filename)


def test_deprecated_only_old_key(tmp_path, tiny_config):
    """Test deprecated key is remapped when only the old key is present."""
    filename = str(tmp_path / "config_deprecated_remap.yml")
    with open(tiny_config) as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        # Remove the replacement key and insert only the deprecated key.
        del cfg["cosine_schedule_period_iters"]
        cfg["max_iters"] = 1
        yaml.safe_dump(cfg, f_out)

    with pytest.warns(DeprecationWarning, match="remapped to"):
        config = Config(filename)

    assert config.cosine_schedule_period_iters == 1


def test_deprecated_both_keys(tmp_path, tiny_config):
    """Test that when both a deprecated key and its replacement are present,
    the replacement value is kept and the deprecated key is discarded."""
    filename = str(tmp_path / "config_deprecated_both.yml")
    with open(tiny_config) as f_in, open(filename, "w") as f_out:
        cfg = yaml.safe_load(f_in)
        cfg["cosine_schedule_period_iters"] = 500
        cfg["max_iters"] = 999  # deprecated, should be discarded
        yaml.safe_dump(cfg, f_out)

    with pytest.warns(DeprecationWarning, match="ignored"):
        config = Config(filename)

    assert config.cosine_schedule_period_iters == 500


def test_getattr_private_guard(tiny_config):
    """Test that __getattr__ raises AttributeError for non-existent
    private attributes without attempting a config key lookup."""
    config = Config(str(tiny_config))
    with pytest.raises(AttributeError):
        _ = config._nonexistent_private_xyz


def test_getattr_unknown_public(tiny_config):
    """Test that accessing a non-existent public config parameter raises
    AttributeError."""
    config = Config(str(tiny_config))
    with pytest.raises(AttributeError):
        _ = config.nonexistent_public_param


def test_precursor_mass_tol_inf(tmp_path):
    """Test that precursor_mass_tol accepts .inf (YAML float infinity)."""
    filename = str(tmp_path / "config_inf.yml")
    with open(filename, "w") as f:
        f.write("precursor_mass_tol: .inf\n")
    config = Config(filename)
    assert math.isinf(config.precursor_mass_tol)


def test_isotope_error_range_type():
    """Test that isotope_error_range is returned as a tuple of two ints."""
    config = Config()
    r = config.isotope_error_range
    assert isinstance(r, tuple)
    assert len(r) == 2
    assert all(isinstance(v, int) for v in r)


def test_residues_are_floats():
    """Test that residue masses are returned as float values."""
    config = Config()
    residues = config.residues
    assert isinstance(residues, dict)
    assert all(isinstance(v, float) for v in residues.values())
