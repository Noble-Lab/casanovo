"""Test configuration loading"""
import pytest

from casanovo.config import Config


def test_default():
    """Test that loading the default works"""
    config = Config()
    assert config.random_seed == 454
    assert config["random_seed"] == 454
    assert not config.no_gpu


def test_override(tmp_path):
    """Test overriding the default"""
    yml = tmp_path / "test.yml"
    with yml.open("w+") as f_out:
        f_out.write("random_seed: 42")

    config = Config(yml)
    assert config.random_seed == 42
    assert config["random_seed"] == 42
    assert not config.no_gpu