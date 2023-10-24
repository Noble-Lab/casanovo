"""Test configuration loading"""
from casanovo.config import Config
import pytest
import yaml


def test_default():
    """Test that loading the default works"""
    config = Config()
    assert config.random_seed == 454
    assert config["random_seed"] == 454
    assert config.accelerator == "auto"
    assert config.file == "default"


def test_override(tmp_path, tiny_config):
    """Test overriding the default"""
    yml = tmp_path / "test.yml"
    with yml.open("w+") as f_out:
        f_out.write(
            """random_seed: 42
top_match: 3
residues:
  W: 1
  O: 2
  U: 3
  T: 4
"""
        )

    with open (tiny_config, 'r') as read_file:
        contents = yaml.safe_load(read_file)
        contents['random_seed_'] = 354
        print(contents)

    with open('output.yml', 'w') as write_file:
        yaml.safe_dump(contents, write_file)
    with pytest.raises(KeyError):
        config = Config('output.yml')
    
    with pytest.raises(KeyError):
        config = Config(yml)
    