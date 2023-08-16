"""Test configuration loading"""
from casanovo.config import Config


def test_default():
    """Test that loading the default works"""
    config = Config()
    assert config.random_seed == 454
    assert config["random_seed"] == 454
    assert not config.no_gpu
    assert config.file == "default"


def test_override(tmp_path):
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

    config = Config(yml)
    assert config.random_seed == 42
    assert config["random_seed"] == 42
    assert not config.no_gpu
    assert config.top_match == 3
    assert len(config.residues) == 4
    for i, residue in enumerate("WOUT", 1):
        assert config["residues"][residue] == i
    assert config.file == str(yml)
