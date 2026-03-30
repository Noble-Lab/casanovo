"""Tests for the Casanovo CLI (casanovo/casanovo.py)."""

from pathlib import Path
from click.testing import CliRunner
from casanovo.casanovo import main


def test_output_root_same_name_as_existing_dir():
    """Regression test for issue #568.

    --output_root must be accepted even when its value matches the name
    of an existing directory.The directory-existence check is only
    meaningful for individual output *files*, not for the root name
    itself.
    """
    dir_name = "train-casa562-mssv"
    runner = CliRunner()

    with runner.isolated_filesystem():
        # Simulating the situation: the output directory already exists
        # (e.g. created by a previous run or by --output_dir).
        Path(dir_name).mkdir()
        result = runner.invoke(
            main,
            [
                "configure",
                "--output_dir",
                dir_name,
                "--output_root",
                dir_name,
                "--force_overwrite",
            ],
        )
    # Before the fix this failed with:
    #   "Invalid value for '-o' / '--output_root':
    #    File 'train-casa562-mssv' is a directory."
    assert "is a directory" not in (result.output or ""), (
        f"CLI unexpectedly rejected --output_root. Output:\n{result.output}"
    )
    assert result.exit_code == 0, (
        f"CLI exited with code {result.exit_code}. Output:\n{result.output}"
    )
