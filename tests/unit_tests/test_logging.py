"""Tests for Casanovo logging setup."""

import logging
import warnings

from casanovo import casanovo


def test_setup_logging_warning_once_single_line(tmp_path):
    """Warnings are logged once and collapsed onto a single line."""
    log_file = tmp_path / "casanovo.log"
    root_logger = logging.getLogger()
    existing_handlers = list(root_logger.handlers)
    try:
        casanovo.setup_logging(log_file, "info")
        warnings.warn(
            "Skipped 3 spectra with invalid information."
            "Last error was: \n boom"
        )
        log_text = log_file.read_text()
    finally:
        for handler in list(root_logger.handlers):
            if handler not in existing_handlers:
                root_logger.removeHandler(handler)
                handler.close()

    warning_lines = [
        line for line in log_text.splitlines() if "Skipped 3 spectra" in line
    ]
    assert len(warning_lines) == 1
    assert "\n" not in warning_lines[0]
    assert "Last error was: boom" in warning_lines[0]
