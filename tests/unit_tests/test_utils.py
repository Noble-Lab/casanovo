import logging
import platform
import unittest.mock

import pandas as pd
import pytest

from casanovo import utils
from casanovo.data import psm


def test_log_annotate_report_average_precision(monkeypatch):
    mock_logger = unittest.mock.MagicMock()
    monkeypatch.setattr("casanovo.utils.logger", mock_logger)
    monkeypatch.setattr(utils, "log_run_report", lambda **_: None)
    utils.log_annotate_report([], average_precision=0.75)
    calls = [
        c
        for c in mock_logger.info.call_args_list
        if "Average Precision" in c[0][0]
    ]
    assert calls and calls[0][0][1] == 0.75


@pytest.mark.skip(reason="Skipping due to Linux deadlock issue")
def test_n_workers(monkeypatch):
    """Check that n_workers is correct without a GPU."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    def cpu_fun(_: object) -> list[str]:
        return ["foo"] * 31

    with monkeypatch.context() as mnk:
        mnk.setattr("psutil.Process.cpu_affinity", cpu_fun, raising=False)
        expected = 0 if platform.system() in ["Windows", "Darwin"] else 31
        assert utils.n_workers() == expected

    with monkeypatch.context() as mnk:
        mnk.delattr("psutil.Process.cpu_affinity", raising=False)
        mnk.setattr("os.cpu_count", lambda: 41)
        expected = 0 if platform.system() in ["Windows", "Darwin"] else 41
        assert utils.n_workers() == expected

    with monkeypatch.context() as mnk:
        mnk.setattr("torch.cuda.device_count", lambda: 4)
        mnk.setattr("psutil.Process.cpu_affinity", cpu_fun, raising=False)
        expected = 0 if platform.system() in ["Windows", "Darwin"] else 7
        assert utils.n_workers() == expected

    with monkeypatch.context() as mnk:
        mnk.delattr("psutil.Process.cpu_affinity", raising=False)
        mnk.delattr("os.cpu_count")
        if platform.system() not in ["Windows", "Darwin"]:
            with pytest.raises(AttributeError):
                utils.n_workers()
        else:
            assert utils.n_workers() == 0


@pytest.mark.parametrize(
    ("start_time", "end_time", "expected_throughput"),
    [
        (0, 10, "Throughput: 0.20 spectra/s"),
        (0, 100, "Throughput: 0.02 spectra/s"),
        (None, 10, None),
        (0, None, None),
        (0, 0, None),
        (10, 0, None),
    ],
)
def test_log_annotate_report(
    caplog, start_time, end_time, expected_throughput
):
    """Report unique spectra and guard missing or invalid run times."""
    predictions = [
        psm.PepSpecMatch(
            sequence="AAAA",
            spectrum_id=("test.mgf", spectrum_id),
            peptide_score=score,
            charge=3,
            calc_mz=100.0,
            exp_mz=100.0,
            aa_scores=[0.5] * 4,
        )
        for spectrum_id, score in [
            ("0", 0.05),
            ("0", 0.5),
            ("1", 0.9),
        ]
    ]

    with caplog.at_level(logging.INFO, logger="casanovo"):
        utils.log_annotate_report(
            predictions, start_time=start_time, end_time=end_time
        )

    messages = [record.getMessage() for record in caplog.records]
    assert "Sequenced 2 spectra (3 PSMs)" in messages
    assert "3 PSMs (100.00%) scored ≥ 0.00" in messages
    assert "2 PSMs (66.67%) scored ≥ 0.50" in messages
    assert "1 PSM (33.33%) scored ≥ 0.90" in messages
    distribution_messages = [
        message for message in messages if "scored ≥" in message
    ]
    assert len(distribution_messages) == 7
    throughput_messages = [
        message for message in messages if message.startswith("Throughput:")
    ]
    if expected_throughput is None:
        assert throughput_messages == []
    else:
        assert throughput_messages == [expected_throughput]


def test_get_report_dict_default_bins():
    """Default fixed bins include their boundaries and exclude lower scores."""
    scores = [
        -0.1,
        0.0,
        0.499,
        0.5,
        0.599,
        0.6,
        0.699,
        0.7,
        0.799,
        0.8,
        0.849,
        0.85,
        0.899,
        0.9,
        1.0,
    ]
    results = pd.DataFrame(
        {"sequence": ["AAAA"] * len(scores), "score": scores}
    )
    report = utils._get_report_dict(results)
    assert report["num_psms"] == 15
    assert report["score_bins"] == {
        0.0: 14,
        0.5: 12,
        0.6: 10,
        0.7: 8,
        0.8: 6,
        0.85: 4,
        0.9: 2,
    }


@pytest.mark.parametrize("scores", [[0.42], [0.42, 0.42, 0.42]])
def test_get_report_dict_fixed_bins_ties(scores):
    """Include every tied PSM and retain bins with no matches."""
    report = utils._get_report_dict(
        pd.DataFrame({"sequence": ["AAAA"] * len(scores), "score": scores}),
        score_bins=(0.4, 0.42, 0.5),
    )
    assert report["score_bins"] == {
        0.4: len(scores),
        0.42: len(scores),
        0.5: 0,
    }


def test_log_annotate_report_custom_threshold_precision(caplog):
    """Printed custom thresholds reproduce the exact counted PSM set."""
    threshold = 0.992123456
    predictions = [
        psm.PepSpecMatch(
            sequence="AAAA",
            spectrum_id=("test.mgf", str(index)),
            peptide_score=score,
            charge=3,
            calc_mz=100.0,
            exp_mz=100.0,
            aa_scores=[0.5] * 4,
        )
        for index, score in enumerate([0.9921233, threshold])
    ]
    with caplog.at_level(logging.INFO, logger="casanovo"):
        utils.log_annotate_report(predictions, score_bins=(threshold,))
    assert "1 PSM (50.00%) scored ≥ 0.992123456" in caplog.messages


def test_split_version():
    """Test that splitting the version number works as expected."""
    version = utils.split_version("2.0.1")
    assert version == ("2", "0", "1")

    version = utils.split_version("0.1.dev1+g39f8c53")
    assert version == ("0", "1", "")

    version = utils.split_version("3.0.1.dev10282blah")
    assert version == ("3", "0", "1")


def test_check_dir(tmp_path):
    exists_path = tmp_path / "exists-1234.ckpt"
    exists_pattern = "exists-*.ckpt"
    exists_path.touch()
    dne_pattern = "dne-*.ckpt"

    with pytest.raises(FileExistsError):
        utils.check_dir_file_exists(tmp_path, [exists_pattern, dne_pattern])

    with pytest.raises(FileExistsError):
        utils.check_dir_file_exists(tmp_path, exists_pattern)

    utils.check_dir_file_exists(tmp_path, [dne_pattern])
    utils.check_dir_file_exists(tmp_path, dne_pattern)


def test_log_annotate_report_missing_predictions(monkeypatch):
    mock_logger = unittest.mock.MagicMock()
    monkeypatch.setattr("casanovo.utils.logger", mock_logger)
    monkeypatch.setattr(utils, "log_run_report", lambda **_: None)
    utils.log_annotate_report([], n_missing_predictions=4)
    calls = [
        c
        for c in mock_logger.info.call_args_list
        if "did not receive a prediction" in c[0][0]
    ]
    assert calls and calls[0][0][1] == 4

    # Positional score_bins callers still bind correctly, not to the
    # keyword-only n_missing_predictions (which would otherwise make
    # the missing-prediction message fire for a positional argument).
    mock_logger.reset_mock()
    utils.log_annotate_report([], None, None, [0.5])
    # An empty report still warns that no predictions were logged.
    mock_logger.warning.assert_called_once()
    missing_calls = [
        c
        for c in mock_logger.info.call_args_list
        if "did not receive a prediction" in c[0][0]
    ]
    assert not missing_calls
