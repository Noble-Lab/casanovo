"""Tests for profiling utilities and benchmark command."""

import json
import logging
import time
from unittest import mock

import pytest

from casanovo import utils
from casanovo.config import Config
from casanovo.denovo.model_runner import ModelRunner


def test_stage_timer_records_elapsed():
    timings = {}
    with utils.stage_timer("test_stage", timings):
        time.sleep(0.01)
    assert "test_stage" in timings
    assert timings["test_stage"] >= 0.01


def test_stage_timer_multiple_stages():
    timings = {}
    with utils.stage_timer("stage_a", timings):
        pass
    with utils.stage_timer("stage_b", timings):
        pass
    assert set(timings.keys()) == {"stage_a", "stage_b"}
    assert all(v >= 0 for v in timings.values())


def test_stage_timer_records_on_exception():
    timings = {}
    with pytest.raises(ValueError):
        with utils.stage_timer("boom", timings):
            raise ValueError("oops")
    assert "boom" in timings


def test_log_run_report_stage_times(caplog):
    stage_times = {"data_loading": 0.123, "inference": 4.567}
    with caplog.at_level(logging.INFO, logger="casanovo"):
        utils.log_run_report(stage_times=stage_times)
    assert "data_loading" in caplog.text
    assert "inference" in caplog.text


def test_log_run_report_no_stage_times(caplog):
    with caplog.at_level(logging.INFO, logger="casanovo"):
        utils.log_run_report()
    assert "End of Run Report" in caplog.text


def test_predict_creates_profile_trace(tmp_path, mgf_small, tiny_config):
    config = Config(tiny_config)
    config.max_epochs = 1
    config.n_layers = 1

    ckpt = tmp_path / "model.ckpt"
    mztab = tmp_path / "results.mztab"
    trace_path = tmp_path / "trace.json"

    with ModelRunner(config=config, output_dir=tmp_path) as runner:
        runner.train([mgf_small], [mgf_small])
        runner.trainer.save_checkpoint(ckpt)

    with ModelRunner(config=config, model_filename=str(ckpt)) as runner:
        stage_times = runner.predict(
            [mgf_small], str(mztab), profile_output=str(trace_path)
        )

    assert trace_path.exists()
    with open(trace_path) as f:
        trace = json.load(f)
    assert isinstance(trace, (dict, list))
    assert "data_loading" in stage_times
    assert "inference" in stage_times
    assert stage_times["inference"] > 0


def test_predict_no_profile_output(tmp_path, mgf_small, tiny_config):
    config = Config(tiny_config)
    config.max_epochs = 1
    config.n_layers = 1

    ckpt = tmp_path / "model.ckpt"
    mztab = tmp_path / "results.mztab"

    with ModelRunner(config=config, output_dir=tmp_path) as runner:
        runner.train([mgf_small], [mgf_small])
        runner.trainer.save_checkpoint(ckpt)

    with ModelRunner(config=config, model_filename=str(ckpt)) as runner:
        stage_times = runner.predict([mgf_small], str(mztab))

    assert len(list(tmp_path.glob("*.json"))) == 0
    assert "data_loading" in stage_times
    assert "inference" in stage_times


def test_sequence_cli_profile_flag(tmp_path, mgf_small, tiny_config):
    from click.testing import CliRunner
    from casanovo.casanovo import sequence

    trace_path = tmp_path / "cli_trace.json"
    runner = CliRunner()

    with (
        mock.patch("casanovo.casanovo.setup_model") as mock_setup,
        mock.patch("casanovo.casanovo.ModelRunner") as MockRunner,
    ):
        mock_setup.return_value = (Config(tiny_config), None)
        mock_instance = mock.MagicMock()
        mock_instance.__enter__ = mock.Mock(return_value=mock_instance)
        mock_instance.__exit__ = mock.Mock(return_value=False)
        mock_instance.predict.return_value = {
            "data_loading": 0.1,
            "inference": 1.0,
        }
        mock_instance.writer.psms = []
        MockRunner.return_value = mock_instance

        result = runner.invoke(
            sequence,
            [
                str(mgf_small),
                "--profile",
                str(trace_path),
                "--config",
                str(tiny_config),
                "--output_dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0, result.output
    _, call_kwargs = mock_instance.predict.call_args
    assert call_kwargs.get("profile_output") == str(trace_path)


def test_benchmark_cli_smoke(tmp_path, mgf_small, tiny_config):
    from click.testing import CliRunner
    from casanovo.casanovo import benchmark

    runner = CliRunner()

    with (
        mock.patch("casanovo.casanovo.setup_model") as mock_setup,
        mock.patch("casanovo.casanovo.ModelRunner") as MockRunner,
    ):
        mock_setup.return_value = (Config(tiny_config), None)
        mock_instance = mock.MagicMock()
        mock_instance.__enter__ = mock.Mock(return_value=mock_instance)
        mock_instance.__exit__ = mock.Mock(return_value=False)
        mock_instance.predict_timed.return_value = 150.0
        MockRunner.return_value = mock_instance

        result = runner.invoke(
            benchmark,
            [
                str(mgf_small),
                "--config",
                str(tiny_config),
                "--output_dir",
                str(tmp_path),
                "--batch_sizes",
                "256",
                "--n_beams_list",
                "1",
                "--n_iter",
                "1",
                "--warmup",
                "0",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "spectra/s" in result.output
    assert "256" in result.output
    mock_instance.predict_timed.assert_called_once()
