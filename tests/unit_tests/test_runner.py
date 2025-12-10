"""Unit tests specifically for the model_runner module."""

import shutil
import unittest.mock
from pathlib import Path

import depthcharge.tokenizers.peptides
import numpy as np
import pytest
import torch

from casanovo.config import Config
from casanovo.data.psm import PepSpecMatch
from casanovo.denovo.model_runner import ModelRunner


def test_initialize_model(tmp_path, mgf_small):
    """Test initializing a new or existing model."""
    config = Config()
    config.model_save_folder_path = tmp_path
    # Initializing model without initializing tokenizer raises an error.
    with pytest.raises(RuntimeError):
        ModelRunner(config=config).initialize_model(train=True)

    # No model filename given, so train from scratch.
    runner = ModelRunner(config=config)
    runner.initialize_tokenizer()
    runner.initialize_model(train=True)

    # No model filename given during inference = error.
    with pytest.raises(ValueError):
        runner = ModelRunner(config=config)
        runner.initialize_tokenizer()
        runner.initialize_model(train=False)

    # Non-existing model filename given during inference = error.
    with pytest.raises(FileNotFoundError):
        runner = ModelRunner(config=config, model_filename="blah")
        runner.initialize_tokenizer()
        runner.initialize_model(train=False)

    # Train a quick model.
    config.max_epochs = 1
    config.n_layers = 1
    ckpt = tmp_path / "existing.ckpt"
    with ModelRunner(config=config, output_dir=tmp_path) as runner:
        runner.train([mgf_small], [mgf_small])
        runner.trainer.save_checkpoint(ckpt)

    # Resume training from previous model.
    runner = ModelRunner(config=config, model_filename=str(ckpt))
    runner.initialize_tokenizer()
    with unittest.mock.patch.object(
        ModelRunner, "_verify_tokenizer", wraps=ModelRunner._verify_tokenizer
    ) as mock_verify:
        runner.initialize_model(train=True)
        mock_verify.assert_called_once()

    # Inference with previous model.
    runner = ModelRunner(config=config, model_filename=str(ckpt))
    runner.initialize_tokenizer()
    with unittest.mock.patch.object(
        ModelRunner, "_verify_tokenizer", wraps=ModelRunner._verify_tokenizer
    ) as mock_verify:
        runner.initialize_model(train=False)
        mock_verify.assert_called_once()

    # If the model initialization throws and EOFError, then the Spec2Pep model
    # has tried to load the weights.
    weights = tmp_path / "blah"
    weights.touch()
    with pytest.raises(EOFError):
        runner = ModelRunner(config=config, model_filename=str(weights))
        runner.initialize_tokenizer()
        runner.initialize_model(train=False)


def test_save_and_load_weights(tmp_path, mgf_small, tiny_config):
    """Test saving and loading weights"""
    config = Config(tiny_config)
    config.max_epochs = 1
    config.n_layers = 1
    ckpt = tmp_path / "test.ckpt"
    mztab = tmp_path / "test.mztab"

    with ModelRunner(config=config, output_dir=tmp_path) as runner:
        runner.train([mgf_small], [mgf_small])
        runner.trainer.save_checkpoint(ckpt)

    # Try changing model arch:
    other_config = Config(tiny_config)
    other_config.n_layers = 50  # lol
    other_config.n_beams = 12
    other_config.cosine_schedule_period_iters = 2
    with torch.device("meta"):
        # Now load the weights into a new model
        # The device should be meta for all the weights.
        runner = ModelRunner(config=other_config, model_filename=str(ckpt))
        runner.initialize_tokenizer()
        runner.initialize_model(train=False)

    obs_layers = runner.model.encoder.transformer_encoder.num_layers
    assert obs_layers == 1  # Match the original arch.
    assert runner.model.n_beams == 12  # Match the config
    assert runner.model.cosine_schedule_period_iters == 2  # Match the config
    assert next(runner.model.parameters()).device == torch.device("meta")

    # If the Trainer correctly moves the weights to the accelerator,
    # then it should fail if the weights are on the "meta" device.
    with torch.device("meta"):
        with ModelRunner(other_config, model_filename=str(ckpt)) as runner:
            with pytest.raises(NotImplementedError) as err:
                runner.predict([mgf_small], mztab)

    assert "meta tensor; no data!" in str(err.value)

    # Try without arch:
    ckpt_data = torch.load(ckpt, weights_only=False)
    del ckpt_data["hyper_parameters"]
    torch.save(ckpt_data, ckpt)

    # Shouldn't work:
    with ModelRunner(other_config, model_filename=str(ckpt)) as runner:
        with pytest.raises(RuntimeError):
            runner.predict([mgf_small], mztab)

    # Should work:
    with ModelRunner(config=config, model_filename=str(ckpt)) as runner:
        runner.predict([mgf_small], mztab)


def test_save_and_load_weights_deprecated(tmp_path, mgf_small, tiny_config):
    """Test saving and loading weights with deprecated config options."""
    config = Config(tiny_config)
    config.max_epochs = 1
    config.cosine_schedule_period_iters = 5
    ckpt = tmp_path / "test.ckpt"

    with ModelRunner(config=config, output_dir=tmp_path) as runner:
        runner.train([mgf_small], [mgf_small])
        runner.trainer.save_checkpoint(ckpt)

    # Replace the new config option with the deprecated one.
    ckpt_data = torch.load(ckpt, weights_only=False)
    ckpt_data["hyper_parameters"]["max_iters"] = 5
    del ckpt_data["hyper_parameters"]["cosine_schedule_period_iters"]
    torch.save(ckpt_data, str(ckpt))

    # Inference.
    with ModelRunner(
        config=config, model_filename=str(ckpt), overwrite_ckpt_check=False
    ) as runner:
        runner.initialize_tokenizer()
        runner.initialize_model(train=False)
        assert runner.model.cosine_schedule_period_iters == 5
    # Fine-tuning.
    with ModelRunner(
        config=config,
        model_filename=str(ckpt),
        output_dir=tmp_path,
        overwrite_ckpt_check=False,
    ) as runner:
        with pytest.warns(DeprecationWarning):
            runner.train([mgf_small], [mgf_small])
            assert "max_iters" not in runner.model.opt_kwargs


def test_calculate_precision(tmp_path, mgf_small, tiny_config, monkeypatch):
    """Test that this parameter is working correctly."""
    config = Config(tiny_config)
    config.n_layers = 1
    config.max_epochs = 1
    config.calculate_precision = False
    config.tb_summarywriter = str(tmp_path)

    with monkeypatch.context() as ctx:
        mock_logger = unittest.mock.MagicMock()
        ctx.setattr("casanovo.denovo.model.logger", mock_logger)
        runner = ModelRunner(config=config, output_dir=tmp_path)
        with runner:
            runner.train([mgf_small], [mgf_small])

        logged_items = [
            item
            for call in mock_logger.info.call_args_list
            for arg in call.args
            for item in (arg.split("\t") if isinstance(arg, str) else [arg])
        ]

        assert "AA precision" not in logged_items
        assert "Peptide precision" not in logged_items

    config.calculate_precision = True
    with monkeypatch.context() as ctx:
        mock_logger = unittest.mock.MagicMock()
        ctx.setattr("casanovo.denovo.model.logger", mock_logger)
        runner = ModelRunner(
            config=config, output_dir=tmp_path, overwrite_ckpt_check=False
        )
        with runner:
            runner.train([mgf_small], [mgf_small])

        logged_items = [
            item
            for call in mock_logger.info.call_args_list
            for arg in call.args
            for item in (arg.split("\t") if isinstance(arg, str) else [arg])
        ]

        assert "AA precision" in logged_items
        assert "Peptide precision" in logged_items


def test_save_final_model(tmp_path, mgf_small, tiny_config):
    """Test that final model checkpoints are saved."""
    # Test checkpoint saving when val_check_interval is greater than training steps
    config = Config(tiny_config)
    config.val_check_interval = 50
    model_file = tmp_path / "epoch=19-step=20.ckpt"
    with ModelRunner(config, output_dir=tmp_path) as runner:
        runner.train([mgf_small], [mgf_small])

    assert model_file.exists()

    # Test that training again raises file exists error
    with pytest.raises(FileExistsError):
        with ModelRunner(config, output_dir=tmp_path) as runner:
            runner.train([mgf_small], [mgf_small])

    assert model_file.exists()
    Path.unlink(model_file)

    # Test checkpoint saving when val_check_interval is not a factor of training steps
    config.val_check_interval = 15
    validation_file = tmp_path / "foobar.best.ckpt"
    model_file = tmp_path / "foobar.epoch=19-step=20.ckpt"
    with ModelRunner(
        config, output_dir=tmp_path, output_rootname="foobar"
    ) as runner:
        runner.train([mgf_small], [mgf_small])

    assert model_file.exists()
    assert validation_file.exists()


def test_evaluate(
    tmp_path, mgf_small, mzml_small, mgf_small_unannotated, tiny_config
):
    """Test model evaluation during sequencing"""
    # Train tiny model
    config = Config(tiny_config)
    config.max_epochs = 1
    model_file = tmp_path / "epoch=0-step=1.ckpt"
    with ModelRunner(config, output_dir=tmp_path) as runner:
        runner.train([mgf_small], [mgf_small])

    assert model_file.is_file()

    # Test evaluation with annotated peak file
    result_file = tmp_path / "result.mztab"
    with ModelRunner(
        config, model_filename=str(model_file), overwrite_ckpt_check=False
    ) as runner:
        runner.predict([mgf_small], result_file, evaluate=True)

    assert result_file.is_file()
    result_file.unlink()

    exception_string = (
        "Error creating annotated spectrum dataloaders. This may "
        "be the result of having an unannotated peak file present "
        "in the validation peak file path list."
    )

    with pytest.raises(TypeError):
        with ModelRunner(
            config, model_filename=str(model_file), overwrite_ckpt_check=False
        ) as runner:
            runner.predict([mzml_small], result_file, evaluate=True)

    with pytest.raises(TypeError, match=exception_string):
        with ModelRunner(
            config, model_filename=str(model_file), overwrite_ckpt_check=False
        ) as runner:
            runner.predict([mgf_small_unannotated], result_file, evaluate=True)

    with pytest.raises(TypeError, match=exception_string):
        with ModelRunner(
            config, model_filename=str(model_file), overwrite_ckpt_check=False
        ) as runner:
            runner.predict(
                [mgf_small_unannotated, mzml_small], result_file, evaluate=True
            )

    # MzTab with just metadata is written in the case of FileNotFound
    # or TypeError early exit
    assert result_file.is_file()
    result_file.unlink()

    # Test mix of annotated an unannotated peak files
    with pytest.raises(TypeError):
        with ModelRunner(
            config, model_filename=str(model_file), overwrite_ckpt_check=False
        ) as runner:
            runner.predict([mgf_small, mzml_small], result_file, evaluate=True)

    assert result_file.is_file()
    result_file.unlink()

    with pytest.raises(TypeError, match=exception_string):
        with ModelRunner(
            config, model_filename=str(model_file), overwrite_ckpt_check=False
        ) as runner:
            runner.predict(
                [mgf_small, mgf_small_unannotated], result_file, evaluate=True
            )

    assert result_file.is_file()
    result_file.unlink()

    with pytest.raises(TypeError, match=exception_string):
        with ModelRunner(
            config, model_filename=str(model_file), overwrite_ckpt_check=False
        ) as runner:
            runner.predict(
                [mgf_small, mgf_small_unannotated, mzml_small],
                result_file,
                evaluate=True,
            )

    result_file.unlink()


def test_metrics_logging(tmp_path, mgf_small, tiny_config):
    config = Config(tiny_config)
    config.log_metrics = True
    config.log_every_n_steps = 1
    config.tb_summarywriter = True
    config.max_epochs = 1

    curr_model_path = tmp_path / "foo.epoch=0-step=1.ckpt"
    best_model_path = tmp_path / "foo.best.ckpt"
    tb_path = tmp_path / "tensorboard"
    csv_path = tmp_path / "csv_logs"

    with ModelRunner(
        config, output_dir=tmp_path, output_rootname="foo"
    ) as runner:
        runner.train([mgf_small], [mgf_small])

    assert curr_model_path.is_file()
    assert best_model_path.is_file()
    assert tb_path.is_dir()
    assert csv_path.is_dir()

    curr_model_path.unlink()
    best_model_path.unlink()
    shutil.rmtree(tb_path)

    with pytest.raises(FileExistsError):
        with ModelRunner(
            config, output_dir=tmp_path, output_rootname="foo"
        ) as runner:
            runner.train([mgf_small], [mgf_small])

    assert not curr_model_path.is_file()
    assert not best_model_path.is_file()
    assert not tb_path.is_dir()
    assert csv_path.is_dir()


@pytest.mark.parametrize(
    "pred_args, expected_pep, expected_aa, expected_recall",
    [
        pytest.param(
            [
                ("PEP", ("foo", "1"), "PEP"),
                ("PET", ("foo", "2"), "PET"),
            ],
            100,
            100,
            100,
            id="perfect_precision",
        ),
        pytest.param(
            [
                ("PEP", ("foo", "1"), "PEP"),
                ("PEP", ("foo", "2"), "PET"),
            ],
            50,
            100 * (5 / 6),
            100 * (5 / 6),
            id="half_precision_one_wrong",
        ),
        pytest.param(
            [
                ("PEP", ("foo", "1"), "PEP"),
                ("PET", ("foo", "2"), "PET"),
                ("PEI", ("foo", "3"), "PEI"),
                (None, ("foo", "4"), "PEG"),
                ("PEA", ("foo", "5"), "PEA"),
            ],
            100 * (4 / 5),
            100,
            100 * (4 / 5),
            id="skipped_spectra_four_of_five_correct",
        ),
        pytest.param(
            [
                ("PEP", ("foo", "1"), "PEP"),
                (None, ("foo", "2"), "PET"),
                ("PEI", ("foo", "3"), "PEI"),
                (None, ("foo", "4"), "PEG"),
                (None, ("foo", "5"), "PEA"),
            ],
            100 * (2 / 5),
            100,
            100 * (2 / 5),
            id="two_of_five_correct_two_predictions_made",
        ),
        pytest.param(
            [
                ("PE", ("foo", "1"), "PEP"),
                ("PE", ("foo", "2"), "PET"),
                ("PE", ("foo", "3"), "PEI"),
                ("PE", ("foo", "4"), "PEG"),
                (None, ("foo", "5"), "PEA"),
            ],
            0,
            100,
            100 * (8 / 15),
            id="uninferred_spectra_all_predictions_wrong_or_short",
        ),
    ],
)
def test_log_metrics(
    monkeypatch,
    tiny_config,
    pred_args,
    expected_pep,
    expected_aa,
    expected_recall,
):
    def get_mock_psm(sequence, spectrum_id, ground_truth_sequence):
        return PepSpecMatch(
            sequence=sequence,
            spectrum_id=spectrum_id,
            peptide_score=np.nan,
            charge=-1,
            calc_mz=np.nan,
            exp_mz=np.nan,
            aa_scores=[],
            ground_truth_sequence=ground_truth_sequence,
        )

    with monkeypatch.context() as ctx:
        mock_logger = unittest.mock.MagicMock()
        ctx.setattr("casanovo.denovo.model_runner.logger", mock_logger)

        with ModelRunner(Config(tiny_config)) as runner:
            runner.tokenizer = (
                depthcharge.tokenizers.peptides.MskbPeptideTokenizer()
            )
            runner.writer = unittest.mock.MagicMock()
            runner.writer.psms = [get_mock_psm(*args) for args in pred_args]

            runner.log_metrics()
            calls = mock_logger.info.call_args_list[-3:]
            pep_val = calls[0][0][1]
            aa_val = calls[1][0][1]
            recall_val = calls[2][0][1]

            assert pep_val == pytest.approx(expected_pep)
            assert aa_val == pytest.approx(expected_aa)
            assert recall_val == pytest.approx(expected_recall)


@pytest.mark.parametrize(
    "checkpoint_attrs, config_attrs, expected_substring",
    [
        # residues differ: warning about residues/masses
        (
            {"residues": {"A": 42.0, "B": 43.0}, "index": {"A": 0, "B": 1}},
            {"residues": {"A": 42.0, "C": 43.0}, "index": {"A": 0, "B": 1}},
            "residues and/or residue masses",
        ),
        # Residue masses differ: warning about residues/masses
        (
            {"residues": {"A": 42.0, "B": 43.0}, "index": {"A": 0, "B": 1}},
            {"residues": {"A": 42.0, "B": 42.0}, "index": {"A": 0, "B": 1}},
            "residues and/or residue masses",
        ),
        # residues same but index differs: warning about indices
        (
            {"residues": {"A": 42.0, "B": 43.0}, "index": {"A": 0, "B": 1}},
            {"residues": {"A": 42.0, "B": 43.0}, "index": {"A": 1, "B": 0}},
            "residue indices",
        ),
        # identical: no warning
        (
            {"residues": {"A": 42.0, "B": 43.0}, "index": {"A": 0, "B": 1}},
            {"residues": {"A": 42.0, "B": 43.0}, "index": {"A": 0, "B": 1}},
            None,
        ),
    ],
)
def test_verify_tokenizer(
    checkpoint_attrs, config_attrs, expected_substring, caplog
):
    """Test ModelRunner._verify_tokenizer for warning and non-warning behavior."""
    checkpoint = unittest.mock.MagicMock(**checkpoint_attrs)
    config = unittest.mock.MagicMock(**config_attrs)

    with caplog.at_level("WARNING"):
        ModelRunner._verify_tokenizer(checkpoint, config)

    if expected_substring:
        # Check that exactly one warning was logged
        warnings_logged = [
            rec for rec in caplog.records if rec.levelname == "WARNING"
        ]
        assert len(warnings_logged) == 1
        msg = warnings_logged[0].getMessage()
        assert expected_substring in msg
        assert "model checkpoint tokenizer vs config file tokenizer;" in msg
    else:
        # Ensure no warnings were logged
        assert not any(rec.levelname == "WARNING" for rec in caplog.records)
