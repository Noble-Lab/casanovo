"""The command line entry point for Casanovo."""
import datetime
import logging
import os
import sys

import click
import pytorch_lightning as pl
import yaml

from . import __version__
from casanovo.denovo import model_runner


logger = logging.getLogger("casanovo")


@click.command()
@click.option(
    "--mode",
    required=True,
    default="denovo",
    help="\b\nThe mode in which to run Casanovo:\n"
    '- "denovo" will predict peptide sequences for\nunknown MS/MS spectra.\n'
    '- "train" will train a model (from scratch or by\ncontinuing training a '
    "previously trained model).\n"
    '- "eval" will evaluate the performance of a\ntrained model using '
    "previously acquired spectrum\nannotations.",
    type=click.Choice(["denovo", "train", "eval"]),
)
@click.option(
    "--model",
    help="The file name of the model weights (.ckpt file).",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--peak_dir",
    required=True,
    help="The directory with peak files for predicting peptide sequences or "
    "training Casanovo.",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--peak_dir_val",
    help="The directory with peak files to be used as validation data during "
    "training.",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--config",
    help="The file name of the configuration file with custom options. If not "
    "specified, a default configuration will be used.",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output",
    help="The base output file name to store logging (extension: .log) and "
    "(optionally) prediction results (extension: .csv).",
    type=click.Path(dir_okay=False),
)
@click.option(
    "--num_workers",
    default=None,
    help="The number of worker threads to use.",
    type=click.INT,
)
def main(
    mode: str,
    model: str,
    peak_dir: str,
    peak_dir_val: str,
    config: str,
    output: str,
    num_workers: int,
):
    """
    \b
    Casanovo: De novo mass spectrometry peptide sequencing with a transformer model.
    ================================================================================

    Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. De novo
    mass spectrometry peptide sequencing with a transformer model. Proceedings
    of the 39th International Conference on Machine Learning - ICML '22 (2022)
    doi:10.1101/2022.02.07.479481.

    Official code website: https://github.com/Noble-Lab/casanovo
    """
    if output is None:
        output = os.path.join(
            os.getcwd(),
            f"casanovo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        )

    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    file_handler = logging.FileHandler(f"{os.path.splitext(output)[0]}.log")
    file_handler.setFormatter(log_formatter)
    root.addHandler(file_handler)
    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(logging.INFO)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    # Read parameters from the config file.
    if config is None:
        config = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "config.yaml"
        )
    with open(config) as f_in:
        config = yaml.safe_load(f_in)
    # Overwrite any parameters that were provided as command-line arguments.
    if num_workers is not None:
        config["num_workers"] = num_workers
    # Ensure that the config values have the correct type.
    for t, keys in (
        (
            int,
            (
                "random_seed",
                "n_peaks",
                "num_workers",
                "dim_model",
                "n_head",
                "dim_feedforward",
                "n_layers",
                "dim_intensity",
                "max_length",
                "max_charge",
                "n_log",
                "warmup_iters",
                "max_iters",
                "train_batch_size",
                "predict_batch_size",
                "max_epochs",
                "num_sanity_val_steps",
                "every_n_epochs",
            ),
        ),
        (
            float,
            (
                "min_mz",
                "max_mz",
                "min_intensity",
                "remove_precursor_tol",
                "dropout",
                "learning_rate",
                "weight_decay",
            ),
        ),
        (bool, ("train_from_scratch", "save_model", "save_weights_only")),
    ):
        for key in keys:
            try:
                if config[key] is not None:
                    config[key] = t(config[key])
            except (TypeError, ValueError) as e:
                logger.error(
                    "Incorrect type for configuration value %s: %s", key, e
                )
                raise TypeError(
                    f"Incorrect type for configuration value {key}: {e}"
                )
    config["residues"] = {
        str(aa): float(mass) for aa, mass in config["residues"].items()
    }

    pl.utilities.seed.seed_everything(seed=config["random_seed"], workers=True)

    # Log the active configuration.
    logger.info("Casanovo version %s", str(__version__))
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    # Run Casanovo in the specified mode.
    if mode == "denovo":
        logger.info("Predict peptide sequences with Casanovo.")
        model_runner.predict(peak_dir, model, output, config)
    elif mode == "eval":
        logger.info("Evaluate a trained Casanovo model.")
        model_runner.evaluate(peak_dir, model, config)
    elif mode == "train":
        logger.info("Train the Casanovo model.")
        model_runner.train(peak_dir, peak_dir_val, model, config)


if __name__ == "__main__":
    main()
