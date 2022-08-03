import logging
import os
import sys
from typing import List

import click
import pytorch_lightning as pl
import yaml

from . import __version__
from casanovo.denovo import denovo, evaluate, train


logger = logging.getLogger("casanovo")


@click.command()
@click.option(
    "--mode",
    required=True,
    default="denovo",
    help='\b\nThe mode in which to run Casanovo:\n'
         '- "denovo" will predict peptide sequences for\nunknown MS/MS spectra.\n'
         '- "train" will train a model (from scratch or by\ncontinuing training a '
         'previously trained model).\n'
         '- "eval" will evaluate the performance of a\ntrained model using previously '
         'acquired spectrum\nannotations.',
    type=click.Choice(["denovo", "train", "eval"]),
)
@click.option(
    "--model",
    required=True,
    help="The file name of the model weights (.ckpt file).",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--denovo_dir",
    help="The directory with peak files for predicting peptide sequences.",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--train_dir",
    help="The directory with peak files to be used as training data.",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--val_dir",
    help="The directory with peak files to be used as validation data.",
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
    help="The output file name for the prediction results (format: .csv).",
    type=click.Path(dir_okay=False),
)
@click.option(
    "--num_workers",
    default=None,
    help="The number of worker threads to use.",
    type=click.INT,
)
@click.option(
    "--gpu",
    default=(),
    help="The identifier of the GPU to use. Multiple GPUs can be requested using the "
         "following format: --gpus=0 --gpus=1 --gpus=2 ...",
    type=click.INT,
    multiple=True,
)
def main(
    mode: str,
    model: str,
    denovo_dir: str,
    train_dir: str,
    val_dir: str,
    config: str,
    output: str,
    num_workers: int,
    gpu: List[int],
):
    """
    \b
    Casanovo: De novo mass spectrometry peptide sequencing with a transformer model.
    ================================================================================

    Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. De novo mass
    spectrometry peptide sequencing with a transformer model. Proceedings of the 39th
    International Conference on Machine Learning - ICML '22 (2022)
    doi:10.1101/2022.02.07.479481.

    Official code website: https://github.com/Noble-Lab/casanovo
    """
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : {message}",
        style="{"
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    file_handler = logging.FileHandler(f"{os.path.splitext(output)[0]}.log")
    file_handler.setFormatter(log_formatter)
    root.addHandler(file_handler)
    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
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
    if len(gpu) > 0:
        config["gpus"] = gpu

    pl.utilities.seed.seed_everything(seed=config["random_seed"], workers=True)

    # Log the active configuration.
    logger.info('Casanovo version %s', str(__version__))
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    # Run Casanovo in the specified mode.
    if mode == "denovo":
        logger.info("Predict peptide sequences with Casanovo.")
        denovo(denovo_dir, model, output, config)
    elif mode == "train":
        logger.info("Train the Casanovo model.")
        train(train_dir, val_dir, model, config)
    elif mode == "eval":
        logger.info("Evaluate a trained Casanovo model.")
        evaluate(denovo_dir, model, config)


if __name__ == "__main__":
    main()
