import logging
import os
import sys
from typing import List

import click
import yaml

from casanovo.denovo import denovo, evaluate, train


logger = logging.getLogger("casanovo")


@click.command()
@click.option(
    "--mode",
    required=True,
    default="denovo",
    help='The mode in which to run Casanovo:\n'
         '- "denovo" will predict peptide sequence for unknown MS/MS spectra.\n'
         '- "train" will train a model (from scratch or by continuing training a '
         'previously trained model).\n'
         '- "eval" will evaluate the performance of a trained model using previously '
         'acquired spectrum annotations.',
    type=click.Choice(["denovo", "train", "eval"]),
)
@click.option(
    "--model",
    required=True,
    help="The file name of the model weights (.ckpt file).",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--denovo_dir",
    help="The directory with peak files for predicting peptide sequences.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--train_dir",
    help="The directory with peak files to be used as training data.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--val_dir",
    help="The directory with peak files to be used as validation data.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--config",
    help="The file name of the configuration file with custom options. If not "
         "specified, a default configuration will be used.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--output",
    help="The output file name for the prediction results (format: .csv).",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
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
    model: click.Path,
    denovo_dir: click.Path,
    train_dir: click.Path,
    val_dir: click.Path,
    config: click.Path,
    output: click.Path,
    num_workers: int,
    gpu: List[int],
):
    """
    Command-line interface for the Casanovo de novo peptide sequencer.

    Parameters
    ----------
    mode : str
        Whether to run Casanovo in prediction ("denovo"), training ("train"), or
        evaluation ("eval") mode.
    model : click.Path
        The file name of the model weights (.ckpt file).
    denovo_dir : click.Path
        The directory with peak files for predicting peptide sequences.
    train_dir : click.Path
        The directory with peak files to be used as training data.
    val_dir : click.Path
        The directory with peak files to be used as validation data.
    config : click.Path
        The file name of the configuration file with custom options. If not specified, a
        default configuration will be used.
    output : click.Path
        The output file name for the prediction results (format: .csv).
    num_workers : int
        The number of worker threads to use.
    gpu : List[int]
        The identifiers of the GPUs to use.
    """
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
            "{message}",
            style="{"
        )
    )
    root.addHandler(handler)

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

    # Run Casanovo in the specified mode.
    if mode == "denovo":
        logger.info("Predict peptide sequences with Casanovo.")
        denovo(denovo_dir, model, config, output)
    elif mode == "train":
        logger.info("Train the Casanovo model.")
        train(train_dir, val_dir, model, config)
    elif mode == "eval":
        logger.info("Evaluate a trained Casanovo model.")
        evaluate(denovo_dir, model, config)


if __name__ == "__main__":
    main()
