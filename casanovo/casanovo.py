"""The command line entry point for Casanovo."""
import collections
import csv
import datetime
import logging
import os
import pathlib
import re
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

    pl.utilities.seed.seed_everything(seed=config["random_seed"], workers=True)

    # Log the active configuration.
    logger.info("Casanovo version %s", str(__version__))
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    # Run Casanovo in the specified mode.
    if mode == "denovo":
        logger.info("Predict peptide sequences with Casanovo.")
        # Derive the fixed and variable modifications from the residue alphabet.
        residues = collections.defaultdict(set)
        for aa, mass in config["residues"].items():
            aa_mod = re.match(r"([A-Z]?)([+-]?(?:[0-9]*[.])?[0-9]+)", aa)
            if aa_mod is None:
                residues[aa].add(None)
            else:
                residues[aa_mod[1]].add(aa_mod[2])
        fixed_mods, variable_mods = [], []
        for aa, mods in residues.items():
            if len(mods) > 1:
                for mod in mods:
                    if mod is not None:
                        variable_mods.append((aa, mod))
            elif None not in mods:
                fixed_mods.append((aa, mods.pop()))
        # Write the mzTab output file header.
        stub_name = os.path.splitext(os.path.basename(output))[0]
        metadata = [
            ("mzTab-version", "1.0.0"),
            ("mzTab-mode", "Summary"),
            ("mzTab-type", "Identification"),
            ("description", f"Casanovo identification file {stub_name}"),
            (
                "ms_run[1]-location",
                pathlib.Path(os.path.abspath(peak_dir)).as_uri(),
            ),
            (
                "psm_search_engine_score[1]",
                "[MS, MS:1001143, search engine specific score for PSMs, ]",
            ),
            ("software[1]", f"[MS, MS:1001456, Casanovo, {__version__}, ]"),
        ]
        if len(fixed_mods) == 0:
            metadata.append(
                (
                    "fixed_mod[1]",
                    "[MS, MS:1002453, No fixed modifications searched, ]",
                )
            )
        else:
            for i, (aa, mod) in enumerate(fixed_mods):
                metadata.append(
                    (f"fixed_mod[{i}]", f"[CHEMMOD, CHEMMOD:{mod}, , ]")
                )
                metadata.append(
                    (f"fixed_mod[{i}]-site", aa if aa else "N-term")
                )
        if len(variable_mods) == 0:
            metadata.append(
                (
                    "variable_mod[1]",
                    "[MS, MS:1002454, No variable modifications searched,]",
                )
            )
        else:
            for i, (aa, mod) in enumerate(variable_mods):
                metadata.append(
                    (f"variable_mod[{i}]", f"[CHEMMOD, CHEMMOD:{mod}, , ]")
                )
                metadata.append(
                    (f"variable_mod[{i}]-site", aa if aa else "N-term")
                )
        for i, (key, value) in enumerate(config.items()):
            if key not in ("residues",):
                metadata.append(
                    (f"software[1]-setting[{i}]", f"{key} = {value}")
                )
        with open(f"{os.path.splitext(output)[0]}.mztab", "w") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            for row in metadata:
                writer.writerow(row)
        # Get the peptide predictions.
        model_runner.predict(peak_dir, model, output, config)
    elif mode == "eval":
        logger.info("Evaluate a trained Casanovo model.")
        model_runner.evaluate(peak_dir, model, config)
    elif mode == "train":
        logger.info("Train the Casanovo model.")
        model_runner.train(peak_dir, peak_dir_val, model, config)


if __name__ == "__main__":
    main()
