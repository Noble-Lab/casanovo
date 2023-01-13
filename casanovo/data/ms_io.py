"""Mass spectrometry file type input/output operations."""
import collections
import csv
import os
import re
from pathlib import Path
from typing import Any, Dict

from .. import __version__


class MztabWriter:
    """
    Export spectrum identifications to an mzTab file.

    Parameters
    ----------
    filename : str
        The name of the mzTab file.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.metadata = [
            ("mzTab-version", "1.0.0"),
            ("mzTab-mode", "Summary"),
            ("mzTab-type", "Identification"),
            (
                "description",
                f"Casanovo identification file "
                f"{os.path.splitext(os.path.basename(self.filename))[0]}",
            ),
            ("software[1]", f"[MS, MS:1003281, Casanovo, {__version__}]"),
            (
                "psm_search_engine_score[1]",
                "[MS, MS:1001143, search engine specific score for PSMs, ]",
            ),
        ]
        self.psms = []

    def set_metadata(
        self, filename_in: str, config: Dict[str, Any], **kwargs
    ) -> None:
        """
        Specify metadata information to write to the mzTab header.

        Parameters
        ----------
        filename_in : str
            The name or directory of the input file(s).
        config : Dict[str, Any]
            The active configuration options.
        kwargs
            Additional configuration options (i.e. from command-line arguments).
        """
        # Derive the fixed and variable modifications from the residue alphabet.
        known_mods = {
            "+57.021": "[UNIMOD, UNIMOD:4, Carbamidomethyl, ]",
            "+15.995": "[UNIMOD, UNIMOD:35, Oxidation, ]",
            "+0.984": "[UNIMOD, UNIMOD:7, Deamidated, ]",
            "+42.011": "[UNIMOD, UNIMOD:1, Acetyl, ]",
            "+43.006": "[UNIMOD, UNIMOD:5, Carbamyl, ]",
            "-17.027": "[UNIMOD, UNIMOD:385, Ammonia-loss, ]",
        }
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

        # Add all config values to the mzTab metadata section.
        self.metadata.append(
            (
                "ms_run[1]-location",
                Path(os.path.abspath(filename_in)).as_uri(),
            ),
        )
        if len(fixed_mods) == 0:
            self.metadata.append(
                (
                    "fixed_mod[1]",
                    "[MS, MS:1002453, No fixed modifications searched, ]",
                )
            )
        else:
            for i, (aa, mod) in enumerate(fixed_mods, 1):
                self.metadata.append(
                    (
                        f"fixed_mod[{i}]",
                        known_mods.get(mod, f"[CHEMMOD, CHEMMOD:{mod}, , ]"),
                    )
                )
                self.metadata.append(
                    (f"fixed_mod[{i}]-site", aa if aa else "N-term")
                )
        if len(variable_mods) == 0:
            self.metadata.append(
                (
                    "variable_mod[1]",
                    "[MS, MS:1002454, No variable modifications searched,]",
                )
            )
        else:
            for i, (aa, mod) in enumerate(variable_mods, 1):
                self.metadata.append(
                    (
                        f"variable_mod[{i}]",
                        known_mods.get(mod, f"[CHEMMOD, CHEMMOD:{mod}, , ]"),
                    )
                )
                self.metadata.append(
                    (f"variable_mod[{i}]-site", aa if aa else "N-term")
                )
        for i, (key, value) in enumerate(kwargs.items(), 1):
            self.metadata.append(
                (f"software[1]-setting[{i}]", f"{key} = {value}")
            )
        for i, (key, value) in enumerate(config.items(), len(kwargs) + 1):
            if key not in ("residues",):
                self.metadata.append(
                    (f"software[1]-setting[{i}]", f"{key} = {value}")
                )

    def save(self) -> None:
        """
        Export the spectrum identifications to the mzTab file.
        """
        with open(self.filename, "w") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator=os.linesep)
            # Write metadata.
            for row in self.metadata:
                writer.writerow(["MTD", *row])
            # Write PSMs.
            writer.writerow(
                [
                    "PSH",
                    "sequence",
                    "PSM_ID",
                    "accession",
                    "unique",
                    "database",
                    "database_version",
                    "search_engine",
                    "search_engine_score[1]",
                    "modifications",
                    "retention_time",
                    "charge",
                    "exp_mass_to_charge",
                    "calc_mass_to_charge",
                    "spectra_ref",
                    "pre",
                    "post",
                    "start",
                    "end",
                    "opt_ms_run[1]_aa_scores",
                ]
            )
            for psm in self.psms:
                writer.writerow(
                    [
                        "PSM",
                        psm[0],  # sequence
                        psm[1],  # PSM_ID
                        "null",  # accession
                        "null",  # unique
                        "null",  # database
                        "null",  # database_version
                        f"[MS, MS:1003281, Casanovo, {__version__}]",
                        psm[2],  # search_engine_score[1]
                        # FIXME: Modifications should be specified as
                        #  controlled vocabulary terms.
                        "null",  # modifications
                        # FIXME: Can we get the retention time from the data
                        #  loader?
                        "null",  # retention_time
                        psm[3],  # charge
                        psm[4],  # exp_mass_to_charge
                        psm[5],  # calc_mass_to_charge
                        f"ms_run[1]:index={psm[1]}",  # spectra_ref
                        "null",  # pre
                        "null",  # post
                        "null",  # start
                        "null",  # end
                        psm[6],  # opt_ms_run[1]_aa_scores
                    ]
                )
