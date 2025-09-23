"""Mass spectrometry file type input/output operations."""

import collections
import csv
import operator
import os
import re
from pathlib import Path
from typing import List, Tuple

import natsort
import pyteomics.proforma

from .. import __version__
from ..config import Config
from .psm import PepSpecMatch


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
        self._run_map = {}
        self.psms: List[PepSpecMatch] = []

    def set_metadata(self, config: Config, **kwargs) -> None:
        """
        Specify metadata information to write to the mzTab header.

        Parameters
        ----------
        config : Config
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

    def set_ms_run(self, peak_filenames: List[str]) -> None:
        """
        Add input peak files to the mzTab metadata section.

        Parameters
        ----------
        peak_filenames : List[str]
            The input peak file name(s).
        """
        for i, filename in enumerate(natsort.natsorted(peak_filenames), 1):
            filename = os.path.abspath(filename)
            self.metadata.append(
                (f"ms_run[{i}]-location", Path(filename).as_uri()),
            )
            self._run_map[Path(filename).name] = i

    @staticmethod
    def get_mod_string(
        mod: pyteomics.proforma.TagBase,
        residue: str = "N-term",
        position: int = 0,
    ) -> str:
        """
        Format a ProForma modification into an mzTab-style string.

        Parameters
        ----------
        mod : pyteomics.proforma.TagBase
            A modification tag object parsed from a ProForma sequence.
            This can be a Unimod/PSI-MOD modification, a generic
            modification, or a mass-only delta.
        residue : str, default="N-term"
            The residue associated with the modification. For
            N-terminal modifications, use `"N-term"`.
        position : int, default=0
            Position of the modification in the peptide sequence.
            Use `0` for N-terminal, `len(sequence)+1` for C-terminal,
            and a 1-based index for internal residues.

        Returns
        -------
        str
            The mzTab-formatted modification string. Examples:
        """
        if hasattr(mod, "name"):
            mod_str = f"{position}-{mod.name} ({residue})"
            # If known unimod modification, add id
            if hasattr(mod, "id") and mod.id is not None:
                mod_str = f"{mod_str}:UNIMOD:{mod.id}"
        elif hasattr(mod, "mass"):
            mod_str = f"{position}-[{mod.mass:+.4f}]"
        else:
            mod_str = f"{position}-{str(mod)}"

        return mod_str

    @staticmethod
    def parse_sequence(seq: str) -> Tuple[str, str]:
        """
        Parse a ProForma peptide sequence into a plain amino acid sequence and
        an mzTab-formatted modifications string.

        Parameters
        ----------
        seq : str
            A peptide sequence in ProForma notation.

        Returns
        -------
        aa_seq : str
            The plain amino acid sequence with modifications stripped.
        mod_string : str
            A semicolon-delimited string of modifications in mzTab
            format, suitable for reporting in the PSM section.
        """
        seq_mod, term = pyteomics.proforma.parse(seq)
        aa_seq = "".join(res for res, _ in seq_mod)
        n_term_mods = term["n_term"]

        if n_term_mods is None:
            mod_strings = []
        else:
            mod_strings = [
                MztabWriter.get_mod_string(curr) for curr in n_term_mods
            ]

        for position, (res, mods) in enumerate(seq_mod, start=1):
            if mods is None:
                continue

            for mod in mods:
                mod = MztabWriter.get_mod_string(
                    mod, residue=res, position=position
                )
                mod_strings.append(mod)

        combined_mod_string = "; ".join(mod_strings)
        return aa_seq, combined_mod_string

    def save(self) -> None:
        """
        Export the spectrum identifications to the mzTab file.
        """
        with open(self.filename, "w", newline="") as f:
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
            by_id = operator.attrgetter("spectrum_id")
            for i, psm in enumerate(
                natsort.natsorted(self.psms, key=by_id),
                1,
            ):
                filename, idx = psm.spectrum_id
                if Path(filename).suffix.lower() == ".mgf" and idx.isnumeric():
                    idx = f"index={idx}"

                seq, mods = self.parse_sequence(psm.sequence)

                writer.writerow(
                    [
                        "PSM",
                        seq,  # sequence
                        i,  # PSM_ID
                        psm.protein,  # accession
                        "null",  # unique
                        "null",  # database
                        "null",  # database_version
                        f"[MS, MS:1003281, Casanovo, {__version__}]",
                        psm.peptide_score,  # search_engine_score[1]
                        # FIXME: Modifications should be specified as
                        #  controlled vocabulary terms.
                        mods if mods else "null",  # modifications
                        # FIXME: Can we get the retention time from the data
                        #  loader?
                        "null",  # retention_time
                        psm.charge,  # charge
                        psm.exp_mz,  # exp_mass_to_charge
                        psm.calc_mz,  # calc_mass_to_charge
                        f"ms_run[{self._run_map[filename]}]:{idx}",
                        "null",  # pre
                        "null",  # post
                        "null",  # start
                        "null",  # end
                        # opt_ms_run[1]_aa_scores
                        ",".join(list(map("{:.5f}".format, psm.aa_scores))),
                    ]
                )
