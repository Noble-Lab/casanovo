"""Mass spectrometry file type input/output operations."""

import collections
import csv
import operator
import os
import re
from pathlib import Path
from typing import List

import natsort

from .. import __version__
from ..config import Config
from .psm import PepSpecMatch

def _build_mgf_scan_index(mgf_path: str) -> dict:
    """
    Build an index mapping spectrum position to SCANS value for an MGF file.

    Reads only the header lines of each spectrum entry (never the peak
    data), so this is very fast even for large files.

    Parameters
    ----------
    mgf_path : str
        Path to the MGF file.

    Returns
    -------
    dict
        Mapping from spectrum index (as a string, e.g. ``"0"``) to the
        SCANS field value (e.g. ``"17"``). Only spectra that contain a
        ``SCANS``, ``SCAN``, or ``SCAN ID`` header field are included.
    """
    index2scan = {}
    index = 0
    current_scan = None
    in_ions = False
    try:
        with open(mgf_path, "r", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                upper = stripped.upper()
                if upper == "BEGIN IONS":
                    in_ions = True
                    current_scan = None
                elif upper == "END IONS":
                    if current_scan is not None:
                        index2scan[str(index)] = current_scan
                    index += 1
                    in_ions = False
                elif in_ions:
                    for prefix in ("SCANS=", "SCAN=", "SCAN ID="):
                        if upper.startswith(prefix):
                            current_scan = stripped.split("=", 1)[1].strip()
                            break
    except OSError:
        pass
    return index2scan


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
        self._mgf_scan_index = {}  # {(filename_base, index_str): scan_num_str}
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

    def set_mgf_scan_index(self, peak_filenames: List[str]) -> None:
        """
        Pre-compute the MGF scan number index for the given peak files.

        For each MGF file, reads the ``SCANS`` (or ``SCAN`` / ``SCAN ID``)
        header field of every spectrum entry and stores a mapping from
        spectrum index to scan number.  Called once before prediction so
        that ``save()`` can embed the scan number in the ``spectra_ref``
        column without re-reading the files at write time.

        Parameters
        ----------
        peak_filenames : List[str]
            The input peak file paths (mzML files are silently ignored).
        """
        for path in peak_filenames:
            if Path(path).suffix.lower() != ".mgf":
                continue
            name = Path(path).name
            for idx_str, scan_num in _build_mgf_scan_index(path).items():
                self._mgf_scan_index[(name, idx_str)] = scan_num

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
                    "opt_ms_run[1]_proforma",
                ]
            )
            by_id = operator.attrgetter("spectrum_id")
            for i, psm in enumerate(
                natsort.natsorted(self.psms, key=by_id),
                1,
            ):
                filename, idx = psm.spectrum_id
                if Path(filename).suffix.lower() == ".mgf" and idx.isnumeric():
                    scan_num = self._mgf_scan_index.get((filename, idx))
                    idx = f"index={idx}"
                    if scan_num:
                        idx = f"{idx} scan={scan_num}"

                writer.writerow(
                    [
                        "PSM",
                        psm.aa_sequence,  # sequence
                        i,  # PSM_ID
                        psm.protein,  # accession
                        "null",  # unique
                        "null",  # database
                        "null",  # database_version
                        f"[MS, MS:1003281, Casanovo, {__version__}]",
                        psm.peptide_score,  # search_engine_score[1]
                        # FIXME: Modifications should be specified as
                        #  controlled vocabulary terms.
                        psm.modifications,  # modifications
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
                        psm.sequence,  # op_ms_run[1]_proforma
                    ]
                )
