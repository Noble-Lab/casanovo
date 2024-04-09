"""Methods used to annotate an .mgf so that it can be used by Casanovo-DB"""

from pathlib import Path
from typing import Optional, Tuple
import os
import re
import logging

import pandas as pd
import pyteomics.mgf as mgf


def _normalize_mods(seq: str) -> str:
    """
    Turns tide-style modifications into the format used by Casanovo-DB.

        Parameters
        ----------
        seq : str
            The peptide sequence with tide-style modifications.

        Returns
        -------
        str
            The peptide sequence with Casanovo-DB-style modifications.
    """
    logger = logging.getLogger("casanovo")
    seq = seq.replace("C", "C+57.021")
    seq = re.sub(r"M\[15\.[0-9]*\]", r"M+15.995", seq)
    seq = re.sub(r"N\[0\.9[0-9]*\]", r"N+0.984", seq)
    seq = re.sub(r"Q\[0\.9[0-9]*\]", r"Q+0.984", seq)
    seq = re.sub(r"(.*)\[42\.[0-9]*\]", r"+42.011\1", seq)
    seq = re.sub(r"(.*)\[43\.[0-9]*\]", r"+43.006\1", seq)
    seq = re.sub(r"(.*)\[\-17\.[0-9]*\]", r"-17.027\1", seq)
    seq = re.sub(r"(.*)\[25\.[0-9]*\]", r"+43.006-17.027\1", seq)
    return seq


def annotate_mgf(peak_path: str, tide_path: str, output: Optional[str]):
    """
    Accepts a directory containing the results of a successful tide search, and an .mgf file containing MS/MS spectra.
    The .mgf file is then annotated in the SEQ field with all of the candidate peptides for each spectrum, as well as their target/decoy status.
    This annotated .mgf can be given directly to Casanovo-DB to perfrom a database search.

        Parameters
        ----------
        tide_dir_path : str
            Path to the directory containing the results of a successful tide search.
        mgf_file : str
            Path to the .mgf file containing MS/MS spectra.
        output_file : str
            Path to where the annotated .mgf will be written.

    """
    logger = logging.getLogger("casanovo")
    # Get paths to tide search text files
    tdf_path = os.path.join(tide_path, "tide-search.target.txt")
    ddf_path = os.path.join(tide_path, "tide-search.decoy.txt")
    try:
        target_df = pd.read_csv(
            tdf_path, sep="\t", usecols=["scan", "sequence", "target/decoy"]
        )
        decoy_df = pd.read_csv(
            ddf_path, sep="\t", usecols=["scan", "sequence", "target/decoy"]
        )
    except FileNotFoundError as e:
        logger.error(
            "Could not find tide search results in the specified directory. "
            "Please ensure that the directory contains the following files: "
            "tide-search.target.txt and tide-search.decoy.txt"
        )
        raise e

    logger.info("Successfully read tide search results from %s.", tide_path)

    df = pd.concat([target_df, decoy_df])
    scan_groups = df.groupby("scan")[["sequence", "target/decoy"]]

    scan_map = {}

    for scan, item in scan_groups:
        td_group = item.groupby("target/decoy")["sequence"].apply(list)
        if "target" in td_group.index:
            target_candidate_list = list(
                map(
                    _normalize_mods,
                    td_group["target"],
                )
            )
        else:
            target_candidate_list = []
            logger.warn(f"No target peptides found for scan {scan}.")
        if "decoy" in td_group.index:
            decoy_candidate_list = list(
                map(
                    _normalize_mods,
                    td_group["decoy"],
                )
            )
            decoy_candidate_list = list(
                map(lambda x: "decoy_" + str(x), decoy_candidate_list)
            )
        else:
            decoy_candidate_list = []
            logger.warn(f"No decoy peptides found for scan {scan}.")

        scan_map[scan] = target_candidate_list + decoy_candidate_list

    all_spec = []
    for idx, spec_dict in enumerate(mgf.read(peak_path)):
        try:
            scan = int(spec_dict["params"]["scans"])
        except KeyError as e:
            logger.error(
                "Could not find the scan number in the .mgf file. Please ensure that the .mgf file contains the scan number in the 'SCANS' field."
            )
            raise e
        try:
            spec_dict["params"]["seq"] = ",".join(list(scan_map[scan]))
            all_spec.append(spec_dict)
        except KeyError as e:
            # No need to do anything if the scan is not found in the scan map
            pass
    try:
        output = str(output)
        logger.info(output)
        mgf.write(all_spec, output, file_mode="w")
        logger.info("Annotated .mgf file written to %s.", output)
    except Exception as e:
        print(f"Write to {output} failed. Check if the file path is correct.")
        print(e)
