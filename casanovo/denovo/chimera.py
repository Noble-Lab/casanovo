import dataclasses
import pathlib
import copy
import os
import warnings
import tempfile
import uuid
import collections.abc
from os import PathLike
from typing import Dict, Iterable, List, Tuple, Callable, TextIO, Generator

import depthcharge.constants
import lance
import depthcharge.utils
import depthcharge.data.parsers
import depthcharge.data.fields
import depthcharge.data.spectrum_datasets
import depthcharge.primitives
import pandas as pd
import polars as pl
import pyarrow as pa
import torch
import tqdm
import cloudpathlib
import numpy as np
from numpy.typing import ArrayLike


class ChimeraTokenizer(depthcharge.tokenizers.peptides.PeptideTokenizer):
    def __init__(
        self,
        residues: Dict[str, float] | None = None,
        replace_isoleucine_with_leucine: bool = False,
        reverse: bool = False,
        start_token: str | None = None,
        stop_token: str | None = "$",
        chimeric_separator_token: str = ":",
    ) -> None:
        self.chimeric_separator_token = chimeric_separator_token
        residues = dict() if residues is None else residues
        residues[chimeric_separator_token] = 0.0

        super().__init__(
            residues=residues,
            replace_isoleucine_with_leucine=replace_isoleucine_with_leucine,
            reverse=reverse,
            start_token=start_token,
            stop_token=stop_token,
        )

    def compliment(
        self,
        sequences: Iterable[str] | str,
    ) -> Iterable[str]:
        """Get compliment sequences"""
        compliment_sequences = []
        for seq in depthcharge.utils.listify(sequences):
            peptides = seq.split(self.chimeric_separator_token)
            compliment = self.chimeric_separator_token.join(peptides[::-1])
            compliment_sequences.append(compliment)

        return compliment_sequences

    def tokenize_compliment(
        self,
        sequences: Iterable[str] | str,
        add_start: bool = False,
        add_stop: bool = False,
        to_strings: bool = False,
    ) -> torch.tensor | List[List[str]]:
        """Tokenize compliment sequences"""
        return self.tokenize(
            self.compliment(sequences),
            add_start=add_start,
            add_stop=add_stop,
            to_strings=to_strings,
        )

    def split(self, sequence: str) -> list[str]:
        """Split chimera peptide sequence"""
        peptides = sequence.split(self.chimeric_separator_token)
        if len(peptides) in [1, 2]:
            split = super().split(peptides[0])
            if len(peptides) == 2:
                split += [self.chimeric_separator_token]
                split += peptides[1]
        else:
            raise ValueError(
                f"Sequence {sequence} contains more than chimeric separator,"
                " sequences can contain at most one chimeric separators."
            )

        return split

    def calculate_precursor_ions(
        self,
        tokens: torch.Tensor | Iterable[str],
        charges: torch.Tensor,
        give_max_mz: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the m/z for precursor ions.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_sequences, len_seq)
            The tokens corresponding to the peptide sequence.
        charges : torch.Tensor of shape (n_sequences,)
            The charge state for each peptide.
        give_max_mz : bool (default True)
            Whether to return the max m/z for each peptide in a chimera, or
            whether to return both

        Returns
        -------
        torch.Tensor.
            The monoisotopic m/z for each charged peptide. Will be size
            (n_sequences,) if max_mz is set to true, (n_sequences, 2)
            otherwise. In the case that give_max is set to true the

        """
        if isinstance(tokens[0], str):
            tokens = self.tokenize(depthcharge.utils.listify(tokens))

        if not isinstance(charges, torch.Tensor):
            charges = torch.tensor(charges)
            if not charges.shape:
                charges = charges[None]

        chimera_separator = self.index[self.chimeric_separator_token]
        masses = self.masses[tokens].cumsum(dim=1)
        is_separator = tokens == chimera_separator
        is_chimeric = is_separator.sum(dim=1)
        if is_chimeric.max().item() > 1:
            raise ValueError(
                "Sequences can contain at most one chimeric separator."
            )
        is_chimeric = is_chimeric.to(torch.bool)

        mass_one = (masses * is_separator).sum(dim=1, keepdim=True)
        mass_one[~is_chimeric] = masses[~is_chimeric, -1].unsqueeze(1)

        if give_max_mz:
            mass_two = masses[:, -1] - mass_one
            calc_mz = torch.cat((mass_one, mass_two), dim=1)
            calc_mz = calc_mz.max(dim=1).values
        else:
            calc_mz = mass_one.squeeze(-1)

        return calc_mz


class MskbChimeraTokenizer(ChimeraTokenizer):
    _parse_peptide = depthcharge.primitives.Peptide.from_massivekb


class ChimeraSpectrumDataset(depthcharge.data.SpectrumDataset):
    def __init__(
        self,
        spectra: pl.DataFrame | PathLike | Iterable[PathLike],
        batch_size: int,
        path: PathLike | None = None,
        parse_kwargs: dict | None = None,
        **kwargs: dict,
    ) -> None:
        """Initialize a SpectrumDataset."""
        self._parse_kwargs = {} if parse_kwargs is None else parse_kwargs
        self._init_kwargs = copy.copy(self._parse_kwargs)
        self._init_kwargs["batch_size"] = 128
        self._init_kwargs["progress"] = False

        self._tmpdir = None
        if path is None:
            # Create a random temporary file:
            self._tmpdir = tempfile.TemporaryDirectory()
            path = pathlib.Path(self._tmpdir.name) / f"{uuid.uuid4()}.lance"

        self._path = cloudpathlib.AnyPath(path)
        if self._path.suffix != ".lance":
            self._path = self._path.with_suffix(".lance")

        # Now parse spectra.
        if spectra is not None:
            spectra = depthcharge.utils.listify(spectra)
            batch = next(_get_records(spectra, **self._init_kwargs))
            lance.write_dataset(
                _get_records(spectra, **self._parse_kwargs),
                str(self._path),
                mode="overwrite" if self._path.exists() else "create",
                schema=batch.schema,
            )

        elif not self._path.exists():
            raise ValueError("No spectra were provided")

        dataset = lance.dataset(str(self._path))
        if "to_tensor_fn" not in kwargs:
            kwargs["to_tensor_fn"] = self._to_tensor

        super(depthcharge.data.SpectrumDataset, self).__init__(
            dataset, batch_size, **kwargs
        )


class ChimeraAnnotatedSpectrumDataset(ChimeraSpectrumDataset):
    """See depthcharge.AnnotatedSpectrumDataset"""

    def __init__(
        self,
        spectra: pd.DataFrame | os.PathLike | Iterable[os.PathLike],
        annotations: str,
        tokenizer: ChimeraTokenizer,
        batch_size: int,
        path: os.PathLike = None,
        parse_kwargs: Dict | None = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.annotations = annotations
        super().__init__(
            spectra=spectra,
            batch_size=batch_size,
            path=path,
            parse_kwargs=parse_kwargs,
            **kwargs,
        )

    def _to_tensor(self, batch):
        """Convert a record batch to tensor

        see depthcharge.AnnotatedSpectrumDataset._to_tensor
        """
        batch = super()._to_tensor(batch)
        sequence = batch[self.annotations]
        batch[self.annotations] = self.tokenizer.tokenize(
            sequence,
            add_start=self.tokenizer.start_token is not None,
            add_stop=self.tokenizer.stop_token is not None,
        )
        batch[
            self.annotations + "_compliment"
        ] = self.tokenizer.tokenize_compliment(
            sequence,
            add_start=self.tokenizer.start_token is not None,
            add_stop=self.tokenizer.stop_token is not None,
        )

        return batch


@dataclasses.dataclass
class ChimericMassSpectrum(depthcharge.primitives.MassSpectrum):
    def __init__(
        self,
        filename: str,
        scan_id: str,
        precursor_charge_one: int,
        precursor_charge_two: int,
        precursor_mass: float,
        mz: ArrayLike,
        intensity: ArrayLike,
        **kwargs,
    ) -> None:
        super().__init__(
            filename=filename,
            scan_id=scan_id,
            mz=mz,
            intensity=intensity,
            # We have to lie to depthcharge here
            precursor_mz=precursor_mass + depthcharge.constants.PROTON,
            precursor_charge=1,
            **kwargs,
        )

        self.precursor_charge_one = precursor_charge_one
        self.precursor_charge_two = precursor_charge_two


class ChimeraMgfParser(depthcharge.data.parsers.MgfParser):
    Spectrum = Dict[str, Dict[str, str] | List[str]]

    def __init__(
        self,
        peak_file: PathLike,
        ms_level: int = 2,
        preprocessing_fn: Callable | Iterable[Callable] | None = None,
        valid_charge: Iterable[int] | None = None,
        custom_fields: dict[str, Iterable[str]] | None = None,
        progress: bool = True,
    ) -> None:
        super().__init__(
            peak_file=peak_file,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            custom_fields=custom_fields,
            progress=progress,
        )

        # Re-define schema
        self.schema = pa.schema(
            [
                pa.field("peak_file", pa.string()),
                pa.field("scan_id", pa.string()),
                pa.field("ms_level", pa.uint8()),
                pa.field("precursor_mass", pa.float64()),
                pa.field("precursor_charge_one", pa.int16()),
                pa.field("precursor_charge_two", pa.int16()),
                pa.field("mz_array", pa.list_(pa.float64())),
                pa.field("intensity_array", pa.list_(pa.float64())),
            ]
        )

        if self.custom_fields is not None:
            self.custom_fields = depthcharge.utils.listify(self.custom_fields)
            for field in self.custom_fields:
                self.schema = self.schema.append(
                    pa.field(field.name, field.dtype)
                )

    def read_mgf(self, mgf_file: TextIO) -> Generator[Spectrum, None, None]:
        def get_spectrum_dict():
            return {
                "params": dict(),
                "mz_array": list(),
                "intensity_array": list(),
            }

        WHITE_SPACE = 0
        HEADERS = 1
        PEAKS = 2

        result = get_spectrum_dict()
        curr_state = WHITE_SPACE
        for line in mgf_file:
            line = line.strip()
            if curr_state == WHITE_SPACE:
                if line == "BEGIN IONS":
                    curr_state = HEADERS
                continue
            elif curr_state == HEADERS:
                if "=" not in line:
                    mz, intensity = line.split(" ")
                    result["mz_array"].append(float(mz))
                    result["intensity_array"].append(float(intensity))
                    curr_state = PEAKS
                else:
                    param, value = line.split("=")
                    result["params"][param.lower()] = value
            else:
                if line == "END IONS":
                    result["mz_array"] = np.array(result["mz_array"])
                    result["intensity_array"] = np.array(
                        result["intensity_array"]
                    )
                    yield result

                    result = get_spectrum_dict()
                    curr_state = WHITE_SPACE
                else:
                    mz, intensity = line.split(" ")
                    result["mz_array"].append(float(mz))
                    result["intensity_array"].append(float(intensity))

    def iter_batches(self, batch_size: int | None) -> pa.RecordBatch:
        """Iterate over batches of mass spectra in the Arrow format.

        Parameters
        ----------
        batch_size : int or None
            The number of spectra in a batch. ``None`` loads all of
            the spectra in a single batch.

        Yields
        ------
        RecordBatch
            A batch of spectra and their metadata.

        """
        batch_size = float("inf") if batch_size is None else batch_size
        pbar_args = {
            "desc": self.peak_file.name,
            "unit": " spectra",
            "disable": not self.progress,
        }

        n_skipped = 0
        last_exc = None
        with open(self.peak_file) as spectra:
            self._batch = None
            for spectrum in tqdm.tqdm(self.read_mgf(spectra), **pbar_args):
                try:
                    parsed = self.parse_spectrum(spectrum)
                    if parsed is None:
                        continue

                    if self.preprocessing_fn is not None:
                        for processor in self.preprocessing_fn:
                            parsed = processor(parsed)

                    entry = {
                        "peak_file": self.peak_file.name,
                        "scan_id": parsed.scan_id,
                        "ms_level": parsed.ms_level,
                        "precursor_mass": parsed.precursor_mass,
                        "precursor_charge_one": parsed.precursor_charge_one,
                        "precursor_charge_two": parsed.precursor_charge_two,
                        "mz_array": parsed.mz,
                        "intensity_array": parsed.intensity,
                    }

                except (IndexError, KeyError, ValueError) as exc:
                    last_exc = exc
                    n_skipped += 1
                    continue

                # Parse custom fields:
                entry.update(self.parse_custom_fields(spectrum))
                self._update_batch(entry)

                # Update the batch:
                if len(self._batch["scan_id"]) == batch_size:
                    yield self._yield_batch()

            # Get the remainder:
            if self._batch is not None:
                yield self._yield_batch()

        if n_skipped:
            warnings.warn(
                f"Skipped {n_skipped} spectra with invalid information."
                f"Last error was: \n {str(last_exc)}"
            )

    def parse_spectrum(self, spectrum: dict) -> ChimericMassSpectrum:
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in MGF format.

        """
        self._counter += 1
        if self.ms_level is not None and 1 not in self.ms_level:
            precursor_mass = float(spectrum["params"]["pepmass"])
            precursor_charge = spectrum["params"]["charge"]
            precursor_charges = precursor_charge.split(":")
            precursor_charge_one = int(precursor_charges[0][0])

            if len(precursor_charges) == 2:
                precursor_charge_two = int(precursor_charges[1][0])
            else:
                precursor_charge_two = 0
        else:
            precursor_mass, precursor_charge_one, precursor_charge_two = (
                None,
                0,
                0,
            )

        scan_id = str(self._counter)
        if "title" in spectrum["params"]:
            scan_id = spectrum["params"]["title"]

        if (
            self.valid_charge is None
            or precursor_charge_one in self.valid_charge
        ):
            return ChimericMassSpectrum(
                filename=str(self.peak_file),
                scan_id=scan_id,
                precursor_charge_one=precursor_charge_one,
                precursor_charge_two=precursor_charge_two,
                precursor_mass=precursor_mass,
                mz=spectrum["mz_array"],
                intensity=spectrum["intensity_array"],
                ms_level=min(self.ms_level),
            )

        raise ValueError("Invalid precursor charge.")


def _get_records(
    data: list[PathLike], **kwargs: dict
) -> collections.abc.Generator[pa.RecordBatch]:
    for spectra in data:
        spectra = spectra_to_stream(spectra, **kwargs)
        yield from spectra


def spectra_to_stream(
    peak_file: PathLike,
    *,
    batch_size: int | None = 100_000,
    metadata_df: pl.DataFrame | pl.LazyFrame | None = None,
    ms_level: int | Iterable[int] | None = 2,
    preprocessing_fn: Callable | Iterable[Callable] | None = None,
    valid_charge: Iterable[int] | None = None,
    custom_fields: (
        depthcharge.data.fields.CustomField
        | Iterable[depthcharge.data.fields.CustomField]
        | None
    ) = None,
    progress: bool = True,
) -> collections.abc.Generator[pa.RecordBatch]:
    """see depthcharge.arror.spectra_to_stream"""
    parser_args = {
        "ms_level": ms_level,
        "valid_charge": valid_charge,
        "preprocessing_fn": preprocessing_fn,
        "custom_fields": custom_fields,
        "progress": progress,
    }

    on_cols = ["scan_id"]
    validation = "1:1"
    if metadata_df is not None:
        metadata_df = metadata_df.lazy()
        if "peak_file" in metadata_df.columns:
            # Validation is only supported when on is a single column.
            # Adding a footgun here to remove later...
            validation = "m:m"
            on_cols.append("peak_file")

    parser = ChimeraMgfParser(peak_file, **parser_args)
    for batch in parser.iter_batches(batch_size=batch_size):
        if metadata_df is not None:
            batch = (
                pl.from_arrow(batch)
                .lazy()
                .join(
                    metadata_df,
                    on=on_cols,
                    how="left",
                    validate=validation,
                )
                .collect()
                .to_arrow()
                .to_batches(max_chunksize=batch_size)[0]
            )

        yield batch
