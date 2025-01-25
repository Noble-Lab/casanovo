import os
from typing import Dict, Iterable, List, Tuple

import depthcharge.utils
import pandas as pd
import torch
from depthcharge.constants import PROTON, H2O

from .dataloaders import AnnotatedSpectrumDataset


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
        give_max_mz: bool = True,
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

        mass_one = (masses * is_separator).sum(dim=1, keepdim=True)
        mass_two = masses[:, -1] - mass_one
        mz_two = (mass_two + H2O) / charges + PROTON
        mz_one = ((mass_one + H2O) / charges + PROTON) * is_chimeric
        calc_mz = torch.cat((mz_one, mz_two), dim=1)

        if give_max_mz:
            calc_mz = calc_mz.max(dim=1).values

        return calc_mz


class MskbChimeraTokenizer(ChimeraTokenizer):
    _parse_peptide = depthcharge.primitives.Peptide.from_massivekb


class ChimeraAnnotatedSpectrumDataset(AnnotatedSpectrumDataset):
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
        super().__init__(
            spectra,
            annotations,
            tokenizer,
            batch_size,
            path,
            parse_kwargs,
            **kwargs,
        )

    def _to_tensor(self, batch):
        """Convert a record batch to tensor

        see depthcharge.AnnotatedSpectrumDataset._to_tensor
        """
        batch = super(AnnotatedSpectrumDataset, self)._to_tensor(batch)
        batch[
            self.annotations + "_compliment"
        ] = self.tokenizer.tokenize_compliment(
            batch[self.annotations],
            add_start=self.tokenizer.start_token is not None,
            add_stop=self.tokenizer.stop_token is not None,
        )

        batch[self.annotations] = self.tokenizer.tokenize(
            batch[self.annotations],
            add_start=self.tokenizer.start_token is not None,
            add_stop=self.tokenizer.stop_token is not None,
        )

        return batch
