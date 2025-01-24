import os
from typing import Dict, Iterable, List

import depthcharge.utils
import pandas as pd
import torch

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
