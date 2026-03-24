"""Chimera tokenizer and dataset for co-fragmented spectrum sequencing."""

import os
from typing import Dict, Iterable, List, Tuple

import depthcharge.primitives
import depthcharge.utils
import pandas as pd
import torch
from depthcharge.constants import H2O, PROTON

from .dataloaders import AnnotatedSpectrumDataset


class ChimeraTokenizer(depthcharge.tokenizers.peptides.PeptideTokenizer):
    """A peptide tokenizer extended with a chimeric separator token.

    Adds a separator character (default ``:``) with mass 0 to the amino acid
    alphabet so that two co-fragmented peptides can be represented as a single
    concatenated sequence ``pep1:pep2``.

    Parameters
    ----------
    residues : Dict[str, float] | None
        Custom residue mass dictionary.  If ``None``, uses the default
        canonical amino acids.
    replace_isoleucine_with_leucine : bool
        Replace isoleucine with leucine.
    reverse : bool
        Reverse peptide sequences during tokenization.
    start_token : str | None
        Start token string.
    stop_token : str | None
        Stop token string (default ``"$"``).
    chimeric_separator_token : str
        The separator character inserted between the two peptides of a chimera
        (default ``":"``).
    """

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
    ) -> List[str]:
        """Return the complement ordering of chimeric sequences.

        For a sequence ``"pep1:pep2"`` returns ``"pep2:pep1"``.

        Parameters
        ----------
        sequences : Iterable[str] | str
            One or more chimeric peptide strings.

        Returns
        -------
        List[str]
            Sequences with the order of sub-peptides reversed.
        """
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
    ) -> torch.Tensor | List[List[str]]:
        """Tokenize the complement ordering of chimeric sequences.

        Parameters
        ----------
        sequences : Iterable[str] | str
            One or more chimeric peptide strings.
        add_start : bool
            Prepend a start token.
        add_stop : bool
            Append a stop token.
        to_strings : bool
            Return token strings instead of integer indices.

        Returns
        -------
        torch.Tensor | List[List[str]]
            Tokenized complement sequences.
        """
        return self.tokenize(
            self.compliment(sequences),
            add_start=add_start,
            add_stop=add_stop,
            to_strings=to_strings,
        )

    def split(self, sequence: str) -> list[str]:
        """Split a chimeric sequence into individual token strings.

        Parameters
        ----------
        sequence : str
            A peptide sequence, optionally containing one chimeric separator.

        Returns
        -------
        list[str]
            The sequence split into token strings.
        """
        peptides = sequence.split(self.chimeric_separator_token)
        if len(peptides) in [1, 2]:
            split = super().split(peptides[0])
            if len(peptides) == 2:
                split += [self.chimeric_separator_token]
                split += super().split(peptides[1])
        else:
            raise ValueError(
                f"Sequence {sequence} contains more than one chimeric "
                "separator; sequences can contain at most one chimeric "
                "separator."
            )
        return split

    def calculate_precursor_ions(
        self,
        tokens: torch.Tensor | Iterable[str],
        charges: torch.Tensor,
        give_max_mz: bool = True,
    ) -> torch.Tensor:
        """Calculate precursor m/z for one or two sub-peptides.

        For a chimeric sequence the mass is split at the separator token.  For
        a non-chimeric sequence (no separator) the standard precursor m/z is
        returned.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_sequences, len_seq) or list of str
            Token indices or string sequences.
        charges : torch.Tensor of shape (n_sequences,)
            Precursor charge states.
        give_max_mz : bool
            If ``True`` (default) return the maximum m/z across both
            sub-peptides.  If ``False`` return a ``(n_sequences, 2)`` tensor
            with both m/z values.

        Returns
        -------
        torch.Tensor
            Shape ``(n_sequences,)`` when *give_max_mz* is ``True``, otherwise
            ``(n_sequences, 2)``.
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
        mass_two = masses[:, -1:] - mass_one
        mz_two = (mass_two + H2O) / charges.unsqueeze(-1) + PROTON
        mz_one = (
            (mass_one + H2O) / charges.unsqueeze(-1) + PROTON
        ) * is_chimeric.unsqueeze(-1)
        calc_mz = torch.cat((mz_one, mz_two), dim=1)

        if give_max_mz:
            calc_mz = calc_mz.max(dim=1).values

        return calc_mz


class MskbChimeraTokenizer(ChimeraTokenizer):
    """ChimeraTokenizer using MassIVE-KB peptide parsing."""

    _parse_peptide = depthcharge.primitives.Peptide.from_massivekb


class ChimeraAnnotatedSpectrumDataset(AnnotatedSpectrumDataset):
    """AnnotatedSpectrumDataset that also provides complement token sequences.

    In addition to the standard ``batch["seq"]`` field this dataset populates
    ``batch["seq_compliment"]`` with the complement ordering of chimeric
    sequences (``pep2:pep1`` when the annotation is ``pep1:pep2``).  Both
    fields are required by the chimera min-loss training objective.

    Parameters
    ----------
    spectra : pd.DataFrame | os.PathLike | Iterable[os.PathLike]
        Input spectra.
    annotations : str
        Name of the annotation column / field.
    tokenizer : ChimeraTokenizer
        A chimera-capable tokenizer.
    batch_size : int
        Batch size.
    path : os.PathLike, optional
        Optional path for the Lance index.
    parse_kwargs : Dict | None, optional
        Extra keyword arguments for the parser.
    **kwargs
        Additional arguments forwarded to the parent class.
    """

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
        """Convert a record batch to tensors, adding the complement sequence.

        Overrides the parent to additionally store
        ``batch["<annotations>_compliment"]`` containing the complement
        ordering tokenized sequence alongside the standard annotation tokens.
        """
        batch = super(AnnotatedSpectrumDataset, self)._to_tensor(batch)
        batch[self.annotations + "_compliment"] = (
            self.tokenizer.tokenize_compliment(
                batch[self.annotations],
                add_start=self.tokenizer.start_token is not None,
                add_stop=self.tokenizer.stop_token is not None,
            )
        )
        batch[self.annotations] = self.tokenizer.tokenize(
            batch[self.annotations],
            add_start=self.tokenizer.start_token is not None,
            add_stop=self.tokenizer.stop_token is not None,
        )
        return batch
