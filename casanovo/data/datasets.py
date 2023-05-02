"""A PyTorch Dataset class for annotated spectra."""
from typing import Optional, Tuple

import depthcharge
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):
    """
    Parse and retrieve collections of MS/MS spectra.

    Parameters
    ----------
    spectrum_index : depthcharge.data.SpectrumIndex
        The MS/MS spectra to use as a dataset.
    random_state : Optional[int]
        The NumPy random state. ``None`` leaves mass spectra in the order they
        were parsed.
    """

    def __init__(
        self,
        spectrum_index: depthcharge.data.SpectrumIndex,
        random_state: Optional[int] = None,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.rng = np.random.default_rng(random_state)
        self._index = spectrum_index

    def __len__(self) -> int:
        """The number of spectra."""
        return self.n_spectra

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, float, int, Tuple[str, str]]:
        """
        Return the MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the spectrum to return.

        Returns
        -------
        spectrum : torch.Tensor
            A tensor of the spectrum with the m/z and intensity peak values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.
        spectrum_id: Tuple[str, str]
            The unique spectrum identifier, formed by its original peak file and
            identifier (index or scan number) therein.
        """
        mz_array, int_array, precursor_mz, precursor_charge = self.index[idx]
        spectrum = torch.tensor(np.array([mz_array, int_array])).T.float()
        return (
            spectrum,
            precursor_mz,
            precursor_charge,
            self.get_spectrum_id(idx),
        )

    def get_spectrum_id(self, idx: int) -> Tuple[str, str]:
        """
        Return the identifier of the MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the MS/MS spectrum within the SpectrumIndex.

        Returns
        -------
        ms_data_file : str
            The peak file from which the MS/MS spectrum was originally parsed.
        identifier : str
            The MS/MS spectrum identifier, per PSI recommendations.
        """
        with self.index:
            return self.index.get_spectrum_id(idx)

    @property
    def n_spectra(self) -> int:
        """The total number of spectra."""
        return self.index.n_spectra

    @property
    def index(self) -> depthcharge.data.SpectrumIndex:
        """The underlying SpectrumIndex."""
        return self._index

    @property
    def rng(self):
        """The NumPy random number generator."""
        return self._rng

    @rng.setter
    def rng(self, seed):
        """Set the NumPy random number generator."""
        self._rng = np.random.default_rng(seed)


class AnnotatedSpectrumDataset(SpectrumDataset):
    """
    Parse and retrieve collections of annotated MS/MS spectra.

    Parameters
    ----------
    annotated_spectrum_index : depthcharge.data.SpectrumIndex
        The MS/MS spectra to use as a dataset.
    random_state : Optional[int]
        The NumPy random state. ``None`` leaves mass spectra in the order they
        were parsed.
    """

    def __init__(
        self,
        annotated_spectrum_index: depthcharge.data.SpectrumIndex,
        random_state: Optional[int] = None,
    ):
        super().__init__(annotated_spectrum_index, random_state=random_state)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, int, str]:
        """
        Return the annotated MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the spectrum to return.

        Returns
        -------
        spectrum : torch.Tensor
            A tensor of the spectrum with the m/z and intensity peak values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.
        annotation : str
            The peptide annotation of the spectrum.
        """
        (
            mz_array,
            int_array,
            precursor_mz,
            precursor_charge,
            peptide,
        ) = self.index[idx]
        spectrum = torch.tensor(np.array([mz_array, int_array])).T.float()
        return spectrum, precursor_mz, precursor_charge, peptide
