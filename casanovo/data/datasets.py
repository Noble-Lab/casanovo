"""A PyTorch Dataset class for annotated spectra."""
import math

import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset


class SpectrumDataset(Dataset):
    """Parse and retrieve collections of mass spectra.

    Parameters
    ----------
    spectrum_index : depthcharge.data.SpectrumIndex
        The collection of spectra to use as a dataset.
    n_peaks : int, optional
        Keep only the top-n most intense peaks in any spectrum. ``None``
        retains all of the peaks.
    min_mz : float, optional
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    random_state : int or RandomState, optional.
        The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.
    preprocess_spec : bool, optional
        Preprocess the provided spectra
        
    Attributes
    ----------
    n_peaks : int
        The maximum number of mass speak to consider for each mass spectrum.
    min_mz : float
        The minimum m/z to consider for each mass spectrum.
    n_spectra : int
    index : depthcharge.data.SpectrumIndex
    rng : numpy.random.Generator
    preprocess_spec : bool
    """

    def __init__(
        self,
        spectrum_index,
        n_peaks=200,
        min_mz=140,
        random_state=None,
        preprocess_spec=False
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.rng = np.random.default_rng(random_state)
        self._index = spectrum_index
        self.preprocess_spec = preprocess_spec

    def __len__(self):
        """The number of spectra."""
        return self.n_spectra

    def __getitem__(self, idx):
        """Return a single mass spectrum.

        Parameters
        ----------
        idx : int
            The index to return.

        Returns
        -------
        spectrum : torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where ``spectrum[:, 0]`` are the m/z values and
            ``spectrum[:, 1]`` are their associated intensities.
        precursor_mz : float
            The m/z of the precursor.
        precursor_charge : int
            The charge of the precursor.
        """
        mz_array, int_array, prec_mz, prec_charge = self.index[idx]
        spec = self._process_peaks(mz_array, int_array)
        if not spec.sum():
            spec = torch.tensor([[0, 1]]).float()

        return spec, prec_mz, prec_charge

    def _process_peaks(self, mz_array, int_array):
        """Choose the top n peaks and normalize the spectrum intensities.

        Parameters
        ----------
        mz_array : numpy.ndarray of shape (n_peaks,)
            The m/z values of the peaks in the spectrum.
        int_array : numpy.ndarray of shape (n_peaks,)
            The intensity values of the peaks in the spectrum.

        Returns
        -------
        torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where ``spectrum[:, 0]`` are the m/z values and
            ``spectrum[:, 1]`` are their associated intensities.
        """
        if self.min_mz is not None:
            keep = mz_array >= self.min_mz
            mz_array = mz_array[keep]
            int_array = int_array[keep]

        if len(int_array) > self.n_peaks:
            top_p = np.argpartition(int_array, -self.n_peaks)[-self.n_peaks :]
            top_p = np.sort(top_p)
            mz_array = mz_array[top_p]
            int_array = int_array[top_p]

        int_array = np.sqrt(int_array)
        int_array = int_array / np.linalg.norm(int_array)
        return torch.tensor([mz_array, int_array]).T.float()

    @property
    def n_spectra(self):
        """The total number of spectra."""
        return self.index.n_spectra

    @property
    def index(self):
        """The underyling SpectrumIndex."""
        return self._index

    @property
    def rng(self):
        """The numpy random number generator."""
        return self._rng

    @rng.setter
    def rng(self, seed):
        """Set the numpy random number generator."""
        self._rng = np.random.default_rng(seed)

class AnnotatedSpectrumDataset(SpectrumDataset):
    """Parse and retrieve collections of mass spectra

    Parameters
    ----------
    annotated_spectrum_index : depthcharge.data.SpectrumIndex
        The collection of annotated mass spectra to use as a dataset.
    n_peaks : int, optional
        Keep only the top-n most intense peaks in any spectrum. ``None``
        retains all of the peaks.
    min_mz : float, optional
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    random_state : int or RandomState, optional.
        The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.
    preprocess_spec : bool, optional
        Preprocess the provided spectra

    Attributes
    ----------
    n_peaks : int
        The maximum number of mass speak to consider for each mass spectrum.
    min_mz : float
        The minimum m/z to consider for each mass spectrum.
    n_spectra : int
    index : depthcharge.data.SpectrumIndex
    rng : numpy.random.Generator
    preprocess_spec : bool
    """

    def __init__(
        self,
        annotated_spectrum_index,
        n_peaks=200,
        min_mz=140,
        random_state=None,
        preprocess_spec=False
    ):
        """Initialize an AnnotatedSpectrumDataset"""
        super().__init__(
            annotated_spectrum_index,
            n_peaks=n_peaks,
            min_mz=min_mz,
            random_state=random_state,
            preprocess_spec=preprocess_spec
        )

    def __getitem__(self, idx):
        """Return a single annotated mass spectrum.

        Parameters
        ----------
        idx : int
            The index to return.

        Returns
        -------
        spectrum : torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where ``spectrum[:, 0]`` are the m/z values and
            ``spectrum[:, 1]`` are their associated intensities.
        precursor_mz : float
            The m/z of the precursor.
        precursor_charge : int
            The charge of the precursor.
        annotation : str
            The annotation for the mass spectrum.
        """
        mz_array, int_array, prec_mz, prec_charge, pep = self.index[idx]
        if self.preprocess_spec == True:
            spec = self._process_peaks(mz_array, int_array)
        else:
            spec = torch.tensor([mz_array, int_array]).T.float()
        return spec, prec_mz, prec_charge, pep
