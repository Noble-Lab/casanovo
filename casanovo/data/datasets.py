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
    max_mz : float, optional
        The maximum m/z to include.
    min_intensity : float, optional
        Remove peaks whose intensity is below `min_intensity` percentage
        of the intensity of the most intense peak
    fragment_tol_mass : float, optional
        Fragment mass tolerance around the precursor mass in Da to remove the
        precursor peak.
    random_state : int or RandomState, optional.
        The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.

    Attributes
    ----------
    n_peaks : int
        The maximum number of mass speak to consider for each mass spectrum.
    min_mz : float
        The minimum m/z to consider for each mass spectrum.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage
        of the intensity of the most intense peak
    fragment_tol_mass : float
        Fragment mass tolerance around the precursor mass in Da to remove the
        precursor peak.
    n_spectra : int
    index : depthcharge.data.SpectrumIndex
    rng : numpy.random.Generator
    """

    def __init__(
        self,
        spectrum_index,
        n_peaks=200,
        min_mz=140,
        max_mz=2500,
        min_intensity=0.01,
        fragment_tol_mass=2,
        random_state=None,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.fragment_tol_mass = fragment_tol_mass
        self.rng = np.random.default_rng(random_state)
        self._index = spectrum_index

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
        spectrum_order_id: str
            The unique identifier for spectrum based on its order in the original mgf file
        """
        mz_array, int_array, prec_mz, prec_charge = self.index[idx]

        spec = self._process_peaks(mz_array, int_array, prec_mz, prec_charge)

        if not spec.sum():
            spec = torch.tensor([[0, 1]]).float()

        spectrum_order_id = f"{idx}"

        return spec, prec_mz, prec_charge, spectrum_order_id

    def _get_non_precursor_peak_mask(
        self,
        mz: np.ndarray,
        pep_mass: float,
        max_charge: int,
        fragment_tol_mass: float,
    ):
        """
        Get a mask to remove peaks that are close to the precursor mass peak (at
        different charges and isotopes).
        ----------
        mz : np.ndarray
            The mass-to-charge ratios of the spectrum fragment peaks.
        pep_mass : float
            The mono-isotopic mass of the uncharged peptide.
        max_charge : int
            The maximum precursor loss charge.
        fragment_tol_mass : float
                Fragment mass tolerance around the precursor mass to remove the
                precursor peak (Da).

        Returns
        -------
        np.ndarray
            Index mask specifying which peaks are retained after precursor peak
            filtering.
        """
        isotope = 0
        remove_mz = []
        for charge in range(max_charge, 0, -1):
            for iso in range(isotope + 1):
                remove_mz.append((pep_mass + iso) / charge + 1.0072766)

        mask = np.full_like(mz, True, np.bool_)
        mz_i = remove_i = 0
        while mz_i < len(mz) and remove_i < len(remove_mz):
            md = mz[mz_i] - remove_mz[remove_i]  # in Da
            if md < -fragment_tol_mass:
                mz_i += 1
            elif md > fragment_tol_mass:
                remove_i += 1
            else:
                mask[mz_i] = False
                mz_i += 1

        return mask

    def _get_filter_intensity_mask(
        self, intensity, min_intensity, max_num_peaks
    ):
        """
        Get a mask to remove low-intensity peaks and retain only the given number
        of most intense peaks.

        Parameters
        ----------
        intensity : np.ndarray
            The intensities of the spectrum fragment peaks.
        min_intensity : float
            Remove peaks whose intensity is below `min_intensity` percentage of the
            intensity of the most intense peak.
        max_num_peaks : int
            Only retain the `max_num_peaks` most intense peaks.
        Returns
        -------
        np.ndarray
            Index mask specifying which peaks are retained after filtering the at
            most `max_num_peaks` most intense intensities above the minimum
            intensity threshold.
        """
        intensity_idx = np.argsort(intensity)
        min_intensity *= intensity[intensity_idx[-1]]
        # Discard low-intensity noise peaks.
        start_i = 0
        for intens in intensity[intensity_idx]:
            if intens > min_intensity:
                break
            start_i += 1
        # Only retain at most the `max_num_peaks` most intense peaks.
        mask = np.full_like(intensity, False, np.bool_)
        mask[
            intensity_idx[max(start_i, len(intensity_idx) - max_num_peaks) :]
        ] = True
        return mask

    def _process_peaks(self, mz_array, int_array, prec_mz, prec_charge):
        """Choose the top n peaks and normalize the spectrum intensities.

        Parameters
        ----------
        mz_array : numpy.ndarray of shape (n_peaks,)
            The m/z values of the peaks in the spectrum.
        int_array : numpy.ndarray of shape (n_peaks,)
            The intensity values of the peaks in the spectrum.
        precursor_mz : float
            The m/z of the precursor.
        precursor_charge : int
            The charge of the precursor.

        Returns
        -------
        torch.Tensor of shape (n_peaks, 2)
            The mass spectrum where ``spectrum[:, 0]`` are the m/z values and
            ``spectrum[:, 1]`` are their associated intensities.
        """
        # Set m/z range for fragments
        if self.min_mz is not None:
            keep = (mz_array >= self.min_mz) & (mz_array <= self.max_mz)
            mz_array = mz_array[keep]
            int_array = int_array[keep]

        # Remove fragment peak(s) close to the precursor m/z
        neutral_mass = (prec_mz - 1.0072766) * prec_charge
        peak_mask = self._get_non_precursor_peak_mask(
            mz_array,
            neutral_mass,
            prec_charge,
            self.fragment_tol_mass,
        )
        mz_array = mz_array[peak_mask]
        int_array = int_array[peak_mask]

        # Remove low-intensity fragment peaks and keep a maximum pre-specified number of peaks
        top_p = self._get_filter_intensity_mask(
            intensity=int_array,
            min_intensity=self.min_intensity,
            max_num_peaks=self.n_peaks,
        )
        mz_array = mz_array[top_p]
        int_array = int_array[top_p]

        # Square root normalize the peak intensities
        int_array = np.sqrt(int_array)
        int_array = int_array / np.linalg.norm(int_array)

        return torch.tensor(np.array([mz_array, int_array])).T.float()

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
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage
        of the intensity of the most intense peak
    fragment_tol_mass : float
        Fragment mass tolerance around the precursor mass in Da to remove the
        precursor peak.
    random_state : int or RandomState, optional.
        The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.

    Attributes
    ----------
    n_peaks : int
        The maximum number of mass speak to consider for each mass spectrum.
    min_mz : float
        The minimum m/z to consider for each mass spectrum.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage
        of the intensity of the most intense peak
    fragment_tol_mass : float
        Fragment mass tolerance around the precursor mass in Da to remove the
        precursor peak.
    n_spectra : int
    index : depthcharge.data.SpectrumIndex
    rng : numpy.random.Generator
    """

    def __init__(
        self,
        annotated_spectrum_index,
        n_peaks=200,
        min_mz=140,
        max_mz=2500,
        min_intensity=0.01,
        fragment_tol_mass=2,
        random_state=None,
    ):
        """Initialize an AnnotatedSpectrumDataset"""
        super().__init__(
            annotated_spectrum_index,
            n_peaks=n_peaks,
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            fragment_tol_mass=fragment_tol_mass,
            random_state=random_state,
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
        spec = self._process_peaks(mz_array, int_array, prec_mz, prec_charge)
        return spec, prec_mz, prec_charge, pep
