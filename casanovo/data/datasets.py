"""A PyTorch Dataset class for annotated spectra."""
from typing import Optional, Tuple

import depthcharge
import numpy as np
import spectrum_utils.spectrum as sus
import torch
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):
    """
    Parse and retrieve collections of MS/MS spectra.

    Parameters
    ----------
    spectrum_index : depthcharge.data.SpectrumIndex
        The MS/MS spectra to use as a dataset.
    n_peaks : Optional[int]
        The number of top-n most intense peaks to keep in each spectrum. `None` retains
        all peaks.
    min_mz : float
        The minimum m/z to include. The default is 140 m/z, in order to exclude TMT and
        iTRAQ reporter ions.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage of the base
        peak intensity.
    remove_precursor_tol : float
        Remove peaks within the given mass tolerance in Dalton around the precursor
        mass.
    random_state : Optional[int]
        The NumPy random state. ``None`` leaves mass spectra in the order they were
        parsed.
    """

    def __init__(
        self,
        spectrum_index: depthcharge.data.SpectrumIndex,
        n_peaks: int = 150,
        min_mz: float = 140.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        random_state: Optional[int] = None,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.rng = np.random.default_rng(random_state)
        self._index = spectrum_index

    def __len__(self) -> int:
        """The number of spectra."""
        return self.n_spectra

    def __getitem__(self, idx) -> Tuple[torch.Tensor, float, int, str]:
        """
        Return the MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the spectrum to return.

        Returns
        -------
        spectrum : torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.
        spectrum_id: str
            The unique spectrum identifier, as determined by its index in the original
            peak file.
        """
        mz_array, int_array, precursor_mz, precursor_charge = self.index[idx]
        spectrum = self._process_peaks(
            mz_array, int_array, precursor_mz, precursor_charge
        )
        # Replace invalid spectra by a dummy spectrum.
        if not spectrum.sum():
            spectrum = torch.tensor([[0, 1]]).float()
        return spectrum, precursor_mz, precursor_charge, str(idx)

    def _get_non_precursor_peak_mask(
        self,
        mz: np.ndarray,
        pep_mass: float,
        max_charge: int,
        remove_precursor_tol: float,
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
        remove_precursor_tol : float
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
            if md < -remove_precursor_tol:
                mz_i += 1
            elif md > remove_precursor_tol:
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
            self.remove_precursor_tol,
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
    def n_spectra(self) -> int:
        """The total number of spectra."""
        return self.index.n_spectra

    @property
    def index(self) -> depthcharge.data.SpectrumIndex:
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
    """
    Parse and retrieve collections of annotated MS/MS spectra.

    Parameters
    ----------
    annotated_spectrum_index : depthcharge.data.SpectrumIndex
        The MS/MS spectra to use as a dataset.
    n_peaks : Optional[int]
        The number of top-n most intense peaks to keep in each spectrum. `None` retains
        all peaks.
    min_mz : float
        The minimum m/z to include. The default is 140 m/z, in order to exclude TMT and
        iTRAQ reporter ions.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage of the base
        peak intensity.
    remove_precursor_tol : float
        Remove peaks within the given mass tolerance in Dalton around the precursor
        mass.
    random_state : Optional[int]
        The NumPy random state. ``None`` leaves mass spectra in the order they were
        parsed.
    """

    def __init__(
        self,
        annotated_spectrum_index: depthcharge.data.SpectrumIndex,
        n_peaks: int = 150,
        min_mz: float = 140.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            annotated_spectrum_index,
            n_peaks=n_peaks,
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            remove_precursor_tol=remove_precursor_tol,
            random_state=random_state,
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, int, str]:
        """
        Return the annotated MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the spectrum to return.

        Returns
        -------
        spectrum : torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.
        annotation : str
            The peptide annotation of the spectrum.
        """
        mz_array, int_array, precursor_mz, precursor_charge, peptide = self.index[idx]
        spectrum = self._process_peaks(
            mz_array, int_array, precursor_mz, precursor_charge
        )
        return spectrum, precursor_mz, precursor_charge, peptide
