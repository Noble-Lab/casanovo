"""A PyTorch Dataset class for annotated spectra."""
import math

import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset


class SpectrumDataset(Dataset):
    """
    Parse and retrieve collections of mass spectra.

    :param spectrum_index: The collection of spectra to use as a dataset.
    :type spectrum_index: depthcharge.data.SpectrumIndex
    :param n_peaks: Keep only the top-n most intense peaks in any spectrum. ``None``
        retains all of the peaks.
    :type n_peaks: int, optional
    :param min_mz: The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    :type min_mz: float, optional
    :param max_mz: The maximum m/z to include. 
    :type max_mz: float, optional
    :param min_intensity: Remove peaks whose intensity is below `min_intensity` percentage
        of the intensity of the most intense peak
    :type min_intensity: float, optional
    :param fragment_tol_mass: Fragment mass tolerance around the precursor mass in Da to remove the
        precursor peak.   
    :type fragment_tol_mass: float, optional
    :param random_state: The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.
    :type random_state: int or RandomState, optional.
    :param preprocess_spec: Preprocess the provided spectra
    :type preprocess_spec: bool, optional
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
        preprocess_spec=False
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
        self.preprocess_spec = preprocess_spec

    def __len__(self):
        """The number of spectra."""
        return self.n_spectra

    def __getitem__(self, idx):
        """
        Return a single mass spectrum.
        
        :param idx: The index to return.
        :type idx: int
        :return: *spectrum* - The mass spectrum where ``spectrum[:, 0]`` are the m/z values and
            ``spectrum[:, 1]`` are their associated intensities.
        :rtype: torch.Tensor of shape (n_peaks, 2)
        :return: *precursor_mz* - The m/z of the precursor.
        :rtype: float
        :return: *precursor_charge* - The charge of the precursor.
        :rtype: int
        :return: *spectrum_order_id* - The unique identifier for spectrum based on its order in the original mgf file
        :rtype: str
        """
        mz_array, int_array, prec_mz, prec_charge = self.index[idx]
 
        if self.preprocess_spec == True:
            spec = self._process_peaks(mz_array, int_array, prec_mz, prec_charge)
        else:
            spec = torch.tensor(np.array([mz_array, int_array])).T.float()
            
        if not spec.sum():
            spec = torch.tensor([[0, 1]]).float()            
            
        spectrum_order_id = f'{idx}'
        
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

        :param mz: The mass-to-charge ratios of the spectrum fragment peaks.
        :type mz: np.ndarray
        :param pep_mass: The mono-isotopic mass of the uncharged peptide. 
        :type pep_mass: float
        :param max_charge: The maximum precursor loss charge.
        :type max_charge: int
        :param fragment_tol_mass: Fragment mass tolerance around the precursor mass to remove the
                precursor peak (Da).
        :type: fragment_tol_mass: float
        :return: *mask* - Index mask specifying which peaks are retained after precursor peak
            filtering.
        :rtype: np.ndarray
        """
        isotope = 0
        remove_mz = []
        for charge in range(max_charge, 0, -1):
            for iso in range(isotope + 1):
                remove_mz.append((pep_mass + iso) / charge + 1.0072766)

        mask = np.full_like(mz, True, np.bool_)
        mz_i = remove_i = 0
        while mz_i < len(mz) and remove_i < len(remove_mz):
            md = mz[mz_i] - remove_mz[remove_i] # in Da
            if md < -fragment_tol_mass:
                mz_i += 1
            elif md > fragment_tol_mass:
                remove_i += 1
            else:
                mask[mz_i] = False
                mz_i += 1

        return mask    
    
    def _get_filter_intensity_mask(self, intensity, min_intensity, max_num_peaks):
        """
        Get a mask to remove low-intensity peaks and retain only the given number
        of most intense peaks.

        :param intensity: The intensities of the spectrum fragment peaks.
        :type intensity: np.ndarray
        :param min_intensity: Remove peaks whose intensity is below `min_intensity` percentage of the
            intensity of the most intense peak.
        :type min_intensity: float
        :param max_num_peaks: Only retain the `max_num_peaks` most intense peaks.
        :type max_num_peaks: int
        :return: *mask* -  Index mask specifying which peaks are retained after filtering the at
            most `max_num_peaks` most intense intensities above the minimum
            intensity threshold.
        :rtype: np.ndarray
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

        :param mz_array: The m/z values of the peaks in the spectrum.
        :type mz_array: numpy.ndarray of shape (n_peaks,)
        :param int_array: The intensity values of the peaks in the spectrum.
        :type: int_array: numpy.ndarray of shape (n_peaks,)
        :param prec_mz: The m/z of the precursor.
        :type prec_mz: float
        :param prec_charge: The charge of the precursor.
        :type prec_charge: int
        :return: *ms* - The mass spectrum where ``spectrum[:, 0]`` are the m/z values and
            ``spectrum[:, 1]`` are their associated intensities.
        :rtype: torch.Tensor of shape (n_peaks, 2)
        """
        #Set m/z range for fragments
        if self.min_mz is not None:
            keep = (mz_array >= self.min_mz) & (mz_array <= self.max_mz)
            mz_array = mz_array[keep]
            int_array = int_array[keep]
        
        #Remove fragment peak(s) close to the precursor m/z
        neutral_mass = (prec_mz - 1.0072766) * prec_charge
        peak_mask = self._get_non_precursor_peak_mask(
            mz_array,
            neutral_mass,
            prec_charge,
            self.fragment_tol_mass,
        )
        mz_array = mz_array[peak_mask]
        int_array = int_array[peak_mask]
        
        #Remove low-intensity fragment peaks and keep a maximum pre-specified number of peaks
        top_p = self._get_filter_intensity_mask(intensity=int_array, min_intensity=self.min_intensity, max_num_peaks=self.n_peaks)
        mz_array = mz_array[top_p]
        int_array = int_array[top_p]
        
        #Square root normalize the peak intensities
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
    """
    Parse and retrieve collections of mass spectra.

    :param annotated_spectrum_index: The collection of annotated mass spectra to use as a dataset.
    :type annotated_spectrum_index: depthcharge.data.SpectrumIndex
    :param n_peaks: Keep only the top-n most intense peaks in any spectrum. ``None``
        retains all of the peaks.
    :type n_peaks: int, optional
    :param min_mz: The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    :type min_mz: float, optional
    :param max_mz: The maximum m/z to include. 
    :type max_mz: float, optional
    :param min_intensity: Remove peaks whose intensity is below `min_intensity` percentage
        of the intensity of the most intense peak
    :type min_intensity: float, optional
    :param fragment_tol_mass: Fragment mass tolerance around the precursor mass in Da to remove the
        precursor peak.   
    :type fragment_tol_mass: float, optional
    :param random_state: The numpy random state. ``None`` leaves mass spectra in the order
        they were parsed.
    :type random_state: int or RandomState, optional.
    :param preprocess_spec: Preprocess the provided spectra
    :type preprocess_spec: bool, optional
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
        preprocess_spec=False
    ):
        """
        Initialize an AnnotatedSpectrumDataset
        """
        super().__init__(
            annotated_spectrum_index,
            n_peaks=n_peaks,
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            fragment_tol_mass=fragment_tol_mass,            
            random_state=random_state,
            preprocess_spec=preprocess_spec
        )

    def __getitem__(self, idx):
        """
        Return a single annotated mass spectrum.

        :param idx:
        :type idx:

        :return: *spectrum* - The mass spectrum where ``spectrum[:, 0]`` are the m/z values and
            ``spectrum[:, 1]`` are their associated intensities.
        :rtype: torch.Tensor of shape (n_peaks, 2)
        :return: *precursor_mz* - The m/z of the precursor.
        :rtype: float
        :return: *precursor_charge* - The charge of the precursor.
        :rtype: int
        :return: *annotation* - The annotation for the mass spectrum.
        :rtype: str
        """
        mz_array, int_array, prec_mz, prec_charge, pep = self.index[idx]
        if self.preprocess_spec == True:
            spec = self._process_peaks(mz_array, int_array, prec_mz, prec_charge)
        else:
            spec = torch.tensor(np.array([mz_array, int_array])).T.float()
        return spec, prec_mz, prec_charge, pep
