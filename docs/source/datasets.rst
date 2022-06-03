datasets.py
===========

SpectrumDataset
-------------------------------

.. autoclass:: casanovo.data.datasets.SpectrumDataset

Attributes:

- n_peaks - (int) The maximum number of mass speak to consider for each mass spectrum.
- min_mz - (float) The minimum m/z to consider for each mass spectrum.
- max_mz - (float) The maximum m/z to include.
- min_intensity - (float) Remove peaks whose intensity is below `min_intensity` percentage of the intensity of the most intense peak
- fragment_tol_mass - (float) Fragment mass tolerance around the precursor mass in Da to remove the precursor peak.      
- n_spectra - (int)
- index - (depthcharge.data.SpectrumIndex)
- rng - (numpy.random.Generator)
- preprocess_spec - (bool)


SpectrumDataset.\__getitem\__
-------------------------------

.. autofunction:: casanovo.data.datasets.SpectrumDataset.__getitem__()

SpectrumDataset._get_non_precursor_peak_mask
----------------------------------------------

.. autofunction:: casanovo.data.datasets.SpectrumDataset._get_non_precursor_peak_mask()

SpectrumDataset._get_filter_intensity_mask
-------------------------------------------

.. autofunction:: casanovo.data.datasets.SpectrumDataset._get_filter_intensity_mask()

SpectrumDataset._process_peaks
-------------------------------

.. autofunction:: casanovo.data.datasets.SpectrumDataset._process_peaks()

AnnotatedSpectrumDataset
----------------------------------------

.. autoclass:: casanovo.data.datasets.AnnotatedSpectrumDataset

Attributes:

- n_peaks - (int) The maximum number of mass speak to consider for each mass spectrum.
- min_mz - (float) The minimum m/z to consider for each mass spectrum.
- max_mz - (float) The maximum m/z to include.
- min_intensity - (float) Remove peaks whose intensity is below `min_intensity` percentage of the intensity of the most intense peak
- fragment_tol_mass - (float) Fragment mass tolerance around the precursor mass in Da to remove the precursor peak.      
- n_spectra - (int)
- index - (depthcharge.data.SpectrumIndex)
- rng - (numpy.random.Generator)
- preprocess_spec - (bool)

AnnotatedSpectrumDataset.\__getitem\__
----------------------------------------

.. autofunction:: casanovo.data.datasets.AnnotatedSpectrumDataset.__getitem__()