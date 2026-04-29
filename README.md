# Casanovo

**_De Novo_ Mass Spectrometry Peptide Sequencing with a Transformer Model**

<img src="casanovo.svg" alt="Casanovo logo" width="300">

[![PyPI version](https://badge.fury.io/py/casanovo.svg)](https://pypi.org/project/casanovo/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://casanovo.readthedocs.io/en/latest/)

Casanovo is a state-of-the-art deep learning tool designed for _de novo_ peptide sequencing.
Powered by a transformer neural network, Casanovo "translates" peaks in MS/MS spectra into amino acid sequences with remarkable precision.
Casanovo can be used to find unexpected peptide sequences in any data-dependent acquisition, bottom-up tandem mass spectrometry dataset, and is particularly useful for immunopeptidomics, metaproteomics, paleoproteomics, venomics, or any setting in which you are interested in identifying peptides that may not be in your protein database.

## Why choose Casanovo?

- **No database required:** Casanovo sequences peptides directly from spectra without needing a protein database — making it ideal for novel or uncharacterized samples.
- **Unmatched accuracy:** Cutting-edge transformer AI ensures precise and reliable peptide sequencing, even in challenging datasets.
- **Open-source innovation:** Freely available and easy to integrate into existing proteomics workflows.
- **Actively maintained:** Join a growing network of researchers and developers to stay at the forefront of technology.

## Application areas

| Area | Description |
|---|---|
| **Immunopeptidomics** | Identify MHC-presented peptides including those from novel antigens not in reference databases |
| **Metaproteomics** | Sequence peptides from complex microbial communities where complete databases are unavailable |
| **Paleoproteomics** | Recover and identify heavily degraded ancient proteins from archaeological or fossil samples |
| **Venomics** | Characterize toxin peptides from venomous organisms with limited genomic resources |
| **General proteomics** | Any DDA bottom-up MS/MS experiment where unexpected sequences may be present |

## Quick start

```bash
pip install casanovo
```

Download a pre-trained model and run _de novo_ sequencing:

```bash
casanovo sequence --model casanovo_pretrained.ckpt spectra.mgf
```

For training on your own data:

```bash
casanovo train --config config.yaml train.mgf validation.mgf
```

See the [documentation](https://casanovo.readthedocs.io/en/latest/) for full configuration options and workflows.

## [Documentation](https://casanovo.readthedocs.io/en/latest/)

## [Citation information](https://casanovo.readthedocs.io/en/latest/cite.html)
