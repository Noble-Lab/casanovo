# Getting Started

## Installation

We recommend to run Casanovo in a dedicated [conda environment](https://docs.conda.io/en/latest/).
This helps keep your environment for Casanovo and its dependencies separate from your other Python environments.

```{Note}
Don't know what conda is?
Conda is a package manager for Python packages and many others.
We recommend installing the Anaconda Python distribution which includes conda.
Check out the [Windows](https://docs.anaconda.com/anaconda/install/windows/#), [MacOS](https://docs.anaconda.com/anaconda/install/mac-os/), and [Linux](https://docs.anaconda.com/anaconda/install/linux/) installation instructions.
```

Once you have conda installed, you can use this helpful [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) to see common commands and what they do.

### Create a conda environment

First, open the terminal (MacOS and Linux) or the Anaconda Prompt (Windows).
All of the commands that follow should be entered into this terminal or Anaconda Prompt window---that is, your *shell*.
To create a new conda environment for Casanovo, run the following:

```sh
conda create --name casanovo_env python=3.10
```

This will create an anaconda environment called `casanovo_env` that has Python 3.10 installed.

Activate this environment by running:

```sh
conda activate casanovo_env
```

Your shell should now say **(casanovo_env)** instead of **(base)**.
If this is the case, then you have set up conda and the environment correctly.

```{note}
Be sure to retype in the activation command into your terminal when you reopen anaconda and want to use Casanovo.
```

### *Optional:* Install PyTorch manually

Casanovo employs the PyTorch machine learning framework, which by default will be installed automatically along with the other dependencies.
However, if you have a graphics processing unit (GPU) that you want Casanovo to use, we recommend installing PyTorch manually.
This will ensure that the version of PyTorch used by Casanovo will be compatible with your GPU.
For installation instructions, see the [PyTorch documentation](https://pytorch.org/get-started/locally/#start-locally)

### Install Casanovo

You can now install the Casanovo Python package (dependencies will be installed automatically as needed):

```sh
pip install casanovo
```

After installation, test that it was successful by viewing the Casanovo command line interface help:
```sh
casanovo --help
```
![`casanovo --help`](images/help.svg)


All auxiliary data, model, and training-related parameters can be specified in a YAML configuration file. 
To generate a YAML file containing the current Casanovo defaults, run:
```sh
casanovo configure
```
![`casanovo configure --help`](images/configure-help.svg)

When using Casanovo to sequence peptides from mass spectra or evaluate a previous model's performance, you can change some of the parameters in the first section of this file.
Parameters in the second section will not have an effect unless you are training a new Casanovo model.

### Download model weights

Using Casanovo to sequence peptides from new mass spectra, Casanovo needs compatible pretrained model weights to make its predictions.
By default, Casanovo will try to download the latest compatible model weights from GitHub when it is run. 

However, our model weights are uploaded with new Casanovo versions on the [Releases page](https://github.com/Noble-Lab/casanovo/releases) under the "Assets" for each release (file extension: `.ckpt`).
This model file or a custom one can then be specified using the `--model` command-line parameter when executing Casanovo.

Not all releases will have a model file included on the [Releases page](https://github.com/Noble-Lab/casanovo/releases), in which case model weights for alternative releases with the same major version number can be used.

The most recent model weights for Casanovo version 3.x are currently provided under [Casanovo v3.0.0](https://github.com/Noble-Lab/casanovo/releases/tag/v3.0.0):
- `casanovo_massivekb.ckpt`: Default Casanovo weights to use when analyzing tryptic data. These weights will be downloaded automatically if no weights are explicitly specified.
- `casanovo_non-enzy.checkpt`: Casanovo weights to use when analyzing non-tryptic data, obtained by fine-tuning the tryptic model on multi-enzyme data. These weights need to be downloaded manually.

## Running Casanovo

```{note}
We recommend a Linux system with a dedicated GPU to achieve optimal runtime performance.
```

### Sequence new mass spectra

To sequence your own mass spectra with Casanovo, use the `casanovo sequence` command:

```sh
casanovo sequence -o results.mztab spectra.mgf
```
![`casanovo sequence --help`](images/sequence-help.svg)

Casanovo can predict peptide sequences for MS/MS spectra in mzML, mzXML, and MGF files.
This will write peptide predictions for the given MS/MS spectra to the specified output file in mzTab format.

### Evaluate *de novo* sequencing performance

To evaluate _de novo_ sequencing performance based on known mass spectrum annotations, use the `casanovo evaluate` command:

```sh
casanovo evaluate annotated_spectra.mgf
```
![`casanovo evaluate --help`](images/evaluate-help.svg)


To evaluate the peptide predictions, ground truth peptide labels must to be provided as an annotated MGF file where the peptide sequence is denoted in the `SEQ` field. 
Compatible MGF files are available from [MassIVE-KB](https://massive.ucsd.edu/ProteoSAFe/static/massive-kb-libraries.jsp).

### Train a new model

To train a model from scratch, run:

```sh
casanovo train --validation_peak_path validation_spectra.mgf training_spectra.mgf
```
![`casanovo train --help`](images/train-help.svg)

Training and validation MS/MS data need to be provided as annotated MGF files, where the peptide sequence is denoted in the `SEQ` field.

If a training is continued for a previously trained model, specify the starting model weights using `--model`.

## Try Casanovo on a small example

Let's use Casanovo to sequence peptides from a small collection of mass spectra in an MGF file (~100 MS/MS spectra).
The example MGF file is available at [`sample_data/sample_preprocessed_spectra.mgf`](https://github.com/Noble-Lab/casanovo/blob/main/sample_data/sample_preprocessed_spectra.mgf).

To obtain *de novo* sequencing predictions for these spectra:
1. Download the example MGF above.
2. [Install Casanovo](#installation).
3. Ensure your Casanovo conda environment is activated by typing `conda activate casanovo_env`. (If you named your environment differently, type in that name instead.)
4. Sequence the mass spectra with Casanovo, replacing `[PATH_TO]` with the path to the example MGF file that you downloaded:
```sh
casanovo sequence [PATH_TO]/sample_preprocessed_spectra.mgf
```

```{note}
If you want to store the output mzTab file in a different location than the current working directory, specify an alternative output location using the `--output` parameter.
```

This job should complete in < 1 minute.

Congratulations! Casanovo is installed and running.
