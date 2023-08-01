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

Fist, open the terminal (MacOS and Linux) or the Anaconda Prompt (Windows).
All of the commands that follow should be entered this terminal or Anaconda Prompt window---that is, your *shell*.
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

All auxiliary data, model, and training-related parameters can be specified in a user created `.yaml` configuration file.
See [`casanovo/config.yaml`](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml) for the default configuration that was used to obtain the reported results. When running Casanovo in eval or denovo mode, you can change some of the parameters in this file, indicated with "(I)" in the file. You should not change other parameters unless you are training a new Casanovo model.


### Download model weights

When running Casanovo in `denovo` or `eval` mode, Casanovo needs compatible pretrained model weights to make predictions.
Our model weights are uploaded with new Casanovo versions on the [Releases page](https://github.com/Noble-Lab/casanovo/releases) under the "Assets" for each release (file extension: .ckpt).
The model file can then be specified using the `--model` command-line parameter when executing Casanovo.
To assist users, if no model file is specified Casanovo will try to download and use a compatible model file automatically.

Not all releases might have a model file included on the [Releases page](https://github.com/Noble-Lab/casanovo/releases), in which case model weights for alternative releases with the same major version number can be used.

The most recent model weights for Casanovo version 3.x are currently provided under [Casanovo v3.0.0](https://github.com/Noble-Lab/casanovo/releases/tag/v3.0.0):
- `casanovo_massivekb.ckpt`: Default Casanovo weights to use when analyzing tryptic data. These weights will be downloaded automatically if no weights are explicitly specified.
- `casanovo_non-enzy.checkpt`: Casanovo weights to use when analyzing non-tryptic data, obtained by fine-tuning the tryptic model on multi-enzyme data. These weights need to be downloaded manually.

## Running Casanovo

```{note}
We recommend a Linux system with a dedicated GPU to achieve optimal runtime performance.
Notably, Casanovo is restricted to single-threaded execution only on Windows and MacOS.
```

> **Warning**
> Casanovo can currently crash if no GPU is available.
> We are actively trying to fix this known issue.

### Sequence new mass spectra

To sequence your own mass spectra with Casanovo, use the `denovo` mode:

```sh
casanovo --mode=denovo --peak_path=path/to/predict/spectra.mgf --output=path/to/output
```

Casanovo can predict peptide sequences for MS/MS spectra in mzML, mzXML, and MGF files.
This will write peptide predictions for the given MS/MS spectra to the specified output file in mzTab format.

> **Warning**
> If you are running inference with Casanovo on a system that has multiple GPUs, it is necessary to restrict Casanovo to (maximum) a single GPU.
> For example, for CUDA-capable GPUs, GPU visibility can be controlled by setting the `CUDA_VISIBLE_DEVICES` shell variable.

### Evaluate *de novo* sequencing performance

To evaluate _de novo_ sequencing performance based on known mass spectrum annotations, run:

```sh
casanovo --mode=eval --peak_path=path/to/test/annotated_spectra.mgf
```

To evaluate the peptide predictions, ground truth peptide labels must to be provided as an annotated MGF file where the peptide sequence is denoted in the `SEQ` field.

### Train a new model

To train a model from scratch, run:

```sh
casanovo --mode=train --peak_path=path/to/train/annotated_spectra.mgf --peak_path_val=path/to/validation/annotated_spectra.mgf
```

Training and validation MS/MS data need to be provided as annotated MGF files, where the peptide sequence is denoted in the `SEQ` field.

If a training is continued for a previously trained model, specify the starting model weights using `--model`.

## Try Casanovo on a small example

Here, we demonstrate how to use Casanovo using a small collection of mass spectra in an MGF file (~100 MS/MS spectra).
The example MGF file is available at [`sample_data/sample_preprocessed_spectra.mgf`](https://github.com/Noble-Lab/casanovo/blob/main/sample_data/sample_preprocessed_spectra.mgf).

To obtain *de novo* sequencing predictions for these spectra:
1. Download the example MGF above.
2. [Install Casanovo](#installation).
3. Ensure your Casanovo conda environment is activated by typing `conda activate casanovo_env`. (If you named your environment differently, type in that name instead.)
4. Sequence the mass spectra with Casanovo, replacing `[PATH_TO]` with the path to the example MGF file that you downloaded:
```sh
casanovo --mode=denovo --peak_path=[PATH_TO]/sample_preprocessed_spectra.mgf
```

```{note}
If you want to store the output mzTab file in a different location than the current working directory, specify an alternative output location using the `--output` parameter.
```

This job should complete in < 1 minute.

Congratulations! Casanovo is installed and running.
