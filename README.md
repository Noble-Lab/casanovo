# Casanovo

**_De Novo_ Mass Spectrometry Peptide Sequencing with a Transformer Model**

![image](https://user-images.githubusercontent.com/32707537/152622912-ca87da20-a64c-4e3f-9ca1-721c6b0d9c64.png)

If you use Casanovo in your work, please cite the following publication:

- Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. *De novo* mass spectrometry peptide sequencing with a transformer model. in *Proceedings of the 39th International Conference on Machine Learning - ICML '22* vol. 162 25514–25522 (PMLR, 2022). [https://proceedings.mlr.press/v162/yilmaz22a.html](https://proceedings.mlr.press/v162/yilmaz22a.html)

Data and pre-trained model weights are available [on Zenodo](https://zenodo.org/record/6791263).

## Documentation

#### https://casanovo.readthedocs.io/en/latest/

## Getting started with Casanovo

We recommend to run Casanovo in a dedicated **Anaconda** environment.
This helps keep your environment for Casanovo and its dependencies separate from your other Python environments.
**This is especially helpful because Casanovo works within a specific range of Python versions (3.7 > Python version > 3.10).**

- Check out the [Windows](https://docs.anaconda.com/anaconda/install/windows/#), [MacOS](https://docs.anaconda.com/anaconda/install/mac-os/), and [Linux](https://docs.anaconda.com/anaconda/install/linux/) installation instructions.

Once you have Anaconda installed, you can use this helpful [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) to see common commands and what they do.

### Environment creation

Open up the anaconda prompt and run this command:

```
conda create --name casanovo_env python=3.7
```

This will create an anaconda environment called `casanovo_env` that has Python 3.7 installed.
(You can check if it was created by typing `conda env list`.)

You can activate this environment by typing:

```
conda activate casanovo_env
```

To the left of your anaconda prompt line it should now say **(casanovo_env)** instead of **(base)**.
If this is the case, then you have set up anaconda and the environment properly.

**Be sure to retype in the activation command into your terminal when you reopen anaconda and want to use Casanovo.**
The base environment most likely will not work.

### Installation

Install Casanovo as a Python package from this repository (requires 3.7 > [Python version] > 3.10 , dependencies will be installed automatically as needed):

```
pip install git+https://github.com/Noble-Lab/casanovo.git#egg=casanovo
```

Once installed, Casanovo can be used with a simple command line interface.
**Run `casanovo --help` for more details.**
All auxiliary data, model, and training-related parameters can be specified in a user created `.yaml` configuration file.
See [`casanovo/config.yaml`](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml) for the default configuration that was used to obtain the reported results.

### Example commands

- To run _de novo_ sequencing:

```
casanovo --mode=denovo --model='path/to/pretrained.ckpt' --peak_dir='path/to/predict/mgf/files/dir' --config='path/to/config.yaml' --output='path/to/output'
```

This will write peptide predictions for the given MS/MS spectra to the specified output file in a tab-separated format (extension: .csv).

- To evaluate _de novo_ sequencing performance based on known spectrum annotations:

```
casanovo --mode=eval --model='path/to/pretrained.ckpt' --peak_dir='path/to/test/predict/files/dir' --config='path/to/config.yaml'
```

Note that to evaluate the peptide predictions, ground truth peptide labels in an annotated MGF file need to be present.

- To train a model from scratch:

```
casanovo --mode=train --peak_dir='path/to/train/mgf/files/dir' --peak_dir_val='path/to/validation/mgf/files/dir' --config='path/to/config.yaml'
```

If a training is continued for a previously trained model, specify the starting model weights using `--model`.

### Example job

We will demonstrate how to use Casanovo using a small walkthrough example on a small MGF file (~100 MS/MS spectra).
The example MGF file is available at [`sample_data/sample_preprocessed_spectra.mgf`](https://github.com/Noble-Lab/casanovo/blob/main/sample_data/sample_preprocessed_spectra.mgf`).

1. Install Casanovo (see above for details).
2. Download the `casanovo_pretrained_model_weights.zip` from [Zenodo](https://zenodo.org/record/6791263). Place these models in a location that you can easily access and know the path of.
    - We will be `using pretrained_excl_mouse.ckpt` for this job.
3. Copy the example `config.yaml` file into a location you can easily access. 
4. Ensure you are in the proper anaconda environment by typing `conda activate casanovo_env`. (If you named your environment differently, type in that name instead.)
5. Run this command:
```
casanovo --mode=denovo --model='[PATH_TO]/pretrained_excl_mouse.ckpt' --peak_dir='sample_data' --config='path/to/config.yaml'
```
Make sure you use the proper filepath to the `pretrained_excl_mouse.ckpt` file.
    - Note: If you want to get the ouput CSV file in different location than the working directory, specify an alternative output location using the `--output` parameter.

This job will take very little time to run (< 1 minute).

If the first few lines look like:

```
spectrum_id     sequence            score       aa_scores
0               LAHYNKR             0.98197     1.0,0.999...
1               VKEDYGQM+15.995PR   0.77206     0.999,0.999...
```

Congratulations! You got Casanovo to work.

## Common Troubleshooting / FAQ

**I installed Casanovo and it worked before, but I after reopening Anaconda it says that Casanovo is not installed.**

Make sure you are in the `casanovo_env` environment. You can ensure this by typing:

```
conda activate casanovo_env
```

**Which command-line options are available?**

Run the following command in your command prompt to see all possible command-line configuration options:
```
casanovo --help
```

**I get a "CUDA out of memory" error when trying to run Casanovo. Help!**

This means that there was not enough (free) memory available on your GPU to run Casanovo, which is especially likely to happen when you are using a smaller, consumer-grade GPU.
We recommend trying to decrease the `train_batch_size` or `predict_batch_size` options in the [config file](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml) (depending on whether the error occurred during `train` or `denovo` mode) to reduce the number of spectra that are processed simultaneously.
Additionally, we recommend shutting down any other processes that may be running on the GPU, so that Casanovo can exclusively use the GPU.

## Release notes

- Release 2.1.1 (2022-07-27): Update tutorial
- Release 2.1.0 (2022-07-02): Import latest depthcharge version with stable memory usage and fix to positional encoding for amino acids
- Release 2.0.1 (2022-06-13): Release notes added
- Release 2.0.0 (2022-06-05): Added additional CLI functionality, changed config file format, added pytest functionality, tutorial added, documentation with sphinx/ReadTheDocs added
- Release 1.2.0 (2022-03-07): Add peptide and amino acid confidence scores to output file
- Release 1.1.2 (2022-02-20): Add support for multiple MGF files in a directory
- Release 1.1.1 (2022-02-10): Add more CLI options and specify custom config file
- Release 1.1.0 (2022-02-04): Add data infrastructure, model and training/testing functionality
- Release 1.0.0 (2022-01-28): Initial commit

