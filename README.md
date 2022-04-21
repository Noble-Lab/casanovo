# casanovo
**_De Novo_ Mass Spectrometry Peptide Sequencing with a Transformer Model**

![image](https://user-images.githubusercontent.com/32707537/152622912-ca87da20-a64c-4e3f-9ca1-721c6b0d9c64.png)

Data and pre-trained model weights are available [here](https://zenodo.org/record/5976003).

# How to get started with Casanovo?
## Our recommendation:

Install **Anaconda**! It helps keep your environment for casanovo and its dependencies separate from your other Python environments. **This is especially helpful because casanovo works within a specific range of Python versions (3.7 > Python version > 3.10).**

- Check out the [Windows](https://docs.anaconda.com/anaconda/install/windows/#), [MacOS](https://docs.anaconda.com/anaconda/install/mac-os/), and [Linux](https://docs.anaconda.com/anaconda/install/linux/) installation instructions.

Once you have Anaconda installed, you can use this helpful [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) to see common commands and what they do.

## Environment creation:

Open up the anaconda prompt and run this command:
```
conda create --name casanovo_env python=3.7
```
This will create an anaconda environment called `casanovo_env` that has python 3.7 installed (You can check if it was created by typing `conda env list`). 

You can activate this environment by typing:
```
conda activate casanovo_env
```
To the left of your anaconda prompt line it should now say **(casanovo_env)** instead of **(base)**. If this is the case, then you have set up anaconda and the environment properly.

**Be sure to retype in the activation command into your terminal when you reopen anaconda and want to use casanovo.** The base environment most likely will not work.

## Installation:

Install `casanovo` as a Python package from this repo (requires 3.7 > [Python version] > 3.10 , dependencies will be installed automatically as needed):
```
pip install git+https://github.com/Noble-Lab/casanovo.git#egg=casanovo
```

Once installed, Casanovo can be used with a simple command line interface. **Run `casanovo --help` for more details.** All auxiliary data, model and training-related variables can be specified in a user created `.yaml` file, see `casanovo/config.yaml` for the default configuration that was used to obtain the reported results.

# Example Commands:

- To evaluate _de novo_ sequencing performance of a pre-trained model (peptide annotations are needed for spectra):
```
casanovo --mode=eval --model_path='path/to/pretrained' --test_data_path='path/to/test/mgf/files/dir' --config_path='path/to/config'
```

- To run _de novo_ sequencing without evaluation (specificy directory path for output csv file with _de novo_ sequences, see `casanovo_sample_output.csv` for a sample output file):
```
casanovo --mode=denovo --model_path='path/to/pretrained' --test_data_path='path/to/test/mgf/files/dir' --config_path='path/to/config' --output_path='path/to/output'
```

- To train a model from scratch or continue training a pre-trained model:
```
casanovo train --mode=train --model_path='path/to/pretrained' --train_data_path='path/to/train/mgf/files/dir'  --val_data_path='path/to/validation/mgf/files/dir' --config_path='path/to/config'
```
# Common Troubleshooting

## Installed casanovo and it worked before, but I reopened Anaconda again and now it says casanovo is not installed
Make sure you are in the `casanovo_env` environment. You can make sure you are in it by typing
```
conda activate casanovo_env
```

