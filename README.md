# casanovo
**_De Novo_ Mass Spectrometry Peptide Sequencing with a Transformer Model**

![image](https://user-images.githubusercontent.com/32707537/152622912-ca87da20-a64c-4e3f-9ca1-721c6b0d9c64.png)

Data and pre-trained model weights are available [here](https://zenodo.org/record/5976003).

A link to the preprint of the paper where we discuss our methods and tests can be found [here](https://www.biorxiv.org/content/10.1101/2022.02.07.479481v1).

# Documentation:
#### https://casanovo.readthedocs.io/en/latest/

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
# Example Job:
## A small walkthrough on how to use casanovo with a very small spectra (~100) set

### The spectra file (.mgf) that we will be running this job on can be seen in the sample_data folder.

- Step 1: Install casanovo (see above for details)
- Step 2: Download the casanovo_pretrained_model_weights.zip from [here](https://zenodo.org/record/5976003). Place these models in a location that you can easly access and know the path of.
    - We will be using pretrained_excl_mouse.ckpt for this job.
- Step 3: Ensure you are in the proper anaconda environment by typing ```conda activate casanovo_env```. (If you named it differently, type in that name instead)
- Step 4: Run this command:
```
casanovo --mode=denovo --model_path='[PATH_TO]/pretrained_excl_mouse.ckpt' --test_data_path='sample_data' --preprocess_spec=False
```
Make sure you have the proper filepath to the pretrained_excl_mouse.ckpt file.
 - Note: If you want to get the ouput csv in a place OTHER than where you ran this command, specify where you would like the output to be placed by specifying a directory in the --output_path CLI field
    - It would look like ```--output_path='path/to/output/location'``` appended onto the end of the above command. Be sure to provide a directory, not a file!

This job should take very little time to run (< 1 minute), and the result should be a file named ```casanovo_output.csv``` wherever you specified.

If the first few lines look like:
```
spectrum_id,denovo_seq,peptide_score,aa_scores
0,LAHYNKR,0.9912219984190804,"[1.0, 1.0, 1.0, 0.99948...
```
Congratulations! You got casanovo to work!

# Common Troubleshooting/FAQ

## Installed casanovo and it worked before, but I reopened Anaconda again and now it says casanovo is not installed
Make sure you are in the `casanovo_env` environment. You can make sure you are in it by typing
```
conda activate casanovo_env
```
## What CLI Prompts can I use?
Run the following command in your command prompt:
```
casanovo --help
```
It should give you a comprehensive list of all CLI options you can tag onto a casanovo job and how/why to use them.

