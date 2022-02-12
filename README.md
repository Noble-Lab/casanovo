# casanovo
**_De Novo_ Mass Spectrometry Peptide Sequencing with a Transformer Model**

![image](https://user-images.githubusercontent.com/32707537/152622912-ca87da20-a64c-4e3f-9ca1-721c6b0d9c64.png)

Data and pre-trained model weights are available [here](https://zenodo.org/record/5976003).

## How to get started with Casanovo?

Install `casanovo` as a Python package from this repo (requires Python 3.7+, dependencies will be installed automatically as needed):
```
pip install git+https://github.com/Noble-Lab/casanovo.git#egg=casanovo
```

Once installed, Casanovo can be used with a simple command line interface. Run `casanovo --help` for more details. All auxiliary data, model and training-related variables can be specified in a user created `.py` file, see `casanovo/config.py` for the default configuration that was used to obtain the reported results.

- To evaluate _de novo_ sequencing performance of a pre-trained model (peptide annotations are needed for spectra):
```
casanovo --mode=eval --model_path='path/to/pretrained' --test_data_path='path/to/test' --config_path='path/to/config'
```

- To run _de novo_ sequencing without evaluation (specificy directory path for output csv file with _de novo_ sequences):
```
casanovo --mode=denovo --model_path='path/to/pretrained' --test_data_path='path/to/test' --config_path='path/to/config' --output_path='path/to/output'
```

- To train a model from scratch or continue training a pre-trained model:
```
casanovo train --mode=train --model_path='path/to/pretrained' --train_data_path='path/to/train'  --val_data_path='path/to/validation' --config_path='path/to/config'
```


