# casanovo
**_De Novo_ Mass Spectrometry Peptide Sequencing with a Transformer Model**

![image](https://user-images.githubusercontent.com/32707537/152622912-ca87da20-a64c-4e3f-9ca1-721c6b0d9c64.png)

Data and pre-trained model weights are available [here](https://zenodo.org/record/5976003).

## How to get started with Casanovo?

Install `casanovo` as a Python package from this repo (dependencies will be installed automatically as needed):
```
pip install git+https://github.com/Noble-Lab/casanovo.git#egg=casanovo
```

Once installed, Casanovo can be used with a simple command line interface. All data, model and training-related options can be specified in `casanovo/config.py`, default configuration was used to obtain the reported results.
- To run _de novo_ sequencing with a pre-trained model:
```
casanovo test
```

- To train a model from scratch or continue training a pre-trained model:
```
casanovo train
```


