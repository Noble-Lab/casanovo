# Frequently Asked Questions

**I installed Casanovo and it worked before, but I after reopening Anaconda it says that Casanovo is not installed.**

Make sure you are in the `casanovo_env` environment. You can ensure this by typing:

```sh
conda activate casanovo_env
```

**Which command-line options are available?**

Run the following command in your command prompt to see all possible command-line configuration options:

```sh
casanovo --help
```

Additionally, you can use a configuration file to fully customize Casanovo.
You can find the `config.yaml` configuration file that is used by default [here](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml).

**How do I solve a "PermissionError: GitHub API rate limit exceeded" error when trying to run Casanovo?**

When running Casanovo in `denovo` or `eval` mode, Casanovo needs compatible pretrained model weights to make predictions.
If no model weights file is specified using the `--model` command-line parameter, Casanovo will automatically try to download the latest compatible model file from GitHub and save it to its cache for subsequent use.
However, the GitHub API is limited to maximum 60 requests per hour per IP address.
Consequently, if Casanovo has been executed multiple times already, it might temporarily not be able to communicate with GitHub.
You can avoid this error by explicitly specifying the model file using the `--model` parameter.

**I get a "CUDA out of memory" error when trying to run Casanovo. Help!**

This means that there was not enough (free) memory available on your GPU to run Casanovo, which is especially likely to happen when you are using a smaller, consumer-grade GPU.
We recommend trying to decrease the `train_batch_size` or `predict_batch_size` options in the [config file](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml) (depending on whether the error occurred during `train` or `denovo` mode) to reduce the number of spectra that are processed simultaneously.
Additionally, we recommend shutting down any other processes that may be running on the GPU, so that Casanovo can exclusively use the GPU.

**I see "NotImplementedError: The operator 'aten::index.Tensor'..." when using a Mac with an Apple Silicon chip.**

Casanovo can leverage Apple's Metal Performance Shaders (MPS) on newer Mac computers, which requires that the `PYTORCH_ENABLE_MPS_FALLBACK` is set to `1`:

```sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

This will need to be set with each new shell session, or you can add it to your `.bashrc` / `.zshrc` to set this environment variable by default.

**Where can I find the data that Casanovo was trained on?**

The [Casanovo results reported ](https://doi.org/10.1101/2023.01.03.522621) were obtained by training on two different datasets: (i) a commonly used nine-species benchmark dataset, and (ii) a large-scale training dataset derived from the MassIVE Knowledge Base (MassIVE-KB).

All data for the _nine-species benchmark_ is available as annotated MGF files [on MassIVE](https://doi.org/doi:10.25345/C52V2CK8J).
Using these data, Casanovo was trained in a cross-validated fashion, training on eight species and testing on the remaining species.

The _MassIVE-KB training data_ was derived from PSMs used to compile the MassIVE-KB v1 spectral library and consists of 30 million PSMs.
These PSMs were obtained by collecting up to the top 100 PSMs for each of the precursors (as defined by a peptidoform and charge) included in MassIVE-KB.
To compile this dataset yourself, on the [MassIVE website](https://massive.ucsd.edu/ProteoSAFe/static/massive.jsp), go to [MassIVE Knowledge Base](https://massive.ucsd.edu/ProteoSAFe/static/massive-kb-libraries.jsp) > [Human HCD Spectral Library](https://massive.ucsd.edu/ProteoSAFe/status.jsp?task=82c0124b6053407fa41ba98f53fd8d89) > [All Candidate library spectra](https://massive.ucsd.edu/ProteoSAFe/result.jsp?task=82c0124b6053407fa41ba98f53fd8d89&view=candidate_library_spectra) > Download.
This will give you a zipped TSV file with the metadata and peptide identifications for all 30 million PSMs.
Using the filename (column "filename") you can then retrieve the corresponding peak files from the MassIVE FTP server and extract the desired spectra using their scan number (column "scan").

**How do I know which model to use after training Casanovo?**

By default, Casanovo saves a snapshot of the model weights after every 50,000 training steps.
Note that the number of samples that are processed during a single training step depends on the batch size.
Therefore, when using the default training batch size of 32, this correspond to saving a model snapshot after every 1.6 million training samples.
You can optionally modify the snapshot frequency in the [config file](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml) (parameter `every_n_train_steps`), depending on your dataset size.
Note that taking very frequent model snapshots will result in somewhat slower training time because Casanovo will evaluate its performance on the validation data for every snapshot.

When saving a model snapshot, Casanovo will use the validation data to compute performance measures (training loss, validation loss, amino acid precision, and peptide precision) and print this information to the console and log file.
After your training job is finished, you can identify the best performing model that achieves the maximum peptide and amino acid precision from the log file and use the corresponding model snapshot.
