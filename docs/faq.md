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

**I get a "CUDA out of memory" error when trying to run Casanovo. Help!**

This means that there was not enough (free) memory available on your GPU to run Casanovo, which is especially likely to happen when you are using a smaller, consumer-grade GPU.
We recommend trying to decrease the `train_batch_size` or `predict_batch_size` options in the [config file](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml) (depending on whether the error occurred during `train` or `denovo` mode) to reduce the number of spectra that are processed simultaneously.
Additionally, we recommend shutting down any other processes that may be running on the GPU, so that Casanovo can exclusively use the GPU.

**How do I solve a "PermissionError: GitHub API rate limit exceeded" error when trying to run Casanovo?**

When running Casanovo in `denovo` or `eval` mode, Casanovo needs compatible pretrained model weights to make predictions.
If no model weights file is specified using the `--model` command-line parameter, Casanovo will automatically try to download the latest compatible model file from GitHub and save it to its cache for subsequent use.
However, the GitHub API is limited to maximum 60 requests per hour per IP address.
Consequently, if Casanovo has been executed multiple times already, it might temporarily not be able to communicate with GitHub.
You can avoid this error by explicitly specifying the model file using the `--model` parameter.

**I see "NotImplementedError: The operator 'aten::index.Tensor'..." when using a Mac with an Apple Silicon chip.**

Casanovo can leverage Apple's Metal Performance Shaders (MPS) on newer Mac computers, which requires that the `PYTORCH_ENABLE_MPS_FALLBACK` is set to `1`:

```sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

This will need to be set with each new shell session, or you can add it to your `.bashrc` / `.zshrc` to set this environment variable by default.

**How do I know which model to use after training Casanovo?**

By default, Casanovo saves a snapshot of the model weights after every 50,000 training steps.
Note that the number of samples that are processed during a single training step depends on the batch size.
Therefore, when using the default training batch size of 32, this correspond to saving a model snapshot after every 1.6 million training samples.
You can optionally modify the snapshot frequency in the [config file](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml) (parameter `every_n_train_steps`), depending on your dataset size.
Note that taking very frequent model snapshots will result in somewhat slower training time because Casanovo will evaluate its performance on the validation data for every snapshot.

When saving a model snapshot, Casanovo will use the validation data to compute performance measures (training loss, validation loss, amino acid precision, and peptide precision) and print this information to the console and log file.
After your training job is finished, you can identify the best performing model that achieves the maximum peptide and amino acid precision from the log file and use the corresponding model snapshot.
