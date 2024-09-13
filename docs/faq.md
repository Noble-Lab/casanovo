# Frequently Asked Questions

## Running Casanovo

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

## GPU Troubleshooting

**Casanovo is very slow even when running on the GPU. How can I speed it up?**

It is highly recommended to run Casanovo on the GPU to get the maximum performance.
If Casanovo is slow despite your system having a GPU, then the GPU might not be configured correctly.
A quick test to verify that Casanovo is using your (CUDA-enabled) GPU is to run `watch nvidia-smi` in your terminal.
If Casanovo has access to the GPU, then you should see it listed in the bottom process table, and the "Volatile GPU-Util" column at the top right should show activity while Casanovo is processing the data.

If Casanovo is not listed in the `nvidia-smi` output, then it is not using your GPU.
This is commonly caused by an incompatibility between your NVIDIA drivers and Pytorch.
Although Pytorch is installed automatically when installing Casanovo, in this case we recommend reinstalling it manually according to the following steps:

1. Uninstall the current version of Pytorch: `pip uninstall torch`
2. Install the latest version of the NVIDIA drivers using the [official CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). If supported by your system, an easy alternative can be conda using `conda install -c nvidia cuda-toolkit`.
3. Install the latest version of Pytorch according to the [instructions on the Pytorch website](https://pytorch.org/get-started/locally/).

Try to run Casanovo again and use `watch nvidia-smi` to inspect whether it can use the GPU now.
If this is still not the case, please [open an issue on GitHub](https://github.com/Noble-Lab/casanovo/issues/new).
Include full information about your system setup, the installed CUDA toolkit and Pytorch versions, and the troubleshooting steps you have performed.

**Why do I get a "CUDA out of memory" error when trying to run Casanovo?**

This means that there was not enough (free) memory available on your GPU to run Casanovo, which is especially likely to happen when you are using a smaller, consumer-grade GPU.
Depending on whether the error occurred during `train` or `denovo` mode, we recommend decreasing the `train_batch_size` or `predict_batch_size` options, respectively, in the [config file](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml) to reduce the number of spectra that are processed simultaneously.
Additionally, we recommend shutting down any other processes that may be running on the GPU, so that Casanovo can exclusively use the GPU.

**How can I run Casanovo on a specific GPU device?**

You can control which GPU(s) Casanovo uses by setting the `devices` option in the [configuration file](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml).
This setting also controls the number of cores to use when running on a CPU only (which can be specified using the `accelerator` option).

By default, Casanovo will automatically try to use the maximum number of devices available.
I.e., if your system has multiple GPUs, then Casanovo will use all of them for maximum efficiency.
Alternatively, you can select a specific GPU by specifying the GPU number as the value for `devices`.
For example, if you have a four-GPU system, when specifying `devices: 1` in your config file Casanovo will only use the GPU with identifier `1`.

The config file functionality only allows specifying a single GPU, by setting its id under `devices`, or all GPUs, by setting `devices: -1`.
If you want more fine-grained control to use some but not all GPUs on a multi-GPU system, then the `CUDA_VISIBLE_DEVICES` environment variable can be used instead.
For example, by setting `CUDA_VISIBLE_DEVICES=1,3`, only GPUs `1` and `3` will be visible to Casanovo, and specifying `devices: -1` will allow it to utilize both of these.

Note that when using `CUDA_VISIBLE_DEVICES`, the GPU numbers (potentially to be specified under `devices`) are reset to consecutively increase from `0`.

**I see "NotImplementedError: The operator 'aten::index.Tensor'..." when using a Mac with an Apple Silicon chip.**

Casanovo can leverage Apple's Metal Performance Shaders (MPS) on newer Mac computers, which requires that the `PYTORCH_ENABLE_MPS_FALLBACK` is set to `1`:

```sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

This will need to be set with each new shell session, or you can add it to your `.bashrc` / `.zshrc` to set this environment variable by default.

## Training Casanovo

**Where can I find the data that Casanovo was trained on?**

The [reported Casanovo results](https://doi.org/10.1101/2023.01.03.522621) were obtained by training on two different datasets: (i) a commonly used nine-species benchmark dataset, and (ii) a large-scale training dataset derived from the MassIVE Knowledge Base (MassIVE-KB).

All data for the _nine-species benchmark_ are available as annotated MGF files on MassIVE with [dataset identifier MSV000090982](https://doi.org/doi:10.25345/C52V2CK8J).
Annotated MGF files that are directly compatible with Casanovo are available in the `/MSV000090982/updates/2024-05-14_woutb_71950b89/peak/9speciesbenchmark` FTP directory.
Using these data, Casanovo was trained in a cross-validated fashion, training on eight species and testing on the remaining species.

The _MassIVE-KB training data_ was derived from PSMs used to compile the MassIVE-KB v1 spectral library and consists of 30 million PSMs.
These PSMs were obtained by collecting up to the top 100 PSMs for each of the precursors (as defined by a peptidoform and charge) included in MassIVE-KB.
To compile this dataset yourself, on the [MassIVE website](https://massive.ucsd.edu/ProteoSAFe/static/massive.jsp), go to [MassIVE Knowledge Base](https://massive.ucsd.edu/ProteoSAFe/static/massive-kb-libraries.jsp) > [Human HCD Spectral Library](https://massive.ucsd.edu/ProteoSAFe/status.jsp?task=82c0124b6053407fa41ba98f53fd8d89) > [All Candidate library spectra](https://massive.ucsd.edu/ProteoSAFe/result.jsp?task=82c0124b6053407fa41ba98f53fd8d89&view=candidate_library_spectra) > Download.
This will give you a zipped TSV file with the metadata and peptide identifications for all 30 million PSMs.
Using the filename (column "filename") you can then retrieve the corresponding peak files from the MassIVE FTP server and extract the desired spectra using their scan number (column "scan").

The _non-enzymatic dataset_, used to train a non-tryptic version of Casanovo, was created by selecting PSMs with a uniform distribution of amino acids at the C-terminal peptide positions from two datasets: MassIVE-KB and PROSPECT.
Training, validation, and test splits for the non-enzymatic dataset are available as annotated MGF files on MassIVE with [dataset identifier MSV000094014]](https://doi.org/doi:10.25345/C5KS6JG0W).

**How do I know which model to use after training Casanovo?**

By default, Casanovo saves a snapshot of the model weights after every 50,000 training steps.
Note that the number of samples that are processed during a single training step depends on the batch size.
Therefore, the default training batch size of 32 corresponds to saving a model snapshot after every 1.6 million training samples.
You can optionally modify the snapshot (and validation) frequency in the [config file](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml) (parameter `val_check_interval`), depending on your dataset size.
Note that taking very frequent model snapshots will result in slower training time because Casanovo will evaluate its performance on the validation data for every snapshot.

When saving a model snapshot, Casanovo will use the validation data to compute performance measures (training loss, validation loss, amino acid precision, and peptide precision) and print this information to the console and log file.
After your training job is finished, you can identify the model that achieves the maximum peptide and amino acid precision from the log file and use the corresponding model snapshot.

**Even though I added new post-translational modifications to the configuration file, Casanovo didn't identify those peptides.**

Casanovo can only make predictions using post-translational modifications (PTMs) that were included when training the model.
If you want to add new types of PTMs, then you will need to retrain the model.

The [`config.yaml` configuration file](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml) contains all amino acids and PTMs that Casanovo knows.
By default, this includes oxidation of methionine, deamidation of asparagine and glutamine, N-terminal acetylation, N-terminal carbamylation, and an N-terminal loss of ammonia.
(Additionally, cysteines are _always_ considered to be carbamidomethylated.)
Simply making changes to the `residues` alphabet in the configuration file is insufficient to identify new types of PTMs with Casanovo, however.
This is indicated by the fact that this option is not marked with `(I)` in the configuration file, which indicates options that can be modified during inference.
All remaining options require training a new Casanovo model.

Therefore, to learn the spectral signature of previously unknown PTMs, a new Casanovo version needs to be _trained_.
To include new PTMs in Casanovo, you need to:
1. Update the `residues` alphabet in the configuration file accordingly.
2. Compile a large training dataset that includes those PTMs and format this as an annotated MGF file. Note that you can include some or all of the data that was originally used to train Casanovo (see above), in addition to the data that includes your new types of PTMs.
3. Train a new version of Casanovo on this dataset.

It is unfortunately not possible to finetune a pre-trained Casanovo model to add new types of PTMs.
Instead, such a model must be trained from scratch.

**How can I change the learning rate schedule used during training?**

By default, Casanovo uses a learning rate schedule that combines linear warm up followed by a cosine decay (as implemented in `CosineWarmupScheduler` in `casanovo/denovo/model.py`) during training.
To use a different learning rate schedule, you can specify an alternative learning rate scheduler as follows (in the `lr_scheduler` variable in function `Spec2Pep.configure_optimizers` in `casanovo/denovo/model.py`):

```
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=self.warmup_iters)
```

You can use any of the scheduler classes available in [`torch.optim.lr_scheduler`](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) or implement your custom learning rate schedule similar to `CosineWarmupScheduler`.

## Miscellaneous

**Can I use Casanovo to sequence antibodies?**

Yes, antibody sequencing is one of the popular uses for de novo sequencing technology.
[This article](https://academic.oup.com/bib/article/24/1/bbac542/6955273) carried out a systematic comparison of six de novo sequencing tools (Novor, pNovo 3, DeepNovo, SMSNet, PointNovo and Casanovo).  Casanovo fared very well in this comparison: "Casanovo exhibits the highest number of correct peptide predictions compared with all other de novo algorithms across all enzymes demonstrating the advantage of using transformers for peptide sequencing. Furthermore, Casanovo predicts amino acids with overall superior precision."

In practice, you may want to try providing your Casanovo output file to the [Stitch software](https://github.com/snijderlab/stitch), which performs template-based assembly of de novo peptide reads to reconstruct antibody sequences ([Schulte and Snyder 2024](https://www.biorxiv.org/content/10.1101/2024.02.20.581155v1)).

**Where can I find Casanovo model weights trained on the nine-species benchmark?**

You can find the Casanovo weights corresponding to the nine-species benchmark [on Zenodo](https://doi.org/10.5281/zenodo.10694984), compatible with Casanovo v4.x.x.
These weights correspond to training and validation on eight species using the default configurations, with the remaining species held out for testing, as indicated by the file names.
Note that these weights are only intended for evaluation purposes on this specific benchmark dataset.
For general-purpose usage of Casanovo, use its [default weights](https://casanovo.readthedocs.io/en/latest/getting_started.html#download-model-weights) instead, as these will give significantly improved performance.

**How can I generate a precision–coverage curve?**

You can evaluate a trained Casanovo model compared to ground-truth peptide labels using a precision–coverage curve.

1. Run Casanovo in sequencing or evaluation mode on your MS/MS data, [as described here](https://casanovo.readthedocs.io/en/latest/getting_started.html#running-casanovo).
2. Collect the ground-truth peptide labels as well as the peptide labels predicted by Casanovo. Note that Casanovo might not report a peptide for every spectrum if the spectra are invalid (e.g. not enough peaks), so make sure that both pieces of information are correctly linked to each other (using the `spectra_ref` column in the mzTab output file produced by Casanovo).
3. Use the following script to plot a precision–coverage curve:
```python
import depthcharge
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

from casanovo.denovo import evaluate


# `psm_sequences` is assumed to be a DataFrame with at least the following
# three columns:
#   - "sequence": The ground-truth peptide labels.
#   - "sequence_pred": The predicted peptide labels.
#   - "search_engine_score[1]": The prediction scores.
psm_sequences = ...  # TODO: Get the PSM information.

# Sort the PSMs by descreasing prediction score.
psm_sequences = psm_sequences.sort_values(
    "search_engine_score[1]", ascending=False
)
# Find matches between the true and predicted peptide sequences.
aa_matches_batch = evaluate.aa_match_batch(
    psm_sequences["sequence"],
    psm_sequences["sequence_pred"],
    depthcharge.masses.PeptideMass("massivekb").masses,
)
# Calculate the peptide precision and coverage.
peptide_matches = np.asarray([aa_match[1] for aa_match in aa_matches_batch[0]])
precision = np.cumsum(peptide_matches) / np.arange(1, len(peptide_matches) + 1)
coverage = np.arange(1, len(peptide_matches) + 1) / len(peptide_matches)
# Calculate the score threshold at which peptide predictions don't fit the
# precursor m/z tolerance anymore.
threshold = np.argmax(psm_sequences["search_engine_score[1]"] < 0)

# Print the performance values.
print(f"Peptide precision = {precision[threshold]:.3f}")
print(f"Coverage = {coverage[threshold]:.3f}")
print(f"Peptide precision @ coverage=1 = {precision[-1]:.3f}")

# Plot the precision–coverage curve.
width = 4
height = width / 1.618
fig, ax = plt.subplots(figsize=(width, width))

ax.plot(
    coverage, precision, label=f"Casanovo AUC = {auc(coverage, precision):.3f}"
)
ax.scatter(
    coverage[threshold],
    precision[threshold],
    s=50,
    marker="D",
    edgecolors="black",
    zorder=10,
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_xlabel("Coverage")
ax.set_ylabel("Peptide precision")
ax.legend(loc="lower left")

plt.savefig("prec_cov.png", dpi=300, bbox_inches="tight")
plt.close()
```
