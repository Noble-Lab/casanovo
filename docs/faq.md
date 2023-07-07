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

**Even though I added new post-translational modifications to the configuration file, Casanovo didn't identify those peptides.**

Casanovo can only make predictions using post-translational modifications (PTMs) that were included when training the model.
If you want to add new types of PTMs, then you will need to retrain the model.

The [`config.yaml` configuration file](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml) contains all amino acids and PTMs that Casanovo knows.
By default, this includes oxidation of methionine, deamidation of asparagine and glutamine, N-terminal acetylation, N-terminal carbamylation, and an N-terminal loss of ammonia.
(Additionally, cysteines are _always_ considered to be carbamidomethylated.)
Simply making changes to the `residues` alphabet in the configuration file is insufficient to identify new types of PTMs with Casanovo, however.
This is indicated by the fact that this option is not marked with `(I)` in the configuration file, which indicates options that can be modified during inference.
Al remaining options require training a new Casanovo model.

Therefore, to learn the spectral signature of previously unknown PTMs, a new Casanovo version needs to be _trained_.
To include new PTMs in Casanovo, you need to:
1. Update the `residues` alphabet in the configuration file accordingly.
2. Compile a large training dataset that includes those PTMs and format this as an annotated MGF file. Note that you can include some or all of the data that was originally used to train Casanovo (see above), in addition to the data that includes your new types of PTMs.
3. Train a new version of Casanovo on this dataset.

It is unfortunately not possible to finetune a pre-trained Casanovo model to add new types of PTMs.
Instead, such a model must be trained from scratch.

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
