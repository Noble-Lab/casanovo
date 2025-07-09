# File Formats

## Input File Formats For Casanovo

### MS/MS spectra

When you are ready to use Casanovo, you can input your MS/MS spectra in one of the following formats:

- **[mzML](https://doi.org/10.1074/mcp.R110.000133)**: XML-based mass spectrometry community standard file format developed by the Proteomics Standards Initiative (PSI).
- **[mzXML](https://doi.org/10.1038/nbt1031)**: XML-based predecessor of mzML. Although supported by Casanovo, mzML should typically be preferred instead.
- **[MGF](https://www.matrixscience.com/help/data_file_help.html)**: A simple text-based peak file, though not as rich in detail as mzML.

All three of the above file formats can be used as input to Casanovo for *de novo* peptide sequencing and database searching.
As the official PSI standard format containing the complete information from a mass spectrometry run, mzML should typically be preferred.

### FASTA (optional)

When using Casanovo for database searching, you will additionally need to provide a relevant FASTA file.
This is not necessary when using Casanovo for *de novo* peptide sequencing.

- **[FASTA](https://www.ncbi.nlm.nih.gov/WebSub/html/help/protein.html)**: A simple text-based file format that stores genetic/proteomic sequence information.

```{note}
Remember to add decoy sequences and common contaminants to your FASTA file, as Casanovo will not do this automatically.
```

```{warning}
In case the FASTA file contains amino acids that are not in Casanovo's vocabulary, peptides containing those residues will be ignored.
```

### Model weights

In addition to MS/MS spectra, Casanovo also optionally accepts a model weights (.ckpt extension) input file when running in training, sequencing, or evaluating mode.
These weights define the functionality of the Casanovo neural network.

If no input weights file is provided, Casanovo will automatically use the most recent compatible weights from the [official Casanovo GitHub repository](https://github.com/Noble-Lab/casanovo), which will be downloaded and cached locally if they are not already.
Model weights are retrieved by matching Casanovo release version, which is of the form (major, minor, patch).
If no model weights for an identical release are available, alternative releases with matching (i) major and minor, or (ii) major versions will be used.

Alternatively, you can input custom model weights in the form of a local file system path or a URL pointing to a compatible Casanovo model weights file.
If a URL is provided, the upstream weights file will be downloaded and cached locally for later use.
See the [command line interface documentation](cli.rst) for more details.

## Output: Understanding the mzTab format

After Casanovo processes your input file(s), it provides the results in an **[mzTab]((https://doi.org/10.1074/mcp.O113.036681))** file.
This file is divided into two main sections:

1. **Metadata section**: This part describes general information about the file and the Casanovo task.
2. **Peptide–spectrum match (PSM) section**: Details of the peptide sequences that Casanovo predicted for the MS/MS spectra.

mzTab files can contain additional sections to include protein identifications and quantification information as well.
However, as these levels of information are not relevant for Casanovo, these are not included in its output mzTab files.

```{tip}
mzTab is a human and machine readable format.
It can be inspected manually by opening it with a text editor or with spreadsheet software (specify tab as the delimiter).
Additionally, you can use tools like [Pyteomics](https://pyteomics.readthedocs.io/en/latest/api/mztab.html) for Python or [MSnbase](https://rdrr.io/bioc/MSnbase/man/readMzTabData.html) for R to programmatically read mzTab files.
```

**Metadata section**

The metadata section consists of three columns, each separated by a tab:
1. The prefix `MTD` indicating that this is the metadata section.
2. A key describing a metadata item.
3. The value corresponding to the metadata key.

As an example, these are the first few lines in an mzTab output file produced by Casanovo:

```
MTD	mzTab-version	1.0.0
MTD	mzTab-mode	Summary
MTD	mzTab-type	Identification
MTD	description	Casanovo identification file my_example_output
MTD	software[1]	[MS, MS:1003281, Casanovo, 4.0.1]
```

This identifies this mzTab file with filename "my_example_output" as a summary-level identification file produced by Casanovo.
On the final line you can see a typical key–value entry using information defined in the [PSI-MS controlled vocabulary](https://github.com/HUPO-PSI/psi-ms-CV/).
In this case, the line indicates that the file is produced by the Casanovo software, which is recorded in the `MS` controlled vocabulary with accession number `MS:1003281`.
The final element is the version number of Casanovo that produced this file.

The next few lines typically list the post-translational modifications (PTMs) that Casanovo knew:

```
MTD	fixed_mod[1]	[UNIMOD, UNIMOD:4, Carbamidomethyl, ]
MTD	fixed_mod[1]-site	C
MTD	variable_mod[1]	[UNIMOD, UNIMOD:7, Deamidated, ]
MTD	variable_mod[1]-site	N
MTD	variable_mod[2]	[UNIMOD, UNIMOD:7, Deamidated, ]
MTD	variable_mod[2]-site	Q
MTD	variable_mod[3]	[UNIMOD, UNIMOD:35, Oxidation, ]
MTD	variable_mod[3]-site	M
MTD	variable_mod[4]	[UNIMOD, UNIMOD:385, Ammonia-loss, ]
MTD	variable_mod[4]-site	N-term
MTD	variable_mod[5]	[UNIMOD, UNIMOD:1, Acetyl, ]
MTD	variable_mod[5]-site	N-term
MTD	variable_mod[6]	[UNIMOD, UNIMOD:5, Carbamyl, ]
MTD	variable_mod[6]-site	N-term
```

This indicates that cysteine carbamidomethylation was used as a static modification (this time defined by the [Unimod](https://www.unimod.org/) controlled vocabulary), and that deamidation of asparagine and glutamine, oxidation of methionine, N-terminal loss of ammonia, N-terminal acetylation, and N-terminal carbamylation were used as variable modifications.
Different PTMs in Casanovo can only be enabled or disabled by training a new model.

The final piece of information in the metadata section is the active configuration settings, allowing for replication or review of the analysis parameters:

```
MTD	software[1]-setting[1]	model = casanovo_massivekb_v4_0_0.ckpt
MTD	software[1]-setting[2]	config_filename = default
MTD	software[1]-setting[3]	precursor_mass_tol = 50.0
MTD	software[1]-setting[4]	isotope_error_range = (0, 1)
MTD	software[1]-setting[5]	min_peptide_len = 6
MTD	software[1]-setting[6]	max_peptide_len = 100
MTD	software[1]-setting[7]	predict_batch_size = 1024
MTD	software[1]-setting[8]	top_match = 1
MTD	software[1]-setting[9]	accelerator = auto
MTD	software[1]-setting[10]	devices = None
MTD	software[1]-setting[11]	n_beams = 10
MTD	software[1]-setting[12]	enzyme = trypsin
MTD	software[1]-setting[13]	digestion = full
MTD	software[1]-setting[14]	missed_cleavages = 0
MTD	software[1]-setting[15]	max_mods = 1
MTD	software[1]-setting[16]	allowed_fixed_mods = C:C+57.021
MTD	software[1]-setting[17]	allowed_var_mods = M:M+15.995,N:N+0.984,Q:Q+0.984,nterm:+42.011,nterm:+43.006,nterm:-17.027,nterm:+43.006-17.027
MTD	software[1]-setting[18]	random_seed = 454
MTD	software[1]-setting[19]	n_log = 1
MTD	software[1]-setting[20]	tb_summarywriter = False
MTD	software[1]-setting[21]	log_metrics = False
MTD	software[1]-setting[22]	log_every_n_steps = 50
MTD	software[1]-setting[23]	val_check_interval = 50000
MTD	software[1]-setting[24]	n_peaks = 150
MTD	software[1]-setting[25]	min_mz = 50.0
MTD	software[1]-setting[26]	max_mz = 2500.0
MTD	software[1]-setting[27]	min_intensity = 0.01
MTD	software[1]-setting[28]	remove_precursor_tol = 2.0
MTD	software[1]-setting[29]	max_charge = 10
MTD	software[1]-setting[30]	dim_model = 512
MTD	software[1]-setting[31]	n_head = 8
MTD	software[1]-setting[32]	dim_feedforward = 1024
MTD	software[1]-setting[33]	n_layers = 9
MTD	software[1]-setting[34]	dropout = 0.0
MTD	software[1]-setting[35]	dim_intensity = None
MTD	software[1]-setting[36]	warmup_iters = 100000
MTD	software[1]-setting[37]	cosine_schedule_period_iters = 600000
MTD	software[1]-setting[38]	learning_rate = 0.0005
MTD	software[1]-setting[39]	weight_decay = 1e-05
MTD	software[1]-setting[40]	train_label_smoothing = 0.01
MTD	software[1]-setting[41]	train_batch_size = 32
MTD	software[1]-setting[42]	max_epochs = 30
MTD	software[1]-setting[43]	num_sanity_val_steps = 0
MTD	software[1]-setting[44]	calculate_precision = False
MTD	software[1]-setting[46]	n_workers = 20
MTD	ms_run[1]-location	file://[...]/my_example_input.mgf
```

```{info}
Some of these configuration settings may only apply to specific modes of operation (`sequence`, `db-search`, `train`, etc.).
Irrespective of the mode of operation used, all settings will be reported in the mzTab file.
```

**PSM section**

The PSM section in mzTab files starts with a header line, indicated by the `PSH` key, which defines the subsequent tabular PSM information.
Next, the following lines each start with the `PSM` key and contain information for an individual PSM predicted by Casanovo.

```
PSH	sequence	PSM_ID	accession	unique	database	database_version	search_engine	search_engine_score[1]	modifications	retention_time	charge	exp_mass_to_charge	calc_mass_to_charge	spectra_ref	pre	post	start	end	opt_ms_run[1]_aa_scores
PSM	EPPTPLTYVAGAGSGVR	1	null	null	null	null	[MS, MS:1003281, Casanovo, 4.0.1]	0.968312939008077	null	null	2.0	836.439	836.4386613168799	ms_run[1]:index=0	null	null	null	null	0.96454,0.90841,0.97874,0.97979,0.97915,0.98254,0.98184,0.97898,0.86762,0.97782,0.97771,0.97899,0.97987,0.97788,0.97949,0.98074,0.97561
PSM	VVHGFYNPAVSRVLEQ	2	null	null	null	null	[MS, MS:1003281, Casanovo, 4.0.1]	0.9652494998539195	null	null	3.0	605.6572	605.65644936688	ms_run[1]:index=1	null	null	null	null	0.96870,0.97701,0.85667,0.97274,0.97827,0.97790,0.97829,0.97706,0.97654,0.97725,0.97778,0.95544,0.95622,0.96240,0.96992,0.96909
PSM	EPPTPLTYVAGGSLNR	3	null	null	null	null	[MS, MS:1003281, Casanovo, 4.0.1]	0.813004752730622	null	null	2.0	836.4398	836.4386608168799	ms_run[1]:index=2	null	null	null	null	0.78636,0.45168,0.64947,0.68432,0.89344,0.90091,0.90124,0.56938,0.89757,0.90204,0.90129,0.90190,0.80076,0.90097,0.90233,0.87599
PSM	LERPFVHLM+15.995VFLGGSGR	4	null	null	null	null	[MS, MS:1003281, Casanovo, 4.0.1]	0.758128507890635	null	null	4.0	483.7627	483.51345739187997	ms_run[1]:index=3	null	null	null	null	0.86884,0.85508,0.87392,0.39732,0.87556,0.87291,0.69642,0.87083,0.79858,0.86588,0.86291,0.84178,0.45706,0.52835,0.85704,0.41526,0.83419
PSM	GEYKLLPFNKLMLGEG	5	null	null	null	null	[MS, MS:1003281, Casanovo, 4.0.1]	-0.18260370983796959	null	null	3.0	602.99817	603.6586910335465	ms_run[1]:index=4	null	null	null	null	0.64453,0.77152,0.90248,0.84100,0.65059,0.89975,0.87886,0.82220,0.90324,0.90056,0.88657,0.86091,0.66978,0.63579,0.81815,0.90577
...
```

Key information for each PSM is as follows:
- `sequence`: The predicted peptide sequence.
- `PSM_ID`: A monotonically increasing index, serving as a unique identifier for each PSM.
- `search_engine_score[1]`: The score of this PSM.
- `spectra_ref`: Unique identifier linking the prediction back to the original spectrum in the input file(s).
- `opt_ms_run[1]_aa_scores`: Casanovo predicts peptides in an autoregressive fashion, one amino acid at a time. This column contains comma-separated scores of the individual amino acid predictions.

When running Casanovo in _database searching mode_ rather than *de novo* peptide sequencing mode, the PSM section will look slightly differently:

```
PSH	sequence	PSM_ID	accession	unique	database	database_version	search_engine	search_engine_score[1]	modifications	retention_time	charge	exp_mass_to_charge	calc_mass_to_charge	spectra_ref	pre	post	start	end	opt_ms_run[1]_aa_scores
PSM	THM+15.995ELGGK	1	sp|A5A616|MGTS_ECOLI	null	null	null	[MS, MS:1003281, Casanovo, 4.1.1.dev8+g258edb4.d20240329]	0.6994086	null	null	2	444.71582381688	444.7159	ms_run[1]:index=0	null	null	null	null	0.84454,0.81027,0.83296,0.56239,0.40844,0.83554,0.82437,0.84730,0.84514
...
```

In this case, each PSM contains additional information in the `accession` column referring to the identifier of the protein the matched peptide is derived from.

```{note}
Scores in Casanovo range from -1 to 1, where 1 indicates high confidence in the prediction.
A score below 0 occurs for a predicted peptide sequence that mismatches the observed precursor mass, in which case the score is penalized by subtracting 1.
This will also be evident from a difference in the observed precursor _m_/_z_, in the `exp_mass_to_charge` column, and the precursor _m_/_z_ calculated from the predicted peptide sequence, in the `calc_mass_to_charge` column.
Hence, it is important to properly configure settings that impact the precursor mass filter, such as the precursor mass tolerance (option `precursor_mass_tol`) and the isotopes to consider (option `isotope_error_range`).
```

The `spectra_ref` column is essential for connecting predictions back to the corresponding MS/MS spectra in the input file(s).
This column consists of two parts: the run index and the spectrum reference, separated by a colon.
- The run index is of the form `ms_run[FILE_INDEX]`, with `FILE_INDEX` referring to the corresponding run location in the metadata section. In the typical case when only a single input file was processed, this will be `1`.
- The spectrum reference can take the form of either a scan number or a spectrum index.
    - When using mzML or mzXML files as input, the spectrum reference will take the form of a scan number, encoded as `scan=SCAN`, with `SCAN` the scan number specified in the input file for this spectrum.
    - When using MGF files as input, the spectrum reference will be an index, encoded as `index=INDEX`, with `INDEX` the zero-based index of the spectrum in its input file. This is because MGF is not a standardized format that is not guaranteed to contain specific spectrum identifiers.

```{warning}
Be mindful of the input peak file format when linking Casanovo PSMs to their input spectra.
Even when the same raw file is converted to both mzML and MGF, scan numbers in the mzML file will generally not match spectrum indices in the MGF file, as the former contains both MS and MS/MS spectra while the latter only contains MS/MS spectra.
```

```{note}
The PSM identifier in the `PSM_ID` column is not necessarily identical to the spectrum index in the `spectra_ref` column, even for MGF files.
- `PSM_ID` is one-based, whereas spectrum indices in `spectra_ref` are zero-based.
- If multiple predictions are included per spectrum (configuration option `top_match`), each PSM will have a different identifier, but spectrum references will overlap.
```

## Casanovo Configuration

Casanovo operates based on settings defined in a [YAML configuration file](https://github.com/Noble-Lab/casanovo/blob/main/casanovo/config.yaml).
This file contains several options that affect how Casanovo processes your data and predicts peptide sequences.
If you run Casanovo without specifying a configuration file, it uses a set of default settings.
However, you might want to adjust these settings for several reasons, such as to capture specific characteristics of your data or to experiment with different training configurations.

To create a custom configuration file, you can start by generating a copy of the default configuration:

```sh
casanovo configure
```

You can then edit this file to adjust various settings.
After editing, specify your custom configuration file when running Casanovo with the `--config` option.

The configuration file is divided into sections, each containing options that are relevant to different phases of Casanovo's operation.
The first section contains options used to configure Casanovo during *de novo* peptide sequencing, followed by options in the second section that can only be modified when training a new model.
For example, the `top_match` option in the first section makes it possible to flexibly report multiple PSMs per spectrum during _de novo_ peptide sequencing.
In contrast, setting a different value for the `n_peaks` option in the second section is only possible when training a new model, and cannot be modified when predicting with a previously trained model that uses a different configuration.

```{tip}
Each change in the configuration can lead to different outcomes in the peptide sequencing process, so it may be beneficial to experiment with various settings to find the optimal configuration for your data.
Always consider your experimental design and the nature of your data when adjusting these settings.
```

## Logging

Casanovo generates detailed log files during operation, providing insights into its performance and aiding in troubleshooting.
These log files are named similarly to the output mzTab files but with a `.log` extension.
Log files detail every step Casanovo takes, including:

- Starting and ending timestamps of the sequencing or training process.
- Configuration options used.
- Warnings or errors encountered during processing, providing clues for troubleshooting.
- Summary statistics upon completion, offering a quick overview of the results.

```{tip}
Tips for using log files:
- Bug reporting: When encountering issues, including the relevant log file in your bug report can significantly aid in diagnosing the problem.
- Performance monitoring: Log files can be used to monitor the efficiency of Casanovo's operation over time, identifying potential bottlenecks.
```

## For Advanced Users: Training Casanovo

To train a new Casanovo model, the training and validation data must be provided as **annotated MGF files**.
Annotated MGF files are similar to standard MGF files but include a `SEQ` key–value pair in the spectrum header, indicating the peptide sequence for the corresponding spectrum.

Example of an annotated MGF file entry:

```
BEGIN IONS
TITLE=My spectrum title
PEPMASS=602.2881
CHARGE=2+
RTINSECONDS=985.44604
SEQ=HQGVM+15.995VGM+15.995GQK
84.08081817626953 0.1001848503947258
87.74003601074219 0.07622149586677551
...
END IONS
```

```{note}
In case the peptide sequence includes PTMs, ensure that these are formatted correctly and match the amino acid and modification vocabulary in the Casanovo configuration.
```

mzML or mzXML files are not supported as input during training, as these formats do not provide a mechanism to annotate their spectra with peptide sequences.
Similarly, in Casanovo evaluation mode only annotated MGF files are supported.

<!-- TODO: when index files can be reused, document this here -->

During training, Casanovo will save **checkpoint files** at every `val_check_interval` steps, specified in the configuration.
Model checkpoints will be saved to the folder specified by the `--output_dir` command line option with filename format `epoch=EPOCH-step=STEP.ckpt`, with `EPOCH` the epoch and `STEP` the training step at which the checkpoint was taken, helping you track progress and select the best model based on validation performance.

<!-- TODO: when checkpointing is made more flexible, update this information -->
