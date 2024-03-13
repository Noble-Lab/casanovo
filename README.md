# PSM scoring with Casanovo-DB 

This branch of the Casanovo project contains code that implements the Casanovo-DB score function for database search. The preprint version of the paper can be found [here](https://www.biorxiv.org/content/10.1101/2024.01.26.577425v2). Note that this is currently only a proof-of-concept implementation for using Casanovo as a score function, and thus is not yet intended for use as a complete database search workflow. Our eventual goal is to add more comprehensive database search functionality as part of Casanovo.  For now, however, this branch allows for testing of the methodology by making use of some important functionality available in the Crux mass spectrometry toolkit (http://crux.ms).
You can install this branch using the following command:
```
  pip install git+https://github.com/Noble-Lab/casanovo.git@db_search
```
To use Casanovo-DB, you must also install the Crux toolkit.  Given a set of spectra in a file named, for example, `spectra.mgf` and a corresponding proteome fasta `proteome.fasta`, you can run a database search via the following commands:
1. Build a peptide index in the directory `my_proteome`:
- `crux tide-index proteome.fasta my_proteome`

Please note that your `.fasta` file cannot contain any 'U' amino acids because it is not in the vocabulary of Casanovo. Replace all occurrences of this character with 'X' to denote a missing amino acid. The minimal example attached has that done for you.

2. Identify candidate peptides for each spectrum. Be sure to set `top-match` to a very high number so every candidate PSM is considered:
- `crux tide-search --output-dir search_results --top-match 1000000 spectra.mgf my_proteome`
3. Extract the candidate peptides from the search results into a format readable by Casanovo-DB (`annotated.mgf`).
- `casanovo --mode=annotate --peak_path spectra.mgf --tide_dir_path search_results --output annotated.mgf`

Please note that `spectra.mgf` must contain the `SCANS=` field for this to work properly. The output `annotated.mgf` will have a unique `SEQ=` field for each spectrum where all candidates are comma-separated, with the decoys having the "decoy" prefix.

4. Run Casanovo-DB:
- `casanovo --mode=db --peak_path annotated.mgf --output casanovo_db_result.mztab`

The resulting file is in mztab format, similar to that produced by Casanovo's `sequence` command, except that there are scores for every candidate peptide against their respective spectrum (pairs as specified in `annotated.mgf`).

**Minimal Example Files:**

Here are some demo files for testing the current Casanovo-DB workflow:
- E. Coli `.fasta`: [ecoli_ux.fasta](sample_data/ecoli_ux.fasta)
- E. Coli `.mgf`: [minimal_example_ecoli.mgf](sample_data/minimal_example_ecoli.mgf)

**_De Novo_ Mass Spectrometry Peptide Sequencing with a Transformer Model**

![image](https://user-images.githubusercontent.com/32707537/152622912-ca87da20-a64c-4e3f-9ca1-721c6b0d9c64.png)

If you use Casanovo in your work, please cite the following publications:

- Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. *De novo* mass spectrometry peptide sequencing with a transformer model. in *Proceedings of the 39th International Conference on Machine Learning - ICML '22* vol. 162 25514–25522 (PMLR, 2022). [https://proceedings.mlr.press/v162/yilmaz22a.html](https://proceedings.mlr.press/v162/yilmaz22a.html)
- Yilmaz, M., Fondrie, W. E., Bittremieux, W., Nelson, R., Ananth, V., Oh, S. & Noble, W. S. Sequence-to-sequence translation from mass spectra to peptides with a transformer model. in *bioRxiv* (2023). [doi:10.1101/2023.01.03.522621](https://doi.org/10.1101/2023.01.03.522621)

## Documentation

#### https://casanovo.readthedocs.io/en/latest/


