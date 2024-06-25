# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.2.1] - 2024-06-25

### Fixed

- Pin NumPy version to below v2.0 to ensure compatibility with current DepthCharge version.

## [4.2.0] - 2024-05-14

### Added

- A deprecation warning will be issued when deprecated config options are used in the config file or in the model weights file.

### Changed

- Config option `max_iters` has been renamed to `cosine_schedule_period_iters` to better reflect that it controls the number of iterations for the cosine half period of the learning rate.

### Fixed

- Fix beam search caching failure when multiple beams have an equal predicted peptide score by breaking ties randomly.
- The mzTab output file now has proper line endings regardless of platform, fixing the extra `\r` found when run on Windows.

## [4.1.0] - 2024-02-16

### Changed

- Instead of having to specify `train_from_scratch` in the config file, training will proceed from an existing model weights file if this is given as an argument to `casanovo train`.

### Fixed

- Fixed beam search decoding error due to non-deterministic selection of beams with equal scores.

## [4.0.1] - 2023-12-25

### Fixed

- Fix automatic PyPI upload.

## [4.0.0] - 2023-12-22

### Added

-  Checkpoints include model parameters, allowing for mismatches with the provided configuration file.
- `accelerator` parameter controls the accelerator (CPU, GPU, etc) that is used.
- `devices` parameter controls the number of accelerators used.
- `val_check_interval` parameter controls the frequency of both validation epochs and model checkpointing during training.
- `train_label_smoothing` parameter controls the amount of label smoothing applied when calculating the training loss.

### Changed

- The CLI has been overhauled to use subcommands.
- Upgraded to Lightning >=2.0.
- Checkpointing is configured to save the top-k models instead of all.
- Log steps rather than epochs as units of progress during training.
- Validation performance metrics are logged (and added to tensorboard) at the validation epoch, and training loss is logged at the end of training epoch, i.e. training and validation metrics are logged asynchronously.
- Irrelevant warning messages on the console output and in the log file are no longer shown.
- Nicely format logged warnings.
- `every_n_train_steps` has been renamed to `val_check_interval` in accordance to the corresponding Pytorch Lightning parameter.
- Training batches are randomly shuffled.
- Upgraded to Torch >=2.1.

### Removed

- Remove config option for a custom Pytorch Lightning logger.
- Remove superfluous `custom_encoder` config option.

### Fixed

- Casanovo runs on CPU and can pass all tests.
- Correctly refer to input peak files by their full file path.
- Specifying custom residues to retrain Casanovo is now possible.
- Upgrade to depthcharge v0.2.3 to fix sinusoidal encoding and for the `PeptideTransformerDecoder` hotfix.
- Correctly report amino acid precision and recall during validation.

## [3.5.0] - 2023-08-16

### Fixed

- Don't try to assign non-existing output writer during eval mode.
- Specifying custom residues to retrain Casanovo is now possible.

## [3.4.0] - 2023-06-19

### Added

- `every_n_train_steps` parameter now controls the frequency of both validation epochs and model checkpointing during training.

### Changed

- We now log steps rather than epochs as units of progress during training.
- Validation performance metrics are logged (and added to tensorboard) at the validation epoch, and training loss is logged at the end of training epoch, i.e. training and validation metrics are logged asynchronously.

### Fixed

- Correctly refer to input peak files by their full file path.

## [3.3.0] - 2023-04-04

### Added

- Included the `min_peptide_len` parameter in the configuration file to restrict predictions to peptide with a minimum length.
- Export multiple PSMs per spectrum using the `top_match` parameter in the configuration file.

### Changed

- Calculate the amino acid scores as the average of the amino acid scores and the peptide score.
- Spectra from mzML and mzXML peak files are referred to by their scan numbers in the mzTab output instead of their indexes.

### Fixed

- Verify that the final predicted amino acid is the stop token.
- Spectra are correctly matched to their input peak file when analyzing multiple files simultaneously.
- The score of the stop token is taken into account when calculating the predicted peptide score.
- Peptides with incorrect N-terminal modifications (multiple or internal positions) are no longer predicted.

## [3.2.0] - 2022-11-18

### Changed

- Update PyTorch Lightning global seed setting.
- Use beam search decoding rather than greedy decoding to predict the peptides.

### Fixed

- Don't use model weights with incorrect major version number.

## [3.1.0] - 2022-11-03

### Added

- Matching model weights are automatically downloaded from GitHub.
- Automatically calculate testing code coverage.

### Changed

- Maximum supported Python version updated to 3.10.
- No need to explicitly specify a config file, the default config will be used instead.
- Initialize Tensorboard during training by passing its directory location.

### Fixed

- Don't use worker threads on Windows and MacOS.
- Fix for running when no GPU is available.

## [3.0.0] - 2022-10-10

### Added

- The first PyPI release! :tada:
- Tests are run on every PR automatically.
- Test code coverage must be maintained or improved with each change.
- Log the active Casanovo configuration.
- Log to both the console and a log file.
- Use all available hardware resources (GPU and CPU).
- Add ICML paper citation info.
- Document GPU out of memory error in the README.
- Allow mzML and mzXML peak files as input during predicting.
- Ability to reuse an existing HDF5 index during training.
- Move the changelog information from the README to CHANGELOG.

### Changed

- Consistent code formatting using black.
- Assign a negative score to peptide predictions that don't fit the precursor _m_/_z_ tolerance.
- Faster empty token detection during decoding.
- Consistently set the random seed to get reproducible results.
- Spectrum indexes are written to temporary HDF5 files.
- Use spectrum_utils for spectrum preprocessing.
- Rename the mode to predict peptides for unknown spectra from `test` to `predict`.
- Export spectrum predictions to mzTab files.
- Update the residue alphabet to include N-terminal modifications (acetylation, carbamidomethylation, NH3 loss).
- Specify input peak files as a shell pattern rather than by their directory.
- Make the config file optional to specify.

### Removed

- Always preprocess spectra, rather than having this as a user option.

### Fixed

- Don't log overly detailed messages from dependencies.
- Don't crash on invalid spectrum preprocessing.
- Ensure that config values have the correct type.
- Don't crash when an invalid residue is encountered during predicting (i.e. an N-terminal modification in another position).
- Don't penalize peptide scores if the precursor _m_/_z_ fits a C13 difference.

## [2.1.1] - 2022-07-27

### Added

- Update tutorial in the README.

## [2.1.0] - 2022-07-02

### Fixed

- Use latest depthcharge version with stable memory usage and fix to positional encoding for amino acids.

## [2.0.1] - 2022-06-13

### Added

- Include release notes in the README.

## [2.0.0] - 2022-06-05

### Added

- Additional CLI functionality.
- Unit testing using pytest.
- Include a tutorial in the README.
- Publish documentation using sphinx/ReadTheDocs.

### Changed

- Specify config as a YAML file.

## [1.2.0] - 2022-03-07

### Added

- Include peptide and amino acid confidence scores in output file.

## [1.1.2] - 2022-02-20

### Added

- Support for multiple input MGF files in a directory.

## [1.1.1] - 2022-02-10

### Added

- Provide more CLI options.
- Ability to specify a custom config file.

## [1.1.0] - 2022-02-04

### Added

- Data infrastructure.
- Model and training/testing functionality.

## [1.0.0] - 2022-01-28

### Added

- Initial Casanovo version.

[Unreleased]: https://github.com/Noble-Lab/casanovo/compare/v4.2.1...HEAD
[4.2.1]: https://github.com/Noble-Lab/casanovo/compare/v4.2.0...v4.2.1
[4.2.0]: https://github.com/Noble-Lab/casanovo/compare/v4.1.0...v4.2.0
[4.1.0]: https://github.com/Noble-Lab/casanovo/compare/v4.0.1...v4.1.0
[4.0.1]: https://github.com/Noble-Lab/casanovo/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/Noble-Lab/casanovo/compare/v3.5.0...v4.0.0
[3.5.0]: https://github.com/Noble-Lab/casanovo/compare/v3.4.0...v3.5.0
[3.4.0]: https://github.com/Noble-Lab/casanovo/compare/v3.3.0...v3.4.0
[3.3.0]: https://github.com/Noble-Lab/casanovo/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/Noble-Lab/casanovo/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/Noble-Lab/casanovo/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/Noble-Lab/casanovo/compare/v2.1.1...v3.0.0
[2.1.1]: https://github.com/Noble-Lab/casanovo/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/Noble-Lab/casanovo/compare/v2.0.1...v2.1.0
[2.0.1]: https://github.com/Noble-Lab/casanovo/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/Noble-Lab/casanovo/compare/v1.2.0...v2.0.0
[1.2.0]: https://github.com/Noble-Lab/casanovo/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/Noble-Lab/casanovo/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/Noble-Lab/casanovo/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Noble-Lab/casanovo/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Noble-Lab/casanovo/releases/tag/v1.0.0
