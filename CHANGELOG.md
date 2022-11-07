# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Noble-Lab/casanovo/compare/v3.1.0...HEAD
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
