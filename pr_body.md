## What changed
Log the number of training and validation spectra during model training setup. After the datasets are created in `DeNovoDataModule.setup()`, the total number of spectra for each dataset is logged via the casanovo logger.

## Why
Users need visibility into the size of their training and validation sets. This information is useful for monitoring and debugging training runs.

Fixes #583

## Testing
- Added `test_log_training_set_size` unit test that verifies training and validation spectra counts are logged when setting up the data module.
- All existing tests pass.
