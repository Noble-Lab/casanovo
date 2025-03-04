"""Parse the YAML configuration."""

import logging
import shutil
import warnings
from pathlib import Path
from typing import Optional, Dict, Callable, Tuple, Union

import yaml

from . import utils

logger = logging.getLogger("casanovo")


# FIXME: This contains deprecated config options to be removed in the next
#  major version update.
_config_deprecated = dict(
    every_n_train_steps="val_check_interval",
    max_iters="cosine_schedule_period_iters",
    max_length="max_peptide_len",
    save_top_k=None,
    model_save_folder_path=None,
)


class Config:
    """The Casanovo configuration options.

    If a parameter is missing from a user's configuration file, the
    default value is assumed.

    Parameters
    ----------
    config_file : str, optional
        The provided user configuration file.

    Examples
    --------
    ```
    config = Config("casanovo.yaml")
    config.n_peaks    # the n_peaks parameter
    config["n_peaks"] # also the n_peaks parameter
    ```
    """

    _default_config = Path(__file__).parent / "config.yaml"
    _config_types = dict(
        random_seed=int,
        n_peaks=int,
        min_mz=float,
        max_mz=float,
        min_intensity=float,
        remove_precursor_tol=float,
        max_charge=int,
        precursor_mass_tol=float,
        isotope_error_range=lambda min_max: (int(min_max[0]), int(min_max[1])),
        min_peptide_len=int,
        dim_model=int,
        n_head=int,
        dim_feedforward=int,
        n_layers=int,
        dropout=float,
        dim_intensity=int,
        max_peptide_len=int,
        residues=dict,
        n_log=int,
        tb_summarywriter=bool,
        log_metrics=bool,
        log_every_n_steps=int,
        train_label_smoothing=float,
        warmup_iters=int,
        cosine_schedule_period_iters=int,
        learning_rate=float,
        weight_decay=float,
        train_batch_size=int,
        predict_batch_size=int,
        n_beams=int,
        top_match=int,
        max_epochs=int,
        num_sanity_val_steps=int,
        val_check_interval=int,
        calculate_precision=bool,
        accelerator=str,
        devices=int,
        lance_dir=str,
        shuffle=bool,
        buffer_size=int,
        reverse_peptides=bool,
        replace_isoleucine_with_leucine=bool,
        accumulate_grad_batches=int,
        gradient_clip_val=float,
        gradient_clip_algorithm=str,
        precision=str,
        early_stopping_patience=int,
        resume_training_from=str,
        mskb_tokenizer=bool,
    )

    def __init__(self, config_file: Optional[str] = None):
        """Initialize a Config object."""
        self.file = str(config_file) if config_file is not None else "default"
        with self._default_config.open() as f_in:
            self._params = yaml.safe_load(f_in)

        if config_file is None:
            self._user_config = {}
        else:
            with Path(config_file).open() as f_in:
                self._user_config = yaml.safe_load(f_in)
                # Remap deprecated config entries.
                for old, new in _config_deprecated.items():
                    if old in self._user_config:
                        if new is not None:
                            self._user_config[new] = self._user_config.pop(old)
                            warning_msg = (
                                f"Deprecated config option '{old}' "
                                f"remapped to '{new}'"
                            )
                        else:
                            del self._user_config[old]
                            warning_msg = (
                                f"Deprecated config option '{old}' "
                                "is no longer in use"
                            )

                        warnings.warn(warning_msg, DeprecationWarning)
                # Check for missing entries in config file.
                config_missing = self._params.keys() - self._user_config.keys()
                if len(config_missing) > 0:
                    raise KeyError(
                        "Missing expected config option(s): "
                        f"{', '.join(config_missing)}"
                    )
                # Check for unrecognized config file entries.
                config_unknown = self._user_config.keys() - self._params.keys()
                if len(config_unknown) > 0:
                    raise KeyError(
                        "Unrecognized config option(s): "
                        f"{', '.join(config_unknown)}"
                    )
        # Validate:
        for key, val in self._config_types.items():
            self.validate_param(key, val)

        self._params["n_workers"] = utils.n_workers()

    def __getitem__(self, param: str) -> Union[int, bool, str, Tuple, Dict]:
        """Retrieve a parameter"""
        return self._params[param]

    def __getattr__(self, param: str) -> Union[int, bool, str, Tuple, Dict]:
        """Retrieve a parameter"""
        return self._params[param]

    def validate_param(self, param: str, param_type: Callable):
        """Verify a parameter is the correct type.

        Parameters
        ----------
        param : str
            The Casanovo parameter
        param_type : Callable
            The expected callable type of the parameter.
        """
        try:
            param_val = self._user_config.get(param, self._params[param])
            if param == "residues":
                residues = {
                    str(aa): float(mass) for aa, mass in param_val.items()
                }
                self._params["residues"] = residues
            elif param_val is not None:
                self._params[param] = param_type(param_val)
        except (TypeError, ValueError) as err:
            logger.error(
                "Incorrect type for configuration value %s: %s", param, err
            )
            raise TypeError(
                f"Incorrect type for configuration value {param}: {err}"
            )

    def items(self) -> Tuple[str, ...]:
        """Return the parameters"""
        return self._params.items()

    @classmethod
    def copy_default(cls, output: str) -> None:
        """Copy the default YAML configuration.

        Parameters
        ----------
        output : str
            The output file.
        """
        shutil.copyfile(cls._default_config, output)
