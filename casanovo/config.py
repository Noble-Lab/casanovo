"""Parse the YAML configuration."""

import logging
import shutil
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import yaml

from . import utils


logger = logging.getLogger("casanovo")


# FIXME: This contains deprecated config options to be removed in the next
#  major version update.
def _coerce_val_check_interval(value):
    """Accept ``val_check_interval`` as either an int or a float.

    PyTorch Lightning's ``Trainer(val_check_interval=...)`` accepts
    both, with different semantics:

    * ``int``, run validation every N training **steps** (batches).
    * ``float`` in ``[0.0, 1.0]``, run validation at that fraction of
      each **epoch** (``1.0`` = once per epoch end).

    Casanovo's config schema previously cast everything through
    ``int``, which silently truncated a user-supplied ``0.5`` to ``0``
    (equivalent to "validate every step", almost certainly not what
    the user wanted). Accept both shapes here, reject anything else,
    and reject out-of-range floats up front so the error message
    points at the config field rather than failing deep inside
    PyTorch Lightning's trainer setup.

    See https://github.com/Noble-Lab/casanovo/issues/627.
    """
    # ``bool`` is a subclass of ``int`` in Python, guard explicitly so
    # ``val_check_interval: true`` is not silently coerced to 1.
    if isinstance(value, bool):
        raise TypeError(
            "val_check_interval must be an int or a float in [0.0, 1.0], "
            f"got bool ({value!r})"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "val_check_interval as a float must be in [0.0, 1.0] "
                f"(fraction of epoch); got {value!r}"
            )
        return value
    if isinstance(value, str):
        # Prefer int over float when the string has no decimal point,
        # so ``"50000"`` keeps step-count semantics.
        if "." in value or "e" in value.lower():
            return _coerce_val_check_interval(float(value))
        return _coerce_val_check_interval(int(value))
    raise TypeError(
        "val_check_interval must be an int or a float in [0.0, 1.0], "
        f"got {type(value).__name__} ({value!r})"
    )


_config_deprecated = dict(
    n_peaks="max_peaks",
    every_n_train_steps="val_check_interval",
    max_iters="cosine_schedule_period_iters",
    max_length="max_peptide_len",
    save_top_k=None,
    model_save_folder_path=None,
    reverse_peptides=None,
)


class Config:
    """
    The Casanovo configuration options.

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
    config.max_peaks    # the max_peaks parameter
    config["max_peaks"] # also the max_peaks parameter
    ```
    """

    _default_config = Path(__file__).parent / "config.yaml"
    _config_types = dict(
        precursor_mass_tol=float,
        isotope_error_range=lambda min_max: (int(min_max[0]), int(min_max[1])),
        min_peptide_len=int,
        max_peptide_len=int,
        predict_batch_size=int,
        top_match=int,
        accelerator=str,
        devices=int,
        n_beams=int,
        enzyme=str,
        digestion=str,
        missed_cleavages=int,
        max_mods=int,
        allowed_fixed_mods=str,
        allowed_var_mods=str,
        random_seed=int,
        n_log=int,
        tb_summarywriter=bool,
        log_metrics=bool,
        log_every_n_steps=int,
        lance_dir=str,
        val_check_interval=lambda v: _coerce_val_check_interval(v),
        min_peaks=int,
        max_peaks=int,
        min_mz=float,
        max_mz=float,
        min_intensity=float,
        remove_precursor_tol=float,
        max_charge=int,
        dim_model=int,
        n_head=int,
        dim_feedforward=int,
        n_layers=int,
        dropout=float,
        dim_intensity=int,
        warmup_iters=int,
        cosine_schedule_period_iters=int,
        learning_rate=float,
        weight_decay=float,
        train_label_smoothing=float,
        train_batch_size=int,
        max_epochs=int,
        shuffle=bool,
        shuffle_buffer_size=int,
        num_sanity_val_steps=int,
        calculate_precision=bool,
        accumulate_grad_batches=int,
        gradient_clip_val=float,
        gradient_clip_algorithm=str,
        precision=str,
        replace_isoleucine_with_leucine=bool,
        massivekb_tokenizer=bool,
        residues=dict,
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
        # Validate.
        for key, val in self._config_types.items():
            self.validate_param(key, val)

        self._params["n_workers"] = utils.n_workers()

        if self._params["accelerator"] == "auto" and utils.is_apple_silicon():
            self._params["accelerator"] = "cpu"
            logger.warning(
                "accelerator='auto' will be overwritten to 'cpu' on Apple Silicon"
                " devices due to incompatibility with MPS accelerators.\n"
                "Note: If you want to use a different accelerator (other than MPS),"
                " please specify it explicitly in the config file."
            )

    def __getitem__(self, param: str) -> Union[int, bool, str, Tuple, Dict]:
        """Retrieve a parameter."""
        return self._params[param]

    def __getattr__(self, param: str) -> Union[int, bool, str, Tuple, Dict]:
        """Retrieve a parameter."""
        return self._params[param]

    def validate_param(self, param: str, param_type: Callable) -> None:
        """
        Verify that a parameter is the correct type.

        Parameters
        ----------
        param : str
            The Casanovo parameter.
        param_type : Callable
            The expected callable type of the parameter.

        Raises
        ------
        TypeError
            If the parameter is not the correct type.
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
        """Return the parameters."""
        return self._params.items()

    @classmethod
    def copy_default(cls, output: str) -> None:
        """
        Copy the default YAML configuration.

        Parameters
        ----------
        output : str
            The output file.
        """
        shutil.copyfile(cls._default_config, output)
