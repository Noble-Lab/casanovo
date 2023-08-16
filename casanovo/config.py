"""Parse the YAML configuration."""
import logging
from pathlib import Path
from typing import Optional, Dict, Callable, Tuple, Union

import yaml
import torch

from . import utils

logger = logging.getLogger("casanovo")


class Config:
    """The Casanovo configuration options.

    If a parameter is missing from a user's configuration file, the default
    value is assumed.

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
        max_length=int,
        residues=dict,
        n_log=int,
        tb_summarywriter=str,
        warmup_iters=int,
        max_iters=int,
        learning_rate=float,
        weight_decay=float,
        train_batch_size=int,
        predict_batch_size=int,
        n_beams=int,
        top_match=int,
        max_epochs=int,
        num_sanity_val_steps=int,
        train_from_scratch=bool,
        save_model=bool,
        model_save_folder_path=str,
        save_weights_only=bool,
        every_n_train_steps=int,
        no_gpu=bool,
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

        # Validate:
        for key, val in self._config_types.items():
            self.validate_param(key, val)

        # Add extra configuration options and scale by the number of GPUs.
        n_gpus = 0 if self["no_gpu"] else torch.cuda.device_count()
        self._params["n_workers"] = utils.n_workers()
        if n_gpus > 1:
            self._params["train_batch_size"] = (
                self["train_batch_size"] // n_gpus
            )

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
