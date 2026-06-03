"""Parse the YAML configuration using OmegaConf."""

import logging
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError as OmegaConfAttributeError

from . import utils

logger = logging.getLogger("casanovo")


# FIXME: This contains deprecated config options to be removed in the next
#  major version update.
_config_deprecated = dict(
    n_peaks="max_peaks",
    every_n_train_steps="val_check_interval",
    max_iters="cosine_schedule_period_iters",
    max_length="max_peptide_len",
    save_top_k=None,
    model_save_folder_path=None,
    reverse_peptides=None,
)


@dataclass
class CasanovoConfig:
    """Typed schema for Casanovo configuration parameters.

    This dataclass defines the expected names and types for every
    Casanovo configuration option. OmegaConf uses it as a *structured
    config* schema so that type mismatches in a loaded YAML file are
    caught at merge time rather than deep inside the model.

    Actual default values are stored in ``config.yaml``; fields here
    are marked ``MISSING`` so that the schema acts as a type contract
    rather than duplicating defaults.
    """

    # Inference / fine-tuning
    precursor_mass_tol: float = MISSING
    isotope_error_range: List[int] = MISSING
    min_peptide_len: int = MISSING
    max_peptide_len: int = MISSING
    predict_batch_size: int = MISSING
    top_match: int = MISSING
    accelerator: str = MISSING
    devices: Optional[int] = None
    n_beams: int = MISSING
    # Database search
    enzyme: str = MISSING
    digestion: str = MISSING
    missed_cleavages: int = MISSING
    max_mods: Optional[int] = None
    allowed_fixed_mods: str = MISSING
    allowed_var_mods: str = MISSING
    # Training – output
    random_seed: int = MISSING
    n_log: int = MISSING
    tb_summarywriter: bool = MISSING
    log_metrics: bool = MISSING
    log_every_n_steps: int = MISSING
    lance_dir: Optional[str] = None
    val_check_interval: int = MISSING
    # Spectrum processing
    min_peaks: int = MISSING
    max_peaks: int = MISSING
    min_mz: float = MISSING
    max_mz: float = MISSING
    min_intensity: float = MISSING
    remove_precursor_tol: float = MISSING
    max_charge: int = MISSING
    # Model architecture
    dim_model: int = MISSING
    n_head: int = MISSING
    dim_feedforward: int = MISSING
    n_layers: int = MISSING
    dropout: float = MISSING
    dim_intensity: Optional[int] = None
    # Optimiser / scheduler
    warmup_iters: int = MISSING
    cosine_schedule_period_iters: int = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    train_label_smoothing: float = MISSING
    # Training / inference options
    train_batch_size: int = MISSING
    max_epochs: int = MISSING
    shuffle: bool = MISSING
    shuffle_buffer_size: int = MISSING
    num_sanity_val_steps: int = MISSING
    calculate_precision: bool = MISSING
    accumulate_grad_batches: int = MISSING
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None
    precision: str = MISSING
    replace_isoleucine_with_leucine: bool = MISSING
    massivekb_tokenizer: bool = MISSING
    # Amino acid / modification vocabulary
    residues: Dict[str, float] = MISSING
    # Token initialization mapping for fine-tuning with extended vocabularies.
    new_token_init: Dict[str, str] = MISSING


class Config:
    """
    The Casanovo configuration options.

    Loads and merges YAML configuration using OmegaConf. The
    :class:`CasanovoConfig` dataclass provides a typed schema; actual
    default values are read from ``config.yaml``. User-supplied
    configuration files are merged on top of these defaults, so
    **partial override files are fully supported** — only the keys you
    want to change need to be present.

    Type mismatches (e.g. a string where a float is expected) are
    caught at load time by OmegaConf's structured-config validation.
    Unknown configuration keys raise a ``KeyError``.

    Parameters
    ----------
    config_file : str, optional
        Path to a user YAML configuration file. If ``None``, the
        built-in defaults from ``config.yaml`` are used.

    Examples
    --------
    config = Config("casanovo.yaml")
    config.max_peaks    # the max_peaks parameter
    config["max_peaks"] # also the max_peaks parameter
    """

    _default_config = Path(__file__).parent / "config.yaml"

    def __init__(self, config_file: Optional[str] = None):
        """Initialize a Config object."""
        self.file = str(config_file) if config_file is not None else "default"

        # Build a typed, struct-mode schema from the dataclass, then
        # populate it with the defaults from config.yaml.
        schema = OmegaConf.structured(CasanovoConfig)
        default_cfg = OmegaConf.load(self._default_config)
        base = OmegaConf.merge(schema, default_cfg)

        if config_file is None:
            self._cfg = base
        else:
            user_cfg = OmegaConf.load(config_file)
            if not isinstance(user_cfg, DictConfig):
                raise TypeError(
                    f"Config file {config_file!r} must contain a YAML "
                    f"mapping, got {type(user_cfg).__name__}"
                )

            # Remap deprecated config entries directly on the OmegaConf
            # object (user_cfg is non-struct, so keys can be mutated).
            for old, new in _config_deprecated.items():
                if old in user_cfg:
                    if new is not None:
                        if new in user_cfg:
                            # Both deprecated key and its replacement are
                            # present. Keep the replacement value and
                            # discard the deprecated key so it cannot
                            # silently overwrite the user's explicit value.
                            del user_cfg[old]
                            warning_msg = (
                                f"Deprecated config option '{old}' "
                                f"ignored; '{new}' is already set"
                            )
                        else:
                            val = user_cfg[old]
                            del user_cfg[old]
                            user_cfg[new] = val
                            warning_msg = (
                                f"Deprecated config option '{old}' "
                                f"remapped to '{new}'"
                            )
                    else:
                        del user_cfg[old]
                        warning_msg = (
                            f"Deprecated config option '{old}' "
                            "is no longer in use"
                        )
                    warnings.warn(
                        warning_msg, DeprecationWarning, stacklevel=2
                    )

            # Merge user overrides on top of the schema-validated base.
            # The struct-mode base causes OmegaConf to raise
            # ConfigAttributeError for any key not present in the schema.
            try:
                self._cfg = OmegaConf.merge(base, user_cfg)
            except OmegaConfAttributeError as e:
                raise KeyError(f"Unrecognized config option(s): {e}") from e

        # Allow adding runtime-computed keys not defined in the schema
        # (e.g. n_workers).
        OmegaConf.set_struct(self._cfg, False)
        self._cfg.n_workers = utils.n_workers()

        if self._cfg.accelerator == "auto" and utils.is_apple_silicon():
            self._cfg.accelerator = "cpu"
            logger.warning(
                "accelerator='auto' will be overwritten to 'cpu' on Apple "
                "Silicon devices due to incompatibility with MPS "
                "accelerators.\nNote: If you want to use a different "
                "accelerator (other than MPS), please specify it explicitly "
                "in the config file."
            )

    def __getitem__(self, param: str) -> Any:
        """Retrieve a parameter."""
        try:
            val = self._cfg[param]
        except KeyError:
            raise KeyError(param) from None
        if isinstance(val, (DictConfig, ListConfig)):
            result = OmegaConf.to_container(val, resolve=True)
            # Preserve historical tuple contract for isotope_error_range.
            if param == "isotope_error_range" and isinstance(result, list):
                return tuple(int(v) for v in result)
            return result
        return val

    def __getattr__(self, param: str) -> Any:
        """Retrieve a parameter.

        ``__getattr__`` is only called when normal attribute lookup
        fails, so instance attributes set in ``__init__`` (e.g.
        ``self.file``, ``self._cfg``) are returned directly without
        ever reaching this method.
        """
        # Guard against infinite recursion for private/dunder attrs
        # that have not been set yet (e.g. _cfg during __init__).
        if param.startswith("_"):
            raise AttributeError(param)
        try:
            return self[param]
        except KeyError:
            raise AttributeError(
                f"'Config' object has no attribute '{param}'"
            ) from None

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Return the parameters as (key, value) pairs."""
        return OmegaConf.to_container(self._cfg, resolve=True).items()

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
