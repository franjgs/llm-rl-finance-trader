# src/gen_utils.py
"""
gen_utils.py

Shared configuration utilities for the RL finance pipeline.

Features
--------
- ``load_config()``: Load YAML configuration with error handling.
- Centralizes config loading for:
    * ``train_walk_forward.py``
    * ``analyze_walk_forward.py``
    * Any future script

All docstrings and comments in **English**.
"""

from pathlib import Path
import yaml
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config_walk_forward.yaml") -> Dict:
    """
    Load the configuration from a YAML file.

    This function is shared across training and analysis scripts to ensure
    consistent configuration loading and error reporting.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML config file.
        Default: ``configs/config_walk_forward.yaml``

    Returns
    -------
    Dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    yaml.YAMLError
        If the YAML is malformed.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded: {config_path.resolve()}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config {config_path}: {e}")
        raise