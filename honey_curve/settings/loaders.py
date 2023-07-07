"""
This module contains the functions to load and validated the settings.
"""
import logging
from pathlib import Path

from honey_curve.settings import constants
from honey_curve.settings.models import ConfigDotEnv


def load_config_dotenv() -> ConfigDotEnv:
    """Load and parse the config.env settings with the pydantic model.

    Use the pydantic dotenv support to load from the .env file
    https://pydantic-docs.helpmanual.io/usage/settings/#dotenv-env-support
    """
    logging.info(f"Loading config from: {Path(constants.PATH_TO_CONFIG) / constants.FILE_CONFIG}")

    config = ConfigDotEnv(  # type: ignore[call-arg]
        _env_file=Path(constants.PATH_TO_CONFIG) / constants.FILE_CONFIG,
        _env_file_encoding="utf-8",
    )
    return config
