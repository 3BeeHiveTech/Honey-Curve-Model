"""
Tests of the settings module.
"""

from honey_curve.settings.loaders import load_config_dotenv


def test_load_config_dotenv() -> None:
    """Test that the config.env setting are loaded properly from the config file stored at
    '~/.config/honey_curve/config.env'."""
    config = load_config_dotenv()
