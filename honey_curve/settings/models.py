"""
This module contains all of the data models that are needed to validate the settings files.
"""
from pydantic import BaseSettings


class ConfigDotEnv(BaseSettings):
    """Pydantic settings used to validate the config.env file at '~/.config/honey_curve/config.env'"""

    # CREDENTIALS
    # 3Bee production database
    CRED_3BEE_PROD_DB_NAME: str  # Name of the production database
    CRED_3BEE_PROD_DB_USER: str  # Username
    CRED_3BEE_PROD_DB_PASSWORD: str  # Password
    CRED_3BEE_PROD_DB_URL: str  # URL (e.g. "xxx.yyy.eu-central-1.rds.amazonaws.com")
    CRED_3BEE_PROD_DB_PORT: str  # Port (e.g. "3306")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
