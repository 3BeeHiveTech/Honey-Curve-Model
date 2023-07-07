"""Define here all constant variables used in the package"""

## --- INPUT CONSTANTS --- ##

# PATHS
PATH_TO_CONFIG = "~/.config/honey_curve/"  # Path to the config folder on Linux/MacOS

# FOLDER NAMES
FOLDER_DATA_PARENT = "data"
FOLDER_DATA_RAW = "0_raw"
FOLDER_DATA_PROCESSED = "1_processed"
FOLDER_DATA_DATASET = "2_dataset"
FOLDER_DATA_TMP = "3_tmp"
FOLDER_MODELS_PARENT = "models"
FOLDER_FIGURES_PARENT = "figures"
FOLDER_BIGDATA_PARENT = "bigdata"

# FILE NAMES
FILE_CONFIG = "config.env"  # File containing all of the config info. To be validated when loading.


# AWS S3 BUCKETS
AWS_S3_PROJECT_BUCKET_DATA = "s3://3bee-generali-parametrica-alveari-assets"
AWS_S3_PROJECT_BUCKET_BIGDATA = "s3://3bee-generali-parametrica-alveari-assets"  # NOTE: For the
# moment, it is the same bucket! Create a new one when you will start using it.
