# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: honey_curve_light
#     language: python
#     name: honey_curve_light
# ---

# %% [markdown]
# # M221124_002 Load and run model

# %% [markdown]
# In this notebook we test the loading and run of the **M221124_002** model on a time series downloaded from the 3Bee database

# %%
from datetime import timezone
import pandas as pd
from honey_curve.database.models_threebee_production import (
    Apicultures,
    Devices,
    Hives,
    Locations,
    Notes,
    Temperatures,
    Weights,
)
from honey_curve.models.m221124_002.loader import M221124_002
from honey_curve.settings.loaders import load_config_dotenv
from sqlalchemy import String, create_engine, func, select, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

pd.set_option("display.max_columns", None)


# %% [markdown]
# # Download raw trace from DB


# %%
def download_weight_given_year_and_hive_id(connect_url, hive_id, year):
    """Select the data from the 'weights' table for the given year and hive_id."""

    engine = create_engine(connect_url, echo=False, future=True)
    session = Session(engine)

    print(f"    * Downloading weights for {year=} and {hive_id=}")

    # Download the weights for the year
    stmt = (
        select(Weights.id, Weights.hive_id, Weights.acquired_at, Weights.total_weight_value)  # type: ignore[attr-defined,arg-type]
        .filter(Weights.hive_id == hive_id)
        .filter(Weights.acquired_at.between(f"{year}-01-01", f"{year}-12-31"))
    )
    weights = session.execute(stmt).all()
    df_weights = pd.DataFrame(weights)

    return df_weights


def download_weight_given_year_and_device_id(connect_url, device_id, year):
    """Select the data from the 'weights' table for the given year and device_id."""
    engine = create_engine(connect_url, echo=False, future=True)
    session = Session(engine)
    device_id = str(device_id)
    print(f"    * Downloading weights for {year=} and {device_id=}")
    stmt = (
        select(
            Devices.id,
            Devices.hive_id,
            Devices.external_id,
            Weights.acquired_at,
            Weights.total_weight_value,
        )
        .filter(Devices.external_id.cast(String).contains(device_id))
        .join(Weights, Weights.device_id == Devices.id)
        .filter(Weights.acquired_at.between(f"{year}-01-01", f"{year}-12-31"))
    )
    results = session.execute(stmt).all()
    df_results = pd.DataFrame(results)
    return df_results


# %% [markdown]
# Connect to the DB

# %%
config = load_config_dotenv()

connect_url = URL.create(
    "mysql+pymysql",
    username=config.CRED_3BEE_PROD_DB_USER,
    password=config.CRED_3BEE_PROD_DB_PASSWORD,
    host=config.CRED_3BEE_PROD_DB_URL,
    port=config.CRED_3BEE_PROD_DB_PORT,
    database=config.CRED_3BEE_PROD_DB_NAME,
)
print("- Connected to the threebee_production database.")

# %% [markdown]
# Select year and hive_id, download data

# %%
device_id = 694218
year = 2022
print(f"{year=}")
print(f"{device_id=}")

# %%
df_weights_raw = download_weight_given_year_and_device_id(connect_url, device_id, year)

# %%
df_weights_raw.head(2)

# %%
df_weights_raw.shape

# %%

# %% [markdown]
# # Run the model to get the honey curve

# %%
model = M221124_002()

# %%
start_date = f"{year}-01-01"
end_date = f"{year}-12-31"
df_weights_honey = model.preprocess_and_tag_and_calc_honey(
    df_weights_raw, start_date=start_date, end_date=end_date
)

# %%
df_weights_honey.head()

# %%
df_weights_honey.shape

# %% tags=[]
model.jump_classification_model

# %%

# %% [markdown]
# # Calculate the monthly increments

# %% tags=[]
df_month = (
    df_weights_honey["honey_curve"].resample("M").last()
    - df_weights_honey["honey_curve"].resample("M").first()
).to_frame()
df_month.index = df_month.index.month_name(locale="it_IT.utf8")
df_month = df_month.rename(columns={"honey_curve": "Variazione Miele"})

# %% tags=[]
df_month

# %%

# %%
