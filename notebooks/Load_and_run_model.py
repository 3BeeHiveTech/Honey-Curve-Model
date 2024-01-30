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
#     display_name: honey_curve
#     language: python
#     name: honey_curve
# ---

# %% [markdown]
# # M221124_002 Load and run model

# %% [markdown]
# In this notebook we test the loading and run of the **M221124_002** model on a time series downloaded from the 3Bee database

# %%
from datetime import timezone
import pandas as pd
import plotly.graph_objects as go
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
from plotly.subplots import make_subplots
from sqlalchemy import String, create_engine, func, select, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

pd.set_option("display.max_columns", None)

# %% [markdown]
# # Download raw trace from DB

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


# def download_weight_given_year_and_device_id(connect_url, device_id, year):
#     """Select the data from the 'weights' table for the given year and device_id."""
#     engine = create_engine(connect_url, echo=False, future=True)
#     session = Session(engine)
#     device_id = str(device_id)
#     print(f"    * Downloading weights for {year=} and {device_id=}")
#     stmt = (
#         select(
#             Devices.id,
#             Devices.hive_id,
#             Devices.external_id,
#             Weights.acquired_at,
#             Weights.total_weight_value,
#         )
#         .filter(Devices.external_id.cast(String).endswith(device_id))
#         .join(Weights, Weights.device_id == Devices.id)
#         .filter(Weights.acquired_at.between(f"{year}-01-01", f"{year}-12-31"))
#     )
#     results = session.execute(stmt).all()
#     df_results = pd.DataFrame(results)
#     return df_results


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
        .filter(Devices.external_id.cast(String).endswith(device_id))
        .join(Weights, Weights.hive_id == Devices.hive_id)
        .filter(Weights.acquired_at.between(f"{year}-01-01", f"{year}-12-31"))
    )
    results = session.execute(stmt).all()
    df_results = pd.DataFrame(results)
    return df_results


# %% [markdown]
# Select year and hive_id, download data

# %%
device_id = 765022
hive_id = 11521
year = 2023
print(f"{year=}")
print(f"{device_id=}")
print(f"{hive_id=}")

# %%
# df_weights_raw = download_weight_given_year_and_device_id(connect_url, device_id, year)
df_weights_raw = download_weight_given_year_and_hive_id(connect_url, hive_id, year)

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

# %% tags=[]
# model.jump_classification_model._predictors

# %%

# %% [markdown]
# # Plot the results

# %%
# Colormap states -> color
color_map = {
    # Jump Types
    "is_big_jump": "violet",
    "is_small_jump": "lightblue",
    "no_type": "gray",
    "start": "red",
    "end": "red",
}


def add_vrect(irow, icol, x0, x1, y0, y1, name, color):
    """Add a vrect on the selected plot, from x0 to x1
    and from y0 to y1, with the given hover text name and color."""

    # Add shape regions
    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor=color,
        opacity=0.8,
        layer="below",
        line_width=0,
        name="",
        row=irow,
        col=icol,
    )
    # HACK to add hover_text for the vrect: add a trace with a fill, setting opacity to 0
    fig.add_trace(
        go.Scatter(
            x=[x0, x0, x1, x1, x0],
            y=[y0, y1, y1, y0, y0],
            fill="toself",
            mode="lines",
            name="",
            text=name,
            opacity=0,
            fillcolor=color,
        ),
        row=irow,
        col=icol,
    )


## FIGURE
fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

#### First row
fig.add_trace(
    go.Scatter(
        x=df_weights_raw["acquired_at"],
        y=df_weights_raw["total_weight_value"],
        mode="lines+markers",
        name="raw weights",
        marker=dict(color="#b6002a", size=4),
    ),
    row=1,
    col=1,
)
# Add jumps vrects
ymin = df_weights_honey["total_weight_value"].min()
ymax = df_weights_honey["total_weight_value"].max()
for idx, row in df_weights_honey[
    (df_weights_honey["is_jump"])
    & (df_weights_honey["jump_type"].isin(["is_big_jump", "is_small_jump"]))
].iterrows():
    x0 = idx - pd.Timedelta("2 hours", tz=timezone.utc)
    x1 = idx + pd.Timedelta("2 hours", tz=timezone.utc)
    name = row["jump_type"]
    color = color_map[name]
    add_vrect(irow=1, icol=1, x0=x0, x1=x1, y0=ymin, y1=ymax, name=name, color=color)


#### Second row
fig.add_trace(
    go.Scatter(
        x=df_weights_honey.index,
        y=df_weights_honey["total_weight_value"],
        mode="lines+markers",
        name="filtered weights",
        marker=dict(color="#636EFA", size=4),
    ),
    row=2,
    col=1,
)
# Add jumps vrects
ymin = df_weights_honey["total_weight_value"].min()
ymax = df_weights_honey["total_weight_value"].max()
for idx, row in df_weights_honey[
    (df_weights_honey["is_jump"])
    & (df_weights_honey["jump_type"].isin(["is_big_jump", "is_small_jump"]))
].iterrows():
    x0 = idx - pd.Timedelta("2 hours", tz=timezone.utc)
    x1 = idx + pd.Timedelta("2 hours", tz=timezone.utc)
    name = row["jump_type"]
    color = color_map[name]
    add_vrect(irow=2, icol=1, x0=x0, x1=x1, y0=ymin, y1=ymax, name=name, color=color)

#### THird row
fig.add_trace(
    go.Scatter(
        x=df_weights_honey.index,
        y=df_weights_honey["honey_curve"],
        mode="lines+markers",
        name="honey_curve",
        marker=dict(color="gold", size=4),
    ),
    row=3,
    col=1,
)
# Add jumps vrects
ymin = df_weights_honey["honey_curve"].min()
ymax = df_weights_honey["honey_curve"].max()
for idx, row in df_weights_honey[
    (df_weights_honey["is_jump"])
    & (df_weights_honey["jump_type"].isin(["is_big_jump", "is_small_jump"]))
].iterrows():
    x0 = idx - pd.Timedelta("2 hours", tz=timezone.utc)
    x1 = idx + pd.Timedelta("2 hours", tz=timezone.utc)
    name = row["jump_type"]
    color = color_map[name]
    add_vrect(irow=3, icol=1, x0=x0, x1=x1, y0=ymin, y1=ymax, name=name, color=color)

#### Layout
fig.update_layout(
    height=400,
    autosize=True,
    # width=800,
    font=dict(size=14),
    margin=go.layout.Margin(
        l=0,
        r=0,
        b=0,
        t=10,  # left margin  # right margin  # bottom margin  # top margin
    ),
    # paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

# Update xaxis properties
fig.update_yaxes(title_text="weight", row=1, col=1)
fig.update_yaxes(title_text="filtered_weight", row=2, col=1)
fig.update_yaxes(title_text="honey_curve", row=3, col=1)
fig.update_xaxes(title_text="Time", row=3, col=1)

# Update ylim properties

fig.show()

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

# %% [markdown]
# # Test plotly

# %%
# import plotly.express as px
# df = px.data.iris()
# fig = px.scatter(df, x='sepal_width', y='sepal_length', color="species")

# # all three of these worked
# fig.show(renderer='iframe')
# fig.show(renderer='iframe_connected')
# fig.show(renderer='colab')

# %%
