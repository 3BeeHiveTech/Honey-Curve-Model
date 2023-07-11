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
# # Load back the light model

# %% tags=[]
import pickle
from importlib.resources import as_file, files
from pathlib import Path

import honey_curve

with as_file(files(honey_curve).joinpath("models", "m221124_002")) as path:
    path_to_model_folder = Path(path)

# 2. Load the  custom sklearn_light model
path_to_model = path_to_model_folder / "m221124_002_barebone.pickle"
with open(path_to_model, "rb") as f:
    m221124_002_bare = pickle.load(f)

# %%
