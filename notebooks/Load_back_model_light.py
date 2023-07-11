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
#     display_name: esabloom_invest_venv
#     language: python
#     name: esabloom_invest_venv
# ---

# %% [markdown]
# # Load back the light model

# %% tags=[]
import cloudpickle

path_to_model = "m221124_002_barebone.pickle"
with open(path_to_model, "rb") as f:
    m221124_002_bare = cloudpickle.load(f)

# %% tags=[]
m221124_002_bare

# %% tags=[]
path_to_model = "test_sklearn.pickle"
with open(path_to_model, "rb") as f:
    sklearn = cloudpickle.load(f)

# %%

# %%
