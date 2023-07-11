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
# # Convert the full model to light

# %% tags=[]
import pickle
from importlib.resources import as_file, files
from pathlib import Path

import honey_curve
from honey_curve.models.m221124_002.loader import M221124_002
from honey_curve.sklearn_light.binning import _BareboneBinMapper
from honey_curve.sklearn_light.gradient_boosting import (
    BareboneHistGradientBoostingClassifier,
)
from honey_curve.sklearn_light.loss import BareboneHalfMultinomialLoss
from honey_curve.sklearn_light.predictor import BareboneTreePredictor


def convert_TreePredictor(treepred):
    """Convert the input sklearn TreePredictor into the sklearn_light version."""
    return BareboneTreePredictor(
        nodes=treepred.nodes,
        binned_left_cat_bitsets=treepred.binned_left_cat_bitsets,
        raw_left_cat_bitsets=treepred.raw_left_cat_bitsets,
    )


def convert_predictors(predictors):
    """Convert the input sklearn predictors into the sklearn_light version."""
    return list(
        map(
            lambda x: list(map(lambda y: convert_TreePredictor(y), x)),
            predictors,
        )
    )


def convert_binmapper(binmapper):
    """Convert the input sklearn binmapper into the sklearn_light version."""

    return _BareboneBinMapper(
        n_bins=binmapper.n_bins,
        subsample=binmapper.subsample,
        is_categorical=binmapper.is_categorical,
        known_categories=binmapper.known_categories,
        random_state=binmapper.random_state,
        n_threads=binmapper.n_threads,
        is_categorical_=binmapper.is_categorical_,
        bin_thresholds_=binmapper.bin_thresholds_,
    )


with as_file(files(honey_curve).joinpath("models", "m221124_002")) as path:
    path_to_model_folder = Path(path)

# 1. Load the full sklearn model
path_to_model = path_to_model_folder / "m221124_002.pickle"
with open(path_to_model, "rb") as f:
    m221124_002_full = pickle.load(f)

m221124_002_bare = BareboneHistGradientBoostingClassifier(
    classes_=m221124_002_full.classes_,
    _loss=BareboneHalfMultinomialLoss(),
    _n_features=m221124_002_full._n_features,
    n_trees_per_iteration_=m221124_002_full.n_trees_per_iteration_,
    _baseline_prediction=m221124_002_full._baseline_prediction,
    _predictors=convert_predictors(m221124_002_full._predictors),
    _bin_mapper=convert_binmapper(m221124_002_full._bin_mapper),
    _check_feature_names=None,
)

# Save model
path_to_model = "m221124_002_barebone.pickle"
with open(path_to_model, "wb") as f:
    pickle.dump(m221124_002_bare, f)

# %%

# %% [markdown]
# ## Check that all attributes are non-sklearn

# %% tags=[]
m221124_002_bare.classes_

# %% tags=[]
m221124_002_bare._loss

# %% tags=[]
m221124_002_bare._n_features

# %% tags=[]
m221124_002_bare.n_trees_per_iteration_

# %% tags=[]
m221124_002_bare._baseline_prediction

# %% tags=[]
m221124_002_bare._predictors[0][0]

# %% tags=[]
type(m221124_002_bare._bin_mapper)

# %% tags=[]
m221124_002_bare._check_feature_names

# %%
m221124_002_bare._bin_mapper

# %%

# %% [markdown]
# # Check the binmapper attributes

# %% tags=[]
binmapper = m221124_002_bare._bin_mapper

# %% tags=[]
binmapper.n_bins

# %% tags=[]
binmapper.subsample

# %% tags=[]
binmapper.is_categorical

# %% tags=[]
binmapper.known_categories

# %% tags=[]
binmapper.random_state

# %% tags=[]
binmapper.n_threads

# %% tags=[]
binmapper.is_categorical_

# %% tags=[]
binmapper.bin_thresholds_[0][0]

# %% [markdown] tags=[]
# # TODO:
#
# - Finisci la traduzione di tutti gli attributi di m221124_002_bare;
# - Salva il nuovo modello in pickle;
# - Testa che il nuovo modello può essere caricato da pickle senza sklearn e che funziona in Load_and_run;
# - Pulisci i requirements, ritesta che funziona tutto da installazione.

# %% [markdown]
# https://github.com/scikit-learn/scikit-learn/tree/1.1.2

# %% [markdown]
# https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html

# %%
