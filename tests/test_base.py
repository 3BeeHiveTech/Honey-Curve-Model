import pickle
from importlib.resources import as_file, files
from pathlib import Path

import pandas as pd

import honey_curve
from honey_curve.base import NAME


def test_base():
    assert NAME == "honey_curve"


def test_BareboneHistGradientBoostingClassifier():
    """Test that the model loaded with the full sklearn library and the model loaded with the custom
    sklearn_light library have the same predictions."""

    with as_file(files(honey_curve).joinpath("models", "m221124_002")) as path:
        path_to_model_folder = Path(path)
    with as_file(files(honey_curve)) as path:
        path_to_tests_folder = Path(path).parent / "tests"

    # 1. Load the full sklearn model
    path_to_model = path_to_model_folder / "m221124_002.pickle"
    with open(path_to_model, "rb") as f:
        m221124_002_full = pickle.load(f)

    # 2. Load the  custom sklearn_light model
    path_to_model = path_to_model_folder / "m221124_002_barebone.pickle"
    with open(path_to_model, "rb") as f:
        m221124_002_bare = pickle.load(f)

    path_to_df = path_to_tests_folder / "df_features.pickle"
    df_features = pd.read_pickle(path_to_df)

    # 3. Compare the predictions
    pred_full = m221124_002_full.predict(df_features)
    pred_bare = m221124_002_bare.predict(df_features)

    assert (pred_full == pred_bare).all()
