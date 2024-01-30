"""
Loader function for the m221124_002 model.
"""
import pickle
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

import honey_curve
import honey_curve.matrixops.interpolation as matint
from honey_curve.models.m221124_001.loader import M221124_001, JumpDetectionModel
from honey_curve.models.m221124_002.functions import calc_honey_curve


class M221124_002:
    """Custom model class for the m221124_002 model."""

    def __init__(self, jump_detection_model: JumpDetectionModel = M221124_001()) -> None:
        """Load the m221124_002 model, which is a HistGradientBoostingClassifier from sklearn,
        which can classify the jumps of a weights function in three classes:
        - "is_big_jump" (the jump is of type "Metti Arnia", "Togli Arnia", "Metti Melario", "Togli Melario")
        - "is_small_jump" (the jump is small, but still a true jump)
        - "no_type" (the jump is not really a jump, but just a normal small variation)

        This model depends on the preprocessing done by the jump_detection_model, which is the
        is the m221124_001 model"""
        with as_file(files(honey_curve).joinpath("models", "m221124_002")) as path:  # type: ignore[call-arg]
            self.path_to_model_folder = Path(path)
        self.path_to_model = (
            self.path_to_model_folder / "m221124_002_barebone.pickle"
        )  # switch to the barebone model
        with open(self.path_to_model, "rb") as f:
            self.jump_classification_model = pickle.load(f)
        self.jump_detection_model = jump_detection_model

    def preprocess(self, df_weights: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Apply all data processing that is needed in order before launching the tag algorithm.

        This preprocessing is based on the same preprocessing of the jump_detection_model.

        Args:
            df_weights: Raw Weights time series dataframe of the type:

            |    | acquired_at         |   total_weight_value |
            |---:|:--------------------|---------------------:|
            |  0 | 2019-01-01 00:08:14 |              28.4    |
            |  1 | 2019-01-01 01:08:14 |              28.3542 |
            |  2 | 2019-01-01 02:08:14 |              28.3543 |
            |  3 | 2019-01-01 03:08:14 |              28.3486 |
            |  4 | 2019-01-01 04:08:14 |              28.2857 |

            that come from a select from the threebee_production.weights table for a given hive_id
            and a given year.
            start_date: Date in format 'YYYY-MM-DD' (e.g. '2023-01-30') to use as the start of the
                reindex.
            end_date: Date in format 'YYYY-MM-DD' (e.g. '2023-01-30') to use as the end of the
                reindex.

        Returns:
            df_weights_preproc: Preprocessed weights time series of the type:

            |                           |   total_weight_value |
            |:--------------------------|---------------------:|
            | 2019-01-01 00:00:00+00:00 |              28.4    |
            | 2019-01-01 00:30:00+00:00 |              28.4    |
            | 2019-01-01 01:00:00+00:00 |              28.4    |
            | 2019-01-01 01:30:00+00:00 |              28.3542 |
            | 2019-01-01 02:00:00+00:00 |              28.3542 |

            where all of the needed preprocessing steps have been applied.
        """
        df_weights_preproc = self.jump_detection_model.preprocess(
            df_weights, start_date=start_date, end_date=end_date
        )
        return df_weights_preproc

    def tag(self, df_weights_preproc: pd.DataFrame) -> pd.DataFrame:
        """Apply the m221124_002 model algorithm to tag the input weights time series.

        Args:
            df_weights: Weights time series dataframe of the type:

            |                           |   total_weight_value |
            |:--------------------------|---------------------:|
            | 2021-01-01 00:00:00+00:00 |              28.1829 |
            | 2021-01-01 00:30:00+00:00 |              28.1829 |
            | 2021-01-01 01:00:00+00:00 |              28.1829 |
            | 2021-01-01 01:30:00+00:00 |              28.1829 |
            | 2021-01-01 02:00:00+00:00 |              28.1829 |

            where the sample frequency is "30min". This can be the output of the preprocess()
            function on a raw weights dataframe from the threebee_production.weights table:
            >>> df_weights_preproc = model.preprocess(df_weights)
            >>> df_weights_tagged = model.tag(df_weights_preproc)

        Returns:
            df_weights_tagged: A tagged version of the input dataframes where the weight jumps have
                been tagged and several jump features are added.

                Note that the HistGradientBoostingClassifier only adds the last 'jump_type' column,
                while all of the other feature columns are added by the previous
                jump_detection_model.

            |                           |   total_weight_value | is_jump   |   jump_size |   jump_baseline_left |   jump_baseline_right |   jump_max |    jump_min |   jump_center |   jump_step_factor_180 |   jump_step_factor_360 |   jump_slope_factor_180 |   jump_slope_factor_360 | jump_type   |
            |:--------------------------|---------------------:|:----------|------------:|---------------------:|----------------------:|-----------:|------------:|--------------:|-----------------------:|-----------------------:|------------------------:|------------------------:|:------------|
            | 2019-03-03 17:30:00+00:00 |            0.0571429 | False     |    nan      |          nan         |              nan      |   nan      | nan         |        nan    |          nan           |          nan           |             nan         |            nan          |             |
            | 2019-03-03 18:00:00+00:00 |            0.0571429 | False     |    nan      |          nan         |              nan      |   nan      | nan         |        nan    |          nan           |          nan           |             nan         |            nan          |             |
            | 2019-03-03 18:30:00+00:00 |           27.1828    | True      |     27.1257 |            0.0571429 |               27.1828 |    27.1828 |   0.0571429 |         13.62 |            0.000160239 |            0.000565211 |               0.0032299 |              0.00174138 | is_big_jump |
            | 2019-03-03 19:00:00+00:00 |           27.1828    | False     |    nan      |          nan         |              nan      |   nan      | nan         |        nan    |          nan           |          nan           |             nan         |            nan          |             |
            | 2019-03-03 19:30:00+00:00 |           27.1828    | False     |    nan      |          nan         |              nan      |   nan      | nan         |        nan    |          nan           |          nan           |             nan         |            nan          |             |
        """

        features = [
            "jump_size",
            "jump_baseline_left",
            "jump_baseline_right",
            "jump_max",
            "jump_min",
            "jump_center",
            "jump_step_factor_180",
            "jump_step_factor_360",
            "jump_slope_factor_180",
            "jump_slope_factor_360",
        ]

        df_weights_tagged = self.jump_detection_model.tag(df_weights_preproc)

        # Add the "jump_type" column to the tagged dataframe using the jump_classification_model
        df_jumps = df_weights_tagged[df_weights_tagged["is_jump"]]
        df_weights_tagged["jump_type"] = ""
        if not df_jumps.empty:
            df_weights_tagged.loc[
                df_jumps.index, "jump_type"
            ] = self.jump_classification_model.predict(df_jumps[features])

        # Manual correction 1: if the jump_size > threshold, set it as a
        # "is_big_jump" by default.
        threshold = 3.5
        df_big_jumps = df_jumps[df_jumps["jump_size"].abs() > threshold]
        df_weights_tagged.loc[df_big_jumps.index, "jump_type"] = "is_big_jump"

        # # Manual correction 2: convert some of the "real jumps" (the "is_big_jump" or
        # # "is_small_jump") to "no_type". These are events when some bee food has been added, so in
        # # order not to remove them, they are defines as "no_type". These type of events are
        # # characterized by a very negative slope in the 12 hours after the jump.
        # real_jump_types = ["is_big_jump", "is_small_jump"]
        # df_real_jumps = df_weights_tagged[df_weights_tagged["jump_type"].isin(real_jump_types)]
        # # Parameters for the slope. Calculated from examples
        # # slope_min = -7.4e-05
        # # slope_max = -3.3e-05
        # slope_min = -7.4e-05
        # slope_max = -4.3e-05
        # list_of_idx_of_bee_food_events = []
        # for idx, row in df_real_jumps.iterrows():
        #     # Consider a slice of 12 hours (720 minutes) after the jump
        #     df_slice_720 = df_weights_tagged.loc[
        #         (df_weights_tagged.index >= idx + pd.Timedelta("1 hours", tz=timezone.utc))  # type: ignore[operator, arg-type]
        #         & (df_weights_tagged.index <= idx + pd.Timedelta("15 hours", tz=timezone.utc))  # type: ignore[operator, arg-type]
        #     ]
        #     x = df_slice_720["total_weight_value"].index.astype(np.int64) // 10**9  # type: ignore[operator]
        #     y = df_slice_720["total_weight_value"].values
        #     bhat, _ = matint.linear_interp_1d(x, y)  # type: ignore[arg-type]
        #     slope = bhat[1]
        #     if (slope > slope_min) and (slope < slope_max):
        #         list_of_idx_of_bee_food_events.append(idx)
        # df_weights_tagged.loc[list_of_idx_of_bee_food_events, "jump_type"] = "no_type"

        return df_weights_tagged

    def calc_honey_curve(self, df_weights_tagged: pd.DataFrame) -> pd.DataFrame:
        """Calculate the "honey_curve" of the given tagged df_weights_tagged dataframe.

        The calculation is just given by:
            honey_curve = total_weight_value - jump_shift - start_baseline
        where:
            total_weight_value = Value of the "total_weight_value" column;
            jump_shift = Cumulative sum of the jumps, taken with sign;
            start_baseline = Baseline of the "total_weight_value" at the start, taken as the median
                of a certain timedelta.

        Args:
            df_weights_tagged: A tagged version of the input dataframes where the weight jumps have
                            been tagged and several jump features are added.

            |                           |   total_weight_value | is_jump   |   jump_size |   jump_baseline_left |   jump_baseline_right |   jump_max |   jump_min |   jump_center |   jump_step_factor_180 |   jump_step_factor_360 |   jump_slope_factor_180 |   jump_slope_factor_360 | jump_type   |
            |:--------------------------|---------------------:|:----------|------------:|---------------------:|----------------------:|-----------:|-----------:|--------------:|-----------------------:|-----------------------:|------------------------:|------------------------:|:------------|
            | 2020-01-01 10:30:00+00:00 |              25.7372 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |             nan        |             nan        |           nan           |           nan           |             |
            | 2020-01-01 11:00:00+00:00 |              25.7372 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |             nan        |             nan        |           nan           |           nan           |             |
            | 2020-01-01 11:30:00+00:00 |              25.0229 | True      |     -1.4743 |              25.7372 |               24.2629 |    25.7372 |    24.2629 |       25.0001 |               0.183124 |               0.152275 |            -0.000175511 |            -0.000104865 | no_type     |
            | 2020-01-01 12:00:00+00:00 |              24.2629 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |             nan        |             nan        |           nan           |           nan           |             |
            | 2020-01-01 12:30:00+00:00 |              24.2629 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |             nan        |             nan        |           nan           |           nan           |             |

        Returns:
            df_weights_honey: A tagged version with the "honey_curve" as the last column.

            |                           |   total_weight_value | is_jump   |   jump_size |   jump_baseline_left |   jump_baseline_right |   jump_max |   jump_min |   jump_center |   jump_step_factor_180 |   jump_step_factor_360 |   jump_slope_factor_180 |   jump_slope_factor_360 | jump_type   |   honey_curve |
            |:--------------------------|---------------------:|:----------|------------:|---------------------:|----------------------:|-----------:|-----------:|--------------:|-----------------------:|-----------------------:|------------------------:|------------------------:|:------------|--------------:|
            | 2020-01-01 10:30:00+00:00 |              25.7372 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |             nan        |             nan        |           nan           |           nan           |             |       -0.1428 |
            | 2020-01-01 11:00:00+00:00 |              25.7372 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |             nan        |             nan        |           nan           |           nan           |             |       -0.1428 |
            | 2020-01-01 11:30:00+00:00 |              25.0229 | True      |     -1.4743 |              25.7372 |               24.2629 |    25.7372 |    24.2629 |       25.0001 |               0.183124 |               0.152275 |            -0.000175511 |            -0.000104865 | no_type     |       -0.8571 |
            | 2020-01-01 12:00:00+00:00 |              24.2629 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |             nan        |             nan        |           nan           |           nan           |             |       -1.6171 |
            | 2020-01-01 12:30:00+00:00 |              24.2629 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |             nan        |             nan        |           nan           |           nan           |             |       -1.6171 |
        """
        df_weights_honey = calc_honey_curve(df_weights_tagged=df_weights_tagged, drop_tmpcols=True)
        return df_weights_honey

    def preprocess_and_tag(
        self, df_weights: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Conveniance method to apply both preprocessing and tagging."""
        df_weights_preproc = self.preprocess(df_weights, start_date=start_date, end_date=end_date)
        df_weights_tagged = self.tag(df_weights_preproc)
        return df_weights_tagged

    def preprocess_and_tag_and_calc_honey(
        self, df_weights: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Conveniance method to apply preprocessing, tagging and honey calculation."""
        df_weights_preproc = self.preprocess(df_weights, start_date=start_date, end_date=end_date)
        df_weights_tagged = self.tag(df_weights_preproc)
        df_weights_honey = self.calc_honey_curve(df_weights_tagged)
        return df_weights_honey
