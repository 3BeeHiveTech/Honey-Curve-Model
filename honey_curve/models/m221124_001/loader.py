"""
Loader function for the m221101_001 model.
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype  # type: ignore[attr-defined]

import honey_curve.timeseries.outlier as timout
import honey_curve.timeseries.resampling as timres
from honey_curve.models.m221124_001.algorithm import apply_weight_jump_detection_algorithm_v03


class M221124_001:
    """Custom model class for the m221124_001 model, used for jump detection."""

    def __init__(self) -> None:
        """Load the model. Since it is just an algorithm in this case, no data needs to be loaded."""

    def preprocess(self, df_weights: pd.DataFrame) -> pd.DataFrame:
        """Apply all data processing that is needed in order before launching the tag algorithm.

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

        assert ("acquired_at" in df_weights) and (
            is_datetime_or_timedelta_dtype(df_weights["acquired_at"])
        ), "The input df_weights must have a 'acquired_at' timestamp/datetime column."

        assert ("total_weight_value" in df_weights) and (
            is_numeric_dtype(df_weights["total_weight_value"])
        ), "The input df_weights must have a 'total_weight_value' numeric column."

        df_weights_series = (
            df_weights[["acquired_at", "total_weight_value"]]
            .sort_values("acquired_at")
            .set_index("acquired_at")
            .tz_localize("utc")
        )

        # Set a time threshold. This is used both when detecting the outliers to remove (so that
        # they are not removed if they are solitary points in the given time window) and when
        # filling the gaps in the data when resampling (so that gaps greater than this time window
        # are not filled).
        timestep_thresh = pd.Timedelta("18 hours")
        start_year = df_weights["acquired_at"].min().year
        start_date = f"{start_year}-01-01"
        end_year = df_weights["acquired_at"].max().year
        end_date = f"{end_year}-12-31"

        # Remove outliers via derivative, replace with median
        df_weights_outlier = timout.detect_outliers_via_derivative(
            series=df_weights_series["total_weight_value"],
            der_thresh=5,
            timestep_thresh=timestep_thresh,
            drop_tmpcols=True,
        ).rename(columns={"values": "total_weight_value"})
        weights_median = (
            df_weights_outlier["total_weight_value"].rolling("2 H", center=True).median()
        )
        df_weights_outlier.loc[
            df_weights_outlier["is_outlier"], "total_weight_value"
        ] = weights_median[df_weights_outlier["is_outlier"]]
        df_weights_outlier.drop(columns=["is_outlier"], inplace=True)

        # Remove outliers via median, replace with median
        df_weights_outlier = timout.detect_outliers_via_median(
            series=df_weights_outlier["total_weight_value"],
            median_thresh=2,
            median_window=pd.Timedelta("6 hours"),
            drop_tmpcols=True,
        ).rename(columns={"values": "total_weight_value"})
        weights_median = (
            df_weights_outlier["total_weight_value"].rolling("12 H", center=True).median()
        )
        df_weights_outlier.loc[
            df_weights_outlier["is_outlier"], "total_weight_value"
        ] = weights_median[df_weights_outlier["is_outlier"]]
        df_weights_outlier.drop(columns=["is_outlier"], inplace=True)
        ## TODO: Add other outlier preprocessing

        # Resample series with ffill
        df_weights_preproc = timres.resample_timeseries_dataframe(
            df_in=df_weights_outlier,
            start=start_date,
            end=end_date,
            freq="30min",
            method="ffill",
            mode="nearest",
            timestep_thresh=timestep_thresh,
        )

        return df_weights_preproc

    def tag(self, df_weights_preproc: pd.DataFrame) -> pd.DataFrame:
        """Apply the m221011_001 model algorithm to tag the input weights time series.

        Args:
            df_weights_preproc: Weights time series dataframe of the type:

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
            been tagged and several jump features are added. Have a look at the
            apply_weight_jump_detection_algorithm_v01() docstring to know more.

            |      | acquired_at               |   total_weight_value | is_jump   |   jump_size |   jump_baseline_left |   jump_baseline_right |   jump_max |   jump_min |   jump_center |   jump_step_factor_180 |   jump_step_factor_360 |   jump_slope_factor_180 |   jump_slope_factor_360 |
            |-----:|:--------------------------|---------------------:|:----------|------------:|---------------------:|----------------------:|-----------:|-----------:|--------------:|-----------------------:|-----------------------:|------------------------:|------------------------:|
            | 1524 | 2019-02-01 18:00:00+00:00 |              35.1714 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |
            | 1525 | 2019-02-01 18:30:00+00:00 |              35.1714 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |
            | 1526 | 2019-02-01 19:00:00+00:00 |              36.3828 | True      |      1.2114 |              35.1714 |               36.3828 |    36.4114 |    35.1714 |       35.7771 |              0.0922064 |               0.212966 |             0.000165076 |             0.000106627 |
            | 1527 | 2019-02-01 19:30:00+00:00 |              36.3828 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |
            | 1528 | 2019-02-01 20:00:00+00:00 |              36.4114 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |
        """

        df_weights_tagged = apply_weight_jump_detection_algorithm_v03(
            df_weights_preproc,
            der_thresh_small=0.75,
            der_thresh_big=3.5,
            jump_size_thresh=0.8,
            drop_tmpcols=True,
        )

        return df_weights_tagged

    def preprocess_and_tag(self, df_weights: pd.DataFrame) -> pd.DataFrame:
        """Conveniance method to apply both preprocessing and tagging."""
        df_weights_preproc = self.preprocess(df_weights)
        df_weights_tagged = self.tag(df_weights_preproc)
        return df_weights_tagged
