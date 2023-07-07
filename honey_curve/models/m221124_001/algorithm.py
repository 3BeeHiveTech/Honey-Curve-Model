"""
Main m221101_001 model algorithm.
"""
from datetime import timezone

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import honey_curve.matrixops.interpolation as matint


def apply_weight_jump_detection_algorithm_v03(
    df_weights: pd.DataFrame,
    der_thresh_small: float = 0.75,
    der_thresh_big: float = 3.5,
    jump_size_thresh: float = 0.8,
    drop_tmpcols: bool = True,
) -> pd.DataFrame:
    """Apply the weights jump detection algoritm on the input df_weights time series dataframe.

    Args:
        df_weights: Weights time series dataframe of the type:

            |                           |   total_weight_value |
            |:--------------------------|---------------------:|
            | 2021-01-01 00:00:00+00:00 |              28.1829 |
            | 2021-01-01 00:30:00+00:00 |              28.1829 |
            | 2021-01-01 01:00:00+00:00 |              28.1829 |
            | 2021-01-01 01:30:00+00:00 |              28.1829 |
            | 2021-01-01 02:00:00+00:00 |              28.1829 |

            NOTE: The df_weights is supposed to be already resampled with frequency "30min".
            Furthermore, it has been tested only with resampling method "ffill", since in this
            case the jump detection are clearer. So an example of a compatible input df_weights
            would be something like:

                df_weights_raw = (
                    df_weights_select[["acquired_at", "total_weight_value"]]
                    .set_index("acquired_at")
                    .tz_localize("utc")
                )

                df_weights = timres.resample_timeseries_dataframe(
                    df_in=df_weights_raw,
                    start=start,
                    end=end,
                    freq="30min",
                    method="ffill",
                    mode="nearest",
                    timestep_thresh=pd.Timedelta("5 hours"),
                )

            where 'df_weights_select' is the raw dataframe that comes from a select of the
            'weights' production database table.

        der_thresh: Theshold applied on the derivative of the weight signal. Of the derivative is
            over der_thresh, then the sample point is a candidate weight jump. Default is 0.5 kg,
            so all changes lower than this will not be considered as weight jumps.

        jump_size_thresh: Theshold on the jump_size. If the difference is greater than
            jump_size_thresh, the point is a candidate weight jump. Default is 0.8 kg, so that a
            jump_size minimum size is 0.8 kg. Less than this, it is not considered a weight jump.

        drop_tmpcols: If True, drop all auxiliary columns in the output dataframe.


    Returns:
        df_weights_tagged: The output dataframe with the added columns:
            'is_jump': which is set to True if the related time sample corresponds to a weight jump;
            'jump_size': the size of the weight jump, calculated from the median difference of 5
                points before and after the weight jump.
            'jump_baseline_left': the left baseline of the jump calculated on 5 points before the
                jump.
            'jump_baseline_right': the right baseline of the jump calculated on 5 points after the
                jump.
            'jump_max': the maximux of the 5 points centered on the jump.
            'jump_min': the minimum of the 5 points centered on the jump.
            'jump_center': the center of the jump. It is just ('jump_baseline_right' +
                'jump_baseline_left')/2.
            'jump_step_factor': a factor that goes from 0 to +inf. It is close to zero if the
                jump resembles a step function, otherwise it is >> 0. It is calulated on a time
                window of either 180 minutes or 360 minutes.
            'jump_slope_factor': the slope of the linear fit of a window centered on the jump. It
                is calulated on a time window of either 180 minutes or 360 minutes.

            |      | acquired_at               |   total_weight_value | is_jump   |   jump_size |   jump_baseline_left |   jump_baseline_right |   jump_max |   jump_min |   jump_center |   jump_step_factor_180 |   jump_step_factor_360 |   jump_slope_factor_180 |   jump_slope_factor_360 |
            |-----:|:--------------------------|---------------------:|:----------|------------:|---------------------:|----------------------:|-----------:|-----------:|--------------:|-----------------------:|-----------------------:|------------------------:|------------------------:|
            | 1524 | 2019-02-01 18:00:00+00:00 |              35.1714 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |
            | 1525 | 2019-02-01 18:30:00+00:00 |              35.1714 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |
            | 1526 | 2019-02-01 19:00:00+00:00 |              36.3828 | True      |      1.2114 |              35.1714 |               36.3828 |    36.4114 |    35.1714 |       35.7771 |              0.0922064 |               0.212966 |             0.000165076 |             0.000106627 |
            | 1527 | 2019-02-01 19:30:00+00:00 |              36.3828 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |
            | 1528 | 2019-02-01 20:00:00+00:00 |              36.4114 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |
    """

    df = df_weights  # Rename df for convenience
    px = "TMPCOL_"  # Columns prefix for all of the auxiliary columns

    assert isinstance(
        df.index, pd.DatetimeIndex
    ), "The input df_weights must have a pd.DatetimeIndex index."

    assert ("total_weight_value" in df) and (
        is_numeric_dtype(df["total_weight_value"])
    ), "The input df_weights must have a 'total_weight_value' numeric column."

    diff = np.array(df.index[1:] - df.index[:-1])[:-1].astype(int)
    assert np.allclose(
        diff, pd.Timedelta("30min").to_timedelta64().astype(int), rtol=0.000_000_001
    ), "The input df_weights.index must have a sampling frequency of 30min."

    # Set column to be considered as 'signal' for all of the algorithm calculations
    df[f"{px}signal"] = df["total_weight_value"]

    # Calculate the first derivative
    df[f"{px}signal_shift+1"] = df[f"{px}signal"].shift(periods=1)
    df[f"{px}der"] = df[f"{px}signal"] - df[f"{px}signal_shift+1"]

    # PART1: Logic for small jumps.
    # In the case of small jumps, consider both the derivative and the
    # median signal. A point is set as a jump only if both signals
    # are over a threshold. The size of the jump is given by the baseline
    # differences.
    df[f"{px}is_der_over_thresh_small"] = df[f"{px}der"].abs() > der_thresh_small
    # Define a derivative spike
    df[f"{px}is_der_spike"] = df[f"{px}is_der_over_thresh_small"]

    # Calculate a rolling median on a window of 5 data points
    median_window = 5
    df[f"{px}median"] = df[f"{px}signal"].rolling(median_window).median()
    df[f"{px}median_right"] = df[f"{px}median"].shift(periods=-3)
    df[f"{px}median_left"] = df[f"{px}median"].shift(periods=0)
    df[f"{px}median_diff"] = df[f"{px}median_right"] - df[f"{px}median_left"]
    # Apply threshold on median_diff (the jump_size)
    df[f"{px}is_median_diff_over_thresh"] = df[f"{px}median_diff"].abs() > jump_size_thresh
    # Calculate the weights_jump
    df["is_jump"] = df[f"{px}is_der_spike"] & df[f"{px}is_median_diff_over_thresh"]

    # PART2: Logic for big jumps.
    # In the case of big jumps, consider only the derivative. If it is
    # greater than the threshold for big jumps, set that point as jump (even though
    # it was previously suppressed). The size of the jump is given by the
    # derivative.
    df[f"{px}is_der_over_thresh_big"] = df[f"{px}der"].abs() > der_thresh_big
    # Update "is_jump" for the big jumps
    df.loc[
        df[f"{px}is_der_over_thresh_big"],
        f"is_jump",
    ] = True

    # CALCULATION OF JUMP SIZE AND BASELINE.
    # Calculate the jump size and baseline. For all types of jumps, the jump size
    # is the derivative calculated on that point, while the baselines are calculated
    # from the median_left and meadian_right signal.
    # NOTE: with this assumption, the size of the jump is no more an indication of
    # what an human would assign for the "jump event". For that, maybe a new
    # type of algorithm is needed.
    df[f"{px}is_jump_float"] = df["is_jump"].astype(float).replace(0.0, np.nan)
    df["jump_baseline_left"] = df[f"{px}is_jump_float"] * df[f"{px}median_left"]
    df["jump_baseline_right"] = df[f"{px}is_jump_float"] * df[f"{px}median_right"]
    # Calculate the jump size
    df["jump_size"] = df[f"{px}is_jump_float"] * df[f"{px}der"]

    # LAST PART: Calculate some features for each jump
    # Calculate the max and min on a window of 5 data points
    extrema_window = 5
    df["jump_max"] = (
        df[f"{px}signal"].rolling(extrema_window, center=True).max() * df[f"{px}is_jump_float"]
    )
    df["jump_min"] = (
        df[f"{px}signal"].rolling(extrema_window, center=True).min() * df[f"{px}is_jump_float"]
    )

    # Calculate the jump_center as the mean of the baselines
    df["jump_center"] = (df["jump_baseline_left"] + df["jump_baseline_right"]) / 2

    # Calculate the step_factor and the slope_factor for each jump
    df["jump_step_factor_180"] = np.nan
    df["jump_step_factor_360"] = np.nan
    df["jump_slope_factor_180"] = np.nan
    df["jump_slope_factor_360"] = np.nan
    for idx, row in df[df["is_jump"]].iterrows():
        center = (row["jump_baseline_left"] + row["jump_baseline_right"]) / 2
        half_jump_size = row["jump_size"] / 2
        abs_jump_size = np.abs(row["jump_size"])

        # Consider a slice of 3 hours (180 minutes), calculate the step_factor and the slope_factor
        df_slice_180 = df.loc[
            (df.index >= idx - pd.Timedelta("1.5 hours", tz=timezone.utc))  # type: ignore[operator, arg-type]
            & (df.index <= idx + pd.Timedelta("1.5 hours", tz=timezone.utc))  # type: ignore[operator, arg-type]
        ]
        # step factor (how close is the step to a step function)
        step_factor_180 = (
            half_jump_size - (df_slice_180["total_weight_value"] - center).abs()
        ).std() / abs_jump_size
        df.loc[idx, "jump_step_factor_180"] = step_factor_180  # type: ignore[index]
        # slope factor (the slope of the linear fit of the slice)
        x = df_slice_180["total_weight_value"].index.astype(np.int64) // 10**9  # type: ignore[operator]
        y = df_slice_180["total_weight_value"].values
        slope_factor_180 = matint.linear_interp_1d(x, y)[0][1]  # type: ignore[arg-type]
        df.loc[idx, "jump_slope_factor_180"] = slope_factor_180  # type: ignore[index]

        # Consider a slice of 6 hours (360 minutes), calculate the step_factor and the slope_factor
        df_slice_360 = df.loc[
            (df.index >= idx - pd.Timedelta("3.0 hours", tz=timezone.utc))  # type: ignore[operator, arg-type]
            & (df.index <= idx + pd.Timedelta("3.0 hours", tz=timezone.utc))  # type: ignore[operator, arg-type]
        ]
        # step factor (how close is the step to a step function)
        step_factor_360 = (
            half_jump_size - (df_slice_360["total_weight_value"] - center).abs()
        ).std() / abs_jump_size
        df.loc[idx, "jump_step_factor_360"] = step_factor_360  # type: ignore[index]
        # slope factor (the slope of the linear fit of the slice)
        x = df_slice_360["total_weight_value"].index.astype(np.int64) // 10**9  # type: ignore[operator]
        y = df_slice_360["total_weight_value"].values
        slope_factor_360 = matint.linear_interp_1d(x, y)[0][1]  # type: ignore[arg-type]
        df.loc[idx, "jump_slope_factor_360"] = slope_factor_360  # type: ignore[index]

    # Drop TMPCOLs if requested
    if drop_tmpcols:
        df.drop(columns=df.columns[df.columns.str.startswith(px)], inplace=True)

    # Rename output df
    df_weights_tagged = df

    return df_weights_tagged
