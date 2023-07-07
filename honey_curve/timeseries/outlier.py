"""
Outlier detection functions for time series.
"""
from typing import Optional

import numpy as np
import pandas as pd


def detect_outliers_via_derivative(
    series: pd.Series,
    der_thresh: float = 5,
    timestep_thresh: Optional[pd.Timedelta] = None,
    drop_tmpcols: bool = True,
) -> pd.DataFrame:
    """Detect the outliers of an input timeseries. Return a dataframe with the input series and the
    "is_outlier" boolean column.

    NOTE on the algorithm: A point is defined as an outlier if the following conditions are true:
    - The derivative of its neighbor at position i-1 is lower than timestep_thresh;
    - The derivative at its position i is higher than the timestep_thresh, either positive or
        negative;
    - The derivative at its position i+1 is higher than the timestep_thresh, either positive or
        negative, and with opposite sign with respect to position i+1;
    - The derivative of its at position i+2 is lower than timestep_thresh;
    This guaranties that the point is an isolated outlier which either jumps up and then down over
    the threshold, or jumps down and then up over the threshold, while the previous and next
    samples both have low derivatives.

    Args:
        series: Input timeseries to check. It has to have a pd.DatetimeIndex index.
        der_thresh: Theshold on the derivative. Basically, how high or low the jump needs to be in
            order to be considered an outlier.
        timestep_thresh: pd.Timedelta timestep_thresh on the distance between one sample and the
            next. If the distance is greater than the timestep_thresh in both directions, the given
            point is not considered an outlier in any case.
        drop_tmpcols: If True, drop all auxiliary columns in the output dataframe.
    """
    assert isinstance(series, pd.Series), "The input series must be a pd.Series."

    assert isinstance(
        series.index, pd.DatetimeIndex
    ), "The input series must have a pd.DatetimeIndex index."

    px = "TMPCOL_"  # Columns prefix for all of the auxiliary columns

    # Calculate the derivative of neighboring points
    series_np = series.to_numpy()
    series_np_der = np.concatenate([np.array([0]), series_np[1:] - series_np[:-1]])

    df = pd.DataFrame(
        {
            f"values": series_np,
            f"{px}der": series_np_der,
        },
        index=series.index,
    )
    # Apply threshold on derivatives
    df[f"{px}is_der_over_thresh_up"] = df[f"{px}der"] > der_thresh
    df[f"{px}is_der_over_thresh_down"] = df[f"{px}der"] < -der_thresh
    is_der_over_thresh_up_np = df[f"{px}is_der_over_thresh_up"].to_numpy()
    is_der_over_thresh_down_np = df[f"{px}is_der_over_thresh_down"].to_numpy()

    # Calculate logical conditions based on derivatives, for the steps:
    # i-1 (suffix m1, as in "minus one")
    # i: the current step
    # i+1 (suffix p1, as in "plus one")
    # i+2 (suffix p2, as in "plus two")
    df[f"{px}is_der_over_thresh_xor"] = np.logical_xor(
        df[f"{px}is_der_over_thresh_up"], df[f"{px}is_der_over_thresh_down"]
    )
    df[f"{px}is_der_over_thresh_xor_p1"] = np.concatenate(
        [df[f"{px}is_der_over_thresh_xor"].to_numpy()[1:], np.array([False])]
    )
    df[f"{px}is_der_over_thresh_up_m1"] = np.concatenate(
        [np.array([False]), is_der_over_thresh_up_np[:-1]]
    )
    df[f"{px}is_der_over_thresh_up_p2"] = np.concatenate(
        [is_der_over_thresh_up_np[2:], np.array([False, False])]
    )
    df[f"{px}is_der_over_thresh_down_m1"] = np.concatenate(
        [np.array([False]), is_der_over_thresh_down_np[:-1]]
    )
    df[f"{px}is_der_over_thresh_down_p2"] = np.concatenate(
        [is_der_over_thresh_down_np[2:], np.array([False, False])]
    )

    # Set the condition to be outlier. This corresponds to:
    # - at i-1, the derivative is low
    # - at i, either the derivative is > thresh or < - thresh
    # - at i+1, same as step i, but with opposite sign
    # - at i+1, the derivative is low
    # This corresponds to a singular outlier in the time series.
    df["is_outlier"] = (
        (~df[f"{px}is_der_over_thresh_up_m1"])
        & (~df[f"{px}is_der_over_thresh_down_m1"])
        & (df[f"{px}is_der_over_thresh_xor"])
        & (df[f"{px}is_der_over_thresh_xor_p1"])
        & (~df[f"{px}is_der_over_thresh_up_p2"])
        & (~df[f"{px}is_der_over_thresh_down_p2"])
    )

    # Fix the
    if timestep_thresh:
        # Calculate the index difference
        index_diff = series.index[1:] - series.index[:-1]
        df[f"{px}index_diff_right"] = pd.concat(
            [
                pd.Series(index_diff),
                pd.Series(pd.Timedelta(np.nan)),
            ]
        ).to_list()

        df[f"{px}index_diff_left"] = pd.concat(
            [
                pd.Series(pd.Timedelta(np.nan)),
                pd.Series(index_diff),
            ]
        ).to_list()
        # Set the threshold (excluding np.nan)
        df[f"{px}index_diff_thresh"] = (df[f"{px}index_diff_right"] > timestep_thresh) | (
            df[f"{px}index_diff_left"] > timestep_thresh
        )
        # Apply threshold
        df.loc[df[f"{px}index_diff_thresh"], "is_outlier"] = False

    # Drop TMPCOLs if requested
    if drop_tmpcols:
        df.drop(columns=df.columns[df.columns.str.startswith(px)], inplace=True)

    return df


def detect_outliers_via_median(
    series: pd.Series,
    median_thresh: float = 2,
    median_window: Optional[pd.Timedelta] = pd.Timedelta("6 hours"),
    drop_tmpcols: bool = True,
) -> pd.DataFrame:
    """Detect the outliers of an input timeseries. Return a dataframe with the input series and the
    "is_outlier" boolean column.

    NOTE on the algorithm: A point is detected as an outlier if its value is greater than the "difference
    median" calculated from the left and from the right. The "difference median" is such
    that it displays some steps close to jumps in baseline, while it is flat when there are not jumps.
    By checking if the median centered on the sample is greater than a threshold calculated over the
    "difference median", one can spot the outliers when the curve is flat.

    NOTE replace median with mean for the left-right difference since it seems to work better this
    way.

    Args:
        series: Input timeseries to check. It has to have a pd.DatetimeIndex index.
        der_thresh: Theshold on the derivative. Basically, how high or low the jump needs to be in
            order to be considered an outlier.
        timestep_thresh: pd.Timedelta timestep_thresh on the distance between one sample and the
            next. If the distance is greater than the timestep_thresh in both directions, the given
            point is not considered an outlier in any case.
        drop_tmpcols: If True, drop all auxiliary columns in the output dataframe.
    """

    assert isinstance(series, pd.Series), "The input series must be a pd.Series."

    assert isinstance(
        series.index, pd.DatetimeIndex
    ), "The input series must have a pd.DatetimeIndex index."

    px = "TMPCOL_"  # Columns prefix for all of the auxiliary columns

    series_name = series.name
    df = series.to_frame()

    # Calculate the median with a window centered on the sample
    df[f"{px}median_center"] = df[series_name].rolling(median_window, center=True).median()  # type: ignore[call-overload]
    df[f"{px}median_center_diff"] = (df[series_name] - df[f"{px}median_center"]).abs()  # type: ignore[call-overload]

    # Calculate the right and left median by setting center="False" and then using the trick
    # of flipping the data to calculate the median in the other direction.
    df[f"{px}median_right"] = df[series_name].rolling(median_window, center=False).mean()  # type: ignore[call-overload]
    df[f"{px}median_left"] = np.flip(
        np.flip(df[series_name]).rolling(median_window, center=False).mean()  # type: ignore[call-overload]
    )
    df[f"{px}median_leftright_diff"] = (df[f"{px}median_left"] - df[f"{px}median_right"]).abs()

    # Calculate the "difference median" value. This is such that it is grater than zero only when
    # the signal is far away from a jump and is very different than the median calculated over
    # double the time window.
    df[f"{px}diff_median"] = (
        np.maximum(df[f"{px}median_center_diff"], df[f"{px}median_leftright_diff"])
        - df[f"{px}median_leftright_diff"]
    )

    # Calculate the outlier vie the threshold
    df["is_outlier"] = df[f"{px}diff_median"] > median_thresh

    # Drop TMPCOLs if requested
    if drop_tmpcols:
        df.drop(columns=df.columns[df.columns.str.startswith(px)], inplace=True)

    return df
