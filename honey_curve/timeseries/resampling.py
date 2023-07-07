"""
Resampling functions for time series.
"""
from datetime import timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd


def resample_timeseries_dataframe(
    df_in: pd.DataFrame,
    start: str,
    end: str,
    freq: Optional[str] = None,
    periods: Optional[int] = None,
    method: str = "linear",
    mode: str = "constant",
    cval: float = np.nan,
    tz: timezone = timezone.utc,
    timestep_thresh: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """Resample an input timeseries dataframe with the given method. If the threshold is set, fill
    only the points there the distance from one sample to the next is less than the threshold.

    NOTE: Since some transformation apply "inplace", the original df_in dataframe will be
    transformed as well.

    TODO: write a version with inplace=False.

    TODO: write a version with drop_tmpcols=False.

    Args:
        df_in: Input timeseries dataframe to resample with a pd.DatetimeIndex index.
        start: Start of the new DatetimeIndex as string (e.g. "2020-01-01").
        end: End of the new DatetimeIndex as string (e.g. "2020-12-31").
        freq: Frequency of the resampled DatetimeIndex. Examples are: "1S" (1 second),
            "1H" (1 hour), "1D" (1 day), "1M" (1 month). See more at:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        periods: Number of new resample samples for the output dataframe. To be specified when the
            "freq" is not set.
        method: Use the given method to fill in the nan data. Avaiable methods are:
            "linear" : use linear interpolation;
            "ffill": use forward filling.
            The method is passed directly to the pandas.DataFrame.interpolate function.
        mode: If 'constant', fill past the edges of the input with 'cval'. If 'nearest', fill them
            with the last not null sample value.
        cval: Value to fill past the edges of the input if mode is 'constant'. Default is np.nan.
        tz: Selected timezone. Defaults to timezone.utc.
        timestep_thresh: pd.Timedelta timestep_thresh on the distance between one sample and the
            next. If the distance is greater than the timestep_thresh, replace the values with
            np.nan.

    Returns:
        df_out: Output timeseries dataframe resampled with reindex.
    """

    assert isinstance(
        df_in.index, pd.DatetimeIndex
    ), "The input df_in must have a pd.DatetimeIndex index."

    methods = ["linear", "ffill"]
    if method not in methods:
        raise ValueError(f"'method' must be one of: {methods}.")

    if not ((freq is not None) ^ (periods is not None)):
        raise ValueError(
            "Resample the dataframe by specifying either the 'freq' parameter or the"
            + " 'periods' parameter."
        )

    modes = ["constant", "nearest"]
    if mode not in modes:
        raise ValueError(f"'mode' must be one of: {modes}.")

    # Save the original df_in columns
    columns_in = df_in.columns.copy()

    # Drop the rows with only nan data
    df_in.drop(df_in.index[pd.isnull(df_in).all(axis=1)], axis=0, inplace=True)

    # Add the start and end timestamps as an index if they are not included already. Include also
    # the same start and end with a 50 microseconds delay, because the pandas resampling function
    # works in mysterious ways.
    reindex = pd.date_range(start=start, end=end, periods=2, inclusive="both", tz=tz)
    reindex2 = pd.date_range(
        start=reindex[0] - timedelta(microseconds=50),  # type: ignore[arg-type, operator]
        end=reindex[1] + timedelta(microseconds=50),  # type: ignore[arg-type, operator]
        periods=2,
        inclusive="both",
        tz=tz,
    )
    df_in_reindex = df_in.reindex(df_in.index.union(reindex))
    df_in_reindex = df_in_reindex.reindex(df_in_reindex.index.union(reindex2))
    df_in_reindex = df_in_reindex.sort_index()

    # Calculate the difference between samples in timedelta.
    df_in_reindex["TMPCOL_diff"] = pd.concat(
        [
            pd.Series(df_in_reindex.index[1:] - df_in_reindex.index[:-1]),
            pd.Series(pd.Timedelta(np.nan)),
        ]
    ).to_list()

    # If periods is specified, resample with timedelta given by the specified timestep.
    if periods is not None:
        timestep = (reindex[1] - reindex[0]).to_pytimedelta() / (periods - 1)  # type: ignore[union-attr, operator]
        resampling_rule = timestep
    # If freq is specified, resample with the given frequency.
    elif freq is not None:
        resampling_rule = freq
    else:
        raise ValueError("This should never happen.")

    # Resample dataframe. Use the pandas.DataFrame.resample method under the hood.
    # Resample the dataframe with a given frequency, starting from start. Use the
    # pandas.DataFrame.resample convenience method.
    if method == "linear":
        df_resampled = (
            df_in_reindex[columns_in]
            .resample(resampling_rule, origin=pd.Timestamp(start, tz=tz))
            .mean()
            .interpolate(method="linear", limit_direction="forward", axis=0)
        )
    elif method == "ffill":
        df_resampled = (
            df_in_reindex[columns_in]
            .resample(resampling_rule, origin=pd.Timestamp(start, tz=tz))
            .ffill()
        )

    # Resample "TMPCOL_diff" always with ffill. This will be used if the timestep_thresh is set.
    df_resampled["TMPCOL_diff"] = (
        df_in_reindex["TMPCOL_diff"]
        .resample(resampling_rule, origin=pd.Timestamp(start, tz=tz))
        .ffill()
    )
    # Fix the points that are over the last time sample as np.nan
    df_resampled.loc[
        df_resampled.index >= (df_in.index.max() + timedelta(microseconds=50)), columns_in
    ] = np.nan
    # Fix the points that are before the first time sample as np.nan
    df_resampled.loc[
        df_resampled.index <= (df_in.index.min() - timedelta(microseconds=50)), columns_in
    ] = np.nan

    # Filter out the values before start and after end. Allow for en error of 50 microseconds in
    # the time calculations.
    df_resampled.drop(
        df_resampled.index[
            (df_resampled.index <= (reindex[0] - timedelta(microseconds=50)))  # type: ignore[operator]
            | (
                df_resampled.index >= (reindex[1] + timedelta(microseconds=50))  # type: ignore[operator]
            )
        ],
        axis=0,
        inplace=True,
    )

    # Apply the fill out the edges as spedified.
    if mode == "nearest":
        df_resampled[columns_in] = df_resampled[columns_in].ffill().bfill()
    elif mode == "constant":
        df_resampled[columns_in] = df_resampled[columns_in].fillna(cval)

    # Drop TMPCOL for the original df_in. Use the inplace method to delete it in the original df_in
    # (side effect)
    df_in_reindex.drop(columns=["TMPCOL_diff"], inplace=True)

    # Fix the points that are greater than the timestep_thresh
    if timestep_thresh:
        # Set the timestep_thresh (excluding np.nan)
        df_resampled["TMPCOL_timestep_thresh"] = ~(df_resampled["TMPCOL_diff"] < timestep_thresh)
        # Apply timestep_thresh
        df_resampled.loc[df_resampled["TMPCOL_timestep_thresh"], columns_in] = np.nan
        # Remove TMPCOL for the df_resampled
        df_resampled.drop(columns=["TMPCOL_timestep_thresh"], inplace=True)

    # Remove TMPCOL for the df_out
    df_resampled.drop(columns=["TMPCOL_diff"], inplace=True)

    return df_resampled
