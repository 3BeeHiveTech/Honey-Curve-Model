"""
Auxiliary functions to be used when testing the timeseries functions (either with pytest or via 
notebooks).
"""
from datetime import timezone

import numpy as np
import pandas as pd


def create_dummy_df_series_v01(
    start: str = "2020-01-01",
    end: str = "2020-12-31",
    totsamples: int = 24,
    nsamples: int = 12,
    nansamples: int = 4,
) -> pd.DataFrame:
    """Create a dummy df_series dataframe object to be used for testing purposes.

    The dataframe is of the type:
        |                               |    col_1 |    col_2 |
        |:------------------------------|---------:|---------:|
        | 2020-01-01 00:00:00           | nan      | nan      |
        | 2020-02-17 14:36:31.304347826 |  13.0435 |  26.087  |
        | 2020-03-04 11:28:41.739130435 |  17.3913 |  34.7826 |
        | 2020-05-22 19:49:33.913043478 |  39.1304 |  78.2609 |
        | 2020-06-07 16:41:44.347826088 | nan      | nan      |
        | 2020-06-23 13:33:54.782608696 |  47.8261 |  95.6522 |
        | 2020-07-09 10:26:05.217391304 | nan      | nan      |
        | 2020-10-28 12:31:18.260869568 |  82.6087 | 165.217  |
        | 2020-11-13 09:23:28.695652176 |  86.9565 | 173.913  |
        | 2020-11-29 06:15:39.130434784 |  91.3043 | 182.609  |
        | 2020-12-15 03:07:49.565217392 |  95.6522 | 191.304  |
        | 2020-12-31 00:00:00           | nan      | nan      |
    where the position of the nan samples is random, the index goes from start to end and the two
    columns come from a linspace, where the first goes from 0 to 100, and the second goes from 0 to
    200.

    Args:
        start: Starting date for the DatetimeIndex.
        end: Ending date for the DatetimeIndex.
        totsamples: Total samples before the random sampling.
        nsamples: Number of samples in the output dataframe.
        nansamples: Number of samples in the output dataframe that have been set as np.nan.

    Returns:
        df_series: A dummy dataframe object with DatetimeIndex.
    """

    df_series = (
        pd.DataFrame(
            data={
                "col_1": np.linspace(0, 100, totsamples),
                "col_2": np.linspace(0, 100, totsamples) * 2,
            },
            index=pd.date_range(
                start=start, end=end, periods=24, inclusive="both", tz=timezone.utc
            ),
        )
        .sample(nsamples)
        .sort_index()
    )
    df_series.loc[df_series.sample(nansamples).index] = np.nan  # type: ignore[call-overload]

    # Add start and end back if they have been removed
    reindex = pd.date_range(start=start, end=end, periods=2, inclusive="both", tz=timezone.utc)
    df_series = df_series.reindex(df_series.index.union(reindex))
    df_series = df_series.sort_index()

    return df_series
