"""
Tests of the timeseries module.
"""
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

from honey_curve.timeseries import resampling as timres
from honey_curve.timeseries import test as timtst


def test_resample_timeseries_dataframe() -> None:
    """Test that resampling the a test dataframe of the type df_series_v01:
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
    using the function:
    :func:`~honey_curve.timeseries.resampling.resample_timeseries_dataframe`
    we will get a resampled df_resampled with the following properties:
    - The minimum of the resampled df is the same as start;
    - The maximum of the resampled df is the same as end;
    - The new index has time steps which are equal or almost equal;
    - The interpolated columns col_1 and col_2 have increasing values;
    """
    df_test = timtst.create_dummy_df_series_v01(
        start="2020-01-01", end="2020-12-31", totsamples=24, nsamples=12, nansamples=4
    )

    start = "2020-02-17"
    end = "2020-12-15"
    periods = 12
    freq = None
    # Resample with linear method
    df_resampled_linear = timres.resample_timeseries_dataframe(
        df_test, start=start, end=end, freq=freq, periods=periods, method="linear", mode="nearest"
    )
    # Resample with ffill method
    df_resampled_ffill = timres.resample_timeseries_dataframe(
        df_test, start=start, end=end, freq=freq, periods=periods, method="ffill", mode="nearest"
    )

    date_min, date_max = pd.date_range(
        start=start, end=end, periods=2, inclusive="both", tz=timezone.utc
    )

    # The number of samples is given by 'periods'
    assert df_resampled_linear.shape[0] == periods
    assert df_resampled_ffill.shape[0] == periods

    # The minimum of the resampled df is the same as start (with error = 50 microseconds)
    assert (df_resampled_linear.index.min() - date_min) <= timedelta(microseconds=50)
    assert (df_resampled_ffill.index.min() - date_min) <= timedelta(microseconds=50)

    # The maximum of the resampled df is the same as end (with error = 50 microseconds)
    assert (df_resampled_linear.index.max() - date_max) <= timedelta(microseconds=50)
    assert (df_resampled_ffill.index.max() - date_max) <= timedelta(microseconds=50)

    # The new index has time steps which are equal or almost equal. Apart from the last one
    diff = np.array(df_resampled_linear.index[1:] - df_resampled_linear.index[:-1])[:-1].astype(
        int
    )
    assert np.allclose(diff, diff[0], rtol=0.00000001)
    diff = np.array(df_resampled_ffill.index[1:] - df_resampled_ffill.index[:-1])[:-1].astype(int)
    assert np.allclose(diff, diff[0], rtol=0.00000001)

    # The interpolated columns col_1 and col_2 have increasing values
    assert (
        (
            df_resampled_linear["col_1"][1:].fillna(0).to_numpy()
            - df_resampled_linear["col_1"][:-1].fillna(0).to_numpy()
        )
        >= 0
    ).all()
    assert (
        (
            df_resampled_ffill["col_1"][1:].fillna(0).to_numpy()
            - df_resampled_ffill["col_1"][:-1].fillna(0).to_numpy()
        )
        >= 0
    ).all()
