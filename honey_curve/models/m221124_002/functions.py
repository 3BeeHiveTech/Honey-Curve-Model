"""Auxiliary functions specific to the m221124_002 model."""


import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

import honey_curve.timeseries.outlier as timout


def calc_honey_curve(
    df_weights_tagged: pd.DataFrame,
    drop_tmpcols: bool = True,
) -> pd.DataFrame:
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
                        been tagged and several jump features are added. The minimal columns in
                        order for the algorithm to work are:
                        "total_weight_value" : the weights trace;
                        "jump_size": the size of the jump, with sign;
                        "jump_type": the jump classification, which can be either:
                            - "is_big_jump" (the jump is of type "Metti Arnia", "Togli Arnia", "Metti Melario", "Togli Melario")
                            - "is_small_jump" (the jump is small, but still a true jump)
                            - "no_type" (the jump is not really a jump, but just a normal small variation)

        |                           |   total_weight_value |   jump_size | jump_type   |
        |:--------------------------|---------------------:|------------:|:------------|
        | 2020-01-01 10:00:00+00:00 |              25.7372 |    nan      |             |
        | 2020-01-01 10:30:00+00:00 |              25.7372 |    nan      |             |
        | 2020-01-01 11:00:00+00:00 |              25.7372 |    nan      |             |
        | 2020-01-01 11:30:00+00:00 |              25.0229 |     -1.4743 | no_type     |
        | 2020-01-01 12:00:00+00:00 |              24.2629 |    nan      |             |
        | 2020-01-01 12:30:00+00:00 |              24.2629 |    nan      |             |
        | 2020-01-01 13:00:00+00:00 |              24.2629 |    nan      |             |

        drop_tmpcols: If True, drop all auxiliary columns in the output dataframe.


    Returns:
        df_weights_honey: A tagged version with the "honey_curve" as the last column.

        |                           |   total_weight_value |   jump_size | jump_type   |   honey_curve |
        |:--------------------------|---------------------:|------------:|:------------|--------------:|
        | 2020-01-01 10:00:00+00:00 |              25.7372 |    nan      |             |       -0.1428 |
        | 2020-01-01 10:30:00+00:00 |              25.7372 |    nan      |             |       -0.1428 |
        | 2020-01-01 11:00:00+00:00 |              25.7372 |    nan      |             |       -0.1428 |
        | 2020-01-01 11:30:00+00:00 |              25.0229 |     -1.4743 | no_type     |       -0.8571 |
        | 2020-01-01 12:00:00+00:00 |              24.2629 |    nan      |             |       -1.6171 |
        | 2020-01-01 12:30:00+00:00 |              24.2629 |    nan      |             |       -1.6171 |
        | 2020-01-01 13:00:00+00:00 |              24.2629 |    nan      |             |       -1.6171 |

    """

    df = df_weights_tagged  # Rename df for convenience
    px = "TMPCOL_"  # Columns prefix for all of the auxiliary columns

    assert isinstance(
        df.index, pd.DatetimeIndex
    ), "The input df_weights_tagged must have a pd.DatetimeIndex index."

    assert ("total_weight_value" in df) and (
        is_numeric_dtype(df["total_weight_value"])
    ), "The input df_weights_tagged must have a 'total_weight_value' numeric column."

    assert ("jump_size" in df) and (
        is_numeric_dtype(df["jump_size"])
    ), "The input df_weights_tagged must have a 'jump_size' numeric column."

    assert ("jump_type" in df) and (
        is_string_dtype(df["jump_type"])
    ), "The input df_weights_tagged must have a 'jump_type' string column."

    ## Calculate the start baseline
    delta = pd.Timedelta("6 hours")
    df_notnull = df[df["total_weight_value"].notnull()]
    start_baseline = df[(df.index < df_notnull.index.min() + delta)]["total_weight_value"].median()

    ## Calculate the jump_shift, removing the "no_type" jumps
    jump_types_to_remove = ["no_type"]
    df[f"{px}jump_size_filtered"] = df["jump_size"].copy()
    remove_jumps = df[(df["jump_type"].isin(jump_types_to_remove))]
    df.loc[remove_jumps.index, f"{px}jump_size_filtered"] = 0  # Set remove_jumps as zero size
    df[f"{px}jump_size_filtered"] = df[f"{px}jump_size_filtered"].fillna(
        0
    )  # Set all nan values as zero size jumps.
    # Calculate the "jump_shift" as the cumsum of the jumps.
    df[f"{px}jump_shift"] = df[f"{px}jump_size_filtered"].cumsum()

    ## Calculate the honey_curve
    df["honey_curve"] = df["total_weight_value"] - df[f"{px}jump_shift"] - start_baseline

    ## Remove outliers in the honey_curve via derivative, replace with median
    df_outlier = timout.detect_outliers_via_derivative(
        series=df["honey_curve"],
        der_thresh=5,
        timestep_thresh=pd.Timedelta("18 hours"),
        drop_tmpcols=True,
    ).rename(columns={"values": "honey_curve"})
    trace_median = df["honey_curve"].rolling("2 H", center=True).median()
    df.loc[df_outlier["is_outlier"], "honey_curve"] = trace_median[df_outlier["is_outlier"]]

    # #### New additional jump detection remotion in the honey_curve via derivative, replace with median
    # ## Jump detection algorithm
    # df_out = df[["honey_curve"]].rename(columns={"honey_curve": "x"})
    # df_out["x_der"] = df_out["x"].diff()
    # df_out["x_der_abs"] = df_out["x_der"].abs()
    # df_out["x_der_mean"] = df_out["x_der_abs"].rolling(11, center=True).mean()
    # df_out["x_der_over_mean"] = df_out["x_der_abs"] > df_out["x_der_mean"] * 5.0
    # df_out["x_der_over"] = df_out["x_der_abs"] > 0.5
    # df_out["is_outlier"] = df_out["x_der_over_mean"] & df_out["x_der_over"]
    # ## Remove the jumps from the honey_curve
    # new_jumps_index = df[df_out["is_outlier"]].index
    # df[f"{px}jump_size_filtered_2"] = 0
    # df.loc[new_jumps_index, f"{px}jump_size_filtered_2"] = df_out.loc[new_jumps_index, "x_der"]
    # df[f"{px}jump_shift_2"] = df[f"{px}jump_size_filtered_2"].cumsum()
    # ## Update the honey_curve
    # df["honey_curve"] = df["honey_curve"] - df[f"{px}jump_shift_2"]
    # #### END of New additional jump detection remotion

    # Drop TMPCOLs if requested
    if drop_tmpcols:
        df.drop(columns=df.columns[df.columns.str.startswith(px)], inplace=True)

    # Rename output df
    df_weights_honey = df

    return df_weights_honey
