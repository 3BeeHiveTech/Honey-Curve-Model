# Model m221124_001
Weight tagging model (algorithm) for a weights time series.

**NOTE**: This model is a copy of the m `m221101_001` model with some preprocessing/algorithm
changes in order to improve the jump detection algorithm (hopefully).

## Description
This model is a first version of a weight jump detection algorithm, loosly based on Marco Croci's
weight tagging algorithm. It works by analyzing the "total_weight_value" time series and it will 
find all relevant weight jumps and their size in kg by looking and the derivative of the resampled
signal.

## Usage
Given a time series **df_weight_raw** of the type:

|    | acquired_at         |   total_weight_value |
|---:|:--------------------|---------------------:|
|  0 | 2019-01-01 00:08:14 |              28.4    |
|  1 | 2019-01-01 01:08:14 |              28.3542 |
|  2 | 2019-01-01 02:08:14 |              28.3543 |
|  3 | 2019-01-01 03:08:14 |              28.3486 |
|  4 | 2019-01-01 04:08:14 |              28.2857 |

which comes from a select of `the threebee_production.weights` table, we can apply the tagging 
algorithm to find all the detected weight jumps by:

```python
from honey_curve.models.y2022_11.m221124_001.loader import M221124_001 

model = M221124_001()
df_weights_tagged = model.preprocess_and_tag(df_weights_raw)
```

The output of this function is:
- **df_weight_tagged**: A preprocessed version of the input dataframe which has been resampled to 
    `30 min` frequency, and where all of the weight jump events are tagged, along with other jump 
    features.

|      | acquired_at               |   total_weight_value | is_jump   |   jump_size |   jump_baseline_left |   jump_baseline_right |   jump_max |   jump_min |   jump_center |   jump_step_factor_180 |   jump_step_factor_360 |   jump_slope_factor_180 |   jump_slope_factor_360 |
|-----:|:--------------------------|---------------------:|:----------|------------:|---------------------:|----------------------:|-----------:|-----------:|--------------:|-----------------------:|-----------------------:|------------------------:|------------------------:|
| 1524 | 2019-02-01 18:00:00+00:00 |              35.1714 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |
| 1525 | 2019-02-01 18:30:00+00:00 |              35.1714 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |
| 1526 | 2019-02-01 19:00:00+00:00 |              36.3828 | True      |      1.2114 |              35.1714 |               36.3828 |    36.4114 |    35.1714 |       35.7771 |              0.0922064 |               0.212966 |             0.000165076 |             0.000106627 |
| 1527 | 2019-02-01 19:30:00+00:00 |              36.3828 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |
| 1528 | 2019-02-01 20:00:00+00:00 |              36.4114 | False     |    nan      |             nan      |              nan      |   nan      |   nan      |      nan      |            nan         |             nan        |           nan           |           nan           |


Note that you can also apply separately the preprocessing and the tagging algorithms as:

```python
from honey_curve.models.y2022_11.m221124_001.loader import M221124_001 

model = M221124_001()
df_weights_preproc = model.preprocess(df_weights_raw)
df_weights_tagged = model.tag(df_weights_preproc)
```

See the model docs for more info.