# Model m221124_002
Weight jump classification model (algorithm) for a weights time series.

**NOTE**: This model is a copy of the `m221102_001` model but with slight modification to improve 
(hopefully) the jump detection and classification algorithms. Have a look at the report notebooks
to see the model improvement

**CHANGES**:
- Jump detection algorithm has a different deduplication of duplicate peaks, by looking at only
consecutive values;
- Jump detection algorithm has a new derivative threshold at 0.75 kg.

## Description
This model is a first version of a jump classification algorithm, which will first find all 
possible weight jumps in a weight time series, and then it will tag all jumps with the column
`jump_type` which can be either of three classes:
- **is_big_jump** (the jump is of type "Metti Arnia", "Togli Arnia", "Metti Melario", "Togli Melario")
- **is_small_jump** (the jump is small, but still a true jump)
- **no_type** (the jump is not really a jump, but just a normal small variation)

Note that this model depend on the **M221124_001**, which is the jump detection model/algorithm 
used to find all possible weight jumps in the time series. The classification will later be 
computed by a `HistGradientBoostingClassifier` trained on the following jump features:

```python
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
```

## Usage
Given a time series **df_weight_raw** of the type:

|    | acquired_at         |   total_weight_value |
|---:|:--------------------|---------------------:|
|  0 | 2019-01-01 00:08:14 |              28.4    |
|  1 | 2019-01-01 01:08:14 |              28.3542 |
|  2 | 2019-01-01 02:08:14 |              28.3543 |
|  3 | 2019-01-01 03:08:14 |              28.3486 |
|  4 | 2019-01-01 04:08:14 |              28.2857 |

which comes from a select of `the threebee_production.weights` table, we can apply the 
classification algorithm to find all the detected weight jumps by:

```python
from honey_curve.models.y2022_11.m221124_002.loader import M221124_002 

model = M221124_002()
df_weights_tagged = model.preprocess_and_tag(df_weights_raw)
```

The output of this function is:
- **df_weight_tagged**: A preprocessed version of the input dataframe which has been resampled to 
    `30 min` frequency, and where all of the weight jump events are tagged, along with other jump 
    features. A `jump_type` classification column is added in the end which classifies the jumps 
    as **is_big_jump**, **is_small_jump**, or **no_type**.

|                           |   total_weight_value | is_jump   |   jump_size |   jump_baseline_left |   jump_baseline_right |   jump_max |    jump_min |   jump_center |   jump_step_factor_180 |   jump_step_factor_360 |   jump_slope_factor_180 |   jump_slope_factor_360 | jump_type   |
|:--------------------------|---------------------:|:----------|------------:|---------------------:|----------------------:|-----------:|------------:|--------------:|-----------------------:|-----------------------:|------------------------:|------------------------:|:------------|
| 2019-03-03 17:30:00+00:00 |            0.0571429 | False     |    nan      |          nan         |              nan      |   nan      | nan         |        nan    |          nan           |          nan           |             nan         |            nan          |             |
| 2019-03-03 18:00:00+00:00 |            0.0571429 | False     |    nan      |          nan         |              nan      |   nan      | nan         |        nan    |          nan           |          nan           |             nan         |            nan          |             |
| 2019-03-03 18:30:00+00:00 |           27.1828    | True      |     27.1257 |            0.0571429 |               27.1828 |    27.1828 |   0.0571429 |         13.62 |            0.000160239 |            0.000565211 |               0.0032299 |              0.00174138 | is_big_jump |
| 2019-03-03 19:00:00+00:00 |           27.1828    | False     |    nan      |          nan         |              nan      |   nan      | nan         |        nan    |          nan           |          nan           |             nan         |            nan          |             |
| 2019-03-03 19:30:00+00:00 |           27.1828    | False     |    nan      |          nan         |              nan      |   nan      | nan         |        nan    |          nan           |          nan           |             nan         |            nan          |             |


Note that you can also apply separately the preprocessing and the tagging plus classification 
algorithms as:

```python
from honey_curve.models.y2022_11.m221124_002.loader import M221124_002

model = M221124_002()
df_weights_preproc = model.preprocess(df_weights_raw)
df_weights_tagged = model.tag(df_weights_preproc)
```

See the model docs for more info.