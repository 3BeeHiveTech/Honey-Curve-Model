# Honey-Curve-Model
Repo of the Honey-Curve Model that can clean the raw data from the threbee_production.weights table
in order to generate the Honey curve for a given hive.

This is a "spin off" model from the [Generali-Parametrica-Alveari](https://github.com/3BeeHiveTech/Generali-Parametrica-Alveari)
repository. Specifically, the model here reported is the [M221124_002 model](https://github.com/3BeeHiveTech/Generali-Parametrica-Alveari/tree/main/honey_curve/models/y2022_11/m221124_002#model-m221124_002).



## Installation (testing)
To install the package in development mode, you can use the following commands to clone the repo:

```bash
git clone git@github.com:3BeeHiveTech/Honey-Curve-Model.git
cd Honey-Curve-Model
```

Next you can install the package in a local conda environment named `honey_curve` when inside the
`Honey-Curve-Model` folder like so:

```bash
conda create -n honey_curve python=3.10
conda activate honey_curve
(honey_curve) pip install -e ".[test]"
```

This will install the current package and its dependencies in editable mode, with all of the test
requirements (used to format the code).

Now you can run the `make help` to see what are the avaiable commands. Remember to run each command
with the `(honey_curve)` environment always activated.

```bash
(honey_curve) make help  # show the make help
```


## Installation (production)
To install the package in production mode, you can use the following command to create a new conda
environment and install the package from git:

```bash
conda create -n honey_curve python=3.10
conda activate honey_curve
(honey_curve) pip install git+https://github.com/3BeeHiveTech/Honey-Curve-Model.git
```

This will install the `honey_curve` package directly from the git repository.


## Configuring the italian language package

To add the locale it_IT to an Ubuntu-like system you can run:

```bash
sudo apt-get install language-pack-it
sudo locale-gen it_IT
sudo update-locale
```

This is not strictly needed for the code but the loading notebooks requires on the italian local
when displaying the honey summary by month.


## Configuring honey_curve
Before using it, the package `honey_curve` must be configured by setting a config file at
`~/.config/honey_curve/config.env`. This file must contain all of the setted variables that are 
needed as configuration, which are:

```.env
# CREDENTIALS
# 3Bee production database
CRED_3BEE_PROD_DB_NAME="xxx"
CRED_3BEE_PROD_DB_USER="xxx.yyy"
CRED_3BEE_PROD_DB_PASSWORD="zzz"
CRED_3BEE_PROD_DB_URL="xxx.yyy.eu-central-1.rds.amazonaws.com"
CRED_3BEE_PROD_DB_PORT="pppp"
```

## Usage
**TODO ...**


## Notes on the BareboneHistGradientBoostingClassifier model
In order to have fewer dependencies and a lighter repository, the `HistGradientBoostingClassifier` 
`sklearn` model has been converted into a `BareboneHistGradientBoostingClassifier`, following a 
procedure similar to what is described here: [Speeding up a sklearn model pipeline to serve single predictions with very low latency](https://blog.telsemeyer.com/2020/08/13/speeding-up-a-sklearn-model-pipeline-to-serve-single-predictions-with-very-low-latency/).

The code for the light version is in the `sklearn_light` module of this repo. The barebone version
of the model has been generated from the full model with the following code:

```python
from honey_curve.models.m221124_002.loader import M221124_002
from honey_curve.sklearn_light.gradient_boosting import BareboneHistGradientBoostingClassifier

model = M221124_002()

bmodel = BareboneHistGradientBoostingClassifier(
    classes_=model.jump_classification_model.classes_,
    _loss=model.jump_classification_model._loss,
    _n_features=model.jump_classification_model._n_features,
    n_trees_per_iteration_=model.jump_classification_model.n_trees_per_iteration_,
    _baseline_prediction=model.jump_classification_model._baseline_prediction,
    _predictors=model.jump_classification_model._predictors,
    _bin_mapper=model.jump_classification_model._bin_mapper,
    _check_feature_names=model.jump_classification_model._check_feature_names,
)

# Save model
path_to_model = "m221124_002_barebone.pickle"
with open(path_to_model, "wb") as f:
    pickle.dump(bmodel, f)
```

In the most recent version, the `M221124_002` loader has been modified so that the barebone model 
is loaded instad. 

A check on the equivalence of the two models is performed by: `test_BareboneHistGradientBoostingClassifier()`.