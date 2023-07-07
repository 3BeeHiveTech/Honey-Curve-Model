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