# Honey-Curve-Model
Repo of the Honey-Curve Model that can clean the raw data from the threbee_production.weights table
in order to generate the Honey curve for a given hive.

This is a "spin off" model from the [Generali-Parametrica-Alveari](https://github.com/3BeeHiveTech/Generali-Parametrica-Alveari)
repository. Specifically, the model here reported is the [M221124_002 model](https://github.com/3BeeHiveTech/Generali-Parametrica-Alveari/tree/main/honey_curve/models/y2022_11/m221124_002#model-m221124_002).



## Installation (testing)

To add the locale it_IT (needed for the notebooks)

```bash
sudo apt-get install language-pack-it
sudo locale-gen it_IT
sudo update-locale
```


**TODO ...**

```bash
conda create -n honey_curve python=3.10
```


```bash
pip install -e ".[test]"
```

All of the the MAKE commands MUST be run with the `.venv` virtual environment activated.

You can now run:
```bash
make help  # show the make help
```
to see other avaible commands for updating the envirnoment, formatting and launching the tests.


## Installation (production)
**TODO ...**


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