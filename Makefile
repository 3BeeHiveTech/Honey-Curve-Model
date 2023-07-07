.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")


.PHONY: help
help:                             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: show
show:                             ## Show the current environment.
	@echo "Current environment:"
	@echo "Running using $(ENV_PREFIX)"
	@$(ENV_PREFIX)python -V
	@$(ENV_PREFIX)python -m site


.PHONY: env_ipykernel
env_ipykernel:                    ## Install the current venv as an ipython kernel for jupyter.
	$(ENV_PREFIX)python -m ipykernel install --name honey_curve --user
	@echo "Installed ipython kernel honey_curve"


.PHONY: fmt
fmt:                              ## Format code using black, isort, mypy and autoflake.
	$(ENV_PREFIX)isort honey_curve/
	$(ENV_PREFIX)isort tests/
	$(ENV_PREFIX)black -l 99 honey_curve/
	$(ENV_PREFIX)black -l 99 tests/
	$(ENV_PREFIX)python -m autoflake --in-place -r honey_curve/
	$(ENV_PREFIX)python -m autoflake --in-place -r tests/


.PHONY: test
test:                             ## Run tests and generate coverage report.
	$(ENV_PREFIX)pytest -v -l --tb=short --maxfail=1 tests/


.PHONY: clean
clean:                            ## Clean unused files.
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build
