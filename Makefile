.PHONY: install install-dev install-notebooks format lint lint-check type-check test test-cov clean build venv venv-recreate start deps-update deps-sync quality style

# Use stage 0 container pip constraints (only if file exists)
CONSTRAINT_FILE := /etc/pip/constraint.txt
CONSTRAINTS := $(if $(wildcard $(CONSTRAINT_FILE)),--constraint $(CONSTRAINT_FILE),)

export PYTHONPATH = src
check_dirs := src tests
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/python
UV := $(shell which uv)
# Find Python 3.11+ (prefer /usr/local/bin/python for PyTorch container compatibility)
PYTHON := $(shell \
	for cmd in /usr/local/bin/python /usr/bin/python3; do \
		if [ -x "$$cmd" ] && $$cmd --version 2>/dev/null | grep -qE "3\.(1[1-9]|[2-9][0-9])"; then \
			echo $$cmd; \
			exit 0; \
		fi; \
	done; \
	command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || echo python3)

# Create venv with access to system packages (from stage 0 container)
$(VENV_DIR)/bin/activate:
	rm -rf $(VENV_DIR)
	$(UV) venv $(VENV_DIR) --python $(PYTHON) --system-site-packages

venv: $(VENV_DIR)/bin/activate

# Installation targets
install: venv
	$(UV) pip install -e .

install-dev: venv
	$(UV) pip install -e ".[dev]"

install-notebooks: venv
	$(UV) pip install -e ".[notebooks]"
	$(VENV_PY) -m ipykernel install --user --name python3 --display-name "Python 3 (tcc)"

# Development workflow targets
format: venv
	$(VENV_DIR)/bin/ruff format $(check_dirs)

lint: venv
	$(VENV_DIR)/bin/ruff check $(check_dirs) --fix

lint-check: venv
	$(VENV_DIR)/bin/ruff check $(check_dirs)

type-check: venv
	$(VENV_DIR)/bin/mypy src/tcc

quality: lint-check type-check
	@echo "All quality checks passed!"

style: format lint
	@echo "Code formatting and linting completed!"

# Testing targets
test:
	$(UV) run pytest

test-cov:
	$(UV) run pytest --cov=tcc --cov-report=html --cov-report=term

# Dependency management
deps-update: venv
	$(UV) pip compile pyproject.toml --upgrade $(CONSTRAINTS)

deps-sync: venv
	$(UV) pip compile pyproject.toml $(CONSTRAINTS) | $(UV) pip sync -

# Build targets
clean:
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	$(UV) build

# Utility targets
venv-recreate:
	rm -rf $(VENV_DIR)
	$(MAKE) venv

start:
	$(MAKE) venv-recreate
	$(MAKE) deps-sync
	$(MAKE) install
	$(VENV_PY) -m ipykernel install --user --name python3 --display-name "Python 3 (tcc)"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"
