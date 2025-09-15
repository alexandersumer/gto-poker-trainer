# Enforce Python 3.12.11 exactly for all tasks.
PYTHON := python3.12

.PHONY: ensure-python
ensure-python:
	@$(PYTHON) -c "import sys; assert sys.version_info[:3]==(3,12,11), 'Expected 3.12.11, got %s' % sys.version.split()[0]; print('Using Python %s' % sys.version.split()[0])"

.PHONY: venv install-dev test lint fix format check clean

venv:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && $(PYTHON) -m pip install -U pip

install-dev: ensure-python
	$(PYTHON) -m pip install -e .[dev]

test: ensure-python
	$(PYTHON) -m pytest -q

lint: ensure-python
	$(PYTHON) -m ruff check .

fix: ensure-python
	$(PYTHON) -m ruff check . --fix

format: ensure-python
	$(PYTHON) -m ruff format .

check: ensure-python lint test

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.py[cod]" -delete
	rm -rf .pytest_cache .ruff_cache
	rm -rf src/*.egg-info
