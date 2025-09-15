PYTHON := python3

.PHONY: venv install-dev test lint fix format check clean

venv:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && $(PYTHON) -m pip install -U pip

install-dev:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

fix:
	$(PYTHON) -m ruff check . --fix

format:
	$(PYTHON) -m ruff format .

check: lint test

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.py[cod]" -delete
	rm -rf .pytest_cache .ruff_cache
	rm -rf src/*.egg-info
