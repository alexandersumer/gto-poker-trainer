# Run project automation through uv for consistent environments.
UV_RUN := uv run --no-config --locked --extra dev --
UV_SYNC := uv sync --no-config --locked --extra dev --no-build-isolation-package eval7

.PHONY: ensure-python
ensure-python:
	@$(UV_RUN) python -c "import sys; major, minor, patch = sys.version_info[:3]; assert (major, minor) == (3, 13) and patch >= 5, 'Expected Python 3.13.5+, got %s' % sys.version.split()[0]; print(f'Using Python {sys.version.split()[0]}')"

.PHONY: venv install-dev test lint fix format check render-smoke clean

venv:
	uv venv

install-dev: ensure-python
	$(UV_SYNC)

test: ensure-python
	$(UV_RUN) pytest -q

lint: ensure-python
	$(UV_RUN) ruff check .

fix: ensure-python
	$(UV_RUN) ruff check . --fix

format: ensure-python
	$(UV_RUN) ruff format .

check: ensure-python lint test

render-smoke: ensure-python
	$(UV_RUN) python scripts/check_render.py

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.py[cod]" -delete
	rm -rf .pytest_cache .ruff_cache
	rm -rf src/*.egg-info
