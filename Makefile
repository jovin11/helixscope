.PHONY: install install-dev install-full run test lint format format-check compile check

install:
	python -m pip install -e ".[ml]"

install-dev:
	python -m pip install -e ".[dev]"

install-full:
	python -m pip install -e ".[ml,dev]"

run:
	python python_service/app.py

test:
	python -m pytest

lint:
	python -m ruff check .

format:
	python -m ruff format .

format-check:
	python -m ruff format --check .

compile:
	python -m compileall -q python_service tests scripts

check: lint format-check compile test