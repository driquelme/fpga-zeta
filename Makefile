.PHONY: lint test regress tables

lint:
	./scripts/lint.sh

test:
	uv run pytest -m "not nightly"

regress:
	uv run pytest

tables:
	uv run python scripts/gen_tables.py
