.PHONY: lint test regress tables

lint:
	./scripts/lint.sh

test:
	uv run pytest -m "not nightly"

regress:
	uv run pytest

tables:
	@echo "coefficient/ROM generation lands in M4 (tools/coeffgen, tools/romgen)"
