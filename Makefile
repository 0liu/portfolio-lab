# Task runner
.PHONY: fmt check test clean

fmt:    ## auto-format and auto-fix lint findings
	uv run ruff format .
	uv run ruff check --fix .

check:  ## ruff + import contracts + tests (identical to CI)
	uv run ruff format --check .
	uv run ruff check .
	uv run pytest

test:
	uv run pytest

clean:  ## remove tool caches and build junk
	rm -rf .pytest_cache .ruff_cache .import_linter_cache dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
