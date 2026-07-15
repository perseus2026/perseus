.PHONY: install
install: ## Install dependencies
	uv sync && uv run prek prepare-hooks

.PHONY: lint
lint: ## Run pre-commit hooks on the whole repo
	uv run prek run -a

.PHONY: test
test: ## Run tests with coverage
	uv run pytest -n auto --cov --cov-report=term-missing --cov-report=xml --cov-fail-under=80
