.PHONY: help test lint format type-check lint-imports build clean install-dev coverage audit api-snapshot decision-index decision-check changelog-check gate-step gate-push docs docs-serve docs-clean

# Default target
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run test suite
	python -m pytest tests/ -v

lint: ## Run linter (ruff)
	python -m ruff check src/ tests/

format: ## Format code with ruff
	python -m ruff format src/ tests/

type-check: ## Run type checker (mypy)
	python -m mypy -p fsm_llm

lint-imports: ## Check the import contracts in .importlinter (import-linter)
	lint-imports

coverage: ## Run tests with coverage report
	python -m pytest tests/ --cov=fsm_llm --cov-branch --cov-report=term-missing --cov-report=html

build: ## Build wheel and sdist
	@echo "Building package..."
	python -m build

clean: ## Remove build artifacts and caches
	@echo "Cleaning artifacts..."
	rm -rf build/ dist/ *.egg-info/ src/*.egg-info/ logs/
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/
	rm -rf tests/__pycache__
	rm -rf tests/test_fsm_llm/__pycache__
	rm -rf src/logs* src/fsm_llm/logs*
	rm -rf src/fsm_llm/__pycache__
	rm -rf htmlcov/ .coverage
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

install-dev: ## Install package in development mode with all extras
	pip install -c constraints.txt -e ".[dev,workflows,reasoning,agents,monitor,eval,synthesis,testing,otel,a2a]"
	pre-commit install

audit: ## Audit site-packages for suspicious .pth files
	python scripts/audit_pth.py

api-snapshot: ## Regenerate tests/fixtures/api_snapshot.txt, the public API snapshot
	python scripts/api_snapshot.py --write tests/fixtures/api_snapshot.txt

decision-index: ## Write a decision report to build/decision-index.md (on demand, gitignored, never committed)
	python scripts/decision_index.py --write build/decision-index.md

decision-check: ## Fail when a decision anchor has no line in plans/ANCHORS.md or names no plan id
	python scripts/decision_index.py --check

# Green on the real CHANGELOG.md: the form of every heading and BREAKING bullet. CI runs the same
# command in the lint job, scripts/gate.sh push runs it and test_changelog_check.py runs it on every pytest run.
changelog-check: ## Check the CHANGELOG form: ### headings, compare links, the BREAKING marker spelling
	python scripts/changelog_check.py check

# The lean gates (scripts/gate.sh): the step gate before a commit, the push gate before a push (the pre-push hook).
gate-step: ## Run the lean step gate on the working tree (scripts/gate.sh step)
	scripts/gate.sh step

gate-push: ## Run the lean push gate against origin/main (scripts/gate.sh push, the pre-push hook)
	scripts/gate.sh push

docs: ## Build the docs site into site/ (strict: any warning fails); needs `pip install --group docs -c constraints.txt`
	NO_MKDOCS_2_WARNING=1 mkdocs build --strict

docs-serve: ## Serve the docs site locally with live reload (http://127.0.0.1:8000)
	NO_MKDOCS_2_WARNING=1 mkdocs serve

docs-clean: ## Remove the built docs site (site/)
	rm -rf site/
