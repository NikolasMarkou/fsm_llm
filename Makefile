.PHONY: test build clean

# Run tests
test:
	python -m pytest tests/ -vvv

# Build wheel
build:
	@echo "Building package..."
	python -m build

# Clean build artifacts (optional but useful)
clean:
	@echo "Cleaning artifacts..."
	rm -rf build/ dist/ *.egg-info/ logs/
	rm -rf tests/__pycache__
	rm -rf tests/test_fsm_llm/__pycache__
	rm -rf src/logs*
	rm -rf src/fsm_llm*egg-info
	rm -rf src/fsm_llm/__pycache__
	rm -rf src/fsm_llm_workflows/__pycache__
