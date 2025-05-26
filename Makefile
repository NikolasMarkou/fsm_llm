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
	rm -rf tests/test_llm_fsm/__pycache__
	rm -rf src/llm_fsm*egg-info
	rm -rf src/llm_fsm/__pycache__
	rm -rf src/llm_fsm_workflows/__pycache__