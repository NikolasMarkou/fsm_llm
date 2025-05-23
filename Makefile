.PHONY: test build clean

# Run tests
test:
	python -m pytest tests/ -vv

# Build wheel
build:
	@echo "Building package..."
	python -m build

# Clean build artifacts (optional but useful)
clean:
	@echo "Cleaning artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf src/llm_fsm*egg-info