.PHONY: test build clean version

# Generate version file
version:
	@echo "Generating version info..."
	@GIT_COMMIT=$$(git rev-parse HEAD); \
	GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ"); \
	VERSION=$$(python -c "import tomllib; config = tomllib.load(open('pyproject.toml', 'rb')); print(config.get('project', {}).get('version', 'dev'))"); \
	sed -e "s/{version}/$$VERSION/g" \
	    -e "s/{git_commit}/$$GIT_COMMIT/g" \
	    -e "s/{git_branch}/$$GIT_BRANCH/g" \
	    -e "s/{build_date}/$$BUILD_DATE/g" \
	    src/llm_fsm/_version.py.template > src/llm_fsm/_version.py

# Run tests
test:
	python -m pytest tests/ -vv

# Build wheel
build: version
	python -m build

# Clean build artifacts (optional but useful)
clean:
	rm -rf build/ dist/ *.egg-info/