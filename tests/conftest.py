import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture
def has_workflows():
    """Check if workflows extension is available."""
    try:
        import llm_fsm_workflows
        return True
    except ImportError:
        return False

# Skip workflows tests if not installed
def pytest_collection_modifyitems(config, items):
    try:
        import llm_fsm_workflows
    except ImportError:
        skip_workflows = pytest.mark.skip(
            reason="workflows extension not installed"
        )
        for item in items:
            if "test_workflows" in str(item.fspath):
                item.add_marker(skip_workflows)