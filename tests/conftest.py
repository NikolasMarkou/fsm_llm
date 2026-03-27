"""
Global test configuration and fixtures for the entire test suite.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import after path adjustment
from fsm_llm.definitions import (
    DataExtractionResponse,
    FSMDefinition,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface


@pytest.fixture
def has_workflows():
    """Check if workflows extension is available."""
    try:
        import fsm_llm_workflows  # noqa: F401

        return True
    except ImportError:
        return False


# Skip workflows tests if not installed
def pytest_collection_modifyitems(config, items):
    """Skip workflows tests if extension not installed."""
    try:
        import fsm_llm_workflows  # noqa: F401
    except ImportError:
        skip_workflows = pytest.mark.skip(reason="workflows extension not installed")
        for item in items:
            if "test_workflows" in str(item.fspath):
                item.add_marker(skip_workflows)


# Example testing fixtures
@pytest.fixture(scope="session")
def examples_root():
    """Get the examples directory path."""
    return Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="session")
def test_fixtures_root():
    """Get the test fixtures directory path."""
    fixtures_path = Path(__file__).parent / "fixtures"
    # Create fixtures directory if it doesn't exist
    fixtures_path.mkdir(exist_ok=True)
    return fixtures_path


@pytest.fixture(scope="session")
def all_example_paths(examples_root):
    """Get all FSM definition paths from examples."""
    fsm_paths = []
    if not examples_root.exists():
        return fsm_paths

    for fsm_file in examples_root.rglob("fsm.json"):
        fsm_paths.append(fsm_file)
    return fsm_paths


@pytest.fixture(scope="session")
def example_categories(examples_root):
    """Categorize examples by type."""
    categories = {
        "basic": [],
        "intermediate": [],
        "advanced": [],
        "classification": [],
        "reasoning": [],
        "workflows": [],
        "agents": [],
    }

    if not examples_root.exists():
        return categories

    for fsm_file in examples_root.rglob("fsm.json"):
        relative_path = fsm_file.relative_to(examples_root)
        category = relative_path.parts[0]

        if category in categories:
            categories[category].append(fsm_file)
        else:
            categories.setdefault("other", []).append(fsm_file)

    return categories


class MockLLM2Interface(LLMInterface):
    """Mock LLM implementing the 2-pass architecture for fsm_llm functional tests."""

    def __init__(
        self,
        extraction_data=None,
        response_text="Hello! How can I help you?",
        transition_target=None,
    ):
        self.extraction_data = extraction_data or {}
        self.response_text = response_text
        self.transition_target = transition_target
        self.call_history = []

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.call_history.append(("generate_response", request))
        return ResponseGenerationResponse(
            message=self.response_text,
            message_type="response",
            reasoning="Mock response",
        )

    def extract_field(self, request):
        self.call_history.append(("extract_field", request))
        from fsm_llm.definitions import FieldExtractionResponse

        value = self.extraction_data.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="Mock field extraction",
            is_valid=value is not None,
        )



def configure_mock_extract_field(mock_llm, mock_data=None):
    """Configure a Mock(spec=LLMInterface) with a working extract_field side_effect.

    Call this on any mock LLM interface that may be used with the pipeline,
    since the pipeline now calls extract_field instead of extract_data.
    """
    from fsm_llm.definitions import FieldExtractionResponse

    data = mock_data or {"name": "TestUser", "email": "test@test.com", "age": "25"}

    def _mock_extract_field(request):
        value = data.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="Mock field extraction",
            is_valid=value is not None,
        )

    mock_llm.extract_field.side_effect = _mock_extract_field
    return mock_llm


@pytest.fixture
def mock_llm2_interface():
    """Mock LLM interface for fsm_llm 2-pass architecture testing."""
    return MockLLM2Interface()


@pytest.fixture
def mock_llm_interface():
    """Mock LLM interface for deterministic testing."""
    from fsm_llm.definitions import FieldExtractionResponse

    mock = Mock(spec=LLMInterface)

    # extract_field returns per-field response — uses the field_name from request
    def _mock_extract_field(request):
        # Default mock data keyed by field name
        mock_data = {"name": "TestUser", "email": "test@test.com", "age": "25"}
        value = mock_data.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="Mock field extraction",
            is_valid=value is not None,
        )

    mock.extract_field.side_effect = _mock_extract_field

    # generate_response returns a simple string
    mock.generate_response.return_value = ResponseGenerationResponse(
        message="Hello! How can I help you?",
        message_type="response",
        reasoning="Mock response",
    )

    return mock


@pytest.fixture
def sample_fsm_definition(test_fixtures_root):
    """Load a sample FSM definition for testing."""
    # Create a minimal FSM definition for testing if fixture doesn't exist
    minimal_fsm = {
        "name": "test_fsm",
        "description": "A minimal FSM for testing",
        "version": "3.0",
        "initial_state": "greeting",
        "states": {
            "greeting": {
                "id": "greeting",
                "description": "Initial greeting state",
                "purpose": "Greet the user",
                "transitions": [
                    {
                        "target_state": "greeting",
                        "description": "Stay in greeting",
                        "priority": 100,
                    }
                ],
            }
        },
    }

    # Try to load from fixtures, or create minimal one
    fixtures_fsm_dir = test_fixtures_root / "test_fsm_definitions"
    fixtures_fsm_dir.mkdir(exist_ok=True)

    minimal_fsm_path = fixtures_fsm_dir / "minimal_fsm.json"
    if not minimal_fsm_path.exists():
        with open(minimal_fsm_path, "w") as f:
            json.dump(minimal_fsm, f, indent=2)

    with open(minimal_fsm_path) as f:
        fsm_data = json.load(f)

    return FSMDefinition.model_validate(fsm_data)


@pytest.fixture
def sample_fsm_definition_v2():
    """Minimal FSM definition for fsm_llm testing."""
    fsm_data = {
        "name": "test_greeting",
        "description": "A minimal greeting FSM for testing",
        "version": "4.1",
        "initial_state": "greeting",
        "states": {
            "greeting": {
                "id": "greeting",
                "description": "Initial greeting state",
                "purpose": "Greet the user and collect their name",
                "transitions": [
                    {
                        "target_state": "farewell",
                        "description": "User wants to end conversation",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "User name is collected",
                                "requires_context_keys": ["user_name"],
                            }
                        ],
                    }
                ],
            },
            "farewell": {
                "id": "farewell",
                "description": "Farewell state",
                "purpose": "Say goodbye to the user",
                "transitions": [],
            },
        },
    }
    return FSMDefinition.model_validate(fsm_data)


# Optional: Environment variable based configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration from environment variables."""
    return {
        "skip_slow_tests": os.getenv("SKIP_SLOW_TESTS", "false").lower() == "true",
        "test_real_llm": os.getenv("TEST_REAL_LLM", "false").lower() == "true",
        "llm_model": os.getenv("TEST_LLM_MODEL", "gpt-3.5-turbo"),
        "api_key": os.getenv("OPENAI_API_KEY"),
    }


# Optional: Skip real LLM tests if no API key
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "examples: mark test as example test")
    config.addinivalue_line("markers", "real_llm: mark test as requiring real LLM API")
