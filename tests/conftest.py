"""
Global test configuration and fixtures for the entire test suite.
"""
import os
import sys
import json
import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import after path adjustment
from llm_fsm.llm import LLMInterface
from llm_fsm.definitions import FSMDefinition


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
    """Skip workflows tests if extension not installed."""
    try:
        import llm_fsm_workflows
    except ImportError:
        skip_workflows = pytest.mark.skip(
            reason="workflows extension not installed"
        )
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
        "tutorials": [],
        "use_cases": [],
        "features": [],
        "legacy": []
    }

    if not examples_root.exists():
        return categories

    for fsm_file in examples_root.rglob("fsm.json"):
        relative_path = fsm_file.relative_to(examples_root)
        category = relative_path.parts[0]

        if category in categories:
            categories[category].append(fsm_file)
        else:
            # Handle new organizational structure or unknown categories
            categories.setdefault("other", []).append(fsm_file)

    return categories


class MockLLMWithResponses:
    """Mock LLM that returns predefined responses based on conversation state."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        """
        Initialize mock LLM with optional response mappings.

        Args:
            responses: Dict mapping patterns to LLM responses
        """
        self.responses = responses or {}
        self.call_count = 0
        self.call_history = []
        self.default_response = {
            "transition": {
                "target_state": "greeting",
                "context_update": {}
            },
            "message": "Hello! How can I help you?",
            "reasoning": "Default response for testing"
        }

    def send_request(self, system_prompt: str, user_message: str, **kwargs) -> Dict[str, Any]:
        """Return appropriate mock response based on conversation context."""
        self.call_count += 1

        # Store call info for debugging
        call_info = {
            "call_number": self.call_count,
            "system_prompt": system_prompt,
            "user_message": user_message,
            "kwargs": kwargs
        }
        self.call_history.append(call_info)

        # Extract current state from system prompt for better matching
        current_state = self._extract_current_state(system_prompt)

        # Try to match specific state first
        if current_state in self.responses:
            return self.responses[current_state].copy()

        # Pattern matching for different scenarios
        system_lower = system_prompt.lower()
        message_lower = user_message.lower()

        if "greeting" in system_lower or "hello" in message_lower:
            return self.responses.get("greeting", self._greeting_response())
        elif "form" in system_lower or "name" in system_lower:
            return self.responses.get("form_filling", self._form_response(user_message))
        elif "story" in system_lower:
            return self.responses.get("story", self._story_response())
        elif "book" in system_lower or "recommend" in system_lower:
            return self.responses.get("recommendation", self._recommendation_response())

        return self.default_response.copy()

    def _extract_current_state(self, system_prompt: str) -> str:
        """Extract the current state from the system prompt."""
        import re
        # Look for <id>state_name</id> pattern in system prompt
        match = re.search(r'<id>([^<]+)</id>', system_prompt)
        return match.group(1) if match else "unknown"

    def _greeting_response(self) -> Dict[str, Any]:
        """Generate a greeting response."""
        return {
            "transition": {
                "target_state": "greeting",
                "context_update": {}
            },
            "message": "Hello! Nice to meet you. How can I help you today?",
            "reasoning": "Greeting response"
        }

    def _form_response(self, user_message: str) -> Dict[str, Any]:
        """Generate a form filling response."""
        # Simple pattern to extract potential name or email
        context_update = {}
        if "@" in user_message:
            context_update["email"] = user_message.strip()
        elif len(user_message.split()) <= 3:  # Likely a name
            context_update["name"] = user_message.strip()

        return {
            "transition": {
                "target_state": "collect_info",
                "context_update": context_update
            },
            "message": "Thank you! I've recorded that information. What else would you like to share?",
            "reasoning": "Form filling response"
        }

    def _story_response(self) -> Dict[str, Any]:
        """Generate a story response."""
        return {
            "transition": {
                "target_state": "telling_story",
                "context_update": {"story_started": True}
            },
            "message": "Once upon a time, in a land far away...",
            "reasoning": "Story response"
        }

    def _recommendation_response(self) -> Dict[str, Any]:
        """Generate a recommendation response."""
        return {
            "transition": {
                "target_state": "making_recommendation",
                "context_update": {"preference_noted": True}
            },
            "message": "Based on what you've told me, I'd recommend checking out some popular options.",
            "reasoning": "Recommendation response"
        }

    def add_response(self, pattern: str, response: Dict[str, Any]):
        """Add a new pattern-response mapping."""
        self.responses[pattern] = response

    def get_call_count(self) -> int:
        """Get the number of times the LLM was called."""
        return self.call_count

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get information about the last LLM call."""
        return self.call_history[-1] if self.call_history else None

    def reset(self):
        """Reset call history and count."""
        self.call_count = 0
        self.call_history.clear()


@pytest.fixture
def mock_llm_interface():
    """Mock LLM interface for deterministic testing."""
    return Mock(spec=LLMInterface)


@pytest.fixture
def mock_llm_with_responses():
    """Create a mock LLM with realistic responses."""
    return MockLLMWithResponses()


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
                        "priority": 100
                    }
                ]
            }
        }
    }

    # Try to load from fixtures, or create minimal one
    fixtures_fsm_dir = test_fixtures_root / "test_fsm_definitions"
    fixtures_fsm_dir.mkdir(exist_ok=True)

    minimal_fsm_path = fixtures_fsm_dir / "minimal_fsm.json"
    if not minimal_fsm_path.exists():
        with open(minimal_fsm_path, 'w') as f:
            json.dump(minimal_fsm, f, indent=2)

    with open(minimal_fsm_path) as f:
        fsm_data = json.load(f)

    return FSMDefinition.model_validate(fsm_data)


@pytest.fixture
def example_fsm_factory(mock_llm_with_responses):
    """Factory to create LLM_FSM instances for testing examples."""

    def _create_fsm(fsm_path: Path, **kwargs):
        """Create an LLM_FSM instance with mocked LLM for testing."""
        # Import here to avoid circular imports
        from llm_fsm import API
        from llm_fsm.utilities import load_fsm_from_file

        fsm_def = load_fsm_from_file(fsm_path)

        # Create LLM_FSM with mock LLM
        llm_fsm = API(
            fsm_definition=fsm_def,
            llm_interface=mock_llm_with_responses,
            **kwargs
        )
        return llm_fsm

    return _create_fsm


@pytest.fixture
def conversation_tester():
    """Utility for testing conversation flows."""

    class ConversationTester:
        def __init__(self, fsm):
            """
            Initialize conversation tester.

            Args:
                fsm: LLM_FSM instance to test
            """
            self.fsm = fsm
            self.conversation_id = None
            self.responses = []

        def start_conversation(self, initial_message: str = ""):
            """Start a new conversation."""
            self.conversation_id, response = self.fsm.converse(initial_message)
            self.responses.append(response)
            return response

        def send_message(self, message: str):
            """Send a message and get response."""
            if not self.conversation_id:
                raise ValueError("Conversation not started. Call start_conversation() first.")

            _, response = self.fsm.converse(message, self.conversation_id)
            self.responses.append(response)
            return response

        def get_context(self) -> Dict[str, Any]:
            """Get current conversation context."""
            if not self.conversation_id:
                return {}
            return self.fsm.get_data(self.conversation_id)

        def get_current_state(self) -> str:
            """Get current FSM state."""
            if not self.conversation_id:
                return ""
            return self.fsm.get_current_state(self.conversation_id)

        def assert_state(self, expected_state: str):
            """Assert current state matches expected."""
            current_state = self.get_current_state()
            assert current_state == expected_state, f"Expected state '{expected_state}', got '{current_state}'"

        def assert_context_has_key(self, key: str):
            """Assert context contains a specific key."""
            context = self.get_context()
            assert key in context, f"Key '{key}' not found in context: {context}"

        def assert_context_value(self, key: str, expected_value: Any):
            """Assert context key has expected value."""
            context = self.get_context()
            actual_value = context.get(key)
            assert actual_value == expected_value, f"Expected {key}='{expected_value}', got '{actual_value}'"

        def assert_response_contains(self, text: str, case_sensitive: bool = False):
            """Assert the last response contains specific text."""
            if not self.responses:
                pytest.fail("No responses recorded")

            last_response = self.responses[-1]
            if case_sensitive:
                assert text in last_response, f"Response does not contain '{text}': {last_response}"
            else:
                assert text.lower() in last_response.lower(), f"Response does not contain '{text}': {last_response}"

        def get_response_count(self) -> int:
            """Get the number of responses received."""
            return len(self.responses)

        def get_all_responses(self) -> List[str]:
            """Get all responses received."""
            return self.responses.copy()

    return ConversationTester


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
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "examples: mark test as example test"
    )
    config.addinivalue_line(
        "markers", "real_llm: mark test as requiring real LLM API"
    )