"""
Global test configuration and fixtures for the entire test suite.
"""

import json
import re
import socket
import sys
from pathlib import Path
from typing import NoReturn
from unittest.mock import Mock

import pytest

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import after path adjustment
from fsm_llm.constants import DEFAULT_LLM_MODEL
from fsm_llm.definitions import (
    BulkExtractionRequest,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    FSMDefinition,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface

#: Ollama tag of the package default model (``DEFAULT_LLM_MODEL`` minus its
#: ``ollama_chat/`` provider prefix). Live suites pass their own tag explicitly.
OLLAMA_MODEL_TAG = DEFAULT_LLM_MODEL.split("/", 1)[1]
#: Where a stock Ollama daemon publishes its model list.
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def ollama_available(model_tag: str = OLLAMA_MODEL_TAG) -> bool:
    """Whether Ollama is reachable AND *model_tag* is pulled.

    Interface contract (2 call sites -- ``tests/test_integration_ollama.py``
    and ``tests/test_fsm_llm_harness/test_live_ollama.py``; centralised so the
    two live suites cannot drift into probing different daemons):
        - Parameter: a substring of the tag as ``/api/tags`` reports it.
        - Returns ``True`` only when the daemon answered 200 AND some pulled
          model name contains *model_tag*.  Any failure -- httpx absent, the
          daemon down, a non-200, malformed JSON -- returns ``False``.
        - Never raises.  Performs ONE bounded HTTP GET, so callers must keep it
          behind a short-circuit rather than calling it at import time when the
          live suite is switched off.
    """
    try:
        import httpx

        resp = httpx.get(OLLAMA_TAGS_URL, timeout=3)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model_tag in name for name in models)
    except Exception:
        return False


#: Markers whose tests may open real TCP connections under ``block_network``.
NETWORK_EXEMPT_MARKERS = ("real_llm", "integration")
#: Address families ``block_network`` refuses (Unix sockets stay open).
_BLOCKED_FAMILIES = (socket.AF_INET, socket.AF_INET6)


def network_exempt(node: pytest.Item) -> bool:
    """Whether *node* carries one of ``NETWORK_EXEMPT_MARKERS``."""
    return any(node.get_closest_marker(name) for name in NETWORK_EXEMPT_MARKERS)


def block_network(monkeypatch: pytest.MonkeyPatch, node: pytest.Item) -> None:
    """Refuse every IPv4/IPv6 connect for the rest of *node*'s test.

    Interface contract (2 call sites -- the autouse fixtures in
    ``tests/test_fsm_llm_agents/conftest.py`` and
    ``tests/test_fsm_llm_meta/conftest.py``; kept here so the two suites
    cannot drift into different guards):
        - Patches ``socket.socket.connect`` and ``connect_ex`` through
          *monkeypatch*, so the patch is undone at teardown. Loopback is
          blocked too: the default model is a local Ollama.
        - A blocked call raises ``ConnectionRefusedError`` at once (no
          timeout), naming the address. Unix sockets (MCP stdio, asyncio
          self-pipes) are untouched.
        - No-op when ``network_exempt(node)`` is true (live suites).
    """
    if network_exempt(node):
        return
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _refuse(address: object) -> NoReturn:
        raise ConnectionRefusedError(
            f"network blocked in offline tests: connect to {address!r} "
            f"(mark the test real_llm or integration to allow it)"
        )

    def _connect(sock: socket.socket, address: object) -> None:
        if sock.family in _BLOCKED_FAMILIES:
            _refuse(address)
        real_connect(sock, address)

    def _connect_ex(sock: socket.socket, address: object) -> int:
        if sock.family in _BLOCKED_FAMILIES:
            _refuse(address)
        return real_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _connect_ex)


@pytest.fixture(scope="session")
def test_fixtures_root():
    """Get the test fixtures directory path."""
    fixtures_path = Path(__file__).parent / "fixtures"
    # Create fixtures directory if it doesn't exist
    fixtures_path.mkdir(exist_ok=True)
    return fixtures_path


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


#: Where the response prompt names the state it answers for (``prompts.py``).
_CURRENT_STATE_TAG = re.compile(r"<current_state>([^<]+)</current_state>")


class PromptGroundedLLM(LLMInterface):
    """Fake LLM that knows a fact only when the prompt shows its evidence.

    ``MockLLM2Interface`` answers every field from a fixed dict whatever the
    prompt says, so it cannot tell a context-free prompt from a grounded one.
    This fake can: a value comes back only when the text it would ground on
    is in the request, like a real model reading its prompt.

    Interface contract (shared by the agents Phase-1 tests):
        - ``facts``: ``{field_name: (value, evidence)}``. ``evidence`` is a
          substring; the field resolves to ``value`` only when it occurs in
          the request text.
        - ``extract_field``: the text is ``system_prompt``, ``user_message``
          and the JSON-dumped ``context``. Returns ``value`` (``is_valid=True``)
          or ``None`` (``is_valid=False``, confidence 0) for a missing fact or
          absent evidence.
        - ``extract_bulk_data``: returns only facts whose name the prompt asks
          for (``"name"`` quoted in ``system_prompt``) and whose evidence is in
          ``system_prompt`` or ``user_message``. A context-free prompt
          therefore yields ``{}``.
        - ``generate_response``: ``responses[state]`` for the state named by
          the prompt's ``<current_state>`` tag, else ``default_response``.
        - ``requests``: every call as ``(kind, request)`` in call order, kind
          being the method name. Never raises.
    """

    def __init__(
        self,
        facts: dict[str, tuple[object, str]] | None = None,
        responses: dict[str, str] | None = None,
        default_response: str = "ok",
    ) -> None:
        self.model = "prompt-grounded-fake"
        self.facts = dict(facts or {})
        self.responses = dict(responses or {})
        self.default_response = default_response
        self.requests: list[tuple[str, object]] = []

    def calls(self, kind: str) -> list:
        """The recorded requests of one kind, in call order."""
        return [request for k, request in self.requests if k == kind]

    def _grounded(self, name: str, text: str) -> object | None:
        value, evidence = self.facts.get(name, (None, ""))
        return value if evidence and evidence in text else None

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        self.requests.append(("extract_field", request))
        text = "\n".join(
            [
                request.system_prompt,
                request.user_message,
                json.dumps(request.context or {}, default=str),
            ]
        )
        value = self._grounded(request.field_name, text)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="prompt-grounded fake",
            is_valid=value is not None,
        )

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        self.requests.append(("extract_bulk_data", request))
        text = f"{request.system_prompt}\n{request.user_message}"
        data = {}
        for name in self.facts:
            if f'"{name}"' not in request.system_prompt:
                continue
            value = self._grounded(name, text)
            if value is not None:
                data[name] = value
        return DataExtractionResponse(extracted_data=data)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.requests.append(("generate_response", request))
        match = _CURRENT_STATE_TAG.search(request.system_prompt)
        state = match.group(1) if match else None
        message = self.responses.get(state or "", self.default_response)
        return ResponseGenerationResponse(
            message=message, message_type="response", reasoning="prompt-grounded fake"
        )


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


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "examples: mark test as example test")
    config.addinivalue_line("markers", "real_llm: mark test as requiring real LLM API")
