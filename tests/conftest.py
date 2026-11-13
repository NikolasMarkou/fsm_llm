"""
Global test configuration and fixtures for the entire test suite.
"""

import json
import math
import os
import socket
import sys
from pathlib import Path
from typing import NoReturn
from unittest.mock import Mock

import pytest

from fsm_llm.constants import ENV_LLM_MODEL
from fsm_llm.definitions import FSMDefinition, ResponseGenerationResponse
from fsm_llm.llm import LLMInterface
from fsm_llm.testing import MockLLM2Interface

#: Ollama tag the live suites expect to be pulled (a model name without its
#: ``ollama_chat/`` provider prefix). The package has no default model: live
#: suites pass the model explicitly, and this is the one place the tag lives.
OLLAMA_MODEL_TAG = "bonsai-2:27b"
#: Where a stock Ollama daemon publishes its model list.
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def ollama_available(model_tag: str = OLLAMA_MODEL_TAG) -> bool:
    """Whether Ollama is reachable AND *model_tag* is pulled.

    Interface contract (2 call sites -- ``tests/test_integration_ollama.py``
    and ``tests/test_fsm_llm/test_live_classification_memory.py``; centralised so the
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


def posix_only(reason: str) -> pytest.MarkDecorator:
    """A ``skipif`` mark for a test that needs POSIX file modes, signals or pids.

    Interface contract (many call sites; the AST rule in
    ``tests/test_packaging.py::TestPortabilityRules`` lists every use):
        - Parameter: *reason*, a non-blank string literal saying what the test
          needs that Windows does not give (a mode bit, a signal, a pid probe).
        - Returns a ``pytest.mark.skipif`` for ``sys.platform == "win32"``;
          macOS and Linux run the test.  This is a capability skip, never a
          place to hide a failure: the reason names the missing capability.
    """
    return pytest.mark.skipif(sys.platform == "win32", reason=reason)


def skip_if_root() -> None:
    """Skip the running test when the process is root (modes are not enforced).

    Interface contract: call it in the test BODY, never at import or decorator
    time (``os.geteuid`` does not exist on Windows).  No return value; raises
    ``pytest.skip.Exception`` for root, does nothing otherwise.
    """
    geteuid = getattr(os, "geteuid", None)
    if geteuid is not None and geteuid() == 0:
        pytest.skip("root ignores file and directory modes")


def symlink_or_skip(
    link: Path, target: Path | str, *, target_is_directory: bool = False
) -> None:
    """Create ``link`` pointing at ``target``; skip the test where that is refused.

    Interface contract (the one way a test makes a symlink; the rule in
    ``TestPortabilityRules`` bans a bare ``symlink_to``/``os.symlink``):
        - Parameters: the link path, what it points at (may not exist: a
          dangling link is valid), and ``target_is_directory`` for Windows
          (an existing directory target sets it on its own).
        - Returns ``None``.  ``OSError`` or ``NotImplementedError`` (Windows
          without the symlink privilege, a filesystem without links) skips the
          calling test; no other error is caught.
    """
    # Windows needs the flag for a directory; an existing directory is detected
    # here so a call site cannot forget it (a dangling link stays a file link).
    is_dir = target_is_directory or (link.parent / target).is_dir()
    try:
        link.symlink_to(target, target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported here")


#: Env knob that raises ``timing_factor`` (a positive float; a smaller or
#: invalid value is ignored, so the factor never drops below 1).
TIMING_FACTOR_ENV = "FSM_LLM_TEST_TIMING_FACTOR"
#: Budget multiplier while a tracer or coverage is active (measured 1.3x to
#: 1.9x of CPU-bound work under ``--cov --cov-branch``, more on a loaded host).
TRACED_TIMING_FACTOR = 5.0


def _coverage_running() -> bool:
    """Whether a ``coverage.Coverage`` is measuring this process right now.

    Python 3.12+ can measure through ``sys.monitoring``, which leaves
    ``sys.gettrace()`` empty, so the tracer check alone would miss it. Never
    imports ``coverage``: absent from ``sys.modules`` means not running.
    """
    module = sys.modules.get("coverage")
    current = getattr(getattr(module, "Coverage", None), "current", None)
    try:
        return current is not None and current() is not None
    except Exception:
        return False


def current_timing_factor() -> float:
    """The multiplier a wall-clock budget needs on this run, never below 1.

    Interface contract (the ``timing_factor`` fixture below, and tests that
    simulate a condition by patching ``sys.gettrace`` or the environment):
        - No parameters; reads ``sys.gettrace()``, ``COV_CORE_SOURCE`` (set by
          pytest-cov for the whole run), a live ``coverage.Coverage`` and
          ``FSM_LLM_TEST_TIMING_FACTOR`` at call time.
        - Returns ``TRACED_TIMING_FACTOR`` (5.0) when a tracer, pytest-cov or
          coverage is active, else 1.0; a valid ``FSM_LLM_TEST_TIMING_FACTOR``
          (a finite float above 0) raises it when larger. A value that does
          not parse, is not finite or is not above 0 is ignored; one below 1
          cannot lower the floor.
        - Never raises. Budgets are scaled, never skipped or deselected.
        - Scale an ABSOLUTE time budget only. A ratio bound (the spread of
          per-character times over a doubling sweep, a speed-up) is never
          multiplied by it: a uniform slowdown cancels in a ratio, so scaling
          the bound only lets a quadratic pattern through. Retry a noisy
          ratio instead (``tests/test_fsm_llm/test_timing_factor.py`` scans
          every test file for a scaled ratio name).
    """
    traced = (
        sys.gettrace() is not None
        or "COV_CORE_SOURCE" in os.environ
        or _coverage_running()
    )
    factor = TRACED_TIMING_FACTOR if traced else 1.0
    raw = os.environ.get(TIMING_FACTOR_ENV, "").strip()
    if raw:
        try:
            override = float(raw)
        except ValueError:
            override = 0.0
        if math.isfinite(override) and override > 0:
            factor = max(factor, override)
    return factor


@pytest.fixture
def timing_factor() -> float:
    """Multiplier for a wall-clock budget: 1.0 normally, 5.0 under a tracer.

    Use it as ``budget * timing_factor`` in a test that asserts an elapsed
    time, joins a thread or waits on an event (never on a ratio bound); a test
    about a time budget in product code injects a fake monotonic clock instead. See
    ``current_timing_factor`` for the rules and the env override.
    """
    return current_timing_factor()


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


#: The model name every offline test sees in ``LLM_MODEL``. Nothing resolves it
#: to a real provider: the network is blocked and the interfaces are doubles.
TEST_LLM_MODEL = "test/model"


@pytest.fixture(scope="session", autouse=True)
def _session_llm_model():
    """``LLM_MODEL`` for session and module fixtures, and a stray ``.env``.

    Interface contract: sets ``LLM_MODEL`` to ``TEST_LLM_MODEL`` for the whole
    session through a ``MonkeyPatch`` context and restores the previous value
    at the end. A fixture wider than a function (built before the per-test
    fixture below) therefore resolves a model too, and the repo-root ``.env``
    (git-ignored, loaded by litellm at import until row 0.8 finishes) cannot
    decide a result. Nothing here reads or writes the file.
    """
    with pytest.MonkeyPatch.context() as session_patch:
        session_patch.setenv(ENV_LLM_MODEL, TEST_LLM_MODEL)
        yield


@pytest.fixture(autouse=True)
def _llm_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LLM_MODEL`` is set for every test (plan d454fd62 D-005, row 0.8).

    The library has no default model: ``resolve_model`` raises
    ``ModelNotConfiguredError`` when nothing names one. The 159 offline
    ``AgentConfig()`` sites and every model-less ``API`` read it from here. A
    test that needs the unset state deletes it itself with
    ``monkeypatch.delenv(ENV_LLM_MODEL, raising=False)``; ``monkeypatch`` undoes
    that at teardown, so the next test starts set again. Tests that clear the
    whole environment (``patch.dict(os.environ, ..., clear=True)``) are
    unaffected: they state their own.
    """
    monkeypatch.setenv(ENV_LLM_MODEL, TEST_LLM_MODEL)


@pytest.fixture
def reloadable_logging_module():
    """``fsm_llm.logging`` with its namespace restored after the test.

    Interface contract: yields the module object. A test that calls
    ``importlib.reload`` on it re-executes the module body in the same
    namespace: every function becomes a new object (so ``fsm_llm.setup_logging``
    and ``fsm_llm.logging.setup_logging`` stop being one function) and the
    handler registries (``_library_handler_ids`` and the others) are replaced
    by empty ones. Teardown puts back every name the module held on entry and
    drops any name the reload added, so a reload cannot reach a later test.
    Failure mode: none; it only reads and writes the module's ``__dict__``.
    """
    import fsm_llm.logging as log_module

    saved = dict(vars(log_module))
    yield log_module
    namespace = vars(log_module)
    for name in [name for name in namespace if name not in saved]:
        del namespace[name]
    namespace.update(saved)


@pytest.fixture(scope="session")
def test_fixtures_root():
    """Get the test fixtures directory path."""
    fixtures_path = Path(__file__).parent / "fixtures"
    # Create fixtures directory if it doesn't exist
    fixtures_path.mkdir(exist_ok=True)
    return fixtures_path


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
    mock = configure_mock_extract_field(Mock(spec=LLMInterface))

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
        with open(minimal_fsm_path, "w", encoding="utf-8") as f:
            json.dump(minimal_fsm, f, indent=2)

    with open(minimal_fsm_path, encoding="utf-8") as f:
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
