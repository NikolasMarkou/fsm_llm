"""
Global test configuration and fixtures for the entire test suite.
"""

import errno
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import Mock

import pytest

from fsm_llm.constants import ENV_LLM_MODEL
from fsm_llm.core.definitions import FSMDefinition, ResponseGenerationResponse
from fsm_llm.llm import LLMInterface
from fsm_llm.testing import MockLLM2Interface
from tests.cputime import (  # noqa: F401  (re-exported: the one import path)
    CpuTimer,
    cpu_timer,
)

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
          every other platform runs the test.  This is a capability skip, never a
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


#: ``OSError.winerror`` values Windows gives a file name it cannot hold (invalid name,
#: bad path or file name syntax, label too long, invalid file or directory name).
_BAD_NAME_WINERRORS = frozenset({123, 161, 206, 267})


def name_refused(exc: OSError) -> bool:
    """Whether *exc* is a file system refusing a NAME (not a permission or a missing parent).

    Interface contract (``write_if_creatable`` and the tests that create a folder
    with a hostile name): ``True`` for ``EINVAL`` or a Windows ``winerror`` in
    ``_BAD_NAME_WINERRORS`` (NTFS cannot hold an ESC or a newline in a name, nor
    ``<>:"|?*``: ``[Errno 22]`` or ``[WinError 123]``); ``False`` for every other
    ``OSError``, so a real failure is never swallowed.
    """
    return exc.errno == errno.EINVAL or getattr(exc, "winerror", None) in (
        _BAD_NAME_WINERRORS
    )


def write_if_creatable(path: Path, content: bytes = b"x") -> bool:
    """Write *content* to *path* (parents created) unless the file system refuses its name.

    Interface contract (the tests of hostile file names): returns ``True`` when the file
    was written and ``False`` when the NAME is refused (``name_refused``); any other
    ``OSError`` propagates. Bytes, so no newline translation. A test uses it to try an
    ESC or newline name and checks what exists afterwards: Linux holds both, NTFS
    refuses both, and the assertions must follow what was created.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    except OSError as exc:
        if name_refused(exc):
            return False
        raise
    return True


def skip_if_backslash_separates(directory: Path) -> None:
    """Skip the running test when a backslash in a file name is a path separator.

    Interface contract: call it in the test BODY with a scratch *directory*. It writes
    ``a\\b.txt`` in a ``_backslash_probe`` folder there and skips (reason: such a name
    cannot exist) when that made a folder ``a`` (Windows) instead of one file; removes
    the probe either way. Returns ``None``; the file system is probed, not the platform.
    """
    import shutil

    probe = directory / "_backslash_probe"
    probe.mkdir()
    try:
        write_if_creatable(probe / "a\\b.txt")
        if (probe / "a").is_dir():
            pytest.skip(
                "a backslash is a path separator here, so that name cannot exist"
            )
    finally:
        shutil.rmtree(probe, ignore_errors=True)


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


def bind_api_clock(monkeypatch: pytest.MonkeyPatch, fake: Any) -> list[str]:
    """Bind *fake* as the ``time`` of every ``fsm_llm.engine.api`` module that reads one.

    Interface contract (4 call sites: the budget and idle-sweep tests of core and of the
    agents suite): replaces the module attribute ``time`` through *monkeypatch* (undone at
    teardown) in each module of the package that did ``import time``, found by walking the
    package (the old ``fsm_llm.api`` was one module; a module of the package that reads the
    clock and is not bound would mix real and fake readings, e.g. ``_last_accessed`` written by
    one module against the idle sweep of another). Never touches the global ``time`` module.
    Returns the dotted names bound; fails when none is found.
    """
    import importlib
    import pkgutil

    import fsm_llm.engine.api as package

    bound: list[str] = []
    for info in pkgutil.iter_modules(package.__path__):
        name = f"{package.__name__}.{info.name}"
        module = importlib.import_module(name)
        if getattr(module, "time", None) is time:
            monkeypatch.setattr(module, "time", fake)
            bound.append(name)
    assert bound, "no module of fsm_llm.engine.api reads the clock"
    return bound


#: Markers whose tests may open real TCP connections under ``block_network``.
NETWORK_EXEMPT_MARKERS = ("real_llm", "integration")
#: Address families ``block_network`` refuses (Unix sockets stay open; on Windows
#: asyncio self-pipes are loopback TCP, see ``_called_by_socketpair_fallback``).
_BLOCKED_FAMILIES = (socket.AF_INET, socket.AF_INET6)


def network_exempt(node: pytest.Item) -> bool:
    """Whether *node* carries one of ``NETWORK_EXEMPT_MARKERS``."""
    return any(node.get_closest_marker(name) for name in NETWORK_EXEMPT_MARKERS)


def _called_by_socketpair_fallback() -> bool:
    """Whether the caller of the patched connect is ``socket._fallback_socketpair``.

    Interface contract (``block_network`` only): compares the code object of the
    frame two levels up (the caller of ``_connect`` or ``_connect_ex``) with the
    stdlib function's, so a test-code function that merely shares the name is
    not exempt. Returns ``False`` when the stdlib has no such function or the
    frame is absent; never raises.
    """
    fallback = getattr(socket, "_fallback_socketpair", None)
    if fallback is None:
        return False
    try:
        return sys._getframe(2).f_code is fallback.__code__
    except ValueError:
        return False


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
          self-pipes on POSIX) are untouched. On Windows there is no AF_UNIX
          ``socketpair``: the stdlib builds one from two loopback TCP sockets
          in ``socket._fallback_socketpair`` (asyncio's self-pipe, every event
          loop), and that one function, identified by its code object and not
          by its name, may connect. Every other caller stays refused.
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
        if sock.family in _BLOCKED_FAMILIES and not _called_by_socketpair_fallback():
            _refuse(address)
        real_connect(sock, address)

    def _connect_ex(sock: socket.socket, address: object) -> int:
        if sock.family in _BLOCKED_FAMILIES and not _called_by_socketpair_fallback():
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
    from fsm_llm.core.definitions import FieldExtractionResponse

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


#: A parametrize value (``str`` or ``bytes``) longer than this gets a short id.
MAX_PARAM_VALUE_ID_CHARS = 1_000
#: A node id longer than this fails collection. pytest copies the node id into the
#: ``PYTEST_CURRENT_TEST`` environment variable, and Windows refuses a variable of
#: more than 32,767 characters (``ValueError`` in setup and teardown: the test never
#: runs), so the limit keeps a wide margin below it.
MAX_NODEID_CHARS = 4_000


def compact_param_id(value: object, argname: str) -> str | None:
    """A short, stable id for a huge ``str`` or ``bytes`` parameter, else ``None``.

    Interface contract (the ``pytest_make_parametrize_id`` hook below, and a test
    that builds its own ``ids=`` callable and wants the same form):
        - Parameters: the parametrize *value* and its *argname*.
        - Returns ``"<argname>-<str|bytes><length>-<sha1 prefix>"`` for a ``str`` or
          ``bytes`` longer than ``MAX_PARAM_VALUE_ID_CHARS`` (the same value always
          gives the same id, two values of one length differ by the digest), and
          ``None`` for everything else, so pytest builds its usual id.
        - Never raises for a ``str`` (a lone surrogate is encoded with
          ``surrogatepass``).
    """
    import hashlib

    if not isinstance(value, (str, bytes)) or len(value) <= MAX_PARAM_VALUE_ID_CHARS:
        return None
    raw = value if isinstance(value, bytes) else value.encode("utf-8", "surrogatepass")
    kind = "bytes" if isinstance(value, bytes) else "str"
    return f"{argname}-{kind}{len(value)}-{hashlib.sha1(raw).hexdigest()[:8]}"


def pytest_make_parametrize_id(config, val, argname):
    """Short ids for the 100,000 and 1,000,000 character parameters of the suite."""
    return compact_param_id(val, argname)


def pytest_collection_modifyitems(config, items):
    """Fail collection on a node id too long for a Windows environment variable."""
    too_long = [item.nodeid for item in items if len(item.nodeid) > MAX_NODEID_CHARS]
    if too_long:
        raise pytest.UsageError(
            f"{len(too_long)} test id(s) longer than {MAX_NODEID_CHARS} characters "
            "(Windows cannot set PYTEST_CURRENT_TEST that long); give the "
            "parametrize call short ids= or an id= per pytest.param, first: "
            + too_long[0][:150]
        )


#: ``huge_operand_explanation`` explains a failed ``==`` itself, without ``difflib``,
#: when an operand is a string or bytes of more than this many characters ...
MAX_COMPARE_CHARS = 10_000
#: ... or lines, or a list, tuple, dict or set of more than ``MAX_COMPARE_ITEMS``.
MAX_COMPARE_LINES = 200
MAX_COMPARE_ITEMS = 1_000


def _too_big_to_diff(value: object) -> bool:
    if isinstance(value, (str, bytes)):
        newline = b"\n" if isinstance(value, bytes) else "\n"
        return (
            len(value) > MAX_COMPARE_CHARS or value.count(newline) > MAX_COMPARE_LINES
        )
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return len(value) > MAX_COMPARE_ITEMS
    return False


def _describe(value: object) -> str:
    size = len(value) if hasattr(value, "__len__") else "?"
    return f"{type(value).__name__} of length {size}"


def huge_operand_explanation(op: str, left: object, right: object) -> list[str] | None:
    """Explain a failed ``==`` on huge operands without ``difflib``, else ``None``.

    Interface contract (``tests/test_fsm_llm/test_conftest_guards.py`` pins it):
        - Returns ``None`` (pytest explains the comparison as usual) unless *op* is
          ``==`` and an operand is a string or bytes of more than
          ``MAX_COMPARE_CHARS`` characters or ``MAX_COMPARE_LINES`` lines, or a
          list, tuple, dict, set or frozenset of more than ``MAX_COMPARE_ITEMS``.
        - Otherwise returns a short explanation (the first line is the summary, as
          pytest's own): the two types and lengths, the index of the first
          difference (strings, bytes, lists, tuples) with a short window of each
          side, and the size of the symmetric difference (dicts and sets). Linear
          time; never raises for the types above.
    Why: pytest renders a failed string ``==`` with ``difflib.ndiff`` over the
    lines; on Python 3.11 to 3.13 that is cubic in the number of near-equal lines
    (400 by 560 lines took a minute, 5,000 by 7,000 never finished: the Windows
    3.11 leg of CI ran for 35 minutes inside one failed assert), so a mismatch of a
    large text must not reach it. The ``pytest_assertrepr_compare`` hook cannot do
    this: pytest calls every implementation (it is not ``firstresult``), so the
    builtin one runs whatever a conftest returns; ``pytest_configure`` below puts
    this function in front of pytest's own ``util.assertrepr_compare`` instead.
    """
    if op != "==" or not (_too_big_to_diff(left) or _too_big_to_diff(right)):
        return None
    lines = [
        f"{_describe(left)} == {_describe(right)}",
        "full diff skipped: the operands are too large for difflib "
        "(guard in tests/conftest.py, huge_operand_explanation)",
    ]
    if isinstance(left, (str, bytes, list, tuple)) and type(left) is type(right):
        index = next(
            (i for i, (a, b) in enumerate(zip(left, right, strict=False)) if a != b),
            min(len(left), len(right)),
        )
        lines.append(f"first difference at index {index}")
        window = slice(max(index - 20, 0), index + 60)
        lines.append(f"left : {left[window]!r}")
        lines.append(f"right: {right[window]!r}")
    elif isinstance(left, (dict, set, frozenset)) and type(left) is type(right):
        lines.append(f"{len(set(left) ^ set(right))} keys or members on one side only")
    return lines


def guard_assertrepr_compare(original):
    """Wrap pytest's ``util.assertrepr_compare`` so a huge operand skips ``difflib``.

    Interface contract (``pytest_configure`` only): *original* is the function
    ``_pytest.assertion.util.assertrepr_compare``, which pytest 9 calls as
    ``assertrepr_compare(op=..., left=..., right=..., verbose=..., ...)`` and which
    returns an iterator of lines. The wrapper reads ``op``, ``left`` and ``right``
    from the keywords; a call that does not pass them by keyword (another pytest
    version) goes to *original* unchanged, so the guard fails open and never hides
    an assertion. Returns an iterator, as *original* does.
    """

    def guarded(*args, **kwargs):
        if {"op", "left", "right"} <= kwargs.keys():
            explanation = huge_operand_explanation(
                kwargs["op"], kwargs["left"], kwargs["right"]
            )
            if explanation is not None:
                return iter(explanation)
        return original(*args, **kwargs)

    guarded.__wrapped__ = original
    return guarded


def pytest_configure(config):
    """Configure pytest with custom markers and the assertion-diff guard."""
    from _pytest.assertion import util as assertion_util

    original = assertion_util.assertrepr_compare
    if not hasattr(original, "__wrapped__"):  # one guard, however often it runs
        assertion_util.assertrepr_compare = guard_assertrepr_compare(original)
        config.add_cleanup(
            lambda: setattr(assertion_util, "assertrepr_compare", original)
        )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "examples: mark test as example test")
    config.addinivalue_line("markers", "real_llm: mark test as requiring real LLM API")
