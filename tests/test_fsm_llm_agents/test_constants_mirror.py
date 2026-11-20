"""A shared constant has one value for every reader, whichever class it is set on.

Plan d454fd62, review gate 2 concern 1 (D-045). Before step 85 one class held each of
these values, so ``Defaults.MAX_OBSERVATION_LENGTH = 5000`` reached every reader. The
tool and policy layers then got their own constants classes and the agents classes
copied the values: an override of the agents class reached only the agents readers
(`ToolResult.observation` kept cutting at 2000). The classes now write a change of a
shared name through to every class that holds it (`MirroredConstants`).

The tests below set a shared name on ONE class at a time and compare what every reader
sees with what it sees when the name is set on ALL of its classes. They are order
independent: every override goes through a `pytest.MonkeyPatch` that is undone inside
the test.
"""

from __future__ import annotations

import ast
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import fsm_llm.agents.constants as agents_constants
import fsm_llm.policy.constants as policy_constants
import fsm_llm.tools as tools_pkg
import fsm_llm.tools.constants as tool_constants
from fsm_llm.agents.definitions import AgentConfig
from fsm_llm.core.definitions import AgentError
from fsm_llm.policy import HumanInTheLoop
from fsm_llm.tools import (
    ToolCall,
    ToolExecutionError,
    ToolRegistry,
    ToolResult,
    ToolTimedOut,
    call_with_timeout,
    check_timeout,
    require_int_limit,
)

SRC = Path(tools_pkg.__file__).resolve().parents[1]

# (class name, attribute) -> the classes that hold it, as module + class name
_AGENTS = agents_constants
_TOOLS = tool_constants
_POLICY = policy_constants

_SHARED: dict[tuple[str, str], tuple[Any, ...]] = {
    **{
        ("ContextKeys", a): (_AGENTS, _TOOLS)
        for a in ("TOOL_NAME", "TOOL_INPUT", "NO_TOOL")
    },
    **{
        ("Defaults", a): (_AGENTS, _TOOLS)
        for a in (
            "MAX_INT_LIMIT",
            "MAX_OBSERVATION_LENGTH",
            "MAX_TOOL_TIMEOUT_S",
            "TOOL_CONFIDENCE_THRESHOLD",
        )
    },
    ("Defaults", "CONFIDENCE_THRESHOLD"): (_AGENTS, _POLICY),
    **{
        ("ErrorMessages", a): (_AGENTS, _TOOLS)
        for a in (
            "TOOL_NOT_FOUND",
            "TOOL_EXECUTION_FAILED",
            "TOOL_TIMED_OUT",
            "TOOL_BAD_TIMEOUT",
        )
    },
    ("LogMessages", "TOOL_LATE_RESULT"): (_AGENTS, _TOOLS),
    **{
        ("LogMessages", a): (_AGENTS, _POLICY)
        for a in ("APPROVAL_REQUESTED", "APPROVAL_RESULT", "ESCALATION")
    },
}


class _Recorder:
    """A stand-in for a module's ``logger`` that keeps the messages."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def __getattr__(self, level: str) -> Callable[[str], None]:
        return self.messages.append


# ---------------------------------------------------------------------------
# One probe per shared name: a deterministic value that depends on every reader of
# the name (tool layer, policy layer, agents modules, synthesis tools).
# ---------------------------------------------------------------------------


def _probe_observation_length(mp: pytest.MonkeyPatch) -> object:
    from fsm_llm.agents.handlers.recording import record_tool_step
    from fsm_llm.agents.okf import _budget
    from fsm_llm.agents.summarization import _default_summary
    from fsm_llm.synthesis.tools import _room

    delta = record_tool_step(
        {},
        tool_name="t",
        shown_input={},
        observation="y" * 4000,
        status="success",
        thought="",
        action="",
    )
    return (
        len(ToolResult(tool_name="t", success=True, result="x" * 4000).observation),
        len(delta["observations"][0]),
        len(_default_summary(["z" * 4000])),
        _budget("t"),
        _room("t"),
    )


def _probe_timeout_limit(mp: pytest.MonkeyPatch) -> object:
    try:
        check_timeout("t", 10.0)
    except ToolExecutionError as exc:
        return str(exc)
    return "accepted"


def _probe_int_limit(mp: pytest.MonkeyPatch) -> object:
    seen = []
    try:
        seen.append(require_int_limit("n", 50, minimum=1))
    except AgentError as exc:
        seen.append(str(exc))
    try:
        AgentConfig(model="m", max_iterations=50)
        seen.append("accepted")
    except ValueError as exc:
        seen.append("rejected" if "at most" in str(exc) else str(exc))
    return seen


def _probe_selection_schema(mp: pytest.MonkeyPatch) -> object:
    schema = ToolRegistry().to_classification_schema()
    return (
        schema["confidence_threshold"],
        schema["fallback_intent"],
        [intent["name"] for intent in schema["intents"]],
    )


def _probe_tool_call_keys(mp: pytest.MonkeyPatch) -> object:
    from fsm_llm.agents.tools import tool_call_from_context

    call = tool_call_from_context(
        {"tool_name": "a", "tool_input": {"x": 1}, "tname": "b", "tinput": {"y": 2}}
    )
    return (call.tool_name, call.parameters, *_probe_selection_schema(mp))


def _probe_unknown_tool(mp: pytest.MonkeyPatch) -> object:
    return ToolRegistry().execute(ToolCall(tool_name="ghost", parameters={})).error


def _probe_failing_tool(mp: pytest.MonkeyPatch) -> object:
    registry = ToolRegistry()

    def boom() -> str:
        raise RuntimeError("kaput")

    registry.register_function(boom, name="boom", description="d")
    from fsm_llm.tools import registry as registry_module

    recorder = _Recorder()
    mp.setattr(registry_module, "logger", recorder)
    registry.execute(ToolCall(tool_name="boom", parameters={}))
    return list(recorder.messages)


def _wait_for_threads(before: int) -> None:
    """Let the timed-out tool threads of a probe end, so none logs into a later one."""
    deadline = time.monotonic() + 5
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.005)


def _probe_timed_out(mp: pytest.MonkeyPatch) -> object:
    before = threading.active_count()
    release = threading.Event()
    try:
        call_with_timeout(release.wait, "slow", 0.02)
    except ToolTimedOut as exc:
        return str(exc)
    finally:
        release.set()
        _wait_for_threads(before)
    return "finished"


def _probe_bad_timeout(mp: pytest.MonkeyPatch) -> object:
    try:
        check_timeout("t", -1)
    except ToolExecutionError as exc:
        return str(exc)
    return "accepted"


def _probe_late_result(mp: pytest.MonkeyPatch) -> object:
    from fsm_llm.tools import timeouts

    recorder = _Recorder()
    mp.setattr(timeouts, "logger", recorder)
    before = threading.active_count()
    release = threading.Event()
    with pytest.raises(ToolTimedOut):
        call_with_timeout(release.wait, "slow", 0.02)
    release.set()
    _wait_for_threads(before)
    return list(recorder.messages)


def _probe_approval_log(mp: pytest.MonkeyPatch) -> object:
    from fsm_llm.policy import hitl

    recorder = _Recorder()
    mp.setattr(hitl, "logger", recorder)
    manager = HumanInTheLoop(approval_callback=lambda request: True)
    manager.request_approval(ToolCall(tool_name="x", parameters={}), {})
    manager.escalate("why", {})
    return list(recorder.messages)


def _probe_confidence_default(mp: pytest.MonkeyPatch) -> object:
    # the HITL default is bound when the module loads, as before step 86; what can
    # move at run time is the value every class of the name holds
    return (
        _AGENTS.Defaults.CONFIDENCE_THRESHOLD,
        _POLICY.Defaults.CONFIDENCE_THRESHOLD,
    )


_PROBES: dict[
    tuple[str, str], tuple[Callable[[pytest.MonkeyPatch], object], object]
] = {
    ("Defaults", "MAX_OBSERVATION_LENGTH"): (_probe_observation_length, 5000),
    ("Defaults", "MAX_TOOL_TIMEOUT_S"): (_probe_timeout_limit, 5.0),
    ("Defaults", "MAX_INT_LIMIT"): (_probe_int_limit, 10),
    ("Defaults", "TOOL_CONFIDENCE_THRESHOLD"): (_probe_selection_schema, 0.9),
    ("Defaults", "CONFIDENCE_THRESHOLD"): (_probe_confidence_default, 0.6),
    ("ContextKeys", "NO_TOOL"): (_probe_tool_call_keys, "nothing"),
    ("ContextKeys", "TOOL_NAME"): (_probe_tool_call_keys, "tname"),
    ("ContextKeys", "TOOL_INPUT"): (_probe_tool_call_keys, "tinput"),
    ("ErrorMessages", "TOOL_NOT_FOUND"): (_probe_unknown_tool, "NF {name}"),
    ("ErrorMessages", "TOOL_EXECUTION_FAILED"): (
        _probe_failing_tool,
        "EF {name} {error}",
    ),
    ("ErrorMessages", "TOOL_TIMED_OUT"): (_probe_timed_out, "TO {name} {timeout}"),
    ("ErrorMessages", "TOOL_BAD_TIMEOUT"): (
        _probe_bad_timeout,
        "BT {name} {timeout} {limit}",
    ),
    ("LogMessages", "TOOL_LATE_RESULT"): (
        _probe_late_result,
        "LR {name} {timeout} {outcome}",
    ),
    ("LogMessages", "APPROVAL_REQUESTED"): (_probe_approval_log, "AQ {action}"),
    ("LogMessages", "APPROVAL_RESULT"): (_probe_approval_log, "AR {result} {action}"),
    ("LogMessages", "ESCALATION"): (_probe_approval_log, "ES {reason}"),
}

_CASES = sorted(_SHARED)


def _holders(key: tuple[str, str]) -> list[type]:
    return [getattr(module, key[0]) for module in _SHARED[key]]


def _label(cls: type) -> str:
    return cls.__module__.split(".")[1]


class TestEveryReaderSeesAnOverrideOnAnyClass:
    def test_the_table_covers_every_shared_name(self) -> None:
        assert set(_PROBES) == set(_SHARED)

    @pytest.mark.parametrize("key", _CASES, ids=lambda k: ".".join(k))
    def test_setting_one_class_equals_setting_all_of_them(
        self, key: tuple[str, str]
    ) -> None:
        probe, override = _PROBES[key]
        holders = _holders(key)
        attr = key[1]
        with pytest.MonkeyPatch.context() as mp:
            baseline = probe(mp)
        with pytest.MonkeyPatch.context() as mp:
            for cls in holders:
                mp.setattr(cls, attr, override)
            everywhere = probe(mp)
        assert everywhere != baseline, "the probe does not depend on the constant"
        for cls in holders:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(cls, attr, override)
                one_class = probe(mp)
            assert one_class == everywhere, (
                f"an override of {cls.__module__}.{cls.__name__}.{attr} did not reach "
                "every reader"
            )
            # the undo of the patch put every class and every reader back
            with pytest.MonkeyPatch.context() as mp:
                assert probe(mp) == baseline, (
                    f"restoring {cls.__module__}.{attr} did not restore every reader"
                )

    def test_the_reviewers_probe_the_agents_limit_reaches_the_tool_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_AGENTS.Defaults, "MAX_OBSERVATION_LENGTH", 5000)
        result = ToolResult(tool_name="t", success=True, result="x" * 4000)
        assert len(result.observation) == 4000

    def test_the_tool_limit_reaches_the_agents_readers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fsm_llm.agents.handlers.recording import record_tool_step

        monkeypatch.setattr(_TOOLS.Defaults, "MAX_OBSERVATION_LENGTH", 300)
        delta = record_tool_step(
            {},
            tool_name="t",
            shown_input={},
            observation="y" * 4000,
            status="success",
            thought="",
            action="",
        )
        assert len(delta["observations"][0]) <= 300
        assert _AGENTS.Defaults.MAX_OBSERVATION_LENGTH == 300


class TestEveryClassOfAnAttributeMoves:
    @pytest.mark.parametrize("key", _CASES, ids=lambda k: ".".join(k))
    def test_a_set_reaches_every_class_and_the_undo_restores_them(
        self, key: tuple[str, str]
    ) -> None:
        attr = key[1]
        holders = _holders(key)
        before = [vars(cls)[attr] for cls in holders]
        for source in holders:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(source, attr, "sentinel")
                assert [vars(cls)[attr] for cls in holders] == ["sentinel"] * len(
                    holders
                )
            assert [vars(cls)[attr] for cls in holders] == before

    @pytest.mark.parametrize("key", _CASES, ids=lambda k: ".".join(k))
    def test_a_plain_assignment_and_a_manual_restore_propagate_too(
        self, key: tuple[str, str]
    ) -> None:
        attr = key[1]
        holders = _holders(key)
        original = vars(holders[0])[attr]
        try:
            setattr(holders[-1], attr, "manual")
            assert all(vars(cls)[attr] == "manual" for cls in holders)
        finally:
            setattr(holders[-1], attr, original)
        assert all(vars(cls)[attr] is original for cls in holders)

    def test_a_delete_reaches_every_class_and_the_undo_brings_the_name_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # scratch classes: deleting and restoring a real constant would move it to the
        # end of `vars()` for the rest of the session (as on one plain class)
        meta = tool_constants.MirroredConstants

        class Low(metaclass=meta):
            LIMIT = 1
            OTHER = 2

        class High(metaclass=meta):
            LIMIT = Low.LIMIT
            OTHER = Low.OTHER

        tool_constants.mirror(Low, High, ("LIMIT",))
        monkeypatch.delattr(High, "LIMIT")
        assert "LIMIT" not in vars(Low)
        assert vars(Low)["OTHER"] == vars(High)["OTHER"] == 2
        monkeypatch.undo()
        assert vars(Low)["LIMIT"] == vars(High)["LIMIT"] == 1
        with pytest.raises(AttributeError):
            delattr(High, "MISSING")

    def test_a_name_the_class_does_not_share_stays_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_AGENTS.Defaults, "MAX_ITERATIONS", 7)
        assert not hasattr(_TOOLS.Defaults, "MAX_ITERATIONS")
        assert not hasattr(_POLICY.Defaults, "MAX_ITERATIONS")
        monkeypatch.setattr(_TOOLS.Defaults, "ONLY_HERE", 1, raising=False)
        assert not hasattr(_AGENTS.Defaults, "ONLY_HERE")

    def test_the_tool_and_policy_classes_do_not_mirror_each_other(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_POLICY.Defaults, "CONFIDENCE_THRESHOLD", 0.77)
        assert _TOOLS.Defaults.TOOL_CONFIDENCE_THRESHOLD == 0.4
        assert not hasattr(_TOOLS.Defaults, "CONFIDENCE_THRESHOLD")


class TestTheClassesStayPlainClasses:
    @pytest.mark.parametrize(
        "name", ["ContextKeys", "Defaults", "ErrorMessages", "LogMessages"]
    )
    def test_vars_holds_the_values_in_the_old_order_and_nothing_else(
        self, name: str
    ) -> None:
        for module in (_AGENTS, _TOOLS):
            cls = getattr(module, name)
            assert cls.__bases__ == (object,)
            names = [k for k in vars(cls) if not k.startswith("__")]
            assert all(k.isupper() for k in names), names
        # the registry lives in the module, never on the classes
        assert not any(
            "mirror" in k.lower() or "peer" in k.lower()
            for name in ("ContextKeys", "Defaults", "ErrorMessages", "LogMessages")
            for module in (_AGENTS, _TOOLS, _POLICY)
            if hasattr(module, name)
            for k in vars(getattr(module, name))
        )

    def test_the_agents_class_keeps_the_position_of_each_shared_name(self) -> None:
        order = [k for k in vars(_AGENTS.Defaults) if not k.startswith("__")]
        assert order.index("MAX_INT_LIMIT") < order.index("MAX_OBSERVATION_LENGTH")
        assert order.index("MAX_OBSERVATION_LENGTH") < order.index("MAX_TOOL_TIMEOUT_S")
        assert order.index("CONFIDENCE_THRESHOLD") < order.index(
            "TOOL_CONFIDENCE_THRESHOLD"
        )


# ---------------------------------------------------------------------------
# The registry is pinned against what the source really copies.
# ---------------------------------------------------------------------------

_LOWER_MODULES = {"fsm_llm.tools.constants", "fsm_llm.policy.constants"}


def _assigned_from_lower_layers() -> set[tuple[str, str, str, str]]:
    """(module, class, attribute, source module) for every class-body assignment of
    ``attr = _Alias.attr`` in the agents constants where ``_Alias`` is a class imported
    from a lower layer's constants module. Read from the source, not from a table."""
    found: set[tuple[str, str, str, str]] = set()
    for path in sorted((SRC / "agents" / "constants").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        alias: dict[str, tuple[str, str]] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module in _LOWER_MODULES:
                for item in node.names:
                    alias[item.asname or item.name] = (node.module or "", item.name)
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            for stmt in cls.body:
                if not (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Attribute)
                    and isinstance(stmt.value.value, ast.Name)
                    and stmt.value.value.id in alias
                ):
                    continue
                module, source_class = alias[stmt.value.value.id]
                assert stmt.targets[0].id == stmt.value.attr, (
                    f"{cls.name}.{stmt.targets[0].id} is assigned from a differently "
                    f"named attribute {stmt.value.attr}; the mirror pairs equal names"
                )
                assert source_class == cls.name, (cls.name, source_class)
                found.add((path.stem, cls.name, stmt.targets[0].id, module))
    return found


class TestTheRegistryMatchesTheSource:
    def test_the_source_copies_exactly_the_shared_table_of_this_file(self) -> None:
        derived = {
            (cls, attr, module)
            for _stem, cls, attr, module in _assigned_from_lower_layers()
        }
        table = {
            (cls, attr, module.__name__)
            for (cls, attr), modules in _SHARED.items()
            for module in modules
            if module is not _AGENTS
        }
        assert derived == table

    def test_the_registry_holds_exactly_the_copied_attributes(self) -> None:
        # scratch classes of other tests register too: judge the library's own only
        links = {
            key: peers
            for key, peers in tool_constants.mirror_links().items()
            if key[0].__module__.startswith("fsm_llm.")
        }
        expected: dict[tuple[type, str], set[type]] = {}
        for _stem, cls_name, attr, module in _assigned_from_lower_layers():
            lower = getattr(
                _TOOLS if module == "fsm_llm.tools.constants" else _POLICY, cls_name
            )
            copy = getattr(_AGENTS, cls_name)
            expected[(copy, attr)] = {lower}
            expected[(lower, attr)] = {copy}
        assert {k: set(v) for k, v in links.items()} == expected

    def test_a_registered_pair_must_hold_the_same_object(self) -> None:
        class A(metaclass=tool_constants.MirroredConstants):
            X = 1

        class B(metaclass=tool_constants.MirroredConstants):
            X = 2

        with pytest.raises(ValueError, match="X"):
            tool_constants.mirror(A, B, ("X",))
        assert (A, "X") not in tool_constants.mirror_links()

    def test_a_registered_class_needs_the_metaclass(self) -> None:
        class Plain:
            X = 1

        class Mirrored(metaclass=tool_constants.MirroredConstants):
            X = 1

        with pytest.raises(TypeError, match="MirroredConstants"):
            tool_constants.mirror(Plain, Mirrored, ("X",))

    def test_every_class_that_copies_or_is_copied_uses_the_metaclass(self) -> None:
        for (cls, _attr), _peers in tool_constants.mirror_links().items():
            assert isinstance(cls, tool_constants.MirroredConstants)

    def test_every_constants_module_that_assigns_from_a_lower_layer_registers(
        self,
    ) -> None:
        """A new copy in any other `agents/constants` module, or a new class there,
        fails here until it is registered (the source scan above reads them all)."""
        stems = {stem for stem, *_rest in _assigned_from_lower_layers()}
        assert stems == {"core"}


# ---------------------------------------------------------------------------
# Concurrent overrides (final REFLECT concern 4, D-053): one lock serialises
# set-and-propagate, so the copies of a shared name can never end up different.
# ---------------------------------------------------------------------------

_V0 = "start value"


class _UnlockedMirrored(tool_constants.MirroredConstants):
    """The metaclass of D-045 as it was before D-053: no lock around the write-through.

    It is the mutant that proves the interleaving scenarios below can tell a locked
    write from an unlocked one: the same scenario must diverge here and must not on
    the real metaclass."""

    def __setattr__(cls, name: str, value: Any) -> None:
        type.__setattr__(cls, name, value)
        for peer in tool_constants._MIRRORS.get((cls, name), ()):
            type.__setattr__(peer, name, value)

    def __delattr__(cls, name: str) -> None:
        type.__delattr__(cls, name)
        for peer in tool_constants._MIRRORS.get((cls, name), ()):
            if name in vars(peer):
                type.__delattr__(peer, name)


class _HookedRegistry(dict):  # type: ignore[type-arg]
    """A registry whose lookup of one armed key runs a callback once.

    The write-through looks the peers up right after it wrote its own class and right
    before it writes the peers: the one point where another thread must not get in.
    The callback runs on the writing thread at that point, so the interleaving is
    exact, with no timing involved."""

    armed: tuple[type, str] | None = None
    callback: Callable[[], None] | None = None

    def get(self, key: Any, default: Any = None) -> Any:
        if key == self.armed and self.callback is not None:
            callback, self.callback = self.callback, None
            callback()
        return super().get(key, default)


def _group(
    monkeypatch: pytest.MonkeyPatch, meta: type
) -> tuple[type, type, _HookedRegistry]:
    """Two classes of one throwaway group on a private hooked registry."""
    registry = _HookedRegistry(tool_constants.mirror_links())
    monkeypatch.setattr(tool_constants, "_MIRRORS", registry)
    a = meta("A", (), {"X": _V0})
    b = meta("B", (), {"X": _V0})
    tool_constants.mirror(a, b, ("X",))
    return a, b, registry


def _wait_finished(thread: threading.Thread, seconds: float) -> None:
    """Wait until *thread* ends or *seconds* pass, without a timed join (D-057)."""
    deadline = time.monotonic() + seconds
    while thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)


def _interleave(
    registry: _HookedRegistry,
    armed: tuple[type, str],
    first: Callable[[], None],
    second: Callable[[], None],
) -> None:
    """Run *first* and make *second* start on another thread at the hooked point.

    Without a lock *second* runs to its end inside the hook (it is not alive after
    the sleep); with one it blocks until *first* is done (it is still alive after
    the sleep and *first* goes on). Either way both are finished when this returns.
    No timed Thread.join is issued here: the competitor can be blocked on a lock
    this thread holds, and on CPython 3.15 Windows such a join hung a CI leg for 45
    minutes (plan ace7a8c8 D-057). The waits are a sleep and a bounded poll."""
    other: list[threading.Thread] = []

    def compete() -> None:
        started = threading.Event()

        def run() -> None:
            started.set()
            second()

        thread = threading.Thread(target=run, daemon=True)
        other.append(thread)
        thread.start()
        assert started.wait(5)
        time.sleep(0.3)

    registry.armed, registry.callback = armed, compete
    first()
    _wait_finished(other[0], 5)
    assert not other[0].is_alive(), "the competing write never finished"


def _state(a: type, b: type) -> tuple[object, object]:
    missing = "<absent>"
    return vars(a).get("X", missing), vars(b).get("X", missing)


class TestConcurrentOverridesStayMirrored:
    """The deterministic scenarios are the proof; the stress test below is a
    probabilistic second line (it can only fail when a bad interleaving happens)."""

    # (name, first op on A, second op on B, final state when the writes are serial,
    # the second one running after the first)
    _SCENARIOS = (
        ("set against set", "set", "set", ("two", "two")),
        ("delete against set", "del", "set", ("two", "two")),
        ("set against delete", "set", "del", ("<absent>", "<absent>")),
    )

    @staticmethod
    def _op(cls: type, kind: str, value: str) -> Callable[[], None]:
        if kind == "set":
            return lambda: setattr(cls, "X", value)
        return lambda: delattr(cls, "X")

    @pytest.mark.parametrize(
        ("label", "first_kind", "second_kind", "final"),
        _SCENARIOS,
        ids=[s[0] for s in _SCENARIOS],
    )
    def test_a_competing_write_cannot_slip_between_the_class_and_its_peers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        first_kind: str,
        second_kind: str,
        final: tuple[str, str],
    ) -> None:
        a, b, registry = _group(monkeypatch, tool_constants.MirroredConstants)
        _interleave(
            registry,
            (a, "X"),
            self._op(a, first_kind, "one"),
            self._op(b, second_kind, "two"),
        )
        assert _state(a, b) == final, label

    @pytest.mark.parametrize(
        ("label", "first_kind", "second_kind", "final"),
        _SCENARIOS,
        ids=[s[0] for s in _SCENARIOS],
    )
    def test_the_same_scenario_diverges_without_the_lock(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        first_kind: str,
        second_kind: str,
        final: tuple[str, str],
    ) -> None:
        """Mutant: the unlocked metaclass leaves the copies different, so the scenario
        above would fail when the lock of ``MirroredConstants`` is removed."""
        a, b, registry = _group(monkeypatch, _UnlockedMirrored)
        _interleave(
            registry,
            (a, "X"),
            self._op(a, first_kind, "one"),
            self._op(b, second_kind, "two"),
        )
        left, right = _state(a, b)
        assert left != right, (label, left, right)

    def test_the_lock_is_re_entrant(self) -> None:
        """A hook that writes a second mirrored name from inside a write (a descriptor,
        a ``__set_name__``) must not block the thread on itself: the lock is an
        ``RLock``. Checked by type, not by trying: a plain ``Lock`` would hang the
        process on the failing run."""
        assert isinstance(tool_constants._LOCK, type(threading.RLock()))

    def test_stress_two_threads_never_leave_the_copies_different(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """3,000 forced trials (barrier, a switch interval of 1e-7 s), 0 divergences.

        Probabilistic on its own: before D-053 one trial in about 3,000 diverged, so a
        pass here proves little by itself. The deterministic scenarios above are the
        RED proof of the lock."""
        assert _stress_divergences(monkeypatch, tool_constants.MirroredConstants) == 0


def _stress_divergences(
    monkeypatch: pytest.MonkeyPatch, meta: type, trials: int = 3000
) -> int:
    """Trials in which two threads, each writing a different value to one class of a
    two-class group at the same moment, left the classes different.

    Three parties meet at each barrier (the two writers and the checking thread), so a
    trial is read only after both writes are over and the next one starts only after the
    read: the one thing that can make the classes differ is the write-through itself."""
    import sys

    a, b, _registry = _group(monkeypatch, meta)
    start, done = threading.Barrier(3), threading.Barrier(3)
    errors: list[BaseException] = []

    def worker(cls: type, base: int) -> None:
        try:
            for i in range(trials):
                start.wait(10)
                cls.X = base + i  # type: ignore[attr-defined]
                done.wait(10)
        except BaseException as exc:  # a broken barrier ends the test, not hangs it
            errors.append(exc)
            start.abort()
            done.abort()

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-7)
    threads = [
        threading.Thread(target=worker, args=(a, 0), daemon=True),
        threading.Thread(target=worker, args=(b, 10**6), daemon=True),
    ]
    divergent = 0
    try:
        for thread in threads:
            thread.start()
        for _ in range(trials):
            start.wait(10)
            done.wait(10)
            left, right = _state(a, b)
            divergent += left != right
    finally:
        sys.setswitchinterval(old)
        for thread in threads:
            thread.join(10)
    assert not errors, errors
    return divergent
