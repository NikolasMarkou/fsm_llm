"""Fixtures for the ``fsm_llm_harness`` unit tests.

Everything here is deterministic and offline: the FSM is driven by
``MockLLM2Interface`` (the repo's 2-pass mock, ``tests/conftest.py``) and every
worker dispatch goes through a recording stand-in.  No network, no sleeps, no
ollama.

Design notes that matter for the tests that use these fixtures:

* :class:`RecordingWorker` records the **whole** ``RoleRequest``, not a
  hand-picked tuple.  A mock whose attribute surface is narrower than the real
  object hides exactly the bugs these tests exist to catch
  (``plans/LESSONS.md``, fixture anti-patterns).
* :class:`ApprovalRecorder` records the real
  :class:`~fsm_llm_agents.definitions.ApprovalRequest` objects the driver
  produced, so a test can assert the callback **was consulted** rather than
  only that a gate stayed shut (decisions.md D-023).
* Nothing here pre-satisfies a gate.  The default script is empty, so a state
  whose worker writes no gate flag BLOCKS -- the interesting case stays
  interesting.
* The FSM's own Pass-1 extraction is a **first-class, selectable axis**
  (``make_harness(..., extraction_data=...)``), because it is the SECOND writer
  into gate context and the one review C4 found the suite had mocked into
  silence.  :data:`FABRICATED_DRIVER_OWNED` is derived from the driver-owned
  tables rather than hand-listed, so a driver-owned key added later is covered
  by ``TestExtractionCannotOpenAGate`` without anyone remembering to add it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from fsm_llm.definitions import FSMDefinition
from fsm_llm.logging import logger
from fsm_llm.transition_evaluator import TransitionEvaluator
from fsm_llm_agents.definitions import AgentResult, ApprovalRequest
from fsm_llm_harness import build_harness_fsm
from fsm_llm_harness.constants import (
    DRIVER_OWNED_SEEDS,
    DRIVER_OWNED_UNSET,
    ContextKeys,
)
from fsm_llm_harness.harness import (
    _APPROVAL_CLOSE,
    _APPROVAL_LEASH,
    _APPROVAL_PLAN,
    _APPROVAL_REVERT,
    _DISPATCH_LOCAL,
    HarnessAgent,
    RoleRequest,
)
from tests.conftest import MockLLM2Interface

#: The three human approval gates, re-exported under public names.  Imported
#: from the driver rather than re-typed so a rename cannot leave the tests
#: asserting a string nothing produces; ``test_approval_gate_names_are_stable``
#: pins the literal wire values separately.
APPROVAL_PLAN = _APPROVAL_PLAN
APPROVAL_CLOSE = _APPROVAL_CLOSE
APPROVAL_LEASH = _APPROVAL_LEASH
#: The fourth gate: the leash-cap revert, consulted only when the caller
#: supplied a ``revert_callback`` (D-039).
APPROVAL_REVERT = _APPROVAL_REVERT

#: Every gate name the driver may emit.
APPROVAL_GATES = (APPROVAL_PLAN, APPROVAL_CLOSE, APPROVAL_LEASH, APPROVAL_REVERT)

#: One entry of a worker script: either a literal reply spec or a callable
#: taking the ``RoleRequest`` and returning one.
ScriptEntry = Mapping[str, Any] | Callable[[RoleRequest], Mapping[str, Any]]


# ---------------------------------------------------------------------------
# The fabricating-LLM axis (review C1 / C4, success criterion 18)
# ---------------------------------------------------------------------------

#: A fabricated counter large enough to satisfy every ``>=`` gate the protocol
#: has (``findings_count >= 3``) and to corrupt every ``<`` one.
_FABRICATED_INT = 99

#: A fabricated string for the driver-owned keys that carry free prose or a
#: filesystem root.  Absolute and outside any test root on purpose: if it ever
#: reaches ``plan_dir`` the protocol's own memory has been re-pointed.
_FABRICATED_STR = "/fabricated/by/the/model"


def _fabricated_value(key: str) -> Any:
    """Return a value for *key* that an LLM could plausibly hallucinate.

    Interface contract (one call site, the comprehension below; kept as a
    function so the type mapping is stated once and readably):
        - Parameter: a driver-owned context key.
        - Returns ``True`` for a boolean-seeded key, :data:`_FABRICATED_INT` for
          an integer-seeded one, and :data:`_FABRICATED_STR` for a key whose
          default is absence (``DRIVER_OWNED_UNSET``).
        - Never raises; the value is always DIFFERENT from the driver's own.
    """
    seed = DRIVER_OWNED_SEEDS.get(key)
    if isinstance(seed, bool):
        return True
    if isinstance(seed, int):
        return _FABRICATED_INT
    return _FABRICATED_STR


# DECISION plan-2026-07-21T125237-191b2eb2/D-056
# DERIVED from the driver-owned tables, and the fixture default stays SILENT.
# Two things here look wrong and are deliberate:
#   1. Do NOT replace this comprehension with a literal dict of the nine gate
#      flags review C1 enumerated. The point of deriving it is that a
#      driver-owned key added by a later step gets a writer-provenance test
#      without anyone remembering to write one -- a hand-listed copy is exactly
#      the drift that let C1 (nine keys, diagnosed as one) and C2 (a hand-kept
#      tool-scope table) through.
#   2. Do NOT flip `make_harness`'s `extraction_data` default to this table "so
#      the hostile case is always on". That would make all 139 pre-existing
#      tests simultaneously guard tests: one guard regression would fail ~139
#      of them at once with no diagnostic value, and a genuine worker-seam
#      regression would arrive buried in the noise. What review C4 punished was
#      the hostile case being UNREACHABLE, not the default being gentle -- and
#      it is reachable, parametrised over every key, and pinned against vacuity
#      by `test_seeding_is_what_holds_the_gate`.
# See decisions.md D-056.
#: Every driver-owned context key mapped to a value the driver must never let
#: the FSM's own Pass-1 extraction write.
FABRICATED_DRIVER_OWNED: Mapping[str, Any] = MappingProxyType(
    {key: _fabricated_value(key) for key in (*DRIVER_OWNED_SEEDS, *DRIVER_OWNED_UNSET)}
)


# ---------------------------------------------------------------------------
# Global-state hygiene
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_dispatch_guard():
    """Clear the module-level re-entrancy flag around every test.

    The guard is a ``threading.local`` shared by every ``HarnessAgent``
    (decisions.md D-014).  A test that leaves it set would make an unrelated
    later test fail under random ordering, so it is cleared on both sides.
    """
    _DISPATCH_LOCAL.role = None
    yield
    _DISPATCH_LOCAL.role = None


# ---------------------------------------------------------------------------
# FSM fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harness_fsm_dict() -> dict[str, Any]:
    """The FSM-JSON dict at default thresholds."""
    return build_harness_fsm("exercise the harness protocol")


@pytest.fixture
def harness_fsm(harness_fsm_dict: dict[str, Any]) -> FSMDefinition:
    """The validated ``FSMDefinition`` at default thresholds."""
    return FSMDefinition(**harness_fsm_dict)


@pytest.fixture
def evaluator() -> TransitionEvaluator:
    """The real core transition evaluator, at its default configuration."""
    return TransitionEvaluator()


# ---------------------------------------------------------------------------
# Filesystem fixtures (plan directory / workspace roots)
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_dir(tmp_path: Path) -> Path:
    """A bootstrapped, empty plan directory with its two subdirectories."""
    directory = tmp_path / "plans" / "plan-2026-07-21T000000-testplan"
    (directory / "findings").mkdir(parents=True)
    (directory / "checkpoints").mkdir(parents=True)
    return directory


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty workspace root, separate from the plan directory."""
    directory = tmp_path / "workspace"
    directory.mkdir()
    return directory


# DECISION plan-2026-07-21T125237-191b2eb2/D-057
# Do NOT "simplify" any log assertion in this package to pytest's `caplog`.
# The harness logs through loguru (`fsm_llm.logging`), which does not propagate
# to the stdlib `logging` tree at all, so a `caplog`-based assertion passes
# whether the message was emitted or not -- a test that cannot fail, which is
# the same class of false green as review C4. A loguru sink is the only way to
# observe these messages. See decisions.md D-057.
@pytest.fixture
def captured_logs() -> Any:
    """Every loguru message emitted during the test, as plain strings."""
    lines: list[str] = []
    sink_id = logger.add(lines.append, level="DEBUG", format="{level}|{message}")
    try:
        yield lines
    finally:
        logger.remove(sink_id)


@pytest.fixture
def roots(plan_dir: Path, workspace: Path) -> dict[str, str]:
    """The two filesystem-as-memory roots, shaped for ``run(initial_context=)``.

    Hand this to ``make_harness(..., roots=roots)`` for any test that cares
    about the plan directory.  Without it a dispatch has ``plan_dir=None``,
    which is a real production shape (the driver degrades to "no plan-file
    tools") but makes every plan-directory assertion vacuous.
    """
    return {
        ContextKeys.PLAN_DIR: str(plan_dir),
        ContextKeys.WORKSPACE_ROOT: str(workspace),
    }


# ---------------------------------------------------------------------------
# Recording worker factory
# ---------------------------------------------------------------------------


class RecordingWorker:
    """A ``WorkerFactory`` that records every dispatch and replies to script.

    Interface contract (this class has 2+ call sites across the suite):

    * ``script`` maps a state id to either a reply spec or a callable taking
      the :class:`RoleRequest` and returning one.  A reply spec understands
      ``success`` (default ``True``), ``ctx`` (the ``final_context`` handed
      back), ``answer`` and ``raises``.
    * A state absent from the script gets a successful, empty reply -- which
      writes no gate flag, so the state's exit gate stays BLOCKED.
    * ``requests`` holds every :class:`RoleRequest` in dispatch order;
      :attr:`calls` projects it to ``(state, iteration, step_number,
      fix_attempts)`` tuples for assertions.
    """

    def __init__(self, script: Mapping[str, ScriptEntry] | None = None) -> None:
        self.script: dict[str, ScriptEntry] = dict(script or {})
        self.requests: list[RoleRequest] = []

    @property
    def calls(self) -> list[tuple[str, int, int, int]]:
        """``(state, iteration, step_number, fix_attempts)`` per dispatch."""
        return [
            (
                request.state,
                request.iteration,
                request.step_number,
                request.fix_attempts,
            )
            for request in self.requests
        ]

    @property
    def states(self) -> list[str]:
        """The state of each dispatch, in order."""
        return [request.state for request in self.requests]

    @property
    def roles(self) -> list[str]:
        """The role of each dispatch, in order."""
        return [request.role for request in self.requests]

    def calls_for(self, state: str) -> list[tuple[str, int, int, int]]:
        """Every dispatch tuple recorded for *state*."""
        return [call for call in self.calls if call[0] == state]

    def count_for(self, state: str) -> int:
        """How many times *state*'s worker was dispatched."""
        return len(self.calls_for(state))

    def __call__(self, request: RoleRequest) -> AgentResult:
        self.requests.append(request)
        entry = self.script.get(request.state, {})
        if callable(entry):
            entry = entry(request)
        error = entry.get("raises")
        if error is not None:
            raise error
        return AgentResult(
            answer=entry.get("answer", f"{request.role} completed"),
            success=bool(entry.get("success", True)),
            final_context=dict(entry.get("ctx", {})),
        )


# ---------------------------------------------------------------------------
# Recording approval callback
# ---------------------------------------------------------------------------


class ApprovalRecorder:
    """An ``ApprovalCallback`` that records every gate it is consulted on.

    Interface contract (2+ call sites):

    * ``verdicts`` maps a gate ``tool_name`` to either a bool or a callable
      taking the 1-based call index **for that gate** and returning a bool
      (used to grant the first N leash-continues and then deny).
    * ``default`` answers any gate not named in ``verdicts``.
    * ``raises`` makes the callback itself fail, which the driver must treat as
      DENIED rather than as an open gate.
    * ``requests`` holds the real :class:`ApprovalRequest` objects, so a test
      can assert consultation, ordering and parameters -- not merely the
      outcome (decisions.md D-023).
    """

    def __init__(
        self,
        verdicts: Mapping[str, bool | Callable[[int], bool]] | None = None,
        *,
        default: bool = True,
        raises: BaseException | None = None,
    ) -> None:
        self.verdicts: dict[str, bool | Callable[[int], bool]] = dict(verdicts or {})
        self.default = default
        self.raises = raises
        self.requests: list[ApprovalRequest] = []

    @property
    def names(self) -> list[str]:
        """The gate name of each consultation, in order."""
        return [request.tool_name for request in self.requests]

    def count(self, gate: str) -> int:
        """How many times *gate* was put to the callback."""
        return self.names.count(gate)

    def __call__(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        verdict = self.verdicts.get(request.tool_name, self.default)
        if callable(verdict):
            return verdict(self.count(request.tool_name))
        return verdict


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


@dataclass
class HarnessUnderTest:
    """One configured driver plus the two recorders wired into it."""

    agent: HarnessAgent
    worker: RecordingWorker | None
    approvals: ApprovalRecorder
    llm: MockLLM2Interface
    default_context: dict[str, Any] = field(default_factory=dict)

    def run(self, goal: str = "test goal", **kwargs: Any) -> AgentResult:
        """Run the driver and return its result.

        ``make_harness(roots=...)`` is merged UNDER any explicit
        ``initial_context``, so a test can still override a root it was given.
        """
        if self.default_context:
            supplied = kwargs.pop("initial_context", None) or {}
            kwargs["initial_context"] = {**self.default_context, **supplied}
        return self.agent.run(goal, **kwargs)


#: Sentinel distinguishing "no worker argument given" from an explicit
#: ``worker=None`` (which exercises the degrade path).
_UNSET = object()


@pytest.fixture
def make_harness() -> Callable[..., HarnessUnderTest]:
    """Factory building a ``HarnessAgent`` wired to the two recorders.

    Interface contract::

        harness = make_harness(script, approvals=..., worker=None, **agent_kwargs)

    * ``script`` -- a :class:`RecordingWorker` script (see that class).
    * ``worker`` -- pass ``None`` explicitly to run with no worker factory at
      all (the degrade path); omit it to get a scripted recorder.
    * ``approvals`` -- an :class:`ApprovalRecorder`; the default approves every
      gate, so a test that cares about denial must say so.
    * ``extraction_data`` -- what the FSM's own Pass-1 extraction "finds" in
      the user's prose.  The DEFAULT IS ``None`` (a silent LLM); pass
      :data:`FABRICATED_DRIVER_OWNED` (or a one-key slice of it) to select the
      hostile axis.  See ``decisions.md`` D-056 for why the default is silent
      rather than fabricating.
    * ``roots`` -- ``{plan_dir, workspace_root}`` merged into every ``run()``'s
      ``initial_context``; use the ``roots`` fixture.
    * remaining keyword arguments go to ``HarnessAgent.__init__`` (thresholds,
      ``config``, ...).  The mocked LLM is wired through ``**api_kwargs``.
    """

    def _make(
        script: Mapping[str, ScriptEntry] | None = None,
        *,
        worker: Any = _UNSET,
        approvals: ApprovalRecorder | None = None,
        extraction_data: Mapping[str, Any] | None = None,
        roots: Mapping[str, str] | None = None,
        **agent_kwargs: Any,
    ) -> HarnessUnderTest:
        recorder: RecordingWorker | None
        if worker is _UNSET:
            recorder = RecordingWorker(script)
        else:
            recorder = worker
        callback = approvals if approvals is not None else ApprovalRecorder()
        llm = MockLLM2Interface(
            extraction_data=dict(extraction_data) if extraction_data else None
        )
        agent = HarnessAgent(
            worker_factory=recorder,
            approval_callback=callback,
            llm_interface=llm,
            **agent_kwargs,
        )
        return HarnessUnderTest(
            agent=agent,
            worker=recorder,
            approvals=callback,
            llm=llm,
            default_context=dict(roots or {}),
        )

    return _make
