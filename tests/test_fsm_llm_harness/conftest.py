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
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.definitions import FSMDefinition
from fsm_llm.transition_evaluator import TransitionEvaluator
from fsm_llm_agents.definitions import AgentResult, ApprovalRequest
from fsm_llm_harness import build_harness_fsm
from fsm_llm_harness.harness import (
    _APPROVAL_CLOSE,
    _APPROVAL_LEASH,
    _APPROVAL_PLAN,
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

#: Every gate name the driver may emit.
APPROVAL_GATES = (APPROVAL_PLAN, APPROVAL_CLOSE, APPROVAL_LEASH)

#: One entry of a worker script: either a literal reply spec or a callable
#: taking the ``RoleRequest`` and returning one.
ScriptEntry = Mapping[str, Any] | Callable[[RoleRequest], Mapping[str, Any]]


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

    def run(self, goal: str = "test goal", **kwargs: Any) -> AgentResult:
        """Run the driver and return its result."""
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
    * remaining keyword arguments go to ``HarnessAgent.__init__`` (thresholds,
      ``config``, ...).  The mocked LLM is wired through ``**api_kwargs``.
    """

    def _make(
        script: Mapping[str, ScriptEntry] | None = None,
        *,
        worker: Any = _UNSET,
        approvals: ApprovalRecorder | None = None,
        **agent_kwargs: Any,
    ) -> HarnessUnderTest:
        recorder: RecordingWorker | None
        if worker is _UNSET:
            recorder = RecordingWorker(script)
        else:
            recorder = worker
        callback = approvals if approvals is not None else ApprovalRecorder()
        agent = HarnessAgent(
            worker_factory=recorder,
            approval_callback=callback,
            llm_interface=MockLLM2Interface(),
            **agent_kwargs,
        )
        return HarnessUnderTest(agent=agent, worker=recorder, approvals=callback)

    return _make
