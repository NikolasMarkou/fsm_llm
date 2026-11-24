"""The rows of the K7 review fix round (plan baa37f40, steps K7R.1 to K7R.4, D-055 to D-058).

Each row reproduces one finding of ``findings/review-k7.md`` with the reviewer's probe shape, on a
REAL ``ExecutableTargets`` and ``RunHost`` over a code-backed stub (and, where the finding names
one, a real executable): a workflow-kind executable is no delegated target and the call's
``max_steps`` reaches an accepting run (K7R.1, warnings 7 and 1), and the rows of the later steps
(the stream run loops, a popped frame, the finished results, the launcher guards) are appended
by their steps. Rules of the rows (LESSONS ace7a8c8, D-048): no ``Thread.join``, events and
bounded waits, no ordered scripted reply on two threads, thread rows skipped under a tracer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm import API
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions.executable import Contract
from fsm_llm.core.results import RunResult
from fsm_llm.engine.api import delegated
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.code_backed import CodeBackedExecutable
from fsm_llm.engine.executables.runs import ExecutableTargets
from fsm_llm.reasoning.executables import SolveExecutable
from fsm_llm.runtime import EventLog
from tests.test_fsm_llm.test_call_delegated import _initial_call
from tests.test_fsm_llm.test_call_delegated_settle import _TARGET, _Probe
from tests.test_fsm_llm.test_call_settle import _Gate
from tests.test_fsm_llm.test_call_state import _call, _parent

_ROOT = Path(__file__).resolve().parents[2]


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(_ROOT / "src")


@pytest.fixture(autouse=True)
def _short_wait(monkeypatch):
    monkeypatch.setattr(delegated, "DELEGATED_WAIT_SECONDS", 0.05)


def _build(
    tmp_path: Path,
    *executables: Any,
    config: dict[str, Any],
    context: RunContext | None = None,
    log: bool = True,
    **state_over: Any,
):
    """A bare API whose conversation begins in one delegated call state."""
    registry = ExecutableRegistry()
    for executable in executables:
        registry.register(executable)
    event_log = EventLog(tmp_path / "logs") if log else None
    base = RunContext(event_log=event_log) if context is None else context
    targets = ExecutableTargets(registry, base)
    state_over.setdefault("response_instructions", "")
    api = API.from_definition(
        _initial_call(_parent(config, **state_over)),
        llm_interface=_Gate(default_response="r"),
        targets=targets,
        **({} if event_log is None else {"event_log": event_log}),
    )
    return api, targets, event_log


def _drive(api: API, cid: str, steps: int = 30) -> None:
    api.run_until_terminal(cid, max_steps=steps)


class _Workflow(CodeBackedExecutable):
    """A host-registered code-backed executable of the ``workflow`` kind."""

    def __init__(self) -> None:
        super().__init__(
            id=_TARGET,
            kind="workflow",
            impl="tests.probe.Workflow",
            contract=Contract(inputs=["topic"], outputs=["findings"]),
        )
        self.runs = 0

    async def execute(self, inputs, objects, context, identity) -> RunResult:
        self.runs += 1
        return RunResult(
            status=Status.OK, outputs={"findings": ["f"]}, answer="a", identity=identity
        )


class TestAWorkflowKindIsNoDelegatedTarget:
    """K7R.1, review warning 7 (D-053: delegated mode is pattern targets only)."""

    def test_the_registry_does_not_resolve_a_workflow_kind(self, tmp_path) -> None:
        """Kills: ``_DELEGATED_KINDS`` holding ``workflow`` (the registry answers a target)."""
        workflow = _Workflow()
        api, targets, log = _build(
            tmp_path, workflow, config=_call(target=_TARGET, inputs=["topic"])
        )
        try:
            assert targets.resolve_run(_TARGET, input_keys=["topic"]) is None
        finally:
            api.close()
            assert log is not None
            log.close()

    def test_a_call_to_a_workflow_ends_unknown_target_and_runs_nothing(
        self, tmp_path
    ) -> None:
        workflow = _Workflow()
        api, _targets, log = _build(
            tmp_path,
            workflow,
            config=_call(
                target=_TARGET, inputs=["topic"], returns={"found": "findings"}
            ),
        )
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            _drive(api, cid)
            record = api.get_data(cid)["call_result"]
            assert (record["status"], record["reason"]) == ("failed", "unknown_target")
            assert workflow.runs == 0
        finally:
            api.close()
            assert log is not None
            log.close()

    def test_a_pattern_kind_still_resolves(self, tmp_path) -> None:
        probe = _Probe()
        api, targets, log = _build(
            tmp_path, probe, config=_call(target=_TARGET, inputs=["topic"])
        )
        try:
            assert targets.resolve_run(_TARGET, input_keys=["topic"]) is not None
        finally:
            api.close()
            assert log is not None
            log.close()


class _Refusing(_Probe):
    refused_budgets = ("max_steps",)


def _steps_config(**over: Any) -> dict[str, Any]:
    return _call(
        target=_TARGET,
        inputs=["topic"],
        returns={"found": "findings"},
        max_steps=3,
        **over,
    )


class TestTheCallStepBudgetReachesTheRun:
    """K7R.1, review warning 1: the call's ``max_steps`` was dropped in delegated mode."""

    def test_an_accepting_executable_receives_the_budget(self, tmp_path) -> None:
        """The reviewer's probe: ``max_steps: 3`` and ``context.max_steps is None`` in the worker.
        Kills: ``ExecutableTargets.start`` deriving no ``max_steps``."""
        probe = _Probe()
        api, _targets, log = _build(tmp_path, probe, config=_steps_config())
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            _drive(api, cid)
            assert api.get_data(cid)["call_result"]["status"] == "ok"
            assert [run["context"].max_steps for run in probe.runs] == [3]
        finally:
            api.close()
            assert log is not None
            log.close()

    def test_reasoning_solve_receives_the_budget(self, tmp_path, monkeypatch) -> None:
        """``reasoning.solve`` accepts a step budget since step 16 (``refused_budgets=()``): it
        must be the one that RECEIVES it. Kills: the budget dropped by ``start``."""
        seen: list[Any] = []

        def solve(problem, initial_context, context):
            seen.append((problem, context.max_steps))
            return "the solution", {}

        monkeypatch.setattr(SolveExecutable, "_solve", staticmethod(solve))
        api, _targets, log = _build(
            tmp_path,
            SolveExecutable(),
            config=_call(
                target="reasoning.solve",
                inputs=["problem"],
                returns={"answer": "solution"},
                max_steps=3,
            ),
        )
        try:
            cid, _ = api.start_conversation({"problem": "why is the sky blue"})
            _drive(api, cid)
            data = api.get_data(cid)
            assert data["call_result"]["status"] == "ok", data["call_result"]
            assert data["answer"] == "the solution"
            assert seen == [("why is the sky blue", 3)]
        finally:
            api.close()
            assert log is not None
            log.close()

    def test_a_call_with_no_budget_gives_the_run_the_host_limit_only(
        self, tmp_path
    ) -> None:
        probe = _Probe()
        config = _call(target=_TARGET, inputs=["topic"], returns={"found": "findings"})
        api, _targets, _log = _build(
            tmp_path, probe, config=config, context=RunContext(max_steps=7), log=False
        )
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            _drive(api, cid)
            assert [run["context"].max_steps for run in probe.runs] == [7]
        finally:
            api.close()

    def test_the_call_budget_is_narrowed_against_the_host_limit(self, tmp_path) -> None:
        """A call above the host limit runs under the host limit (it does not fail in the
        worker with the ``ValueError`` of ``derive``); a call below it keeps its own."""
        for call_steps, expected in ((5, 2), (1, 1)):
            probe = _Probe()
            api, _targets, _log = _build(
                tmp_path / str(call_steps),
                probe,
                config=_call(
                    target=_TARGET,
                    inputs=["topic"],
                    returns={"found": "findings"},
                    max_steps=call_steps,
                ),
                context=RunContext(max_steps=2),
                log=False,
            )
            try:
                cid, _ = api.start_conversation({"topic": "why"})
                _drive(api, cid)
                assert api.get_data(cid)["call_result"]["status"] == "ok"
                assert [run["context"].max_steps for run in probe.runs] == [expected]
            finally:
                api.close()

    def test_an_executable_that_refuses_a_step_budget_is_refused_at_resolution(
        self, tmp_path
    ) -> None:
        """The pattern refusal stays (``refused_budgets``): ``unknown_target``, nothing started."""
        probe = _Refusing()
        api, _targets, log = _build(tmp_path, probe, config=_steps_config())
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            _drive(api, cid)
            record = api.get_data(cid)["call_result"]
            assert (record["status"], record["reason"]) == ("failed", "unknown_target")
            assert probe.runs == []
        finally:
            api.close()
            assert log is not None
            log.close()

    def test_the_real_pattern_names_max_steps_in_its_refused_budgets(self) -> None:
        from fsm_llm.agents.executables import PatternExecutable

        assert "max_steps" in PatternExecutable("react").refused_budgets
        assert "max_steps" not in SolveExecutable().refused_budgets
