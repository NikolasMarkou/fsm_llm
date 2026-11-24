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

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm import API
from fsm_llm.aio import aoffload
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions import FSMDefinition, FSMError
from fsm_llm.core.definitions.executable import Contract
from fsm_llm.core.results import RunResult
from fsm_llm.engine.api import delegated
from fsm_llm.engine.api.delegated import RunMark, flush_run_ends
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.code_backed import CodeBackedExecutable
from fsm_llm.engine.executables.runs import ExecutableTargets, RunHost
from fsm_llm.engine.kinds.targets import ResolvedTarget
from fsm_llm.reasoning.executables import SolveExecutable
from fsm_llm.runtime import EventLog
from tests.conftest import current_timing_factor
from tests.test_fsm_llm.test_async_engine import _CHAT
from tests.test_fsm_llm.test_call_delegated import _initial_call
from tests.test_fsm_llm.test_call_delegated_settle import _TARGET, _lock_free, _Probe
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


# ---- K7R.2: the stream run loops wait, a popped frame cancels and ends its run ----------------


class _Slow(_Probe):
    """A pattern stub that works for ``seconds`` (a worker-thread sleep) and then returns."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds

    async def execute(self, inputs, objects, context, identity):
        self.runs.append({"context": context})
        await aoffload(time.sleep, self.seconds)
        return RunResult(
            status=Status.OK,
            outputs={"findings": ["f"]},
            answer="a",
            identity=identity,
        )


def _slow_flow(tmp_path: Path, seconds: float):
    probe = _Slow(seconds)
    api, targets, log = _build(
        tmp_path,
        probe,
        config=_call(target=_TARGET, inputs=["topic"], returns={"found": "findings"}),
    )
    cid, _ = api.start_conversation({"topic": "why"})
    return api, targets, log, probe, cid


@pytest.mark.skipif(
    current_timing_factor() > 1, reason="a wall-clock row: skipped under a tracer"
)
class TestTheStreamRunLoopsWait:
    """K7R.2, review warning 2. The probe of the reviewer: a 1 second pattern and ``max_steps=10``
    ended ``RunBudgetExceededError`` after 0.007 s in ``run_until_terminal_stream``."""

    def test_the_sync_stream_run_loop_does_not_burn_its_step_budget(
        self, tmp_path, monkeypatch
    ) -> None:
        """Kills: the stream step entry without the wait (the budget goes in milliseconds)."""
        monkeypatch.setattr(delegated, "DELEGATED_WAIT_SECONDS", 0.3)
        api, _targets, log, _probe, cid = _slow_flow(tmp_path, 1.0)
        try:
            started = time.monotonic()
            list(api.run_until_terminal_stream(cid, max_steps=10))
            assert time.monotonic() - started >= 0.7  # it really waited for the run
            assert api.get_data(cid)["call_result"]["status"] == "ok"
        finally:
            api.close()
            assert log is not None
            log.close()

    async def test_the_async_stream_run_loop_does_not_burn_its_step_budget(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(delegated, "DELEGATED_WAIT_SECONDS", 0.3)
        api, _targets, log, _probe, cid = _slow_flow(tmp_path, 1.0)
        try:
            async for _chunk in api.arun_until_terminal_stream(cid, max_steps=10):
                pass
            assert api.get_data(cid)["call_result"]["status"] == "ok"
        finally:
            api.close()
            assert log is not None
            log.close()

    async def test_the_non_stream_loop_is_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(delegated, "DELEGATED_WAIT_SECONDS", 0.3)
        api, _targets, log, _probe, cid = _slow_flow(tmp_path, 1.0)
        try:
            await api.arun_until_terminal(cid, max_steps=10)
            assert api.get_data(cid)["call_result"]["status"] == "ok"
        finally:
            api.close()
            assert log is not None
            log.close()


class TestTheStreamEntryWaitsOnce:
    def _spy(self, monkeypatch) -> list[str]:
        waits: list[str] = []
        real = RunHost.wait

        def spy(self, handle, timeout_seconds):
            waits.append(handle)
            return real(self, handle, timeout_seconds)

        monkeypatch.setattr(RunHost, "wait", spy)
        return waits

    async def test_one_advance_stream_makes_one_bounded_wait_outside_every_lock(
        self, tmp_path, monkeypatch
    ) -> None:
        probe = _Probe(hold=True)
        api, _targets, log = _build(
            tmp_path, probe, config=_call(target=_TARGET, inputs=["topic"])
        )
        seen: list[dict[str, Any]] = []
        real = RunHost.wait

        def spy(self, handle, timeout_seconds):
            seen.append(
                {
                    "timeout": timeout_seconds,
                    "stack": _lock_free(api._stack_lock),
                    "manager": _lock_free(api.fsm_manager._lock),
                }
            )
            return real(self, handle, timeout_seconds)

        monkeypatch.setattr(RunHost, "wait", spy)
        try:
            cid, _ = await api.astart_conversation({"topic": "why"})
            async for _chunk in api.aadvance_stream(cid):
                pass
            assert seen == [
                {
                    "timeout": delegated.DELEGATED_WAIT_SECONDS,
                    "stack": True,
                    "manager": True,
                }
            ]
        finally:
            probe.release.set()
            api.close()
            assert log is not None
            log.close()

    async def test_the_sync_advance_stream_waits_too(
        self, tmp_path, monkeypatch
    ) -> None:
        waits = self._spy(monkeypatch)
        probe = _Probe(hold=True)
        api, _targets, log = _build(
            tmp_path, probe, config=_call(target=_TARGET, inputs=["topic"])
        )
        try:
            cid, _ = await api.astart_conversation({"topic": "why"})
            await asyncio.to_thread(lambda: list(api.advance_stream(cid)))
            assert len(waits) == 1
        finally:
            probe.release.set()
            api.close()
            assert log is not None
            log.close()

    async def test_a_message_stream_never_waits(self, tmp_path, monkeypatch) -> None:
        """``aconverse`` and its stream have no run loop to pace."""
        probe = _Probe(hold=True)
        api, _targets, log = _build(
            tmp_path, probe, config=_call(target=_TARGET, inputs=["topic"])
        )
        try:
            cid, _ = await api.astart_conversation({"topic": "why"})
            await api.aadvance(cid)  # starts the run
            waits = self._spy(monkeypatch)
            async for _chunk in api.aconverse_stream("hello", cid):
                pass
            assert waits == []
        finally:
            probe.release.set()
            api.close()
            assert log is not None
            log.close()


class _FrameTargets(ExecutableTargets):
    """The registry's delegated resolution, plus one FRAME target ``child.fsm`` whose body is a
    delegated call state: a pushed child that holds a parked run."""

    def __init__(self, registry, context, child: dict[str, Any]) -> None:
        super().__init__(registry, context)
        self._child = child

    def resolve_target(self, target_id, *, conversation_id=None):
        if target_id == "child.fsm":
            return ResolvedTarget(FSMDefinition(**self._child), "t:child", "i:child")
        return super().resolve_target(target_id, conversation_id=conversation_id)


def _child_definition() -> dict[str, Any]:
    return _initial_call(
        _parent(
            _call(target=_TARGET, inputs=["topic"], returns={"found": "findings"}),
            response_instructions="",
        )
    )


def _types(log: EventLog, cid: str, *types: str) -> list[dict[str, Any]]:
    return [e for e in log.read(cid) if e["type"] in types]


class TestAPoppedFrameCancelsAndEndsItsRun:
    """K7R.2, review warning 3: ``pop_fsm`` ends the child through the manager, not through
    ``API.end_conversation``, so the run of the popped frame was never told to stop and its
    ``run/end`` was never written (the mark was dropped with the frame)."""

    def _flow(self, tmp_path: Path, *, call_steps: int | None):
        probe = _Probe(hold=True)
        registry = ExecutableRegistry()
        registry.register(probe)
        log = EventLog(tmp_path / "logs")
        targets = _FrameTargets(
            registry, RunContext(event_log=log), _child_definition()
        )
        config = _call(target="child.fsm", inputs=["topic"])
        if call_steps is not None:
            config["max_steps"] = call_steps
        api = API.from_definition(
            _initial_call(_parent(config, response_instructions="")),
            llm_interface=_Gate(default_response="r"),
            targets=targets,
            event_log=log,
        )
        cid, _ = api.start_conversation({"topic": "why"})
        return api, targets, log, probe, cid

    def test_a_child_whose_budget_is_spent_cancels_its_parked_run(
        self, tmp_path
    ) -> None:
        """The child holds 2 steps: the first starts its run, the second spends the budget and the
        engine pops it with the run still parked. Kills (D7): ``pop_fsm`` without the
        ``cancel_for`` hook (the token of the run stays unset, the run keeps going detached)."""
        api, targets, log, probe, cid = self._flow(tmp_path, call_steps=2)
        try:
            for _ in range(6):
                api.advance(cid)
                if probe.runs and len(api.conversation_stacks[cid]) == 1:
                    break
            assert len(probe.runs) == 1 and len(api.conversation_stacks[cid]) == 1
            assert api.get_data(cid)["call_result"]["reason"] == "budget_spent"
            assert probe.runs[0]["context"].cancel.cancelled is True
            assert (
                list(targets.host._running) != []
            )  # detached: the thread is still parked
        finally:
            probe.release.set()
            api.close()
            log.close()

    def test_the_popped_run_ends_in_the_log_at_the_parent_frame(self, tmp_path) -> None:
        """Kills (D8): ``pop_fsm`` without ``end_runs_of`` and the mark dropped with the frame
        (a ``run/start`` with no ``run/end`` after a CLEAN pop)."""
        api, _targets, log, probe, cid = self._flow(tmp_path, call_steps=2)
        try:
            for _ in range(6):
                api.advance(cid)
                if probe.runs and len(api.conversation_stacks[cid]) == 1:
                    break
            events = _types(log, cid, "stack/pop", "run/start", "run/end")
            assert [e["type"] for e in events] == ["run/start", "stack/pop", "run/end"]
            start, _pop, end = events
            assert end["handle"] == start["handle"] and end["target"] == _TARGET
            assert (end["status"], end["reason"]) == ("aborted", "cancelled")
            assert (end["template_identity"], end["instance_identity"]) == (
                start["template_identity"],
                start["instance_identity"],
            )
            assert end["frame"] == 0  # the top frame after the pop: the parent
            assert api._call_marks.runs == {}
        finally:
            probe.release.set()
            api.close()
            log.close()

    def test_a_child_popped_before_its_run_started_drops_its_reservation(
        self, tmp_path
    ) -> None:
        """One step: the child's budget is spent by the very turn that reserved its handle, so the
        pop comes before the start. Nothing started, nothing is owed, nothing stays reserved."""
        api, targets, log, probe, cid = self._flow(tmp_path, call_steps=1)
        try:
            for _ in range(6):
                api.advance(cid)
                if (
                    len(api.conversation_stacks[cid]) == 1
                    and api.get_stack_depth(cid) == 1
                ):
                    break
            assert probe.runs == []
            assert list(targets.host._reserved) == []
            assert _types(log, cid, "run/start", "run/end") == []
        finally:
            probe.release.set()
            api.close()
            log.close()

    def test_a_hand_pop_cancels_and_ends_the_run_too(self, tmp_path) -> None:
        probe = _Probe(hold=True)
        registry = ExecutableRegistry()
        registry.register(probe)
        log = EventLog(tmp_path / "logs")
        targets = ExecutableTargets(registry, RunContext(event_log=log))
        api = API.from_definition(
            _CHAT(),
            llm_interface=_Gate(default_response="r"),
            targets=targets,
            event_log=log,
        )
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            api.push_fsm(cid, _child_definition(), context_to_pass={"topic": "why"})
            api.advance(cid)  # the child decides, settle starts its run
            assert len(probe.runs) == 1
            api.pop_fsm(cid)
            assert probe.runs[0]["context"].cancel.cancelled is True
            end = _types(log, cid, "run/end")
            assert [(e["status"], e["reason"]) for e in end] == [
                ("aborted", "cancelled")
            ]
        finally:
            probe.release.set()
            api.close()
            log.close()

    def test_a_refused_pop_keeps_the_run(self, tmp_path, monkeypatch) -> None:
        """A pop whose end was refused leaves the frame on the stack, so its run is not told to
        stop and no ``run/end`` is written."""
        probe = _Probe(hold=True)
        registry = ExecutableRegistry()
        registry.register(probe)
        log = EventLog(tmp_path / "logs")
        targets = ExecutableTargets(registry, RunContext(event_log=log))
        api = API.from_definition(
            _CHAT(),
            llm_interface=_Gate(default_response="r"),
            targets=targets,
            event_log=log,
        )
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            api.push_fsm(cid, _child_definition(), context_to_pass={"topic": "why"})
            api.advance(cid)

            def refuse(frame_id):
                raise FSMError("a turn holds the child")

            monkeypatch.setattr(api.fsm_manager, "end_conversation", refuse)
            monkeypatch.setattr(api.fsm_manager, "has_instance", lambda frame_id: True)
            with pytest.raises(FSMError):
                api.pop_fsm(cid)
            assert probe.runs[0]["context"].cancel.cancelled is False
            assert _types(log, cid, "run/end") == []
            assert len(api._call_marks.runs) == 1
        finally:
            monkeypatch.undo()
            probe.release.set()
            api.close()
            log.close()

    def test_a_mark_whose_frame_is_off_the_stack_ends_at_the_next_flush(
        self, tmp_path
    ) -> None:
        """The other road to an orphan mark (a pop that raised after the teardown): the flush
        writes the owed ``run/end`` instead of dropping the mark silently. Kills: the orphan branch
        of ``flush_run_ends`` dropping without the event."""
        probe = _Probe()
        api, _targets, log = _build(
            tmp_path, probe, config=_call(target=_TARGET, inputs=["topic"])
        )
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            api._call_marks.runs["run-orphan"] = RunMark(
                cid, "gone-frame", "work", "call_result", _TARGET, "t:x", "i:x"
            )
            flush_run_ends(api, cid)
            [end] = _types(log, cid, "run/end")
            assert (end["handle"], end["status"], end["reason"]) == (
                "run-orphan",
                "aborted",
                "cancelled",
            )
            assert (end["template_identity"], end["instance_identity"]) == (
                "t:x",
                "i:x",
            )
            assert api._call_marks.runs == {}
        finally:
            api.close()
            assert log is not None
            log.close()
