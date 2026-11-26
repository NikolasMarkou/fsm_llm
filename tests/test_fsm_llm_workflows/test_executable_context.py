"""A composed run inherits its parent: ``ExecutableStep`` under ``_RUN_CONTEXT`` (plan 7d9c0955 step 4, D-007).

A workflow run by the adapter (step 5) puts its ``RunContext`` in one ``ContextVar``; an ``ExecutableStep`` derives
its child context from it, so a strategy, a pattern or a procedure that the workflow composes keeps the model, the
event log, the cancel token and the registry of its parent (a blocker row each), and a limit of the step narrows and
never widens (D-068 of plan ace7a8c8; O-4: a step limit above the inherited one is the step's failure route, no
clamp). With the variable unset the step is today's (the PARITY rows, and ``test_executable_step.py`` unedited).
Doubles only: a scripted model, a recording code-backed executable. No sleep decides a row; waits are bounded.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.constants import DEFAULT_MAX_STACK_DEPTH
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions.executable import Contract, Requires
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.results import RunResult
from fsm_llm.engine.context import CancelToken, RunContext
from fsm_llm.engine.executables import (
    CodeBackedExecutable,
    ExecutableRegistry,
)
from fsm_llm.engine.executables import adapters as adapters_module
from fsm_llm.testing import ScriptedLLM
from fsm_llm.workflows import (
    WorkflowEngine,
    WorkflowStatus,
    auto_step,
    create_workflow,
)
from fsm_llm.workflows.steps import ExecutableStep, ParallelStep
from fsm_llm.workflows.steps.helpers import _RUN_CONTEXT, _RUN_DEPTH, run_context_scope
from tests.test_fsm_llm_workflows.test_executable_step import _fsm_doc

_BOUND = 10  # seconds: a wait that would hang fails instead


class _Probe(CodeBackedExecutable):
    """A code-backed executable that records the ``RunContext`` of every run it gets."""

    def __init__(
        self, ident: str = "probe", *, callables: tuple[str, ...] = ()
    ) -> None:
        super().__init__(
            id=ident,
            kind="strategy",
            impl="tests",
            contract=Contract(inputs=["topic"], outputs=[]),
            requires=Requires(callables=list(callables)),
        )
        self.contexts: list[RunContext] = []

    async def execute(self, inputs, objects, context, identity) -> RunResult:
        self.contexts.append(context)
        return RunResult(status=Status.OK, outputs={}, answer="ok", identity=identity)


def _step(registry: Any, target: str = "probe", **kwargs: Any) -> ExecutableStep:
    kwargs.setdefault("inputs", {"topic": "subject"})
    kwargs.setdefault("success_step_id", "next")
    kwargs.setdefault("error_step_id", "err")
    return ExecutableStep(
        step_id="x", name="X", target=target, registry=registry, **kwargs
    )


def _registry(*executables: Any) -> ExecutableRegistry:
    registry = ExecutableRegistry()
    for executable in executables:
        registry.register(executable)
    return registry


async def _run(step: ExecutableStep, parent: RunContext | None):
    """Run ``step`` the way the adapter's engine does: inside the scope of ``parent`` (or none)."""
    if parent is None:
        return await asyncio.wait_for(step.execute({"subject": "t"}), _BOUND)
    with run_context_scope(parent):
        return await asyncio.wait_for(step.execute({"subject": "t"}), _BOUND)


class _Resolver:
    """A ``TargetResolver`` stand-in: the context only checks that ``resolve_target`` exists."""

    def resolve_target(self, target_id: str, *, conversation_id: str | None = None):
        raise AssertionError("never resolved in these rows")


# ---- unset: today's step (PARITY, rows that must pass on both trees) ----------------------------


class TestUnset:
    async def test_with_no_parent_the_step_builds_the_bare_context(self) -> None:
        probe = _Probe()
        step = _step(_registry(probe), max_steps=7, max_seconds=3.5)
        assert (await _run(step, None)).success is True
        (context,) = probe.contexts
        assert context.parent is None
        assert (context.max_steps, context.max_seconds) == (7, 3.5)
        assert context.llm_interface is None and context.llm_factory is None
        assert context.event_log is None and context.registry is None
        assert context.callables is None and context.cancel.cancelled is False

    async def test_with_no_parent_a_needed_callables_registry_is_still_refused(
        self,
    ) -> None:
        probe = _Probe(callables=("tool",))
        result = await _run(_step(_registry(probe)), None)
        assert (result.success, result.next_step_id) == (False, "err")
        assert "needs the callables" in str(result.error)
        assert probe.contexts == []

    def test_the_variable_is_unset_by_default(self) -> None:
        assert _RUN_CONTEXT.get() is None


# ---- set: the blocker rows (a composed run keeps model, log, token, registry) -------------------


class TestInherited:
    async def test_the_child_shares_every_host_member_and_names_its_parent(
        self,
    ) -> None:
        llm = ScriptedLLM()
        names: NameRegistry[Any] = NameRegistry("callables")
        resolver = _Resolver()
        token = CancelToken()
        parent = RunContext(
            llm_interface=llm,
            cancel=token,
            callables=names,
            registry=resolver,  # type: ignore[arg-type]
        )
        probe = _Probe()
        assert (await _run(_step(_registry(probe)), parent)).success is True
        (child,) = probe.contexts
        assert child.llm_interface is llm
        assert child.cancel is token
        assert child.callables is names
        assert child.registry is resolver
        assert child.parent == "x"  # the id of the step that asks
        assert child is not parent

    async def test_no_approval_grant_is_carried_into_the_child(self) -> None:
        """D1: the context has no member that could hold a grant and the child gains none (the
        approval grant is driver-only and an executable never supplies one)."""
        import dataclasses

        probe = _Probe()
        await _run(_step(_registry(probe)), RunContext())
        (child,) = probe.contexts
        members = {f.name for f in dataclasses.fields(RunContext)}
        assert set(vars(child)) == members
        assert not any("grant" in m or "approv" in m for m in members)

    async def test_the_step_callables_win_for_names_when_set(self) -> None:
        mine: NameRegistry[Any] = NameRegistry("callables")
        theirs: NameRegistry[Any] = NameRegistry("callables")
        probe = _Probe()
        parent = RunContext(callables=theirs)
        await _run(_step(_registry(probe), callables=mine), parent)
        assert probe.contexts[0].callables is mine
        assert probe.contexts[0].cancel is parent.cancel

    async def test_a_needed_registry_is_met_by_the_inherited_callables(self) -> None:
        probe = _Probe(callables=("tool",))
        names: NameRegistry[Any] = NameRegistry("callables")
        names.register("tool", lambda: None)
        result = await _run(_step(_registry(probe)), RunContext(callables=names))
        assert result.success is True and probe.contexts[0].callables is names

    async def test_the_model_of_the_parent_answers_the_child_fsm(self) -> None:
        """BLOCKER: the child FSM is asked through the parent's interface (not a second model)."""
        llm = ScriptedLLM(extraction_data={"result": "42"}, default_response="r")
        registry = ExecutableRegistry()
        registry.add_document(_fsm_doc())
        step = _step(
            registry, "demo", returns={"final": "result"}, success_step_id="next"
        )
        result = await _run(step, RunContext(llm_interface=llm))
        assert result.success is True and result.data["final"] == "42"
        assert llm.call_history, "the parent's interface was never asked"
        assert llm.unexpected == []

    async def test_with_no_parent_model_the_same_step_fails_for_want_of_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The contrast of the row above: the model comes from the inherited context, not from
        anywhere else (no silent default, D11)."""
        monkeypatch.delenv("LLM_MODEL", raising=False)
        registry = ExecutableRegistry()
        registry.add_document(_fsm_doc())
        result = await _run(_step(registry, "demo"), None)
        assert (result.success, result.next_step_id) == (False, "err")

    async def test_the_event_log_of_the_parent_receives_the_child_turns(
        self, tmp_path: Path
    ) -> None:
        """BLOCKER: the child writes to the INHERITED log (invariant 6)."""
        llm = ScriptedLLM(extraction_data={"result": "42"}, default_response="r")
        registry = ExecutableRegistry()
        registry.add_document(_fsm_doc())
        parent = RunContext(llm_interface=llm, event_log=tmp_path)
        assert (await _run(_step(registry, "demo"), parent)).success is True
        (log,) = sorted((tmp_path / "logs").glob("*.jsonl"))
        text = log.read_text(encoding="utf-8")
        assert "turn/start" in text and "llm/request" in text

    async def test_the_cancel_token_of_the_parent_ends_the_child_cancelled(
        self,
    ) -> None:
        """BLOCKER: setting the PARENT's token ends the composed run ``aborted/cancelled`` (the
        child shares the token, it did not get a fresh one) and the step takes its failure route.
        The next call of the same caller, with a token that is not set, runs normally."""
        llm = ScriptedLLM(extraction_data={"result": "42"}, default_response="r")
        registry = ExecutableRegistry()
        registry.add_document(_fsm_doc())
        token = CancelToken()
        token.cancel()
        step = _step(registry, "demo")
        result = await _run(step, RunContext(llm_interface=llm, cancel=token))
        assert (result.success, result.next_step_id) == (False, "err")
        assert result.data["executable_status"] == "aborted"
        assert "cancelled" in str(result.error)
        assert llm.call_history == []  # nothing was asked
        again = await _run(step, RunContext(llm_interface=llm))
        assert again.success is True

    async def test_the_registry_of_the_parent_is_the_resolver_of_the_child_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BLOCKER: a ``call`` state of the composed FSM resolves through the inherited registry:
        the child ``API`` is built with ``targets=`` the parent's resolver."""
        seen: list[Any] = []

        class _Spy(API):
            @classmethod
            def from_definition(cls, definition, **kwargs):  # type: ignore[override]
                seen.append(kwargs.get("targets"))
                return super().from_definition(definition, **kwargs)

        monkeypatch.setattr(adapters_module, "API", _Spy)
        resolver = _Resolver()
        llm = ScriptedLLM(extraction_data={"result": "42"}, default_response="r")
        registry = ExecutableRegistry()
        registry.add_document(_fsm_doc())
        parent = RunContext(llm_interface=llm, registry=resolver)  # type: ignore[arg-type]
        assert (await _run(_step(registry, "demo"), parent)).success is True
        assert seen == [resolver]


# ---- limits narrow and never widen (D-068, O-4) -------------------------------------------------


class TestLimits:
    async def test_an_unset_step_limit_keeps_the_parents(self) -> None:
        probe = _Probe()
        parent = RunContext(max_steps=5, max_seconds=4.0)
        await _run(_step(_registry(probe)), parent)
        assert (probe.contexts[0].max_steps, probe.contexts[0].max_seconds) == (5, 4.0)

    async def test_a_lower_step_limit_narrows(self) -> None:
        probe = _Probe()
        parent = RunContext(max_steps=5, max_seconds=4.0)
        await _run(_step(_registry(probe), max_steps=2, max_seconds=1.5), parent)
        assert (probe.contexts[0].max_steps, probe.contexts[0].max_seconds) == (2, 1.5)

    async def test_an_equal_limit_is_not_widening(self) -> None:
        probe = _Probe()
        await _run(_step(_registry(probe), max_steps=5), RunContext(max_steps=5))
        assert probe.contexts[0].max_steps == 5

    async def test_a_parent_with_no_limit_accepts_the_step_limit(self) -> None:
        probe = _Probe()
        await _run(_step(_registry(probe), max_steps=9), RunContext())
        assert probe.contexts[0].max_steps == 9

    @pytest.mark.parametrize(
        ("parent_kwargs", "step_kwargs", "named"),
        [
            ({"max_seconds": 5.0}, {"max_seconds": 10.0}, "max_seconds"),
            ({"max_steps": 3}, {"max_steps": 4}, "max_steps"),
        ],
    )
    async def test_a_step_limit_above_the_parents_is_the_failure_route(
        self, parent_kwargs: dict, step_kwargs: dict, named: str
    ) -> None:
        """O-4: ``derive`` raises, the step reports it on its failure route, nothing is run and
        the limit is not clamped to the parent's."""
        probe = _Probe()
        result = await _run(
            _step(_registry(probe), **step_kwargs), RunContext(**parent_kwargs)
        )
        assert (result.success, result.next_step_id) == (False, "err")
        assert "cannot widen" in str(result.error) and named in str(result.error)
        assert probe.contexts == []


# ---- the variable never outlives a run ----------------------------------------------------------


class TestScope:
    def test_the_scope_sets_and_resets(self) -> None:
        parent = RunContext()
        with run_context_scope(parent):
            assert _RUN_CONTEXT.get() is parent
        assert _RUN_CONTEXT.get() is None

    def test_the_scope_nests_and_restores_the_outer_value(self) -> None:
        outer, inner = RunContext(), RunContext()
        with run_context_scope(outer):
            with run_context_scope(inner):
                assert _RUN_CONTEXT.get() is inner
            assert _RUN_CONTEXT.get() is outer
        assert _RUN_CONTEXT.get() is None

    def test_the_scope_resets_after_an_exception(self) -> None:
        with pytest.raises(RuntimeError), run_context_scope(RunContext()):
            raise RuntimeError("boom")
        assert _RUN_CONTEXT.get() is None

    async def test_the_scope_resets_after_a_cancel(self) -> None:
        entered = asyncio.Event()

        async def parked() -> None:
            with run_context_scope(RunContext()):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(parked())
        await asyncio.wait_for(entered.wait(), _BOUND)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _BOUND)
        assert (
            _RUN_CONTEXT.get() is None
        )  # a task has its own copy; the caller never saw it

    async def test_the_variable_does_not_leak_into_an_unrelated_task(self) -> None:
        """A task created BEFORE the scope never sees it; one created inside does (a child task
        inherits), and neither sees it after the scope ends."""
        before = asyncio.create_task(asyncio.sleep(0, result="x"))

        async def read() -> Any:
            return _RUN_CONTEXT.get()

        parent = RunContext()
        with run_context_scope(parent):
            inside = await asyncio.wait_for(asyncio.create_task(read()), _BOUND)
        after = await asyncio.wait_for(asyncio.create_task(read()), _BOUND)
        await before
        assert inside is parent and after is None

    async def test_a_second_engine_started_afterwards_sees_no_parent(self) -> None:
        """A second run in the same task after the first run's scope ended builds the bare
        context again (no stale model or log)."""
        probe = _Probe()
        step = _step(_registry(probe))
        await _run(step, RunContext(llm_interface=ScriptedLLM()))
        await _run(step, None)
        first, second = probe.contexts
        assert first.llm_interface is not None and first.parent == "x"
        assert second.llm_interface is None and second.parent is None


# ---- through a real engine ----------------------------------------------------------------------


def _workflow(step: Any):
    return (
        create_workflow("wf", "W")
        .with_initial_step(step)
        .with_step(auto_step("next", "Next", next_step_id=""))
        .with_step(auto_step("err", "Err", next_step_id=""))
    )


class TestDepth:
    """Review of plan 7d9c0955 (D-012, finding 3): one level per adapter scope, whatever the context."""

    def test_each_scope_is_one_level_and_the_depth_comes_back(self) -> None:
        assert _RUN_DEPTH.get() == 0
        with run_context_scope(RunContext()):
            assert _RUN_DEPTH.get() == 1
            with run_context_scope(RunContext()):
                assert _RUN_DEPTH.get() == 2
            assert _RUN_DEPTH.get() == 1
        assert _RUN_DEPTH.get() == 0

    def test_a_fresh_or_derived_context_does_not_reset_the_depth(self) -> None:
        parent = RunContext(max_steps=5)
        with run_context_scope(parent):
            with run_context_scope(parent.derive(parent="s", max_steps=3)):
                with run_context_scope(RunContext()):
                    assert _RUN_DEPTH.get() == 3

    def test_the_depth_is_reset_after_an_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with run_context_scope(RunContext()):
                raise RuntimeError("x")
        assert _RUN_DEPTH.get() == 0

    @pytest.mark.parametrize(("levels", "refused"), [(9, False), (10, True)])
    async def test_a_step_below_the_limit_runs_and_at_the_limit_is_refused(
        self, levels: int, refused: bool
    ) -> None:
        asked: list[str] = []

        class _Registry:
            def get(self, target: str) -> Any:
                asked.append(target)
                raise KeyError(target)

        step = ExecutableStep(
            step_id="e", name="e", target="t", registry=_Registry(), error_step_id="f"
        )
        asked.clear()  # building the step asks the registry about the target once
        with contextlib.ExitStack() as stack:
            for _ in range(levels):
                stack.enter_context(run_context_scope(RunContext()))
            result = await step.execute({})

        assert result.success is False and result.next_step_id == "f"
        assert (asked == []) is refused
        if refused:
            assert result.error == (
                f"composed workflow runs nest at most {DEFAULT_MAX_STACK_DEPTH} deep"
            )


class TestThroughTheEngine:
    async def test_a_step_run_by_an_engine_under_the_scope_inherits(self) -> None:
        probe = _Probe()
        parent = RunContext(llm_interface=ScriptedLLM(), max_steps=6)
        engine = WorkflowEngine()
        engine.register_workflow(_workflow(_step(_registry(probe))))
        with run_context_scope(parent):
            iid = await engine.start("wf", initial_context={"subject": "t"})
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        assert probe.contexts[0].llm_interface is parent.llm_interface
        assert probe.contexts[0].max_steps == 6
        await engine.close()

    async def test_a_second_engine_after_the_scope_sees_none(self) -> None:
        probe = _Probe()
        engine = WorkflowEngine()
        engine.register_workflow(_workflow(_step(_registry(probe))))
        with run_context_scope(RunContext(llm_interface=ScriptedLLM())):
            await engine.start("wf", initial_context={"subject": "t"})
        await engine.start("wf", initial_context={"subject": "t"})
        assert probe.contexts[0].llm_interface is not None
        assert probe.contexts[1].llm_interface is None
        await engine.close()

    async def test_both_children_of_a_parallel_step_inherit(self) -> None:
        a, b = _Probe("a"), _Probe("b")
        registry = _registry(a, b)
        children = [
            ExecutableStep(
                step_id=f"x{i}",
                name="X",
                target=target,
                registry=registry,
                inputs={"topic": "subject"},
            )
            for i, target in enumerate(("a", "b"))
        ]
        parallel = ParallelStep(
            step_id="p", name="P", steps=children, next_step_id="next"
        )
        token = CancelToken()
        engine = WorkflowEngine()
        engine.register_workflow(
            create_workflow("wf", "W")
            .with_initial_step(parallel)
            .with_step(auto_step("next", "Next", next_step_id=""))
        )
        with run_context_scope(RunContext(cancel=token, max_seconds=30.0)):
            iid = await engine.start("wf", initial_context={"subject": "t"})
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        for probe, child_id in ((a, "x0"), (b, "x1")):
            (context,) = probe.contexts
            assert context.cancel is token and context.max_seconds == 30.0
            assert context.parent == child_id
        await engine.close()
