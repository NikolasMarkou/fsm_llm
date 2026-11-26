"""The adapter of the ``workflow`` kind (plan 7d9c0955 step 5, D-008; invariants 1, 3, 4, 6, 9).

``fsm_llm.workflows.executable`` serves a ``workflow`` envelope by id: one ``WorkflowEngine`` per run,
the sync ``run`` is ``asyncio.run`` (and refuses inside a running loop), the cancel goes through the
engine, a budget is ``aborted/budget_spent``, outputs are the declared contract only and a composed
run inherits its parent (model, event log, token, registry). Every row that runs a workflow has a
sync and an async twin (``how``), the callables are plain functions, the model is ``ScriptedLLM``:
no provider, no network, no sleep decides a row (a wait that never ends is bounded by a budget or by
``_BOUND``). Rows that run a thread or a loop have the hard per-test timeout of the suite.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions.executable import (
    ExecutableDocumentError,
    ExecutableError,
    InstantiationError,
    WorkflowEnvelope,
    parse_executable,
)
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.context import CancelToken, RunContext
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables import registry as registry_module
from fsm_llm.engine.executables.directory import load_directory_registry
from fsm_llm.engine.executables.protocol import DEFAULT_EXECUTABLE_MAX_STEPS
from fsm_llm.testing import ScriptedLLM
from fsm_llm.workflows import WorkflowEngine, WorkflowStatus
from fsm_llm.workflows import executable as wfexec
from fsm_llm.workflows.exceptions import WorkflowValidationError
from fsm_llm.workflows.executable import (
    WorkflowExecutable,
    workflow_adapter,
)
from tests.test_fsm_llm_workflows.test_executable_step import _fsm_doc

_BOUND = 20  # seconds: a wait that would hang fails instead

HOW = pytest.mark.parametrize("how", ["sync", "async"])


def _auto(step_id: str, nxt: str = "", action: str | None = None, **extra: Any) -> Any:
    step: dict[str, Any] = {
        "type": "AutoStep",
        "step_id": step_id,
        "name": step_id,
        "next_step_id": nxt,
    }
    if action is not None:
        step["action"] = action
    return {**step, **extra}


def _chain(count: int, action: str | None = None) -> dict[str, Any]:
    """s1 -> s2 -> ... -> s<count> (acyclic: a loop is refused when a workflow registers)."""
    return {
        f"s{i}": _auto(f"s{i}", f"s{i + 1}" if i < count else "", action)
        for i in range(1, count + 1)
    }


def _document(
    steps: dict[str, Any],
    *,
    ident: str = "wf",
    initial: str | None = None,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    callables: tuple[str, ...] = (),
    executables: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "schema": "fsm-llm/executable",
        "kind": "workflow",
        "id": ident,
        "name": ident,
        "contract": {"inputs": list(inputs), "outputs": list(outputs)},
        "requires": {
            "tools": "none",
            "callables": list(callables),
            "executables": list(executables),
        },
        "lifecycle": "draft",
        "body": {"initial_step_id": initial or next(iter(steps)), "steps": steps},
    }


def _executable(document: dict[str, Any], **options: Any) -> WorkflowExecutable:
    envelope = parse_executable(document)
    assert isinstance(envelope, WorkflowEnvelope)
    return workflow_adapter(envelope, **options)


def _names(**objects: Callable[..., Any]) -> NameRegistry[Any]:
    registry: NameRegistry[Any] = NameRegistry("callables")
    for name, obj in objects.items():
        registry.register(name, obj)
    return registry


async def _go(
    how: str,
    executable: Any,
    inputs: dict[str, Any] | None = None,
    context: RunContext | None = None,
) -> Any:
    """Run through the sync name (on a worker thread: ``asyncio.run`` needs a thread of its own) or
    the async one."""
    if how == "async":
        work = executable.arun(inputs or {}, context=context)
    else:
        work = asyncio.to_thread(executable.run, inputs or {}, context=context)
    return await asyncio.wait_for(work, _BOUND)


class _SpyEngine(WorkflowEngine):
    """A real engine that records that it was built and closed."""

    made: ClassVar[list[_SpyEngine]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.closed = 0
        _SpyEngine.made.append(self)

    async def close(self) -> None:
        self.closed += 1
        await super().close()


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> type[_SpyEngine]:
    _SpyEngine.made = []
    monkeypatch.setattr(wfexec, "WorkflowEngine", _SpyEngine)
    return _SpyEngine


# ---- the kind, the header, both identities ------------------------------------------------------


class TestKind:
    def test_a_registry_loads_a_workflow_document_and_serves_it_by_id(self) -> None:
        registry = ExecutableRegistry()
        served = registry.add_document(_document(_chain(2)))
        assert isinstance(served, WorkflowExecutable)
        assert registry.get("wf") is served
        assert (served.id, served.kind, served.version) == ("wf", "workflow", "1")
        assert served.lifecycle.value == "draft"
        assert served.template_identity and len(served.template_identity) == 64

    def test_the_instance_identity_is_the_digest_of_the_body_and_a_workflow_takes_no_option(
        self,
    ) -> None:
        a = _executable(_document(_chain(2), ident="one")).instantiate()
        b = _executable(_document(_chain(2), ident="two")).instantiate()
        c = _executable(_document(_chain(3), ident="one")).instantiate()
        assert a.instance_identity == b.instance_identity != c.instance_identity
        assert dict(a.record) == {"options": {}, "variables": {}}
        with pytest.raises(InstantiationError):
            _executable(_document(_chain(2))).instantiate(options={"x": 1})
        with pytest.raises(InstantiationError):
            _executable(_document(_chain(2))).instantiate(variables={"x": "1"})

    @HOW
    async def test_a_result_carries_both_identities(self, how: str) -> None:
        executable = _executable(_document(_chain(2)))
        result = await _go(how, executable)
        assert result.status is Status.OK
        assert result.identity.template_identity == executable.template_identity
        assert (
            result.identity.instance_identity
            == executable.instantiate().instance_identity
        )
        assert result.handle  # the engine instance id


# ---- the result mapping -----------------------------------------------------------------------


class TestOutcomes:
    @HOW
    async def test_a_completed_run_returns_the_declared_outputs_only(
        self, how: str
    ) -> None:
        steps = {"a": _auto("a", "b", "make"), "b": _auto("b")}
        executable = _executable(
            _document(steps, outputs=("made",), callables=("make",))
        )
        context = RunContext(
            callables=_names(make=lambda ctx: {"made": 1, "stray": 2, "api_key": "k"})
        )
        result = await _go(how, executable, context=context)
        assert result.status is Status.OK and result.reason is None
        assert result.outputs == {"made": 1}
        assert result.detail["kind"] == "workflow" and result.detail["steps"] == 2

    @HOW
    async def test_no_declared_output_gives_no_outputs(self, how: str) -> None:
        executable = _executable(
            _document({"a": _auto("a", action="make")}, callables=("make",))
        )
        context = RunContext(callables=_names(make=lambda ctx: {"made": 1}))
        result = await _go(how, executable, context=context)
        assert (result.status, result.outputs) == (Status.OK, {})

    @HOW
    async def test_a_declared_output_that_never_appears_is_return_keys_missing(
        self, how: str
    ) -> None:
        executable = _executable(_document(_chain(1), outputs=("nope",)))
        result = await _go(how, executable)
        assert (result.status, result.reason) == (
            Status.FAILED,
            Reason.RETURN_KEYS_MISSING,
        )
        assert result.detail["missing"] == ["nope"]

    @HOW
    async def test_inputs_reach_the_context_and_a_missing_one_is_refused_before_a_step(
        self, how: str, spy: type[_SpyEngine]
    ) -> None:
        seen: list[Any] = []
        executable = _executable(
            _document(
                {"a": _auto("a", action="look")},
                inputs=("amount",),
                callables=("look",),
            )
        )
        context = RunContext(callables=_names(look=lambda ctx: seen.append(dict(ctx))))
        missing = await _go(how, executable, {}, context)
        assert (missing.status, missing.reason) == (
            Status.FAILED,
            Reason.INPUTS_MISSING,
        )
        assert missing.detail["missing"] == ["amount"]
        assert spy.made == [] and seen == []
        ok = await _go(how, executable, {"amount": 7}, context)
        assert ok.status is Status.OK and seen[0]["amount"] == 7

    @HOW
    @pytest.mark.parametrize(
        "key", ["bogus", "_workflow_info", "_user_input", "_approval_granted"]
    )
    async def test_an_undeclared_or_internal_prefixed_input_is_refused_before_a_step(
        self, how: str, key: str, spy: type[_SpyEngine]
    ) -> None:
        executable = _executable(_document(_chain(1), inputs=("amount",)))
        with pytest.raises(ValueError, match="declares no input"):
            await _go(how, executable, {"amount": 1, key: "x"})
        assert spy.made == []

    @HOW
    async def test_a_name_the_context_does_not_supply_is_an_executable_error(
        self, how: str, spy: type[_SpyEngine]
    ) -> None:
        executable = _executable(
            _document({"a": _auto("a", action="make")}, callables=("make",))
        )
        for context in (None, RunContext(callables=_names(other=lambda c: None))):
            with pytest.raises(ExecutableError, match="needs the callables"):
                await _go(how, executable, context=context)
        assert spy.made == []

    @HOW
    async def test_a_raising_callable_is_failed_with_no_reason_and_a_capped_error(
        self, how: str
    ) -> None:
        def boom(ctx: Any) -> None:
            raise RuntimeError("x" * 5000)

        steps = {
            "a": {
                "type": "ConditionStep",
                "step_id": "a",
                "name": "a",
                "condition": "boom",
                "true_step_id": "b",
                "false_step_id": "b",
            },
            "b": _auto("b"),
        }
        executable = _executable(_document(steps, callables=("boom",)))
        result = await _go(
            how, executable, context=RunContext(callables=_names(boom=boom))
        )
        assert (result.status, result.reason) == (Status.FAILED, None)
        assert result.detail["workflow_status"] == "failed"
        assert 0 < len(result.detail["error"]) <= 300


# ---- budgets (invariant 4: a budget narrows, never a failure) ----------------------------------


class TestBudgets:
    @HOW
    async def test_a_chain_longer_than_the_step_cap_is_budget_spent(
        self, how: str
    ) -> None:
        executable = _executable(_document(_chain(6)))
        result = await _go(how, executable, context=RunContext(max_steps=5))
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert (result.detail["budget"], result.detail["limit"]) == ("max_steps", 5)
        assert result.detail["steps"] == 5

    @HOW
    async def test_a_chain_that_fits_the_cap_completes(self, how: str) -> None:
        executable = _executable(_document(_chain(5)))
        result = await _go(how, executable, context=RunContext(max_steps=5))
        assert result.status is Status.OK

    @HOW
    async def test_a_failure_one_below_the_cap_is_a_failure_and_not_a_budget(
        self, how: str
    ) -> None:
        def boom(ctx: Any) -> None:
            raise RuntimeError("no")

        steps = _chain(6)
        steps["s5"] = _auto("s5", "s6", "boom")
        executable = _executable(_document(steps, callables=("boom",)))
        context = RunContext(max_steps=5, callables=_names(boom=boom))
        result = await _go(how, executable, context=context)
        assert (result.status, result.reason) == (Status.FAILED, None)

    @HOW
    async def test_failures_that_take_their_error_route_still_count_toward_the_cap(
        self, how: str
    ) -> None:
        """Review of plan 7d9c0955 (D-012, finding 2): the engine counts every step it runs."""

        def boom(ctx: Any) -> None:
            raise RuntimeError("no")

        steps = _chain(6)
        steps["s1"] = _auto("s1", "s2", "boom", error_step_id="s2")
        executable = _executable(_document(steps, callables=("boom",)))
        context = RunContext(max_steps=5, callables=_names(boom=boom))
        result = await _go(how, executable, context=context)
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert (result.detail["budget"], result.detail["limit"]) == ("max_steps", 5)
        assert result.detail["steps"] == 5

    @HOW
    async def test_a_run_that_routes_around_a_failure_and_fits_counts_every_step(
        self, how: str
    ) -> None:
        def boom(ctx: Any) -> None:
            raise RuntimeError("no")

        steps = _chain(3)
        steps["s1"] = _auto("s1", "s2", "boom", error_step_id="s2")
        executable = _executable(_document(steps, callables=("boom",)))
        context = RunContext(max_steps=3, callables=_names(boom=boom))
        result = await _go(how, executable, context=context)
        assert result.status is Status.OK and result.detail["steps"] == 3

    @HOW
    async def test_a_failure_on_the_last_allowed_step_with_no_route_is_a_failure(
        self, how: str
    ) -> None:
        def boom(ctx: Any) -> None:
            raise RuntimeError("no")

        steps = _chain(5)
        steps["s5"] = _auto("s5", "", "boom")
        executable = _executable(_document(steps, callables=("boom",)))
        context = RunContext(max_steps=5, callables=_names(boom=boom))
        result = await _go(how, executable, context=context)
        assert (result.status, result.reason) == (Status.FAILED, None)
        assert result.detail["steps"] == 5

    @HOW
    async def test_a_failure_routed_on_from_the_last_allowed_step_is_budget_spent(
        self, how: str
    ) -> None:
        def boom(ctx: Any) -> None:
            raise RuntimeError("no")

        steps = _chain(6)
        steps["s5"] = _auto("s5", "s6", "boom", error_step_id="s6")
        executable = _executable(_document(steps, callables=("boom",)))
        context = RunContext(max_steps=5, callables=_names(boom=boom))
        result = await _go(how, executable, context=context)
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)

    @HOW
    async def test_the_default_cap_bounds_a_run_with_no_limit(self, how: str) -> None:
        executable = _executable(_document(_chain(DEFAULT_EXECUTABLE_MAX_STEPS + 1)))
        result = await _go(how, executable)
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert result.detail["limit"] == DEFAULT_EXECUTABLE_MAX_STEPS

    @HOW
    async def test_a_step_that_never_ends_is_stopped_by_the_deadline(
        self, how: str
    ) -> None:
        async def hang(ctx: Any) -> None:
            await asyncio.Event().wait()

        executable = _executable(
            _document(
                {"a": _auto("a", "b", "hang"), "b": _auto("b")}, callables=("hang",)
            )
        )
        context = RunContext(max_seconds=0.2, callables=_names(hang=hang))
        result = await _go(how, executable, context=context)
        assert (result.status, result.reason) == (Status.ABORTED, Reason.BUDGET_SPENT)
        assert (
            result.detail["budget"] == "max_seconds" and result.detail["limit"] == 0.2
        )


# ---- cancel (invariant 4: through the engine, never swallowed) ---------------------------------


class TestCancel:
    @HOW
    async def test_a_token_set_before_the_first_step_starts_nothing(
        self, how: str, spy: type[_SpyEngine]
    ) -> None:
        calls: list[str] = []
        executable = _executable(
            _document({"a": _auto("a", action="mark")}, callables=("mark",))
        )
        token = CancelToken()
        token.cancel()
        context = RunContext(
            cancel=token, callables=_names(mark=lambda c: calls.append("a"))
        )
        result = await _go(how, executable, context=context)
        assert (result.status, result.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert calls == [] and spy.made == []
        # the NEXT call of the same caller: a fresh token runs the same executable
        again = await _go(
            how, executable, context=RunContext(callables=context.callables)
        )
        assert again.status is Status.OK and calls == ["a"]

    @HOW
    async def test_a_token_set_between_steps_stops_the_later_steps(
        self, how: str
    ) -> None:
        calls: list[str] = []
        token = CancelToken()

        def make(name: str) -> Callable[[Any], None]:
            def act(ctx: Any) -> None:
                calls.append(name)
                if name == "s1":
                    token.cancel()

            return act

        steps = {
            f"s{i}": _auto(f"s{i}", f"s{i + 1}" if i < 5 else "", f"f{i}")
            for i in range(1, 6)
        }
        executable = _executable(
            _document(steps, callables=tuple(f"f{i}" for i in range(1, 6)))
        )
        names = _names(**{f"f{i}": make(f"s{i}") for i in range(1, 6)})
        result = await _go(
            how, executable, context=RunContext(cancel=token, callables=names)
        )
        assert (result.status, result.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert (
            calls[:2] == ["s1", "s2"] and "s4" not in calls
        )  # the step in flight finishes
        # the NEXT call of the same caller: the token stays set, so nothing starts; a new one runs
        calls.clear()
        again = await _go(
            how, executable, context=RunContext(cancel=token, callables=names)
        )
        assert (again.status, again.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert calls == []
        fresh = await _go(how, executable, context=RunContext(callables=names))
        assert fresh.status is Status.OK and len(calls) == 5

    @HOW
    async def test_a_token_set_during_an_executable_step_is_cancelled_not_failed(
        self, how: str
    ) -> None:
        registry = ExecutableRegistry()
        registry.register(_CancellingChild())
        step = {
            "type": "ExecutableStep",
            "step_id": "call",
            "name": "call",
            "target": "child",
            "success_step_id": "",
        }
        executable = _executable(_document({"call": step}, executables=("child",)))
        token = CancelToken()
        context = RunContext(cancel=token, registry=registry)
        result = await _go(how, executable, context=context)
        assert token.cancelled
        assert (result.status, result.reason) == (Status.ABORTED, Reason.CANCELLED)

    async def test_a_cancelled_task_is_never_swallowed_and_the_engine_is_closed(
        self, spy: type[_SpyEngine]
    ) -> None:
        started = asyncio.Event()
        calls: list[int] = []

        async def maybe_hang(ctx: Any) -> None:
            calls.append(1)
            if len(calls) == 1:
                started.set()
                await asyncio.Event().wait()

        executable = _executable(
            _document({"a": _auto("a", action="hang")}, callables=("hang",))
        )
        context = RunContext(callables=_names(hang=maybe_hang))
        task = asyncio.ensure_future(executable.arun({}, context=context))
        await asyncio.wait_for(started.wait(), _BOUND)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _BOUND)
        assert [engine.closed for engine in spy.made] == [1]
        # the NEXT call of the same caller runs to the end on a fresh engine
        again = await asyncio.wait_for(executable.arun({}, context=context), _BOUND)
        assert again.status is Status.OK
        assert [engine.closed for engine in spy.made] == [1, 1]

    async def test_a_failing_cancel_request_is_logged_and_the_outcome_is_the_engines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(self: Any, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("cancel is broken")

        monkeypatch.setattr(WorkflowEngine, "cancel", boom)
        token = CancelToken()
        steps = {"a": _auto("a", "b", "stop"), "b": _auto("b", "c"), "c": _auto("c")}
        executable = _executable(_document(steps, callables=("stop",)))
        names = _names(stop=lambda ctx: token.cancel())
        result = await executable.arun(
            {}, context=RunContext(cancel=token, callables=names)
        )
        # the token is set and the instance completed anyway: a set token wins only when the
        # instance did not complete (the commit wins)
        assert result.status is Status.OK


class _CancellingChild:
    """A code-backed-shaped executable that sets the token of the run that asked and says so."""

    id = "child"
    kind = "strategy"
    version = "1"
    template_identity = "t" * 64
    capabilities: ClassVar[list[Any]] = []

    def __init__(self) -> None:
        from fsm_llm.core.codes import Lifecycle
        from fsm_llm.core.definitions.executable import Contract, Requires

        self.contract = Contract()
        self.requires = Requires()
        self.lifecycle = Lifecycle.BUILTIN

    def instantiate(self, options: Any = None, variables: Any = None) -> Any:
        raise NotImplementedError

    def run(self, inputs: Any, *, context: Any = None) -> Any:
        raise NotImplementedError

    async def arun(self, inputs: Any, *, context: RunContext | None = None) -> Any:
        from fsm_llm.core.results import Identity, RunResult

        assert context is not None
        context.cancel.cancel()
        return RunResult(
            status=Status.ABORTED, reason=Reason.CANCELLED, identity=Identity()
        )


# ---- the engine is always closed; a run that does not end is a failure -------------------------


class TestEngineLifecycle:
    @HOW
    async def test_the_engine_is_closed_exactly_once_on_every_ending(
        self, how: str, spy: type[_SpyEngine]
    ) -> None:
        def boom(ctx: Any) -> None:
            raise RuntimeError("no")

        ok = _executable(_document(_chain(2), ident="ok"))
        failed = _executable(
            _document(
                {"a": _auto("a", action="boom")}, ident="bad", callables=("boom",)
            )
        )
        budget = _executable(_document(_chain(4), ident="long"))
        assert (await _go(how, ok)).status is Status.OK
        context = RunContext(callables=_names(boom=boom))
        assert (await _go(how, failed, context=context)).status is Status.FAILED
        spent = await _go(how, budget, context=RunContext(max_steps=2))
        assert spent.status is Status.ABORTED
        assert [engine.closed for engine in spy.made] == [1, 1, 1]

    @HOW
    async def test_a_close_that_raises_is_logged_and_never_replaces_the_result(
        self, how: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def bad_close(self: Any) -> None:
            raise RuntimeError("close failed")

        monkeypatch.setattr(WorkflowEngine, "close", bad_close)
        result = await _go(how, _executable(_document(_chain(2))))
        assert result.status is Status.OK

    @HOW
    @pytest.mark.parametrize("marker", ["waiting", "running"])
    async def test_an_instance_that_is_not_terminal_after_start_is_a_failure_not_a_hang(
        self, how: str, marker: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The factory refuses pausing steps; this is the guard behind it (a definition the document
        could not have produced is injected)."""
        from fsm_llm.workflows import WorkflowDefinition, auto_step, wait_event_step
        from fsm_llm.workflows.models import WorkflowStepResult
        from fsm_llm.workflows.steps import WorkflowStep

        class _Drift(WorkflowStep):
            next_step_id: str = "end"

            async def execute(self, context: dict[str, Any]) -> WorkflowStepResult:
                return WorkflowStepResult.success_result(next_step_id=None)

        definition = WorkflowDefinition(workflow_id="wf", name="wf")
        if marker == "waiting":
            definition.with_initial_step(
                wait_event_step("w", "w", "never", success_step_id="")
            )
        else:
            definition.with_initial_step(_Drift(step_id="w", name="w"))
            definition.with_step(auto_step("end", "end", next_step_id=""))
        executable = _executable(_document(_chain(1)))
        monkeypatch.setattr(
            WorkflowExecutable, "build_definition", lambda self, *a, **k: definition
        )
        result = await _go(how, executable)
        assert (result.status, result.reason) == (Status.FAILED, None)
        assert result.detail["workflow_status"] == marker

    async def test_two_concurrent_runs_of_one_executable_share_no_state(self) -> None:
        async def tag(ctx: Any) -> dict[str, Any]:
            await asyncio.sleep(0)  # interleave the two runs
            return {"tag": ctx["who"]}

        executable = _executable(
            _document(
                {"a": _auto("a", "b", "tag"), "b": _auto("b")},
                inputs=("who",),
                outputs=("tag",),
                callables=("tag",),
            )
        )
        context = RunContext(callables=_names(tag=tag))
        first, second = await asyncio.wait_for(
            asyncio.gather(
                executable.arun({"who": "one"}, context=context),
                executable.arun({"who": "two"}, context=context),
            ),
            _BOUND,
        )
        assert first.outputs == {"tag": "one"} and second.outputs == {"tag": "two"}
        assert first.handle != second.handle

    @HOW
    async def test_the_template_is_pinned_for_the_run_and_unpinned_after_it(
        self, how: str
    ) -> None:
        events: list[str] = []

        class Pins:
            def pin(self, executable: Any) -> None:
                events.append("pin")

            def unpin(self, executable: Any) -> None:
                events.append("unpin")

        seen: list[list[str]] = []
        executable = _executable(
            _document({"a": _auto("a", action="look")}, callables=("look",)),
            pins=Pins(),
        )
        context = RunContext(callables=_names(look=lambda c: seen.append(list(events))))
        assert (await _go(how, executable, context=context)).status is Status.OK
        assert seen == [["pin"]] and events == ["pin", "unpin"]


class TestSyncName:
    async def test_run_inside_a_running_loop_is_refused_and_starts_nothing(
        self, spy: type[_SpyEngine]
    ) -> None:
        executable = _executable(_document(_chain(1)))
        with pytest.raises(RuntimeError, match="await arun"):
            executable.run({})
        with pytest.raises(RuntimeError, match="await arun"):
            executable.instantiate().run({})
        assert spy.made == []

    def test_run_outside_a_loop_is_asyncio_run_and_leaves_no_loop_behind(self) -> None:
        executable = _executable(_document(_chain(1)))
        assert executable.run({}).status is Status.OK
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        assert (
            executable.run({}).status is Status.OK
        )  # the next call of the same caller


# ---- invariant 6: a composed run inherits (one model, one log, one token, one registry) --------


class TestComposition:
    @HOW
    async def test_a_workflow_that_runs_an_fsm_executable_uses_one_model_one_log_one_token(
        self, how: str, tmp_path: Path
    ) -> None:
        registry = ExecutableRegistry()
        registry.add_document(_fsm_doc())
        step = {
            "type": "ExecutableStep",
            "step_id": "call",
            "name": "call",
            "target": "demo",
            "inputs": {"topic": "subject"},
            "returns": {"final": "result"},
            "success_step_id": "",
        }
        executable = registry.add_document(
            _document(
                {"call": step},
                ident="compose",
                inputs=("subject",),
                outputs=("final",),
                executables=("demo",),
            )
        )
        llm = ScriptedLLM(extraction_data={"result": "42"}, default_response="r")
        token = CancelToken()
        context = RunContext(
            llm_interface=llm, event_log=tmp_path, cancel=token, registry=registry
        )
        result = await _go(how, executable, {"subject": "t"}, context)
        assert result.status is Status.OK and result.outputs == {"final": "42"}
        assert llm.call_history and llm.unexpected == []  # the ONE interface answered
        (log,) = sorted((tmp_path / "logs").glob("*.jsonl"))
        assert "turn/start" in log.read_text(encoding="utf-8")  # the INHERITED log
        assert token.cancelled is False

    @HOW
    async def test_without_a_registry_every_target_is_the_steps_failure_route(
        self, how: str
    ) -> None:
        step = {
            "type": "ExecutableStep",
            "step_id": "call",
            "name": "call",
            "target": "demo",
            "success_step_id": "",
            "error_step_id": "fallback",
        }
        steps = {"call": step, "fallback": _auto("fallback", action="mark")}
        executable = _executable(
            _document(steps, executables=("demo",), callables=("mark",))
        )
        marks: list[int] = []
        context = RunContext(callables=_names(mark=lambda c: marks.append(1)))
        result = await _go(how, executable, context=context)
        assert result.status is Status.OK and marks == [1]
        assert WorkflowStatus.COMPLETED.value == result.detail["workflow_status"]


# ---- one bad document stops nothing (review of plan 7d9c0955, D-012, finding 1) ----------------


def _write_documents(directory: Path, bad_action: str) -> None:
    bad = _document({"a": _auto("a", "", action=bad_action)}, callables=(bad_action,))
    (directory / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    good = _document(_chain(2), ident="good")
    (directory / "good.json").write_text(json.dumps(good), encoding="utf-8")


class TestOneBadFileStopsNothing:
    @pytest.mark.parametrize("name", ["$run.callables", "$run.registry", " "])
    def test_a_directory_with_a_bad_file_serves_the_good_one_and_warns_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        _write_documents(tmp_path, name)
        warned: list[str] = []
        registry = ExecutableRegistry([tmp_path])
        monkeypatch.setattr(registry_module.logger, "warning", warned.append)

        report = registry.reload()

        assert [e.id for e in registry.list()] == ["good"]
        assert list(report.failed) and all("bad.json" in key for key in report.failed)
        assert len(warned) == 1 and "bad.json" in warned[0]

    def test_load_executable_reports_the_document(self) -> None:
        document = _document(
            {"a": _auto("a", "", action="$run.callables")},
            callables=("$run.callables",),
        )

        envelope, result = load_executable(document)

        assert envelope is None and not result.is_valid

    def test_the_directory_loader_of_the_monitor_process_serves_the_good_file(
        self, tmp_path: Path
    ) -> None:
        _write_documents(tmp_path, "$run.callables")

        registry = load_directory_registry(tmp_path, [], prog="fsm-llm-monitor")

        assert [e.id for e in registry.list(kind="workflow")] == ["good"]

    @pytest.mark.parametrize("name", ["$run.callables", "$run.registry"])
    def test_the_factory_alone_refuses_the_document_as_an_executable_error(
        self, name: str
    ) -> None:
        envelope = parse_executable(
            _document({"a": _auto("a", "", action=name)}, callables=(name,))
        )

        with pytest.raises(ExecutableError) as caught:
            workflow_adapter(envelope)

        assert len(str(caught.value)) < 400

    def test_a_host_registry_holding_a_reserved_name_is_a_validation_error(
        self,
    ) -> None:
        executable = _executable(_document(_chain(1)))
        host = _names(**{"$run.registry": lambda c: None})

        with pytest.raises(WorkflowValidationError) as caught:
            executable.build_definition(host)

        assert "$run.registry" not in str(caught.value)

    def test_a_run_with_such_a_host_registry_is_refused_not_crashed(self) -> None:
        executable = _executable(_document(_chain(1)))
        context = RunContext(callables=_names(**{"$run.callables": lambda c: None}))

        with pytest.raises(WorkflowValidationError):
            asyncio.run(executable.arun({}, context=context))


# ---- the factory (V2 in structure-only form) at the registry ------------------------------------


class TestFactory:
    @pytest.mark.parametrize(
        "step",
        [
            {
                "type": "WaitEventStep",
                "step_id": "w",
                "name": "w",
                "config": {"event_type": "go", "success_step_id": ""},
            },
            {
                "type": "TimerStep",
                "step_id": "w",
                "name": "w",
                "delay_seconds": 1,
                "next_step_id": "",
            },
            {
                "type": "RetryStep",
                "step_id": "w",
                "name": "w",
                "step": {
                    "type": "TimerStep",
                    "step_id": "inner",
                    "name": "i",
                    "delay_seconds": 1,
                    "next_step_id": "",
                },
            },
        ],
    )
    def test_a_registry_refuses_a_pausing_step_with_one_capped_line(
        self, step: dict[str, Any]
    ) -> None:
        document = _document({"w": step})
        with pytest.raises(ExecutableDocumentError) as caught:
            ExecutableRegistry().add_document(document)
        assert (
            len(caught.value.problems) == 1
            and "cannot pause" in caught.value.problems[0]
        )
        assert len(str(caught.value)) < 400
