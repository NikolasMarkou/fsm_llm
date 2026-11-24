"""The async seam of the agents base: the loop, the standard run and its stream (plan ace7a8c8, step 2).

``_arun_conversation_loop``, ``_astandard_run`` and ``_astandard_run_stream`` are the one bodies;
the sync names are thin wrappers. A row runs the same scripted work sync and async on a double whose
``a*`` twins really suspend (``_Suspending``) and compares what a caller sees. Rows use events and
bounded waits, never sleeps or the clock. The mutant each row kills is named in its docstring.
"""

from __future__ import annotations

import ast
import asyncio
import gc
import inspect
import textwrap
import threading
import time
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.agents import AgentConfig, HumanInTheLoop, ReactAgent, ToolRegistry
from fsm_llm.agents.base import BaseAgent
from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError
from fsm_llm.testing import PromptGroundedLLM
from tests.test_fsm_llm_agents.test_advance_driver import (
    _ANSWER,
    _FACTS,
    _HITL_TASK,
    _TASK,
    _endless_fsm,
    _EndlessAgent,
    _HitlLLM,
    _registry,
)

_BOUND = 10  # seconds: a wait that would hang fails instead


class _Suspending(PromptGroundedLLM):
    """The ``a*`` twins really suspend: every call counts a hop, and the hops in ``gates`` park
    until ``release`` is set (``entered`` is set once ``want`` calls are parked)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hops, self.parked, self.want, self.gates = 0, 0, 1, set[int]()
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def _hop(self) -> None:
        self.hops += 1
        if self.hops not in self.gates:
            return await asyncio.sleep(0)
        self.parked += 1
        if self.parked >= self.want:
            self.entered.set()
        await self.release.wait()

    async def aextract_field(self, request):
        await self._hop()
        return await super().aextract_field(request)

    async def aextract_bulk_data(self, request):
        await self._hop()
        return await super().aextract_bulk_data(request)

    async def agenerate_response(self, request):
        await self._hop()
        return await super().agenerate_response(request)


class _HitlSuspending(_Suspending, _HitlLLM):
    pass


class _Agent(ReactAgent):
    """A real ``BaseAgent`` whose ``arun`` is the native body (the patterns convert in step 3)."""

    async def arun(self, task, initial_context=None):
        fsm_def, context, handlers = self._react_inputs(task, initial_context)
        return await self._astandard_run(
            task, fsm_def, context, "react", handlers=handlers
        )

    def arun_stream(self, task, initial_context=None):
        fsm_def, context, handlers = self._react_inputs(task, initial_context)
        return self._astandard_run_stream(
            task, fsm_def, context, "react", handlers=handlers
        )


class _Rig:
    """An agent over a ``_Suspending`` double and what its runs did to the APIs they built:
    ``ended`` is the history just before each conversation was ended, ``before`` the history
    at each hook call (just before a step)."""

    def __init__(self, llm: _Suspending | None = None, **kwargs: Any) -> None:
        self.llm = llm or _Suspending(facts=_FACTS, default_response=_ANSWER)
        self.agent = _Agent(
            tools=_registry([]),
            config=AgentConfig(max_iterations=6),
            llm_interface=self.llm,
            **kwargs,
        )
        self.apis: list[API] = []
        self.ended: list[Any] = []
        self.before: list[Any] = []
        create, hook = self.agent._create_api, self.agent._on_loop_iteration

        def _create(fsm_def: dict[str, Any]) -> API:
            api = create(fsm_def)
            end = api.end_conversation

            def _end(conversation_id: str) -> None:
                self.ended.append(api.get_conversation_history(conversation_id))
                end(conversation_id)

            api.end_conversation = _end  # type: ignore[method-assign]
            self.apis.append(api)
            return api

        def _hook(api: API, conversation_id: str, step: int) -> None:
            self.before.append(api.get_conversation_history(conversation_id))
            hook(api, conversation_id, step)

        self.agent._create_api = _create  # type: ignore[method-assign]
        self.agent._on_loop_iteration = _hook  # type: ignore[method-assign]


def _view(value: Any) -> Any:
    """A result as plain data with the clock readings left out."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        return {k: _view(v) for k, v in value.items() if k != "timestamp"}
    return [_view(v) for v in value] if isinstance(value, list) else value


async def _cancel_at(rig: _Rig, work: Any) -> asyncio.Task:
    """Run ``work`` until the parked await, cancel it there, return the finished task."""
    task = asyncio.create_task(work)
    await asyncio.wait_for(rig.llm.entered.wait(), _BOUND)
    task.cancel()
    await asyncio.wait({task}, timeout=_BOUND)
    return task


class _AsyncEndless(_EndlessAgent):
    async def arun(self, task, initial_context=None):
        context = self._init_context(task, initial_context)
        return await self._astandard_run(task, _endless_fsm(), context, "endless")


class TestParity:
    async def test_arun_equals_run_and_every_model_call_goes_through_its_twin(self):
        """Kills: an async body that differs from the sync one, a model call that takes the
        sync name in async mode (the hops fall short of the requests)."""
        want = _Rig().agent.run(_TASK)
        rig = _Rig()
        got = await rig.agent.arun(_TASK)
        assert got.success and _view(got) == _view(want)
        assert rig.llm.hops == len(rig.llm.requests) > 0

    async def test_the_stream_chunks_equal_the_sync_stream_and_the_sync_stream_is_lazy(
        self,
    ):
        """Kills: a stream body that differs, a sync stream that runs before its first
        ``next()``."""
        want = list(_Rig().agent.run_stream(_TASK))
        rig = _Rig()
        fsm_def, context, handlers = rig.agent._react_inputs(_TASK, None)
        stream = rig.agent._standard_run_stream(
            _TASK, fsm_def, context, "react", handlers=handlers
        )
        assert rig.apis == [] and rig.llm.requests == []
        assert list(stream) == want == [_ANSWER]
        got = [chunk async for chunk in _Rig().agent.arun_stream(_TASK)]
        assert got == want

    async def test_a_consumer_that_breaks_leaves_no_error_and_no_conversation(self):
        """Kills: a stream cleanup that needs an ``aclose`` the consumer never sends."""
        errors: list[Any] = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: errors.append(context)
        )
        rig = _Rig()
        async for _ in rig.agent.arun_stream(_TASK):
            break
        for _ in range(5):
            gc.collect()
            await asyncio.sleep(0)
        assert errors == [] and rig.apis[0].list_active_conversations() == []

    @pytest.mark.parametrize("mode", ["sync", "async"])
    async def test_a_run_budget_raises_the_same_error_in_both_modes(self, mode):
        """Kills: a budget error that changes type or text in async mode, a run that leaves
        its conversation behind."""
        agent = _AsyncEndless(
            AgentConfig(max_iterations=2, timeout_seconds=5.0),
            llm_interface=PromptGroundedLLM(),
        )
        api = agent._create_api(_endless_fsm())
        spent = time.monotonic() - 6.0
        for error, text, sync_call, async_call in [
            (
                BudgetExhaustedError,
                "Agent budget exhausted: iterations limit (6) reached (6 loop turns",
                lambda: agent.run(_TASK),
                lambda: agent.arun(_TASK),
            ),
            (
                AgentTimeoutError,
                "Agent timed out after 5.0 seconds",
                lambda: agent._run_conversation_loop(api, {}, spent, "endless"),
                lambda: agent._arun_conversation_loop(api, {}, spent, "endless"),
            ),
        ]:
            with pytest.raises(error) as info:
                await async_call() if mode == "async" else sync_call()
            assert str(info.value).startswith(text)
        assert api.list_active_conversations() == []


class TestInterleave:
    async def test_two_runs_overlap_on_one_loop_while_their_model_calls_are_suspended(
        self,
    ):
        """Kills: a model call made on the loop thread (no run could start while one waits),
        a serial run (the second never reaches its call, ``entered`` never sets)."""
        rig = _Rig()
        rig.llm.gates, rig.llm.want = {1, 2}, 2
        ticks = 0

        async def beat() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        runs = [asyncio.create_task(rig.agent.arun(_TASK)) for _ in range(2)]
        heart = asyncio.create_task(beat())
        await asyncio.wait_for(rig.llm.entered.wait(), _BOUND)
        seen = ticks
        while ticks < seen + 3:  # the loop is free while both calls are parked
            await asyncio.sleep(0)
        rig.llm.release.set()
        first, second = await asyncio.wait_for(asyncio.gather(*runs), _BOUND)
        heart.cancel()
        assert _view(first) == _view(second) and first.success


class TestHitlOrdering:
    @pytest.mark.parametrize("mode", ["sync", "async"])
    async def test_the_approval_is_asked_before_the_step_it_gates(
        self, mode, monkeypatch
    ):
        """Kills: a hook scheduled and not awaited (the ask lands after the step), a hook
        passed as a sync ``partial`` (it runs on the loop thread in async mode)."""
        events: list[tuple[Any, ...]] = []
        tools = ToolRegistry()

        def transfer(account: str, amount: int) -> str:
            events.append(("tool", account, amount))
            return f"Transferred {amount} from {account}"

        tools.register_function(
            transfer, name="transfer", description="Move money", requires_approval=True
        )
        hitl = HumanInTheLoop(
            approval_callback=lambda request: (
                events.append(("ask", threading.get_ident())) or True
            )
        )
        agent = _Agent(
            tools=tools,
            config=AgentConfig(max_iterations=6),
            hitl=hitl,
            llm_interface=_HitlSuspending(default_response="Done."),
        )
        aadvance = API.aadvance

        async def _aadvance(api: API, conversation_id: str) -> Any:
            events.append(("step", api.get_current_state(conversation_id)))
            return await aadvance(api, conversation_id)

        monkeypatch.setattr(API, "aadvance", _aadvance)

        result = (
            agent.run(_HITL_TASK) if mode == "sync" else await agent.arun(_HITL_TASK)
        )

        (asked,) = [e for e in events if e[0] == "ask"]
        events[events.index(asked)] = ("ask",)
        assert events == [
            ("step", "think"),
            ("ask",),
            ("step", "await_approval"),
            ("tool", "A-17", 250),
            ("step", "act"),
            ("step", "think"),
        ]
        assert (asked[1] == threading.get_ident()) is (mode == "sync")
        assert (result.success, result.answer) == (True, "Done.")


class TestCancel:
    async def test_a_cancel_at_every_model_await_ends_the_run_and_frees_the_gate(self):
        """Kills: ``_end_run_conversation`` outside the ``finally`` (an active conversation),
        an ``except`` widened to ``BaseException`` (an ``AgentError`` replaces the cancel), a
        step that is not rolled back (the history differs from the one before the step)."""
        control = _Rig()
        await control.agent.arun(_TASK)
        assert control.llm.hops > 0
        for k in range(1, control.llm.hops + 1):
            rig = _Rig()
            rig.llm.gates = {k}
            task = await _cancel_at(rig, rig.agent.arun(_TASK))
            assert task.cancelled(), f"k={k}"  # not swallowed, not an AgentError
            (api,) = rig.apis
            assert api.list_active_conversations() == [], f"k={k}"
            assert not api.fsm_manager._active_turns, f"k={k}"
            assert rig.ended == [rig.before[-1]], f"k={k}"


class TestSyncNames:
    @pytest.mark.parametrize(
        ("name", "wrapper"),
        [
            ("_run_conversation_loop", "run_sync"),
            ("_standard_run", "run_sync"),
            ("_standard_run_stream", "iterate_sync"),
        ],
    )
    def test_each_sync_name_is_a_docstring_and_one_return_of_its_wrapper(
        self, name, wrapper
    ):
        """Kills: a sync name that keeps its own copy of the body (the one-body rule)."""
        source = textwrap.dedent(inspect.getsource(getattr(BaseAgent, name)))
        body = ast.parse(source).body[0].body  # type: ignore[attr-defined]
        assert isinstance(body[0], ast.Expr) and len(body) == 2
        call = body[1].value  # type: ignore[attr-defined]
        assert (call.func.id, call.args[0].func.attr) == (wrapper, "_a" + name[1:])

    def test_a_patched_sync_hop_still_intercepts_in_sync_mode(self):
        """Kills: a sync run that reaches the loop body directly (an instance patch of
        ``_run_conversation_loop`` is bypassed)."""
        agent = _EndlessAgent(
            AgentConfig(max_iterations=2), llm_interface=PromptGroundedLLM()
        )
        calls: list[str] = []

        def spy(*args: Any, **kwargs: Any) -> Any:
            calls.append("loop")
            raise AgentTimeoutError(1.0)

        agent._run_conversation_loop = spy  # type: ignore[method-assign]
        with pytest.raises(AgentTimeoutError):
            agent.run(_TASK)
        assert calls == ["loop"]
