"""The default ``BaseAgent.arun``: one awaitable name for every agent (plan ace7a8c8 X6.1, D-065).

``arun`` is NOT abstract and runs the agent's sync ``run`` through ``aoffload`` (a worker thread in
async mode, inline in sync mode). It never goes through ``acall(self, "run")``: async mode would find
``self.arun`` first and recurse. A sync-only subclass still instantiates; ``run`` stays abstract. A cancelled
await returns at once and the worker keeps running (the documented ``aoffload`` hazard).
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm.agents.base import BaseAgent
from fsm_llm.agents.definitions import AgentResult
from fsm_llm.aio import run_sync
from fsm_llm.core.codes import Status

_BOUND = 10  # seconds: a wait that would hang fails instead


def test_the_module_under_test_is_this_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    assert Path(fsm_llm.__file__).resolve().is_relative_to(root / "src")


class _SyncOnly(BaseAgent):
    """What 8 test classes and every user subclass look like: only ``run``."""

    def __init__(self) -> None:
        super().__init__()
        self.threads: list[int] = []
        self.calls: list[tuple[Any, ...]] = []

    def run(self, task, initial_context=None):
        self.threads.append(threading.get_ident())
        self.calls.append((task, initial_context))
        return AgentResult(answer=f"ran {task}", status=Status.OK)

    def _register_handlers(self, api):
        pass


class _NarrowRun(BaseAgent):
    """A ``run`` that takes no ``initial_context``: the arguments are forwarded verbatim."""

    def run(self, task):
        return AgentResult(answer=f"narrow {task}", status=Status.OK)

    def _register_handlers(self, api):
        pass


class TestTheDefault:
    def test_a_sync_only_subclass_still_instantiates(self) -> None:
        assert _SyncOnly().run("x").answer == "ran x"

    def test_run_stays_abstract(self) -> None:
        class NoRun(BaseAgent):
            def _register_handlers(self, api):
                pass

        with pytest.raises(TypeError, match="abstract"):
            NoRun()  # type: ignore[abstract]

    def test_arun_is_a_coroutine_function_not_an_abstract_method(self) -> None:
        import inspect

        assert inspect.iscoroutinefunction(BaseAgent.arun)
        assert "arun" not in BaseAgent.__abstractmethods__

    async def test_the_worker_runs_the_sync_body_off_the_loop(self) -> None:
        agent = _SyncOnly()
        loop_thread = threading.get_ident()
        result = await agent.arun("task", {"k": "v"})
        assert result.answer == "ran task"
        assert agent.calls == [("task", {"k": "v"})]
        assert agent.threads and agent.threads[0] != loop_thread

    async def test_the_arguments_are_forwarded_verbatim(self) -> None:
        assert (await _NarrowRun().arun("t")).answer == "narrow t"
        agent = _SyncOnly()
        await agent.arun("t", initial_context={"a": 1})
        assert agent.calls == [("t", {"a": 1})]

    def test_in_sync_mode_it_runs_inline_on_the_caller_thread_with_no_loop(
        self,
    ) -> None:
        agent = _SyncOnly()
        result = run_sync(agent.arun("task"))
        assert result.answer == "ran task"
        assert agent.threads == [threading.get_ident()]

    async def test_an_error_of_run_is_raised_by_arun(self) -> None:
        class Boom(_SyncOnly):
            def run(self, task, initial_context=None):
                raise ValueError("from run")

        with pytest.raises(ValueError, match="from run"):
            await Boom().arun("t")

    async def test_arun_does_not_recurse_through_acall(self) -> None:
        """``acall(self, "run")`` would find ``self.arun`` first and call itself forever."""
        agent = _SyncOnly()
        for _ in range(3):
            await agent.arun("t")
        assert len(agent.calls) == 3

    async def test_an_instance_patch_of_run_is_called(self) -> None:
        agent = _SyncOnly()
        seen: list[str] = []

        def fake(task, initial_context=None):
            seen.append(task)
            return AgentResult(answer="patched", status=Status.OK)

        agent.run = fake  # type: ignore[method-assign]
        assert (await agent.arun("t")).answer == "patched"
        assert seen == ["t"]

    async def test_a_class_qualified_patch_of_run_is_called(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(self, task, initial_context=None):
            return AgentResult(answer="class patched", status=Status.OK)

        monkeypatch.setattr(_SyncOnly, "run", fake)
        assert (await _SyncOnly().arun("t")).answer == "class patched"

    async def test_a_subclass_with_its_own_arun_is_native_and_its_run_is_bypassed(
        self,
    ) -> None:
        class Native(_SyncOnly):
            async def arun(self, task, *args, **kwargs):
                return AgentResult(answer="native", status=Status.OK)

        agent = Native()
        assert (await agent.arun("t")).answer == "native"
        assert agent.calls == []


class TestTheCancelHazard:
    async def test_a_cancel_returns_at_once_and_the_worker_runs_to_its_end(
        self,
    ) -> None:
        entered, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        class Slow(_SyncOnly):
            def run(self, task, initial_context=None):
                entered.set()
                assert release.wait(_BOUND)
                finished.set()
                return AgentResult(answer="late", status=Status.OK)

        task = asyncio.ensure_future(Slow().arun("t"))
        assert await asyncio.to_thread(entered.wait, _BOUND)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _BOUND)
        assert not finished.is_set()  # the cancel did not wait for the worker
        release.set()
        assert await asyncio.to_thread(finished.wait, _BOUND)  # it ran on

    async def test_two_runs_overlap_on_one_loop(self) -> None:
        barrier = threading.Barrier(2, timeout=_BOUND)

        class Meet(_SyncOnly):
            def run(self, task, initial_context=None):
                barrier.wait()  # both workers must be inside run at once
                return AgentResult(answer=task, status=Status.OK)

        a, b = await asyncio.gather(Meet().arun("a"), Meet().arun("b"))
        assert (a.answer, b.answer) == ("a", "b")
