"""The workflow steps that call a model await it on the loop (plan ace7a8c8 step 26, X5.1, D-009).

``AgentStep`` awaits ``agent.arun`` when the agent has its own ``async def arun``, ``ConversationStep``
runs its whole conversation as one awaited coroutine (``API.astart_conversation`` / ``aconverse``) and
``LLMStep`` awaits ``agenerate_response`` of a real ``LLMInterface``. A workflow step has no sync mode, so
none of them goes through ``acall`` and none uses the engine ``executor=``. A time limit cancels the
in-flight turn (the core rolls it back whole, f150ec28 D-002) and the conversation is still ended.

Rows run the REAL core with a scripted double that records the thread of every model call. Cancel rows
use events and bounded waits; the only clock is the step's own time limit. Run on 3.12, 3.14 and 3.15 with
``scripts/gate.sh venv tests/test_fsm_llm_workflows/test_native_steps.py`` (D7: a cancel rolls back, the
conversation is ended once, the gate is free).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, ClassVar

import pytest

import fsm_llm
from fsm_llm import API
from fsm_llm.llm import LLMInterface
from fsm_llm.testing import ScriptedLLM
from fsm_llm.workflows.steps import AgentStep, ConversationStep, LLMStep, ParallelStep
from fsm_llm.workflows.steps.helpers import _STEP_EXECUTOR
from tests.test_fsm_llm import kind_corpus as kc

_BOUND = 10  # seconds: a wait that would hang fails instead


def test_the_module_under_test_is_this_checkout() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert Path(fsm_llm.__file__).resolve().is_relative_to(root / "src")


class _NoExecutor:
    """A step executor that fails when asked to run anything."""

    def submit(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("a native step must not use the engine executor")


@pytest.fixture
def no_executor():
    token = _STEP_EXECUTOR.set(_NoExecutor())  # type: ignore[arg-type]
    try:
        yield
    finally:
        _STEP_EXECUTOR.reset(token)


# ---- doubles ----------------------------------------------------------------------------------


class _Probe(ScriptedLLM):
    """A scripted model that records the thread of every call, and may park or meet a barrier.

    ``park_at`` (1-based) parks that ``agenerate_response`` until it is cancelled (``entered`` is set);
    ``barrier`` makes the FIRST ``agenerate_response`` wait for the other party (two of them overlap
    only when both run on one loop at once). ``cancelled`` counts the ``CancelledError`` it saw.
    """

    def __init__(self, *, park_at: int = 0, barrier: Any = None, **kwargs: Any) -> None:
        kwargs.setdefault("extraction_data", {})
        kwargs.setdefault("default_response", "hi")
        super().__init__(**kwargs)
        self.park_at, self.barrier = park_at, barrier
        self.threads: list[int] = []
        self.awaits = 0
        self.cancelled = 0
        self.entered = threading.Event()

    def generate_response(self, request):
        self.threads.append(threading.get_ident())
        return super().generate_response(request)

    def extract_field(self, request):
        self.threads.append(threading.get_ident())
        return super().extract_field(request)

    async def agenerate_response(self, request):
        self.threads.append(threading.get_ident())
        self.awaits += 1
        try:
            if self.park_at == self.awaits:
                self.entered.set()
                await asyncio.Event().wait()
            if self.barrier is not None and self.awaits == 1:
                await asyncio.wait_for(self.barrier.wait(), _BOUND)
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return ScriptedLLM.generate_response(self, request)


def _chat() -> dict[str, Any]:
    """chat (speaks) -> bye; the conversation does not end on its own."""
    return kc._fsm(
        "Chat",
        "chat",
        kc._state(
            "chat",
            response_instructions="Reply.",
            transitions=[
                kc._go(
                    "bye", conditions=kc._when("done", {"==": [{"var": "done"}, True]})
                )
            ],
        ),
        kc._state("bye", response_instructions="Bye."),
    )


class _RecordingAPI(API):
    """The ``API`` a ``ConversationStep`` builds, recording what happened to its conversations."""

    made: ClassVar[list[_RecordingAPI]] = []
    llms: ClassVar[list[_Probe]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ended: list[dict[str, Any]] = []
        type(self).made.append(self)

    def end_conversation(self, conversation_id: str) -> None:
        manager = self.fsm_manager
        context = manager.instances[conversation_id].context
        self.ended.append(
            {
                "id": conversation_id,
                "history": [dict(e) for e in context.conversation.exchanges],
                "gate_free": not manager._active_turns,
            }
        )
        super().end_conversation(conversation_id)


@pytest.fixture
def make_api(monkeypatch: pytest.MonkeyPatch):
    """Patch ``API.from_definition`` (what the step reads) to build a recording API on a probe."""
    _RecordingAPI.made, _RecordingAPI.llms = [], []
    real = API.from_definition.__func__  # type: ignore[attr-defined]

    def install(*probes: _Probe) -> list[_RecordingAPI]:
        queue = list(probes)

        def from_definition(cls, definition, **kwargs):
            probe = queue.pop(0)
            _RecordingAPI.llms.append(probe)
            return real(_RecordingAPI, definition, llm_interface=probe)

        monkeypatch.setattr(API, "from_definition", classmethod(from_definition))
        return _RecordingAPI.made

    return install


def _conversation(step_id: str = "c", **kwargs: Any) -> ConversationStep:
    kwargs.setdefault("auto_messages", ["hello"])
    return ConversationStep(
        step_id=step_id,
        name="C",
        fsm_definition=_chat(),
        success_step_id="next",
        error_step_id="err",
        **kwargs,
    )


# ---- ConversationStep -------------------------------------------------------------------------


class TestConversationStepIsNative:
    async def test_the_whole_conversation_runs_on_the_loop_thread(
        self, make_api, no_executor
    ) -> None:
        """No executor thread for a conversation: every model call of the greeting and of the turn
        runs on the loop thread, and the sync twins are never reached. Kills: the executor hop kept
        (``run_in_executor`` of a sync body)."""
        probe = _Probe()
        made = make_api(probe)
        result = await _conversation(context_mapping={"said": "last_response"}).execute(
            {}
        )
        assert (result.success, result.next_step_id) == (True, "next")
        assert result.data["said"] == "hi"
        assert result.data["conversation_c_ended"] is False
        assert probe.threads and set(probe.threads) == {threading.get_ident()}
        assert len(made) == 1 and len(made[0].ended) == 1

    async def test_the_result_equals_the_one_of_the_sync_path(self, make_api) -> None:
        """The same messages give the same data the executor route gave (``last_response`` and
        ``final_answer`` hold the last SPOKEN reply, 07ad3f8c/D-039)."""
        make_api(_Probe(default_response="spoken"))
        result = await _conversation(
            context_mapping={"a": "final_answer", "b": "last_response"}
        ).execute({})
        assert (result.data["a"], result.data["b"]) == ("spoken", "spoken")
        assert result.message == "Conversation completed in 1 turns"

    async def test_a_time_limit_cancels_the_turn_rolls_it_back_and_ends_the_conversation(
        self, make_api, timing_factor
    ) -> None:
        """Park the model in the user turn (the 2nd Pass-2 await; the 1st is the greeting). The step's
        limit cancels it: the double sees ``CancelledError``, the user message is no longer in the
        history and the gate is free when ``end_conversation`` runs, and it runs exactly once. Kills:
        ``end_conversation`` outside ``finally`` (never called on the cancel), the executor hop kept
        (the limit cannot reach a running thread)."""
        probe = _Probe(park_at=2)
        made = make_api(probe)
        step = _conversation(timeout_seconds=0.2 * timing_factor)
        result = await asyncio.wait_for(step.execute({}), _BOUND)
        assert result.success is False
        assert result.next_step_id == "err"
        assert "timed out after" in str(result.error)
        assert probe.entered.is_set() and probe.cancelled == 1
        (ended,) = made[0].ended
        assert ended["gate_free"] is True
        assert all("hello" not in str(e) for e in ended["history"])  # rolled back
        assert made[0].fsm_manager.instances == {}  # and ended

    async def test_the_smaller_of_the_two_limits_applies(
        self, make_api, timing_factor
    ) -> None:
        probe = _Probe(park_at=2)
        make_api(probe)
        step = _conversation(
            timeout_seconds=60.0, conversation_timeout=0.2 * timing_factor
        )
        result = await asyncio.wait_for(step.execute({}), _BOUND)
        assert "Conversation timed out after" in str(result.error)
        assert probe.cancelled == 1

    async def test_an_outer_cancel_is_never_swallowed(self, make_api) -> None:
        """A cancel from outside (the workflow deadline) ends the conversation and propagates."""
        probe = _Probe(park_at=2)
        made = make_api(probe)
        task = asyncio.create_task(_conversation().execute({}))
        await asyncio.to_thread(probe.entered.wait, _BOUND)
        task.cancel()
        await asyncio.wait({task}, timeout=_BOUND)
        assert task.cancelled()
        assert len(made[0].ended) == 1 and probe.cancelled == 1

    async def test_a_timeout_raised_by_the_conversation_is_a_failure_not_the_step_limit(
        self, make_api
    ) -> None:
        class Raising(_Probe):
            async def agenerate_response(self, request):
                if self.awaits >= 1:
                    raise TimeoutError("provider timeout")
                return await super().agenerate_response(request)

        make_api(Raising())
        result = await _conversation(timeout_seconds=30.0).execute({})
        assert result.success is False and result.next_step_id == "err"
        assert "provider timeout" in str(result.error)

    async def test_two_conversations_of_a_parallel_step_overlap_on_one_loop(
        self, make_api
    ) -> None:
        """Each child waits at a barrier of two inside its model call: they finish only when both are
        in flight at once, on one thread (an executor hop per conversation would not fill an
        ``asyncio`` barrier from two threads' sync calls)."""
        barrier = asyncio.Barrier(2)
        probes = [_Probe(barrier=barrier), _Probe(barrier=barrier)]
        make_api(*probes)
        parallel = ParallelStep(
            step_id="p",
            name="P",
            steps=[_conversation("c1"), _conversation("c2")],
            next_step_id="next",
        )
        result = await asyncio.wait_for(parallel.execute({}), _BOUND)
        assert result.success is True
        assert all(p.awaits >= 1 for p in probes)
        assert {t for p in probes for t in p.threads} == {threading.get_ident()}


# ---- AgentStep --------------------------------------------------------------------------------


class _Answer:
    def __init__(self, answer: str, success: bool = True, final_context: Any = None):
        self.answer, self.success = answer, success
        self.final_context = final_context or {}
        self.structured_output = None


class _NativeAgent:
    """An agent with its own ``async def arun``; its sync ``run`` must never be called."""

    def __init__(self, *, park: bool = False, barrier: Any = None) -> None:
        self.park, self.barrier = park, barrier
        self.threads: list[int] = []
        self.tasks: list[tuple[str, dict]] = []
        self.cancelled = 0
        self.entered = threading.Event()

    def run(self, task, **kwargs):
        raise AssertionError("the sync run of a native agent must not be called")

    async def arun(self, task, **kwargs):
        self.threads.append(threading.get_ident())
        self.tasks.append((task, kwargs))
        try:
            if self.park:
                self.entered.set()
                await asyncio.Event().wait()
            if self.barrier is not None:
                await asyncio.wait_for(self.barrier.wait(), _BOUND)
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return _Answer("native answer", final_context={"k": "v"})


class _SyncAgent:
    def __init__(self) -> None:
        self.threads: list[str] = []

    def run(self, task, **kwargs):
        self.threads.append(threading.current_thread().name)
        return _Answer("sync answer", final_context={"k": "v"})


def _agent_step(agent: Any, **kwargs: Any) -> AgentStep:
    return AgentStep(
        step_id="a",
        name="A",
        agent=agent,
        task_template="Do {topic}",
        success_step_id="next",
        error_step_id="err",
        context_mapping={"got": "k"},
        **kwargs,
    )


class TestAgentStepIsNative:
    async def test_an_agent_with_its_own_arun_is_awaited_on_the_loop(
        self, no_executor
    ) -> None:
        """The task and the input mapping reach ``arun``; the result maps like the sync one. Kills:
        the executor hop kept (``run`` is called, and the executor fails)."""
        agent = _NativeAgent()
        step = _agent_step(agent, input_mapping={"ctx_key": "wf_key"})
        result = await step.execute({"topic": "x", "wf_key": 7})
        assert (result.success, result.next_step_id) == (True, "next")
        assert agent.tasks == [("Do x", {"initial_context": {"ctx_key": 7}})]
        assert agent.threads == [threading.get_ident()]
        assert result.data["agent_answer"] == "native answer"
        assert result.data["agent_a_answer"] == "native answer"
        assert result.data["got"] == "v"

    async def test_a_sync_only_agent_still_runs_in_the_step_executor(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        agent = _SyncAgent()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent26")
        token = _STEP_EXECUTOR.set(executor)
        try:
            result = await _agent_step(agent).execute({"topic": "x"})
        finally:
            _STEP_EXECUTOR.reset(token)
            executor.shutdown()
        assert result.success is True and result.data["agent_answer"] == "sync answer"
        assert agent.threads == ["agent26_0"]

    async def test_a_base_agent_with_the_default_arun_keeps_the_executor_route(
        self,
    ) -> None:
        """The default ``BaseAgent.arun`` (X6.1, D-065) is NOT native: ``AgentStep`` still runs its
        ``run`` in the step executor. Kills: the default treated as native (``arun`` is awaited, which
        runs ``run`` on the default executor and the recorded thread name is not ``agent27``)."""
        from concurrent.futures import ThreadPoolExecutor

        from fsm_llm.agents.base import BaseAgent
        from fsm_llm.agents.definitions import AgentResult
        from fsm_llm.core.codes import Status

        threads: list[str] = []

        class Plain(BaseAgent):
            def run(self, task, initial_context=None):
                threads.append(threading.current_thread().name)
                return AgentResult(answer="plain answer", status=Status.OK)

            def _register_handlers(self, api):
                pass

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent27")
        token = _STEP_EXECUTOR.set(executor)
        try:
            result = await _agent_step(Plain()).execute({"topic": "x"})
        finally:
            _STEP_EXECUTOR.reset(token)
            executor.shutdown()
        assert result.success is True and result.data["agent_answer"] == "plain answer"
        assert threads == ["agent27_0"]

    async def test_a_time_limit_cancels_a_native_arun(self, timing_factor) -> None:
        agent = _NativeAgent(park=True)
        step = _agent_step(agent, timeout_seconds=0.2 * timing_factor)
        result = await asyncio.wait_for(step.execute({"topic": "x"}), _BOUND)
        assert result.success is False and result.next_step_id == "err"
        assert "timed out" in str(result.error)
        assert agent.cancelled == 1  # a real cancel, not a straggler thread

    async def test_an_arun_that_raises_is_the_error_route(self) -> None:
        class Boom(_NativeAgent):
            async def arun(self, task, **kwargs):
                raise RuntimeError("agent crashed")

        result = await _agent_step(Boom()).execute({"topic": "x"})
        assert (result.success, result.next_step_id) == (False, "err")
        assert "agent crashed" in str(result.error)

    async def test_two_native_agents_of_a_parallel_step_overlap(self) -> None:
        barrier = asyncio.Barrier(2)
        agents = [_NativeAgent(barrier=barrier), _NativeAgent(barrier=barrier)]
        parallel = ParallelStep(
            step_id="p",
            name="P",
            steps=[
                AgentStep(step_id=f"a{i}", name="A", agent=agent)
                for i, agent in enumerate(agents)
            ],
            next_step_id="next",
        )
        result = await asyncio.wait_for(parallel.execute({"task": "t"}), _BOUND)
        assert result.success is True
        assert all(len(a.threads) == 1 for a in agents)

    async def test_a_bare_mock_agent_keeps_the_executor_route(self) -> None:
        from unittest.mock import MagicMock

        agent = MagicMock()
        agent.run.return_value = _Answer("from mock")
        result = await AgentStep(step_id="a", name="A", agent=agent).execute(
            {"task": "t"}
        )
        assert result.success is True and result.data["agent_answer"] == "from mock"
        agent.run.assert_called_once_with("t")


# ---- LLMStep ----------------------------------------------------------------------------------


def _llm_step(llm: Any, **kwargs: Any) -> LLMStep:
    return LLMStep(
        step_id="l",
        name="L",
        llm_interface=llm,
        prompt_template="Say {word}",
        context_mapping={"word": "word"},
        output_mapping={"out": ""},
        next_step_id="next",
        error_step_id="err",
        **kwargs,
    )


class _CoreProbe(_Probe):
    """A real ``LLMInterface`` (the doubles are) whose threads the row reads."""


class TestLLMStepIsNative:
    async def test_a_real_interface_is_awaited_on_the_loop(self, no_executor) -> None:
        llm = _CoreProbe(default_response="answer")
        result = await _llm_step(llm).execute({"word": "hi"})
        assert (result.success, result.data["out"]) == (True, "answer")
        assert llm.threads == [threading.get_ident()]
        assert llm.awaits == 1

    async def test_a_time_limit_cancels_the_await(self, timing_factor) -> None:
        llm = _CoreProbe(park_at=1)
        step = _llm_step(llm, timeout_seconds=0.2 * timing_factor)
        result = await asyncio.wait_for(step.execute({"word": "hi"}), _BOUND)
        assert (result.success, result.next_step_id) == (False, "err")
        assert llm.cancelled == 1

    async def test_a_sync_only_interface_goes_through_the_default_twin(self) -> None:
        """An ``LLMInterface`` that defines only the sync method answers through the base class's
        ``agenerate_response`` (a worker thread), with the same result."""
        from fsm_llm.core.definitions import ResponseGenerationResponse

        seen: list[int] = []

        class SyncOnly(LLMInterface):
            def generate_response(self, request):
                seen.append(threading.get_ident())
                return ResponseGenerationResponse(message="sync answer")

        result = await _llm_step(SyncOnly()).execute({"word": "hi"})
        assert result.data["out"] == "sync answer"
        assert seen and seen[0] != threading.get_ident()

    async def test_a_duck_typed_generate_keeps_its_route(self) -> None:
        class Duck:
            def generate(self, prompt):
                return f"duck:{prompt}"

        result = await _llm_step(Duck()).execute({"word": "hi"})
        assert result.data["out"] == "duck:Say hi"

    async def test_two_llm_steps_of_a_parallel_step_overlap(self) -> None:
        barrier = asyncio.Barrier(2)
        llms = [_CoreProbe(barrier=barrier), _CoreProbe(barrier=barrier)]
        steps = [
            LLMStep(
                step_id=f"l{i}",
                name="L",
                llm_interface=llm,
                prompt_template="Say hi",
                output_mapping={f"o{i}": ""},
                next_step_id="next",
            )
            for i, llm in enumerate(llms)
        ]
        result = await asyncio.wait_for(
            ParallelStep(step_id="p", name="P", steps=steps, next_step_id="n").execute(
                {}
            ),
            _BOUND,
        )
        assert result.success is True
        assert all(llm.awaits == 1 for llm in llms)  # both awaited the twin
