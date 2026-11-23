"""The streaming solve (plan baa37f40, step 19, D-036, roadmap E3.4).

``ReasoningEngine.solve_problem_stream(problem, initial_context=None, *, max_steps=None,
max_seconds=None) -> SolveStream``: an iterator of the reply text of one solve, strategy replies
included, as they arrive; ``.result()`` is the ``(solution, trace_info)`` tuple of ``solve_problem``
after exhaustion and raises before. The lock of the engine is taken at the first ``next()`` and
held until the stream ends, fails, is closed or is dropped; the orchestrator conversation is ended
on every one of those ways out. Every row is scripted: no network, no thread, no sleep.
"""

from __future__ import annotations

import asyncio
import collections.abc
import gc
import threading
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm.core.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.reasoning import (
    ReasoningEngine,
    ReasoningExecutionError,
    SolveStream,
)
from fsm_llm.reasoning import engine as engine_module
from fsm_llm.reasoning.handlers import ReasoningHandlers

from .test_engine_scripted import (
    _NEVER_ENDING_SCRIPT,
    _PROBLEM,
    _VALID_SCRIPT,
    _engine,
    _ScriptedLLM,
)

REPO = Path(__file__).resolve().parents[2]


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")
    assert Path(engine_module.__file__).resolve().is_relative_to(REPO / "src")


def _depth(engine: ReasoningEngine) -> int:
    """The depth of the stack of the one open orchestrator conversation."""
    (conversation_id,) = engine.orchestrator.list_active_conversations()
    return engine.orchestrator.get_stack_depth(conversation_id)


def _nothing_open(engine: ReasoningEngine) -> None:
    """The lock is free and no conversation, frame or instance of the orchestrator is left."""
    assert not engine._solve_lock.locked()
    assert engine.orchestrator.list_active_conversations() == []
    assert dict(engine.orchestrator.fsm_manager.instances) == {}


class _HandlerCrash(Exception):
    """What a critical handler of the solve raises."""


def _crash_the_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the critical validator handler raise: a solve that fails in its step."""

    def crash(context: dict[str, Any]) -> dict[str, Any]:
        raise _HandlerCrash("validate_solution crashed")

    monkeypatch.setattr(ReasoningHandlers, "validate_solution", staticmethod(crash))


class _FailingAtField(_ScriptedLLM):
    """Raises ``error`` when the extraction of ``field`` is asked for (inside the strategy frame
    of the valid script, ``conclusion`` is the deductive strategy's)."""

    def __init__(self, script: dict[str, Any], field: str, error: BaseException):
        super().__init__(script)
        self.field = field
        self.error = error

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        if request.field_name == self.field:
            raise self.error
        return super().extract_field(request)


class _CancelledInPass2(_ScriptedLLM):
    """Raises ``CancelledError`` from the Pass 2 call of the first step run in a pushed frame
    and records the state the conversation was in at that moment (mid-turn, after the
    transition)."""

    def __init__(self, script: dict[str, Any]) -> None:
        super().__init__(script)
        self.api: Any = None
        self.state_at_cancel: str | None = None
        self.armed = False

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        if self.armed and self.state_at_cancel is None:
            (conversation_id,) = self.api.list_active_conversations()
            if self.api.get_stack_depth(conversation_id) == 2:
                self.state_at_cancel = self.api.get_current_state(conversation_id)
                raise asyncio.CancelledError
        return super().generate_response(request)


class TestTheStreamIsLazyAndChecksAtCallTime:
    def test_nothing_runs_before_the_first_next(self) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        engine = _engine(llm)

        stream = engine.solve_problem_stream(_PROBLEM)

        assert isinstance(stream, SolveStream)
        assert llm.requests == [] and llm.completions == []
        _nothing_open(engine)
        stream.close()
        _nothing_open(engine)

    def test_the_first_next_takes_the_lock_and_starts_the_conversation(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        stream = engine.solve_problem_stream(_PROBLEM)

        next(stream)

        assert engine._solve_lock.locked()
        assert len(engine.orchestrator.list_active_conversations()) == 1
        stream.close()
        _nothing_open(engine)

    @pytest.mark.parametrize(
        "limits", [{"max_steps": 0}, {"max_steps": True}, {"max_seconds": -1.0}]
    )
    def test_a_bad_limit_is_refused_at_call_time(self, limits: dict[str, Any]) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        engine = _engine(llm)

        with pytest.raises(ValueError):
            engine.solve_problem_stream(_PROBLEM, **limits)

        assert llm.requests == [] and llm.completions == []
        _nothing_open(engine)

    def test_it_is_an_iterator_and_a_context_manager(self) -> None:
        stream = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem_stream(_PROBLEM)

        assert isinstance(stream, collections.abc.Iterator)
        assert iter(stream) is stream
        with stream as entered:
            assert entered is stream
        assert fsm_llm.reasoning.SolveStream is SolveStream
        assert "SolveStream" in fsm_llm.reasoning.__all__


class TestStrategyRepliesArriveBeforeTheResult:
    def test_a_reply_of_the_pushed_strategy_is_yielded_while_its_frame_is_live(
        self,
    ) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        engine = _engine(llm)
        stream = engine.solve_problem_stream(_PROBLEM)
        seen: list[tuple[str, int, int]] = []

        for chunk in stream:
            seen.append((chunk, _depth(engine), len(llm.requests)))
            with pytest.raises(ReasoningExecutionError):
                stream.result()

        depths = [depth for _chunk, depth, _requests in seen]
        assert 2 in depths, (
            depths
        )  # a strategy reply, with the strategy frame on the stack
        first_strategy_reply = depths.index(2)
        assert depths[0] == 1  # the greeting of the orchestrator comes first
        assert depths[-1] == 1  # the answer comes back on the orchestrator frame
        # the solve went on after the first strategy reply: model calls were still to come
        assert seen[first_strategy_reply][2] < len(llm.requests)
        assert stream.result()[0] == _VALID_SCRIPT["final_solution"]

    def test_the_text_of_the_stream_is_the_text_of_all_responses(self) -> None:
        stream = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem_stream(_PROBLEM)
        chunks = list(stream)
        _solution, trace_info = stream.result()

        assert chunks and all(isinstance(chunk, str) for chunk in chunks)
        assert chunks[0] == trace_info["all_responses"][0]  # the start greeting
        # one entry per step that spoke: the chunks of a step, concatenated
        assert "".join(chunks) == "".join(trace_info["all_responses"])
        assert len(trace_info["all_responses"]) <= len(chunks)

    def test_no_thread_is_started(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        before = set(threading.enumerate())

        stream = engine.solve_problem_stream(_PROBLEM)
        for _chunk in stream:
            assert set(threading.enumerate()) == before
        stream.result()

        assert set(threading.enumerate()) == before


class TestTheResultOnlyAfterExhaustion:
    def test_it_raises_before_the_stream_started_and_while_it_runs(self) -> None:
        stream = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem_stream(_PROBLEM)

        with pytest.raises(ReasoningExecutionError, match="not finished"):
            stream.result()
        next(stream)
        with pytest.raises(ReasoningExecutionError, match="not finished"):
            stream.result()
        stream.close()

    def test_it_returns_the_tuple_after_exhaustion_every_time(self) -> None:
        stream = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem_stream(_PROBLEM)
        list(stream)

        solution, trace_info = stream.result()

        assert solution == _VALID_SCRIPT["final_solution"]
        assert set(trace_info) == {
            "reasoning_trace",
            "summary",
            "final_context",
            "all_responses",
        }
        assert stream.result() is stream.result()
        assert list(stream) == []  # exhausted stays exhausted
        with pytest.raises(StopIteration):
            next(stream)

    def test_it_raises_after_a_close_before_the_end(self) -> None:
        stream = _engine(_ScriptedLLM(_VALID_SCRIPT)).solve_problem_stream(_PROBLEM)
        next(stream)
        stream.close()

        with pytest.raises(ReasoningExecutionError, match="closed"):
            stream.result()
        with pytest.raises(StopIteration):
            next(stream)

    def test_it_raises_chained_from_the_error_after_a_failed_solve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _crash_the_validator(monkeypatch)
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        stream = engine.solve_problem_stream(_PROBLEM)

        with pytest.raises(ReasoningExecutionError, match="crashed") as failed:
            list(stream)
        with pytest.raises(ReasoningExecutionError, match="failed") as result:
            stream.result()

        assert result.value.__cause__ is failed.value
        assert failed.value.details["partial_context"] is not None
        _nothing_open(engine)


class TestTheTupleWrapperIsTheSameBody:
    @staticmethod
    def _solve_both() -> tuple[Any, Any, _ScriptedLLM, _ScriptedLLM]:
        plain_llm, stream_llm = _ScriptedLLM(_VALID_SCRIPT), _ScriptedLLM(_VALID_SCRIPT)
        plain = _engine(plain_llm).solve_problem(_PROBLEM)
        stream = _engine(stream_llm).solve_problem_stream(_PROBLEM)
        list(stream)
        return plain, stream.result(), plain_llm, stream_llm

    def test_the_tuple_equals_the_result_of_the_stream(self) -> None:
        (solution, info), (stream_solution, stream_info), _a, _b = self._solve_both()

        assert solution == stream_solution
        assert info["all_responses"] == stream_info["all_responses"]
        assert info["final_context"] == stream_info["final_context"]
        trace, stream_trace = info["reasoning_trace"], stream_info["reasoning_trace"]
        assert trace["steps"] == stream_trace["steps"]
        assert trace["reasoning_types_used"] == stream_trace["reasoning_types_used"]
        assert trace["final_confidence"] == stream_trace["final_confidence"]

    def test_both_make_the_same_model_requests_in_the_same_order(self) -> None:
        _plain, _streamed, plain_llm, stream_llm = self._solve_both()

        def shape(llm: _ScriptedLLM) -> list[Any]:
            return [
                (type(r).__name__, getattr(r, "field_name", None)) for r in llm.requests
            ] + [r.call_type for r in llm.completions]

        assert shape(plain_llm) == shape(stream_llm)
        assert len(plain_llm.requests) > 10

    def test_each_entry_drives_the_run_its_own_way(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``solve_problem`` keeps the message-free ``run_until_terminal`` (the request bytes of
        the non-stream solve are the parent's), the stream drives ``run_until_terminal_stream``."""
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        api = engine.orchestrator
        called: list[str] = []
        for name in ("run_until_terminal", "run_until_terminal_stream"):
            original = getattr(api, name)

            def spy(*args: Any, _name=name, _original=original, **kwargs: Any) -> Any:
                called.append(_name)
                return _original(*args, **kwargs)

            monkeypatch.setattr(api, name, spy)

        engine.solve_problem(_PROBLEM)
        assert called == ["run_until_terminal"]
        called.clear()
        list(engine.solve_problem_stream(_PROBLEM))
        assert called == ["run_until_terminal_stream"]


class TestTheLockAndTheConversationAreReleasedEveryWay:
    """Census D7: the lock and the conversation are released in a ``finally``."""

    def test_the_lock_is_held_across_the_yields(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        stream = engine.solve_problem_stream(_PROBLEM)

        for _chunk in stream:
            assert engine._solve_lock.locked()
            assert not engine._solve_lock.acquire(blocking=False)

        _nothing_open(engine)

    def test_a_close_mid_stream_releases_and_a_following_solve_works(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        stream = engine.solve_problem_stream(_PROBLEM)
        next(stream)
        next(stream)

        stream.close()

        _nothing_open(engine)
        solution, _info = engine.solve_problem(_PROBLEM)
        assert solution == _VALID_SCRIPT["final_solution"]

    def test_the_with_form_closes_on_exit(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))

        with engine.solve_problem_stream(_PROBLEM) as stream:
            next(stream)
            assert engine._solve_lock.locked()

        _nothing_open(engine)

    def test_a_dropped_stream_releases_the_lock_and_ends_the_conversation(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        stream = engine.solve_problem_stream(_PROBLEM)
        next(stream)
        assert engine._solve_lock.locked()

        del stream
        gc.collect()

        _nothing_open(engine)
        assert engine.solve_problem(_PROBLEM)[0] == _VALID_SCRIPT["final_solution"]

    def test_a_close_inside_the_strategy_frame_ends_both_frames(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        stream = engine.solve_problem_stream(_PROBLEM)
        for _chunk in stream:
            if _depth(engine) == 2:
                break
        assert len(engine.orchestrator.fsm_manager.instances) == 2  # root and strategy

        stream.close()

        _nothing_open(engine)

    def test_a_failed_step_ends_the_solve_and_releases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _crash_the_validator(monkeypatch)
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))

        with pytest.raises(ReasoningExecutionError, match="crashed"):
            list(engine.solve_problem_stream(_PROBLEM))

        _nothing_open(engine)
        assert engine.classifier.list_active_conversations() == []

    def test_a_spent_step_budget_is_the_error_of_the_tuple_solve(self) -> None:
        engine = _engine(_ScriptedLLM(_NEVER_ENDING_SCRIPT))

        with pytest.raises(ReasoningExecutionError, match="steps") as info:
            list(engine.solve_problem_stream(_PROBLEM, max_steps=3))

        assert isinstance(info.value.__cause__, fsm_llm.RunBudgetExceededError)
        assert info.value.__cause__.steps_done == 3
        assert info.value.details["partial_context"] is not None
        _nothing_open(engine)

    @pytest.mark.parametrize(
        "signal", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
    )
    def test_a_signal_in_the_strategy_frame_ends_both_and_propagates_unwrapped(
        self, signal: type[BaseException]
    ) -> None:
        llm = _FailingAtField(_VALID_SCRIPT, "conclusion", signal())
        engine = _engine(llm)
        stream = engine.solve_problem_stream(_PROBLEM)

        with pytest.raises(signal):
            list(stream)

        _nothing_open(engine)
        with pytest.raises(ReasoningExecutionError, match="failed"):
            stream.result()

    def test_a_cancel_mid_stream_is_rolled_back_whole_and_ends_the_solve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = _CancelledInPass2(_VALID_SCRIPT)
        engine = _engine(llm)
        llm.api = engine.orchestrator
        state_at_end: list[str] = []
        original = engine.orchestrator.end_conversation

        def spy(conversation_id: str) -> None:
            state_at_end.append(engine.orchestrator.get_current_state(conversation_id))
            original(conversation_id)

        monkeypatch.setattr(engine.orchestrator, "end_conversation", spy)
        stream = engine.solve_problem_stream(_PROBLEM)
        next(stream)  # the greeting: the conversation exists, the next steps run
        llm.armed = True

        with pytest.raises(asyncio.CancelledError):
            list(stream)

        assert llm.state_at_cancel is not None and len(state_at_end) == 1
        # the turn that was cancelled had already moved the frame; core restored it whole
        assert state_at_end[0] != llm.state_at_cancel
        _nothing_open(engine)

    def test_an_unstarted_stream_leaks_nothing(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        abandoned = engine.solve_problem_stream(_PROBLEM)

        assert engine.solve_problem(_PROBLEM)[0] == _VALID_SCRIPT["final_solution"]
        del abandoned
        gc.collect()
        _nothing_open(engine)


class TestASecondSolveOnTheOwnerThreadIsRefusedNotBlocked:
    """D7, D-046 (review iter1, concern 7): the lock is held across the yields of a stream, so a
    consumer that solves again inside its own loop used to wait for itself forever. Every row
    asserts in the SAME thread (no join on a blocked thread); a hang is cut by the row's timeout."""

    @pytest.mark.timeout(60)
    def test_a_solve_inside_an_open_stream_raises_and_the_stream_goes_on(self) -> None:
        """Kills: the owner check removed (the second solve blocks on its own thread's lock)."""
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        stream = engine.solve_problem_stream(_PROBLEM)
        next(stream)

        with pytest.raises(ReasoningExecutionError, match="still open on this thread"):
            engine.solve_problem("another problem")

        assert engine._solve_lock.locked()  # the refusal left the open stream alone
        list(stream)
        assert stream.result()[0] == _VALID_SCRIPT["final_solution"]
        _nothing_open(engine)
        assert engine.solve_problem(_PROBLEM)[0] == _VALID_SCRIPT["final_solution"]

    @pytest.mark.timeout(60)
    def test_a_second_stream_raises_at_its_first_next(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        first = engine.solve_problem_stream(_PROBLEM)
        next(first)
        second = engine.solve_problem_stream(
            "another problem"
        )  # lazy: nothing taken yet

        with pytest.raises(ReasoningExecutionError, match="still open on this thread"):
            next(second)

        first.close()
        _nothing_open(engine)

    @pytest.mark.timeout(60)
    def test_a_closed_stream_frees_the_thread(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        stream = engine.solve_problem_stream(_PROBLEM)
        next(stream)

        stream.close()

        assert engine.solve_problem(_PROBLEM)[0] == _VALID_SCRIPT["final_solution"]


@pytest.mark.timeout(60)
class TestTheSolveLock:
    """The lock of one engine: the surface the engine and the rows use, plus the owner."""

    def _lock(self) -> Any:
        from fsm_llm.reasoning.stream import SolveLock

        return SolveLock()

    def test_the_owner_thread_is_refused_and_a_non_blocking_try_answers_false(
        self,
    ) -> None:
        lock = self._lock()
        with lock:
            assert lock.locked()
            with pytest.raises(ReasoningExecutionError, match="this thread"):
                lock.acquire()
            with pytest.raises(ReasoningExecutionError, match="this thread"):
                with lock:
                    pass  # never reached
            assert (
                lock.acquire(blocking=False) is False
            )  # it never blocks, never raises
        assert not lock.locked()
        assert lock.acquire(blocking=False) is True  # the refusals left it usable
        lock.release()

    def test_another_thread_waits_its_turn_and_is_never_the_owner(self) -> None:
        lock = self._lock()
        trying = threading.Event()
        got: list[object] = []

        def other() -> None:
            trying.set()
            with lock:  # held by the main thread: a wait, never the owner's refusal
                got.append("held")

        lock.acquire()
        worker = threading.Thread(target=other)
        worker.start()
        trying.wait()  # the worker is about to take the lock (or already waits for it)
        lock.release()
        worker.join()  # it cannot block forever: the lock is free

        assert got == ["held"]
        assert not lock.locked()
