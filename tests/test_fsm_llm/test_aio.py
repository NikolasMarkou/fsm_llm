"""Tests for ``fsm_llm.aio``: drive a coroutine from sync code with no event loop.

Every rule has a test that fails when the code is mutated (plan 51fde95b, step 2). The one
version-sensitive rule (a suspended ``asend()`` is ended with ``throw(GeneratorExit)``, never
``close()``) is pinned by ``test_a_stream_that_suspends_is_closed_deterministically``.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import gc
import sys
import threading
import warnings
from contextvars import copy_context
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fsm_llm import aio
from fsm_llm.aio import (
    SYNC,
    SuspendedInSyncModeError,
    acall,
    aiterate,
    async_name,
    iterate_sync,
    run_sync,
)


class Llm:
    """A sync-only interface, like every double in the suite."""

    def __init__(self) -> None:
        self.thread = None

    def generate_response(self, req):
        self.thread = threading.get_ident()
        if req == "boom":
            raise ValueError("provider down")
        if req == "kbd":
            raise KeyboardInterrupt
        if req == "exit":
            raise SystemExit(3)
        return f"reply({req})"


class ALlm(Llm):
    async def agenerate_response(self, req):
        await asyncio.sleep(0)  # a real suspension: legal only in async mode
        return "a" + self.generate_response(req)


async def pipeline(llm, trace, msg):
    try:
        return await acall(llm, "generate_response", msg)
    except Exception:
        trace.append("rollback.Exception")
        raise
    except (KeyboardInterrupt, SystemExit):
        trace.append("rollback.BaseException")
        raise
    finally:
        trace.append("finally")


def reference(llm, trace, msg):
    """The same logic written by hand in sync form."""
    try:
        return llm.generate_response(msg)
    except Exception:
        trace.append("rollback.Exception")
        raise
    except (KeyboardInterrupt, SystemExit):
        trace.append("rollback.BaseException")
        raise
    finally:
        trace.append("finally")


def outcome(fn, msg):
    trace: list[str] = []
    try:
        return "ok", fn(Llm(), trace, msg), trace
    except BaseException as exc:
        return "exc", type(exc), str(exc), getattr(exc, "code", None), trace


class TestRunSync:
    @pytest.mark.parametrize("msg", ["hi", "boom", "kbd", "exit"])
    def test_result_and_exceptions_equal_a_hand_written_sync_twin(self, msg):
        got = outcome(lambda *a: run_sync(pipeline(*a)), msg)

        assert got == outcome(reference, msg)
        assert SYNC.get() is False

    @pytest.mark.parametrize("kind", ["sleep0", "future"])
    def test_a_real_suspension_raises_after_the_coroutine_cleanup(self, kind):
        loop = asyncio.new_event_loop()
        log: list[str] = []

        async def work():
            try:
                await (asyncio.sleep(0) if kind == "sleep0" else loop.create_future())
            except GeneratorExit:
                log.append("GeneratorExit")
                raise
            finally:
                log.append("finally")

        try:
            with pytest.raises(SuspendedInSyncModeError):
                run_sync(work())
        finally:
            loop.close()
        assert log == ["GeneratorExit", "finally"]
        assert SYNC.get() is False
        assert issubclass(SuspendedInSyncModeError, RuntimeError)

    def test_a_cleanup_error_is_not_hidden_by_the_suspension_error(self):
        async def work():
            try:
                await asyncio.sleep(0)
            except GeneratorExit:
                raise ValueError("cleanup failed") from None

        with pytest.raises(ValueError, match="cleanup failed"):
            run_sync(work())

    @pytest.mark.parametrize("scenario", ["done", "suspended", "unstarted"])
    def test_no_never_awaited_warning(self, scenario, monkeypatch):
        async def done():
            return 1

        async def suspended():
            await asyncio.sleep(0)

        def interrupted(_awaitable):
            raise OSError("interrupted before the first step")

        if scenario == "unstarted":
            monkeypatch.setattr(aio, "_step", interrupted)
        work = (suspended if scenario == "suspended" else done)()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            with contextlib.suppress(Exception):
                run_sync(work)
            del work
            gc.collect()
        assert not [w for w in seen if "never awaited" in str(w.message)]

    def test_a_non_coroutine_is_returned_as_is(self):
        sentinel, mock = object(), MagicMock()

        assert run_sync(sentinel) is sentinel  # type: ignore[arg-type]
        assert run_sync(mock.process("m")) is mock.process.return_value
        assert run_sync(AsyncMock(return_value="A")("m")) == "A"

    async def test_called_on_a_running_loop_it_works_and_restores_the_flag(self):
        assert run_sync(pipeline(Llm(), [], "n")) == "reply(n)"
        assert SYNC.get() is False


class TestAcall:
    def test_sync_mode_calls_the_sync_method_even_when_a_twin_exists(self):
        assert run_sync(acall(ALlm(), "generate_response", "x")) == "reply(x)"

    def test_sync_mode_is_intercepted_by_a_patch_or_a_mock_of_the_sync_name(self):
        with patch.object(ALlm, "generate_response", return_value="P") as patched:
            assert run_sync(acall(ALlm(), "generate_response", "x")) == "P"
        assert patched.call_count == 1
        llm = ALlm()
        llm.generate_response = MagicMock(side_effect=ValueError("mocked"))  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="mocked"):
            run_sync(acall(llm, "generate_response", "x"))

    async def test_async_mode_awaits_the_twin(self):
        assert await acall(ALlm(), "generate_response", "x") == "areply(x)"

    async def test_async_mode_without_a_twin_runs_the_sync_method_on_a_worker(self):
        llm = Llm()

        assert await acall(llm, "generate_response", "x") == "reply(x)"
        assert llm.thread != threading.get_ident()

    async def test_a_flag_left_set_in_a_loop_task_takes_the_sync_path(self):
        token = SYNC.set(True)
        try:
            assert await acall(ALlm(), "generate_response", "z") == "reply(z)"
        finally:
            SYNC.reset(token)

    @pytest.mark.parametrize(
        ("name", "twin"),
        [
            ("generate_response", "agenerate_response"),
            ("_send", "_asend"),
            ("_make_llm_call", "_amake_llm_call"),
            ("complete", "acomplete"),
        ],
    )
    def test_async_name(self, name, twin):
        assert async_name(name) == twin


async def counted(n, log):
    try:
        for i in range(n):
            log.append(("step", i, SYNC.get()))
            yield i
    finally:
        log.append("finally")


async def inner(log):
    try:
        yield 1
        yield 2
    finally:
        log.append("inner.finally")


async def outer(log):
    try:
        try:
            async for item in inner(log):
                yield item
        except GeneratorExit:
            log.append("outer.GeneratorExit")
            raise
    finally:
        log.append("outer.finally")


class TestIterateSync:
    def test_it_is_lazy_and_binds_the_flag_only_during_a_step(self):
        log: list = []
        it = iterate_sync(counted(3, log))

        assert log == []
        assert next(it) == 0
        assert SYNC.get() is False
        assert list(it) == [1, 2]
        assert log == [("step", i, True) for i in range(3)] + ["finally"]

    def test_close_runs_the_generator_exit_arm_and_finally_inner_first(self):
        log: list = []
        agen = outer(log)  # held here, so only an explicit aclose() can end it
        it = iterate_sync(agen)

        assert next(it) == 1
        it.close()

        assert log == ["inner.finally", "outer.GeneratorExit", "outer.finally"]
        assert SYNC.get() is False

    def test_a_dropped_iterator_is_closed_on_collection(self):
        log: list = []
        it = iterate_sync(counted(3, log))
        next(it)
        del it
        gc.collect()

        assert log[-1] == "finally"

    @pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
    def test_an_error_mid_stream_propagates_after_the_finally(self, error):
        log: list = []

        async def failing():
            try:
                yield 1
                raise error("mid-stream")
            finally:
                log.append("finally")

        got = []
        with pytest.raises(error):
            for item in iterate_sync(failing()):
                got.append(item)
        assert got == [1]
        assert log == ["finally"]
        assert SYNC.get() is False

    def test_a_stream_that_suspends_is_closed_deterministically(self):
        # With close() instead of throw(GeneratorExit) this leaves the generator "already
        # running" on 3.11 and 3.12, and the aclose() of the finally raises instead.
        log: list = []

        async def suspending():
            try:
                yield 1
                await asyncio.sleep(0)
                yield 2
            finally:
                log.append("finally")

        got = []
        with pytest.raises(SuspendedInSyncModeError):
            for item in iterate_sync(suspending()):
                got.append(item)
        assert got == [1]
        assert log == ["finally"]

    def test_a_consumer_may_hop_contexts_between_steps(self):
        log: list = []
        it = iterate_sync(counted(3, log))
        first, second = copy_context(), copy_context()

        assert first.run(next, it) == 0
        assert second.run(next, it) == 1
        second.run(it.close)

        assert log[-1] == "finally"
        assert SYNC.get() is False

    def test_a_plain_iterator_passes_through(self):
        assert list(iterate_sync(iter([1, 2]))) == [1, 2]

    async def test_dropped_mid_stream_on_a_loop_thread_closes_once_and_the_loop_stays_quiet(
        self,
    ):
        log: list = []
        errors: list = []
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: errors.append(context)
        )
        it = iterate_sync(counted(3, log))
        next(it)
        del it
        gc.collect()
        await asyncio.sleep(0.05)

        assert log.count("finally") == 1
        assert errors == []


@pytest.fixture(params=["sync", "async"])
def drive(request):
    """Run a coroutine in sync mode (``run_sync``) or on a loop."""
    return request.param, (run_sync if request.param == "sync" else asyncio.run)


async def drain(source, stop=None):
    seen = []
    gen = aiterate(source)
    async for item in gen:
        seen.append(item)
        if len(seen) == stop:
            break
    await gen.aclose()
    return seen


def sync_source(log, n=4):
    try:
        for i in range(n):
            log.append(threading.get_ident())
            yield i
    finally:
        log.append("closed")


class TestAiterate:
    def test_a_sync_source_is_read_on_the_right_thread_and_in_order(self, drive):
        mode, run = drive
        log: list = []

        assert run(drain(sync_source(log))) == [0, 1, 2, 3]
        threads = {entry for entry in log if entry != "closed"}
        assert (threads == {threading.get_ident()}) == (mode == "sync")

    def test_stopping_early_closes_a_sync_source(self, drive):
        log: list = []
        source = sync_source(log)  # held here, so only an explicit close() can end it

        assert drive[1](drain(source, stop=2)) == [0, 1]
        assert log[-1] == "closed"

    def test_an_async_source_is_iterated_natively(self, drive):
        async def source():
            yield 1
            yield 2

        assert drive[1](drain(source())) == [1, 2]


class TestModule:
    def test_it_imports_the_standard_library_only(self):
        tree = ast.parse(Path(aio.__file__).read_text(encoding="utf-8"))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add((node.module or "").split(".")[0])

        assert roots <= set(sys.stdlib_module_names)
        assert sorted(aio.__all__) == [
            "SYNC",
            "SuspendedInSyncModeError",
            "acall",
            "aiterate",
            "async_name",
            "iterate_sync",
            "run_sync",
        ]
