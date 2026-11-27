"""Plan 1f0925d2 step 3 (S2): the builder start reserves its slot before the model call,
holds it until the worker thread really ends, and releases it on every exit path.

Interface contract of the doubles: ``_Rig.agent_class()`` is a stand-in for
``MetaBuilderAgent`` whose ``start`` and ``send`` block on ``rig.gate`` (a
``threading.Event`` waited with a hard bound) and count how many workers entered
(``rig.entered``). The message decides the rest: ``"boom"`` fails at once,
``"late-boom"`` fails after the gate opens, anything else answers ``"hello"``. No
test here decides an outcome by a sleep: an Event, a counter and the state of the
slot table decide, and the one poll helper (``_until``) only bounds a wait.
"""

from __future__ import annotations

import asyncio
import gc
import sys
import threading
import time

import httpx
import pytest
from fastapi import HTTPException

import fsm_llm.agents.meta_builder as meta_builder
from fsm_llm.core.definitions import ModelNotConfiguredError
from fsm_llm.monitor.definitions import BuilderSendRequest, BuilderStartRequest
from fsm_llm.monitor.server import app, configure, routes_builder
from tests.test_fsm_llm_monitor.support import _manager

_BOUND_SECONDS = 20.0  # a hard bound on every wait, never the thing that decides


class _Rig:
    """The gate, the counter and the agent class of one test."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self._count = threading.Lock()
        self.entered = 0

    def agent_class(self) -> type:
        rig = self

        class _Agent:
            def __init__(self, config=None):
                self.config = config

            def _work(self, message: str) -> str:
                if message == "boom":
                    raise RuntimeError("worker failed at once")
                if message == "fast":
                    return "hello"  # answers at once, never waits for the gate
                with rig._count:
                    rig.entered += 1
                if not rig.gate.wait(_BOUND_SECONDS):
                    raise RuntimeError("the gate was never opened")
                if message == "late-boom":
                    raise RuntimeError("worker failed after its request gave up")
                return "hello"

            def start(self, message):
                return self._work(message)

            def send(self, message):
                return self._work(message)

            def is_complete(self):
                return False

            def get_internal_state(self):
                return {"phase": "collect", "turn_count": 0}

        return _Agent


def _pending() -> dict:
    """The table of reserved starts (empty on a tree that has none)."""
    return dict(getattr(routes_builder, "_builder_starting", {}))


def _reset() -> None:
    routes_builder._builder_sessions.clear()
    routes_builder._builder_busy.clear()
    getattr(routes_builder, "_builder_starting", {}).clear()
    routes_builder._builder_sessions_lock = None


@pytest.fixture
def rig(monkeypatch):
    rig = _Rig()
    monkeypatch.setattr(meta_builder, "MetaBuilderAgent", rig.agent_class())
    configure(manager=_manager())  # no key: the gate is off
    _reset()
    yield rig
    rig.gate.set()
    _reset()


def _run(rig: _Rig, main):
    """Run ``main()`` on a fresh loop; the gate is always open before the loop closes
    (the default executor joins its threads at shutdown)."""

    async def go():
        try:
            return await main()
        finally:
            rig.gate.set()

    return asyncio.run(go())


async def _until(predicate, what: str) -> None:
    deadline = time.monotonic() + _BOUND_SECONDS
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"never reached: {what}")
        await asyncio.sleep(0.001)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        timeout=_BOUND_SECONDS,
    )


async def _start(client: httpx.AsyncClient, artifact_type: str = ""):
    return await client.post(
        "/api/builder/start", json={"artifact_type": artifact_type}
    )


async def _status_of(call) -> int:
    """The status of the ``HTTPException`` the awaited handler raises. The exception
    is dropped on return: its traceback would keep the handler's frame (and the
    worker future in it) alive past the point a ``never retrieved`` log is due."""
    try:
        await call
    except HTTPException as exc:
        return exc.status_code
    raise AssertionError("the handler did not raise")


def _detail(response: httpx.Response) -> str:
    return response.json()["detail"]


class TestAStartReservesItsSlotBeforeTheModelCall:
    def test_ten_concurrent_starts_at_a_cap_of_three_admit_three(
        self, rig, monkeypatch
    ):
        monkeypatch.setattr(routes_builder, "MAX_BUILDER_SESSIONS", 3)

        async def main():
            async with _client() as client:
                tasks = [asyncio.ensure_future(_start(client)) for _ in range(10)]
                # every request is either inside start() or already answered
                await _until(
                    lambda: rig.entered + sum(t.done() for t in tasks) >= 10,
                    "ten requests placed",
                )
                answered = [t.result() for t in tasks if t.done()]
                assert rig.entered == 3
                assert [r.status_code for r in answered] == [429] * 7
                assert {_detail(r) for r in answered} == {
                    "too many builder sessions (max 3)"
                }
                assert len(_pending()) == 3
                assert not any(_pending().values())  # reserved, none detached
                rig.gate.set()
                rest = [t for t in tasks if not t.done()]
                codes = [r.status_code for r in await asyncio.gather(*rest)]
                assert codes == [200] * 3
            assert len(routes_builder._builder_sessions) == 3
            assert _pending() == {}

        _run(rig, main)

    def test_a_normal_start_adds_the_session_with_the_same_body(self, rig):
        rig.gate.set()

        async def main():
            async with _client() as client:
                response = await _start(client)
            assert response.status_code == 200
            body = response.json()
            assert set(body) == {
                "session_id",
                "response",
                "is_complete",
                "internal_state",
            }
            assert body["response"] == "hello"
            assert body["session_id"] in routes_builder._builder_sessions
            assert _pending() == {}

        _run(rig, main)

    def test_the_429_comes_before_the_501_and_the_400(self, rig, monkeypatch):
        monkeypatch.setattr(routes_builder, "MAX_BUILDER_SESSIONS", 0)
        monkeypatch.setitem(sys.modules, "fsm_llm.agents.meta_builder", None)

        async def main():
            with pytest.raises(HTTPException) as refused:
                await routes_builder.api_builder_start(BuilderStartRequest())
            assert refused.value.status_code == 429
            assert _pending() == {}

        _run(rig, main)


class TestATimedOutOrCancelledStartKeepsItsSlotUntilTheWorkerEnds:
    def test_a_timeout_is_a_504_and_the_slot_counts_against_both_caps(
        self, rig, monkeypatch
    ):
        monkeypatch.setattr(routes_builder, "_LLM_OPERATION_TIMEOUT", 0.05)

        async def main():
            async with _client() as client:
                first = await _start(client)
                assert first.status_code == 504
                assert _detail(first) == "LLM operation timed out"
                assert rig.entered == 1 and not rig.gate.is_set()
                assert list(_pending().values()) == [True]  # held, detached
                with monkeypatch.context() as cap:  # the session cap counts it
                    cap.setattr(routes_builder, "MAX_BUILDER_SESSIONS", 1)
                    second = await _start(client)
                assert second.status_code == 429
                assert _detail(second) == "too many builder sessions (max 1)"
                with monkeypatch.context() as cap:  # the abandoned bound counts it
                    cap.setattr(routes_builder, "MAX_ABANDONED_BUILDERS", 1)
                    third = await _start(client)
                assert third.status_code == 429
                assert _detail(third) == (
                    "1 cancelled builds are still running (max 1); "
                    "try again when one ends"
                )
                assert rig.entered == 1  # neither refused start reached a worker
                rig.gate.set()  # the worker ends: the slot is released
                await _until(lambda: _pending() == {}, "the slot released")
                monkeypatch.setattr(routes_builder, "_LLM_OPERATION_TIMEOUT", 30.0)
                fourth = await _start(client)
                assert fourth.status_code == 200
            assert len(routes_builder._builder_sessions) == 1

        _run(rig, main)

    def test_a_cancelled_request_leaves_the_slot_until_the_worker_ends(self, rig):
        async def main():
            task = asyncio.ensure_future(
                routes_builder.api_builder_start(BuilderStartRequest())
            )
            await _until(lambda: rig.entered == 1, "the worker entered start")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()  # raised, never absorbed
            assert asyncio.current_task().cancelling() == 0
            # no await ran in the arm: the slot is already marked detached
            assert list(_pending().values()) == [True]
            rig.gate.set()
            await _until(lambda: _pending() == {}, "the slot released")
            assert routes_builder._builder_sessions == {}

        _run(rig, main)


class TestEveryOtherExitReleasesTheSlotAtOnce:
    def test_a_start_that_raises_is_the_unchanged_500_and_leaves_no_slot(self, rig):
        async def main():
            async with _client() as client:
                response = await _start(client, "boom")
            assert response.status_code == 500
            assert _detail(response) == "Internal server error"
            assert _pending() == {}

        _run(rig, main)

    def test_a_failing_start_frees_its_own_slot_and_no_other(self, rig, monkeypatch):
        monkeypatch.setattr(routes_builder, "MAX_BUILDER_SESSIONS", 3)

        async def main():
            async with _client() as client:
                blocked = [asyncio.ensure_future(_start(client)) for _ in range(2)]
                await _until(lambda: rig.entered == 2, "two workers inside start")
                failed = await _start(client, "boom")
                assert failed.status_code == 500
                assert len(_pending()) == 2  # the other two slots are still held
                third = asyncio.ensure_future(_start(client))
                await _until(lambda: rig.entered == 3, "a third worker inside start")
                refused = await _start(client)
                assert refused.status_code == 429  # three held, the cap is three
                rig.gate.set()
                codes = [r.status_code for r in await asyncio.gather(*blocked, third)]
                assert codes == [200] * 3
            assert _pending() == {}

        _run(rig, main)

    def test_a_finished_start_frees_its_own_slot_and_no_other(self, rig):
        async def main():
            async with _client() as client:
                blocked = asyncio.ensure_future(_start(client))
                await _until(lambda: rig.entered == 1, "a worker inside start")
                done = await _start(client, "fast")
                assert done.status_code == 200
                assert len(routes_builder._builder_sessions) == 1
                assert len(_pending()) == 1  # the blocked start still holds its slot
                rig.gate.set()
                assert (await blocked).status_code == 200
            assert len(routes_builder._builder_sessions) == 2
            assert _pending() == {}

        _run(rig, main)

    def test_a_missing_meta_builder_is_a_501_and_leaves_no_slot(self, rig, monkeypatch):
        monkeypatch.setitem(sys.modules, "fsm_llm.agents.meta_builder", None)

        async def main():
            with pytest.raises(HTTPException) as refused:
                await routes_builder.api_builder_start(BuilderStartRequest())
            assert refused.value.status_code == 501
            assert _pending() == {}

        _run(rig, main)

    def test_no_model_is_a_400_and_leaves_no_slot(self, rig, monkeypatch):
        def no_model(_name):
            raise ModelNotConfiguredError("no model is set")

        monkeypatch.setattr(routes_builder, "resolve_model", no_model)

        async def main():
            with pytest.raises(HTTPException) as refused:
                await routes_builder.api_builder_start(BuilderStartRequest())
            assert refused.value.status_code == 400
            assert refused.value.detail == "no model is set"
            assert _pending() == {}

        _run(rig, main)

    def test_a_construction_failure_leaves_no_slot(self, rig, monkeypatch):
        class Broken:
            def __init__(self, config=None):
                raise RuntimeError("cannot build the agent")

        monkeypatch.setattr(meta_builder, "MetaBuilderAgent", Broken)

        async def main():
            with pytest.raises(RuntimeError, match="cannot build"):
                await routes_builder.api_builder_start(BuilderStartRequest())
            assert _pending() == {}

        _run(rig, main)


class TestAFailureAfterTheRequestGaveUpIsRetrieved:
    """The loop's exception handler hears of no ``never retrieved`` future."""

    @staticmethod
    async def _after_the_worker_failed(rig, ended) -> list[str]:
        records: list[str] = []
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: records.append(str(context.get("message")))
        )
        rig.gate.set()
        await _until(ended, "the worker ended and the slot or session released")
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0)
        return records

    def test_a_start_that_fails_after_its_504(self, rig, monkeypatch):
        monkeypatch.setattr(routes_builder, "_LLM_OPERATION_TIMEOUT", 0.05)

        async def main():
            status = await _status_of(
                routes_builder.api_builder_start(
                    BuilderStartRequest(artifact_type="late-boom")
                )
            )
            assert status == 504
            records = await self._after_the_worker_failed(rig, lambda: _pending() == {})
            assert [r for r in records if "never retrieved" in r] == []
            assert routes_builder._builder_sessions == {}

        _run(rig, main)

    def test_a_send_that_fails_after_its_504(self, rig, monkeypatch):
        monkeypatch.setattr(routes_builder, "_BUILDER_OPERATION_TIMEOUT", 0.05)
        routes_builder._builder_sessions["s1"] = (rig.agent_class()(), 0.0)

        async def main():
            status = await _status_of(
                routes_builder.api_builder_send(
                    BuilderSendRequest(session_id="s1", message="late-boom")
                )
            )
            assert status == 504
            records = await self._after_the_worker_failed(
                rig, lambda: "s1" not in routes_builder._builder_busy
            )
            assert [r for r in records if "never retrieved" in r] == []

        _run(rig, main)


class TestTheLockBelongsToTheRunningLoop:
    @staticmethod
    async def _contended() -> object:
        """Hold the builder lock while a second task has to wait for it."""
        lock = routes_builder._get_builder_lock()
        assert routes_builder._get_builder_lock() is lock  # one lock per loop

        async def second():
            async with routes_builder._get_builder_lock():
                return True

        async with lock:
            waiter = asyncio.ensure_future(second())
            await asyncio.sleep(0)  # the waiter reaches its wait and binds the lock
        assert await waiter is True
        return lock

    def test_a_second_loop_after_a_contended_first_loop(self, rig):
        first = asyncio.run(self._contended())
        second = asyncio.run(self._contended())  # BASE: RuntimeError, another loop
        assert first is not second

    def test_the_getter_outside_a_loop_still_keeps_one_lock(self, rig):
        first = routes_builder._get_builder_lock()
        assert routes_builder._get_builder_lock() is first


_PATHS = (
    "success",
    "worker_error",
    "no_model",
    "no_meta_builder",
    "construction",
    "timeout",
    "cancel",
    "late_failure",
)


class TestNoPathLeaksASlot:
    """After each way a start can end, ``sessions + reserved == expected`` and a start at
    the cap is refused or admitted exactly as that arithmetic says."""

    @staticmethod
    async def _check(client, expected: int, *, refused: bool) -> None:
        assert len(routes_builder._builder_sessions) + len(_pending()) == expected
        response = await _start(client)
        assert response.status_code == (429 if refused else 200), response.text
        routes_builder._builder_sessions.clear()  # forget the probe's own session

    @staticmethod
    def _inject(fault, path: str) -> None:
        """Make the start fail the way ``path`` names (undone when ``fault`` exits)."""

        def no_model(_name):
            raise ModelNotConfiguredError("no model is set")

        class Broken:
            def __init__(self, config=None):
                raise RuntimeError("cannot build the agent")

        if path == "no_model":
            fault.setattr(routes_builder, "resolve_model", no_model)
        elif path == "no_meta_builder":
            fault.setitem(sys.modules, "fsm_llm.agents.meta_builder", None)
        elif path == "construction":
            fault.setattr(meta_builder, "MetaBuilderAgent", Broken)
        elif path in ("timeout", "cancel", "late_failure"):
            fault.setattr(routes_builder, "_LLM_OPERATION_TIMEOUT", 0.05)

    @pytest.mark.parametrize("path", _PATHS)
    def test_the_count_and_the_next_start_follow_the_arithmetic(
        self, rig, monkeypatch, path
    ):
        monkeypatch.setattr(routes_builder, "MAX_BUILDER_SESSIONS", 1)
        held = path in ("timeout", "cancel", "late_failure")
        if not held:
            rig.gate.set()
        kind = {"late_failure": "late-boom", "worker_error": "boom"}.get(path, "")

        async def attempt(client):
            if path == "cancel":
                task = asyncio.ensure_future(
                    routes_builder.api_builder_start(BuilderStartRequest())
                )
                await _until(lambda: rig.entered == 1, "the worker entered")
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif path in ("no_model", "no_meta_builder", "construction"):
                with pytest.raises(Exception):
                    await routes_builder.api_builder_start(BuilderStartRequest())
            else:
                await _start(client, kind)

        async def main():
            async with _client() as client:
                with monkeypatch.context() as fault:
                    self._inject(fault, path)
                    await attempt(client)
                if held:
                    # the thread still runs: its slot is held, the cap says refuse
                    await self._check(client, 1, refused=True)
                    rig.gate.set()
                    await _until(lambda: _pending() == {}, "the worker ended")
                if path == "success":
                    await self._check(client, 1, refused=True)
                else:
                    await self._check(client, 0, refused=False)
            assert _pending() == {}

        _run(rig, main)
