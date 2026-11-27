"""The builder start reserves its slot before the model
call, holds it until the worker thread really ends, and releases it on every exit path;
a launch checks capacity and registers in ONE ``_lock`` hold, the config setter, the
registry snapshot and the usage fold are single scopes (the classes from
``TestALaunchChecksCapacityAndRegistersInOneHold`` on).

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
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

import fsm_llm.agents.meta_builder as meta_builder
from fsm_llm.core.definitions import ModelNotConfiguredError
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.monitor.definitions import (
    BuilderSendRequest,
    BuilderStartRequest,
    MonitorConfig,
    StubToolConfig,
)
from fsm_llm.monitor.exceptions import MonitorCapacityError
from fsm_llm.monitor.instance_manager import InstanceManager
from fsm_llm.monitor.instance_manager import agent as agent_mod
from fsm_llm.monitor.instance_manager import lifecycle as lifecycle_mod
from fsm_llm.monitor.instance_manager.managed import ManagedAgent, ManagedFSM
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
    """The table of reserved starts."""
    return dict(routes_builder._builder_starting)


def _reset() -> None:
    routes_builder._builder_sessions.clear()
    routes_builder._builder_busy.clear()
    routes_builder._builder_starting.clear()
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
        second = asyncio.run(self._contended())  # a lock per loop
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


# --- one lock scope for launch capacity, the setter, the registry snapshot, the usage fold ---

_STUB = StubToolConfig(name="lookup", description="A stub.", stub_response="R")
_DONE = SimpleNamespace(success=True, answer="done", trace=None, final_context={})


class _SpyLock:
    """A lock that records, per thread, whether it is held and which outermost hold it is in.

    ``hold`` is the number of the outermost hold of the calling thread (``None`` when it
    holds nothing): two reads with the same ``hold`` happened in ONE scope. ``waiting``
    counts threads blocked in ``acquire``. ``on_acquire(spy)`` runs right after each
    outermost acquire, ``after_release()`` (one shot) right after each outermost release.
    """

    def __init__(self, inner, on_acquire=None) -> None:
        self._inner, self._on_acquire = inner, on_acquire
        self._local = threading.local()
        self._count = threading.Lock()
        self.holds, self.waiting, self.after_release = 0, 0, None

    @property
    def depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @property
    def hold(self):
        return getattr(self._local, "hold", None) if self.depth else None

    def acquire(self, blocking=True, timeout=-1):
        with self._count:
            self.waiting += 1
        try:
            ok = self._inner.acquire(blocking, timeout)
        finally:
            with self._count:
                self.waiting -= 1
        if ok:
            if self.depth == 0:
                self.holds += 1  # under the inner lock
                self._local.hold = self.holds
                if self._on_acquire is not None:
                    self._on_acquire(self)
            self._local.depth = self.depth + 1
        return ok

    def release(self):
        self._local.depth = self.depth - 1
        self._inner.release()
        if self.depth == 0 and self.after_release is not None:
            after, self.after_release = self.after_release, None
            after()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def _is_owned(self) -> bool:
        return self.depth > 0


class _Tape(dict):
    """An ``_instances`` that notes the hold of every read that decides capacity and of the insert."""

    def __init__(self, lock: _SpyLock, tape: list) -> None:
        super().__init__()
        self._spy, self.tape = lock, tape

    def __len__(self):
        self.tape.append(("len", self._spy.hold))
        return super().__len__()

    def values(self):
        self.tape.append(("values", self._spy.hold))
        return super().values()

    def __setitem__(self, key, value):
        self.tape.append(("insert", self._spy.hold))
        super().__setitem__(key, value)


def _create_agent(gate: threading.Event | None = None, seen: list | None = None):
    """A ``create_agent`` double: its agent waits on ``gate`` (a hard bound) when given one."""

    def create(pattern, **kwargs):
        if seen is not None:
            seen.append(kwargs.get("registry"))

        class _Agent:
            def run(self, task):
                if gate is not None and not gate.wait(_BOUND_SECONDS):
                    raise RuntimeError("the gate was never opened")
                return _DONE

        return _Agent()

    return create


@pytest.fixture
def launcher(monkeypatch):
    """(gate, seen): ``create_agent`` is a double; the gate is open before the test ends."""
    monkeypatch.setenv("LLM_MODEL", "stub/double")
    gate, seen = threading.Event(), []
    monkeypatch.setattr(agent_mod, "create_agent", _create_agent(gate, seen))
    yield gate, seen
    gate.set()
    configure(manager=InstanceManager())


def _launch(mgr: InstanceManager):
    return mgr.launch_agent(agent_type="react", task="t", tools_config=[_STUB])


def _join_agents(mgr: InstanceManager, gate: threading.Event) -> None:
    gate.set()
    for inst in list(mgr._instances.values()):
        if isinstance(inst, ManagedAgent) and inst.thread is not None:
            inst.thread.join(_BOUND_SECONDS)
            assert not inst.thread.is_alive()


def _two_threads(call) -> list[str]:
    """Run ``call()`` on two threads; each ends ``ok``, ``full`` (capacity) or its repr."""
    outcomes: list[str] = []

    def work():
        try:
            call()
            outcomes.append("ok")
        except MonitorCapacityError:
            outcomes.append("full")
        except BaseException as exc:
            outcomes.append(repr(exc))

    threads = [threading.Thread(target=work, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(_BOUND_SECONDS)
        assert not t.is_alive(), "a launch thread is stuck"
    return sorted(outcomes)


def _after_check(mgr: InstanceManager, hook: str, states: list) -> None:
    """Make the first pass of each thread through ``mgr.<hook>()`` wait for the other thread
    AFTER the check passed, so both pass the check before either registers; the lock state of
    that moment goes to ``states``."""
    real, seen = getattr(mgr, hook), threading.local()
    barrier = threading.Barrier(2, timeout=_BOUND_SECONDS)

    def wrapped(*args, **kwargs):
        out = real(*args, **kwargs)
        if not getattr(seen, "x", False):
            seen.x = True
            states.append(mgr._lock._is_owned())
            barrier.wait()
        return out

    setattr(mgr, hook, wrapped)


class TestALaunchChecksCapacityAndRegistersInOneHold:
    """Two launches that both pass the early check cannot both register."""

    def test_two_agent_launches_at_one_running_agent_admit_one(self, launcher):
        gate, _ = launcher
        mgr = _manager(max_running_agents=1)
        states: list[bool] = []
        _after_check(mgr, "_check_agent_capacity", states)

        outcomes = _two_threads(lambda: _launch(mgr))

        agents = [i for i in mgr._instances.values() if isinstance(i, ManagedAgent)]
        _join_agents(mgr, gate)
        assert states == [False, False]  # the barrier never waited under `_lock`
        assert outcomes == ["full", "ok"]
        assert len(agents) == 1

    def test_two_agent_launches_at_one_instance_admit_one(self, launcher):
        gate, _ = launcher
        mgr = _manager(max_instances=1, max_running_agents=64)
        states: list[bool] = []
        _after_check(mgr, "_make_room", states)

        outcomes = _two_threads(lambda: _launch(mgr))

        count = len(mgr._instances)
        _join_agents(mgr, gate)
        assert states == [False, False]
        assert outcomes == ["full", "ok"]
        assert count == 1

    def test_two_registrations_at_one_instance_admit_one(self):
        """The twin for every launch kind: ``_register_instance`` is the one door."""
        mgr = _manager(max_instances=1)
        states: list[bool] = []
        _after_check(mgr, "_make_room", states)
        ids = iter(("f1", "f2"))

        def register():
            managed = ManagedFSM(instance_id=next(ids), api=object())
            mgr._register_instance(managed, mgr._new_collector(), "FSM")

        outcomes = _two_threads(register)

        assert states == [False, False]
        assert outcomes == ["full", "ok"]
        assert len(mgr._instances) == 1 and len(mgr._collectors) == 1

    def test_a_refused_launch_starts_no_thread_and_registers_nothing(self, launcher):
        gate, seen = launcher
        mgr = _manager(max_running_agents=1)
        _launch(mgr)
        with pytest.raises(MonitorCapacityError):
            _launch(mgr)  # the early check
        _join_agents(mgr, gate)
        assert len(mgr._instances) == 1 and len(seen) == 1

    def test_the_checks_and_the_insert_share_one_hold(self, launcher):
        """The state at each check: the last capacity read before the insert is in the
        insert's own hold (a check in an earlier scope leaves a window)."""
        gate, _ = launcher
        mgr = _manager(max_instances=5, max_running_agents=5)
        spy, tape = _SpyLock(threading.RLock()), []
        mgr._lock = spy
        mgr._instances = _Tape(spy, tape)

        _launch(mgr)

        _join_agents(mgr, gate)
        cut = [kind for kind, _ in tape].index("insert")
        before = tape[:cut]
        holds = {
            kind: [h for k, h in before if k == kind][-1] for kind in ("len", "values")
        }
        assert tape[cut][1] is not None
        assert holds == {"len": tape[cut][1], "values": tape[cut][1]}


class TestNothingBlocksOrNestsUnderTheLock:
    def test_eviction_runs_outside_the_lock(self, launcher):
        """D-005 of plan 5980fbb7: ``_make_room`` evicts (a join, an API close) with no lock."""
        gate, _ = launcher
        mgr = _manager(max_instances=1)
        old = ManagedAgent("old", agent_type="react")
        old.status = "completed"
        mgr._instances["old"] = old
        owned: list[bool] = []
        real = mgr.end_instance

        def end_instance(instance_id):
            owned.append(mgr._lock._is_owned())
            real(instance_id)

        mgr.end_instance = end_instance  # type: ignore[method-assign]

        _launch(mgr)

        _join_agents(mgr, gate)
        assert owned == [False]
        assert "old" not in mgr._instances

    def test_the_catalog_lock_and_the_instance_lock_are_never_held_together(
        self, launcher
    ):
        gate, _ = launcher
        mgr = _manager(max_instances=1)
        bad: list[str] = []
        catalog = _SpyLock(
            threading.Lock(),
            lambda spy: bad.append("lock under catalog") if lock._is_owned() else None,
        )
        lock = _SpyLock(
            threading.RLock(),
            lambda spy: (
                bad.append("catalog under lock") if catalog._is_owned() else None
            ),
        )
        mgr._lock, mgr._catalog_lock = lock, catalog

        managed = _launch(mgr)  # rows, registry, register, thread
        mgr.agent_types()
        mgr.agent_preview("react")
        mgr.attach_executables(ExecutableRegistry())
        mgr.agent_types()  # the rows are rebuilt from the new registry
        mgr.end_instance(managed.instance_id)  # evict, fold, join
        mgr.config = MonitorConfig(log_level="DEBUG")
        mgr.close()

        _join_agents(mgr, gate)
        assert bad == []


class TestTheRegistryOfALaunchIsOneSnapshot:
    """1.3b: the rows and the registry of a preview or a launch come from ONE catalog scope."""

    @staticmethod
    def _swap_after_the_next_catalog_scope(mgr: InstanceManager) -> ExecutableRegistry:
        """``attach_executables`` of an empty registry lands right after the next scope of the
        catalog lock ends (and only then); the empty registry is returned."""
        other = ExecutableRegistry()
        spy = _SpyLock(threading.Lock())
        spy.after_release = lambda: mgr.attach_executables(other)
        mgr._catalog_lock = spy
        return other

    def test_a_preview_survives_a_swap_between_its_rows_and_its_registry(self):
        mgr = _manager()
        before = mgr.agent_preview("react")
        assert before is not None
        self._swap_after_the_next_catalog_scope(mgr)

        assert mgr.agent_preview("react") == before  # no ExecutableNotFoundError

    def test_the_visualize_route_is_not_a_500_after_a_swap(self):
        from fastapi.testclient import TestClient

        mgr = _manager()
        mgr.agent_types()
        configure(manager=mgr)
        self._swap_after_the_next_catalog_scope(mgr)
        try:
            response = TestClient(app, raise_server_exceptions=False).get(
                "/api/agent/visualize", params={"agent_type": "react"}
            )
        finally:
            configure(manager=InstanceManager())
        assert response.status_code == 200, response.text

    def test_a_launch_hands_its_own_registry_to_the_thread(self, launcher):
        gate, seen = launcher
        mgr = _manager()
        mgr.agent_types()
        launch_time = mgr.executables
        other = ExecutableRegistry()
        real = mgr._register_instance

        def register(*args, **kwargs):
            real(*args, **kwargs)
            mgr.attach_executables(
                other
            )  # a swap between the rows and the thread start

        mgr._register_instance = register  # type: ignore[method-assign]

        managed = _launch(mgr)

        _join_agents(mgr, gate)
        assert managed.thread is not None
        assert len(seen) == 1 and seen[0] is launch_time and seen[0] is not other


class TestTheConfigSetterIsOneScope:
    @pytest.fixture
    def mgr(self):
        mgr = InstanceManager(config=MonitorConfig(log_level="ERROR"))
        yield mgr
        mgr.close()

    @staticmethod
    def _logged(mgr: InstanceManager, marker: str) -> int:
        from loguru import logger

        logger.warning(marker)
        return sum(1 for r in mgr.global_collector.get_logs() if r.message == marker)

    def test_the_sink_swap_runs_under_the_lock(self, mgr):
        owned: dict[str, bool] = {}
        for owner, name in (
            (mgr._global_collector, "cleanup"),
            (mgr, "_setup_loguru_sink"),
        ):
            real = getattr(owner, name)

            def wrapped(real=real, name=name):
                owned[name] = mgr._lock._is_owned()
                real()

            setattr(owner, name, wrapped)

        mgr.config = MonitorConfig(log_level="DEBUG")

        assert owned == {"cleanup": True, "_setup_loguru_sink": True}
        assert mgr.config.log_level == "DEBUG"

    def test_two_setters_leave_exactly_one_sink(self, mgr):
        """A pauses between removing the old sink and adding its own; B runs its whole setter
        meanwhile (without the lock B finishes first and its sink is orphaned; with it B waits)."""
        spy = _SpyLock(threading.RLock())
        mgr._lock = spy
        paused, b_done, real = (
            threading.Event(),
            threading.Event(),
            mgr._setup_loguru_sink,
        )
        calls: list[int] = []

        def setup():
            calls.append(1)
            if len(calls) == 1:
                paused.set()
                deadline = time.monotonic() + _BOUND_SECONDS
                while not (
                    b_done.is_set() or spy.waiting >= 1
                ):  # a counter, not a clock
                    assert time.monotonic() < deadline, "B never ran or blocked"
                    time.sleep(0.001)
            real()

        mgr._setup_loguru_sink = setup  # type: ignore[method-assign]

        def a():
            mgr.config = MonitorConfig(log_level="DEBUG")

        def b():
            mgr.config = MonitorConfig(log_level="WARNING")
            b_done.set()

        ta = threading.Thread(target=a, daemon=True)
        ta.start()
        assert paused.wait(_BOUND_SECONDS)
        tb = threading.Thread(target=b, daemon=True)
        tb.start()
        for t in (ta, tb):
            t.join(_BOUND_SECONDS)
            assert not t.is_alive()

        assert self._logged(mgr, "one-sink-marker") == 1

    def test_a_setter_on_a_closed_manager_adds_no_sink(self, mgr):
        mgr.close()

        mgr.config = MonitorConfig(log_level="DEBUG")

        assert mgr.global_collector._log_sink_id is None
        assert self._logged(mgr, "closed-marker") == 0
        assert (
            mgr.config.log_level == "DEBUG"
        )  # the value is stored, nothing else moves

    def test_a_change_that_keeps_the_level_keeps_the_sink(self, mgr):
        before = mgr.global_collector._log_sink_id
        mgr.config = MonitorConfig(log_level="ERROR", max_events=500)
        assert mgr.global_collector._log_sink_id == before


class TestTheUsageOfAnEndedInstanceIsVisibleAtEveryMoment:
    """3.6: ``get_metrics`` counts the instance's usage exactly once while it ends."""

    @staticmethod
    def _two_fsms(mgr: InstanceManager):
        from tests.test_fsm_llm_monitor.test_usage_metrics import (
            _Meter,
            _usage,
            _with_fsm,
        )

        gone = _Meter(_usage(2, 20))
        _with_fsm(mgr, "gone", gone)
        _with_fsm(mgr, "stays", _Meter(_usage(1, 10)))
        return gone

    @staticmethod
    def _calls(mgr: InstanceManager) -> int | None:
        usage = mgr.get_metrics().usage
        return None if usage is None else usage.calls

    def test_a_read_in_the_middle_of_the_move_sees_it_once(self, monkeypatch):
        mgr = _manager()
        self._two_fsms(mgr)
        assert self._calls(mgr) == 3
        seen: list[int | None] = []
        real = lifecycle_mod.api_usage

        def api_usage(api):
            if not seen:  # the reading of the meter is the middle of the move
                seen.append(self._calls(mgr))
            return real(api)

        monkeypatch.setattr(lifecycle_mod, "api_usage", api_usage)

        mgr.end_instance("gone")

        assert seen == [3]  # 3, not 1: the ended instance is listed and folded
        assert self._calls(mgr) == 3

    def test_two_ends_of_one_instance_fold_it_once(self, monkeypatch):
        """The second end runs wholly inside the first one's reading of the meter."""
        mgr = _manager()
        self._two_fsms(mgr)
        real, inner = lifecycle_mod.api_usage, []

        def api_usage(api):
            if not inner:
                inner.append(1)
                mgr.end_instance("gone")  # pops and folds first
            return real(api)

        monkeypatch.setattr(lifecycle_mod, "api_usage", api_usage)

        with pytest.raises(KeyError):
            mgr.end_instance("gone")  # its own pop finds nothing: no second fold

        assert self._calls(mgr) == 3

    def test_an_unknown_id_is_a_key_error_and_folds_nothing(self):
        mgr = _manager()
        self._two_fsms(mgr)
        with pytest.raises(KeyError):
            mgr.end_instance("nobody")
        assert self._calls(mgr) == 3
