"""The thread-per-run host and the end hook (plan baa37f40, step K7.2, D-050, roadmap E3.2).

``RunHost`` runs a body on one DAEMON thread per run (``asyncio.run`` on a fresh context, so
``aio.SYNC`` is unset), keeps the result under a leaf lock and an ``Event``, and never joins:
a cancel sets the per-run token. ``ExecutableTargets`` puts the registry and the host behind the
one ``API(targets=)``. ``API.end_conversation`` (and so ``close`` and the stale sweep) tells the
launcher to ``cancel_for`` the conversation, after its bookkeeping drop.

Rules of the rows (LESSONS ace7a8c8 and D-048): no ``Thread.join``, no sleep as a
synchronization (bounded ``Event.wait`` on the polling side only, the worker holds no lock the
test holds), a body that must park waits on its own token, the shared double is a ``ScriptedLLM``
with a ``responses`` mapping (never an ordered reply list: two threads would race its order), and
the stress row is skipped under a tracer (``timing_factor > 1``). Each guard row names the mutant
it kills.
"""

from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path

import pytest

import fsm_llm
from fsm_llm import API
from fsm_llm.aio import SYNC, aoffload, run_sync
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions import FSMError
from fsm_llm.core.definitions.executable import Capability
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.results import RunResult
from fsm_llm.engine.context import CancelToken, RunContext
from fsm_llm.engine.executables import ExecutableRegistry, runs
from fsm_llm.engine.executables.runs import (
    DEFAULT_MAX_RUNS,
    ExecutableTargets,
    RunHost,
)
from fsm_llm.engine.kinds.targets import (
    DelegatedTarget,
    RunLauncher,
    RunPoll,
    TargetResolver,
)
from fsm_llm.testing import ScriptedLLM
from tests.conftest import current_timing_factor
from tests.test_fsm_llm import kind_corpus as kc
from tests.test_fsm_llm.test_async_engine import _CHAT
from tests.test_fsm_llm.test_delegated_resolution import _Launcher, _Stub

# seconds: a wait that would hang fails instead (scaled under a tracer)
_BOUND = 20 * current_timing_factor()
_ROOT = Path(__file__).resolve().parents[2]


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(_ROOT / "src")


def _ok(answer: str = "done") -> RunResult:
    return RunResult(status=Status.OK, answer=answer)


async def _instant(token: CancelToken) -> RunResult:
    return _ok()


def _parked() -> tuple[object, threading.Event]:
    """A body that runs until its token is set; ``entered`` is set when it started."""
    entered = threading.Event()

    async def body(token: CancelToken) -> RunResult:
        entered.set()
        for _ in range(
            int(_BOUND * 200)
        ):  # at most _BOUND seconds, then it ends by itself
            if token.cancelled:
                return RunResult(status=Status.ABORTED, reason=Reason.CANCELLED)
            await asyncio.sleep(0.005)
        return RunResult(status=Status.FAILED, reason=Reason.UNEXPECTED_ERROR)

    return body, entered


def _wait_gone(host: RunHost, handle: str, conversation: str) -> bool:
    """Bounded poll until the host holds nothing for the handle (an orphaned run's end)."""
    gate = threading.Event()
    for _ in range(int(_BOUND * 200)):
        if host.poll(handle, conversation_id=conversation) is None:
            return True
        gate.wait(0.005)
    return False


def _launch(host: RunHost, handle: str, body, conversation: str = "c1", **kw) -> bool:
    assert host.claim(handle, conversation_id=conversation)
    return host.start(handle, body, conversation_id=conversation, **kw)


# ---- the host --------------------------------------------------------------------------------


class TestTheHostRunsABody:
    def test_a_run_finishes_and_its_result_is_polled(self) -> None:
        host = RunHost()
        assert _launch(host, "h1", _instant)
        assert host.wait("h1", _BOUND)
        poll = host.poll("h1", conversation_id="c1")
        assert poll == RunPoll(True, _ok())
        assert poll.result.answer == "done"

    def test_a_reserved_or_running_run_polls_unfinished(self) -> None:
        host = RunHost()
        assert host.claim("h1", conversation_id="c1")
        assert host.poll("h1", conversation_id="c1") == RunPoll(False, None)
        body, entered = _parked()
        assert host.start("h1", body, conversation_id="c1")
        assert entered.wait(_BOUND)
        assert host.poll("h1", conversation_id="c1") == RunPoll(False, None)
        host.cancel("h1")
        assert host.wait("h1", _BOUND)

    def test_the_body_runs_async_code_with_sync_unset(self) -> None:
        """Kills: a worker that inherits ``aio.SYNC`` (an async body that really suspends then
        raises ``SuspendedInSyncModeError``), or a body driven without an event loop."""
        seen: dict[str, object] = {}

        async def body(token: CancelToken) -> RunResult:
            seen["sync"] = SYNC.get()
            seen["loop"] = asyncio.get_running_loop() is not None
            seen["thread"] = threading.current_thread()
            await asyncio.sleep(0)  # a real suspension
            seen["hop"] = await aoffload(len, "abcd")
            return _ok()

        host = RunHost()
        token = SYNC.set(True)  # the starting thread is inside a sync turn
        try:
            assert _launch(host, "h1", body)
        finally:
            SYNC.reset(token)
        assert host.wait("h1", _BOUND)
        assert host.poll("h1", conversation_id="c1").result.status is Status.OK
        assert seen["sync"] is False
        assert seen["loop"] is True and seen["hop"] == 4
        thread = seen["thread"]
        assert isinstance(thread, threading.Thread)
        assert thread is not threading.current_thread()
        assert thread.daemon  # never keeps the interpreter alive
        assert thread.name.startswith("fsm-llm-run-")

    def test_it_starts_from_inside_a_sync_turn(self) -> None:
        host = RunHost()

        async def start() -> bool:
            return _launch(host, "h1", _instant)

        assert run_sync(start())
        assert host.wait("h1", _BOUND)
        assert host.poll("h1", conversation_id="c1").result.status is Status.OK

    def test_announce_runs_before_the_body_and_outside_the_lock(self) -> None:
        """Kills: ``run/start`` written after the thread started, and a log write (``announce``)
        under the host lock."""
        order: list[str] = []
        free: list[bool] = []
        host = RunHost()

        async def body(token: CancelToken) -> RunResult:
            order.append("body")
            return _ok()

        def announce() -> None:
            got = host._lock.acquire(blocking=False)
            free.append(got)
            if got:
                host._lock.release()
            order.append("announce")

        assert _launch(host, "h1", body, announce=announce)
        assert host.wait("h1", _BOUND)
        assert order == ["announce", "body"]
        assert free == [True]

    def test_a_failing_announce_starts_nothing_and_leaks_no_slot(self) -> None:
        ran = threading.Event()

        async def body(token: CancelToken) -> RunResult:
            ran.set()
            return _ok()

        host = RunHost(max_runs=1)

        def announce() -> None:
            raise OSError("disk full")

        assert host.claim("h1", conversation_id="c1")
        with pytest.raises(OSError):
            host.start("h1", body, conversation_id="c1", announce=announce)
        assert not ran.is_set()
        assert host.poll("h1", conversation_id="c1") is None
        assert _launch(host, "h2", _instant)  # the single slot is free again
        assert host.wait("h2", _BOUND)

    def test_a_thread_that_cannot_start_is_dropped(self, monkeypatch) -> None:
        class _NoThread:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def start(self) -> None:
                raise RuntimeError("can't start new thread")

        monkeypatch.setattr(runs, "Thread", _NoThread)
        host = RunHost(max_runs=1)
        assert host.claim("h1", conversation_id="c1")
        assert host.start("h1", _instant, conversation_id="c1") is False
        assert host.poll("h1", conversation_id="c1") is None
        monkeypatch.undo()
        assert _launch(host, "h2", _instant)
        assert host.wait("h2", _BOUND)


class TestCancelAndErrors:
    def test_cancel_sets_the_per_run_token_and_no_other(self) -> None:
        """Kills: ``cancel`` that sets nothing, or a token shared between runs."""
        host = RunHost()
        one, entered_one = _parked()
        two, entered_two = _parked()
        assert _launch(host, "h1", one)
        assert _launch(host, "h2", two)
        assert entered_one.wait(_BOUND) and entered_two.wait(_BOUND)
        assert host.cancel("h1") is True
        assert host.wait("h1", _BOUND)
        result = host.poll("h1", conversation_id="c1").result
        assert (result.status, result.reason) == (Status.ABORTED, Reason.CANCELLED)
        assert host.poll("h2", conversation_id="c1") == RunPoll(
            False, None
        )  # untouched
        host.cancel("h2")
        assert host.wait("h2", _BOUND)

    def test_a_cancel_before_start_is_seen_by_the_body(self) -> None:
        host = RunHost()
        assert host.claim("h1", conversation_id="c1")
        assert host.cancel("h1") is True
        body, _ = _parked()
        assert host.start("h1", body, conversation_id="c1")
        assert host.wait("h1", _BOUND)
        assert host.poll("h1", conversation_id="c1").result.reason is Reason.CANCELLED

    def test_cancel_of_an_unknown_finished_or_foreign_handle_is_false(self) -> None:
        host = RunHost()
        assert host.cancel("nope") is False
        assert _launch(host, "h1", _instant)
        assert host.wait("h1", _BOUND)
        assert host.cancel("h1") is False  # finished
        body, entered = _parked()
        assert _launch(host, "h2", body, conversation="c2")
        assert entered.wait(_BOUND)
        assert host.cancel("h2", conversation_id="c1") is False  # not its conversation
        assert host.cancel("h2", conversation_id="c2") is True
        assert host.wait("h2", _BOUND)

    def test_an_exception_is_stored_as_its_type_only(self) -> None:
        """Kills: a stored exception text or argument (census D6: it could reach a prompt)."""
        secret = "SECRET-token-4711"

        async def body(token: CancelToken) -> RunResult:
            raise ValueError(secret, {"api_key": secret})

        host = RunHost()
        assert _launch(host, "h1", body)
        assert host.wait("h1", _BOUND)
        result = host.poll("h1", conversation_id="c1").result
        assert (result.status, result.reason) == (
            Status.FAILED,
            Reason.UNEXPECTED_ERROR,
        )
        assert result.detail == {"error": "ValueError"}
        assert secret not in repr(result) and secret not in result.model_dump_json()

    def test_a_cancelled_error_and_a_wrong_return_value_are_mapped_too(self) -> None:
        async def cancelled(token: CancelToken) -> RunResult:
            raise asyncio.CancelledError

        async def wrong(token: CancelToken):
            return "not a result"

        host = RunHost()
        assert _launch(host, "h1", cancelled)
        assert _launch(host, "h2", wrong)
        assert host.wait("h1", _BOUND) and host.wait("h2", _BOUND)
        first = host.poll("h1", conversation_id="c1").result
        second = host.poll("h2", conversation_id="c1").result
        assert first.detail == {"error": "CancelledError"}
        assert second.detail == {"error": "TypeError"}
        assert first.reason is second.reason is Reason.UNEXPECTED_ERROR


class TestCapsAndScope:
    def test_the_cap_refuses_a_claim_and_a_start_beyond_it(self) -> None:
        """Kills: no cap (an unbounded thread count), a cap checked at claim only."""
        host = RunHost(max_runs=2)
        body, entered = _parked()
        assert DEFAULT_MAX_RUNS >= 2
        assert _launch(host, "a", body)
        assert _launch(host, "b", body)
        assert entered.wait(_BOUND)
        assert host.claim("c", conversation_id="c1") is False  # two are running
        host.cancel("a")
        assert host.wait("a", _BOUND)
        assert host.claim("c", conversation_id="c1") is True
        host.cancel("b")
        assert host.wait("b", _BOUND)

    def test_a_reservation_made_below_the_cap_is_refused_at_start_when_full(
        self,
    ) -> None:
        host = RunHost(max_runs=1)
        body, entered = _parked()
        assert host.claim("a", conversation_id="c1")
        assert host.claim("b", conversation_id="c1")  # nothing runs yet
        assert host.start("a", body, conversation_id="c1")
        assert entered.wait(_BOUND)
        assert host.start("b", _instant, conversation_id="c1") is False
        assert host.poll("b", conversation_id="c1") is None
        host.cancel("a")
        assert host.wait("a", _BOUND)

    def test_a_start_needs_the_reservation_of_that_conversation(self) -> None:
        """Kills (the K7.4 D3 mutant, at the host): a start that does not check its reservation,
        so a forged record could start a run, or run twice."""
        ran: list[str] = []

        async def body(token: CancelToken) -> RunResult:
            ran.append("x")
            return _ok()

        host = RunHost()
        assert host.start("never", body, conversation_id="c1") is False  # no claim
        assert host.claim("h1", conversation_id="c1")
        assert (
            host.start("h1", body, conversation_id="c2") is False
        )  # another conversation
        assert host.start("h1", body, conversation_id="c1") is True
        assert host.wait("h1", _BOUND)
        assert host.start("h1", body, conversation_id="c1") is False  # once
        assert ran == ["x"]

    def test_poll_is_scoped_to_the_conversation_and_unknown_is_none(self) -> None:
        """Kills: a poll by handle alone (a forged record adopts another conversation's run)."""
        host = RunHost()
        assert _launch(host, "h1", _instant)
        assert host.wait("h1", _BOUND)
        assert host.poll("h1", conversation_id="c2") is None
        assert host.poll("unknown", conversation_id="c1") is None
        assert host.poll("h1", conversation_id="c1") is not None

    def test_a_handle_is_claimed_once_and_must_be_a_string(self) -> None:
        host = RunHost()
        assert host.claim("h1", conversation_id="c1")
        assert host.claim("h1", conversation_id="c1") is False
        assert host.claim("h1", conversation_id="c2") is False
        for bad in ("", " ", "x" * 300):
            with pytest.raises(ValueError):
                host.claim(bad, conversation_id="c1")
        with pytest.raises(ValueError):
            host.claim("h2", conversation_id="")
        with pytest.raises(ValueError):
            RunHost(max_runs=0)

    def test_the_tables_are_bounded(self, monkeypatch) -> None:
        monkeypatch.setattr(runs, "MAX_RESERVED_RUNS", 3)
        monkeypatch.setattr(runs, "MAX_FINISHED_RUNS", 3)
        host = RunHost()
        for i in range(6):
            assert host.claim(f"r{i}", conversation_id="c1")
        assert host.poll("r0", conversation_id="c1") is None  # the oldest was evicted
        assert host.poll("r5", conversation_id="c1") == RunPoll(False, None)
        for i in range(6):
            assert _launch(host, f"f{i}", _instant, conversation="c2")
            assert host.wait(f"f{i}", _BOUND)
        assert host.poll("f0", conversation_id="c2") is None  # evicted
        assert host.poll("f5", conversation_id="c2").finished is True

    def test_cancel_for_drops_reservations_and_results_and_orphans_a_runner(
        self,
    ) -> None:
        """Kills: leaks after the conversation ended (a reservation, a finished result, a result
        stored for a run nobody will poll)."""
        host = RunHost()
        assert _launch(host, "done", _instant)
        assert host.wait("done", _BOUND)
        assert host.claim("reserved", conversation_id="c1")
        body, entered = _parked()
        assert _launch(host, "running", body)
        assert entered.wait(_BOUND)
        assert _launch(host, "other", _instant, conversation="c2")
        assert host.wait("other", _BOUND)
        assert host.cancel_for("c1") == 1  # one run was told to stop
        for handle in ("done", "reserved"):
            assert host.poll(handle, conversation_id="c1") is None
        assert (
            host.poll("other", conversation_id="c2") is not None
        )  # another conversation
        # the running run ends by its token and, being orphaned, leaves no result behind
        assert _wait_gone(host, "running", "c1")
        assert host.cancel_for("c1") == 0
        assert host.cancel_for("nobody") == 0


class TestTheHostIsALeaf:
    def test_runs_py_joins_nothing_sleeps_nowhere_and_imports_no_manager(self) -> None:
        """Kills: a timed ``Thread.join`` (hung Windows 3.15 in plan ace7a8c8), a sleep, an
        import of the API or the manager (a nested manager lock, census D7)."""
        text = Path(runs.__file__).read_text(encoding="utf-8")
        tree = ast.parse(text)
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert "join" not in calls
        assert "sleep" not in calls
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        forbidden = {"fsm_llm.engine.api", "fsm_llm.engine.fsm", "fsm_llm.engine.turn"}
        assert not {m for m in modules if any(m.startswith(f) for f in forbidden)}


# ---- the targets object ----------------------------------------------------------------------


class _Recording(_Stub):
    """A code-backed pattern that records the context it was run with."""

    def __init__(self) -> None:
        super().__init__("pattern", id="probe.record", inputs=["task"])
        self.seen: dict[str, object] = {}
        self.done = threading.Event()

    async def execute(self, inputs, objects, context, identity) -> RunResult:
        self.seen.update(
            inputs=dict(inputs),
            parent=context.parent,
            cancel=context.cancel,
            max_seconds=context.max_seconds,
            registry=context.registry,
        )
        self.done.set()
        return RunResult(status=Status.OK, answer="ran", identity=identity)


def _targets(*executables, context: RunContext | None = None, **kw):
    registry = ExecutableRegistry()
    for executable in executables:
        registry.register(executable)
    return ExecutableTargets(registry, context, **kw)


class TestExecutableTargets:
    def test_it_is_a_resolver_and_a_launcher(self) -> None:
        targets = _targets()
        assert isinstance(targets, TargetResolver)
        assert isinstance(targets, RunLauncher)
        assert isinstance(targets.host, RunHost)
        assert isinstance(targets.registry, ExecutableRegistry)
        assert isinstance(_Launcher(), RunLauncher)

    def test_frame_resolution_goes_to_the_registry(self) -> None:
        from tests.test_fsm_llm.test_executable_adapters import _doc

        registry = ExecutableRegistry()
        registry.add_document(_doc("fsm"))
        targets = ExecutableTargets(registry)
        assert targets.resolve_target("demo") == registry.resolve_target("demo")
        assert targets.resolve_target("nope") is None
        assert targets.resolve_run("demo") is None  # a body kind is no delegated target

    def test_resolve_run_judges_against_its_own_context(self) -> None:
        needs = _Stub(id="probe.needs", callables=["tools"])
        plain = _targets(needs)
        assert plain.resolve_run("probe.needs", input_keys=["task"]) is None
        have: NameRegistry[object] = NameRegistry("callables")
        have.register("tools", [])
        equipped = _targets(needs, context=RunContext(callables=have))
        target = equipped.resolve_run("probe.needs", input_keys=["task"])
        assert isinstance(target, DelegatedTarget) and target.kind == "pattern"
        conv = _Stub(id="probe.chat", capabilities=[Capability.CONVERSATIONAL])
        assert _targets(conv).resolve_run("probe.chat") is None

    def test_a_run_gets_its_own_token_the_conversation_as_parent_and_the_narrowest_limit(
        self,
    ) -> None:
        """Kills: the base token shared by the run (a cancel of one run would stop the host's
        others), a ``parent`` that is not the conversation, a call limit above the host's."""
        stub = _Recording()
        base = RunContext(max_seconds=30)
        targets = _targets(stub, context=base)
        target = targets.resolve_run("probe.record", input_keys=["task"])
        assert target is not None
        assert targets.claim("h1", conversation_id="conv-1")
        assert targets.start(
            "h1", target, {"task": "t"}, conversation_id="conv-1", max_seconds=120
        )
        assert stub.done.wait(_BOUND)
        assert targets.host.wait("h1", _BOUND)
        assert stub.seen["inputs"] == {"task": "t"}
        assert stub.seen["parent"] == "conv-1"
        assert stub.seen["cancel"] is not base.cancel
        assert stub.seen["max_seconds"] == 30  # min(call 120, host 30)
        poll = targets.poll("h1", conversation_id="conv-1")
        assert poll is not None and poll.result.answer == "ran"
        assert not base.cancel.cancelled

    def test_a_tighter_call_limit_wins_and_none_keeps_the_hosts(self) -> None:
        for call, expected in ((5, 5), (None, 30)):
            stub = _Recording()
            targets = _targets(stub, context=RunContext(max_seconds=30))
            target = targets.resolve_run("probe.record", input_keys=["task"])
            assert targets.claim("h", conversation_id="c")
            assert targets.start(
                "h", target, {"task": "t"}, conversation_id="c", max_seconds=call
            )
            assert targets.host.wait("h", _BOUND)
            assert stub.seen["max_seconds"] == expected

    def test_an_instantiation_failure_is_the_mapped_failed_result(self) -> None:
        class _Broken(_Stub):
            def instantiate(self, options=None, variables=None):
                raise RuntimeError("private detail")

        targets = _targets(_Broken(id="probe.broken"))
        target = targets.resolve_run("probe.broken")
        assert targets.claim("h", conversation_id="c")
        assert targets.start("h", target, {"task": "t"}, conversation_id="c")
        assert targets.host.wait("h", _BOUND)
        result = targets.poll("h", conversation_id="c").result
        assert result.detail == {"error": "RuntimeError"}

    def test_cancel_and_cancel_for_reach_the_host(self) -> None:
        released = threading.Event()

        class _Waits(_Stub):
            async def execute(self, inputs, objects, context, identity):
                released.set()
                for _ in range(int(_BOUND * 200)):
                    if context.cancel.cancelled:
                        return self.stopped(Status.ABORTED, Reason.CANCELLED, identity)
                    await asyncio.sleep(0.005)
                return self.stopped(Status.FAILED, None, identity)

        targets = _targets(_Waits(id="probe.waits"))
        target = targets.resolve_run("probe.waits")
        for handle, conversation in (("a", "c1"), ("b", "c2")):
            assert targets.claim(handle, conversation_id=conversation)
            assert targets.start(
                handle, target, {"task": "t"}, conversation_id=conversation
            )
        assert released.wait(_BOUND)
        assert targets.cancel("a") is True
        assert targets.host.wait("a", _BOUND)
        assert targets.poll("a", conversation_id="c1").result.reason is Reason.CANCELLED
        assert targets.cancel_for("c2") == 1
        assert _wait_gone(targets.host, "b", "c2")


class TestSharedDoubleOnTwoThreads:
    def test_two_threads_share_one_scripted_llm_with_a_responses_mapping(self) -> None:
        """A shared ``ScriptedLLM`` on two worker threads, each driving a real conversation:
        the double must answer each by STATE (a ``responses`` mapping), never by call order
        (``replies.pop(0)`` across threads is nondeterministic: D-048)."""
        llm = ScriptedLLM(
            responses={"alpha": "A-reply", "beta": "B-reply"}, default_response="?"
        )
        barrier = threading.Barrier(2, timeout=_BOUND)

        def definition(state: str) -> dict:
            return kc._fsm(
                "Shared", state, kc._state(state, response_instructions="Hi.")
            )

        def body_for(state: str):
            async def body(token: CancelToken) -> RunResult:
                await aoffload(barrier.wait)  # both threads reach the double together
                api = API.from_definition(definition(state), llm_interface=llm)
                _, greeting = await api.astart_conversation()
                return RunResult(status=Status.OK, answer=greeting)

            return body

        host = RunHost()
        assert _launch(host, "a", body_for("alpha"), conversation="c1")
        assert _launch(host, "b", body_for("beta"), conversation="c2")
        assert host.wait("a", _BOUND) and host.wait("b", _BOUND)
        first = host.poll("a", conversation_id="c1").result
        second = host.poll("b", conversation_id="c2").result
        assert (first.status, first.answer) == (Status.OK, "A-reply")
        assert (second.status, second.answer) == (Status.OK, "B-reply")
        assert len(llm.calls("generate_response")) == 2
        assert llm.unexpected == []


@pytest.mark.skipif(
    current_timing_factor() > 1,
    reason="a thread-stress row stalled CPython 3.12 under a tracer (LESSONS ace7a8c8)",
)
class TestStress:
    def test_many_runs_across_conversations_all_finish_and_leave_nothing(self) -> None:
        host = RunHost(max_runs=64)

        async def body(token: CancelToken) -> RunResult:
            for _ in range(3):
                await asyncio.sleep(0)
            return _ok()

        handles = [(f"h{i}", f"c{i % 4}") for i in range(48)]
        for handle, conversation in handles:
            assert _launch(host, handle, body, conversation=conversation)
        for handle, conversation in handles:
            assert host.wait(handle, _BOUND)
            poll = host.poll(handle, conversation_id=conversation)
            assert poll is not None and poll.result.status is Status.OK
        for conversation in ("c0", "c1", "c2", "c3"):
            host.cancel_for(conversation)
        assert all(host.poll(h, conversation_id=c) is None for h, c in handles)


# ---- the end hook of the API -----------------------------------------------------------------


class _Spy(_Launcher):
    """A launcher and resolver that records ``cancel_for`` and what the API looked like then."""

    def __init__(self, api_ref: list, raises: bool = False) -> None:
        self.api_ref = api_ref
        self.raises = raises
        self.calls: list[tuple[str, bool]] = []

    def resolve_target(self, target_id, *, conversation_id=None):
        return None

    def cancel_for(self, conversation_id):
        api = self.api_ref[0]
        self.calls.append(
            (conversation_id, conversation_id in api.list_active_conversations())
        )
        if self.raises:
            raise RuntimeError("launcher is broken")
        return 0


def _api_with(targets) -> API:
    return API.from_definition(
        _CHAT(), llm_interface=ScriptedLLM(default_response="r"), targets=targets
    )


class TestEndConversationCancelsItsRuns:
    def _parked_run(
        self, targets: ExecutableTargets, conversation_id: str, handle: str
    ):
        body, entered = _parked()
        assert targets.host.claim(handle, conversation_id=conversation_id)
        assert targets.host.start(handle, body, conversation_id=conversation_id)
        assert entered.wait(_BOUND)

    def test_end_conversation_cancels_the_runs_of_that_conversation(self) -> None:
        """The named mutant (D7): ``end_conversation`` without the ``cancel_for`` call turns this
        row red (the parked run is never told to stop, so the bounded wait fails)."""
        targets = _targets()
        api = _api_with(targets)
        cid, _ = api.start_conversation()
        other, _ = api.start_conversation()
        self._parked_run(targets, cid, "h1")
        self._parked_run(targets, other, "h2")
        api.end_conversation(cid)
        assert targets.host.wait("h1", _BOUND)
        assert targets.host.poll("h2", conversation_id=other) == RunPoll(False, None)
        api.end_conversation(other)
        assert targets.host.wait("h2", _BOUND)

    def test_close_cancels_every_run(self) -> None:
        targets = _targets()
        api = _api_with(targets)
        cid, _ = api.start_conversation()
        self._parked_run(targets, cid, "h1")
        api.close()
        assert targets.host.wait(
            "h1", _BOUND
        )  # close never joins: the token ended the run

    def test_the_stale_sweep_cancels_the_runs_of_a_swept_conversation(self) -> None:
        targets = _targets()
        api = _api_with(targets)
        cid, _ = api.start_conversation()
        self._parked_run(targets, cid, "h1")
        api._last_accessed[cid] = 0.0  # idle since the clock began
        assert api.cleanup_stale_conversations(max_idle_seconds=1.0) == [cid]
        assert targets.host.wait("h1", _BOUND)

    def test_the_hook_runs_once_with_the_root_id_after_the_bookkeeping_drop(
        self,
    ) -> None:
        ref: list = []
        spy = _Spy(ref)
        api = _api_with(spy)
        ref.append(api)
        cid, _ = api.start_conversation()
        api.end_conversation(cid)
        assert spy.calls == [
            (cid, False)
        ]  # gone from the live table when it was called

    def test_a_refused_end_does_not_cancel(self, monkeypatch) -> None:
        ref: list = []
        spy = _Spy(ref)
        api = _api_with(spy)
        ref.append(api)
        cid, _ = api.start_conversation()

        def refuse(frame_id):
            raise FSMError("refused")

        monkeypatch.setattr(api.fsm_manager, "end_conversation", refuse)
        with pytest.raises(FSMError):
            api.end_conversation(cid)
        assert spy.calls == []
        assert cid in api.list_active_conversations()
        monkeypatch.undo()
        api.end_conversation(cid)
        assert [c for c, _ in spy.calls] == [cid]

    def test_a_failing_launcher_does_not_fail_the_end(self) -> None:
        ref: list = []
        spy = _Spy(ref, raises=True)
        api = _api_with(spy)
        ref.append(api)
        cid, _ = api.start_conversation()
        api.end_conversation(cid)  # no exception
        assert cid not in api.list_active_conversations()
        assert len(spy.calls) == 1

    def test_a_resolver_that_is_no_launcher_is_left_alone(self) -> None:
        api = _api_with(ExecutableRegistry())
        cid, _ = api.start_conversation()
        api.end_conversation(cid)
        assert cid not in api.list_active_conversations()
