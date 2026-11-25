"""A workflow is a delegated ``call`` target and never a frame (plan 7d9c0955, step 6, D-010).

``resolve_target`` (frame mode) answers ``None`` for the ``workflow`` kind by KIND, and
``resolve_delegated`` answers a ``DelegatedTarget`` for it when the host holds a launcher and has
bound every name in ``requires.callables``. A real ``ExecutableTargets`` and ``RunHost`` then run a
real ``workflow`` document on a daemon thread (``asyncio.run`` per run) through ``settle``: the
record lands with the declared returns, ``run/start`` and ``run/end`` are written by the engine
thread alone, the child sees the declared inputs only (no grant), a budget and a cancel reach the
run, and a handle the host never saw ends ``failed/run_lost`` with nothing re-run. Doubles only:
plain callables and a scripted parent interface, no provider, no network. Rules of the rows
(LESSONS ace7a8c8, D-048): no ``Thread.join``, events and bounded waits, no sleep as
synchronization.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm import API
from fsm_llm.core.names import NameRegistry
from fsm_llm.engine.api import delegated
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry
from fsm_llm.engine.executables.runs import ExecutableTargets
from fsm_llm.engine.kinds.targets import DelegatedTarget
from fsm_llm.runtime import EventLog
from tests.conftest import current_timing_factor
from tests.test_fsm_llm.test_call_delegated import _initial_call
from tests.test_fsm_llm.test_call_settle import _Gate
from tests.test_fsm_llm.test_call_state import _call, _parent
from tests.test_fsm_llm.test_delegated_resolution import _Launcher, _Stub
from tests.test_fsm_llm.test_resolve_target_guard import _BodyBacked
from tests.test_fsm_llm_workflows.test_workflow_executable import _auto, _document

_BOUND = 20 * current_timing_factor()  # seconds: a wait that would hang fails instead
_ID = "wf.demo"
_ROOT = Path(__file__).resolve().parents[2]


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(_ROOT / "src")


@pytest.fixture(autouse=True)
def _short_wait(monkeypatch):
    monkeypatch.setattr(delegated, "DELEGATED_WAIT_SECONDS", 0.05)


class _Seen:
    """What the workflow's callables saw, and the gates that hold them."""

    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []
        self.threads: list[str] = []
        self.log_types: list[list[str]] = []
        self.ran: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.log: EventLog | None = None
        self.root = ""

    def callables(self, *, hold: bool = False) -> NameRegistry[Any]:
        names: NameRegistry[Any] = NameRegistry("callables")

        def step(label: str, *, wait: bool = False):
            def action(context: dict[str, Any]) -> dict[str, Any]:
                self.ran.append(label)
                self.contexts.append(dict(context))
                self.threads.append(threading.current_thread().name)
                if self.log is not None:
                    self.log_types.append([e["type"] for e in self.log.read(self.root)])
                if wait:
                    self.entered.set()
                    assert self.release.wait(_BOUND)
                return {"found": f"{label}:{context.get('topic')}"}

            return action

        names.register("mark", step("mark"))
        names.register("first", step("first", wait=hold))
        names.register("middle", step("middle"))
        names.register("last", step("last"))
        return names


def _workflow(steps: dict[str, Any], *callables: str) -> dict[str, Any]:
    return _document(
        steps,
        ident=_ID,
        inputs=("topic",),
        outputs=("found",),
        callables=callables,
    )


def _one() -> dict[str, Any]:
    return _workflow({"s1": _auto("s1", "", "mark")}, "mark")


def _three() -> dict[str, Any]:
    return _workflow(
        {
            "s1": _auto("s1", "s2", "first"),
            "s2": _auto("s2", "s3", "middle"),
            "s3": _auto("s3", "", "last"),
        },
        "first",
        "middle",
        "last",
    )


def _registry(document: dict[str, Any]) -> ExecutableRegistry:
    registry = ExecutableRegistry()
    registry.add_document(document)
    return registry


def _context(seen: _Seen, *, bind: bool = True, hold: bool = False, **kw: Any):
    return RunContext(
        callables=seen.callables(hold=hold) if bind else None, event_log=seen.log, **kw
    )


def _assemble(
    tmp_path: Path,
    document: dict[str, Any],
    seen: _Seen,
    *,
    hold: bool = False,
    config: dict[str, Any] | None = None,
):
    seen.log = EventLog(tmp_path / "logs")
    registry = _registry(document)
    targets = ExecutableTargets(registry, _context(seen, hold=hold))
    definition = _parent(
        _call(target=_ID, inputs=["topic"], returns={"found": "found"})
        if config is None
        else config,
        response_instructions="",
    )
    api = API.from_definition(
        _initial_call(definition),
        llm_interface=_Gate(default_response="r"),
        targets=targets,
        event_log=seen.log,
    )
    return api, targets


def _record(api: API, cid: str) -> dict[str, Any]:
    frame = api.conversation_stacks[cid][0].conversation_id
    record = api.fsm_manager.copy_raw_context_data(frame).get("call_result")
    assert isinstance(record, dict)
    return record


def _types(seen: _Seen, cid: str) -> list[str]:
    assert seen.log is not None
    return [e["type"] for e in seen.log.read(cid)]


# ---- frame mode: a workflow is never a frame, by kind ----------------------------------------


class TestNeverAFrame:
    def test_a_real_workflow_document_is_no_frame_target(self) -> None:
        assert _registry(_one()).resolve_target(_ID) is None

    def test_the_guard_is_by_kind_even_for_an_instance_that_carries_a_body(
        self,
    ) -> None:
        """Kills: the frame guard keyed on ``isinstance(definition, FSMDefinition)`` alone, or
        the kind left out of ``_NEVER_A_FRAME_KINDS`` (a body-carrying workflow instance would
        be pushed with no host callables or registry)."""
        registry = ExecutableRegistry()
        registry.register(_BodyBacked("workflow"))
        assert registry.resolve_target("reasoning.analytical") is None

    def test_a_non_workflow_body_kind_is_still_a_frame(self) -> None:
        registry = ExecutableRegistry()
        registry.register(_BodyBacked("fsm"))
        assert registry.resolve_target("reasoning.analytical") is not None


# ---- delegated resolution ---------------------------------------------------------------------


class TestResolvesDelegated:
    def _resolve(self, registry, *, context=None, launcher=True, **kw):
        return registry.resolve_delegated(
            _ID,
            launcher=_Launcher() if launcher else None,
            context=context,
            **kw,
        )

    def test_a_workflow_with_bound_callables_resolves_with_its_identities(
        self,
    ) -> None:
        registry = _registry(_one())
        target = self._resolve(registry, context=_context(_Seen()))
        assert isinstance(target, DelegatedTarget)
        assert (target.target_id, target.kind) == (_ID, "workflow")
        executable = registry.get(_ID)
        assert target.executable is executable
        assert target.template_identity == executable.template_identity
        assert target.instance_identity is None
        assert target.outputs == ("found",)

    def test_an_unbound_callable_is_nothing(self) -> None:
        """Kills: delegation allowed with an unbound ``requires.callables`` name."""
        registry = _registry(_one())
        assert self._resolve(registry, context=_context(_Seen())) is not None
        assert self._resolve(registry, context=_context(_Seen(), bind=False)) is None
        assert self._resolve(registry, context=RunContext()) is None
        other: NameRegistry[Any] = NameRegistry("callables")
        other.register("something_else", lambda context: {})
        assert self._resolve(registry, context=RunContext(callables=other)) is None

    def test_no_launcher_is_nothing(self) -> None:
        registry = _registry(_one())
        assert (
            self._resolve(registry, context=_context(_Seen()), launcher=False) is None
        )

    def test_an_undeclared_input_key_is_nothing(self) -> None:
        registry = _registry(_one())
        context = _context(_Seen())
        assert self._resolve(registry, context=context, input_keys=["topic"])
        assert self._resolve(registry, context=context, input_keys=["other"]) is None

    def test_a_refused_budget_and_a_deprecated_executable_are_nothing(self) -> None:
        from fsm_llm.core.codes import Lifecycle

        stub = _Stub("workflow", id=_ID, refused=("max_steps",))
        registry = ExecutableRegistry()
        registry.register(stub)
        assert self._resolve(registry, context=RunContext()) is not None
        assert self._resolve(registry, context=RunContext(), max_steps=3) is None
        stub.lifecycle = Lifecycle.DEPRECATED
        assert self._resolve(registry, context=RunContext()) is None


# ---- a real run through the host and settle ---------------------------------------------------


class TestARealRun:
    def test_the_call_runs_the_workflow_delegated_and_lands_the_declared_returns(
        self, tmp_path: Path
    ) -> None:
        seen = _Seen()
        api, targets = _assemble(tmp_path, _one(), seen)
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            seen.root = cid
            api.run_until_terminal(cid, max_steps=30)
            data = api.get_data(cid)
            record = data["call_result"]
            assert (record["status"], record["target"]) == ("ok", _ID)
            assert record["template_identity"] == (
                targets.registry.get(_ID).template_identity
            )
            assert data["found"] == "mark:why"
            assert seen.ran == ["mark"]
            assert seen.threads[0] != threading.current_thread().name  # a worker thread
            assert _types(seen, cid).count("run/start") == 1
            assert _types(seen, cid).count("run/end") == 1
        finally:
            api.close()
            assert seen.log is not None
            seen.log.close()

    def test_run_start_is_on_disk_before_the_worker_and_run_end_is_not(
        self, tmp_path: Path
    ) -> None:
        """Census D8: the engine thread alone writes run events; the worker (the workflow's own
        callable, here) finds ``run/start`` already written and ``run/end`` not yet."""
        seen = _Seen()
        api, _targets = _assemble(tmp_path, _one(), seen)
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            seen.root = cid
            api.run_until_terminal(cid, max_steps=30)
            assert seen.log_types, "the callable never ran"
            assert "run/start" in seen.log_types[0]
            assert "run/end" not in seen.log_types[0]
            events = [e for e in seen.log.read(cid) if e["type"].startswith("run/")]
            assert [e["type"] for e in events] == ["run/start", "run/end"]
            assert events[1]["status"] == "ok"
        finally:
            api.close()
            assert seen.log is not None
            seen.log.close()

    def test_the_child_gets_the_declared_inputs_and_no_grant(
        self, tmp_path: Path
    ) -> None:
        """Census D1: the parent context carries a forged grant and an undeclared key; the run
        receives ``topic`` only (settle builds the inputs from the declared list)."""
        seen = _Seen()
        api, _targets = _assemble(tmp_path, _one(), seen)
        try:
            cid, _ = api.start_conversation(
                {
                    "topic": "why",
                    "_approval_granted": True,
                    "approval_granted": True,
                    "other_note": "x",
                }
            )
            seen.root = cid
            api.run_until_terminal(cid, max_steps=30)
            keys = {k for k in seen.contexts[0] if not k.startswith("_")}
            assert keys == {"topic"}
            assert "_approval_granted" not in seen.contexts[0]
            assert _record(api, cid)["status"] == "ok"
        finally:
            api.close()
            assert seen.log is not None
            seen.log.close()

    def test_a_callable_the_host_never_bound_ends_unknown_target_and_runs_nothing(
        self, tmp_path: Path
    ) -> None:
        seen = _Seen()
        api, targets = _assemble(tmp_path, _one(), seen)
        targets._context = _context(seen, bind=False)  # the host lost its callables
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            seen.root = cid
            api.run_until_terminal(cid, max_steps=30)
            record = _record(api, cid)
            assert (record["status"], record["reason"]) == ("failed", "unknown_target")
            assert seen.ran == []
        finally:
            api.close()
            assert seen.log is not None
            seen.log.close()


class TestBudgetsAndCancel:
    def test_the_calls_step_budget_reaches_the_run(self, tmp_path: Path) -> None:
        """Census D7: ``max_steps`` of the call state is the engine's step budget of the run, so
        a three step chain with a budget of two is cut and its last step never runs."""
        seen = _Seen()
        api, _targets = _assemble(
            tmp_path,
            _three(),
            seen,
            config=_call(
                target=_ID, inputs=["topic"], returns={"found": "found"}, max_steps=2
            ),
        )
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            seen.root = cid
            api.run_until_terminal(cid, max_steps=30)
            record = _record(api, cid)
            assert record["status"] in ("aborted", "failed")
            assert "last" not in seen.ran
            assert seen.ran == ["first", "middle"]
            ended = [e for e in seen.log.read(cid) if e["type"] == "run/end"]
            assert len(ended) == 1 and ended[0]["status"] == "aborted"
        finally:
            api.close()
            assert seen.log is not None
            seen.log.close()

    def test_end_conversation_cancels_the_run_through_the_engine(
        self, tmp_path: Path
    ) -> None:
        """Census D7: the token of a run ``settle`` started is set by ``end_conversation``; the
        workflow stops at the next step boundary (its last step never runs) and the next
        conversation on the same API runs to its end."""
        seen = _Seen()
        api, targets = _assemble(tmp_path, _three(), seen, hold=True)
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            seen.root = cid
            api.advance(cid)
            assert seen.entered.wait(_BOUND)
            handle = _record(api, cid)["handle"]
            api.end_conversation(cid)
            seen.release.set()
            assert targets.wait(handle, _BOUND)
            assert "last" not in seen.ran
            nxt, _ = api.start_conversation({"topic": "again"})
            seen.root = nxt
            api.run_until_terminal(nxt, max_steps=30)
            assert _record(api, nxt)["status"] in ("ok", "aborted", "failed")
        finally:
            seen.release.set()
            api.close()
            assert seen.log is not None
            seen.log.close()


class TestARunTheHostNeverSaw:
    def test_a_fork_inside_the_window_ends_run_lost_and_reruns_nothing(
        self, tmp_path: Path
    ) -> None:
        seen = _Seen()
        api, _targets = _assemble(tmp_path, _three(), seen, hold=True)
        try:
            cid, _ = api.start_conversation({"topic": "why"})
            seen.root = cid
            api.advance(cid)
            assert seen.entered.wait(_BOUND)
            running = _record(api, cid)
            assert running["status"] == "running" and running["mode"] == "delegated"
            assert running["handle"].startswith("run-") and running["deadline"] > 0
            new_id, _state = api.fork(cid, at_turn=1)
            api.advance(new_id)
            record = _record(api, new_id)
            assert (record["status"], record["reason"]) == ("failed", "run_lost")
            assert seen.ran == ["first"]  # nothing was started for the fork
            assert "run/start" not in _types(seen, new_id)
            seen.release.set()
            api.run_until_terminal(cid, max_steps=30)
            assert _record(api, cid)["status"] == "ok"
            assert seen.ran == ["first", "middle", "last"]
        finally:
            seen.release.set()
            api.close()
            assert seen.log is not None
            seen.log.close()
