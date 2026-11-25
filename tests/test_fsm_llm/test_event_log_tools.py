"""The tool wrapper: ``ToolRegistry.execute`` writes ``tool/call`` and ``tool/result`` (plan 3f4530d3
step 3, roadmap rows 1.3 and 1.16, decisions D-005 and D-006).

Every row runs a registry tool through PRODUCT code: a handler of a scripted conversation calls
``registry.execute`` inside a turn (the log scope is bound by ``API``), a ``ParallelReactAgent``
dispatch, or a killed child process whose tool body blocks. Nothing here writes to the log by
hand except the one fold row that plays a late straggler event. The rows pin: the order
(``tool/call`` on disk, fsynced for a side-effecting call, BEFORE the body), the fail-closed
write, the caller-only writer of a timeout, the outcome codes, retry and cache behaviour, the fold
rule and the layer proof.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fsm_llm import API, HandlerTiming, create_handler
from fsm_llm.agents import AgentConfig
from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.parallel_react import ParallelReactAgent
from fsm_llm.agents.tool_registries import CachingToolRegistry, RetryingToolRegistry
from fsm_llm.core.codes import ToolErrorCode
from fsm_llm.core.tool_observer import install_tool_observer, reset_tool_observer
from fsm_llm.runtime import EventLog, LogWriteError, bind, unresolved_tool_calls
from fsm_llm.runtime.stores import fold_stack, jsonl, read_events
from fsm_llm.tools import (
    READ_ONLY_ANNOTATIONS,
    ToolAnnotations,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from fsm_llm.tools.constants import Defaults, ErrorMessages
from tests.conftest import current_timing_factor
from tests.test_fsm_llm import event_log_child as child
from tests.test_fsm_llm.event_log_support import (
    CHILD_MODULE,
    child_process,
    reopen_after_kill,
    wait_ready,
)
from tests.test_fsm_llm.test_event_log_jsonl import can_lock_in_child


@pytest.fixture
def log(tmp_path: Path):
    with EventLog(tmp_path / "root") as opened:
        yield opened


def _call(name: str, **parameters: Any) -> ToolCall:
    return ToolCall(tool_name=name, parameters=parameters)


def _run(
    registry: ToolRegistry,
    calls: list[ToolCall],
    *,
    log: EventLog | None,
) -> tuple[str, list[ToolResult], API]:
    """Run *calls* through ``registry.execute`` from a handler of the ``run_tools`` state of a
    scripted conversation (so the turn scope of ``API`` is the one bound), return the results."""
    results: list[ToolResult] = []

    def run_them(_: dict[str, Any]) -> dict[str, Any]:
        results.extend(registry.execute(c) for c in calls)
        return {}

    api = API.from_definition(
        child.tool_fsm(),
        llm_interface=child.ChildLLM(child.scripted_replies()),
        event_log=log,
        handlers=[
            create_handler("run-tools")
            .at(HandlerTiming.PRE_PROCESSING)
            .on_state("run_tools")
            .do(run_them)
        ],
    )
    cid = child.start(api)
    api.advance(cid)  # intake -> call_model
    api.advance(cid)  # the model asks for a tool: call_model -> run_tools
    api.advance(cid)  # the handler runs the calls
    return cid, results, api


def _tool_events(log: EventLog, cid: str) -> list[dict[str, Any]]:
    return [e for e in log.read(cid) if e["type"].startswith("tool/")]


def _registry(**fns: Callable[..., Any]) -> ToolRegistry:
    registry = ToolRegistry()
    for name, fn in fns.items():
        registry.register_function(fn, name=name)
    return registry


def _add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class TestTheWrapperWritesThePair:
    def test_a_registry_call_in_a_logged_turn_is_a_call_then_a_result(
        self, log: EventLog
    ):
        cid, (result,), _ = _run(_registry(add=_add), [_call("add", a=2, b=3)], log=log)
        assert result.success and result.result == 5 and result.error_code is None
        call, done = _tool_events(log, cid)
        assert (call["type"], done["type"]) == ("tool/call", "tool/result")
        assert call["name"] == done["name"] == "add"
        assert call["call_id"] == done["call_id"] and len(call["call_id"]) == 32
        assert call["arguments"] == {"a": 2, "b": 3}
        assert call["side_effecting"] is True and call["gated"] is False
        assert done["status"] == "ok" and done["truncated"] is False
        assert isinstance(done["elapsed_ms"], float) and done["frame"] == 0
        assert "error_type" not in done and "error_code" not in done
        assert set(done) == {
            "v", "seq", "ts", "conversation_id", "frame", "type",
            "name", "call_id", "status", "truncated", "elapsed_ms",
        }  # fmt: skip

    def test_the_log_off_writes_nothing_and_returns_the_same_result(
        self, log: EventLog, tmp_path: Path
    ):
        registry = _registry(add=_add)
        _, (on,), _ = _run(registry, [_call("add", a=2, b=3)], log=log)
        _, (off,), api = _run(registry, [_call("add", a=2, b=3)], log=None)
        assert api.event_log is None
        ignore = {"execution_time_ms"}
        assert on.model_dump(exclude=ignore) == off.model_dump(exclude=ignore)
        assert not (tmp_path / "off").exists()

    def test_with_no_observer_installed_the_registry_is_the_plain_call(self):
        result = _registry(add=_add).execute(_call("add", a=1, b=2))
        assert result.success and result.result == 3 and result.error_code is None

    def test_a_gated_call_says_so(self, log: EventLog):
        registry = _registry(add=_add)
        with bind(log, "c1"):
            registry.execute(_call("add", a=1, b=1), gated=True)
        assert [e["gated"] for e in log.read("c1") if e["type"] == "tool/call"] == [
            True
        ]


class TestTheOrderIsTheCheckpointRule:
    def test_the_call_is_on_disk_and_fsynced_before_the_body_runs(self, log: EventLog):
        seen: list[str] = []
        real_fsync = jsonl._fsync

        def fsync(fd: int) -> None:
            seen.append("fsync")
            real_fsync(fd)

        def body(n: int) -> str:
            seen.append("body")
            disk = [
                e["type"]
                for e in read_events(next((log.root / "logs").glob("*.jsonl")))
            ]
            seen.append("on-disk:" + ",".join(disk[-1:]))
            return "x"

        registry = _registry(charge=body)
        with mock.patch.object(jsonl, "_fsync", fsync):
            _run(registry, [_call("charge", n=1)], log=log)
        at = seen.index("body")
        assert "fsync" in seen[:at]
        assert seen[at + 1] == "on-disk:tool/call"
        # the fsync that follows the tool/call is the last one before the body
        assert seen[at - 1] == "fsync"

    def test_a_read_only_tool_is_logged_but_not_fsynced(self, log: EventLog):
        fsyncs: list[int] = []
        real_fsync = jsonl._fsync

        def fsync(fd: int) -> None:
            fsyncs.append(fd)
            real_fsync(fd)

        registry = ToolRegistry().register_function(
            _add, name="peek", annotations=READ_ONLY_ANNOTATIONS
        )
        writes: list[str] = []
        real_write = jsonl._write

        def write(fd: int, data: Any) -> int:
            writes.extend(json.loads(x)["type"] for x in bytes(data).split(b"\n")[:-1])
            return real_write(fd, data)

        with bind(log, "c1"):
            with (
                mock.patch.object(jsonl, "_fsync", fsync),
                mock.patch.object(jsonl, "_write", write),
            ):
                registry.execute(_call("peek", a=1, b=1))
        assert writes == ["tool/call", "tool/result"] and fsyncs == []
        (call, _) = log.read("c1")
        assert call["side_effecting"] is False

    def test_an_unknown_annotation_means_side_effecting_and_fsynced(
        self, log: EventLog
    ):
        registry = ToolRegistry().register_function(
            _add, name="w", annotations=ToolAnnotations(destructive=False)
        )
        fsyncs: list[int] = []
        real_fsync = jsonl._fsync
        with (
            bind(log, "c1"),
            mock.patch.object(
                jsonl, "_fsync", lambda fd: (fsyncs.append(fd), real_fsync(fd))
            ),
        ):
            registry.execute(_call("w", a=1, b=1))
        assert len(fsyncs) == 1

    def test_a_poisoned_log_is_a_failed_result_the_body_never_runs_and_nothing_raises(
        self, log: EventLog
    ):
        ran: list[int] = []
        registry = _registry(charge=lambda n: ran.append(n) or "x")
        real_write = jsonl._write

        def write(fd: int, data: Any) -> int:
            if b"tool/call" in bytes(data):
                raise OSError("disk full")
            return real_write(fd, data)

        with bind(log, "c1"), mock.patch.object(jsonl, "_write", write):
            result = registry.execute(_call("charge", n=1))
        assert result.success is False and ran == []
        assert result.error_code is ToolErrorCode.EXECUTION_FAILED
        assert [e for e in log.read("c1") if e["type"].startswith("tool/")] == []

    def test_a_refusing_observer_is_a_failed_result_not_a_raise(self):
        ran: list[int] = []

        class _Refuses:
            def begin_tool(self, **_: Any) -> object:
                raise LogWriteError("refused")

            def end_tool(self, token: object, **_: Any) -> None:
                raise AssertionError("no call was begun")

        registry = _registry(charge=lambda n: ran.append(n) or "x")
        token = install_tool_observer(_Refuses())  # type: ignore[arg-type]
        try:
            result = registry.execute(_call("charge", n=1))
        finally:
            reset_tool_observer(token)
        assert not result.success and ran == []

    def test_a_refused_tool_call_write_tells_the_model_a_fixed_text_not_the_log_error(
        self,
    ):
        """Review 6 of plan 3f4530d3. Kills: ``str(error)`` of the log in the failed result (the
        model saw the log file name and the operating system text: ``append to <id>.jsonl failed:
        OSError No space left on device``)."""
        ran: list[int] = []

        class _Refuses:
            def begin_tool(self, **_: Any) -> object:
                raise LogWriteError(
                    "append to 9f1c-conv.jsonl failed: OSError No space left"
                )

            def end_tool(self, token: object, **_: Any) -> None:
                raise AssertionError("no call was begun")

        registry = _registry(charge=lambda n: ran.append(n) or "x")
        token = install_tool_observer(_Refuses())  # type: ignore[arg-type]
        try:
            result = registry.execute(_call("charge", n=1))
        finally:
            reset_tool_observer(token)
        assert ran == [] and not result.success
        for leak in ("jsonl", "9f1c", "OSError", "space"):
            assert leak not in (result.error or ""), result.error
        assert result.error == ErrorMessages.TOOL_NOT_LOGGED.format(name="charge")
        assert result.error_code is ToolErrorCode.EXECUTION_FAILED

    def test_a_failing_result_write_never_hides_the_tool_outcome(self, log: EventLog):
        registry = _registry(add=_add)
        real_write = jsonl._write

        def write(fd: int, data: Any) -> int:
            if b"tool/result" in bytes(data):
                raise OSError("disk full")
            return real_write(fd, data)

        with bind(log, "c1"), mock.patch.object(jsonl, "_write", write):
            result = registry.execute(_call("add", a=2, b=2))
        assert result.success and result.result == 4
        assert unresolved_tool_calls(log.read("c1"))[0]["status"] == "unknown"


class TestTheObserverChangesNothingTheToolSees:
    def test_the_tool_receives_the_exact_parameters(self, log: EventLog):
        got: list[dict[str, Any]] = []

        def tool(params: dict) -> str:
            got.append(params)
            return "ok"

        registry = ToolRegistry().register_function(tool, name="d", parameter_schema={})
        sent = {"k": "v", "n": [1, 2]}
        with bind(log, "c1"):
            registry.execute(ToolCall(tool_name="d", parameters=sent))
        assert got == [{"k": "v", "n": [1, 2]}] and sent == {"k": "v", "n": [1, 2]}

    def test_the_observer_gets_a_read_only_view(self):
        class _Mutates:
            def begin_tool(self, *, arguments: Any, **_: Any) -> object:
                arguments["a"] = 99
                return None

            def end_tool(self, token: object, **_: Any) -> None:
                pass

        registry = _registry(add=_add)
        call = _call("add", a=1, b=1)
        token = install_tool_observer(_Mutates())  # type: ignore[arg-type]
        try:
            result = registry.execute(call)
        finally:
            reset_tool_observer(token)
        assert call.parameters == {"a": 1, "b": 1}
        assert not result.success  # the mutation was refused, so the call was refused

    def test_secret_looking_arguments_are_redacted_in_the_log_and_raw_for_the_tool(
        self, log: EventLog
    ):
        got: list[tuple[str, str]] = []
        registry = ToolRegistry().register_function(
            lambda api_key, q: got.append((api_key, q)) or "x",
            name="s",
            parameter_schema={},
        )
        with bind(log, "c1"):
            registry.execute(_call("s", api_key="sk-live-1234567890", q="hi"))
        (call, _) = log.read("c1")
        assert "sk-live" not in json.dumps(call)
        assert call["arguments"]["q"] == "hi"
        assert got == [
            ("sk-live-1234567890", "hi")
        ]  # the tool still gets the real value


class TestTheApprovalBoundaryIsUntouched:
    """The event is evidence, never a grant path (census D1): a denial writes no ``tool/call``
    and a granted call's events hold no grant."""

    @staticmethod
    def _handlers(registry: ToolRegistry):
        from fsm_llm.agents.handlers import AgentHandlers

        return AgentHandlers(registry, requires_approval=lambda call, ctx: True)

    def test_a_refused_gated_call_writes_no_tool_event(self, log: EventLog):
        ran: list[int] = []
        handlers = self._handlers(_registry(add=lambda a, b: ran.append(a) or 0))
        context = {
            ContextKeys.TOOL_NAME: "add",
            ContextKeys.TOOL_INPUT: {"a": 1, "b": 2},
        }
        with bind(log, "c1"):
            refusal = handlers.approval_refusal(context)
        assert refusal is not None and ran == []
        assert [e for e in log.read("c1") if e["type"].startswith("tool/")] == []

    def test_a_granted_call_is_logged_gated_and_the_grant_is_nowhere_in_the_log(
        self, log: EventLog
    ):
        from fsm_llm.agents.handlers.approval import approval_grant

        handlers = self._handlers(_registry(add=_add))
        tool_input = {"a": 1, "b": 2}
        context = {
            ContextKeys.TOOL_NAME: "add",
            ContextKeys.TOOL_INPUT: tool_input,
            ContextKeys.DRIVER_APPROVAL: approval_grant("add", tool_input),
        }
        assert handlers.approval_refusal(context) is None
        with bind(log, "c1"):
            handlers.execute_tool(context)
        call, done = (e for e in log.read("c1") if e["type"].startswith("tool/"))
        assert call["gated"] is True and done["status"] == "ok"
        text = json.dumps(list(log.read("c1")))
        assert (
            "_approval_granted" not in text and ContextKeys.DRIVER_APPROVAL not in text
        )


class TestTheOutcomeCodes:
    def test_a_raising_tool_is_failed_with_the_class_name_and_execution_failed(
        self, log: EventLog
    ):
        def boom(n: int) -> str:
            raise ValueError("secret text of the message")

        cid, (result,), _ = _run(_registry(boom=boom), [_call("boom", n=1)], log=log)
        assert result.error_code is ToolErrorCode.EXECUTION_FAILED
        _, done = _tool_events(log, cid)
        assert done["status"] == "failed" and done["error_type"] == "ValueError"
        assert done["error_code"] == "execution_failed"
        assert "secret text" not in json.dumps(done)  # a class name, never the message

    def test_a_bind_failure_is_invalid_args_and_the_text_is_unchanged(
        self, log: EventLog
    ):
        cid, (result,), _ = _run(
            _registry(add=_add), [_call("add", zzz=1, yyy=2)], log=log
        )
        assert result.error_code is ToolErrorCode.INVALID_ARGS
        assert result.error and result.error.startswith("Tool requires parameters")
        _, done = _tool_events(log, cid)
        assert (done["error_type"], done["error_code"]) == (
            "ToolArgumentError",
            "invalid_args",
        )

    def test_a_type_error_of_a_dict_style_body_is_execution_failed_not_invalid_args(
        self, log: EventLog
    ):
        def body(params: dict) -> str:
            raise TypeError("inside the body")

        registry = ToolRegistry().register_function(body, name="d", parameter_schema={})
        with bind(log, "c1"):
            result = registry.execute(_call("d", x=1))
        assert result.error_code is ToolErrorCode.EXECUTION_FAILED

    def test_an_unknown_name_logs_nothing_and_carries_unknown_tool(self, log: EventLog):
        cid, (result,), _ = _run(_registry(add=_add), [_call("nope")], log=log)
        assert not result.success and result.error_code is ToolErrorCode.UNKNOWN_TOOL
        assert _tool_events(log, cid) == []

    def test_a_result_longer_than_the_observation_cap_is_truncated(self, log: EventLog):
        big = "x" * (Defaults.MAX_OBSERVATION_LENGTH + 1)
        cid, (result,), _ = _run(_registry(big=lambda: big), [_call("big")], log=log)
        assert result.truncated is True
        assert _tool_events(log, cid)[1]["truncated"] is True
        assert (
            ToolResult(tool_name="t", success=True, result="short").truncated is False
        )
        assert ToolResult(tool_name="t", success=False, error=big).truncated is False

    def test_the_codes_are_the_closed_vocabulary_of_the_roadmap(self):
        assert [c.value for c in ToolErrorCode] == [
            "unknown_tool",
            "invalid_args",
            "execution_failed",
            "timeout",
            "denied",
            "sandbox_violation",
        ]


class TestATimeoutIsWrittenByTheCallerOnce:
    def test_a_timed_out_tool_is_unknown_once_and_a_late_return_writes_nothing(
        self, log: EventLog
    ):
        hold, finished = threading.Event(), threading.Event()

        def slow(n: int) -> str:
            hold.wait(timeout=30 * current_timing_factor())
            finished.set()
            return "late"

        registry = ToolRegistry().register_function(
            slow, name="slow", timeout_seconds=0.05
        )
        with bind(log, "c1"):
            result = registry.execute(_call("slow", n=1))
            before = list(log.read("c1"))
            hold.set()
            assert finished.wait(timeout=30 * current_timing_factor())
        assert result.timed_out and result.error_code is ToolErrorCode.TIMEOUT
        assert [e["type"] for e in before] == ["tool/call", "tool/result"]
        done = before[1]
        assert (done["status"], done["error_type"], done["error_code"]) == (
            "unknown",
            "ToolTimedOut",
            "timeout",
        )
        assert list(log.read("c1")) == before  # the late value changed nothing
        assert unresolved_tool_calls(before) == []  # the call is resolved, as unknown

    def test_the_scope_reaches_the_timeout_worker_but_the_worker_never_writes(
        self, log: EventLog
    ):
        writer_threads: set[int] = set()
        real_write = jsonl._write

        def write(fd: int, data: Any) -> int:
            writer_threads.add(threading.get_ident())
            return real_write(fd, data)

        registry = ToolRegistry().register_function(_add, name="add", timeout_seconds=5)
        with bind(log, "c1"), mock.patch.object(jsonl, "_write", write):
            registry.execute(_call("add", a=1, b=1))
        assert writer_threads == {threading.get_ident()}


class TestRetryAndCache:
    def test_a_retried_tool_logs_one_call_per_attempt(self, log: EventLog):
        attempts: list[int] = []

        def flaky(n: int) -> str:
            attempts.append(n)
            if len(attempts) < 3:
                raise RuntimeError("again")
            return "ok"

        registry = RetryingToolRegistry(max_retries=2)
        registry.register_function(
            flaky, name="flaky", annotations=READ_ONLY_ANNOTATIONS
        )
        with bind(log, "c1"):
            result = registry.execute(_call("flaky", n=1))
        assert result.success
        events = list(log.read("c1"))
        kinds = [(e["type"], e.get("status")) for e in events]
        assert kinds == [
            ("tool/call", None),
            ("tool/result", "failed"),
            ("tool/call", None),
            ("tool/result", "failed"),
            ("tool/call", None),
            ("tool/result", "ok"),
        ]
        ids = [e["call_id"] for e in events if e["type"] == "tool/call"]
        assert len(set(ids)) == 3

    def test_a_cache_hit_logs_nothing(self, log: EventLog):
        registry = CachingToolRegistry()
        registry.register_function(_add, name="add")
        with bind(log, "c1"):
            first = registry.execute(_call("add", a=1, b=2))
            second = registry.execute(_call("add", a=1, b=2))
        assert first.success and second.success and registry.cache_hits == 1
        assert [e["type"] for e in log.read("c1")] == ["tool/call", "tool/result"]


class TestParallelReact:
    def test_three_parallel_calls_log_three_pairs_one_call_id_each(self, log: EventLog):
        barrier = threading.Barrier(3, timeout=30 * current_timing_factor())

        def meet(n: int) -> str:
            barrier.wait()  # all three bodies run at once, on three threads
            return str(n)

        registry = ToolRegistry().register_function(meet, name="meet")
        agent = ParallelReactAgent(registry, config=AgentConfig(model="m"))
        calls = [{"tool_name": "meet", "tool_input": {"n": i}} for i in range(3)]
        with bind(log, "c1"):
            out = agent._dispatch_parallel({ContextKeys.TOOL_CALLS: calls})
        assert out
        events = list(log.read("c1"))
        starts = [e for e in events if e["type"] == "tool/call"]
        ends = [e for e in events if e["type"] == "tool/result"]
        assert len(starts) == len(ends) == 3
        assert {e["call_id"] for e in starts} == {e["call_id"] for e in ends}
        assert len({e["call_id"] for e in starts}) == 3
        assert {e["status"] for e in ends} == {"ok"}
        assert unresolved_tool_calls(events) == []


class TestTheFoldNeverMovesOnAToolResult:
    def test_a_late_result_of_frame_zero_after_a_push_keeps_the_child(
        self, log: EventLog
    ):
        llm = child.stack_llm([])
        api = API(child.chat_fsm(), llm_interface=llm, event_log=log)
        cid, _ = api.start_conversation()
        api.converse("Ann", cid)
        log.tool_begin(cid, name="slow", call_id="late", arguments={}, frame=0)
        api.push_fsm(cid, child.tool_fsm())
        stack = fold_stack(log.read(cid), log.get_json)
        assert stack is not None and len(stack.frames) == 1
        # the straggler thread of the frame-0 tool returns after the push
        log.tool_end(cid, name="slow", call_id="late", status="ok", frame=0)
        after = fold_stack(log.read(cid), log.get_json)
        assert after is not None and len(after.frames) == 1
        assert after.frames[0].child_conversation_id == (
            stack.frames[0].child_conversation_id
        )


class TestTheKilledChild:
    """Row 1.3 done-when: a process killed INSIDE a registry tool body leaves ``tool/call`` on
    disk (fsynced before the body), no ``tool/result``; ``API.resume`` reports it ``unknown``
    and never runs the tool again."""

    def test_a_kill_inside_a_registry_tool_is_unknown_on_resume_and_never_rerun(
        self, tmp_path: Path
    ) -> None:
        if not can_lock_in_child():
            pytest.skip("this platform cannot start a child that locks a file")
        root, ready, err = tmp_path / "log", tmp_path / "ready.json", tmp_path / "err"
        args = [CHILD_MODULE, str(root), str(ready), "registry"]
        with child_process(args, err) as proc:
            wait_ready(proc, ready, err)
            proc.kill()  # SIGKILL on POSIX, TerminateProcess on Windows
            proc.wait(timeout=60 * current_timing_factor())
        cid = json.loads(ready.read_text(encoding="utf-8"))["conversation_id"]
        events = list(read_events(root / "logs" / f"{cid}.jsonl"))
        assert [e["type"] for e in events][-1] == "tool/call"
        call = events[-1]
        assert (call["name"], call["side_effecting"], call["arguments"]) == (
            "charge",
            True,
            {"n": 5},
        )
        assert [c["call_id"] for c in unresolved_tool_calls(events)] == [
            call["call_id"]
        ]
        ran: list[int] = []
        spy = (
            create_handler("spy")
            .at(HandlerTiming.PRE_PROCESSING)
            .do(lambda ctx: ran.append(1) or {})
        )
        with reopen_after_kill(lambda: EventLog(root)) as reopened:
            api = API.from_definition(
                child.tool_fsm(),
                llm_interface=child.ChildLLM([]),
                event_log=reopened,
                handlers=[spy],
            )
            _, state = api.resume(cid)
            (unknown,) = state.metadata["unknown_tool_calls"]
            assert (unknown["call_id"], unknown["status"]) == (
                call["call_id"],
                "unknown",
            )
            assert ran == [] and api.llm_interface.call_history == []
            assert unresolved_tool_calls(reopened.read(cid)) == []
            results = [e for e in reopened.read(cid) if e["type"] == "tool/result"]
            assert [(r["call_id"], r["status"]) for r in results] == [
                (call["call_id"], "unknown")
            ]


class TestTheLayerStaysClean:
    """``tools`` and ``policy`` reach the log only through ``core.tool_observer`` (D-005)."""

    _ROOT = Path(__file__).resolve().parents[2] / "src" / "fsm_llm"

    @staticmethod
    def _imports(path: Path) -> set[str]:
        import ast

        names: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names.add(node.module)
            elif isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
        return names

    @pytest.mark.parametrize("package", ["tools", "policy"])
    def test_no_module_imports_runtime_or_llm(self, package: str):
        for path in (self._ROOT / package).rglob("*.py"):
            bad = {
                m
                for m in self._imports(path)
                if m.startswith(("fsm_llm.runtime", "fsm_llm.llm"))
            }
            assert not bad, (path.name, bad)

    def test_the_seam_imports_only_the_standard_library_and_the_logger(self):
        found = self._imports(self._ROOT / "core" / "tool_observer.py")
        assert {m for m in found if m.startswith("fsm_llm")} == {"fsm_llm.logging"}
