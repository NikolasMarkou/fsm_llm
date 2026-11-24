"""Approvals: ``approval/request`` and ``approval/outcome`` (plan 3f4530d3 step 4, decision D-007).

Every row asks through PRODUCT code: a real ``ReactAgent`` run whose driver asks between steps
inside ``API.event_scope``, ``HumanInTheLoop.request_approval`` called inside ``API.event_scope`` or
inside a turn (a handler), never through ``EventLog.append``. The rows pin: the order (the request
is on disk BEFORE the callback runs), the reasons, the log off versus on equality of every verdict
and grant, the best effort rule (a failed write changes no verdict, the callback runs once), that
no event can hold a grant or an internal key, and that the approver still sees the RAW arguments.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest import mock

import pytest

from fsm_llm import API, HandlerTiming, create_handler
from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.react import ReactAgent
from fsm_llm.constants import has_internal_prefix
from fsm_llm.core.tool_observer import (
    ToolHandle,
    begin_approval,
    end_approval,
    install_tool_observer,
    reset_tool_observer,
)
from fsm_llm.policy import ApprovalDeniedError, ApprovalRequest, HumanInTheLoop
from fsm_llm.runtime import EventLog, LogWriteError
from fsm_llm.runtime.stores import fold_stack
from fsm_llm.tools import ToolCall
from tests.test_fsm_llm import event_log_child as child
from tests.test_fsm_llm.event_log_support import tree_files
from tests.test_fsm_llm_agents.test_hitl_security import (
    _config,
    _ForgingLLM,
    _hitl,
    _registry,
)

APPROVAL_TYPES = ("approval/request", "approval/outcome")
_GRANT_KEY = ContextKeys.DRIVER_APPROVAL
_ALLOWED = {
    "v", "seq", "ts", "conversation_id", "frame", "type", "tool_name", "has_callback",
    "arguments", "arguments_ref", "argument_keys", "request_seq", "approved", "reason",
    "error_type",
}  # fmt: skip


@pytest.fixture
def log(tmp_path: Path):
    with EventLog(tmp_path / "root") as opened:
        yield opened


def _code_of(path: Path) -> str:
    """The text of *path* without its comment lines (an anchor may name an event)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _approval_events(log: EventLog, cid: str) -> list[dict[str, Any]]:
    return [e for e in log.read(cid) if e["type"] in APPROVAL_TYPES]


def _agent(
    log: EventLog | None,
    decide: Callable[[int], bool],
    asks: list[Any],
    executions: list[str],
) -> ReactAgent:
    """A ``ReactAgent`` that selects the gated tool ``danger`` and asks ``decide(n)``."""
    llm = _ForgingLLM(lambda: "danger", forge=lambda: False)
    return ReactAgent(
        tools=_registry(executions, "danger"),
        config=_config(),
        llm_interface=llm,
        hitl=_hitl(decide, asks),
        event_log=log,
    )


def _only_cid(log: EventLog) -> str:
    (cid,) = log.conversation_ids()
    return cid


def _pairs(events: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Each ``approval/request`` with its ``approval/outcome``, asserting they alternate."""
    assert [e["type"] for e in events] == list(APPROVAL_TYPES) * (len(events) // 2)
    return [(events[i], events[i + 1]) for i in range(0, len(events), 2)]


class TestAnAgentRunWritesTheAsk:
    def test_an_approved_ask_is_one_request_then_one_outcome(self, log: EventLog):
        asks: list[Any] = []
        executions: list[str] = []
        _agent(log, lambda n: True, asks, executions).run("do it")
        assert asks and "danger" in executions
        events = _approval_events(log, _only_cid(log))
        request, outcome = _pairs(events)[0]
        assert request["tool_name"] == "danger" and request["has_callback"] is True
        assert request["arguments"] == {"x": "1"} and request["frame"] == 0
        assert (
            outcome["request_seq"] == request["seq"] and outcome["seq"] > request["seq"]
        )
        assert outcome["approved"] is True and outcome["reason"] == "approved"
        assert "error_type" not in outcome
        assert len(_pairs(events)) == len(asks)

    def test_a_denied_ask_is_logged_as_denied_and_the_tool_never_runs(
        self, log: EventLog
    ):
        asks: list[Any] = []
        executions: list[str] = []
        _agent(log, lambda n: False, asks, executions).run("do it")
        assert asks and executions == []
        pairs = _pairs(_approval_events(log, _only_cid(log)))
        assert len(pairs) == len(asks)
        for request, outcome in pairs:
            assert outcome["approved"] is False and outcome["reason"] == "denied"
            assert outcome["request_seq"] == request["seq"]

    def test_the_callback_runs_after_the_request_is_on_disk(self, log: EventLog):
        seen: list[list[str]] = []

        def decide(n: int) -> bool:
            events = list(log.read(_only_cid(log)))
            seen.append([e["type"] for e in events if e["type"] in APPROVAL_TYPES])
            return True

        _agent(log, decide, [], []).run("do it")
        assert seen and seen[0] == ["approval/request"]
        # the callback of ask N sees N requests and N - 1 outcomes: never its own outcome
        assert all(
            s.count("approval/request") == s.count("approval/outcome") + 1 for s in seen
        )

    def test_the_driver_asks_in_the_log_scope_of_the_run_conversation(
        self, log: EventLog
    ):
        _agent(log, lambda n: True, [], []).run("do it")
        events = list(log.read(_only_cid(log)))
        kinds = [e["type"] for e in events]
        first = kinds.index("approval/request")
        # between two steps: after a turn/end and before the next turn/start, at frame 0
        assert kinds[first - 1] != "turn/start"
        assert all(e["frame"] == 0 for e in events if e["type"] in APPROVAL_TYPES)


def _scope_api(log: EventLog | None) -> tuple[API, str]:
    api = API(child.chat_fsm(), llm_interface=child.stack_llm([]), event_log=log)
    cid, _ = api.start_conversation()
    return api, cid


def _ask(
    api: API,
    cid: str,
    hitl: HumanInTheLoop,
    name: str = "send",
    **parameters: Any,
) -> Any:
    call = ToolCall(tool_name=name, parameters=parameters or {"to": "a"})
    with api.event_scope(cid):
        return hitl.request_approval(call, {"observations": [], "k": "v"})


class TestTheReasons:
    @pytest.mark.parametrize(
        ("answer", "approved", "reason"),
        [
            (True, True, "approved"),
            (False, False, "denied"),
            (None, False, "denied"),
            (1, False, "denied"),
            ("yes", False, "denied"),
        ],
    )
    def test_approved_mirrors_the_gate_is_true_rule(
        self, log: EventLog, answer: Any, approved: bool, reason: str
    ):
        api, cid = _scope_api(log)
        hitl = HumanInTheLoop(approval_callback=lambda request: answer)
        assert _ask(api, cid, hitl) is answer  # the verdict is returned as given
        ((_, outcome),) = _pairs(_approval_events(log, cid))
        assert outcome["approved"] is approved and outcome["reason"] == reason

    def test_a_slow_callback_is_a_timeout_and_a_denial(self, log: EventLog):
        api, cid = _scope_api(log)
        release = threading.Event()
        hitl = HumanInTheLoop(
            approval_callback=lambda request: release.wait(30) or True,
            approval_timeout=0.05,
        )
        try:
            assert _ask(api, cid, hitl) is False
        finally:
            release.set()
        ((request, outcome),) = _pairs(_approval_events(log, cid))
        assert outcome["reason"] == "timeout" and outcome["approved"] is False
        assert outcome["request_seq"] == request["seq"]
        # the worker thread that outlived the timeout wrote nothing
        assert len(_approval_events(log, cid)) == 2

    def test_no_callback_writes_the_pair_then_raises_as_before(self, log: EventLog):
        api, cid = _scope_api(log)
        hitl = HumanInTheLoop(approval_policy=lambda call, ctx: True)
        with pytest.raises(ApprovalDeniedError):
            _ask(api, cid, hitl)
        ((request, outcome),) = _pairs(_approval_events(log, cid))
        assert request["has_callback"] is False
        assert outcome["reason"] == "no_callback" and outcome["approved"] is False

    def test_a_raising_callback_is_an_error_outcome_and_the_error_is_unchanged(
        self, log: EventLog
    ):
        api, cid = _scope_api(log)

        def boom(request: ApprovalRequest) -> bool:
            raise KeyError("the secret text of the error")

        with pytest.raises(KeyError, match="the secret text"):
            _ask(api, cid, HumanInTheLoop(approval_callback=boom))
        ((_, outcome),) = _pairs(_approval_events(log, cid))
        assert outcome["reason"] == "error" and outcome["approved"] is False
        assert outcome["error_type"] == "KeyError"
        assert "secret" not in json.dumps(outcome)

    def test_an_exit_from_the_callback_is_an_error_outcome_too(self, log: EventLog):
        api, cid = _scope_api(log)

        def leave(request: ApprovalRequest) -> bool:
            raise SystemExit(3)

        with pytest.raises(SystemExit):
            _ask(api, cid, HumanInTheLoop(approval_callback=leave))
        ((_, outcome),) = _pairs(_approval_events(log, cid))
        assert outcome["error_type"] == "SystemExit" and outcome["reason"] == "error"

    def test_an_ask_inside_a_turn_lands_at_the_turn_frame_with_no_event_scope(
        self, log: EventLog
    ):
        hitl = HumanInTheLoop(approval_callback=lambda request: True)
        verdicts: list[Any] = []

        def ask(_: dict[str, Any]) -> dict[str, Any]:
            call = ToolCall(tool_name="send", parameters={"to": "a"})
            verdicts.append(hitl.request_approval(call, {}))
            return {}

        api = API.from_definition(
            child.tool_fsm(),
            llm_interface=child.ChildLLM(child.scripted_replies()),
            event_log=log,
            handlers=[
                create_handler("ask")
                .at(HandlerTiming.PRE_PROCESSING)
                .on_state("run_tools")
                .do(ask)
            ],
        )
        cid = child.start(api)
        for _ in range(3):
            api.advance(cid)
        assert verdicts == [True]
        kinds = [e["type"] for e in log.read(cid)]
        assert kinds.index("approval/request") > kinds.index("turn/start")
        assert kinds.index("approval/outcome") < len(kinds) - 1 - kinds[::-1].index(
            "turn/end"
        )


class TestALogFailureChangesNoVerdict:
    """Best effort (D-007): the log is evidence of the ask, never part of the decision."""

    @pytest.mark.parametrize("answer", [True, False])
    @pytest.mark.parametrize("broken", ["approval_request", "append"])
    def test_a_failed_write_leaves_the_verdict_and_asks_once(
        self, log: EventLog, answer: bool, broken: str
    ):
        api, cid = _scope_api(log)
        asks: list[ApprovalRequest] = []

        def callback(request: ApprovalRequest) -> bool:
            asks.append(request)
            return answer

        hitl = HumanInTheLoop(approval_callback=callback)
        real = EventLog.append

        def refuse_outcome(self: EventLog, conversation_id: str, kind: Any, **kw: Any):
            if str(getattr(kind, "value", kind)) == "approval/outcome":
                raise LogWriteError("disk full")
            return real(self, conversation_id, kind, **kw)

        patches = {
            "approval_request": mock.patch.object(
                EventLog, "approval_request", side_effect=LogWriteError("disk full")
            ),
            "append": mock.patch.object(EventLog, "append", refuse_outcome),
        }
        with patches[broken]:
            assert _ask(api, cid, hitl) is answer
        assert len(asks) == 1
        events = _approval_events(log, cid)
        assert [e["type"] for e in events] == (
            [] if broken == "approval_request" else ["approval/request"]
        )

    def test_a_failed_write_never_turns_an_approval_into_a_denial_or_an_error(
        self, log: EventLog
    ):
        api, cid = _scope_api(log)
        hitl = HumanInTheLoop(approval_callback=lambda request: True)
        with mock.patch.object(
            EventLog, "approval_request", side_effect=OSError("no space")
        ):
            assert _ask(api, cid, hitl) is True  # an OSError is swallowed too

    def test_an_agent_run_whose_log_cannot_take_approvals_still_decides_the_same(
        self, log: EventLog
    ):
        asks: list[Any] = []
        executions: list[str] = []
        with mock.patch.object(
            EventLog, "approval_request", side_effect=LogWriteError("x")
        ):
            _agent(log, lambda n: n == 0, asks, executions).run("do it")
        assert asks and "danger" in executions
        assert _approval_events(log, _only_cid(log)) == []

    def test_no_scope_means_no_event_and_no_error(self, log: EventLog):
        _, cid = _scope_api(log)
        hitl = HumanInTheLoop(approval_callback=lambda request: True)
        call = ToolCall(tool_name="send", parameters={"to": "a"})
        assert hitl.request_approval(call, {}) is True  # no `event_scope`
        assert _approval_events(log, cid) == []

    def test_the_scope_is_a_no_op_with_the_log_off_and_for_an_unknown_id(
        self, log: EventLog
    ):
        off_api, off_cid = _scope_api(None)
        hitl = HumanInTheLoop(approval_callback=lambda request: True)
        assert _ask(off_api, off_cid, hitl) is True
        api, cid = _scope_api(log)
        assert _ask(api, "no-such-conversation", hitl) is True
        assert _approval_events(log, cid) == []


class TestTheSeam:
    class _Spy:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        def begin_approval(self, **kw: Any) -> object:
            self.calls.append(("begin", kw))
            return "token"

        def end_approval(self, token: object, **kw: Any) -> None:
            self.calls.append(("end", (token, kw)))

    def test_with_no_observer_both_helpers_do_nothing(self):
        assert begin_approval("t", {}, has_callback=True) is None
        end_approval(None, approved=True, reason="approved")

    def test_the_observer_gets_a_read_only_view_and_the_token_back(self):
        spy = self._Spy()
        token = install_tool_observer(spy)  # type: ignore[arg-type]
        try:
            params = {"to": "a"}
            handle = begin_approval("send", params, has_callback=True)
            end_approval(handle, approved=True, reason="approved")
        finally:
            reset_tool_observer(token)
        (_, begin), (_, (given, end)) = spy.calls
        assert isinstance(begin["arguments"], MappingProxyType)
        with pytest.raises(TypeError):
            begin["arguments"]["to"] = "b"
        assert params == {"to": "a"} and given == "token"
        assert end == {"approved": True, "reason": "approved", "error_type": None}

    def test_an_observer_that_raises_is_swallowed_in_both_helpers(self):
        class Broken:
            def begin_approval(self, **kw: Any) -> object:
                raise RuntimeError("x")

            def end_approval(self, token: object, **kw: Any) -> None:
                raise RuntimeError("y")

        token = install_tool_observer(Broken())  # type: ignore[arg-type]
        try:
            assert begin_approval("t", {}, has_callback=False) is None
            end_approval(
                ToolHandle(Broken(), None),  # type: ignore[arg-type]
                approved=False,
                reason="denied",
            )
        finally:
            reset_tool_observer(token)

    def test_the_approver_still_sees_the_raw_arguments_the_log_copy_is_redacted(
        self, log: EventLog
    ):
        api, cid = _scope_api(log)
        seen: list[dict[str, Any]] = []

        def callback(request: ApprovalRequest) -> bool:
            seen.append(dict(request.parameters))
            return True

        hitl = HumanInTheLoop(approval_callback=callback)
        _ask(api, cid, hitl, to="a", api_key="sk-live-123456789012345678")
        assert seen == [{"to": "a", "api_key": "sk-live-123456789012345678"}]
        ((request, _),) = _pairs(_approval_events(log, cid))
        assert request["arguments"]["to"] == "a"
        assert "sk-live" not in json.dumps(request)


class TestNoEventCanHoldAGrantOrBeReadAsOne:
    def _flow_log(self, log: EventLog) -> str:
        asks: list[Any] = []
        executions: list[str] = []
        _agent(log, lambda n: n % 2 == 0, asks, executions).run("do it")
        assert asks and executions
        return _only_cid(log)

    def test_an_approval_event_holds_only_the_allowed_names_and_no_internal_key(
        self, log: EventLog
    ):
        events = _approval_events(log, self._flow_log(log))
        assert events
        for event in events:
            assert set(event) <= _ALLOWED
            assert not any(has_internal_prefix(key) for key in event)
            assert "context" not in json.dumps(event)

    def test_no_file_holds_a_grant_value_and_no_blob_or_approval_line_a_grant_key(
        self, log: EventLog
    ):
        self._flow_log(log)
        grant = {"tool_name": "danger", "parameters": {"x": "1"}}
        for path in tree_files(log.root):
            text = path.read_bytes().decode("utf-8", "replace")
            assert json.dumps(grant) not in text, path.name
            if path.suffix != ".jsonl":  # a blob
                assert _GRANT_KEY not in text, path.name
            for line in text.splitlines():
                if '"type":"approval/' in line:
                    assert _GRANT_KEY not in line and "grant" not in line

    def test_the_request_carries_no_context_summary_even_with_a_rich_context(
        self, log: EventLog
    ):
        api, cid = _scope_api(log)
        seen: list[ApprovalRequest] = []
        hitl = HumanInTheLoop(approval_callback=lambda r: seen.append(r) or True)
        call = ToolCall(tool_name="send", parameters={"to": "a"})
        context = {"k": "v-visible", _GRANT_KEY: {"tool_name": "send"}, "token": "t"}
        with api.event_scope(cid):
            hitl.request_approval(call, context)
        assert seen[0].context_summary == {"k": "v-visible"}  # the approver's view
        text = json.dumps(_approval_events(log, cid))
        assert "v-visible" not in text and _GRANT_KEY not in text

    def test_nothing_that_decides_reads_an_approval_event(self):
        root = Path(__file__).resolve().parents[2] / "src" / "fsm_llm"
        reads = re.compile(r"\bAPPROVAL_(REQUEST|OUTCOME)\b|approval/(request|outcome)")
        offenders = [
            str(path.relative_to(root))
            for folder in ("policy", "handlers", "agents")
            for path in (root / folder).rglob("*.py")
            if reads.search(_code_of(path))
        ]
        assert offenders == []


class TestPrivacy:
    def test_at_shape_the_request_keeps_the_names_and_no_value(self, tmp_path: Path):
        with EventLog(tmp_path / "s", privacy="shape") as shaped:
            api, cid = _scope_api(shaped)
            hitl = HumanInTheLoop(approval_callback=lambda request: True)
            _ask(api, cid, hitl, to="a@example.com", body="private text")
            ((request, outcome),) = _pairs(list(_approval_events(shaped, cid)))
        assert request["tool_name"] == "send" and request["has_callback"] is True
        assert request["argument_keys"] == ["body", "to"]
        assert "arguments" not in request and "arguments_ref" not in request
        assert "private text" not in json.dumps([request, outcome])
        assert outcome["approved"] is True and outcome["request_seq"] == request["seq"]

    def test_the_redaction_hook_sees_kind_arguments_and_cannot_change_the_ask(
        self, tmp_path: Path
    ):
        kinds: list[str] = []

        def hook(kind: Any, value: Any) -> Any:
            kinds.append(kind.value)
            return {"hooked": True} if kind.value == "arguments" else value

        with EventLog(tmp_path / "h", redact=hook) as hooked:
            api, cid = _scope_api(hooked)
            seen: list[Any] = []
            hitl = HumanInTheLoop(
                approval_callback=lambda r: seen.append(dict(r.parameters)) or True
            )
            _ask(api, cid, hitl, to="a")
            ((request, _),) = _pairs(list(_approval_events(hooked, cid)))
        assert "arguments" in kinds and request["arguments"] == {"hooked": True}
        assert seen == [{"to": "a"}]

    def test_big_arguments_go_to_a_blob_and_the_approver_still_gets_them_whole(
        self, log: EventLog
    ):
        api, cid = _scope_api(log)
        seen: list[Any] = []
        hitl = HumanInTheLoop(
            approval_callback=lambda r: seen.append(dict(r.parameters)) or True
        )
        _ask(api, cid, hitl, body="z" * 20_000)
        ((request, _),) = _pairs(_approval_events(log, cid))
        assert "arguments" not in request and request["arguments_ref"]
        assert log.get_json(request["arguments_ref"]) == {"body": "z" * 20_000}
        assert seen == [{"body": "z" * 20_000}]


class TestTheFoldAndTheScopeFrame:
    def test_the_scope_binds_the_top_frame(self, log: EventLog):
        api, cid = _scope_api(log)
        api.push_fsm(cid, child.tool_fsm())
        hitl = HumanInTheLoop(approval_callback=lambda request: True)
        _ask(api, cid, hitl)
        assert {e["frame"] for e in _approval_events(log, cid)} == {1}

    def test_a_late_ask_at_frame_zero_after_a_push_keeps_the_child(self, log: EventLog):
        api, cid = _scope_api(log)
        late = api.event_scope(cid)  # bound to frame 0 now
        api.push_fsm(cid, child.tool_fsm())
        hitl = HumanInTheLoop(approval_callback=lambda request: True)
        with late:
            hitl.request_approval(ToolCall(tool_name="send", parameters={}), {})
        assert {e["frame"] for e in _approval_events(log, cid)} == {0}
        stack = fold_stack(log.read(cid), log.get_json)
        assert stack is not None and len(stack.frames) == 1


_CLOCK = {"timestamp", "execution_time_ms"}


def _no_clock(value: Any) -> Any:
    """*value* without the wall-clock fields two identical runs never share."""
    if isinstance(value, dict):
        return {k: _no_clock(v) for k, v in value.items() if k not in _CLOCK}
    if isinstance(value, list):
        return [_no_clock(item) for item in value]
    return value


def _observed(log: EventLog | None, *, use_async: bool) -> dict[str, Any]:
    """Run the approving-then-denying agent flow and return everything a verdict touches: the
    requests the approver saw (RAW), the tool runs, every driver write to the context (grants
    included), the answer and the approval keys of the final context."""
    asks: list[ApprovalRequest] = []
    executions: list[str] = []
    writes: list[dict[str, Any]] = []
    real = API.update_context

    def record(self: API, conversation_id: str, update: dict[str, Any]) -> Any:
        writes.append(json.loads(json.dumps(update, default=repr)))
        return real(self, conversation_id, update)

    def decide(n: int) -> bool:
        return n % 2 == 0

    llm = _ForgingLLM(lambda: "danger", forge=lambda: False)
    agent = ReactAgent(
        tools=_registry(executions, "danger"),
        config=_config(),
        llm_interface=llm,
        hitl=HumanInTheLoop(
            approval_policy=lambda call, ctx: True,
            approval_callback=lambda r: asks.append(r) or decide(len(asks) - 1),
        ),
        event_log=log,
    )
    with mock.patch.object(API, "update_context", record):
        result = asyncio.run(agent.arun("do it")) if use_async else agent.run("do it")
    approval_keys = {
        key: value
        for key, value in result.final_context.items()
        if "approv" in key or key == ContextKeys.REFUSED_ACTIONS
    }
    return _no_clock(
        {
            "asks": [r.model_dump() for r in asks],
            "executions": executions,
            "writes": writes,
            "answer": result.answer,
            "success": result.success,
            "approval_keys": approval_keys,
        }
    )


class TestLogOnEqualsLogOff:
    @pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
    def test_every_verdict_grant_request_and_driver_write_is_equal(
        self, tmp_path: Path, use_async: bool
    ):
        off = _observed(None, use_async=use_async)
        assert off["asks"] and off["executions"]
        assert any(_GRANT_KEY in w for w in off["writes"])  # the grants are in play
        for privacy in ("values", "shape"):
            with EventLog(tmp_path / privacy, privacy=privacy) as on_log:
                on = _observed(on_log, use_async=use_async)
                assert _approval_events(on_log, _only_cid(on_log))
            assert on == off, privacy

    def test_the_comparison_can_fail(self, tmp_path: Path):
        """A planted difference (the log run asks once more) is caught by the same equality."""
        off = _observed(None, use_async=False)
        with EventLog(tmp_path / "x") as on_log:
            on = _observed(on_log, use_async=False)
        on["asks"] = on["asks"] + on["asks"][:1]
        assert on != off
