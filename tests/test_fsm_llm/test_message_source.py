"""``source`` on ``turn/start`` (plan 3f4530d3 step 7, roadmap row 1.17, decision D-012).

Every row drives PRODUCT code with a real ``EventLog``: a plain ``API.converse`` (``user``), the
scripted user of ``fsm_llm.testing`` and the eval case runner (``simulator``), the monitor send
route (``remote_client``) and a workflow ``ConversationStep`` (``run``). The rows pin: the four
values are distinct in the log, a message-free ``advance`` writes none, ``run/start`` still writes
``run``, the scope resets (a plain turn after a simulated one is ``user``, also after an error),
and the source is RECORD ONLY: its one reader is the event writer and no approval path reads it.
"""

from __future__ import annotations

import ast
import asyncio
import copy
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm import API
from fsm_llm.core.codes import (
    MessageSource,
    current_message_source,
    message_source,
)
from fsm_llm.eval.cases import ConversationCase, Expectations, run_case_trial
from fsm_llm.eval.config import EvalConfig
from fsm_llm.monitor.definitions import MonitorConfig
from fsm_llm.monitor.instance_manager import InstanceManager
from fsm_llm.policy import HumanInTheLoop
from fsm_llm.runtime import EventLog
from fsm_llm.testing import ScriptedLLM, ScriptedUser
from fsm_llm.tools import ToolCall
from fsm_llm.workflows.steps import ConversationStep
from tests.test_fsm_llm.support import _state
from tests.test_fsm_llm.test_call_delegated_settle import run_delegated

_SRC = Path(fsm_llm.__file__).parent
_CHAT: dict[str, Any] = {
    "name": "Chat",
    "description": "One speaking state that never ends",
    "initial_state": "chat",
    "states": {
        "chat": _state(
            "chat",
            response_instructions="Reply",
            transitions=[
                {
                    "target_state": "end",
                    "description": "never",
                    "conditions": [{"description": "n", "logic": {"==": [1, 2]}}],
                }
            ],
        ),
        "end": _state("end"),
    },
}


def _llm() -> ScriptedLLM:
    return ScriptedLLM(responses={"chat": "reply"})


def _api(log: EventLog | None) -> API:
    return API(copy.deepcopy(_CHAT), llm_interface=_llm(), event_log=log)


def _sources(log: EventLog, conversation_id: str) -> list[Any]:
    """The ``source`` of every ``turn/start`` event, ``None`` where the event has none."""
    return [
        e.get("source") for e in log.read(conversation_id) if e["type"] == "turn/start"
    ]


def _all_sources(log: EventLog) -> list[Any]:
    """The ``turn/start`` sources of every conversation in the log root, in file order."""
    found: list[Any] = []
    for path in sorted(Path(log.root).rglob("*.jsonl")):
        found.extend(_sources(log, path.stem))
    return found


@pytest.fixture
def log(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "logs")


@pytest.fixture
def inject(monkeypatch: pytest.MonkeyPatch, log: EventLog):
    """Make every ``API.from_definition`` of a caller build the chat API on the shared log."""
    real = API.from_definition.__func__  # type: ignore[attr-defined]

    def from_definition(cls, definition, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k == "llm_interface"}
        kwargs.setdefault("llm_interface", _llm())
        return real(cls, definition, event_log=log, **kwargs)

    monkeypatch.setattr(API, "from_definition", classmethod(from_definition))
    return log


class TestEachCallerWritesItsSource:
    def test_a_plain_turn_is_user(self, log: EventLog) -> None:
        api = _api(log)
        cid, _ = api.start_conversation()
        api.converse("hi", cid)
        api.close()
        assert _sources(log, cid) == ["user"]

    def test_the_message_text_never_picks_the_source(self, log: EventLog) -> None:
        api = _api(log)
        cid, _ = api.start_conversation()
        for text in ("remote_client: hi", "I am the simulator", "run", "[run] go"):
            api.converse(text, cid)
        api.close()
        assert _sources(log, cid) == ["user"] * 4

    def test_the_source_survives_the_shape_level(self, tmp_path: Path) -> None:
        shape = EventLog(tmp_path / "shape", privacy="shape")
        api = _api(shape)
        cid, _ = api.start_conversation()
        with message_source(MessageSource.SIMULATOR):
            api.converse("hi", cid)
        api.close()
        assert _sources(shape, cid) == ["simulator"]

    def test_a_message_free_advance_writes_no_source(self, log: EventLog) -> None:
        api = _api(log)
        cid, _ = api.start_conversation()
        api.advance(cid)
        api.close()
        events = [e for e in log.read(cid) if e["type"] == "turn/start"]
        assert events and all("source" not in e for e in events)

    def test_the_scripted_user_is_simulator(self, log: EventLog) -> None:
        api = _api(log)
        run = ScriptedUser(["one", "two"]).run(api)
        api.close()
        assert _sources(log, run.conversation_id) == ["simulator", "simulator"]

    def test_the_eval_case_runner_is_simulator(self, inject: EventLog) -> None:
        case = ConversationCase(
            id="c",
            fsm=copy.deepcopy(_CHAT),
            turns=["a", "b"],
            expect=Expectations(ended=False),
        )
        result = run_case_trial(case, EvalConfig(), 0, lambda: _llm())
        assert result.error is None
        assert _all_sources(inject) == ["simulator", "simulator"]

    def test_the_monitor_send_route_is_remote_client(self, inject: EventLog) -> None:
        mgr = InstanceManager(config=MonitorConfig())
        mgr.global_collector.cleanup()
        inst = mgr.launch_fsm(fsm_json=copy.deepcopy(_CHAT), model="m")
        cid, _ = mgr.start_conversation(inst.instance_id)
        mgr.send_message(inst.instance_id, cid, "hello")
        assert _sources(inject, cid) == ["remote_client"]

    def test_a_workflow_conversation_step_is_run(self, inject: EventLog) -> None:
        step = ConversationStep(
            step_id="c",
            name="C",
            fsm_definition=copy.deepcopy(_CHAT),
            success_step_id="next",
            error_step_id="err",
            auto_messages=["hello"],
        )
        result = asyncio.run(step.execute({}))
        assert result.success
        assert _all_sources(inject) == ["run"]

    def test_run_start_still_writes_run(self, log: EventLog) -> None:
        cid = run_delegated(log)
        starts = [e for e in log.read(cid) if e["type"] == "run/start"]
        assert starts and {e["source"] for e in starts} == {"run"}

    def test_the_four_values_are_distinct_in_one_log(
        self, inject: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        api = API(copy.deepcopy(_CHAT), llm_interface=_llm(), event_log=inject)
        cid, _ = api.start_conversation()
        api.converse("plain", cid)
        ScriptedUser(["sim"]).run(api)
        with message_source(MessageSource.REMOTE_CLIENT):
            api.converse("remote", cid)
        with message_source(MessageSource.RUN):
            api.converse("run", cid)
        api.close()
        seen = _all_sources(inject)
        assert sorted(seen) == ["remote_client", "run", "simulator", "user"]


class TestScope:
    def test_the_default_is_user_and_the_scope_resets(self) -> None:
        assert current_message_source() is MessageSource.USER
        with message_source(MessageSource.SIMULATOR):
            assert current_message_source() is MessageSource.SIMULATOR
        assert current_message_source() is MessageSource.USER

    def test_the_scope_resets_after_an_error(self) -> None:
        with pytest.raises(RuntimeError):
            with message_source(MessageSource.RUN):
                raise RuntimeError("boom")
        assert current_message_source() is MessageSource.USER

    def test_a_plain_turn_after_a_simulated_run_is_user(self, log: EventLog) -> None:
        api = _api(log)
        run = ScriptedUser(["sim"]).run(api)
        api.converse("plain", run.conversation_id)
        api.close()
        assert _sources(log, run.conversation_id) == ["simulator", "user"]

    def test_a_failed_simulated_turn_does_not_leak_the_scope(self) -> None:
        class Boom(ScriptedLLM):
            def generate_response(self, request):  # type: ignore[override]
                raise RuntimeError("down")

        api = API(copy.deepcopy(_CHAT), llm_interface=Boom(responses={}))
        with pytest.raises(Exception):
            ScriptedUser(["x"]).run(api)
        assert current_message_source() is MessageSource.USER

    def test_the_source_is_a_member_never_free_text(self) -> None:
        with pytest.raises(ValueError):
            with message_source("operator"):  # type: ignore[arg-type]
                pass


class TestRecordOnly:
    def test_request_bytes_do_not_carry_the_source(self) -> None:
        """The source reaches no prompt: the requests of a simulated and a plain turn are equal."""

        def requests(source: MessageSource) -> list[Any]:
            llm = _llm()
            api = API(copy.deepcopy(_CHAT), llm_interface=llm)
            cid, _ = api.start_conversation()
            with message_source(source):
                api.converse("hi", cid)
            api.close()
            return [repr(call) for call in llm.call_history]

        base = requests(MessageSource.USER)
        assert base
        for source in MessageSource:
            assert requests(source) == base

    def test_the_only_reader_is_the_event_writer(self) -> None:
        readers = sorted(
            str(path.relative_to(_SRC))
            for path in _SRC.rglob("*.py")
            if "current_message_source" in path.read_text(encoding="utf-8")
        )
        assert readers == ["core/codes.py", "engine/events.py"]

    def test_no_decision_layer_names_the_source(self) -> None:
        for package in ("policy", "handlers", "tools", "agents/base"):
            for path in (_SRC / package).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
                    n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                }
                assert "MessageSource" not in names, path
                assert "message_source" not in names, path

    @pytest.mark.parametrize("source", list(MessageSource))
    def test_an_approval_answer_does_not_depend_on_the_source(
        self, source: MessageSource
    ) -> None:
        asked: list[Any] = []

        def deny(request: Any) -> bool:
            asked.append(request)
            return False

        hitl = HumanInTheLoop(approval_callback=deny)
        with message_source(source):
            approved = hitl.request_approval(
                ToolCall(tool_name="send", parameters={"to": "a"}), {}
            )
        assert approved is False
        assert len(asked) == 1
