"""The engine producers of the event log (plan 21e6f00d step 8): the seven model call sites, the
streamed reply, the executor and transition events, the stack events and the out-of-turn context
record.

Two model paths: ``ScriptedLLM`` (an engine-path double that never reaches
``LiteLLMInterface._send``, so the engine writes the request and reply) and a real
``LiteLLMInterface`` with the patched ``fsm_llm.llm.providers.litellm.completion`` (the provider
seam writes them). Bytes, counts and sequences, never timing. Unverified until the Windows run:
nothing here opens a file itself (the files are ``stores.jsonl``'s and ``blobs``').

Not produced yet (``NOT_YET_PRODUCED``, declared in ``DECLARED_TYPES``): ``tool/call``,
``approval/*``, ``extraction``, ``extraction/miss``, ``intent/out_of_scope``, ``handler``,
``executable/resolve`` and ``compaction/epoch``. ``TestTheProductPathMatrix`` runs one product
scenario per produced type and bans a direct log write in this module. ``decision`` (the budget
stop, plan 5a6a1f80 step 7) is produced; its tests are in ``test_turn_budgets.py``.
"""

from __future__ import annotations

import ast
import contextvars
import copy
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import fsm_llm.engine
from fsm_llm import API
from fsm_llm.core.definitions import (
    FSMError,
    ResponseGenerationRequest,
    TransitionEvaluationResult,
)
from fsm_llm.engine.turn.events import _priority
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.runtime import (
    DECLARED_TYPES,
    PRODUCED_TYPES,
    EventLog,
    EventLogError,
    EventType,
    LogWriteError,
    bind,
    current_scope,
    llm_call,
)
from fsm_llm.runtime.stores import jsonl
from fsm_llm.testing import ScriptedLLM, final_reply
from tests.test_fsm_llm import event_log_child as crash_child
from tests.test_fsm_llm.event_log_support import (
    _chunk,
    _reply,
    _Wrapper,
    tree_files,
)
from tests.test_fsm_llm.support import _state
from tests.test_fsm_llm.test_call_delegated_settle import run_delegated
from tests.test_fsm_llm.test_event_log_tools import (
    _add as _tools_add,
)
from tests.test_fsm_llm.test_event_log_tools import (
    _call as _tool_call,
)
from tests.test_fsm_llm.test_event_log_tools import (
    _registry as _tools_registry,
)
from tests.test_fsm_llm.test_event_log_tools import (
    _run as _run_registry_tool,
)

COMPLETION = "fsm_llm.llm.providers.litellm.completion"
_ADD: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        },
    },
}
_FINAL_KIND = [
    {
        "description": "The model answered",
        "requires_context_keys": ["completion_result"],
        "logic": {"==": [{"var": "completion_result.kind"}, "final"]},
    }
]
_FSM: dict[str, Any] = {
    "name": "Producers",
    "description": "Every producer: field and bulk extraction, a tie, a stream, a completion",
    "initial_state": "intake",
    "states": {
        "intake": _state(
            "intake",
            required_context_keys=["name"],
            response_instructions="Greet the user",
            transitions=[
                {"target_state": "left", "description": "go left"},
                {"target_state": "alt", "description": "go alt"},
            ],
        ),
        "left": _state(
            "left",
            extraction_instructions="Extract the mood",
            response_instructions="Left reply",
            transitions=[{"target_state": "call_model", "description": "on"}],
        ),
        "alt": _state("alt", response_instructions="Alt reply"),
        "call_model": _state(
            "call_model",
            response_instructions="Working on it",
            completion={"tools": [_ADD], "instructions": "You add."},
            transitions=[
                {
                    "target_state": "done",
                    "description": "answered",
                    "conditions": _FINAL_KIND,
                }
            ],
        ),
        "done": _state("done", response_instructions="Done reply"),
    },
}
_CHILD: dict[str, Any] = {
    "name": "Child",
    "description": "A pushed child",
    "initial_state": "hello",
    "states": {
        "hello": _state(
            "hello",
            response_instructions="Child hello",
            transitions=[{"target_state": "bye", "description": "go"}],
        ),
        "bye": _state("bye", response_instructions="Child bye"),
    },
}
_SILENT_CHILD: dict[str, Any] = {
    "name": "SilentChild",
    "description": "A pushed child that says nothing, so its push makes no model call",
    "initial_state": "quiet",
    "states": {"quiet": _state("quiet")},
}
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
_CLASSIFY = '{"reasoning": "r", "intent": "left", "confidence": 0.9}'

#: (frame, type) per operation of ``_run``: the whole scenario, the expected table.
_TABLE: dict[str, list[tuple[int, str]]] = {
    "start": [(0, "llm/request"), (0, "llm/response"), (0, "context/record")],
    "turn": [
        (0, "turn/start"),
        (0, "executor/start"),
        (0, "llm/request"),
        (0, "llm/response"),
        (0, "executor/result"),
        (0, "llm/request"),
        (0, "llm/response"),
        (0, "transition"),
        (0, "llm/request"),
        (0, "llm/response"),
        (0, "turn/end"),
    ],
    "push": [
        (1, "llm/request"),
        (1, "llm/response"),
        (0, "stack/push"),
    ],
    "child_turn": [
        (1, "turn/start"),
        (1, "executor/start"),
        (1, "executor/result"),
        (1, "transition"),
        (1, "llm/request"),
        (1, "llm/response"),
        (1, "turn/end"),
    ],
    "pop": [(0, "stack/pop")],
    "stream": [
        (0, "turn/start"),
        (0, "executor/start"),
        (0, "llm/request"),
        (0, "llm/response"),
        (0, "executor/result"),
        (0, "transition"),
        (0, "llm/request"),
        (0, "llm/response"),
        (0, "turn/end"),
    ],
    "update": [(0, "context/record")],
    "advance": [
        (0, "turn/start"),
        (0, "executor/start"),
        (0, "llm/request"),
        (0, "llm/response"),
        (0, "executor/result"),
        (0, "transition"),
        (0, "llm/request"),
        (0, "llm/response"),
        (0, "turn/end"),
    ],
}


def _script() -> ScriptedLLM:
    return ScriptedLLM(
        [final_reply(_CLASSIFY), final_reply("4")],
        extraction_data={"name": "Ann", "mood": "happy"},
        responses={
            "intake": "hi",
            "left": "left reply",
            "alt": "alt reply",
            "call_model": "working",
            "done": "all done",
            "hello": "child hello",
            "bye": "child bye",
        },
        stream_chunks=["wor", "king"],
    )


class _Failing(ScriptedLLM):
    """Answers until ``down`` is set, then every Pass-2 call raises a provider-like error."""

    down = False

    def generate_response(self, request: Any) -> Any:
        if self.down:
            raise RuntimeError("PRIVATE provider message")
        return super().generate_response(request)


def _run(
    log: EventLog | None, llm: ScriptedLLM | None = None, **api_kwargs: Any
) -> tuple[API, str, dict[str, tuple[int, int]]]:
    """Run the whole scenario; return the API, the root id and the ``seq`` span of each op."""
    api = API(
        copy.deepcopy(_FSM),
        llm_interface=llm or _script(),
        event_log=log,
        **api_kwargs,
    )
    spans: dict[str, tuple[int, int]] = {}
    last = 0

    def mark(name: str, cid: str | None) -> None:
        nonlocal last
        if log is None or cid is None:
            return
        top = max((e["seq"] for e in log.read(cid)), default=0)
        spans[name] = (last + 1, top)
        last = top

    cid, _ = api.start_conversation()
    mark("start", cid)
    assert api.converse("Ann", cid) == "left reply"
    mark("turn", cid)
    api.push_fsm(cid, copy.deepcopy(_CHILD))
    mark("push", cid)
    assert api.converse("hello child", cid) == "child bye"
    mark("child_turn", cid)
    api.pop_fsm(cid)
    mark("pop", cid)
    assert "".join(api.converse_stream("go", cid)) == "working"
    mark("stream", cid)
    api.update_context(
        cid,
        {
            "_completion_messages": [{"role": "user", "content": "2+2?"}],
            "note": "x",
        },
    )
    mark("update", cid)
    assert api.advance(cid).response == "all done"
    mark("advance", cid)
    return api, cid, spans


def _events(log: EventLog, cid: str, *types: str) -> list[dict[str, Any]]:
    return [e for e in log.read(cid) if not types or e["type"] in types]


class TestTheExpectedTable:
    def test_every_operation_writes_exactly_its_events_in_order(self, log: EventLog):
        _, cid, spans = _run(log)
        assert list(spans) == list(_TABLE)
        events = _events(log, cid)
        for name, (first, last) in spans.items():
            got = [(e["frame"], e["type"]) for e in events if first <= e["seq"] <= last]
            assert got == _TABLE[name], name

    def test_seq_is_strictly_increasing_and_contiguous(self, log: EventLog):
        _, cid, _ = _run(log)
        seqs = [e["seq"] for e in log.read(cid)]
        assert seqs == list(range(1, len(seqs) + 1)) and len(seqs) == 44

    def test_every_logged_request_has_one_outcome_and_the_counts_match(
        self, log: EventLog
    ):
        llm = _script()
        _, cid, _ = _run(log, llm)
        requests = _events(log, cid, "llm/request")
        outcomes = _events(log, cid, "llm/response", "provider/error")
        assert [o["request_seq"] for o in outcomes] == [r["seq"] for r in requests]
        made = len(llm.call_history)
        # every model call is logged, the root greeting included (step 2 of plan 3f4530d3)
        assert made == len(requests) == 10
        assert {r["source"] for r in requests} == {"engine"}
        assert {r["call_type"] for r in requests} == {
            "field_extraction",
            "classification",
            "response_generation",
            "greeting",
            "data_extraction",
            "response_stream",
            "completion",
        }

    def test_the_log_is_off_by_default_and_the_scenario_is_the_same(
        self, tmp_path: Path
    ):
        llm = _script()
        api, cid, _ = _run(None, llm)
        assert current_scope() is None
        assert not list(tmp_path.rglob("*.jsonl"))
        assert api.get_data(cid)["name"] == "Ann"
        assert len(llm.call_history) == 10


class TestTheProducers:
    def test_executor_events_name_the_kind_the_state_and_key_names_only(
        self, log: EventLog
    ):
        _, cid, _ = _run(log)
        starts = _events(log, cid, "executor/start")
        results = _events(log, cid, "executor/result")
        assert [(e["kind"], e["state"]) for e in starts] == [
            ("conversation", "intake"),
            ("conversation", "hello"),
            ("conversation", "left"),
            ("completion", "call_model"),
        ]
        assert [(e["extracted_keys"], e["extraction_failed"]) for e in results] == [
            (["name"], False),
            ([], False),
            (["mood"], False),
            (["completion_result"], False),
        ]
        assert "Ann" not in repr(list(log.read(cid)))

    def test_transition_events_carry_the_outcome_the_target_and_the_priority(
        self, log: EventLog
    ):
        _, cid, _ = _run(log)
        got = [
            (
                e["frame"],
                e["from_state"],
                e["to_state"],
                e["outcome"],
                e["executed"],
                e.get("priority"),
                e.get("candidates"),
            )
            for e in _events(log, cid, "transition")
        ]
        assert got == [
            (0, "intake", "left", "ambiguous", True, 100, ["alt", "left"]),
            (1, "hello", "bye", "deterministic", True, 100, None),
            (0, "left", "call_model", "deterministic", True, 100, None),
            (0, "call_model", "done", "deterministic", True, 100, None),
        ]

    def test_a_blocked_evaluation_is_a_transition_event_that_stays(self, log: EventLog):
        fsm = copy.deepcopy(_FSM)
        fsm["states"]["left"]["transitions"][0]["conditions"] = [
            {"description": "never", "logic": {"==": [1, 2]}}
        ]
        api = API(fsm, llm_interface=_script(), event_log=log)
        cid, _ = api.start_conversation()
        api.converse("Ann", cid)
        api.converse("again", cid)
        last = _events(log, cid, "transition")[-1]
        assert (last["from_state"], last["to_state"], last["outcome"]) == (
            "left",
            None,
            "blocked",
        )
        assert last["executed"] is False and "priority" not in last

    def test_a_deterministic_priority_is_omitted_when_the_target_has_two_edges(self):
        state = SimpleNamespace(
            transitions=[
                SimpleNamespace(target_state="x", priority=10),
                SimpleNamespace(target_state="x", priority=20),
            ]
        )
        one = SimpleNamespace(
            transitions=[SimpleNamespace(target_state="x", priority=30)]
        )
        det = SimpleNamespace(result_type=TransitionEvaluationResult.DETERMINISTIC)
        assert _priority(state, det, "x") == {}  # type: ignore[arg-type]
        assert _priority(one, det, "x") == {"priority": 30}  # type: ignore[arg-type]

    def test_the_stack_events_name_the_child_and_the_frame(self, log: EventLog):
        api = API(copy.deepcopy(_FSM), llm_interface=_script(), event_log=log)
        cid, _ = api.start_conversation()
        api.push_fsm(cid, copy.deepcopy(_CHILD))
        child_id = api.get_sub_conversation_id(cid)
        api.pop_fsm(cid, context_to_return={"result": 1})
        push, pop = _events(log, cid, "stack/push", "stack/pop")
        assert (push["frame"], push["depth"], push["child_conversation_id"]) == (
            0,
            2,
            child_id,
        )
        assert log.get_json(push["definition_ref"])["name"] == "Child"
        assert push["fsm_id"].startswith("fsm_Child")
        assert (pop["frame"], pop["depth"], pop["child_conversation_id"]) == (
            0,
            1,
            child_id,
        )
        assert (pop["merge_strategy"], pop["merged_key_count"]) == ("update", 1)
        assert pop["fsm_id"] == push["fsm_id"]

    def test_a_pushed_childs_greeting_is_logged_under_its_own_frame(
        self, log: EventLog
    ):
        _, cid, spans = _run(log)
        first, last = spans["push"]
        pair = [e for e in _events(log, cid) if first <= e["seq"] <= last]
        assert [(e["frame"], e["call_type"]) for e in pair[:2]] == [
            (1, "greeting"),
            (1, "greeting"),
        ]

    def test_the_root_greeting_is_logged_before_the_start_record_that_holds_its_text(
        self, log: EventLog
    ):
        """Step 2 of plan 3f4530d3 (was D-021: the id was minted inside the manager): the id is
        minted by the API and the greeting runs in the scope of frame 0, as a turn's calls do."""
        llm = _script()
        api = API(copy.deepcopy(_FSM), llm_interface=llm, event_log=log)
        cid, greeting = api.start_conversation()
        assert greeting == "hi" and len(llm.call_history) == 1
        events = list(log.read(cid))
        assert [(e["frame"], e["type"]) for e in events] == [
            (0, "llm/request"),
            (0, "llm/response"),
            (0, "context/record"),
        ]
        request, response, record = events
        assert (request["source"], request["call_type"]) == ("engine", "greeting")
        assert response["request_seq"] == request["seq"]
        state = log.get_json(record["state_ref"])
        assert state["conversation_history"] == [{"system": "hi"}]

    def test_an_update_context_record_carries_only_the_declared_internal_keys(
        self, log: EventLog
    ):
        api = API(copy.deepcopy(_FSM), llm_interface=_script(), event_log=log)
        cid, _ = api.start_conversation()
        transcript = [{"role": "user", "content": "2+2?"}]
        api.update_context(
            cid,
            {
                "_completion_messages": transcript,
                "_approval_granted": True,
                "_run_output_keys": ["x"],
                "_counter": 7,
                "note": "x",
                "gone": None,
                "api_key": "sk-SECRET-corpus-0001",
            },
        )
        (record,) = _events(log, cid, "context/record")[1:]
        assert record["source"] == "update_context"
        assert record["keys"] == sorted(
            [
                "_completion_messages",
                "_approval_granted",
                "_run_output_keys",
                "_counter",
                "note",
                "gone",
                "api_key",
            ]
        )
        delta = log.get_json(record["delta_ref"])
        assert delta["internal"] == {"_completion_messages": transcript}
        assert delta["context"]["note"] == "x" and delta["context"]["gone"] is None
        assert "_approval_granted" not in delta["context"]
        assert delta["context"]["api_key"] == "<redacted>"
        assert set(delta["context"]) == {"note", "gone", "api_key"}

    def test_an_update_context_in_a_pushed_frame_carries_its_frame(self, log: EventLog):
        api = API(copy.deepcopy(_FSM), llm_interface=_script(), event_log=log)
        cid, _ = api.start_conversation()
        api.push_fsm(cid, copy.deepcopy(_CHILD))
        api.update_context(cid, {"note": "child"})
        record = _events(log, cid, "context/record")[-1]
        assert (record["frame"], record["source"]) == (1, "update_context")

    def test_a_failing_provider_writes_the_error_type_only(self, log: EventLog):
        failing = _Failing(responses={"chat": "hi"})
        api = API(copy.deepcopy(_CHAT), llm_interface=failing, event_log=log)
        cid, _ = api.start_conversation()
        failing.down = True
        with pytest.raises(FSMError):
            api.converse("Ann", cid)
        (error,) = _events(log, cid, "provider/error")
        assert error["error_type"] == "RuntimeError" and error["source"] == "engine"
        assert "PRIVATE" not in "".join(
            p.read_text(encoding="utf-8") for p in log_files(log)
        )


def log_files(log: EventLog) -> list[Path]:
    return tree_files(log.root)


def _types_of(log: EventLog, cid: str, after: int = 0) -> set[str]:
    return {e["type"] for e in log.read(cid) if e["seq"] > after}


def _scenario_turns(log: EventLog) -> set[str]:
    """``start_conversation``, ``converse``, ``push_fsm``, ``pop_fsm``, a stream, ``update_context``
    and ``advance`` on a scripted engine."""
    _, cid, _ = _run(log)
    return _types_of(log, cid)


def _scenario_failing_provider(log: EventLog) -> set[str]:
    failing = _Failing(responses={"intake": "hi"})
    api = API(copy.deepcopy(_FSM), llm_interface=failing, event_log=log)
    cid, _ = api.start_conversation()
    failing.down = True
    with pytest.raises(FSMError):
        api.converse("Ann", cid)
    return _types_of(log, cid)


def _scenario_spent_budget(log: EventLog) -> set[str]:
    capped = API(
        copy.deepcopy(_CHAT),
        llm_interface=ScriptedLLM(responses={"chat": "hi"}),
        event_log=log,
        max_steps=1,
    )
    cid, _ = capped.start_conversation()
    capped.converse("one", cid)
    with pytest.raises(FSMError):
        capped.converse("two", cid)
    return _types_of(log, cid)


def _scenario_delegated_call(log: EventLog) -> set[str]:
    return _types_of(log, run_delegated(log))


def _scenario_resume_of_a_tool_in_flight(log: EventLog) -> set[str]:
    """A log a kill left with a ``tool/call`` and no result, resumed: only what ``API.resume``
    writes counts (the ``unknown`` ``tool/result``), never the setup."""
    api = API.from_definition(
        crash_child.tool_fsm(), llm_interface=ScriptedLLM([]), event_log=log
    )
    cid = crash_child.start(api)
    crash_child.leave_a_tool_in_flight(log, cid)
    last = max(e["seq"] for e in log.read(cid))
    log.release(cid)  # the writer of the killed process is gone
    resumed = API.from_definition(
        crash_child.tool_fsm(), llm_interface=ScriptedLLM([]), event_log=log
    )
    resumed.resume(cid)
    return _types_of(log, cid, after=last)


def _scenario_registry_tool(log: EventLog) -> set[str]:
    """A ``ToolRegistry`` tool run by a handler inside a logged turn: ``tool/call`` and its
    ``tool/result`` come from ``ToolRegistry.execute`` through ``core.tool_observer``."""
    cid, _, _ = _run_registry_tool(
        _tools_registry(add=_tools_add), [_tool_call("add", a=2, b=3)], log=log
    )
    return _types_of(log, cid)


#: Per type, the product scenario that writes it: through ``API``, ``API.resume`` or a delegated
#: run, never through ``EventLog.tool_call`` or ``EventLog.append`` (an AST row below bans both
#: in this module). Later steps move a type from ``NOT_YET_PRODUCED`` to this table.
_PRODUCT_SCENARIOS: dict[EventType, Callable[[EventLog], set[str]]] = {
    EventType.TURN_START: _scenario_turns,
    EventType.TURN_END: _scenario_turns,
    EventType.LLM_REQUEST: _scenario_turns,
    EventType.LLM_RESPONSE: _scenario_turns,
    EventType.TRANSITION: _scenario_turns,
    EventType.EXECUTOR_START: _scenario_turns,
    EventType.EXECUTOR_RESULT: _scenario_turns,
    EventType.CONTEXT_RECORD: _scenario_turns,
    EventType.STACK_PUSH: _scenario_turns,
    EventType.STACK_POP: _scenario_turns,
    EventType.PROVIDER_ERROR: _scenario_failing_provider,
    EventType.DECISION: _scenario_spent_budget,
    EventType.RUN_START: _scenario_delegated_call,
    EventType.RUN_END: _scenario_delegated_call,
    EventType.TOOL_CALL: _scenario_registry_tool,
    EventType.TOOL_RESULT: _scenario_resume_of_a_tool_in_flight,
}

#: Declared and with no product producer yet: the approval pair, the extraction family, handler, executable/resolve; and
#: compaction/epoch, which needs roadmap row 2.5 and is out of plan 3f4530d3.
NOT_YET_PRODUCED = frozenset(
    {
        EventType.APPROVAL_REQUEST,
        EventType.APPROVAL_OUTCOME,
        EventType.EXTRACTION,
        EventType.EXTRACTION_MISS,
        EventType.INTENT_OUT_OF_SCOPE,
        EventType.HANDLER,
        EventType.EXECUTABLE_RESOLVE,
        EventType.COMPACTION_EPOCH,
    }
)
_DIRECT_WRITERS = {"tool_call", "append_batch"}
_LOG_NAMES = {"log", "_event_log", "event_log"}


def _direct_writes(source: str) -> list[int]:
    """Line numbers of a direct write to a log object: ``.tool_call(`` and ``.append_batch(``
    on anything, ``.append(`` on a receiver named like a log (a list's ``append`` is fine)."""
    found: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        attr, receiver = node.func.attr, node.func.value
        tail = (
            receiver.id
            if isinstance(receiver, ast.Name)
            else receiver.attr
            if isinstance(receiver, ast.Attribute)
            else ""
        )
        if attr in _DIRECT_WRITERS or (attr == "append" and tail in _LOG_NAMES):
            found.append(node.lineno)
    return found


class TestTheProductPathMatrix:
    """``PRODUCED_TYPES`` is what product code writes: one scenario per type, none seeded by a
    helper call of the test (plan 3f4530d3 step 2; roadmap row 1.1)."""

    def test_the_table_has_one_row_per_produced_type(self):
        assert set(_PRODUCT_SCENARIOS) == PRODUCED_TYPES
        assert PRODUCED_TYPES.isdisjoint(NOT_YET_PRODUCED)
        assert DECLARED_TYPES - PRODUCED_TYPES == NOT_YET_PRODUCED

    @pytest.mark.parametrize("kind", sorted(PRODUCED_TYPES, key=lambda t: t.value))
    def test_each_produced_type_is_written_by_its_product_scenario(
        self, log: EventLog, kind: EventType
    ):
        assert kind.value in _PRODUCT_SCENARIOS[kind](log)

    def test_the_union_of_the_scenarios_is_exactly_the_produced_set(
        self, tmp_path: Path
    ):
        seen: set[str] = set()
        for index, scenario in enumerate(dict.fromkeys(_PRODUCT_SCENARIOS.values())):
            with EventLog(tmp_path / f"m{index}") as fresh:
                seen |= scenario(fresh)
        assert seen == {t.value for t in PRODUCED_TYPES}
        assert {t.value for t in NOT_YET_PRODUCED}.isdisjoint(seen)

    def test_this_module_never_writes_to_a_log_directly(self):
        source = Path(__file__).read_text(encoding="utf-8")
        assert _direct_writes(source) == []

    def test_the_sweep_flags_a_direct_tool_call_and_a_direct_append(self):
        assert _direct_writes("with log.tool_call('c', name='a'):\n    pass\n") == [1]
        assert _direct_writes("api._event_log.append('c', kind)\n") == [1]
        assert _direct_writes("scope.log.append_batch('c', [])\n") == [1]
        assert _direct_writes("errors.append(exc)\nseen.append(1)\n") == []


class TestFailClosedAtTheSites:
    def test_a_refused_pass_two_request_means_no_reply_call(self, log: EventLog):
        llm = ScriptedLLM(responses={"chat": "ok"})
        fsm = copy.deepcopy(_CHAT)
        api = API(fsm, llm_interface=llm, event_log=log)
        # `llm/request` is a checkpoint (fsync) and `start_conversation` makes no logged call.
        cid, _ = api.start_conversation()
        before = len(llm.calls("generate_response"))
        with patch.object(jsonl, "_fsync", side_effect=OSError("disk full")):
            with pytest.raises(FSMError):
                api.converse("hi", cid)
        assert len(llm.calls("generate_response")) == before
        # The request line was appended (then its flush failed), so it is the last event.
        assert [e["type"] for e in log.read(cid)][-1] == "llm/request"

    def test_a_refused_extraction_request_is_swallowed_by_the_site_and_fails_the_next_one(
        self, log: EventLog
    ):
        """`fields.py` swallows any exception of the model call into a WARNING: a refused
        checkpoint on an extraction call does not fail the turn by itself. The poisoned writer
        refuses the NEXT checkpoint (here the Pass-2 request), which does."""
        llm = _script()
        api = API(copy.deepcopy(_FSM), llm_interface=llm, event_log=log)
        cid, _ = api.start_conversation()
        with patch.object(jsonl, "_fsync", side_effect=OSError("disk full")):
            with pytest.raises(FSMError):
                api.converse("Ann", cid)
        assert llm.calls("extract_field") == []
        assert llm.calls("generate_response") == llm.calls("generate_response")[:1]

    def test_a_failing_stack_event_is_a_warning_and_the_push_stands(
        self, log: EventLog
    ):
        api = API(copy.deepcopy(_FSM), llm_interface=_script(), event_log=log)
        cid, _ = api.start_conversation()
        with (
            patch.object(jsonl, "_write", side_effect=OSError("disk full")),
            patch("fsm_llm.runtime.log.logger") as warned,
        ):
            api.push_fsm(cid, copy.deepcopy(_SILENT_CHILD))
        assert api.get_stack_depth(cid) == 2
        assert warned.warning.called

    def test_a_failing_update_context_record_raises(self, log: EventLog):
        api = API(copy.deepcopy(_FSM), llm_interface=_script(), event_log=log)
        cid, _ = api.start_conversation()
        with patch.object(jsonl, "_write", side_effect=OSError("disk full")):
            with pytest.raises(EventLogError):
                api.update_context(cid, {"note": "x"})
        assert (
            api.get_data(cid)["note"] == "x"
        )  # the memory changed; the caller is told


class TestAFrameEndedBeforeItsRecordIsAnFsmError:
    """A5 (engine audit 2.4): ``end_conversation`` ends a frame at the manager before it drops
    the stack entry, so ``update_context`` can resolve a frame that is gone by the time its
    record is built. The producers read ``manager.instances`` without the manager lock (taking
    ``_lock`` there would break the lock order: ``begin_turn`` runs under a ``conv_lock``), and a
    missing frame must read as the ``FSMError`` the rest of the API raises, not a ``KeyError``."""

    def test_update_context_on_a_frame_that_just_ended_raises_fsm_error(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ):
        api = API(copy.deepcopy(_FSM), llm_interface=_script(), event_log=log)
        cid, _ = api.start_conversation()
        api.push_fsm(cid, copy.deepcopy(_CHILD))
        child = api.get_sub_conversation_id(cid)
        real = api.fsm_manager.update_conversation_context

        def write_then_end(conversation_id: str, update: Any, *args: Any) -> None:
            real(conversation_id, update, *args)
            api.fsm_manager.end_conversation(child)  # the end that races the record

        monkeypatch.setattr(
            api.fsm_manager, "update_conversation_context", write_then_end
        )
        with pytest.raises(FSMError, match="Conversation not found"):
            api.update_context(cid, {"note": "x"})

    def test_the_state_blob_of_a_missing_frame_raises_fsm_error(self, log: EventLog):
        api = API(copy.deepcopy(_FSM), llm_interface=_script(), event_log=log)
        with pytest.raises(FSMError, match="Conversation not found"):
            fsm_llm.engine.events.state_ref_of(api.fsm_manager, log, "no-such-frame")


class TestTheProviderPath:
    """A real ``LiteLLMInterface`` with a patched ``completion``: the seam writes the calls."""

    @staticmethod
    def _chat() -> dict[str, Any]:
        return copy.deepcopy(_CHAT)

    def test_one_pair_comes_from_the_provider_seam_and_none_from_the_engine(
        self, log: EventLog
    ):
        llm = LiteLLMInterface(model="gpt-4o", api_key="sk-SECRET-key-0001")
        api = API(self._chat(), llm_interface=llm, event_log=log)
        with patch(COMPLETION, return_value=_reply("ok")) as completion:
            cid, _ = api.start_conversation()
            greeting_calls = completion.call_count
            assert api.converse("hi", cid) == "ok"
            turn_calls = completion.call_count - greeting_calls
        requests = _events(log, cid, "llm/request")
        responses = _events(log, cid, "llm/response")
        # the greeting is the first logged pair, the turn the second (it is no longer skipped)
        assert greeting_calls == 1 and turn_calls == 1
        assert len(requests) == len(responses) == greeting_calls + turn_calls
        assert {e["source"] for e in requests + responses} == {"provider"}
        assert [r["request_seq"] for r in responses] == [r["seq"] for r in requests]
        sent = log.get_json(requests[-1]["prompt_ref"])
        assert sent["model"] == "gpt-4o" and "api_key" not in sent
        assert "sk-SECRET-key-0001" not in "".join(
            p.read_text(encoding="utf-8", errors="replace") for p in log_files(log)
        )
        assert not [e for e in log.read(cid) if e.get("source") == "engine"]

    def test_the_greeting_request_is_logged_and_the_text_is(self, log: EventLog):
        """Was pinned as unlogged (D-021); the greeting is now an `llm/request` and its reply."""
        api = API(
            self._chat(),
            llm_interface=LiteLLMInterface(model="gpt-4o"),
            event_log=log,
        )
        with patch(COMPLETION, return_value=_reply("Welcome")) as completion:
            cid, greeting = api.start_conversation()
        assert completion.call_count == 1 and greeting == "Welcome"
        assert [e["type"] for e in log.read(cid)] == [
            "llm/request",
            "llm/response",
            "context/record",
        ]
        request, response, record = log.read(cid)
        assert request["source"] == "provider" and request["stream"] is False
        assert response["request_seq"] == request["seq"]
        state = log.get_json(record["state_ref"])
        assert state["conversation_history"] == [{"system": "Welcome"}]

    def test_a_streamed_reply_is_one_response_with_the_joined_text(self, log: EventLog):
        api = API(
            self._chat(),
            llm_interface=LiteLLMInterface(model="gpt-4o"),
            event_log=log,
        )
        with patch(COMPLETION, return_value=_reply("Welcome")):
            cid, _ = api.start_conversation()
        stream = iter([_chunk("a"), _chunk("b"), _chunk("c")])
        with patch(COMPLETION, return_value=stream):
            assert "".join(api.converse_stream("hi", cid)) == "abc"
        request = _events(log, cid, "llm/request")[-1]  # [0] is the greeting
        response = _events(log, cid, "llm/response")[-1]
        assert request["stream"] is True and request["source"] == "provider"
        assert response["stream"] is True and response["abandoned"] is False
        assert response["request_seq"] == request["seq"]
        assert log.get_json(response["response_ref"])["text"] == "abc"

    def test_an_abandoned_stream_reports_the_partial_text_and_abandoned(
        self, log: EventLog
    ):
        api = API(
            self._chat(),
            llm_interface=LiteLLMInterface(model="gpt-4o"),
            event_log=log,
        )
        with patch(COMPLETION, return_value=_reply("Welcome")):
            cid, _ = api.start_conversation()
        with patch(COMPLETION, return_value=iter([_chunk("a"), _chunk("b")])):
            stream = api.converse_stream("hi", cid)
            assert next(stream) == "a"
            stream.close()
        response = _events(log, cid, "llm/response")[-1]  # [0] is the greeting
        assert response["abandoned"] is True
        assert log.get_json(response["response_ref"])["text"] == "a"
        end = _events(log, cid, "turn/end")[-1]
        assert end["outcome"] == "abandoned"

    def test_a_failing_provider_call_is_one_provider_error_not_two(self, log: EventLog):
        """`_send` already writes the error of a non-streamed observed call; the engine wrapper
        must not write a second one."""
        api = API(
            self._chat(),
            llm_interface=LiteLLMInterface(model="gpt-4o"),
            event_log=log,
        )
        with patch(COMPLETION, return_value=_reply("Welcome")):
            cid, _ = api.start_conversation()
        with patch(COMPLETION, side_effect=RuntimeError("PRIVATE")):
            with pytest.raises(FSMError):
                api.converse("hi", cid)
        (error,) = _events(log, cid, "provider/error")
        assert error["source"] == "provider"
        start = _events(log, cid, "context/record")[0]["seq"]
        assert [e["type"] for e in _events(log, cid) if e["seq"] > start] == [
            "turn/start",
            "executor/start",
            "executor/result",
            "transition",
            "llm/request",
            "provider/error",
            "turn/end",
        ]

    @pytest.mark.parametrize("when", ["midway", "at the first chunk"])
    def test_a_failed_stream_is_one_provider_error_and_no_response(
        self, log: EventLog, when: str
    ):
        def _broken() -> Iterator[SimpleNamespace]:
            yield _chunk("a")
            raise RuntimeError("PRIVATE stream failure")

        provider: dict[str, Any] = (
            {"return_value": _broken()}
            if when == "midway"
            else {"side_effect": RuntimeError("PRIVATE stream failure")}
        )

        api = API(
            self._chat(),
            llm_interface=LiteLLMInterface(model="gpt-4o"),
            event_log=log,
        )
        with patch(COMPLETION, return_value=_reply("Welcome")):
            cid, _ = api.start_conversation()
        with patch(COMPLETION, **provider):
            with pytest.raises(Exception):
                list(api.converse_stream("hi", cid))
        (error,) = _events(log, cid, "provider/error")
        request = _events(log, cid, "llm/request")[-1]
        assert error["request_seq"] == request["seq"]
        assert [
            e
            for e in _events(log, cid, "llm/response")
            if e["request_seq"] == error["request_seq"]
        ] == []
        # The interface wraps a failing stream as LLMResponseError; the log names that class.
        assert error["source"] == "provider"
        assert error["error_type"] == "LLMResponseError"
        assert "PRIVATE" not in "".join(
            p.read_text(encoding="utf-8", errors="replace") for p in log_files(log)
        )


def _request() -> ResponseGenerationRequest:
    return ResponseGenerationRequest(system_prompt="Greet.", user_message="hi")


def _extracting_fsm(site: str) -> dict[str, Any]:
    """One silent state that needs `name` (bulk or per-field extraction), then `done`."""
    gate = [
        {
            "description": "named",
            "requires_context_keys": ["name"],
            "logic": {"has_context": "name"},
        }
    ]
    where: dict[str, Any] = (
        {"required_context_keys": ["name"]}
        if site == "bulk"
        else {
            "field_extractions": [
                {
                    "field_name": "name",
                    "field_type": "str",
                    "extraction_instructions": "the name",
                }
            ]
        }
    )
    ask = _state(
        "ask",
        transitions=[{"target_state": "done", "description": "go", "conditions": gate}],
        **where,
    )
    return {
        "name": "Extract",
        "description": "d",
        "initial_state": "ask",
        "states": {"ask": ask, "done": _state("done", response_instructions="bye")},
    }


class TestANonJsonContextValue:
    @pytest.mark.parametrize("site", ["bulk", "field"])
    def test_extraction_with_the_log_on_is_the_extraction_with_it_off(
        self, log: EventLog, site: str
    ):
        """A plain object in the context (a handler or `initial_context` can put one there)
        must not make the request blob raise: it degrades to a typed marker."""

        def run(event_log: EventLog | None) -> tuple[str, Any]:
            llm = ScriptedLLM(
                extraction_data={"name": "Ann"}, responses={"done": "bye"}
            )
            api = API(_extracting_fsm(site), llm_interface=llm, event_log=event_log)
            cid, _ = api.start_conversation(initial_context={"thing": object()})
            api.converse("I am Ann", cid)
            return api.get_current_state(cid), api.get_data(cid).get("name")

        assert run(None) == run(log) == ("done", "Ann")
        stored = "".join(
            p.read_text(encoding="utf-8", errors="replace") for p in log_files(log)
        )
        assert "<object>" in stored


class TestTheStreamRecorder:
    def test_a_consumed_stream_writes_one_response_not_abandoned(self, log: EventLog):
        llm = ScriptedLLM(stream_chunks=["a", "b"])
        with bind(log, "root-1"):
            with llm_call(llm, "response_stream", _request(), stream=True) as call:
                got = list(call.stream(llm.generate_response_stream(_request())))
                call.stream_done("".join(got))
        request, response = list(log.read("root-1"))
        assert (request["type"], request["stream"]) == ("llm/request", True)
        assert (
            response["abandoned"] is False and response["request_seq"] == request["seq"]
        )

    def test_an_unfinished_stream_is_abandoned_without_being_told(self, log: EventLog):
        llm = ScriptedLLM(stream_chunks=["a", "b"])
        with bind(log, "root-1"):
            with llm_call(llm, "response_stream", _request(), stream=True) as call:
                chunks = iter(call.stream(llm.generate_response_stream(_request())))
                assert next(chunks) == "a"
                call.stream_done("a")
        assert list(log.read("root-1"))[-1]["abandoned"] is True

    def test_a_stream_that_raised_writes_a_provider_error_and_no_response(
        self, log: EventLog
    ):
        def _broken() -> Iterator[str]:
            yield "a"
            raise RuntimeError("PRIVATE")

        llm = ScriptedLLM()
        with bind(log, "root-1"):
            with pytest.raises(RuntimeError):
                with llm_call(llm, "response_stream", _request(), stream=True) as call:
                    try:
                        list(call.stream(_broken()))
                    finally:
                        call.stream_done("a")
        assert [e["type"] for e in log.read("root-1")] == [
            "llm/request",
            "provider/error",
        ]

    def test_a_wrapper_around_a_real_interface_logs_one_stream_request(
        self, log: EventLog
    ):
        wrapper = _Wrapper()
        stream = iter([_chunk("a"), _chunk("b")])
        with bind(log, "root-1"):
            with patch(COMPLETION, return_value=stream) as completion:
                with llm_call(
                    wrapper, "response_stream", _request(), stream=True
                ) as call:
                    got = list(
                        call.stream(wrapper.generate_response_stream(_request()))
                    )
                    call.stream_done("".join(got))
        assert completion.call_count == 1
        assert [e["type"] for e in log.read("root-1")] == [
            "llm/request",
            "llm/response",
        ]
        assert {e["source"] for e in log.read("root-1")} == {"engine"}

    def test_a_stream_consumed_on_another_thread_leaves_no_observer_behind(
        self, log: EventLog
    ):
        """The suppression is bound per chunk and reset in the context that set it, so a stream
        read from a thread of its own raises nothing and the observer of this context is intact."""
        from fsm_llm.llm.observer import _ACTIVE

        llm = ScriptedLLM(stream_chunks=["a", "b"])
        errors: list[BaseException] = []
        with bind(log, "root-1") as scope:
            with llm_call(llm, "response_stream", _request(), stream=True) as call:
                chunks = iter(call.stream(llm.generate_response_stream(_request())))
                assert next(chunks) == "a"  # the first chunk is read in THIS context

                def read() -> None:
                    try:
                        list(chunks)
                    except BaseException as exc:  # pragma: no cover - reported below
                        errors.append(exc)

                thread = threading.Thread(
                    target=contextvars.Context().run, args=(read,)
                )
                thread.start()
                thread.join()
                assert _ACTIVE.get() is scope.observer
                call.stream_done("ab")
        assert errors == []
        assert [e["type"] for e in log.read("root-1")] == [
            "llm/request",
            "llm/response",
        ]


class TestThePrivacyOfTheRequestBlob:
    def test_a_secret_in_a_request_context_never_reaches_a_blob(self, log: EventLog):
        secrets = ("sk-SECRET-corpus-0001", "pw-SECRET-corpus-0002", "tok-SECRET-3")
        llm = _script()
        api = API(copy.deepcopy(_FSM), llm_interface=llm, event_log=log)
        cid, _ = api.start_conversation(
            {
                "api_key": secrets[0],
                "profile": {"password": secrets[1], "city": "Oslo"},
                "session": {"token": secrets[2]},
            }
        )
        api.converse("Ann", cid)
        api.update_context(cid, {"db_password": secrets[1], "city": "Oslo"})
        # The double saw the secret-looking entry in the request object (non-vacuity) ...
        extraction = llm.calls("extract_field")[0]
        assert extraction.context is not None and "profile" in extraction.context
        # ... and no byte of the log directory holds a secret.
        text = "".join(
            p.read_text(encoding="utf-8", errors="replace") for p in log_files(log)
        )
        assert "Oslo" in text  # non-vacuity: public context is in the log
        for secret in secrets:
            assert secret not in text

    def test_a_request_dump_redacts_secret_named_entries_only(self, log: EventLog):
        from fsm_llm.runtime.scope import _request_body

        body = _request_body(
            SimpleNamespace(
                model_dump=lambda mode: {
                    "system_prompt": "Hello",
                    "context": {"api_key": "sk-X", "city": "Oslo"},
                }
            )
        )
        assert body == {
            "system_prompt": "Hello",
            "context": {"api_key": "<redacted>", "city": "Oslo"},
        }


_CALLS = {
    "generate_response",
    "generate_response_stream",
    "extract_field",
    "extract_bulk_data",
    "complete",
}


class TestEverySiteIsWrapped:
    """A sweep over ``src/fsm_llm/engine``: a new model call site cannot be added unlogged."""

    def _sites(self) -> list[tuple[Path, int, bool]]:
        root = Path(fsm_llm.engine.__file__).parent
        found: list[tuple[Path, int, bool]] = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            parents: dict[ast.AST, ast.AST] = {}
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    parents[child] = node
            for node in ast.walk(tree):
                if self._called(node) is None:
                    continue
                found.append((path, node.lineno, self._inside_llm_call(node, parents)))
        return found

    @staticmethod
    def _called(node: ast.AST) -> str | None:
        """The model method a call node calls: ``x.<name>(...)``, ``acall(x, "<name>", ...)`` or
        ``acall_stream(x, "<name>", ...)``.

        Plan 51fde95b (D-007): a call that goes through the transport helper of ``fsm_llm.aio``
        names the method as a string, and is still a model call site. Plan f150ec28 (step 3.2):
        so is a stream opened through ``acall_stream``.
        """
        if not isinstance(node, ast.Call):
            return None
        if isinstance(node.func, ast.Attribute):
            return node.func.attr if node.func.attr in _CALLS else None
        named = (
            getattr(node.args[1], "value", None)
            if getattr(node.func, "id", None) in ("acall", "acall_stream")
            and len(node.args) > 1
            else None
        )
        return named if named in _CALLS else None

    @staticmethod
    def _inside_llm_call(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.With) and any(
                isinstance(item.context_expr, ast.Call)
                and getattr(item.context_expr.func, "id", None) == "llm_call"
                for item in node.items
            ):
                return True
        return False

    def test_every_model_call_of_the_engine_is_inside_an_llm_call_block(self):
        sites = self._sites()
        unlogged = [(p.name, line) for p, line, wrapped in sites if not wrapped]
        assert unlogged == []

    def test_the_sweep_finds_the_seven_sites(self):
        names = sorted(p.name for p, _, _ in self._sites())
        assert names == [
            "bulk.py",
            "classification.py",
            "completion.py",
            "fields.py",
            "pass2.py",
            "pass2.py",
            "pass2.py",
        ]

    def test_the_sweep_sees_a_call_through_acall_and_flags_it_unwrapped(self):
        for source, name in [
            ('async def f(x):\n    return await acall(x, "complete", 1)\n', "complete"),
            (
                'async def f(x):\n    return acall_stream(x, "generate_response_stream")\n',
                "generate_response_stream",
            ),
            ('async def f(x):\n    return await acall(x, "_send", 1)\n', None),
            ("async def f(x):\n    return await acall(x)\n", None),
        ]:
            tree = ast.parse(source)
            call = next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") in ("acall", "acall_stream")
            )
            assert self._called(call) == name
        tree = ast.parse('async def f(x):\n    return await acall(x, "complete", 1)\n')
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        assert self._inside_llm_call(call, parents) is False

    def test_the_sweep_would_flag_an_unwrapped_call(self):
        tree = ast.parse("def f(x):\n    return x.generate_response(1)\n")
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        assert self._inside_llm_call(call, parents) is False
        tree = ast.parse(
            "def f(x):\n    with llm_call(x, 'a', 1) as c:\n        return x.generate_response(1)\n"
        )
        call = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        )
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        assert self._inside_llm_call(call, parents) is True


class TestTheLogOffPath:
    def test_the_producers_do_nothing_without_a_scope(self):
        with patch("fsm_llm.engine.turn.events.emit", side_effect=AssertionError):
            api, cid, _ = _run(None)
            assert api.get_stack_depth(cid) == 1

    def test_a_log_less_api_leaves_the_event_log_attribute_none(self, tmp_path: Path):
        api = API(copy.deepcopy(_FSM), llm_interface=_script())
        assert api._event_log is None
        cid, _ = api.start_conversation()
        api.update_context(cid, {"note": "x"})
        assert api.get_data(cid)["note"] == "x"
        assert not list(tmp_path.iterdir())

    def test_log_write_error_is_the_refusal_type_of_the_engine_fallback(
        self, log: EventLog
    ):
        llm = ScriptedLLM(responses={"chat": "ok"})
        with bind(log, "root-1"):
            with patch.object(jsonl, "_fsync", side_effect=OSError("disk full")):
                with pytest.raises(LogWriteError):
                    with llm_call(llm, "response_generation", _request()):
                        llm.generate_response(_request())
        assert llm.calls("generate_response") == []
        assert EventType.LLM_REQUEST.value in {e["type"] for e in log.read("root-1")}
