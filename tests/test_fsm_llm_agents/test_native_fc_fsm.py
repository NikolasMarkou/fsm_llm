"""The native function-calling FSM and its handlers (plan 944e2692 step 15, D-009).

``build_native_fc_fsm`` turns ``NativeFunctionCallingReactAgent``'s private
loop into an FSM run by core: ``call_model`` (completion state with tools),
``run_tools`` (handlers run each call through ``ToolRegistry.execute``),
optional ``force_final`` and ``repair`` completion states, ``conclude``.
``NativeFCHandlers`` holds the handlers. Nothing here is wired into the agent
yet (step 16 does that and deletes the loop).

``TestGoldenRequestsThroughCore`` runs the definition through core with the
recording stub of ``test_native_fc_golden.py`` on the provider binding and
compares every request with the fixture captured from the old loop at
e1f63a9 (transport keys excluded, as there): the FSM sends the same bytes.

On the parent commit (eb07e53) none of the names under test exist, so every
test here fails at import.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from fsm_llm import API
from fsm_llm.agents import ToolRegistry
from fsm_llm.agents.base import _output_response_format
from fsm_llm.agents.constants import (
    FRAMEWORK_ONLY_KEYS,
    NATIVE_FC_HANDLER_ONLY_KEYS,
    ContextKeys,
    NativeFCContextKeys,
    NativeFCStates,
    NativeLoopEnd,
    StopReason,
)
from fsm_llm.agents.fsm_definitions import build_native_fc_fsm
from fsm_llm.agents.native_fc import (
    _FORCE_WRITE_PROMPT,
    _REPAIR_PROMPT,
    _SYSTEM_PROMPT,
    NativeFCHandlers,
)
from fsm_llm.constants import CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT
from fsm_llm.definitions import CompletionResponse, FSMDefinition, ModelToolCall
from fsm_llm.llm import check_tool_transcript
from fsm_llm.utilities import extract_json_from_text
from fsm_llm.validator import FSMValidator
from fsm_llm.visualizer import build_fsm_graph, to_mermaid
from tests.test_fsm_llm.test_completion_state import _ScriptedLLM
from tests.test_fsm_llm_agents import test_native_fc_golden as golden

_K = NativeFCContextKeys
_S = NativeFCStates


# --------------------------------------------------------------
# Real tools, recorded calls
# --------------------------------------------------------------


class _Tools:
    """A registry of real-signature tools that records every call it runs."""

    def __init__(self) -> None:
        self.log: list[tuple[str, dict[str, Any]]] = []
        self.registry = ToolRegistry()

        def add(a: int, b: int) -> int:
            """Add two integers."""
            self.log.append(("add", {"a": a, "b": b}))
            return a + b

        def lookup(word: str) -> str:
            """Look a word up in the dictionary."""
            self.log.append(("lookup", {"word": word}))
            return f"{word}: a word"

        def login(user: str, password: str) -> str:
            """Log a user in."""
            self.log.append(("login", {"user": user, "password": password}))
            return f"{user} logged in"

        def broken(reason: str) -> str:
            """A tool that always fails."""
            self.log.append(("broken", {"reason": reason}))
            raise RuntimeError(f"cannot: {reason}")

        def write_note(text: str) -> str:
            """Save a note with the final findings."""
            self.log.append(("write_note", {"text": text}))
            return f"saved {len(text)} chars"

        for fn in (add, lookup, login, broken, write_note):
            self.registry.register_function(fn)

    def names(self) -> list[str]:
        return [name for name, _ in self.log]


class _Answer(BaseModel):
    """The output schema of the repair tests."""

    answer: str


def _parse_answer(text: str) -> _Answer | None:
    data = extract_json_from_text(text)
    if not isinstance(data, dict):
        return None
    try:
        return _Answer(**data)
    except ValueError:
        return None


def _calls(*calls: tuple[str, dict[str, Any]], text: str | None = None) -> Any:
    return CompletionResponse(
        kind="calls",
        text=text,
        calls=tuple(
            ModelToolCall(id=f"call_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls, start=1)
        ),
    )


def _final(text: str | None) -> CompletionResponse:
    return CompletionResponse(kind="final", text=text)


_MALFORMED = CompletionResponse(kind="malformed")


def _run(
    tools: _Tools,
    llm: _ScriptedLLM,
    *,
    task: str = "What is 2 + 3?",
    max_tool_turns: int = 5,
    force_final_tool: str | None = None,
    repair: bool = False,
    extra_context: dict[str, Any] | None = None,
) -> tuple[API, str, list[Any]]:
    """Run the native FSM through core to its terminal; return the run."""
    fsm = build_native_fc_fsm(
        tools.registry.get_json_schemas(),
        instructions=_SYSTEM_PROMPT,
        force_final_tool=force_final_tool,
        repair=repair,
    )
    api = API.from_definition(fsm, llm_interface=llm)
    NativeFCHandlers(
        tools.registry,
        max_tool_turns=max_tool_turns,
        force_final_tool=force_final_tool,
        parse_structured=_parse_answer if repair else None,
    ).register(api)
    context: dict[str, Any] = {ContextKeys.TASK: task, ContextKeys.AGENT_TRACE: []}
    if repair:
        context[CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT] = _output_response_format(_Answer)
    context.update(extra_context or {})
    conv_id, greeting = api.start_conversation(context)
    assert greeting == ""
    steps = list(api.run_until_terminal(conv_id, max_steps=40))
    assert api.has_conversation_ended(conv_id)
    return api, conv_id, steps


def _context(api: API, conv_id: str) -> dict[str, Any]:
    return api.fsm_manager.get_complete_conversation(conv_id)["collected_data"]


def _traced(context: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        (step["action"].split("(")[0], step["tool_input"])
        for step in context[ContextKeys.AGENT_TRACE]
    ]


def _schemas() -> list[dict[str, Any]]:
    return _Tools().registry.get_json_schemas()


# --------------------------------------------------------------
# Definition
# --------------------------------------------------------------


_SHAPES = [
    pytest.param({}, id="loop-only"),
    pytest.param({"force_final_tool": "write_note"}, id="forced"),
    pytest.param({"repair": True}, id="repair"),
    pytest.param({"force_final_tool": "write_note", "repair": True}, id="both"),
]


class TestDefinition:
    @pytest.mark.parametrize("options", _SHAPES)
    def test_loads_and_validates_clean(self, options: dict[str, Any]) -> None:
        fsm = build_native_fc_fsm(_schemas(), instructions=_SYSTEM_PROMPT, **options)
        definition = FSMDefinition(**fsm)
        assert definition.initial_state == _S.CALL_MODEL
        result = FSMValidator(fsm).validate()
        assert result.is_valid, result.errors
        assert result.warnings == []

    @pytest.mark.parametrize("options", _SHAPES)
    def test_graph_export(self, options: dict[str, Any]) -> None:
        fsm = build_native_fc_fsm(_schemas(), instructions=_SYSTEM_PROMPT, **options)
        graph = build_fsm_graph(fsm)
        assert graph.initial_state == _S.CALL_MODEL
        assert {node.id for node in graph.nodes} == set(fsm["states"])
        assert [n.id for n in graph.nodes if n.is_terminal] == [_S.CONCLUDE]
        edges = {(e.source, e.target) for e in graph.edges}
        assert (_S.CALL_MODEL, _S.RUN_TOOLS) in edges
        assert (_S.RUN_TOOLS, _S.CALL_MODEL) in edges
        assert to_mermaid(graph).startswith("stateDiagram-v2")

    def test_optional_states_exist_only_when_configured(self) -> None:
        assert set(build_native_fc_fsm(_schemas(), instructions="s")["states"]) == {
            _S.CALL_MODEL,
            _S.RUN_TOOLS,
            _S.CONCLUDE,
        }
        both = build_native_fc_fsm(
            _schemas(), instructions="s", force_final_tool="write_note", repair=True
        )
        assert set(both["states"]) == {
            _S.CALL_MODEL,
            _S.RUN_TOOLS,
            _S.FORCE_FINAL,
            _S.REPAIR,
            _S.CONCLUDE,
        }

    def test_undeclared_forced_tool_gets_no_forced_turn(self) -> None:
        fsm = build_native_fc_fsm(_schemas(), instructions="s", force_final_tool="nope")
        assert _S.FORCE_FINAL not in fsm["states"]
        targets = {
            t["target_state"] for s in fsm["states"].values() for t in s["transitions"]
        }
        assert _S.FORCE_FINAL not in targets

    @pytest.mark.parametrize("options", _SHAPES)
    def test_every_state_is_silent_and_priorities_are_distinct(
        self, options: dict[str, Any]
    ) -> None:
        fsm = build_native_fc_fsm(_schemas(), instructions="s", **options)
        for state in fsm["states"].values():
            assert state["response_instructions"] == ""
            priorities = [t["priority"] for t in state["transitions"]]
            assert len(priorities) == len(set(priorities)), state["id"]

    @pytest.mark.parametrize("options", _SHAPES)
    def test_every_non_terminal_state_has_an_unconditional_fallback(
        self, options: dict[str, Any]
    ) -> None:
        """D-002 of 3e4eb3e5: a BLOCKED turn runs no handler, so no state may
        block: its lowest-priority edge has no condition."""
        fsm = build_native_fc_fsm(_schemas(), instructions="s", **options)
        for state in fsm["states"].values():
            if not state["transitions"]:
                continue
            last = max(state["transitions"], key=lambda t: t["priority"])
            assert not last.get("conditions"), state["id"]

    def test_completion_states_send_the_instructions_and_their_own_transcript(
        self,
    ) -> None:
        policy = f"{_SYSTEM_PROMPT}\n\nAnswer in one sentence."
        schemas = _schemas()
        fsm = build_native_fc_fsm(
            schemas, instructions=policy, force_final_tool="write_note", repair=True
        )
        call = fsm["states"][_S.CALL_MODEL]["completion"]
        assert call == {
            "tools": schemas,
            "instructions": policy,
            "messages_key": _K.TRANSCRIPT,
            "result_key": _K.MODEL_REPLY,
        }
        forced = fsm["states"][_S.FORCE_FINAL]["completion"]
        assert [s["function"]["name"] for s in forced["tools"]] == ["write_note"]
        assert forced["tool_choice"] == "write_note"
        assert forced["messages_key"] == _K.FORCE_MESSAGES
        repair = fsm["states"][_S.REPAIR]["completion"]
        assert repair["response_format_key"] == CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT
        assert "tools" not in repair
        assert repair["messages_key"] == _K.REPAIR_MESSAGES
        assert {call["instructions"], forced["instructions"]} == {policy}
        assert repair["instructions"] == policy

    def test_handler_written_keys_are_handler_only(self) -> None:
        fsm = build_native_fc_fsm(_schemas(), instructions="s")
        assert set(fsm["handler_only_keys"]) == set(FRAMEWORK_ONLY_KEYS) | set(
            NATIVE_FC_HANDLER_ONLY_KEYS
        )

    def test_no_tools_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one tool schema"):
            build_native_fc_fsm([], instructions="s")


# --------------------------------------------------------------
# Golden requests through core
# --------------------------------------------------------------


def _parser(schema: type[BaseModel]) -> Any:
    def parse(text: str) -> BaseModel | None:
        data = extract_json_from_text(text)
        if not isinstance(data, dict):
            return None
        try:
            return schema(**data)
        except ValueError:
            return None

    return parse


class TestGoldenRequestsThroughCore:
    """Every request of the old loop, byte for byte, from the FSM on core."""

    @pytest.mark.parametrize("name", golden._scenario_names())
    def test_requests_equal_the_golden_capture(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base, label = name.split("/")
        spec = golden._scenario_specs()[base]
        recorder = golden._Recorder(spec["script"]())
        for binding in golden._BINDINGS:
            monkeypatch.setattr(binding, recorder)
        registry = golden._registry()
        options = spec.get("config", {})
        schema = options.get("output_schema")
        force = options.get("force_final_tool")
        policy = spec.get("system_policy")
        instructions = f"{_SYSTEM_PROMPT}\n\n{policy}" if policy else _SYSTEM_PROMPT
        fsm = build_native_fc_fsm(
            registry.get_json_schemas(),
            instructions=instructions,
            force_final_tool=force,
            repair=schema is not None,
        )
        seed = {"seed": spec["seed"]} if spec.get("seed") is not None else {}
        api = API.from_definition(
            fsm,
            model=golden._MODELS[label],
            temperature=0.5,
            max_tokens=1000,
            **seed,
        )
        NativeFCHandlers(
            registry,
            max_tool_turns=5,
            force_final_tool=force,
            parse_structured=_parser(schema) if schema is not None else None,
        ).register(api)
        context: dict[str, Any] = {
            ContextKeys.TASK: spec["task"],
            ContextKeys.AGENT_TRACE: [],
        }
        if schema is not None:
            context[CONTEXT_KEY_OUTPUT_RESPONSE_FORMAT] = _output_response_format(
                schema
            )
        conv_id, _ = api.start_conversation(context)
        api.run_until_terminal(conv_id, max_steps=20)

        expected = golden._golden()["scenarios"][name]
        assert recorder.script == [], "a scripted reply was never requested"
        assert len(recorder.requests) == len(expected["requests"])
        for index, (got, want) in enumerate(
            zip(recorder.requests, expected["requests"], strict=True)
        ):
            assert got == want, f"{name} request {index} differs"
        data = api.get_data(conv_id)
        outcome = expected["outcome"]
        assert data[_K.ANSWER] == outcome["answer"]
        assert [
            [step["action"].split("(")[0], step["tool_input"]]
            for step in data[ContextKeys.AGENT_TRACE]
        ] == outcome["tool_calls"]
        assert (ContextKeys.FORCED_STOP_REASON not in data) == outcome["success"]


# --------------------------------------------------------------
# Handlers, through a scripted interface
# --------------------------------------------------------------


class TestModelLoop:
    def test_direct_answer_makes_one_call_and_runs_nothing(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(_final("5"))
        api, conv_id, steps = _run(tools, llm)
        assert [s.state_after for s in steps] == [_S.CONCLUDE]
        assert tools.log == []
        data = api.get_data(conv_id)
        assert data[_K.ANSWER] == "5"
        assert data[_K.LOOP_END] == NativeLoopEnd.FINAL
        assert ContextKeys.FORCED_STOP_REASON not in data
        (request,) = llm.requests
        assert request.messages == [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "What is 2 + 3?"},
        ]
        assert llm.other_calls == []

    def test_calls_run_in_order_and_the_transcript_is_paired(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(
            _calls(("lookup", {"word": "five"}), ("add", {"a": 2, "b": 3})),
            _final("2 + 3 = 5"),
        )
        api, conv_id, steps = _run(tools, llm)
        assert [s.state_after for s in steps] == [
            _S.RUN_TOOLS,
            _S.CALL_MODEL,
            _S.CONCLUDE,
        ]
        assert tools.names() == ["lookup", "add"]
        second = llm.requests[1].messages
        check_tool_transcript(second)
        assert [m["role"] for m in second] == [
            "system",
            "user",
            "assistant",
            "tool",
            "tool",
        ]
        assert second[3:] == [
            {"role": "tool", "tool_call_id": "call_1", "content": "five: a word"},
            {"role": "tool", "tool_call_id": "call_2", "content": "5"},
        ]
        context = _context(api, conv_id)
        assert _traced(context) == [
            ("lookup", {"word": "five"}),
            ("add", {"a": 2, "b": 3}),
        ]
        assert context[_K.TOOL_TURNS] == 1
        assert api.get_data(conv_id)[_K.ANSWER] == "2 + 3 = 5"

    def test_text_beside_the_calls_stays_in_the_transcript(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(
            _calls(("add", {"a": 1, "b": 1}), text="Let me add."), _final("2")
        )
        _run(tools, llm)
        assistant = llm.requests[1].messages[2]
        assert assistant["content"] == "Let me add."
        assert assistant["tool_calls"][0]["function"] == {
            "name": "add",
            "arguments": json.dumps({"a": 1, "b": 1}),
        }

    def test_failed_tool_is_shown_to_the_model_as_failed(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(_calls(("broken", {"reason": "disk full"})), _final("no"))
        _run(tools, llm)
        tool_message = llm.requests[1].messages[-1]
        assert tool_message["role"] == "tool"
        assert tool_message["content"].startswith("[TOOL FAILED] Error: ")
        assert "disk full" in tool_message["content"]

    def test_trace_is_redacted_but_the_tool_gets_the_real_arguments(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(
            _calls(("login", {"user": "ann", "password": "hunter2"})), _final("ok")
        )
        api, conv_id, _ = _run(tools, llm)
        assert tools.log == [("login", {"user": "ann", "password": "hunter2"})]
        (step,) = _context(api, conv_id)[ContextKeys.AGENT_TRACE]
        assert step["tool_input"] == {"user": "ann", "password": "<redacted>"}
        assert "hunter2" not in step["action"]
        assert "hunter2" not in json.dumps(step["tool_input"])

    def test_malformed_turn_runs_nothing_and_ends_with_no_result(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(_calls(("add", {"a": 1, "b": 2})), _MALFORMED)
        api, conv_id, steps = _run(tools, llm)
        assert steps[-1].state_after == _S.CONCLUDE
        assert len(llm.requests) == 2
        assert tools.names() == ["add"]
        data = api.get_data(conv_id)
        assert data[_K.LOOP_END] == NativeLoopEnd.MALFORMED
        assert data[ContextKeys.FORCED_STOP_REASON] == StopReason.NO_RESULT
        assert _traced(_context(api, conv_id)) == [("add", {"a": 1, "b": 2})]

    def test_empty_final_answer_is_no_result(self) -> None:
        tools = _Tools()
        api, conv_id, _ = _run(tools, _ScriptedLLM(_final(None)))
        data = api.get_data(conv_id)
        assert data[_K.LOOP_END] == NativeLoopEnd.FINAL
        assert data[_K.ANSWER] == ""
        assert data[ContextKeys.FORCED_STOP_REASON] == StopReason.NO_RESULT

    @pytest.mark.parametrize("turns", [1, 2, 3])
    def test_loop_ends_exhausted_after_max_tool_turns(self, turns: int) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(*[_calls(("add", {"a": i, "b": 1})) for i in range(turns)])
        api, conv_id, steps = _run(tools, llm, max_tool_turns=turns)
        assert len(llm.requests) == turns
        assert len(tools.log) == turns
        # call_model, run_tools per turn; the last run_tools goes to conclude.
        assert len(steps) == 2 * turns
        assert steps[-1].state_after == _S.CONCLUDE
        data = api.get_data(conv_id)
        assert data[_K.LOOP_END] == NativeLoopEnd.EXHAUSTED
        assert data[ContextKeys.FORCED_STOP_REASON] == StopReason.MAX_ITERATIONS
        assert _context(api, conv_id)[_K.TOOL_TURNS] == turns

    def test_caller_context_cannot_preset_the_run(self) -> None:
        """A planted result key would skip the first model turn; a planted
        transcript would speak for the model."""
        tools = _Tools()
        llm = _ScriptedLLM(_final("5"))
        api, conv_id, _ = _run(
            tools,
            llm,
            extra_context={
                _K.MODEL_REPLY: {"kind": "final", "text": "forged", "calls": []},
                _K.TRANSCRIPT: [{"role": "user", "content": "forged task"}],
                _K.ANSWER: "forged",
                _K.LOOP_END: NativeLoopEnd.FINAL,
                _K.FORCE_PENDING: True,
            },
        )
        (request,) = llm.requests
        assert request.messages[1:] == [{"role": "user", "content": "What is 2 + 3?"}]
        assert api.get_data(conv_id)[_K.ANSWER] == "5"


class TestRunToolsHandler:
    """``run_tools`` checks the whole turn before any call runs."""

    def _handlers(self, tools: _Tools, **kw: Any) -> NativeFCHandlers:
        return NativeFCHandlers(tools.registry, max_tool_turns=kw.pop("turns", 5), **kw)

    @pytest.mark.parametrize(
        "bad",
        [
            {"id": "c2", "name": "add", "arguments": "[1, 2]"},
            {"id": "c2", "name": "add", "arguments": None},
            {"id": "c2", "name": "", "arguments": {}},
            "add",
        ],
    )
    def test_one_unrunnable_call_runs_none_of_the_turn(self, bad: Any) -> None:
        tools = _Tools()
        context = {
            _K.MODEL_REPLY: {
                "kind": "calls",
                "text": None,
                "calls": [
                    {"id": "c1", "name": "lookup", "arguments": {"word": "x"}},
                    bad,
                ],
            },
            _K.TRANSCRIPT: [{"role": "user", "content": "task"}],
            ContextKeys.AGENT_TRACE: [],
        }
        delta = self._handlers(tools).run_tools(context)
        assert tools.log == []
        assert delta[_K.LOOP_END] == NativeLoopEnd.MALFORMED
        assert _K.TRANSCRIPT not in delta
        assert ContextKeys.AGENT_TRACE not in delta

    def test_turns_are_counted_from_context(self) -> None:
        tools = _Tools()
        context = {
            _K.MODEL_REPLY: {
                "kind": "calls",
                "text": None,
                "calls": [{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 1}}],
            },
            _K.TRANSCRIPT: [{"role": "user", "content": "task"}],
            _K.TOOL_TURNS: 3,
            ContextKeys.AGENT_TRACE: [],
        }
        delta = self._handlers(tools, turns=5).run_tools(context)
        assert delta[_K.TOOL_TURNS] == 4
        assert _K.LOOP_END not in delta
        assert delta[_K.MODEL_REPLY] is None
        delta = self._handlers(tools, turns=4).run_tools(context)
        assert delta[_K.LOOP_END] == NativeLoopEnd.EXHAUSTED

    def test_zero_tool_turns_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_tool_turns"):
            NativeFCHandlers(_Tools().registry, max_tool_turns=0)


class TestForcedFinalTool:
    def test_forced_turn_after_a_final_answer_runs_the_forced_call(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(
            _calls(("lookup", {"word": "paris"})),
            _final("Paris is a word."),
            _calls(("write_note", {"text": "Paris: a word"})),
        )
        api, conv_id, steps = _run(tools, llm, force_final_tool="write_note")
        assert [s.state_after for s in steps][-2:] == [_S.FORCE_FINAL, _S.CONCLUDE]
        assert tools.names() == ["lookup", "write_note"]
        forced = llm.requests[2]
        assert forced.tool_choice == {
            "type": "function",
            "function": {"name": "write_note"},
        }
        assert [t["function"]["name"] for t in forced.tools or []] == ["write_note"]
        assert forced.messages[-1] == {"role": "user", "content": _FORCE_WRITE_PROMPT}
        assert forced.messages[:-1] == llm.requests[1].messages
        data = api.get_data(conv_id)
        assert data[_K.ANSWER] == "Paris is a word."
        assert ContextKeys.FORCED_STOP_REASON not in data
        assert [n for n, _ in _traced(_context(api, conv_id))] == [
            "lookup",
            "write_note",
        ]

    def test_no_forced_turn_when_the_tool_already_ran(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(_calls(("write_note", {"text": "done"})), _final("Saved."))
        _run(tools, llm, force_final_tool="write_note")
        assert len(llm.requests) == 2
        assert tools.names() == ["write_note"]

    def test_forced_turn_follows_a_malformed_loop_and_keeps_no_result(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(_MALFORMED, _calls(("write_note", {"text": "partial"})))
        api, conv_id, _ = _run(tools, llm, force_final_tool="write_note")
        assert tools.names() == ["write_note"]
        assert api.get_data(conv_id)[ContextKeys.FORCED_STOP_REASON] == (
            StopReason.NO_RESULT
        )

    def test_malformed_forced_turn_runs_nothing(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(_final("Done."), _MALFORMED)
        api, conv_id, _ = _run(tools, llm, force_final_tool="write_note")
        assert len(llm.requests) == 2
        assert tools.log == []
        data = api.get_data(conv_id)
        assert data[_K.ANSWER] == "Done."
        assert ContextKeys.FORCED_STOP_REASON not in data


class TestRepairTurn:
    def test_unparsed_answer_is_repaired_by_a_structured_turn(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(_final("It is five."), _final('{"answer": "5"}'))
        api, conv_id, steps = _run(tools, llm, repair=True)
        assert [s.state_after for s in steps] == [_S.REPAIR, _S.CONCLUDE]
        repair = llm.requests[1]
        assert repair.tools is None
        assert repair.response_format == _output_response_format(_Answer)
        assert repair.messages == [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "What is 2 + 3?"},
            {"role": "user", "content": _REPAIR_PROMPT},
        ]
        assert api.get_data(conv_id)[_K.ANSWER] == '{"answer": "5"}'

    def test_parsed_answer_needs_no_repair(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(_final('{"answer": "5"}'))
        _run(tools, llm, repair=True)
        assert len(llm.requests) == 1

    def test_unparsable_repair_is_discarded(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(_final("It is five."), _final("still prose"))
        api, conv_id, _ = _run(tools, llm, repair=True)
        data = api.get_data(conv_id)
        assert data[_K.ANSWER] == "It is five."
        assert ContextKeys.FORCED_STOP_REASON not in data

    def test_repair_never_sees_the_forced_nudge(self) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(
            _final("five"),
            _calls(("write_note", {"text": "five"})),
            _final('{"answer": "5"}'),
        )
        _run(tools, llm, force_final_tool="write_note", repair=True)
        repair = llm.requests[2]
        assert repair.messages[-1] == {"role": "user", "content": _REPAIR_PROMPT}
        assert _FORCE_WRITE_PROMPT not in json.dumps(repair.messages)

    def test_repair_after_a_malformed_loop_fills_the_answer_but_not_success(
        self,
    ) -> None:
        tools = _Tools()
        llm = _ScriptedLLM(_MALFORMED, _final('{"answer": "5"}'))
        api, conv_id, _ = _run(tools, llm, repair=True)
        data = api.get_data(conv_id)
        assert data[_K.ANSWER] == '{"answer": "5"}'
        assert data[ContextKeys.FORCED_STOP_REASON] == StopReason.NO_RESULT
