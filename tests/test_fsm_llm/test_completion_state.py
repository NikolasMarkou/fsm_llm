"""The completion state: ``State.completion`` (plan 944e2692, step 6, D-003, D-021).

A completion state's Pass-1 work is one ``LLMInterface.complete`` call over
``[system(instructions)] + context[messages_key]`` (tools, or a structured
turn); the result ``{kind, text, calls}`` lands in a public result key that
transitions read as ``<result_key>.kind``. Core runs no tool: the FSMs here
run tools in a ``run_tools`` entry handler that appends a ``tool_exchange``
to the transcript and clears the result key. Every FSM starts in a plain
``intake`` state, so the completion state is never the initial state.

On the parent commit (9c34d2f) the field does not exist: pydantic ignores
it, the definition rules do not fire, no ``complete`` call is made and the
result key is never written, so every behaviour test below fails there.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

import fsm_llm
from fsm_llm import API, CompletionStateConfig, State, tool_exchange
from fsm_llm.constants import (
    DEFAULT_COMPLETION_MESSAGES_KEY,
    DEFAULT_COMPLETION_RESULT_KEY,
)
from fsm_llm.definitions import (
    CompletionRequest,
    CompletionResponse,
    FSMDefinition,
    FSMError,
    LLMResponseError,
    ModelToolCall,
    ResponseGenerationResponse,
)
from fsm_llm.handlers import create_handler
from fsm_llm.llm import LiteLLMInterface, LLMInterface, check_tool_transcript
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.validator import FSMValidator
from fsm_llm.visualizer import build_fsm_graph, visualize_fsm_ascii
from tests.conftest import MockLLM2Interface
from tests.test_fsm_llm.test_llm_complete import _call, _Provider, _reply

_SYSTEM = "You answer arithmetic and word questions with the tools you have."
_RESULT = DEFAULT_COMPLETION_RESULT_KEY
_MESSAGES = DEFAULT_COMPLETION_MESSAGES_KEY

_ADD: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}
_LOOKUP: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look a word up in the dictionary.",
        "parameters": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
        },
    },
}
_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
}
_TOOL_FUNCTIONS = {
    "add": lambda a, b: a + b,
    "lookup": lambda word: f"{word}: a word",
}


def _kind_is(kind: str, result_key: str = _RESULT) -> list[dict[str, Any]]:
    return [
        {
            "description": f"The model turn was {kind}",
            "requires_context_keys": [result_key],
            "logic": {"==": [{"var": f"{result_key}.kind"}, kind]},
        }
    ]


def _state(state_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": state_id,
        "description": f"The {state_id} state",
        "purpose": f"Do the {state_id} work",
        "response_instructions": "",
        **extra,
    }


def _tool_fsm(
    *, result_key: str = _RESULT, done_speaks: bool = False, **completion: Any
) -> dict[str, Any]:
    """intake -> call_model (completion) -> run_tools -> call_model ... -> done."""
    config: dict[str, Any] = {"tools": [_ADD, _LOOKUP], "instructions": _SYSTEM}
    config.update(completion)
    if result_key != _RESULT:
        config["result_key"] = result_key
    done = _state("done")
    if done_speaks:
        done["response_instructions"] = "Tell the user the answer."
    return {
        "name": "ToolTurns",
        "description": "A model turn with tools, a tool turn, a final answer",
        "initial_state": "intake",
        "states": {
            "intake": _state(
                "intake",
                transitions=[
                    {"target_state": "call_model", "description": "Ask the model"}
                ],
            ),
            "call_model": _state(
                "call_model",
                completion=config,
                transitions=[
                    {
                        "target_state": "run_tools",
                        "description": "The model asked for tools",
                        "priority": 10,
                        "conditions": _kind_is("calls", result_key),
                    },
                    {
                        "target_state": "done",
                        "description": "The model answered",
                        "priority": 20,
                        "conditions": _kind_is("final", result_key),
                    },
                    {
                        "target_state": "failed",
                        "description": "The tool turn was malformed",
                        "priority": 30,
                        "conditions": _kind_is("malformed", result_key),
                    },
                ],
            ),
            "run_tools": _state(
                "run_tools",
                transitions=[
                    {"target_state": "call_model", "description": "Back to the model"}
                ],
            ),
            "done": done,
            "failed": _state("failed"),
        },
    }


def _structured_fsm() -> dict[str, Any]:
    fsm = _tool_fsm()
    fsm["states"]["call_model"]["completion"] = {
        "response_format_key": "_answer_format",
        "instructions": _SYSTEM,
    }
    return fsm


class _ScriptedLLM(LLMInterface):
    """Answers ``complete`` from a script; records every request it gets."""

    def __init__(self, *replies: CompletionResponse | BaseException) -> None:
        self.replies = list(replies)
        self.requests: list[CompletionRequest] = []
        self.other_calls: list[str] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def generate_response(self, request: Any) -> ResponseGenerationResponse:
        self.other_calls.append("generate_response")
        return ResponseGenerationResponse(message="The answer is 5.")

    def extract_field(self, request: Any) -> Any:
        self.other_calls.append("extract_field")
        raise AssertionError("a completion FSM made a field extraction call")

    def extract_bulk_data(self, request: Any) -> Any:
        self.other_calls.append("extract_bulk_data")
        raise AssertionError("a completion FSM made a bulk extraction call")


def _calls(*calls: tuple[str, dict[str, Any]]) -> CompletionResponse:
    return CompletionResponse(
        kind="calls",
        calls=tuple(
            ModelToolCall(id=f"call_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls, start=1)
        ),
    )


def _final(text: str) -> CompletionResponse:
    return CompletionResponse(kind="final", text=text)


def _tool_runner(log: list[str] | None = None, result_key: str = _RESULT) -> Any:
    """The consumer's ``run_tools`` entry handler: run every call of the
    result, append the paired exchange, clear the result key."""

    def run(context: dict[str, Any]) -> dict[str, Any]:
        result = context[result_key]
        outputs = [
            str(_TOOL_FUNCTIONS[call["name"]](**call["arguments"]))
            for call in result["calls"]
        ]
        if log is not None:
            log.extend(call["name"] for call in result["calls"])
        transcript = list(context.get(_MESSAGES) or [])
        transcript.extend(tool_exchange(result["text"], result["calls"], outputs))
        return {_MESSAGES: transcript, result_key: None}

    return create_handler("run_tools").on_state_entry("run_tools").do(run)


def _api(
    fsm: dict[str, Any], llm: LLMInterface, *, tool_log: list[str] | None = None
) -> API:
    api = API.from_definition(fsm, llm_interface=llm)
    api.register_handler(_tool_runner(tool_log))
    return api


def _start(api: API, question: str = "What is 2 + 3?", **extra: Any) -> str:
    conv_id, greeting = api.start_conversation(
        initial_context={_MESSAGES: [{"role": "user", "content": question}], **extra}
    )
    assert greeting == ""
    return conv_id


def _context(api: API, conv_id: str) -> dict[str, Any]:
    return api.fsm_manager.get_complete_conversation(conv_id)["collected_data"]


def _to_call_model(api: API, conv_id: str) -> None:
    step = api.advance(conv_id)
    assert (step.state_before, step.state_after) == ("intake", "call_model")


# --------------------------------------------------------------
# Definition contract
# --------------------------------------------------------------


class TestCompletionStateDefinition:
    """``State.completion`` is optional and additive; its rules hold at load."""

    def test_field_loads_with_its_defaults(self):
        state = FSMDefinition(**_tool_fsm()).states["call_model"]
        assert isinstance(state.completion, CompletionStateConfig)
        assert state.completion.messages_key == "_completion_messages"
        assert state.completion.result_key == "completion_result"
        assert state.completion.tool_choice is None
        assert state.completion.response_format_key is None

    def test_v41_unchanged_for_definitions_without_it(self):
        fsm = FSMDefinition(**_tool_fsm())
        assert fsm.version == "4.1"
        assert fsm.states["intake"].completion is None
        plain = State(id="s", description="d", purpose="p")
        assert plain.completion is None

    def test_config_is_frozen(self):
        config = CompletionStateConfig(tools=[_ADD])
        with pytest.raises(ValidationError):
            config.tools = None

    @pytest.mark.parametrize(
        "completion, message",
        [
            ({}, "exactly one"),
            ({"tools": [_ADD], "response_format_key": "_f"}, "exactly one"),
            ({"tools": []}, "at least 1"),
            ({"tools": [{"name": "add"}]}, "not an OpenAI function schema"),
            (
                {"tools": [{"type": "function", "function": {"name": ""}}]},
                "not an OpenAI function schema",
            ),
            ({"tools": [_ADD, _ADD]}, "more than once"),
            ({"tools": [_ADD], "tool_choice": "lookup"}, "nor a declared tool"),
            (
                {"response_format_key": "_f", "tool_choice": "auto"},
                "requires tools",
            ),
            ({"tools": [_ADD], "messages_key": "transcript"}, "internal-prefixed"),
            ({"response_format_key": "answer_format"}, "internal-prefixed"),
            ({"tools": [_ADD], "result_key": "_result"}, "public"),
            ({"tools": [_ADD], "result_key": "agent_trace"}, "public"),
            ({"tools": [_ADD], "result_key": "  "}, "public"),
            ({"tools": [_ADD], "tool_turn": True}, "Extra inputs"),
        ],
    )
    def test_config_rules(self, completion: dict[str, Any], message: str):
        with pytest.raises(ValidationError, match=message):
            CompletionStateConfig(**completion)

    @pytest.mark.parametrize("choice", ["auto", "required", "none", "add"])
    def test_tool_choice_accepts_keywords_and_declared_names(self, choice: str):
        config = CompletionStateConfig(tools=[_ADD], tool_choice=choice)
        assert config.tool_choice == choice

    @pytest.mark.parametrize(
        "field, value",
        [
            (
                "field_extractions",
                [
                    {
                        "field_name": "x",
                        "field_type": "str",
                        "extraction_instructions": "X",
                    }
                ],
            ),
            (
                "classification_extractions",
                [
                    {
                        "field_name": "intent",
                        "intents": [
                            {"name": "a", "description": "A"},
                            {"name": "b", "description": "B"},
                        ],
                        "fallback_intent": "a",
                    }
                ],
            ),
            ("required_context_keys", ["answer"]),
            ("extraction_instructions", "Extract the answer."),
        ],
    )
    def test_state_refuses_another_extraction_channel(self, field: str, value: Any):
        fsm = _tool_fsm()
        fsm["states"]["call_model"][field] = value
        with pytest.raises(ValidationError, match="cannot also declare"):
            FSMDefinition(**fsm)

    def test_terminal_completion_state_is_refused(self):
        fsm = _tool_fsm()
        fsm["states"]["done"]["completion"] = {"tools": [_ADD]}
        with pytest.raises(ValidationError, match="needs a transition"):
            FSMDefinition(**fsm)

    def test_validator_graph_and_ascii_accept_the_field(self):
        fsm = _tool_fsm(tool_choice="add")
        result = FSMValidator(fsm).validate()
        assert result.is_valid, result.errors
        assert not [w for w in result.warnings if "completion" in w]
        graph = build_fsm_graph(fsm)
        assert {n.id for n in graph.nodes} == set(fsm["states"])
        assert "call_model" in visualize_fsm_ascii(fsm)

    def test_validator_reports_a_broken_completion_as_an_error(self):
        fsm = _tool_fsm(tool_choice="multiply")
        result = FSMValidator(fsm).validate()
        assert not result.is_valid
        assert any("tool_choice" in e for e in result.errors)

    def test_public_names(self):
        assert "CompletionStateConfig" in fsm_llm.__all__
        assert "tool_exchange" in fsm_llm.__all__


# --------------------------------------------------------------
# tool_exchange and the pairing check
# --------------------------------------------------------------


class TestToolExchange:
    def test_builds_the_provider_shape(self):
        calls = [
            ModelToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3}),
            {"id": "call_2", "name": "lookup", "arguments": {"word": "x"}},
        ]
        messages = tool_exchange(None, calls, ["5", "x: a word"])
        assert messages == [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"word": "x"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "5"},
            {"role": "tool", "tool_call_id": "call_2", "content": "x: a word"},
        ]
        check_tool_transcript(messages)

    def test_text_beside_the_calls_is_kept(self):
        call = ModelToolCall(id="c", name="add", arguments={})
        assert tool_exchange("Let me add.", [call], ["0"])[0]["content"] == (
            "Let me add."
        )
        assert tool_exchange("", [call], ["0"])[0]["content"] is None

    @pytest.mark.parametrize(
        "calls, results",
        [
            ([], []),
            ([{"id": "c", "name": "add", "arguments": {}}], []),
            ([{"id": "c", "name": "", "arguments": {}}], ["0"]),
            ([{"id": "c", "name": "add", "arguments": "{}"}], ["0"]),
            ([{"id": "c", "name": "add", "arguments": {}}], [0]),
        ],
    )
    def test_refuses_what_it_cannot_pair(self, calls: list[Any], results: list[Any]):
        with pytest.raises(ValueError):
            tool_exchange(None, calls, results)

    @pytest.mark.parametrize(
        "transcript",
        [
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_call("add", "{}")],
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_call("add", "{}"), _call("add", "{}", "call_2")],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "0"},
            ],
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_call("add", "{}")],
                },
                {"role": "tool", "tool_call_id": "call_9", "content": "0"},
            ],
            [{"role": "tool", "tool_call_id": "call_1", "content": "0"}],
            [{"role": "user", "content": "hi"}, "not a message"],
            [{"role": "narrator", "content": "hi"}],
            [{"role": "assistant", "content": None, "tool_calls": "add"}],
        ],
    )
    def test_check_refuses_an_unpaired_transcript(self, transcript: list[Any]):
        with pytest.raises(LLMResponseError):
            check_tool_transcript(transcript)


# --------------------------------------------------------------
# The turn (scripted interface)
# --------------------------------------------------------------


class TestCompletionTurn:
    def test_sends_exactly_instructions_plus_transcript(self):
        llm = _ScriptedLLM(_calls(("add", {"a": 2, "b": 3})))
        api = _api(_tool_fsm(), llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        assert llm.requests == []  # entering the state makes no call

        step = api.advance(conv_id)

        (request,) = llm.requests
        assert request.messages == [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "What is 2 + 3?"},
        ]
        assert request.tools == [_ADD, _LOOKUP]
        assert request.tool_choice is None
        assert request.response_format is None
        assert llm.other_calls == []  # no extraction, no Pass 2 (silent)
        assert (step.state_after, step.response) == ("run_tools", None)

    def test_result_is_written_and_drives_the_transition(self):
        llm = _ScriptedLLM(_calls(("add", {"a": 2, "b": 3})))
        api = API.from_definition(_tool_fsm(), llm_interface=llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        api.advance(conv_id)
        assert api.get_current_state(conv_id) == "run_tools"
        assert _context(api, conv_id)[_RESULT] == {
            "kind": "calls",
            "text": None,
            "calls": [{"id": "call_1", "name": "add", "arguments": {"a": 2, "b": 3}}],
        }

    @pytest.mark.parametrize(
        "reply, target",
        [(_final("5"), "done"), (CompletionResponse(kind="malformed"), "failed")],
    )
    def test_transitions_read_the_result_kind(
        self, reply: CompletionResponse, target: str
    ):
        llm = _ScriptedLLM(reply)
        tool_log: list[str] = []
        api = _api(_tool_fsm(), llm, tool_log=tool_log)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        step = api.advance(conv_id)
        assert step.state_after == target
        assert step.ended
        assert tool_log == []
        assert api.get_data(conv_id)[_RESULT]["kind"] == reply.kind

    def test_malformed_turn_carries_no_calls_and_runs_none(self):
        llm = _ScriptedLLM(CompletionResponse(kind="malformed", text="garbled"))
        tool_log: list[str] = []
        api = _api(_tool_fsm(), llm, tool_log=tool_log)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        api.advance(conv_id)
        assert api.get_data(conv_id)[_RESULT] == {
            "kind": "malformed",
            "text": "garbled",
            "calls": [],
        }
        assert tool_log == []

    def test_named_tool_choice_is_sent_as_a_function_choice(self):
        llm = _ScriptedLLM(_calls(("lookup", {"word": "x"})))
        api = _api(_tool_fsm(tool_choice="lookup"), llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        api.advance(conv_id)
        assert llm.requests[0].tool_choice == {
            "type": "function",
            "function": {"name": "lookup"},
        }

    def test_keyword_tool_choice_is_sent_as_given(self):
        llm = _ScriptedLLM(_calls(("lookup", {"word": "x"})))
        api = _api(_tool_fsm(tool_choice="required"), llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        api.advance(conv_id)
        assert llm.requests[0].tool_choice == "required"

    def test_no_instructions_sends_no_system_message(self):
        llm = _ScriptedLLM(_final("5"))
        api = _api(_tool_fsm(instructions=None), llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        api.advance(conv_id)
        assert llm.requests[0].messages == [
            {"role": "user", "content": "What is 2 + 3?"}
        ]

    def test_skip_if_set_makes_no_call(self):
        llm = _ScriptedLLM()
        api = _api(_tool_fsm(), llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        api.update_context(conv_id, {_RESULT: {"kind": "final", "text": "set"}})
        step = api.advance(conv_id)
        assert llm.requests == []
        assert step.state_after == "done"

    def test_result_is_committed_without_cleaning(self):
        reply = _calls(("add", {"_id": 1, "a": None, "b": {"_nested": None}}))
        llm = _ScriptedLLM(reply)
        api = API.from_definition(_tool_fsm(), llm_interface=llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        api.advance(conv_id)
        result = _context(api, conv_id)[_RESULT]
        assert result["text"] is None
        assert result["calls"][0]["arguments"] == {
            "_id": 1,
            "a": None,
            "b": {"_nested": None},
        }

    def test_transcript_is_neither_written_nor_aliased(self):
        llm = _ScriptedLLM(_calls(("add", {"a": 2, "b": 3})))
        api = API.from_definition(_tool_fsm(), llm_interface=llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        before = copy.deepcopy(_context(api, conv_id)[_MESSAGES])
        api.advance(conv_id)
        assert _context(api, conv_id)[_MESSAGES] == before
        llm.requests[0].messages[1]["content"] = "mutated"
        assert _context(api, conv_id)[_MESSAGES] == before


class TestCompletionTurnFailures:
    def _ready(self, llm: LLMInterface, **context: Any) -> tuple[API, str]:
        api = _api(_tool_fsm(), llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        if context:
            api.update_context(conv_id, context)
        return api, conv_id

    def test_outage_raises_and_rolls_back(self):
        llm = _ScriptedLLM(LLMResponseError("Completion call failed: down"))
        api, conv_id = self._ready(llm)
        before = copy.deepcopy(_context(api, conv_id))
        with pytest.raises(LLMResponseError, match="down"):
            api.advance(conv_id)
        assert api.get_current_state(conv_id) == "call_model"
        after = _context(api, conv_id)
        assert _RESULT not in after
        assert after[_MESSAGES] == before[_MESSAGES]

    def test_converse_outage_pops_the_user_message(self):
        llm = _ScriptedLLM(LLMResponseError("Completion call failed: down"))
        api, conv_id = self._ready(llm)
        with pytest.raises(LLMResponseError):
            api.converse("hello", conv_id)
        assert api.get_conversation_history(conv_id) == []
        assert api.get_current_state(conv_id) == "call_model"

    def test_interface_without_complete_is_an_llm_response_error(self):
        api, conv_id = self._ready(MockLLM2Interface())
        with pytest.raises(LLMResponseError, match="MockLLM2Interface"):
            api.advance(conv_id)
        assert api.get_current_state(conv_id) == "call_model"

    def test_orphan_tool_call_is_refused_before_any_call(self):
        llm = _ScriptedLLM(_final("never sent"))
        orphan = [
            {"role": "user", "content": "What is 2 + 3?"},
            {"role": "assistant", "content": None, "tool_calls": [_call("add", "{}")]},
        ]
        api, conv_id = self._ready(llm, **{_MESSAGES: orphan})
        with pytest.raises(LLMResponseError, match="unpaired"):
            api.advance(conv_id)
        assert llm.requests == []
        assert api.get_current_state(conv_id) == "call_model"

    def test_transcript_that_is_not_a_list_is_refused(self):
        llm = _ScriptedLLM(_final("never sent"))
        api, conv_id = self._ready(llm, **{_MESSAGES: "What is 2 + 3?"})
        with pytest.raises(LLMResponseError, match="not a list"):
            api.advance(conv_id)
        assert llm.requests == []

    def test_other_interface_errors_fail_the_turn(self):
        llm = _ScriptedLLM(RuntimeError("bug in a custom interface"))
        api, conv_id = self._ready(llm)
        with pytest.raises(FSMError, match="bug in a custom interface"):
            api.advance(conv_id)
        assert _RESULT not in _context(api, conv_id)


class TestStructuredTurn:
    def test_sends_the_response_format_and_no_tools(self):
        llm = _ScriptedLLM(_final('{"answer": "5"}'))
        api = _api(_structured_fsm(), llm)
        conv_id = _start(api, _answer_format=_FORMAT)
        _to_call_model(api, conv_id)
        step = api.advance(conv_id)
        (request,) = llm.requests
        assert request.response_format == _FORMAT
        assert request.tools is None and request.tool_choice is None
        assert step.state_after == "done"
        assert api.get_data(conv_id)[_RESULT]["text"] == '{"answer": "5"}'

    @pytest.mark.parametrize("value", [None, {}, "json"])
    def test_missing_response_format_is_refused(self, value: Any):
        llm = _ScriptedLLM(_final("never sent"))
        api = _api(_structured_fsm(), llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        api.update_context(conv_id, {"_answer_format": value})
        with pytest.raises(LLMResponseError, match="response format"):
            api.advance(conv_id)
        assert llm.requests == []


class TestOwnedKeysAreNeverMinted:
    def test_completion_state_gets_no_field_configs(self):
        state = FSMDefinition(**_tool_fsm()).states["call_model"]
        assert MessagePipeline._build_field_configs_from_state(state) == []

    def test_result_key_is_handler_only_for_every_state(self):
        """A plain state gated on the result key must not ask the extractor
        for it: the key belongs to the completion state's call alone."""
        fsm = _tool_fsm(result_key="answer_turn")
        fsm["states"]["intake"]["transitions"] = [
            {
                "target_state": "call_model",
                "description": "A previous answer exists",
                "priority": 10,
                "conditions": _kind_is("final", "answer_turn"),
            },
            {"target_state": "call_model", "description": "Ask", "priority": 20},
        ]
        fsm["states"]["intake"]["extraction_instructions"] = "Extract anything."
        llm = MockLLM2Interface(extraction_data={"answer_turn": {"kind": "final"}})
        api = API.from_definition(fsm, llm_interface=llm)
        conv_id, _ = api.start_conversation()
        api.converse("my answer is final", conv_id)
        extracted = [
            request.field_name
            for name, request in llm.call_history
            if name == "extract_field"
        ]
        assert "answer_turn" not in extracted
        assert "answer_turn" not in api.get_data(conv_id)


class TestEntryPoints:
    def test_converse_does_not_add_the_user_message_to_the_transcript(self):
        llm = _ScriptedLLM(_final("5"))
        api = _api(_tool_fsm(done_speaks=True), llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        reply = api.converse("please hurry", conv_id)
        (request,) = llm.requests
        assert all(
            "please hurry" not in str(m.get("content")) for m in request.messages
        )
        assert reply == "The answer is 5."
        assert api.get_conversation_history(conv_id)[0] == {"user": "please hurry"}

    def test_converse_on_a_silent_completion_state_returns_empty(self):
        llm = _ScriptedLLM(_calls(("add", {"a": 2, "b": 3})))
        api = _api(_tool_fsm(), llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        assert api.converse("go", conv_id) == ""
        assert llm.other_calls == []

    def test_advance_stream_runs_the_turn_silently(self):
        llm = _ScriptedLLM(_calls(("add", {"a": 2, "b": 3})))
        api = _api(_tool_fsm(), llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        assert list(api.advance_stream(conv_id)) == []
        assert api.get_current_state(conv_id) == "run_tools"
        assert len(llm.requests) == 1
        transcript = _context(api, conv_id)[_MESSAGES]
        assert [m["role"] for m in transcript] == ["user", "assistant", "tool"]

    def test_run_until_terminal_over_a_tool_loop(self):
        llm = _ScriptedLLM(
            _calls(("add", {"a": 2, "b": 3}), ("lookup", {"word": "five"})),
            _final("2 + 3 = 5"),
        )
        tool_log: list[str] = []
        api = _api(_tool_fsm(), llm, tool_log=tool_log)
        conv_id = _start(api)
        steps = api.run_until_terminal(conv_id, max_steps=10)
        assert [s.state_after for s in steps] == [
            "call_model",
            "run_tools",
            "call_model",
            "done",
        ]
        assert tool_log == ["add", "lookup"]
        assert api.get_data(conv_id)[_RESULT] == {
            "kind": "final",
            "text": "2 + 3 = 5",
            "calls": [],
        }
        second = llm.requests[1].messages
        assert second[:2] == [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "What is 2 + 3?"},
        ]
        assert second[2]["role"] == "assistant"
        assert [c["function"]["name"] for c in second[2]["tool_calls"]] == [
            "add",
            "lookup",
        ]
        assert second[3:] == [
            {"role": "tool", "tool_call_id": "call_1", "content": "5"},
            {"role": "tool", "tool_call_id": "call_2", "content": "five: a word"},
        ]
        check_tool_transcript(second)
        assert llm.other_calls == []

    def test_run_until_terminal_stream_over_a_tool_loop(self):
        llm = _ScriptedLLM(_calls(("add", {"a": 2, "b": 3})), _final("5"))
        api = _api(_tool_fsm(done_speaks=True), llm)
        conv_id = _start(api)
        chunks = list(api.run_until_terminal_stream(conv_id, max_steps=10))
        assert "".join(chunks)
        assert api.has_conversation_ended(conv_id)
        assert len(llm.requests) == 2


class TestThroughLiteLLMInterface:
    """The real request path: builder, Ollama preparation, usage meter."""

    def _run(self, model: str, *replies: Any) -> tuple[LiteLLMInterface, _Provider]:
        llm = LiteLLMInterface(model)
        with _Provider(*replies) as provider:
            api = _api(_tool_fsm(), llm)
            conv_id = _start(api)
            api.run_until_terminal(conv_id, max_steps=10)
        return llm, provider

    def test_usage_is_counted_once_per_model_turn(self):
        llm, provider = self._run(
            "gpt-4o",
            _reply(None, [_call("add", '{"a": 2, "b": 3}')]),
            _reply("5"),
        )
        assert len(provider.calls) == 2
        usage = llm.usage()
        assert usage.calls == 2
        assert usage.by_kind["complete"].calls == 2

    def test_ollama_requests_carry_tools_and_nothink_on_the_last_user_turn(self):
        _, provider = self._run(
            "ollama_chat/qwen3.5:4b",
            _reply(None, [_call("add", '{"a": 2, "b": 3}')]),
            _reply("5"),
        )
        first, second = provider.calls
        for sent in (first, second):
            assert "response_format" not in sent
            assert sent["tools"] == [_ADD, _LOOKUP]
            assert sent["tool_choice"] == "auto"
            assert sent["reasoning_effort"] == "none"
            assert sent["temperature"] == 0.5
        assert first["messages"] == [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "/nothink\nWhat is 2 + 3?"},
        ]
        assert second["messages"][1] == {
            "role": "user",
            "content": "/nothink\nWhat is 2 + 3?",
        }
        assert second["messages"][2]["content"] is None
        assert second["messages"][3] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "5",
        }

    def test_structured_turn_on_ollama_echoes_the_schema_at_temperature_zero(self):
        llm = LiteLLMInterface("ollama_chat/qwen3.5:4b")
        with _Provider(_reply('{"answer": "5"}')) as provider:
            api = _api(_structured_fsm(), llm)
            conv_id = _start(api, _answer_format=_FORMAT)
            api.run_until_terminal(conv_id, max_steps=5)
        (sent,) = provider.calls
        assert "tools" not in sent
        assert sent["response_format"] == _FORMAT
        assert sent["temperature"] == 0
        last = sent["messages"][-1]["content"]
        assert last.startswith("/nothink\nWhat is 2 + 3?")
        assert "Respond in JSON matching this schema" in last
        assert api.get_data(conv_id)[_RESULT]["text"] == '{"answer": "5"}'

    def test_provider_malformed_tool_call_is_a_malformed_result(self):
        llm = LiteLLMInterface("ollama_chat/qwen3.5:4b")
        error = RuntimeError("XML syntax error: element <function> closed")
        with _Provider(error=error):
            api = _api(_tool_fsm(), llm)
            conv_id = _start(api)
            steps = api.run_until_terminal(conv_id, max_steps=5)
        assert steps[-1].state_after == "failed"
        assert api.get_data(conv_id)[_RESULT] == {
            "kind": "malformed",
            "text": None,
            "calls": [],
        }
        assert llm.usage().errors == 1
