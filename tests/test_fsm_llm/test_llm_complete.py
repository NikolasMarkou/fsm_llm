"""``LLMInterface.complete``: the one request primitive of the LLM layer.

Covers the request model (``CompletionRequest``), the reply model
(``CompletionResponse``), the request rules held once in
``LiteLLMInterface._build_call_params`` and the reply normaliser
(plan 944e2692, step 2, D-002). Provider replies are real litellm
``ModelResponse`` objects; the provider is ``fsm_llm.llm.providers.litellm.completion``,
patched.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse
from pydantic import ValidationError

import fsm_llm
import fsm_llm.llm as llm_module
from fsm_llm import constants
from fsm_llm.constants import RESERVED_LLM_CALL_KWARGS
from fsm_llm.core.definitions import (
    CompletionRequest,
    CompletionResponse,
    FieldExtractionRequest,
    FSMError,
    LLMResponseError,
    ModelToolCall,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.engine.classification import Classifier
from fsm_llm.llm import (
    LiteLLMInterface,
    LLMInterface,
    decode_tool_arguments,
    is_malformed_tool_call_error,
)
from fsm_llm.logging import logger
from tests.test_fsm_llm.support import _INTENTS

OLLAMA_MODEL = "ollama_chat/qwen3.5:4b"

_ADD_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}
_LOOKUP_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look a word up",
        "parameters": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
        },
    },
}
_SCHEMA_FORMAT: dict[str, Any] = {
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

# A transcript after one tool round: the assistant tool-call message has
# ``content: None`` (its normal provider shape) and is paired with its result.
_TRANSCRIPT: list[dict[str, Any]] = [
    {"role": "system", "content": "You add numbers with the add tool."},
    {"role": "user", "content": "What is 2 + 3?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "5"},
]


def _reply(
    content: Any = None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning: str | None = None,
) -> ModelResponse:
    message = Message(
        content=content, tool_calls=tool_calls, reasoning_content=reasoning
    )
    return ModelResponse(choices=[Choices(message=message, finish_reason="stop")])


def _call(name: str, arguments: Any, call_id: str = "call_1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class _Provider:
    """Patch the one provider binding; record every request it receives."""

    def __init__(
        self,
        *replies: Any,
        supported: list[str] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.replies = list(replies) or [_reply("ok")]
        self.supported = (
            ["response_format", "tools", "tool_choice"]
            if supported is None
            else supported
        )
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self._patches: list[Any] = []

    def _completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.replies[min(len(self.calls), len(self.replies)) - 1]

    def __enter__(self) -> _Provider:
        self._patches = [
            patch(
                "fsm_llm.llm.providers.litellm.completion", side_effect=self._completion
            ),
            patch(
                "fsm_llm.llm.providers.litellm.get_supported_openai_params",
                return_value=self.supported,
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for p in reversed(self._patches):
            p.stop()


def _complete(
    request: CompletionRequest, *replies: Any, model: str = "gpt-4o", **kw: Any
) -> tuple[CompletionResponse, dict[str, Any]]:
    supported = kw.pop("supported", None)
    with _Provider(*replies, supported=supported) as provider:
        response = LiteLLMInterface(model, **kw).complete(request)
    assert len(provider.calls) == 1
    return response, provider.calls[0]


# --------------------------------------------------------------
# RED on the parent commit (acc8737)
# --------------------------------------------------------------


class TestBuilderKeepsToolMessagesAndRequestScopedKeys:
    """The two builder rules the parent broke."""

    def test_assistant_tool_call_message_passes_through_unchanged(self):
        """Parent: ``_fill_empty_user_turns`` turned ``content: None`` into
        ``""`` for every role, so a tool-call message lost its provider shape."""
        interface = LiteLLMInterface("gpt-4o")
        with patch(
            "fsm_llm.llm.providers.litellm.get_supported_openai_params", return_value=[]
        ):
            params = interface._build_call_params(_TRANSCRIPT, "completion")
        assert params["messages"] == _TRANSCRIPT
        assert params["messages"][2]["content"] is None
        assert params["messages"][2]["tool_calls"] == _TRANSCRIPT[2]["tool_calls"]

    def test_constructor_tools_never_reach_an_extraction_call(self):
        """Parent: ``tools``/``tool_choice`` were not reserved, so a constructor
        kwarg put a tool surface on every Pass-1 call."""
        interface = LiteLLMInterface(
            "gpt-4o", tools=[_ADD_TOOL], tool_choice="required"
        )
        with _Provider(_reply('{"value": "Ada", "confidence": 0.9}')) as provider:
            interface.extract_field(
                FieldExtractionRequest(
                    system_prompt="Extract the name.",
                    user_message="I am Ada",
                    field_name="name",
                    field_type="str",
                )
            )
        assert "tools" not in provider.calls[0]
        assert "tool_choice" not in provider.calls[0]


# --------------------------------------------------------------
# Request and reply models
# --------------------------------------------------------------


class TestCompletionRequestModel:
    def test_tools_and_response_format_together_are_refused(self):
        with pytest.raises(ValidationError, match="not both"):
            CompletionRequest(
                messages=_TRANSCRIPT,
                tools=[_ADD_TOOL],
                response_format=_SCHEMA_FORMAT,
            )

    def test_tool_choice_without_tools_is_refused(self):
        with pytest.raises(ValidationError, match="tool_choice requires tools"):
            CompletionRequest(messages=_TRANSCRIPT, tool_choice="required")

    @pytest.mark.parametrize(
        "fields",
        [
            {"messages": []},
            {"messages": _TRANSCRIPT, "tools": []},
            {"messages": _TRANSCRIPT, "temperature": 2.5},
            {"messages": _TRANSCRIPT, "max_tokens": 0},
            {"messages": _TRANSCRIPT, "call_type": ""},
            {"messages": _TRANSCRIPT, "stream": True},
        ],
    )
    def test_invalid_fields_are_refused(self, fields):
        with pytest.raises(ValidationError):
            CompletionRequest(**fields)

    def test_request_is_frozen(self):
        request = CompletionRequest(messages=_TRANSCRIPT)
        with pytest.raises(ValidationError):
            request.tools = [_ADD_TOOL]
        assert request.call_type == "completion"


class TestCompletionResponseModel:
    def test_calls_kind_needs_calls(self):
        with pytest.raises(ValidationError):
            CompletionResponse(kind="calls")

    @pytest.mark.parametrize("kind", ["final", "malformed"])
    def test_other_kinds_carry_no_calls(self, kind):
        call = ModelToolCall(id="c", name="add", arguments={})
        with pytest.raises(ValidationError):
            CompletionResponse(kind=kind, calls=(call,))

    def test_tool_call_needs_a_name(self):
        with pytest.raises(ValidationError):
            ModelToolCall(id="c", name="", arguments={})

    def test_public_names_are_exported(self):
        for name in ("CompletionRequest", "CompletionResponse", "ModelToolCall"):
            assert name in fsm_llm.__all__
            assert getattr(fsm_llm, name) is getattr(llm_module, name)


# --------------------------------------------------------------
# Request rules (the builder)
# --------------------------------------------------------------


class TestCompleteRequestRules:
    def test_tools_are_sent_with_auto_tool_choice_by_default(self):
        _, sent = _complete(
            CompletionRequest(messages=_TRANSCRIPT[:2], tools=[_ADD_TOOL]),
            _reply("5"),
        )
        assert sent["tools"] == [_ADD_TOOL]
        assert sent["tool_choice"] == "auto"
        assert "response_format" not in sent

    def test_a_named_tool_choice_is_sent_as_given(self):
        choice = {"type": "function", "function": {"name": "lookup"}}
        _, sent = _complete(
            CompletionRequest(
                messages=_TRANSCRIPT[:2],
                tools=[_ADD_TOOL, _LOOKUP_TOOL],
                tool_choice=choice,
            ),
            _reply("x"),
        )
        assert sent["tool_choice"] == choice
        assert "response_format" not in sent

    def test_a_structured_request_carries_no_tool_keys(self):
        """native_fc's repair turn: the schema and NO tool surface at all, not
        merely ``tools: None`` (bf7ffe24/D-002, carried by D-027)."""
        _, sent = _complete(
            CompletionRequest(messages=_TRANSCRIPT, response_format=_SCHEMA_FORMAT),
            _reply('{"answer": "5"}'),
        )
        assert sent["response_format"] == _SCHEMA_FORMAT
        assert "tools" not in sent
        assert "tool_choice" not in sent

    def test_tools_are_never_gated_on_the_supported_params_list(self):
        """litellm lists no tools for ``ollama/`` although calls work."""
        _, sent = _complete(
            CompletionRequest(messages=_TRANSCRIPT[:2], tools=[_ADD_TOOL]),
            _reply("x"),
            model="ollama/qwen3.5:4b",
            supported=[],
        )
        assert sent["tools"] == [_ADD_TOOL]

    def test_ollama_tool_request_shape(self):
        """``/nothink`` on the last user turn only, thinking off, the user's
        temperature kept (a tool turn is not structured), no response_format."""
        messages = [
            *_TRANSCRIPT,
            {"role": "user", "content": "Now call the tool again."},
        ]
        _, sent = _complete(
            CompletionRequest(messages=messages, tools=[_ADD_TOOL]),
            _reply("x"),
            model=OLLAMA_MODEL,
            temperature=0.7,
        )
        users = [m["content"] for m in sent["messages"] if m["role"] == "user"]
        assert users == ["What is 2 + 3?", "/nothink\nNow call the tool again."]
        assert sent["messages"][2]["content"] is None
        assert sent["messages"][2]["tool_calls"] == _TRANSCRIPT[2]["tool_calls"]
        assert sent["messages"][3] == _TRANSCRIPT[3]
        assert sent["reasoning_effort"] == "none"
        assert sent["temperature"] == 0.7
        assert "response_format" not in sent

    def test_ollama_structured_request_runs_at_temperature_zero_with_schema_echo(self):
        _, sent = _complete(
            CompletionRequest(messages=_TRANSCRIPT[:2], response_format=_SCHEMA_FORMAT),
            _reply('{"answer": "5"}'),
            model=OLLAMA_MODEL,
            temperature=0.7,
        )
        assert sent["response_format"] == _SCHEMA_FORMAT
        assert sent["temperature"] == 0
        user = sent["messages"][1]["content"]
        assert user.startswith("/nothink\nWhat is 2 + 3?")
        assert "Respond in JSON matching this schema:" in user

    def test_ollama_preparation_does_not_mutate_the_request_messages(self):
        messages = [dict(m) for m in _TRANSCRIPT[:2]]
        request = CompletionRequest(messages=messages, tools=[_ADD_TOOL])
        _, sent = _complete(request, _reply("x"), model=OLLAMA_MODEL)
        assert sent["messages"][-1]["content"].startswith("/nothink")
        assert request.messages == _TRANSCRIPT[:2]
        assert messages == _TRANSCRIPT[:2]

    def test_non_ollama_request_gets_no_ollama_preparation(self):
        _, sent = _complete(
            CompletionRequest(messages=_TRANSCRIPT[:2], tools=[_ADD_TOOL]),
            _reply("x"),
        )
        assert "reasoning_effort" not in sent
        assert sent["messages"] == _TRANSCRIPT[:2]

    def test_response_format_is_dropped_for_an_unsupported_model(self):
        _, sent = _complete(
            CompletionRequest(messages=_TRANSCRIPT[:2], response_format=_SCHEMA_FORMAT),
            _reply('{"answer": "5"}'),
            supported=[],
        )
        assert "response_format" not in sent

    def test_seed_is_absent_unless_the_interface_has_one(self):
        request = CompletionRequest(messages=_TRANSCRIPT[:2], tools=[_ADD_TOOL])
        _, unseeded = _complete(request, _reply("x"), model=OLLAMA_MODEL)
        _, seeded = _complete(request, _reply("x"), model=OLLAMA_MODEL, seed=7)
        assert "seed" not in unseeded
        assert seeded["seed"] == 7

    def test_per_call_temperature_and_max_tokens(self):
        request = CompletionRequest(
            messages=_TRANSCRIPT[:2], temperature=0.0, max_tokens=64
        )
        _, sent = _complete(request, _reply("x"), temperature=0.9, max_tokens=900)
        assert (sent["temperature"], sent["max_tokens"]) == (0.0, 64)
        _, kept = _complete(
            CompletionRequest(messages=_TRANSCRIPT[:2]),
            _reply("x"),
            temperature=0.9,
            max_tokens=900,
        )
        assert (kept["temperature"], kept["max_tokens"]) == (0.9, 900)

    @pytest.mark.parametrize(
        ("user", "sent_turn"),
        [
            (None, constants.NEUTRAL_USER_TURN),
            ("", constants.EMPTY_USER_MESSAGE_TURN),
            ("  ", constants.EMPTY_USER_MESSAGE_TURN),
        ],
    )
    def test_user_turns_without_text_keep_their_distinct_meanings(
        self, user, sent_turn
    ):
        request = CompletionRequest(
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": user},
            ]
        )
        _, sent = _complete(request, _reply("x"))
        assert sent["messages"][1]["content"] == sent_turn

    def test_reserved_tool_kwargs_are_ignored_with_a_warning(self):
        records: list[str] = []
        sink = logger.add(lambda m: records.append(m.record["message"]))
        logger.enable("fsm_llm")
        try:
            interface = LiteLLMInterface(
                "gpt-4o", tools=[_ADD_TOOL], tool_choice="auto"
            )
        finally:
            logger.disable("fsm_llm")
            logger.remove(sink)
        assert any("['tool_choice', 'tools']" in r for r in records)
        _, sent = _complete(CompletionRequest(messages=_TRANSCRIPT[:2]), _reply("x"))
        assert interface.kwargs["tools"] == [_ADD_TOOL]  # kept, never sent
        assert "tools" not in sent


# --------------------------------------------------------------
# Reply normaliser
# --------------------------------------------------------------


_TOOL_REQUEST = CompletionRequest(messages=_TRANSCRIPT[:2], tools=[_ADD_TOOL])


class TestCompleteReplies:
    def test_a_final_text_reply(self):
        response, _ = _complete(_TOOL_REQUEST, _reply("The answer is 5."))
        assert response == CompletionResponse(
            kind="final", text="The answer is 5.", finish_reason="stop"
        )

    def test_a_tool_call_reply_decodes_arguments(self):
        response, _ = _complete(
            _TOOL_REQUEST, _reply(None, [_call("add", '{"a": 2, "b": 3}')])
        )
        assert response.kind == "calls"
        assert response.text is None
        assert response.calls == (
            ModelToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3}),
        )

    def test_text_beside_tool_calls_is_kept(self):
        response, _ = _complete(
            _TOOL_REQUEST,
            _reply(
                "Let me add.",
                [_call("add", '{"a": 1, "b": 1}'), _call("add", "", "call_2")],
            ),
        )
        assert response.kind == "calls"
        assert response.text == "Let me add."
        assert [c.arguments for c in response.calls] == [{"a": 1, "b": 1}, {}]
        assert [c.id for c in response.calls] == ["call_1", "call_2"]

    @pytest.mark.parametrize("arguments", ["{not json", "[1, 2]", "null", '"5"', "7"])
    def test_arguments_that_are_not_an_object_make_the_turn_malformed(self, arguments):
        response, _ = _complete(
            _TOOL_REQUEST,
            _reply(None, [_call("add", '{"a": 1, "b": 2}'), _call("add", arguments)]),
        )
        assert response.kind == "malformed"
        assert response.calls == ()

    def test_reasoning_only_reply_is_recovered_without_tool_calls(self):
        response, _ = _complete(
            CompletionRequest(messages=_TRANSCRIPT[:2]),
            _reply("", reasoning='{"answer": "5"}'),
        )
        assert response == CompletionResponse(
            kind="final", text='{"answer": "5"}', finish_reason="stop"
        )

    def test_reasoning_is_not_recovered_beside_tool_calls(self):
        response, _ = _complete(
            _TOOL_REQUEST,
            _reply(None, [_call("add", "{}")], reasoning="I should call add"),
        )
        assert response.kind == "calls"
        assert response.text is None

    def test_an_empty_reply_is_a_final_without_text(self):
        response, _ = _complete(_TOOL_REQUEST, _reply(""))
        assert response == CompletionResponse(
            kind="final", text=None, finish_reason="stop"
        )

    def test_a_dict_content_is_returned_as_json_text(self):
        reply = _reply("placeholder")
        reply.choices[0].message.content = {"answer": "5"}
        response, _ = _complete(CompletionRequest(messages=_TRANSCRIPT[:2]), reply)
        assert response.text == '{"answer": "5"}'

    def test_no_choices_is_an_llm_response_error(self):
        with pytest.raises(LLMResponseError, match="Empty response"):
            _complete(_TOOL_REQUEST, ModelResponse(choices=[]))

    def test_an_unreadable_reply_shape_is_not_reported_as_an_outage(self):
        """A reply that cannot be read is named as such, chained from the
        reading error; only the provider call itself is an outage
        (876e7164/D-006 [STALE], reshaped by D-019)."""
        unreadable = type("_Reply", (), {"choices": [object()]})()
        with pytest.raises(
            LLMResponseError, match="Malformed LLM response shape"
        ) as exc:
            _complete(_TOOL_REQUEST, unreadable)
        assert isinstance(exc.value.__cause__, AttributeError)
        assert "Completion call failed" not in str(exc.value)


class TestCompleteFailures:
    @pytest.mark.parametrize(
        "text",
        [
            "Ollama: XML syntax error on line 1",
            "element <function> closed by </parameter>",
            "Invalid tool call in model output",
            # The exact provider error measured 1/35 on qwen3.5:4b
            # (bf7ffe24/D-016).
            'litellm.APIConnectionError: Ollama_chatException - "XML syntax error '
            'on line 5: element <function> closed by </parameter>"',
        ],
    )
    def test_provider_malformed_tool_call_error_is_a_malformed_reply(self, text):
        with _Provider(error=RuntimeError(text)):
            response = LiteLLMInterface(OLLAMA_MODEL).complete(_TOOL_REQUEST)
        assert response == CompletionResponse(kind="malformed")

    def test_the_same_error_without_tools_is_an_outage(self):
        request = CompletionRequest(messages=_TRANSCRIPT[:2])
        with (
            _Provider(error=RuntimeError("XML syntax error")),
            pytest.raises(LLMResponseError) as exc,
        ):
            LiteLLMInterface(OLLAMA_MODEL).complete(request)
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_an_outage_raises_chained(self):
        with (
            _Provider(error=ConnectionError("connection refused")),
            pytest.raises(LLMResponseError, match="connection refused") as exc,
        ):
            LiteLLMInterface("gpt-4o").complete(_TOOL_REQUEST)
        assert isinstance(exc.value.__cause__, ConnectionError)

    def test_malformed_marker_helper_fails_closed(self):
        err = RuntimeError("xml syntax error")
        assert is_malformed_tool_call_error(err, tools_sent=True)
        assert not is_malformed_tool_call_error(err, tools_sent=False)
        assert not is_malformed_tool_call_error(
            RuntimeError("tool server unreachable"), tools_sent=True
        )

    @pytest.mark.parametrize(
        ("raw", "decoded"),
        [
            (None, {}),
            ("", {}),
            ("  ", {}),
            ({"a": 1}, {"a": 1}),
            ('{"a": 1}', {"a": 1}),
            ("{bad", None),
            ("[1]", None),
            ("null", None),
            (42, None),
        ],
    )
    def test_decode_tool_arguments(self, raw, decoded):
        assert decode_tool_arguments(raw) == decoded


# --------------------------------------------------------------
# The interface surface
# --------------------------------------------------------------


class _GenerateOnly(LLMInterface):
    """A third-party interface written before ``complete`` existed."""

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(message="hi")


class TestInterfaceSurface:
    def test_complete_is_not_abstract(self):
        assert "complete" not in LLMInterface.__abstractmethods__
        assert LLMInterface.__abstractmethods__ == frozenset({"generate_response"})

    def test_an_interface_without_complete_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="_GenerateOnly"):
            _GenerateOnly().complete(CompletionRequest(messages=_TRANSCRIPT[:2]))

    def test_one_send_path(self):
        """Every provider request goes through ``_send``: one ``completion``
        call in the whole ``llm`` package (in ``providers/litellm.py``), and no
        per-kind completion method."""
        package_dir = Path(llm_module.__file__).parent
        sends = [
            (path.relative_to(package_dir).as_posix(), node.lineno)
            for path in sorted(package_dir.rglob("*.py"))
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "completion"
        ]
        assert len(sends) == 1
        assert sends[0][0] == "providers/litellm.py"
        assert [n for n in vars(LiteLLMInterface) if n.startswith("complete")] == [
            "complete"
        ]

    def test_native_fc_holds_no_request_or_reply_rule(self):
        """native_fc runs on core (step 16, D-027): the markers, the argument
        rule, the Ollama preparation and the reasoning-trace recovery live
        once, here, and native_fc imports none of them (nor litellm)."""
        from fsm_llm.agents import native_fc

        for name in (
            "decode_tool_arguments",
            "is_malformed_tool_call_error",
            "apply_ollama_params",
            "prepare_ollama_messages",
            "is_ollama_model",
            "resolve_reasoning_trace",
            "_MALFORMED_TOOL_CALL_MARKERS",
        ):
            assert not hasattr(native_fc, name), name
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(inspect.getsource(native_fc)))
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in (
                node.names
                if isinstance(node, ast.Import)
                else [ast.alias(name=node.module or "")]
            )
        }
        assert "litellm" not in imported
        assert constants.MALFORMED_TOOL_CALL_MARKERS


class TestDuplicateToolCallIdsAreMalformed:
    """F-LL-02: refused before any tool runs, like an id-less call."""

    def _request(self) -> CompletionRequest:
        return CompletionRequest(
            messages=[{"role": "user", "content": "add"}], tools=[_ADD_TOOL]
        )

    def test_two_calls_with_one_id_make_the_turn_malformed(self):
        reply = _reply(
            tool_calls=[
                _call("add", '{"a": 1, "b": 2}', "c1"),
                _call("add", '{"a": 3, "b": 4}', "c1"),
            ]
        )
        response, _ = _complete(self._request(), reply)
        assert response.kind == "malformed"
        assert response.calls == ()

    def test_distinct_ids_still_make_a_calls_turn(self):
        reply = _reply(
            tool_calls=[
                _call("add", '{"a": 1, "b": 2}', "c1"),
                _call("add", '{"a": 3, "b": 4}', "c2"),
            ]
        )
        response, _ = _complete(self._request(), reply)
        assert response.kind == "calls"
        assert [c.id for c in response.calls] == ["c1", "c2"]


_USER = [{"role": "user", "content": "What is 2 + 3?"}]


def _sent(llm: LiteLLMInterface, request: CompletionRequest) -> dict[str, Any]:
    """The provider kwargs of one ``complete`` call (supported params incl.
    ``response_format``)."""
    with _Provider(_reply("ok")) as provider:
        llm.complete(request)
    (params,) = provider.calls
    return params


class TestToolRulesHoldWhateverTheCallType:
    """Review area 1, concern 1: the builder sent a ``response_format``
    beside ``tools`` and forced Ollama temperature 0 on a tool turn when the
    free-string ``call_type`` named an extraction or classification."""

    @pytest.mark.parametrize(
        "call_type", ["data_extraction", "field_extraction", "classification"]
    )
    @pytest.mark.parametrize("model", ["ollama_chat/qwen3.5:4b", "gpt-4o"])
    def test_a_tool_turn_gets_no_response_format_and_keeps_its_temperature(
        self, model: str, call_type: str
    ):
        llm = LiteLLMInterface(model, temperature=0.7)
        params = _sent(
            llm,
            CompletionRequest(messages=_USER, tools=[_ADD_TOOL], call_type=call_type),
        )
        assert params["tools"] == [_ADD_TOOL]
        assert "response_format" not in params
        assert params["temperature"] == 0.7
        # Nothing structured leaks into the user turn either (Ollama echo).
        assert "schema" not in params["messages"][-1]["content"].lower()

    def test_the_builder_refuses_tools_beside_a_response_format(self):
        llm = LiteLLMInterface("gpt-4o")
        with (
            _Provider(_reply("ok")),
            pytest.raises(ValueError, match="never both"),
        ):
            llm._build_call_params(
                _USER,
                "completion",
                tools=[_ADD_TOOL],
                response_format={"type": "json_object"},
            )

    def test_a_structured_turn_on_ollama_still_runs_at_temperature_zero(self):
        llm = LiteLLMInterface("ollama_chat/qwen3.5:4b", temperature=0.7)
        params = _sent(
            llm,
            CompletionRequest(messages=_USER, response_format={"type": "json_object"}),
        )
        assert params["temperature"] == 0


class TestCompleteBuildsInsideItsErrorBoundary:
    """Review area 1, concern 8: a builder error escaped ``complete`` raw."""

    def test_a_failing_supported_params_lookup_is_an_llm_response_error(self):
        llm = LiteLLMInterface("gpt-4o")
        with (
            patch(
                "fsm_llm.llm.providers.litellm.get_supported_openai_params",
                side_effect=ConnectionError("registry down"),
            ),
            patch("fsm_llm.llm.providers.litellm.completion") as provider,
            pytest.raises(LLMResponseError, match="registry down") as caught,
        ):
            llm.complete(CompletionRequest(messages=_USER))
        assert isinstance(caught.value.__cause__, ConnectionError)
        provider.assert_not_called()


class TestReservedCallKwargs:
    """Review area 1, concern 11: the legacy tool kwargs and ``n`` set on a
    constructor reached every extraction, classifier and Pass-2 call."""

    @pytest.mark.parametrize("name", ["functions", "function_call", "n"])
    def test_constructor_kwarg_never_reaches_a_request(self, name: str):
        assert name in RESERVED_LLM_CALL_KWARGS
        llm = LiteLLMInterface("gpt-4o", **{name: [{"name": "f"}]})
        params = _sent(llm, CompletionRequest(messages=_USER))
        assert name not in params


class _Garbled:
    """A provider reply whose message raises from ``content``."""

    @property
    def content(self) -> Any:
        raise RuntimeError("garbled message object")


class TestCompleteIsTotal:
    """Pass 8 warning 3: a garbled reply raised a raw TypeError out of
    ``complete`` (and so out of the classifier) instead of LLMResponseError."""

    @pytest.mark.parametrize("tool_calls", [5, "call_1", {"id": "call_1"}, 2.5])
    def test_a_tool_calls_value_that_is_not_a_list_is_a_response_error(
        self, tool_calls
    ):
        reply = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="x", tool_calls=tool_calls)
                )
            ]
        )
        with _Provider(reply), pytest.raises(LLMResponseError, match="tool_calls"):
            LiteLLMInterface("gpt-4o").complete(CompletionRequest(messages=_USER))

    def test_an_unforeseen_reply_error_is_a_response_error(self):
        reply = SimpleNamespace(choices=[SimpleNamespace(message=_Garbled())])
        with _Provider(reply), pytest.raises(LLMResponseError, match="garbled"):
            LiteLLMInterface("gpt-4o").complete(CompletionRequest(messages=_USER))

    def test_the_classifier_turns_a_garbled_reply_into_a_classification_error(self):
        reply = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="x", tool_calls=5))
            ]
        )
        classifier = Classifier(_INTENTS, llm_interface=LiteLLMInterface("gpt-4o"))
        with _Provider(reply), pytest.raises(FSMError) as caught:
            classifier.classify("I want to buy it")
        assert type(caught.value).__name__ == "ClassificationError"


class TestAnIdLessToolCallIsMalformed:
    """Pass 8 note 10: a call without an id was returned as a runnable call
    (id ``""``), so its tool ran before the next turn refused the transcript."""

    @pytest.mark.parametrize("call_id", [None, "", 7])
    def test_the_turn_is_malformed_with_no_calls(self, call_id):
        calls = [
            _call("add", '{"a": 2, "b": 3}', call_id="call_ok"),
            {**_call("add", '{"a": 1, "b": 1}'), "id": call_id},
        ]
        provider_reply = _reply(None, tool_calls=calls)
        # litellm mints an id for a None one; read the shape a non-litellm
        # reply object would carry instead.
        provider_reply.choices[0].message.tool_calls[1].id = call_id
        with _Provider(provider_reply):
            reply = LiteLLMInterface("gpt-4o").complete(
                CompletionRequest(messages=_USER, tools=[_ADD_TOOL])
            )
        assert reply.kind == "malformed"
        assert reply.calls == ()

    def test_a_model_tool_call_needs_an_id(self):
        with pytest.raises(ValidationError):
            ModelToolCall(id="", name="add", arguments={})


_ADD_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
    },
}
