"""Seam regression tests for audit-fix loop 1 (plan-2026-09-19T175721-21cd7f8e).

Every test here drives a public entry point (``API.converse`` or
``LiteLLMInterface`` with a patched ``fsm_llm.llm.completion``), because the
defects these pin were invisible to helper-level tests. Sections are appended
one per plan step.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.api import API
from fsm_llm.definitions import FieldExtractionRequest
from fsm_llm.llm import LiteLLMInterface

OLLAMA_MODEL = "ollama_chat/qwen3.5:9b-q8_0"


def _fake_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


# ══════════════════════════════════════════════════════════════
# Step 2 / LV-01: typed field-extraction `value` + dict rejection
# ══════════════════════════════════════════════════════════════


def _field_request(field_type: str = "str") -> FieldExtractionRequest:
    return FieldExtractionRequest(
        system_prompt="extract the field",
        user_message="My favorite color is blue.",
        field_name="favorite_color",
        field_type=field_type,  # type: ignore[arg-type]
    )


class TestFieldExtractionTypedSchemaOnTheWire:
    """`LiteLLMInterface.extract_field` sends the field-typed schema (Ollama)."""

    @staticmethod
    def _run(model: str, field_type: str):
        interface = LiteLLMInterface(model=model, api_key="test")
        with (
            patch("fsm_llm.llm.completion") as mock_comp,
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ):
            mock_comp.return_value = _fake_response(
                '{"field_name": "favorite_color", "value": "blue", "confidence": 0.9}'
            )
            interface.extract_field(_field_request(field_type))
            return mock_comp.call_args[1]

    @pytest.mark.parametrize("field_type", ["str", "int", "float", "bool", "list"])
    def test_ollama_value_type_forbids_object(self, field_type):
        params = self._run(OLLAMA_MODEL, field_type)
        schema = params["response_format"]["json_schema"]["schema"]
        value_type = schema["properties"]["value"]["type"]
        assert "object" not in value_type

    def test_ollama_dict_field_still_allows_object(self):
        params = self._run(OLLAMA_MODEL, "dict")
        schema = params["response_format"]["json_schema"]["schema"]
        assert "object" in schema["properties"]["value"]["type"]

    def test_ollama_prompt_no_longer_shows_empty_value_schema(self):
        params = self._run(OLLAMA_MODEL, "any")
        prompt = params["messages"][-1]["content"]
        assert '"value": {}' not in prompt
        assert '"value": {"type": [' in prompt

    def test_non_ollama_keeps_json_object_format(self):
        params = self._run("gpt-4o", "str")
        assert params["response_format"] == {"type": "json_object"}


def _color_fsm() -> dict:
    return {
        "name": "ColorBot",
        "description": "typed field seam",
        "version": "4.1",
        "initial_state": "profile",
        "persona": "Concise.",
        "states": {
            "profile": {
                "id": "profile",
                "description": "collect color",
                "purpose": "Learn favorite color",
                "field_extractions": [
                    {
                        "field_name": "favorite_color",
                        "field_type": "str",
                        "extraction_instructions": "Favorite color, one word.",
                        "required": True,
                    }
                ],
                "response_instructions": "Reply in one short sentence.",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "got color",
                        "conditions": [
                            {
                                "description": "has color",
                                "requires_context_keys": ["favorite_color"],
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "end",
                "purpose": "end",
                "response_instructions": "Say goodbye.",
                "transitions": [],
            },
        },
    }


class TestDictValueForStrFieldIsRejected:
    """A dict-wrapped answer for a str field is not stored and does not gate."""

    @staticmethod
    def _completion(field_payload: str):
        def fake(**kwargs):
            fmt = kwargs.get("response_format")
            if fmt is not None:
                return _fake_response(field_payload)
            return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

        return fake

    def _converse(self, field_payload: str):
        with (
            patch(
                "fsm_llm.llm.completion", side_effect=self._completion(field_payload)
            ),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ):
            api = API.from_definition(_color_fsm(), model="gpt-4o", api_key="test")
            cid, _ = api.start_conversation()
            api.converse("My favorite color is blue.", cid)
            return api.get_data(cid), api.get_current_state(cid)

    def test_dict_value_is_not_stored_and_transition_does_not_fire(self):
        data, state = self._converse(
            '{"field_name": "favorite_color", "value": {"blue": "blue"}, '
            '"confidence": 0.9}'
        )
        assert "favorite_color" not in data
        assert state == "profile"

    def test_list_value_is_not_stored_for_str_field(self):
        data, state = self._converse(
            '{"field_name": "favorite_color", "value": ["blue"], "confidence": 0.9}'
        )
        assert "favorite_color" not in data
        assert state == "profile"

    def test_plain_string_value_is_stored_and_transition_fires(self):
        """Vacuity guard: the rejection is a filter, not a dead extractor."""
        data, state = self._converse(
            '{"field_name": "favorite_color", "value": "blue", "confidence": 0.9}'
        )
        assert data.get("favorite_color") == "blue"
        assert state == "done"


class TestZeroConfidenceValueIsNotAccepted:
    """A self-reported confidence of 0.0 is "could not ground it", not a value."""

    @staticmethod
    def _converse(confidence: str):
        payload = (
            '{"field_name": "favorite_color", "value": "blue", '
            f'"confidence": {confidence}}}'
        )

        def fake(**kwargs):
            if kwargs.get("response_format") is not None:
                return _fake_response(payload)
            return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

        with (
            patch("fsm_llm.llm.completion", side_effect=fake),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ):
            api = API.from_definition(_color_fsm(), model="gpt-4o", api_key="test")
            cid, _ = api.start_conversation()
            api.converse("My favorite color is blue.", cid)
            return api.get_data(cid), api.get_current_state(cid)

    def test_zero_confidence_value_is_left_unset(self):
        data, state = self._converse("0.0")
        assert "favorite_color" not in data
        assert state == "profile"

    def test_low_but_nonzero_confidence_is_still_stored(self):
        data, state = self._converse("0.1")
        assert data.get("favorite_color") == "blue"
        assert state == "done"


# ══════════════════════════════════════════════════════════════
# Step 3 / LV-04: plain-text streaming prompt (no JSON envelope)
# ══════════════════════════════════════════════════════════════


def _stream_chunk(text: str) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = text
    chunk.choices[0].delta.reasoning_content = None
    return chunk


def _greeter_fsm(terminal: bool = False) -> dict:
    """chat -> other. ``terminal=True`` gates the edge on ``go`` (the harness
    sets it), so Pass 2 runs in the terminal state ``other``; otherwise the
    edge never fires and Pass 2 runs in the non-terminal state ``chat``."""
    gate = "go" if terminal else "never_set"
    transitions = [
        {
            "target_state": "other",
            "description": "gated",
            "conditions": [{"description": "gated", "requires_context_keys": [gate]}],
        }
    ]
    fsm = {
        "name": "Greeter",
        "description": "stream prompt seam",
        "version": "4.1",
        "initial_state": "chat",
        "persona": "Concise.",
        "states": {
            "chat": {
                "id": "chat",
                "description": "chat",
                "purpose": "Chat briefly",
                "response_instructions": "Answer in one short sentence.",
                "transitions": transitions,
            },
            "other": {
                "id": "other",
                "description": "other",
                "purpose": "other",
                "response_instructions": "Say bye.",
                "transitions": [],
            },
        },
    }
    return fsm


class _StreamHarness:
    """Drive `API.converse` / `converse_stream` against a fake `completion`."""

    def __init__(self, stream_chunks: list[str], terminal: bool = False):
        self.calls: list[dict] = []
        self._chunks = stream_chunks
        self._terminal = terminal

    def _fake(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([_stream_chunk(c) for c in self._chunks])
        return _fake_response(json.dumps({"message": "Hello there.", "reasoning": ""}))

    def run(self, mode: str, context: dict | None = None):
        with (
            patch("fsm_llm.llm.completion", side_effect=self._fake),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ):
            api = API.from_definition(
                _greeter_fsm(self._terminal), model="gpt-4o", api_key="test"
            )
            cid, _ = api.start_conversation()
            ctx = dict(context or {})
            if self._terminal:
                ctx["go"] = True
            if ctx:
                api.update_context(cid, ctx)
            self.calls.clear()  # drop the greeting call
            if mode == "stream":
                self.streamed = list(api.converse_stream("Hi there", cid))
            else:
                self.reply = api.converse("Hi there", cid)
            self.history = api.get_conversation_history(cid)
        return self

    def system_prompt(self, stream: bool) -> str:
        matching = [c for c in self.calls if bool(c.get("stream")) is stream]
        assert matching, f"no {'stream' if stream else 'sync'} call captured"
        return matching[-1]["messages"][0]["content"]


class TestStreamPromptIsPlainText:
    """`converse_stream` neither asks for, yields, nor stores a JSON envelope."""

    def test_stream_system_prompt_has_no_message_schema_block(self):
        h = _StreamHarness(["Hello", " there."]).run("stream")
        prompt = h.system_prompt(stream=True)
        assert '"message":' not in prompt
        assert "valid JSON" not in prompt
        assert "<response_format>" in prompt  # plain-text variant still present

    def test_plain_chunks_are_yielded_verbatim_and_stored_identically(self):
        h = _StreamHarness(["Hello", " there", "."]).run("stream")
        assert h.streamed == ["Hello", " there", "."]
        last = h.history[-1]
        assert list(last.values()) == ["Hello there."]
        assert not any('"message"' in v for e in h.history for v in e.values())

    def test_terminal_state_with_output_response_format_keeps_json_prompt(self):
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "out",
                "schema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
            },
        }
        h = _StreamHarness(['{"message": "hi"}'], terminal=True).run(
            "stream", context={"_output_response_format": schema}
        )
        prompt = h.system_prompt(stream=True)
        assert '"message":' in prompt
        assert "valid JSON" in prompt

    def test_terminal_state_without_output_format_streams_plain_text(self):
        h = _StreamHarness(["Bye."], terminal=True).run("stream")
        prompt = h.system_prompt(stream=True)
        assert '"message":' not in prompt

    def test_sync_converse_prompt_keeps_json_envelope_section(self):
        h = _StreamHarness([]).run("converse")
        prompt = h.system_prompt(stream=False)
        assert '"message": "Your natural response to the user"' in prompt
        assert "Your response must be valid JSON" in prompt

    def test_sync_converse_prompt_is_byte_identical_to_pre_change(self):
        """Hashes recorded on the unfixed tree (step-3 RED run)."""
        import hashlib

        h = _StreamHarness([]).run("converse")
        prompt = h.system_prompt(stream=False)
        section = prompt[
            prompt.index("<response_format>") : prompt.index("</response_format>")
        ]
        assert (
            hashlib.sha256(section.encode()).hexdigest()
            == "f8dbf124c4e2d75d4e7adaa608a532a7fe62ff6a61ad49dc784ae43d40764d67"
        )
        assert (
            hashlib.sha256(prompt.encode()).hexdigest()
            == "9777721720503bc6d0df9aca74d9a177eeaac7e4f2c72c72efd79be1e802c7a8"
        )

    def test_builder_default_matches_explicit_json_variant(self):
        from fsm_llm.prompts import ResponseGenerationPromptBuilder

        b = ResponseGenerationPromptBuilder()
        assert b._build_response_format_section() == b._build_response_format_section(
            plain_text=False
        )
