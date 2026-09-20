"""Seam regression tests for audit-fix loop 1 (plan-2026-09-19T175721-21cd7f8e).

Every test here drives a public entry point (``API.converse`` or
``LiteLLMInterface`` with a patched ``fsm_llm.llm.completion``), because the
defects these pin were invisible to helper-level tests. Sections are appended
one per plan step.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fsm_llm.api import API
from fsm_llm.classification import Classifier
from fsm_llm.definitions import (
    BulkExtractionRequest,
    ClassificationResponseError,
    ClassificationSchema,
    FieldExtractionRequest,
    IntentDefinition,
    LLMResponseError,
)
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.utilities import extract_json_from_text, strip_think_and_fences

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


# ══════════════════════════════════════════════════════════════
# Step 4 / LV-03: bulk-extraction correction overwrite (D-004)
# ══════════════════════════════════════════════════════════════


def _correction_fsm() -> dict:
    """One collecting state: `favorite_color` has a config (required key),
    `nickname` is named only in extraction_instructions. No transitions."""
    return {
        "name": "CorrectionBot",
        "description": "bulk correction seam",
        "version": "4.1",
        "initial_state": "profile",
        "persona": "Concise.",
        "states": {
            "profile": {
                "id": "profile",
                "description": "collect color",
                "purpose": "Learn favorite color",
                "required_context_keys": ["favorite_color"],
                "extraction_instructions": (
                    "Extract favorite_color (one word) and nickname."
                ),
                "response_instructions": "Reply in one short sentence.",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "sentinel, never true",
                        "priority": 10,
                        "conditions": [
                            {
                                "description": "sentinel",
                                "requires_context_keys": ["favorite_color"],
                                "logic": {"==": [{"var": "favorite_color"}, "__x__"]},
                            }
                        ],
                    },
                    {
                        "target_state": "profile",
                        "description": "keep chatting",
                        "priority": 100,
                    },
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


class _CorrectionHarness:
    """`API.converse` over a fake completion with scripted per-turn outputs.

    Per turn: ``field`` is what the per-field extractor returns (``None`` means
    "no value"), ``bulk`` is the bulk `extracted_data` dict.
    """

    def __init__(self, fsm: dict | None = None):
        self.fsm = fsm or _correction_fsm()
        self.field: object = None
        self.bulk: dict = {}
        self.context_updates: list[list[str]] = []

    def _completion(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if '"extracted_data"' in system:
            return _fake_response(
                json.dumps({"extracted_data": self.bulk, "confidence": 0.9})
            )
        if kwargs.get("response_format") is not None:
            return _fake_response(
                json.dumps(
                    {
                        "field_name": "favorite_color",
                        "value": self.field,
                        "confidence": 0.9 if self.field is not None else 0.0,
                    }
                )
            )
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    def run(self, script, presets: dict | None = None):
        """script: list of (field, bulk) per turn. Returns (api, cid) data per turn."""
        from fsm_llm.handlers import HandlerTiming

        snapshots = []
        with (
            patch("fsm_llm.llm.completion", side_effect=self._completion),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        ):
            api = API.from_definition(self.fsm, model="gpt-4o", api_key="test")
            api.register_handler(
                api.create_handler("ctx_spy")
                .at(HandlerTiming.CONTEXT_UPDATE)
                .do(lambda ctx: self.context_updates.append(sorted(ctx.keys())) or {})
            )
            cid, _ = api.start_conversation()
            if presets:
                api.update_context(cid, presets)
            for i, (field, bulk) in enumerate(script):
                self.field, self.bulk = field, bulk
                before = len(self.context_updates)
                api.converse(f"turn {i}", cid)
                snapshots.append(
                    (dict(api.get_data(cid)), len(self.context_updates) - before)
                )
        return snapshots


class TestBulkCorrectionOverwrite:
    """A later-turn correction of a config-covered key reaches `get_data`."""

    def test_config_covered_key_is_corrected_with_one_context_update(self):
        snaps = _CorrectionHarness().run(
            [("blue", {}), (None, {"favorite_color": "red"})]
        )
        assert snaps[0][0]["favorite_color"] == "blue"
        assert snaps[1][0]["favorite_color"] == "red"
        assert snaps[1][1] == 1

    def test_identical_bulk_value_is_a_no_op(self):
        snaps = _CorrectionHarness().run(
            [("blue", {}), (None, {"favorite_color": "blue"})]
        )
        assert snaps[1][0]["favorite_color"] == "blue"
        assert snaps[1][1] == 0

    def test_null_bulk_value_does_not_clear(self):
        snaps = _CorrectionHarness().run(
            [("blue", {}), (None, {"favorite_color": None})]
        )
        assert snaps[1][0]["favorite_color"] == "blue"
        assert snaps[1][1] == 0

    def test_same_turn_per_field_value_wins_over_bulk(self):
        snaps = _CorrectionHarness().run([("blue", {"favorite_color": "red"})])
        assert snaps[0][0]["favorite_color"] == "blue"

    def test_instruction_only_key_is_not_overwritten(self):
        snaps = _CorrectionHarness().run(
            [("blue", {"nickname": "BULK"})], presets={"nickname": "PRESET"}
        )
        assert snaps[0][0]["nickname"] == "PRESET"

    def test_instruction_only_key_is_still_added_when_absent(self):
        """Vacuity guard: the additive pass itself is alive."""
        snaps = _CorrectionHarness().run([("blue", {"nickname": "Bee"})])
        assert snaps[0][0]["nickname"] == "Bee"

    def test_agent_fsm_never_overwrites_a_handler_set_key(self):
        snaps = _CorrectionHarness().run(
            [(None, {"favorite_color": "red"})],
            presets={"favorite_color": "blue", "agent_trace": []},
        )
        assert snaps[0][0]["favorite_color"] == "blue"

    def test_non_agent_handler_set_config_key_is_not_overwritable(self):
        """D-015 supersedes the D-004 trade-off: a value the pipeline did not
        extract itself (handler / update_context) is never overwritten."""
        snaps = _CorrectionHarness().run(
            [(None, {"favorite_color": "red"})], presets={"favorite_color": "blue"}
        )
        assert snaps[0][0]["favorite_color"] == "blue"
        assert snaps[0][1] == 0


# ══════════════════════════════════════════════════════════════
# Step 5 / CF-05: ERROR handlers fire for FSMError and on the stream (D-009)
# ══════════════════════════════════════════════════════════════


class _ErrorHandlerHarness:
    """`API` over a completion that fails on demand; records ERROR handler calls.

    ``fail`` is one of: ``"sync"`` (the non-stream call raises), ``"first"``
    (the stream call raises before any chunk), ``"mid"`` (the stream yields one
    chunk then raises), ``None`` (healthy).
    """

    def __init__(self, fail: str | None, exc: BaseException | None = None):
        self._mode = fail
        self.fail: str | None = None  # armed by ``start`` (greeting must succeed)
        self.exc = exc if exc is not None else RuntimeError("provider down")
        self.error_ctxs: list[dict] = []
        self.handler_raises: BaseException | None = None

    def _completion(self, **kwargs):
        if kwargs.get("stream"):
            if self.fail == "first":
                raise self.exc
            if self.fail == "mid":

                def _gen():
                    yield _stream_chunk("partial")
                    raise self.exc

                return _gen()
            return iter([_stream_chunk("ok")])
        if self.fail == "sync":
            raise self.exc
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    def _on_error(self, ctx):
        self.error_ctxs.append(dict(ctx))
        if self.handler_raises is not None:
            raise self.handler_raises
        return {}

    def start(self, api):
        """Start the conversation on a healthy LLM, then arm the failure."""
        cid, _ = api.start_conversation()
        self.fail = self._mode
        return cid

    def make_api(self):
        from fsm_llm.handlers import HandlerTiming

        api = API.from_definition(_greeter_fsm(False), model="gpt-4o", api_key="test")
        builder = api.create_handler("err_spy").at(HandlerTiming.ERROR)
        if self.handler_raises is not None:
            builder = builder.critical()
        api.register_handler(builder.do(self._on_error))
        return api

    def patches(self):
        return (
            patch("fsm_llm.llm.completion", side_effect=self._completion),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        )


class TestErrorHandlersFireForFSMErrorAndStream:
    """ERROR handlers run for LLMResponseError and on the stream path only."""

    def test_converse_llm_outage_fires_error_handler_once_and_reraises_fsmerror(self):
        from fsm_llm.definitions import FSMError, LLMResponseError

        h = _ErrorHandlerHarness("sync")
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            with pytest.raises(FSMError) as ei:
                api.converse("hi", cid)
        assert isinstance(ei.value, LLMResponseError)
        assert len(h.error_ctxs) == 1
        assert "provider down" in h.error_ctxs[0]["_error"]
        assert "_traceback" in h.error_ctxs[0]

    def test_converse_failure_still_rolls_back_the_user_message(self):
        from fsm_llm.definitions import FSMError

        h = _ErrorHandlerHarness("sync")
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            before = len(api.get_conversation_history(cid))
            with pytest.raises(FSMError):
                api.converse("hi", cid)
            assert len(api.get_conversation_history(cid)) == before

    def test_critical_handler_failure_replaces_fsmerror_with_original_as_cause(self):
        from fsm_llm.definitions import LLMResponseError
        from fsm_llm.handlers import HandlerExecutionError

        h = _ErrorHandlerHarness("sync")
        h.handler_raises = ValueError("handler exploded")
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            with pytest.raises(HandlerExecutionError) as ei:
                api.converse("hi", cid)
        assert "handler exploded" in str(ei.value)
        assert isinstance(ei.value.__cause__, LLMResponseError)

    def test_stream_failure_at_first_token_fires_error_handler(self):
        from fsm_llm.definitions import FSMError

        h = _ErrorHandlerHarness("first")
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            with pytest.raises(FSMError):
                list(api.converse_stream("hi", cid))
        assert len(h.error_ctxs) == 1
        assert "provider down" in h.error_ctxs[0]["_error"]

    def test_stream_failure_mid_stream_fires_error_handler(self):
        h = _ErrorHandlerHarness("mid")
        p1, p2 = h.patches()
        got: list[str] = []
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            try:
                for tok in api.converse_stream("hi", cid):
                    got.append(tok)
            except Exception:
                pass
        assert got[:1] == ["partial"]
        assert len(h.error_ctxs) == 1

    def test_stream_critical_handler_failure_replaces_error_with_cause(self):
        from fsm_llm.handlers import HandlerExecutionError

        h = _ErrorHandlerHarness("first")
        h.handler_raises = ValueError("handler exploded")
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            with pytest.raises(HandlerExecutionError) as ei:
                list(api.converse_stream("hi", cid))
        assert ei.value.__cause__ is not None

    @pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
    def test_interrupts_fire_no_error_handler_sync_and_stream(self, exc_type):
        h = _ErrorHandlerHarness("sync", exc=exc_type())
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            with pytest.raises(exc_type):
                api.converse("hi", cid)
            h.fail = "first"
            with pytest.raises(exc_type):
                list(api.converse_stream("hi", cid))
        assert h.error_ctxs == []

    def test_generator_close_fires_no_error_handler_and_releases_lock(self):
        h = _ErrorHandlerHarness(None)
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            gen = api.converse_stream("hi", cid)
            next(gen)
            gen.close()
            # lock released: a following turn is accepted, not "already processed"
            assert api.converse("again", cid)
        assert h.error_ctxs == []

    def test_healthy_turns_fire_no_error_handler(self):
        h = _ErrorHandlerHarness(None)
        p1, p2 = h.patches()
        with p1, p2:
            api = h.make_api()
            cid = h.start(api)
            api.converse("hi", cid)
            list(api.converse_stream("hi again", cid))
        assert h.error_ctxs == []


# ══════════════════════════════════════════════════════════════
# Step 6 / CF-01: context_scope.read_keys reaches the Pass-2 prompt (D-005)
# ══════════════════════════════════════════════════════════════

HIDDEN = "HIDDEN-VALUE-7431"
SHOWN = "SHOWN-VALUE-2208"
SECRET = "SECRET-TOKEN-9917"


def _scoped_fsm(read_keys: list[str] | None, with_scope: bool = True) -> dict:
    fsm = _greeter_fsm(False)
    if with_scope:
        fsm["states"]["chat"]["context_scope"] = {"read_keys": read_keys}
    return fsm


class _ScopeHarness:
    """Capture the Pass-2 system prompt at the sync, stream and greeting sites."""

    def __init__(self, fsm: dict, context: dict | None = None):
        self.fsm = fsm
        self.context = context if context is not None else {}
        self.calls: list[dict] = []

    def _fake(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([_stream_chunk("ok")])
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    def _patches(self):
        return (
            patch("fsm_llm.llm.completion", side_effect=self._fake),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
        )

    def prompt(self, site: str) -> str:
        p1, p2 = self._patches()
        with p1, p2:
            api = API.from_definition(self.fsm, model="gpt-4o", api_key="test")
            cid, _ = api.start_conversation(initial_context=dict(self.context))
            greeting = self.calls[-1]["messages"][0]["content"]
            if site == "greeting":
                return greeting
            self.calls.clear()
            if site == "stream":
                list(api.converse_stream("Hi there", cid))
                matching = [c for c in self.calls if c.get("stream")]
            else:
                api.converse("Hi there", cid)
                matching = [c for c in self.calls if not c.get("stream")]
            return matching[-1]["messages"][0]["content"]


_SITES = ["sync", "stream", "greeting"]


class TestContextScopeReachesPass2Prompt:
    @pytest.mark.parametrize("site", _SITES)
    def test_key_outside_read_keys_is_absent_from_prompt(self, site):
        h = _ScopeHarness(_scoped_fsm(["shown"]), {"shown": SHOWN, "hidden": HIDDEN})
        prompt = h.prompt(site)
        assert HIDDEN not in prompt
        assert SHOWN in prompt

    @pytest.mark.parametrize("site", _SITES)
    def test_unscoped_state_still_shows_every_key(self, site):
        h = _ScopeHarness(
            _scoped_fsm(None, with_scope=False),
            {"shown": SHOWN, "hidden": HIDDEN},
        )
        prompt = h.prompt(site)
        assert HIDDEN in prompt
        assert SHOWN in prompt

    @pytest.mark.parametrize("site", _SITES)
    def test_scope_without_read_keys_is_unscoped(self, site):
        h = _ScopeHarness(_scoped_fsm([]), {"shown": SHOWN, "hidden": HIDDEN})
        prompt = h.prompt(site)
        assert HIDDEN in prompt

    @pytest.mark.parametrize("site", _SITES)
    def test_secret_named_key_inside_read_keys_is_still_dropped(self, site):
        h = _ScopeHarness(
            _scoped_fsm(["shown", "api_token"]),
            {"shown": SHOWN, "api_token": SECRET, "hidden": HIDDEN},
        )
        prompt = h.prompt(site)
        assert SECRET not in prompt
        assert HIDDEN not in prompt
        assert SHOWN in prompt

    @pytest.mark.parametrize("site", _SITES)
    def test_read_keys_naming_absent_keys_do_not_break_the_prompt(self, site):
        h = _ScopeHarness(_scoped_fsm(["not_yet_set"]), {"hidden": HIDDEN})
        prompt = h.prompt(site)
        assert HIDDEN not in prompt

    def test_builder_context_arg_is_optional_and_defaults_to_none(self):
        import inspect

        from fsm_llm.prompts import ResponseGenerationPromptBuilder

        param = inspect.signature(
            ResponseGenerationPromptBuilder.build_response_prompt
        ).parameters["context"]
        assert param.default is None

    def test_unscoped_prompt_is_byte_identical_to_pre_change(self):
        """Hash recorded on the unfixed tree (step-6 RED run), unscoped state."""
        import hashlib

        h = _ScopeHarness(
            _scoped_fsm(None, with_scope=False),
            {"shown": SHOWN, "hidden": HIDDEN},
        )
        digests = {
            site: hashlib.sha256(h.prompt(site).encode()).hexdigest() for site in _SITES
        }
        assert digests == {
            "sync": "cc5623b3329c5a3b0fddcc4ce98b2f3d3c2d81e8b940608bb23f7f52c8d4daef",
            "stream": "78aecf1510e608bda9d92841f341592b163724ffbaabf689d85e641ddb84721c",
            "greeting": "ee5972fd140e18804a4f6faf7221cc08e51be3aafdf4a0ac42caec98430c410e",
        }


# ══════════════════════════════════════════════════════════════
# Step 7 / CF-02: classification-owned key is not plain-extracted
# ══════════════════════════════════════════════════════════════


def _classified_fsm(explicit_field_extraction: bool = False, topic: bool = False):
    """`intent` is owned by classification_extractions and gates the only
    transition. ``topic`` adds an ordinary required key (vacuity guard)."""
    state: dict = {
        "id": "triage",
        "description": "classify intent",
        "purpose": "route",
        "response_instructions": "Reply briefly.",
        "classification_extractions": [
            {
                "field_name": "intent",
                "intents": [
                    {"name": "buy", "description": "wants to purchase"},
                    {"name": "browse", "description": "just looking"},
                ],
                "fallback_intent": "browse",
                "confidence_threshold": 0.7,
                "required": False,
            }
        ],
        "transitions": [
            {
                "target_state": "shop",
                "description": "buying",
                "conditions": [
                    {
                        "description": "intent is buy",
                        "requires_context_keys": ["intent"],
                        "logic": {"==": [{"var": "intent"}, "buy"]},
                    }
                ],
            }
        ],
    }
    if topic:
        state["required_context_keys"] = ["topic"]
    if explicit_field_extraction:
        state["field_extractions"] = [
            {
                "field_name": "intent",
                "field_type": "str",
                "extraction_instructions": "Explicit intent extraction.",
                "required": False,
            }
        ]
    return {
        "name": "ClassifiedBot",
        "description": "classification ownership seam",
        "version": "4.1",
        "initial_state": "triage",
        "persona": "Concise.",
        "states": {
            "triage": state,
            "shop": {
                "id": "shop",
                "description": "shop",
                "purpose": "shop",
                "response_instructions": "Help shop.",
                "transitions": [],
            },
        },
    }


class _ClassifiedHarness:
    """`API.converse` with a scripted classifier and a spying fake completion.

    ``plain_calls`` records the system prompt of every per-field extraction
    call (the ones that carry a ``response_format`` and name a field).
    """

    def __init__(self, fsm: dict, intent: str, confidence: float, plain_value="buy"):
        self.fsm = fsm
        self.intent = intent
        self.confidence = confidence
        self.plain_value = plain_value
        self.plain_calls: list[str] = []

    def _completion(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if kwargs.get("response_format") is not None:
            self.plain_calls.append(system)
            return _fake_response(
                json.dumps(
                    {"field_name": "x", "value": self.plain_value, "confidence": 0.95}
                )
            )
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    def run(self):
        from fsm_llm.definitions import ClassificationResult

        classifier = MagicMock()
        classifier.classify.return_value = ClassificationResult(
            reasoning="r", intent=self.intent, confidence=self.confidence
        )
        with (
            patch("fsm_llm.llm.completion", side_effect=self._completion),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
            patch("fsm_llm.pipeline.Classifier", return_value=classifier),
        ):
            api = API.from_definition(self.fsm, model="gpt-4o", api_key="test")
            cid, _ = api.start_conversation()
            api.converse("I would like to buy a phone", cid)
            return api.get_data(cid), api.get_current_state(cid)


class TestClassificationOwnedKeyIsNotPlainExtracted:
    def test_no_plain_extract_call_for_classification_key(self):
        h = _ClassifiedHarness(_classified_fsm(), "buy", 0.95)
        data, state = h.run()
        assert h.plain_calls == []
        assert data.get("intent") == "buy"
        assert state == "shop"

    def test_below_threshold_classification_leaves_key_unset_and_gate_closed(self):
        h = _ClassifiedHarness(_classified_fsm(), "buy", 0.2)
        data, state = h.run()
        assert h.plain_calls == []
        assert "intent" not in data
        assert state == "triage"

    def test_fallback_intent_is_still_stored(self):
        h = _ClassifiedHarness(_classified_fsm(), "browse", 0.1)
        data, state = h.run()
        assert data.get("intent") == "browse"
        assert state == "triage"

    def test_ordinary_required_key_is_still_plain_extracted(self):
        """Vacuity guard: the exclusion is per-key, not a disabled extractor."""
        h = _ClassifiedHarness(_classified_fsm(topic=True), "buy", 0.95, "phones")
        data, _ = h.run()
        assert len(h.plain_calls) == 1
        assert "'topic'" in h.plain_calls[0]
        assert "'intent'" not in h.plain_calls[0]
        assert data.get("topic") == "phones"

    def test_explicit_field_extraction_with_same_name_still_wins(self):
        h = _ClassifiedHarness(
            _classified_fsm(explicit_field_extraction=True), "buy", 0.2
        )
        data, state = h.run()
        assert len(h.plain_calls) == 1
        assert data.get("intent") == "buy"
        assert state == "shop"


# ══════════════════════════════════════════════════════════════
# Step 8 / CF-04: "stay" (classifier error / fallback intent) is not a transition
# ══════════════════════════════════════════════════════════════


def _ambiguous_fsm(self_loop: bool = False) -> dict:
    """`hub` has two unconditional, equal-priority exits (AMBIGUOUS), or a single
    declared unconditional self-loop (DETERMINISTIC, target == current state)."""

    def _target(name: str, description: str) -> dict:
        return {"target_state": name, "description": description, "priority": 100}

    if self_loop:
        gated = _target("billing", "billing, gated on a key that is never set")
        gated["conditions"] = [
            {
                "description": "go set",
                "requires_context_keys": ["go"],
                "logic": {"==": [{"var": "go"}, True]},
            }
        ]
        transitions = [_target("hub", "declared self-loop"), gated]
    else:
        transitions = [
            _target("billing", "user asks about billing"),
            _target("support", "user needs tech support"),
        ]
    states = {
        "hub": {
            "id": "hub",
            "description": "hub",
            "purpose": "route",
            "response_instructions": "Reply briefly.",
            "transitions": transitions,
        },
    }
    for name in ("billing",) if self_loop else ("billing", "support"):
        states[name] = {
            "id": name,
            "description": name,
            "purpose": name,
            "response_instructions": f"Handle {name}.",
            "transitions": [],
        }
    return {
        "name": "HubBot",
        "description": "ambiguous transition seam",
        "version": "4.1",
        "initial_state": "hub",
        "persona": "Concise.",
        "states": states,
    }


class _StayHarness:
    """`API.converse` with a scripted (or raising) classifier, transition
    handlers counting their calls, and the Pass-2 system prompt captured."""

    def __init__(self, fsm: dict, intent: str | None, confidence: float = 0.95):
        self.fsm = fsm
        self.intent = intent  # None -> the classifier raises
        self.confidence = confidence
        self.pre = 0
        self.post = 0
        self.response_prompts: list[str] = []

    def _completion(self, **kwargs):
        if kwargs.get("response_format") is None:
            self.response_prompts.append(kwargs["messages"][0]["content"])
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    def run(self):
        from fsm_llm.definitions import ClassificationResult
        from fsm_llm.handlers import HandlerTiming

        classifier = MagicMock()
        if self.intent is None:
            classifier.classify.side_effect = RuntimeError("classifier down")
        else:
            classifier.classify.return_value = ClassificationResult(
                reasoning="r", intent=self.intent, confidence=self.confidence
            )

        def _pre(ctx):
            self.pre += 1
            return {}

        def _post(ctx):
            self.post += 1
            return {}

        with (
            patch("fsm_llm.llm.completion", side_effect=self._completion),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
            patch("fsm_llm.pipeline.Classifier", return_value=classifier),
        ):
            api = API.from_definition(self.fsm, model="gpt-4o", api_key="test")
            api.create_handler("pre", HandlerTiming.PRE_TRANSITION, _pre)
            api.create_handler("post", HandlerTiming.POST_TRANSITION, _post)
            cid, _ = api.start_conversation()
            self.response_prompts.clear()
            api.converse("I have a question", cid)
            return api.get_current_state(cid)


class TestStayIsNotATransition:
    def test_classifier_error_is_not_a_transition(self):
        h = _StayHarness(_ambiguous_fsm(), intent=None)
        assert h.run() == "hub"
        assert (h.pre, h.post) == (0, 0)
        assert "<transition_info>" not in h.response_prompts[-1]

    def test_fallback_intent_is_not_a_transition(self):
        from fsm_llm.constants import TRANSITION_CLASSIFICATION_FALLBACK_INTENT

        h = _StayHarness(
            _ambiguous_fsm(), intent=TRANSITION_CLASSIFICATION_FALLBACK_INTENT
        )
        assert h.run() == "hub"
        assert (h.pre, h.post) == (0, 0)
        assert "<transition_info>" not in h.response_prompts[-1]

    def test_classifier_selecting_a_real_target_still_transitions(self):
        """Vacuity guard: the ambiguous path is reached and still transitions."""
        h = _StayHarness(_ambiguous_fsm(), intent="billing")
        assert h.run() == "billing"
        assert (h.pre, h.post) == (1, 1)
        assert "<transition_info>" in h.response_prompts[-1]

    def test_declared_explicit_self_loop_still_fires_handlers(self):
        """Pins D-007: a DECLARED self-loop is design, not a "stay"."""
        h = _StayHarness(_ambiguous_fsm(self_loop=True), intent="billing")
        assert h.run() == "hub"
        assert (h.pre, h.post) == (1, 1)
        assert "<transition_info>" in h.response_prompts[-1]


# ══════════════════════════════════════════════════════════════
# Step 9 / CF-03: the classifier inherits endpoint, key and timeout
# ══════════════════════════════════════════════════════════════


class _ConnectionHarness:
    """`API.converse` with the REAL Classifier and a spying
    ``fsm_llm.classification.completion``; records every classifier call's
    kwargs. The Pass-1/Pass-2 LLM (``fsm_llm.llm.completion``) is faked."""

    def __init__(self, fsm: dict, intent: str, **api_kwargs):
        self.fsm = fsm
        self.intent = intent
        self.api_kwargs = api_kwargs
        self.classifier_calls: list[dict] = []

    def _classifier_completion(self, **kwargs):
        self.classifier_calls.append(kwargs)
        return _fake_response(
            json.dumps({"reasoning": "r", "intent": self.intent, "confidence": 0.95})
        )

    def _llm_completion(self, **kwargs):
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    def run(self):
        with (
            patch("fsm_llm.llm.completion", side_effect=self._llm_completion),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            ),
            patch(
                "fsm_llm.classification.completion",
                side_effect=self._classifier_completion,
            ),
            patch(
                "fsm_llm.classification.get_supported_openai_params",
                return_value=[],
            ),
        ):
            api = API.from_definition(self.fsm, **self.api_kwargs)
            cid, _ = api.start_conversation()
            api.converse("I would like to buy a phone", cid)
            return api.get_data(cid), api.get_current_state(cid)


_PROXY = {
    "model": "gpt-4o",
    "api_key": "sk-proxy-secret",
    "api_base": "http://proxy:4000",
    "timeout": 7,
}


def _classified_fsm_with_model(model: str | None) -> dict:
    fsm = _classified_fsm()
    if model is not None:
        fsm["states"]["triage"]["classification_extractions"][0]["model"] = model
    return fsm


class TestClassifierInheritsConnection:
    def test_classification_extraction_gets_key_base_and_timeout(self):
        h = _ConnectionHarness(_classified_fsm(), "buy", **_PROXY)
        _, state = h.run()
        assert state == "shop"  # vacuity guard: the real classifier ran
        assert len(h.classifier_calls) == 1
        call = h.classifier_calls[0]
        assert call["api_key"] == "sk-proxy-secret"
        assert call["api_base"] == "http://proxy:4000"
        assert call["timeout"] == 7
        assert call["model"] == "gpt-4o"

    def test_ambiguous_transition_resolution_gets_key_base_and_timeout(self):
        h = _ConnectionHarness(_ambiguous_fsm(), "billing", **_PROXY)
        _, state = h.run()
        assert state == "billing"  # vacuity guard: classifier chose the target
        assert len(h.classifier_calls) == 1
        call = h.classifier_calls[0]
        assert call["api_key"] == "sk-proxy-secret"
        assert call["api_base"] == "http://proxy:4000"
        assert call["timeout"] == 7

    def test_default_timeout_still_bounded_without_explicit_timeout(self):
        h = _ConnectionHarness(_classified_fsm(), "buy", model="gpt-4o", api_key="k")
        h.run()
        # LiteLLMInterface's own default (120.0) is inherited, never unbounded
        assert h.classifier_calls[0]["timeout"] == 120.0

    def test_same_model_config_override_keeps_connection(self):
        h = _ConnectionHarness(_classified_fsm_with_model("gpt-4o"), "buy", **_PROXY)
        h.run()
        assert h.classifier_calls[0]["api_key"] == "sk-proxy-secret"

    def test_other_provider_config_model_does_not_receive_the_key(self):
        h = _ConnectionHarness(
            _classified_fsm_with_model("anthropic/claude-3-haiku"), "buy", **_PROXY
        )
        h.run()
        call = h.classifier_calls[0]
        assert call["model"] == "anthropic/claude-3-haiku"
        assert "api_key" not in call
        assert "api_base" not in call

    def test_helper_tolerates_interfaces_without_attributes(self):
        from fsm_llm.llm import LLMInterface
        from fsm_llm.pipeline import MessagePipeline

        def _helper(llm, model=None):
            pipe = MessagePipeline.__new__(MessagePipeline)
            pipe.llm_interface = llm
            return pipe._classifier_connection_kwargs(model)

        class _Bare:
            pass

        assert _helper(_Bare()) == {}
        assert _helper(MagicMock(spec=LLMInterface)) == {}
        # a MagicMock (non-dict kwargs, non-number timeout) contributes nothing
        assert _helper(MagicMock()) == {}
        # bool is not a timeout
        bare = _Bare()
        bare.timeout = True  # type: ignore[attr-defined]
        assert _helper(bare) == {}


# ══════════════════════════════════════════════════════════════
# Step 10 / DH-01 + DH-19: linear-time fence and think handling
# ══════════════════════════════════════════════════════════════

# The corpora below were captured from the ORIGINAL regex implementations, so
# they pin behavior equality across the rewrite, EXCEPT the six `_STRIP_CORPUS`
# rows annotated `D-030`: those pin the D-030 semantics (only a fence at the
# START of the reply is stripped; a fence inside prose is kept). The `[1, 2]`
# fenced-list row is deliberately absent: step 11 changes that contract (dict-only) on purpose.
_STRIP_CORPUS: list[tuple[str, str]] = [
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('```\n{"a": 1}\n```', '{"a": 1}'),
    ('```json{"a": 1}```', '{"a": 1}'),
    ('  ```json  \n\n {"a": 1}  \n```  \n', '{"a": 1}'),
    ('{"a": 1}\n```', '{"a": 1}'),
    ('```json\n{"a": 1}', '{"a": 1}'),
    ('<think>reason {"x":2}</think>{"a": 1}', '{"a": 1}'),
    ('<think>a</think>b<think>c</think>{"a": 1}', 'b{"a": 1}'),
    ('<think>a<think>b</think>c</think>{"a": 1}', 'c</think>{"a": 1}'),
    ('<think>unterminated {"a": 1}', '<think>unterminated {"a": 1}'),
    ('{"a": 1}<think>unterminated', '{"a": 1}<think>unterminated'),
    ('<think>a</think>\n```json\n{"a": 1}\n```', '{"a": 1}'),
    # D-030: was '{"a": 1}'
    ('```\n```json\n{"a": 1}\n```\n```', '```json\n{"a": 1}\n```'),
    # D-030: was 'text\n{"a": 1}\nmore'
    ('text\n```json\n{"a": 1}\n```\nmore', 'text\n```json\n{"a": 1}\n```\nmore'),
    ('```json\r\n{"a": 1}\r\n```\r\n', '{"a": 1}'),
    ("````\nx\n````", "`\nx\n`"),
    ("```jsonx\ny```", "x\ny"),
    # D-030: was '{}'
    ("```json\n\n\n```json\n{}\n```", "```json\n{}"),
    ("<think></think>```json```", ""),
    ("", ""),
    ("```", ""),
    ("\n```\n", ""),
    # D-030: was 'a\nb'
    ("a\n```\n\n```\nb", "a\n```\n\n```\nb"),
    # D-030: was '{"z":1}'
    ('```json\n```\n```json\n{"z":1}\n```', '```\n```json\n{"z":1}'),
    ('<think>a</think></think>{"a": 1}', '</think>{"a": 1}'),
    ('<THINK>a</THINK>{"a":1}', '<THINK>a</THINK>{"a":1}'),
    ("```json ```", ""),
    # D-030: was 'x```json\n{"a":1}\ny'
    ('x```json\n{"a":1}\n```y', 'x```json\n{"a":1}\n```y'),
    ('  \n```json\n{"a":1}\n```\n\n\n', '{"a":1}'),
    ('``` \xa0 \n{"a":1}\u2003```', '{"a":1}'),
]

_EXTRACT_CORPUS: list[tuple[str, dict[str, Any] | None]] = [
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Here:\n```json\n{"a": 1}\n```\nThanks', {"a": 1}),
    ('```{"a": 1}```', {"a": 1}),
    ('```json{"a":1}```', {"a": 1}),
    ('```jsonx{"a":1}```', {"a": 1}),
    ('```json\n{"a": 1}', {"a": 1}),
    ('```\nnot json\n``` then {"b": 2}', {"b": 2}),
    ('```json\n{"a": 1}\n``` and ```json\n{"b": 2}\n```', {"a": 1}),
    ('```json\n{bad}\n``` {"c": 3}', {"c": 3}),
    ('````\n{"a":1}\n````', {"a": 1}),
    ('```\n```{"a":1}', {"a": 1}),
    ('no fence {"a": 1} here', {"a": 1}),
    ('```json   \n  \n {"a": 1} \n  \n```', {"a": 1}),
    ('``` {"k": "v"} ```', {"k": "v"}),
]


_LINEAR_N = 50_000
_LINEAR_BUDGET_S = 1.0


def _elapsed(fn, arg: str) -> float:
    start = time.perf_counter()
    fn(arg)
    return time.perf_counter() - start


class TestLinearFenceAndThink:
    @pytest.mark.parametrize(("text", "expected"), _STRIP_CORPUS)
    def test_strip_think_and_fences_matches_old_outputs(self, text, expected):
        """The corpus pins the D-030 semantics: only a leading fence is stripped."""
        assert strip_think_and_fences(text) == expected

    @pytest.mark.parametrize(("text", "expected"), _EXTRACT_CORPUS)
    def test_extract_json_from_text_matches_old_outputs(self, text, expected):
        assert extract_json_from_text(text) == expected

    def test_extract_json_unclosed_fence_then_spaces_is_linear(self):
        text = "```json" + " " * _LINEAR_N
        start = time.perf_counter()
        assert extract_json_from_text(text) is None
        assert time.perf_counter() - start < _LINEAR_BUDGET_S

    def test_extract_json_unclosed_fence_then_newlines_is_linear(self):
        text = "```" + "\n" * _LINEAR_N
        start = time.perf_counter()
        assert extract_json_from_text(text) is None
        assert time.perf_counter() - start < _LINEAR_BUDGET_S

    def test_strip_think_repeated_unterminated_open_tags_is_linear(self):
        text = "<think>a" * _LINEAR_N
        start = time.perf_counter()
        assert strip_think_and_fences(text) == text
        assert time.perf_counter() - start < _LINEAR_BUDGET_S

    def test_strip_think_many_closed_blocks_is_linear(self):
        text = "<think>a</think>b" * _LINEAR_N
        start = time.perf_counter()
        assert strip_think_and_fences(text) == "b" * _LINEAR_N
        assert time.perf_counter() - start < _LINEAR_BUDGET_S

    @pytest.mark.parametrize(
        "text",
        [
            "x```" + " " * _LINEAR_N + "x",
            "```" + "\n" * _LINEAR_N,
            "``` " * _LINEAR_N,
            "\n```x" * _LINEAR_N,
        ],
        ids=["ws-run", "fence-newlines", "fence-space-repeat", "newline-fence-repeat"],
    )
    def test_strip_fence_shapes_are_linear(self, text):
        assert _elapsed(strip_think_and_fences, text) < _LINEAR_BUDGET_S

    def test_extract_bulk_data_hostile_think_reply_is_bounded(self):
        """The shared stripper is reached from extract_bulk_data on the real seam."""
        llm = LiteLLMInterface(model=OLLAMA_MODEL)
        hostile = "<think>a" * _LINEAR_N
        request = BulkExtractionRequest(system_prompt="extract", user_message="hi")
        with patch("fsm_llm.llm.completion", return_value=_fake_response(hostile)):
            start = time.perf_counter()
            with pytest.raises(LLMResponseError):
                llm.extract_bulk_data(request)
            assert time.perf_counter() - start < _LINEAR_BUDGET_S


# ══════════════════════════════════════════════════════════════
# Step 11 / DH-08 + LS-05: dict-only JSON contract
# ══════════════════════════════════════════════════════════════


def _classifier() -> Classifier:
    schema = ClassificationSchema(
        intents=[
            IntentDefinition(name="buy", description="wants to buy"),
            IntentDefinition(name="browse", description="just looking"),
        ],
        fallback_intent="browse",
    )
    return Classifier(schema, model="gpt-4o")


class TestDictOnlyJsonContract:
    @pytest.mark.parametrize(
        "text",
        ["42", "[1,2]", "true", '"hi"', "```json\n[1,2]\n```", "3.5", "[]"],
        ids=["int", "list", "bool", "string", "fenced-list", "float", "empty-list"],
    )
    def test_non_dict_json_returns_none(self, text):
        assert extract_json_from_text(text) is None

    def test_top_level_array_of_objects_is_not_recovered(self):
        """D-010: no fall-through into the interior of a valid non-object."""
        assert extract_json_from_text('[{"a": 1}]') is None

    def test_first_object_still_wins(self):
        assert extract_json_from_text('{"a":1} {"b":2}') == {"a": 1}

    def test_fenced_non_dict_falls_through_to_brace_scan(self):
        text = '```json\n[1,2]\n``` then {"a": 1}'
        assert extract_json_from_text(text) == {"a": 1}

    @pytest.mark.parametrize("content", ["42", "[1,2]", "true"])
    def test_classify_non_dict_json_raises_classification_error(self, content):
        clf = _classifier()
        with (
            patch(
                "fsm_llm.classification.completion",
                return_value=_fake_response(content),
            ),
            patch(
                "fsm_llm.classification.get_supported_openai_params",
                return_value=[],
            ),
            pytest.raises(ClassificationResponseError),
        ):
            clf.classify("I want a phone")

    @pytest.mark.parametrize("content", ["[1,2]", "null", "42", "true", '"hi"'])
    def test_extract_bulk_data_non_dict_json_returns_empty(self, content):
        llm = LiteLLMInterface(model=OLLAMA_MODEL)
        request = BulkExtractionRequest(system_prompt="extract", user_message="hi")
        with patch("fsm_llm.llm.completion", return_value=_fake_response(content)):
            result = llm.extract_bulk_data(request)
        assert result.extracted_data == {}

    def test_extract_bulk_data_recovers_embedded_object(self):
        llm = LiteLLMInterface(model=OLLAMA_MODEL)
        request = BulkExtractionRequest(system_prompt="extract", user_message="hi")
        with patch(
            "fsm_llm.llm.completion",
            return_value=_fake_response('Sure! {"a": 1}'),
        ):
            result = llm.extract_bulk_data(request)
        assert result.extracted_data == {"a": 1}

    def test_extract_bulk_data_recovers_wrapped_extracted_data(self):
        llm = LiteLLMInterface(model=OLLAMA_MODEL)
        request = BulkExtractionRequest(system_prompt="extract", user_message="hi")
        reply = 'Here you go: {"extracted_data": {"name": "Ann"}, "confidence": 0.5}'
        with patch("fsm_llm.llm.completion", return_value=_fake_response(reply)):
            result = llm.extract_bulk_data(request)
        assert result.extracted_data == {"name": "Ann"}
        assert result.confidence == 0.5

    @pytest.mark.parametrize("content", ["no json at all", ""])
    def test_extract_bulk_data_unparseable_still_raises(self, content):
        llm = LiteLLMInterface(model=OLLAMA_MODEL)
        request = BulkExtractionRequest(system_prompt="extract", user_message="hi")
        with (
            patch("fsm_llm.llm.completion", return_value=_fake_response(content)),
            pytest.raises(LLMResponseError),
        ):
            llm.extract_bulk_data(request)


# ---------------------------------------------------------------------------
# Step 12: DH-02 + DH-03 -- a dumped FSMDefinition validates and renders
# ---------------------------------------------------------------------------


def _dumped_fsm_with_required_keys() -> dict:
    """model_dump() writes explicit None for absent Optional containers."""
    from fsm_llm.definitions import FSMDefinition

    definition = FSMDefinition(
        **{
            "name": "Dumped",
            "description": "dump round trip",
            "initial_state": "start",
            "persona": "helper",
            "states": {
                "start": {
                    "id": "start",
                    "description": "collect",
                    "purpose": "collect name",
                    "required_context_keys": ["name"],
                    "transitions": [
                        {
                            "target_state": "done",
                            "description": "have name",
                            "conditions": [
                                {
                                    "description": "name set",
                                    "requires_context_keys": ["name"],
                                    "logic": {"has_context": "name"},
                                }
                            ],
                        },
                        {"target_state": "done", "description": "bare"},
                    ],
                },
                "done": {
                    "id": "done",
                    "description": "end",
                    "purpose": "finish",
                    "transitions": [],
                },
            },
        }
    )
    return definition.model_dump()


class TestNullSafeValidatorAndVisualizer:
    def test_dump_carries_explicit_nulls(self):
        dumped = _dumped_fsm_with_required_keys()
        start = dumped["states"]["start"]
        bare = start["transitions"][1]
        # Guards the premise: the dump has None (not absent) somewhere.
        nulls = [
            start.get("required_context_keys") is None,
            dumped["states"]["done"].get("required_context_keys") is None,
            bare.get("conditions") is None,
        ]
        assert any(nulls)

    def test_dumped_fsm_validates_from_file(self, tmp_path):
        from fsm_llm.validator import validate_fsm_from_file

        path = tmp_path / "dumped.json"
        path.write_text(json.dumps(_dumped_fsm_with_required_keys()))
        result = validate_fsm_from_file(str(path))
        assert result.is_valid, result.errors

    def test_explicit_null_containers_validate(self, tmp_path):
        from fsm_llm.validator import validate_fsm_from_file

        data = _dumped_fsm_with_required_keys()
        for state in data["states"].values():
            state["required_context_keys"] = None
            for t in state["transitions"]:
                t["conditions"] = None
        path = tmp_path / "nulls.json"
        path.write_text(json.dumps(data))
        result = validate_fsm_from_file(str(path))
        assert result.is_valid, result.errors

    def test_null_requires_context_keys_in_condition_validates(self, tmp_path):
        from fsm_llm.validator import validate_fsm_from_file

        data = _dumped_fsm_with_required_keys()
        data["states"]["start"]["transitions"][0]["conditions"][0][
            "requires_context_keys"
        ] = None
        path = tmp_path / "condnull.json"
        path.write_text(json.dumps(data))
        result = validate_fsm_from_file(str(path))
        assert result.is_valid, result.errors

    @pytest.mark.parametrize("style", ["full", "compact", "minimal"])
    def test_dumped_fsm_renders_in_every_style(self, tmp_path, style):
        from fsm_llm.visualizer import visualize_fsm_from_file

        path = tmp_path / "dumped.json"
        path.write_text(json.dumps(_dumped_fsm_with_required_keys()))
        out = visualize_fsm_from_file(str(path), style)
        assert "Could not generate diagram" not in out
        assert "start" in out

    @pytest.mark.parametrize("style", ["full", "compact", "minimal"])
    def test_explicit_null_containers_render(self, tmp_path, style):
        from fsm_llm.visualizer import visualize_fsm_from_file

        data = _dumped_fsm_with_required_keys()
        for state in data["states"].values():
            state["required_context_keys"] = None
            for t in state["transitions"]:
                t["conditions"] = None
        data["states"]["start"]["transitions"][0]["conditions"] = [
            {"description": "c", "requires_context_keys": None, "logic": {}}
        ]
        path = tmp_path / "nulls.json"
        path.write_text(json.dumps(data))
        out = visualize_fsm_from_file(str(path), style)
        assert "Could not generate diagram" not in out
