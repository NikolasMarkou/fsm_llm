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

    def test_non_agent_handler_set_config_key_is_overwritable(self):
        """Documents the D-004 trade-off: same setup minus agent_trace."""
        snaps = _CorrectionHarness().run(
            [(None, {"favorite_color": "red"})], presets={"favorite_color": "blue"}
        )
        assert snaps[0][0]["favorite_color"] == "red"


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
