"""Seam regression tests for audit-fix loop 2 (plan-2026-09-19T175721-21cd7f8e).

Every test drives a public entry point (``API.converse`` or ``LiteLLMInterface``
with a patched ``fsm_llm.llm.completion``). Sections are appended one per plan
step; harness helpers are reused from the iteration-1 seam file.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from fsm_llm import API, FileSessionStore
from fsm_llm.definitions import FieldExtractionRequest
from fsm_llm.handlers import HandlerTiming
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.ollama import is_ollama_model
from tests.test_fsm_llm.test_audit_iter1_seam import (
    _ambiguous_fsm,
    _classified_fsm,
    _ConnectionHarness,
    _correction_fsm,
    _fake_response,
)

# ══════════════════════════════════════════════════════════════
# Step 2 / LS-18: Ollama detection is an exact provider prefix
# ══════════════════════════════════════════════════════════════


class TestIsOllamaModelPrefix:
    @pytest.mark.parametrize(
        "model",
        [
            "ollama_chat/qwen3.5:9b-q8_0",
            "ollama/llama3:8b",
            "OLLAMA_CHAT/mixtral:latest",
            "Ollama/Llama3",
        ],
    )
    def test_ollama_prefixes_detected(self, model):
        assert is_ollama_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "openai/my-ollama-proxy",
            "azure/gpt-4o-not-ollama",
            "together_ai/ollama-tuned",
            "my-ollama-finetune",
            "gpt-4o",
            "",
        ],
    )
    def test_name_containing_ollama_is_not_ollama(self, model):
        assert is_ollama_model(model) is False

    def test_extract_field_on_lookalike_model_sends_generic_json_object(self):
        req = FieldExtractionRequest(
            system_prompt="extract the field",
            user_message="My favorite color is blue.",
            field_name="favorite_color",
            field_type="str",  # type: ignore[arg-type]
        )
        formats: dict[str, dict] = {}
        for model in ("azure/gpt-4o-not-ollama", "ollama_chat/qwen3.5:9b-q8_0"):
            with (
                patch("fsm_llm.llm.completion") as mock_comp,
                patch(
                    "fsm_llm.llm.get_supported_openai_params",
                    return_value=["response_format"],
                ),
            ):
                mock_comp.return_value = _fake_response(
                    '{"field_name": "favorite_color", "value": "blue",'
                    ' "confidence": 0.9}'
                )
                LiteLLMInterface(model=model, api_key="k").extract_field(req)
                formats[model] = mock_comp.call_args.kwargs["response_format"]
        assert formats["azure/gpt-4o-not-ollama"]["type"] == "json_object"
        # vacuity guard: a real Ollama model still gets the typed grammar
        assert formats["ollama_chat/qwen3.5:9b-q8_0"]["type"] == "json_schema"


# ══════════════════════════════════════════════════════════════
# Step 2 / RA-05: reserved names in LiteLLMInterface kwargs
# ══════════════════════════════════════════════════════════════


class TestClassifierIgnoresReservedInterfaceKwargs:
    @pytest.mark.parametrize("reserved", ["config", "schema"])
    def test_classification_extraction_survives_reserved_kwarg(self, reserved):
        h = _ConnectionHarness(
            _classified_fsm(),
            "buy",
            model="gpt-4o",
            api_key="sk-x",
            timeout=9,
            **{reserved: {"user_thing": 1}},
        )
        _, state = h.run()
        # RED on HEAD: TypeError swallowed, classification silently skipped
        assert state == "shop"
        assert len(h.classifier_calls) == 1
        call = h.classifier_calls[0]
        assert call["api_key"] == "sk-x"
        assert call["timeout"] == 9
        assert reserved not in call

    @pytest.mark.parametrize("reserved", ["config", "schema"])
    def test_ambiguous_transition_survives_reserved_kwarg(self, reserved):
        h = _ConnectionHarness(
            _ambiguous_fsm(),
            "billing",
            model="gpt-4o",
            api_key="sk-x",
            timeout=9,
            **{reserved: {"user_thing": 1}},
        )
        _, state = h.run()
        assert state == "billing"
        assert len(h.classifier_calls) == 1
        call = h.classifier_calls[0]
        assert call["api_key"] == "sk-x"
        assert call["timeout"] == 9
        assert reserved not in call

    def test_helper_drops_reserved_names_only(self):
        from unittest.mock import MagicMock

        from fsm_llm.pipeline import MessagePipeline

        llm = MagicMock()
        llm.model = "gpt-4o"
        llm.timeout = 5
        llm.kwargs = {
            "schema": 1,
            "model": 2,
            "config": 3,
            "api_key": "k",
            "api_base": "b",
        }
        pipe = MessagePipeline.__new__(MessagePipeline)
        pipe.llm_interface = llm
        assert pipe._classifier_connection_kwargs() == {
            "api_key": "k",
            "api_base": "b",
            "timeout": 5,
        }


# ══════════════════════════════════════════════════════════════
# Step 3 / D-015: provenance-only overwrite + bulk-value coercion
# ══════════════════════════════════════════════════════════════


class _Prov:
    """Scripted ``API`` session: per-field replies by field name, bulk reply dict.

    ``field`` maps a field name to the value its per-field extractor returns
    (absent name means "no value"); ``bulk`` is the bulk ``extracted_data``.
    """

    def __init__(self, fsm: dict, store=None):
        self.fsm = fsm
        self.store = store
        self.field: dict = {}
        self.bulk: dict = {}
        self.context_updates: list[list[str]] = []
        self._stack = ExitStack()

    def _completion(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if '"extracted_data"' in system:
            return _fake_response(
                json.dumps({"extracted_data": self.bulk, "confidence": 0.9})
            )
        if kwargs.get("response_format") is not None:
            for name, value in self.field.items():
                if f"Extract the field '{name}'" in system:
                    return _fake_response(
                        json.dumps(
                            {"field_name": name, "value": value, "confidence": 0.9}
                        )
                    )
            return _fake_response(
                json.dumps({"field_name": "x", "value": None, "confidence": 0.0})
            )
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

    def __enter__(self):
        self._stack.enter_context(
            patch("fsm_llm.llm.completion", side_effect=self._completion)
        )
        self._stack.enter_context(
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            )
        )
        self.api = self.make_api()
        self.cid, _ = self.api.start_conversation()
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def make_api(self) -> API:
        api = API.from_definition(
            self.fsm, model="gpt-4o", api_key="test", session_store=self.store
        )
        api.register_handler(
            api.create_handler("ctx_spy")
            .at(HandlerTiming.CONTEXT_UPDATE)
            .do(lambda ctx: self.context_updates.append(sorted(ctx.keys())) or {})
        )
        return api

    def turn(self, field: dict | None = None, bulk: dict | None = None) -> dict:
        self.field, self.bulk = field or {}, bulk or {}
        self.api.converse("hello", self.cid)
        return dict(self.api.get_data(self.cid))


def _str_color_fsm() -> dict:
    """repro2/repro3: `favorite_color` is a declared `str` config, `nickname`
    is named only in `extraction_instructions`."""
    fsm = _correction_fsm()
    fsm["states"]["profile"]["field_extractions"] = [
        {
            "field_name": "favorite_color",
            "field_type": "str",
            "extraction_instructions": "the color",
            "required": True,
        }
    ]
    return fsm


def _typed_fsm() -> dict:
    """ra01: int and bool configs; a gate fires on ``years_old == 25`` (int)."""
    return {
        "name": "Typed",
        "description": "typed drift seam",
        "version": "4.1",
        "initial_state": "s",
        "persona": "x",
        "states": {
            "s": {
                "id": "s",
                "description": "d",
                "purpose": "collect",
                "extraction_instructions": "Extract years_old and is_subscribed.",
                "field_extractions": [
                    {
                        "field_name": "years_old",
                        "field_type": "int",
                        "extraction_instructions": "age",
                        "required": True,
                    },
                    {
                        "field_name": "is_subscribed",
                        "field_type": "bool",
                        "extraction_instructions": "sub",
                        "required": True,
                    },
                ],
                "response_instructions": "Reply.",
                "transitions": [
                    {
                        "target_state": "adult",
                        "description": "years_old is 25",
                        "priority": 200,
                        "conditions": [
                            {
                                "description": "a",
                                "requires_context_keys": ["years_old"],
                                "logic": {"==": [{"var": "years_old"}, 25]},
                            }
                        ],
                    },
                ],
            },
            "adult": {
                "id": "adult",
                "description": "d",
                "purpose": "p",
                "response_instructions": "ok",
                "transitions": [],
            },
        },
    }


def _gate_fsm() -> dict:
    """repro5: a transition gated on ``is_verified == True``; the bulk pass has
    instructions but no per-field config beyond the auto-minted gate key."""
    return {
        "name": "Gate",
        "description": "gate flip seam",
        "version": "4.1",
        "initial_state": "gate",
        "persona": "x",
        "states": {
            "gate": {
                "id": "gate",
                "description": "d",
                "purpose": "verify",
                "extraction_instructions": "Extract the user's stated details.",
                "response_instructions": "Reply.",
                "transitions": [
                    {
                        "target_state": "admin",
                        "description": "verified users only",
                        "priority": 200,
                        "conditions": [
                            {
                                "description": "verified",
                                "requires_context_keys": ["is_verified"],
                                "logic": {"==": [{"var": "is_verified"}, True]},
                            }
                        ],
                    }
                ],
            },
            "admin": {
                "id": "admin",
                "description": "d",
                "purpose": "admin",
                "response_instructions": "Admin area.",
                "transitions": [],
            },
        },
    }


class TestProvenanceOnlyOverwrite:
    """A bulk value may replace a stored key only while the key still holds
    exactly what the pipeline itself extracted (D-015)."""

    def test_handler_seeded_gate_value_is_not_flipped(self):
        """repro5: RED on HEAD, is_verified False -> True and the gate opens."""
        with _Prov(_gate_fsm()) as p:
            p.api.update_context(p.cid, {"is_verified": False})
            data = p.turn(bulk={"is_verified": True})
            assert data["is_verified"] is False
            assert p.api.get_current_state(p.cid) == "gate"

    def test_start_handler_seeded_gate_value_is_not_flipped(self):
        fsm = _gate_fsm()
        with _Prov(fsm) as p:
            p.api.register_handler(
                p.api.create_handler("seed")
                .at(HandlerTiming.START_CONVERSATION)
                .do(lambda ctx: {"is_verified": False})
            )
            cid, _ = p.api.start_conversation()
            p.cid = cid
            assert p.api.get_data(cid)["is_verified"] is False
            p.turn(bulk={"is_verified": True})
            assert p.api.get_data(cid)["is_verified"] is False
            assert p.api.get_current_state(cid) == "gate"

    def test_pipeline_extracted_value_can_be_corrected_repeatedly(self):
        """Vacuity guard: provenance keeps LV-03 alive, and it is re-recorded
        after each landed correction."""
        with _Prov(_correction_fsm()) as p:
            assert p.turn({"favorite_color": "blue"})["favorite_color"] == "blue"
            assert p.turn(bulk={"favorite_color": "red"})["favorite_color"] == "red"
            assert p.turn(bulk={"favorite_color": "teal"})["favorite_color"] == "teal"

    def test_update_context_value_is_not_overwritten(self):
        with _Prov(_correction_fsm()) as p:
            p.turn({"favorite_color": "blue"})
            p.api.update_context(p.cid, {"favorite_color": "teal"})
            assert p.turn(bulk={"favorite_color": "red"})["favorite_color"] == "teal"

    def test_handler_edit_of_an_extracted_value_fails_closed(self):
        with _Prov(_correction_fsm()) as p:
            fired: list[int] = []

            def _edit(ctx):
                # one-shot: a handler that rewrote the key every turn would
                # mask the overwrite this test looks for
                if "favorite_color" in ctx and not fired:
                    fired.append(1)
                    return {"favorite_color": "HANDLER"}
                return {}

            p.api.register_handler(
                p.api.create_handler("edit").at(HandlerTiming.CONTEXT_UPDATE).do(_edit)
            )
            assert p.turn({"favorite_color": "blue"})["favorite_color"] == "HANDLER"
            assert p.turn(bulk={"favorite_color": "red"})["favorite_color"] == (
                "HANDLER"
            )

    def test_rolled_back_extraction_leaves_no_provenance(self):
        """A CONTEXT_UPDATE failure pops the committed key; the same value
        later written by the application must not be overwritable."""
        with _Prov(_correction_fsm()) as p:
            armed = {"on": True}

            def _boom(ctx):
                if armed["on"]:
                    raise RuntimeError("handler failed")
                return {}

            p.api.register_handler(
                p.api.create_handler("boom")
                .at(HandlerTiming.CONTEXT_UPDATE)
                .critical()
                .do(_boom)
            )
            with pytest.raises(Exception):
                p.turn({"favorite_color": "blue"})
            assert "favorite_color" not in p.api.get_data(p.cid)
            armed["on"] = False
            p.api.update_context(p.cid, {"favorite_color": "blue"})
            assert p.turn(bulk={"favorite_color": "red"})["favorite_color"] == "blue"

    def test_restore_session_fails_closed(self, tmp_path):
        """Provenance is not persisted: after a restore a bulk value does not
        overwrite a restored key (D-015 fail-closed)."""
        store = FileSessionStore(str(tmp_path))
        with _Prov(_correction_fsm(), store=store) as p:
            assert p.turn({"favorite_color": "blue"})["favorite_color"] == "blue"
            p.api.save_session(p.cid)
            saved_cid = p.cid
            p.api = p.make_api()
            restored = p.api.restore_session(saved_cid)
            assert restored is not None
            p.cid = restored[0]
            assert p.api.get_data(p.cid)["favorite_color"] == "blue"
            assert p.turn(bulk={"favorite_color": "red"})["favorite_color"] == "blue"

    def test_agent_managed_fsm_never_overwrites_even_pipeline_values(self):
        with _Prov(_correction_fsm()) as p:
            p.turn({"favorite_color": "blue"})
            p.api.update_context(p.cid, {"agent_trace": []})
            assert p.turn(bulk={"favorite_color": "red"})["favorite_color"] == "blue"


class TestBulkValueIsCoercedAndValidated:
    """A bulk value for a config-covered key goes through the same typed
    validation as a per-field value; containers never land in scalar keys."""

    def test_bulk_dict_does_not_replace_a_str_value(self):
        """repro2 turn 3: RED on HEAD, a dict is stored in the `str` field."""
        with _Prov(_str_color_fsm()) as p:
            p.turn({"favorite_color": "blue"})
            data = p.turn(bulk={"favorite_color": {"blue": "blue"}})
            assert data["favorite_color"] == "blue"
            assert isinstance(data["favorite_color"], str)

    @pytest.mark.parametrize("bad", [{"a": 1}, ["blue"]])
    def test_bulk_container_is_not_added_for_an_absent_str_key(self, bad):
        """repro3 R3a: RED on HEAD, the container is stored."""
        with _Prov(_str_color_fsm()) as p:
            data = p.turn(bulk={"favorite_color": bad})
            assert "favorite_color" not in data

    def test_bulk_dict_is_not_added_for_an_any_typed_gate_key(self):
        """The auto-minted config for a `requires_context_keys` entry is
        `field_type="any"`, which has no coercer; the dict is still refused."""
        with _Prov(_correction_fsm()) as p:
            data = p.turn(bulk={"favorite_color": {"a": 1}})
            assert "favorite_color" not in data
            p.turn({"favorite_color": "blue"})
            assert p.turn(bulk={"favorite_color": {"a": 1}})["favorite_color"] == "blue"

    def test_bulk_dict_is_accepted_for_a_dict_typed_key(self):
        """Vacuity guard: the dict refusal is type-driven, not blanket."""
        fsm = _str_color_fsm()
        fsm["states"]["profile"]["field_extractions"][0]["field_type"] = "dict"
        with _Prov(fsm) as p:
            data = p.turn(bulk={"favorite_color": {"a": 1}})
            assert data["favorite_color"] == {"a": 1}

    def test_typed_values_are_not_replaced_by_string_spellings(self):
        """ra01: RED on HEAD, int 24 -> '24' and bool True -> 'true'."""
        with _Prov(_typed_fsm()) as p:
            data = p.turn({"years_old": 24, "is_subscribed": True})
            assert data["years_old"] == 24 and data["is_subscribed"] is True
            data = p.turn(bulk={"years_old": "24", "is_subscribed": "true"})
            assert data["years_old"] == 24 and type(data["years_old"]) is int
            assert data["is_subscribed"] is True
            assert p.api.get_current_state(p.cid) == "s"

    def test_typed_correction_lands_as_the_typed_value(self):
        with _Prov(_typed_fsm()) as p:
            p.turn({"years_old": 24})
            data = p.turn(bulk={"years_old": "25"})
            assert data["years_old"] == 25 and type(data["years_old"]) is int
            assert p.api.get_current_state(p.cid) == "adult"

    def test_uncoercible_bulk_value_is_skipped(self):
        with _Prov(_typed_fsm()) as p:
            data = p.turn(bulk={"years_old": "abc"})
            assert "years_old" not in data
            p.turn({"years_old": 24})
            assert p.turn(bulk={"years_old": "abc"})["years_old"] == 24

    def test_instruction_only_key_keeps_raw_skip_if_set(self):
        """Vacuity guard: instruction-only keys are still added raw and never
        overwritten (no config, so no type information)."""
        with _Prov(_correction_fsm()) as p:
            p.api.update_context(p.cid, {"nickname": "PRESET"})
            assert p.turn(bulk={"nickname": "BULK"})["nickname"] == "PRESET"
        with _Prov(_correction_fsm()) as p:
            assert p.turn(bulk={"nickname": "Bee"})["nickname"] == "Bee"


# ══════════════════════════════════════════════════════════════
# Step 4 / D-018: re-entrancy guard + ERROR-handler no-merge
# ══════════════════════════════════════════════════════════════


def _reentry_fsm() -> dict:
    """repro1: one loopable state so a turn does Pass 2 only."""
    return {
        "name": "Reentry",
        "description": "d",
        "initial_state": "chat",
        "persona": "p",
        "states": {
            "chat": {
                "id": "chat",
                "description": "d",
                "purpose": "p",
                "response_instructions": "Reply.",
                "transitions": [
                    {"target_state": "chat", "description": "stay", "priority": 100},
                    {"target_state": "done", "description": "bye", "priority": 1},
                ],
            },
            "done": {
                "id": "done",
                "description": "d",
                "purpose": "p",
                "response_instructions": "Bye.",
                "transitions": [],
            },
        },
    }


def _collect_fsm() -> dict:
    """ra05: Pass 1 commits `name` and moves a -> b, Pass 2 can then fail."""
    return {
        "name": "Collect",
        "description": "d",
        "initial_state": "a",
        "persona": "p",
        "states": {
            "a": {
                "id": "a",
                "description": "d",
                "purpose": "collect",
                "response_instructions": "Reply.",
                "field_extractions": [
                    {
                        "field_name": "name",
                        "field_type": "str",
                        "extraction_instructions": "n",
                        "required": True,
                    }
                ],
                "transitions": [
                    {
                        "target_state": "b",
                        "description": "have name",
                        "priority": 1,
                        "conditions": [
                            {
                                "description": "n",
                                "requires_context_keys": ["name"],
                                "logic": {"!=": [{"var": "name"}, None]},
                            }
                        ],
                    }
                ],
            },
            "b": {
                "id": "b",
                "description": "d",
                "purpose": "p",
                "response_instructions": "Reply b.",
                "transitions": [
                    {"target_state": "b", "description": "loop", "priority": 100},
                    {
                        "target_state": "z",
                        "description": "end",
                        "priority": 1,
                        "conditions": [
                            {
                                "description": "never",
                                "requires_context_keys": ["zzz"],
                                "logic": {"==": [{"var": "zzz"}, 1]},
                            }
                        ],
                    },
                ],
            },
            "z": {
                "id": "z",
                "description": "d",
                "purpose": "p",
                "response_instructions": "bye",
                "transitions": [],
            },
        },
    }


class _Provider:
    """Scripted provider with an outage switch and a call counter.

    ``fail_pass2`` fails only the non-extraction, non-stream generation call, so
    a Pass 1 extraction can commit before Pass 2 dies; ``fail_stream`` fails the
    streaming call. ``down`` fails everything.
    """

    def __init__(self):
        self.calls = 0
        self.down = False
        self.fail_pass2 = False
        self.fail_stream = False
        self.name_value = "Bob"

    def completion(self, **kwargs):
        self.calls += 1
        if self.down:
            raise RuntimeError("provider down")
        if kwargs.get("stream"):
            if self.fail_stream:
                raise RuntimeError("stream provider down")
            return iter(_stream_chunks(["Hel", "lo"]))
        system = kwargs["messages"][0]["content"]
        if kwargs.get("response_format") is not None:
            return _fake_response(
                json.dumps(
                    {"field_name": "name", "value": self.name_value, "confidence": 0.9}
                )
            )
        if self.fail_pass2 and "Extract" not in system[:300]:
            raise RuntimeError("provider down")
        return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))


def _stream_chunks(parts):
    from unittest.mock import MagicMock

    out = []
    for p in parts:
        c = MagicMock()
        c.choices = [MagicMock()]
        c.choices[0].delta.content = p
        c.choices[0].delta.reasoning_content = None
        c.choices[0].finish_reason = None
        out.append(c)
    return out


class _Session:
    def __init__(self, fsm: dict):
        self.fsm = fsm
        self.provider = _Provider()
        self._stack = ExitStack()

    def __enter__(self):
        self._stack.enter_context(
            patch("fsm_llm.llm.completion", side_effect=self.provider.completion)
        )
        self._stack.enter_context(
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=["response_format"],
            )
        )
        self.api = API.from_definition(self.fsm, model="gpt-4o", api_key="test")
        self.cid, _ = self.api.start_conversation()
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def on(self, timing, fn, name="h"):
        self.api.register_handler(self.api.create_handler(name).at(timing).do(fn))


class TestSameConversationReentrancyIsBounded:
    def test_error_handler_reentering_converse_does_not_recurse(self):
        """repro1 port: RED on HEAD, the ERROR handler recursed 99 deep."""
        from fsm_llm.definitions import FSMError

        depth = {"n": 0, "max": 0}
        raised: list[BaseException] = []
        with _Session(_reentry_fsm()) as s:

            def on_error(ctx):
                depth["n"] += 1
                depth["max"] = max(depth["max"], depth["n"])
                try:
                    s.api.converse("sorry, retrying", s.cid)
                except FSMError as e:
                    raised.append(e)
                finally:
                    depth["n"] -= 1
                return {}

            s.on(HandlerTiming.ERROR, on_error)
            s.provider.down = True
            with pytest.raises(FSMError):
                s.api.converse("hello", s.cid)
            assert depth["max"] == 1
            assert s.provider.calls <= 2
            assert len(raised) == 1
            assert "already being processed" in str(raised[0])

    def test_error_handler_reentering_converse_stream_does_not_recurse(self):
        from fsm_llm.definitions import FSMError

        depth = {"max": 0, "n": 0}
        with _Session(_reentry_fsm()) as s:

            def on_error(ctx):
                depth["n"] += 1
                depth["max"] = max(depth["max"], depth["n"])
                try:
                    list(s.api.converse_stream("retry", s.cid))
                except FSMError:
                    pass
                finally:
                    depth["n"] -= 1
                return {}

            s.on(HandlerTiming.ERROR, on_error)
            s.provider.fail_stream = True
            with pytest.raises(FSMError):
                list(s.api.converse_stream("hello", s.cid))
            assert depth["max"] == 1
            assert s.provider.calls <= 2

    def test_pre_processing_handler_calling_converse_raises_not_nests(self):
        from fsm_llm.definitions import FSMError

        seen: list[BaseException] = []
        with _Session(_reentry_fsm()) as s:
            armed = {"on": False}

            def pre(ctx):
                if armed["on"]:
                    try:
                        s.api.converse("nested", s.cid)
                    except FSMError as e:
                        seen.append(e)
                return {}

            s.on(HandlerTiming.PRE_PROCESSING, pre)
            armed["on"] = True
            s.api.converse("hello", s.cid)
            assert len(seen) == 1
            assert "already being processed" in str(seen[0])
            # the nested call never ran: its message is nowhere in the history
            users = [
                m["user"] for m in s.api.get_conversation_history(s.cid) if "user" in m
            ]
            assert users == ["hello"]

    def test_error_handler_can_still_write_via_update_context(self):
        from fsm_llm.definitions import FSMError

        with _Session(_reentry_fsm()) as s:
            s.on(
                HandlerTiming.ERROR,
                lambda ctx: s.api.update_context(s.cid, {"error_seen": True}) or {},
            )
            s.provider.down = True
            with pytest.raises(FSMError):
                s.api.converse("hello", s.cid)
            assert s.api.get_data(s.cid).get("error_seen") is True

    def test_guard_is_released_after_a_failed_turn(self):
        from fsm_llm.definitions import FSMError

        with _Session(_reentry_fsm()) as s:
            s.provider.down = True
            with pytest.raises(FSMError):
                s.api.converse("hello", s.cid)
            s.provider.down = False
            assert s.api.converse("again", s.cid)

    def test_abandoned_stream_releases_the_guard(self):
        with _Session(_reentry_fsm()) as s:
            gen = s.api.converse_stream("hello", s.cid)
            next(gen)
            gen.close()
            assert s.api.converse("after close", s.cid)
            gen2 = s.api.converse_stream("hello", s.cid)
            next(gen2)
            del gen2  # CPython closes an unreferenced generator
            assert s.api.converse("after del", s.cid)

    def test_never_iterated_stream_does_not_hold_the_guard(self):
        with _Session(_reentry_fsm()) as s:
            _unused = s.api.converse_stream("hello", s.cid)
            assert s.api.converse("meanwhile", s.cid)
            del _unused

    def test_converse_between_stream_iterations_raises(self):
        """Documented D-018 cost: an open stream owns its conversation."""
        from fsm_llm.definitions import FSMError

        with _Session(_reentry_fsm()) as s:
            gen = s.api.converse_stream("hello", s.cid)
            next(gen)
            with pytest.raises(FSMError, match="already being processed"):
                s.api.converse("interleaved", s.cid)
            gen.close()


class TestErrorHandlerReturnIsNotMerged:
    def test_returned_delta_from_error_handler_is_dropped_sync_and_stream(self):
        """ra05 port: Pass-2 failure after a committed Pass 1."""
        from fsm_llm.definitions import FSMError

        fired: list[int] = []
        with _Session(_collect_fsm()) as s:

            def on_error(ctx):
                fired.append(1)
                return {"handler_marker": "set-by-ERROR-handler"}

            s.on(HandlerTiming.ERROR, on_error)
            s.provider.fail_pass2 = True
            with pytest.raises(FSMError):
                s.api.converse("I am Bob", s.cid)
            assert len(fired) == 1
            data = s.api.get_data(s.cid)
            assert "handler_marker" not in data
            # the turn was rolled back: nothing from Pass 1 survived
            assert "name" not in data
            assert s.api.get_current_state(s.cid) == "a"

            fired.clear()
            s.provider.fail_pass2 = False
            s.provider.fail_stream = True
            # advance to b first so the stream turn has a non-terminal state
            s.api.converse("I am Bob", s.cid)
            assert s.api.get_current_state(s.cid) == "b"
            with pytest.raises(FSMError):
                list(s.api.converse_stream("hi again", s.cid))
            assert len(fired) == 1
            assert "handler_marker" not in s.api.get_data(s.cid)

    def test_non_error_handler_delta_is_still_merged(self):
        """Vacuity guard: only ERROR timing stopped merging."""
        with _Session(_reentry_fsm()) as s:
            s.on(HandlerTiming.PRE_PROCESSING, lambda ctx: {"pre_marker": 1})
            s.api.converse("hello", s.cid)
            assert s.api.get_data(s.cid).get("pre_marker") == 1


# ══════════════════════════════════════════════════════════════
# Step 6 / RA-04 + RA-06: extract_json_from_text fenced non-dict and depth
# ══════════════════════════════════════════════════════════════


class TestExtractJsonFencedNonDictAndDepth:
    def test_fenced_array_interior_object_is_not_returned(self):
        from fsm_llm.utilities import extract_json_from_text

        # RA-04: the fence body is a top-level array, which is not a payload.
        # The brace scan must not recover the object inside it.
        assert extract_json_from_text('```json\n[{"a":1}]\n```') is None

    def test_fenced_array_then_real_object_returns_the_object(self):
        from fsm_llm.utilities import extract_json_from_text

        text = '```json\n[1]\n```\n{"b":2}'
        assert extract_json_from_text(text) == {"b": 2}
        # the interior object of the array is skipped, the later one wins
        text = '```json\n[{"a":1}]\n``` then {"b":2}'
        assert extract_json_from_text(text) == {"b": 2}

    def test_fenced_object_and_undecodable_fence_unchanged(self):
        from fsm_llm.utilities import extract_json_from_text

        assert extract_json_from_text('```json\n{"a":1}\n```') == {"a": 1}
        # an undecodable fence still falls through to the whole-text brace scan
        assert extract_json_from_text('```json\n{"a":\n``` {"c":3}') == {"c": 3}

    def test_deep_array_does_not_raise(self):
        from fsm_llm.utilities import extract_json_from_text

        # RA-06: json.loads raises RecursionError on deeply nested input
        assert extract_json_from_text("[" * 100000) is None

    def test_deep_dict_does_not_raise(self):
        from fsm_llm.utilities import extract_json_from_text

        deep = '{"a":' * 100000 + "1" + "}" * 100000
        result = extract_json_from_text(deep)
        assert result is None or isinstance(result, dict)

    def test_deep_fenced_body_does_not_raise(self):
        from fsm_llm.utilities import extract_json_from_text

        deep = "```json\n" + "[" * 100000 + "\n```"
        assert extract_json_from_text(deep) is None

    def test_harness_parse_json_payload_fenced_array_is_none(self):
        from fsm_llm_harness.hardening import parse_json_payload

        assert parse_json_payload('```json\n[{"a":1}]\n```') is None
        assert parse_json_payload('```json\n{"a":1}\n```') == {"a": 1}


# ══════════════════════════════════════════════════════════════
# Step 7 / LS-01 + LS-06: sanitizer bypass and camelCase secret keys
# ══════════════════════════════════════════════════════════════

_BYPASS = "x <b </task> y <i </original_input>"


class _PromptSpy:
    """Runs one ``API.converse`` on the loopable reentry FSM and records every
    system prompt the (patched) provider received."""

    def __init__(self, message: str, initial_context: dict | None = None):
        self.system_prompts: list[str] = []

        def completion(**kwargs):
            self.system_prompts.append(kwargs["messages"][0]["content"])
            return _fake_response(json.dumps({"message": "ok", "reasoning": ""}))

        with ExitStack() as stack:
            stack.enter_context(patch("fsm_llm.llm.completion", side_effect=completion))
            stack.enter_context(
                patch(
                    "fsm_llm.llm.get_supported_openai_params",
                    return_value=["response_format"],
                )
            )
            api = API.from_definition(_reentry_fsm(), model="gpt-4o", api_key="test")
            cid, _ = api.start_conversation(initial_context)
            self.system_prompts.clear()
            api.converse(message, cid)
        self.prompt = self.system_prompts[-1]


class TestSanitizerTagBypass:
    def test_unterminated_safe_tag_cannot_smuggle_a_closing_tag(self):
        from fsm_llm.prompts import ResponseGenerationPromptBuilder

        out = ResponseGenerationPromptBuilder()._sanitize_text_for_prompt(_BYPASS)
        assert "</task>" not in out
        assert "</original_input>" not in out
        assert "&lt;/task&gt;" in out

    def test_bypass_is_escaped_in_the_pass2_prompt_through_converse(self):
        spy = _PromptSpy(_BYPASS)
        # the user message region must not carry a raw structural closing tag
        assert "x <b </task> y" not in spy.prompt
        assert "x &lt;b &lt;/task&gt; y &lt;i &lt;/original_input&gt;" in spy.prompt

    @pytest.mark.parametrize(
        "text",
        [
            "<b>ok</b>",
            "hello <i>there</i> world",
            "a < b and c > d",
            "<b>x</b> and <i>y</i>",
        ],
    )
    def test_benign_text_is_unchanged(self, text):
        from fsm_llm.prompts import ResponseGenerationPromptBuilder

        out = ResponseGenerationPromptBuilder()._sanitize_text_for_prompt(text)
        assert out == text

    def test_benign_pass2_prompt_hash_is_byte_identical_to_pre_change(self):
        """Hash recorded on the unfixed tree (step-7 RED run)."""
        import hashlib

        spy = _PromptSpy("<b>ok</b> just some ordinary prose, thanks.")
        digest = hashlib.sha256(spy.prompt.encode()).hexdigest()
        assert (
            digest == "f85e32071530b17668df3d9fbfda9b1835938bd6b247d020deea779f06231871"
        )


class TestCamelCaseSecretKeys:
    @pytest.mark.parametrize(
        "key",
        [
            "newPassword",
            "confirmPassword",
            "adminPassword",
            "clientSecret",
            "appSecret",
            "userCredentials",
            "dbCredential",
            "awsSecretKey",
            "sshPrivateKey",
        ],
    )
    def test_camel_case_secret_names_are_forbidden(self, key):
        from fsm_llm.constants import is_forbidden_context_entry

        assert is_forbidden_context_entry(key, "hunter2hunter2") is True

    @pytest.mark.parametrize(
        "key", ["tokenCount", "passwordPolicy", "keyboardLayout", "primaryKey"]
    )
    def test_camel_case_lookalikes_stay_open(self, key):
        from fsm_llm.constants import is_forbidden_context_entry

        assert is_forbidden_context_entry(key, 5) is False

    def test_snake_case_names_unchanged(self):
        from fsm_llm.constants import is_forbidden_context_entry

        assert is_forbidden_context_entry("new_password", "x") is True
        assert is_forbidden_context_entry("user_name", "x") is False

    def test_hostile_str_subclass_key_is_not_camel_split(self):
        from fsm_llm.constants import is_forbidden_context_entry

        class Hostile(str):
            def __getitem__(self, i):
                raise RuntimeError("boom")

        assert is_forbidden_context_entry(Hostile("userName"), "x") is False

    def test_camel_case_secret_never_reaches_the_pass2_prompt(self):
        spy = _PromptSpy("hi", initial_context={"newPassword": "S3CRETVALUE-42x"})
        assert "S3CRETVALUE-42x" not in spy.prompt
