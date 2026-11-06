"""Behavioural fixes of the 2026-11-05 source pass (plan 5980fbb7, step 4).

Each class names the finding it pins. Every test here fails on the parent
commit of the fix it guards.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, ClassVar

import pytest

from fsm_llm import API, APIBuilder, ContextCompactor
from fsm_llm.definitions import (
    Conversation,
    FieldExtractionConfig,
    FieldExtractionResponse,
    FSMContext,
    FSMError,
    FSMInstance,
)
from fsm_llm.fsm import FSMManager
from fsm_llm.llm import _embedding_vectors
from fsm_llm.logging import logger
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.prompts import (
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
)
from fsm_llm.utilities import coerce_confidence, coerce_float, coerce_int
from tests.conftest import MockLLM2Interface

_TWO_STATE = {
    "name": "b",
    "description": "d",
    "initial_state": "s",
    "persona": "p",
    "states": {
        "s": {
            "id": "s",
            "description": "d",
            "purpose": "p",
            "response_instructions": "hi",
            "transitions": [
                {"target_state": "e", "description": "go", "priority": 100}
            ],
        },
        "e": {
            "id": "e",
            "description": "d",
            "purpose": "p",
            "response_instructions": "bye",
        },
    },
}

_ONE_STATE = {
    "name": "one",
    "description": "d",
    "initial_state": "a",
    "states": {
        "a": {
            "id": "a",
            "description": "d",
            "purpose": "p",
            "response_instructions": "hi",
        }
    },
}


@contextmanager
def _debug_records():
    """fsm_llm DEBUG records (the package is disabled at import)."""
    records: list[str] = []
    logger.enable("fsm_llm")
    sink_id = logger.add(lambda msg: records.append(str(msg)), level="DEBUG")
    try:
        yield records
    finally:
        logger.remove(sink_id)
        logger.disable("fsm_llm")


def _validate(field_type: str, value: Any) -> FieldExtractionResponse:
    return MessagePipeline._validate_field_extraction(
        FieldExtractionResponse(
            field_name="n", value=value, confidence=1.0, is_valid=True
        ),
        FieldExtractionConfig(
            field_name="n",
            field_type=field_type,  # type: ignore[arg-type]
            extraction_instructions="get n",
            confidence_threshold=0.0,
        ),
    )


class TestNumericCoercionRejectsWhatIsNotANumber:
    """F-PI-01: inf / nan / bool / truncation for int and float fields."""

    @pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan, 3.7, True, False])
    def test_int_field_marks_the_value_invalid_and_never_raises(self, bad):
        out = _validate("int", bad)
        assert out.is_valid is False
        assert out.validation_error

    @pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan, True, "nan", "inf"])
    def test_float_field_rejects_non_finite_and_bool(self, bad):
        out = _validate("float", bad)
        assert out.is_valid is False

    @pytest.mark.parametrize(
        ("field_type", "huge"),
        [
            pytest.param("float", 10**400, id="float-int-1e400"),
            pytest.param("float", -(10**400), id="float-int-neg-1e400"),
            pytest.param("float", "1" * 400, id="float-400-digit-string"),
            pytest.param("float", Decimal("Infinity"), id="float-decimal-inf"),
            pytest.param("int", Decimal("Infinity"), id="int-decimal-inf"),
            pytest.param("int", Decimal("-Infinity"), id="int-decimal-neg-inf"),
        ],
    )
    def test_a_number_too_large_for_a_float_is_invalid_not_an_exception(
        self, field_type, huge
    ):
        """REVIEW-C3: ``float(10**400)`` and ``int(Decimal('Infinity'))`` raise
        ``OverflowError``, which ``_validate_field_extraction`` does not name;
        a model-written 400-digit JSON number reached it on both extraction
        paths (per field and bulk), which share this function."""
        out = _validate(field_type, huge)
        assert out.is_valid is False
        assert out.validation_error

    def test_the_helpers_keep_their_value_error_contract(self):
        with pytest.raises(ValueError):
            coerce_float(10**400)
        with pytest.raises(ValueError):
            coerce_int(Decimal("Infinity"))

    @pytest.mark.parametrize("good", [7, 3.0, "12", -4])
    def test_int_field_still_takes_integral_values(self, good):
        out = _validate("int", good)
        assert out.is_valid is True
        assert out.value == int(good)
        assert isinstance(out.value, int)

    @pytest.mark.parametrize("good", [7, 3.5, "2.25", -1e3])
    def test_float_field_still_takes_finite_numbers(self, good):
        out = _validate("float", good)
        assert out.is_valid is True
        assert out.value == float(good)


def _validate_with_rules(
    field_type: str, value: Any, rules: dict[str, Any]
) -> FieldExtractionResponse:
    return MessagePipeline._validate_field_extraction(
        FieldExtractionResponse(
            field_name="n", value=value, confidence=1.0, is_valid=True
        ),
        FieldExtractionConfig(
            field_name="n",
            field_type=field_type,  # type: ignore[arg-type]
            extraction_instructions="get n",
            confidence_threshold=0.0,
            validation_rules=rules,
        ),
    )


def _range_machine(field_type: str, rules: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "R",
        "description": "d",
        "initial_state": "ask",
        "states": {
            "ask": {
                "id": "ask",
                "description": "ask",
                "purpose": "get n",
                "field_extractions": [
                    {
                        "field_name": "n",
                        "field_type": field_type,
                        "extraction_instructions": "n",
                        "validation_rules": rules,
                    }
                ],
                "response_instructions": "ask",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "has n",
                        "priority": 1,
                        "conditions": [
                            {"description": "n", "requires_context_keys": ["n"]}
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "end",
                "purpose": "end",
                "response_instructions": "bye",
            },
        },
    }


_HUGE_VALUES = [
    pytest.param(10**400, id="int-1e400"),
    pytest.param(-(10**400), id="int-neg-1e400"),
    pytest.param("1" * 400, id="digit-string-400"),
]


class TestRangeRulesNeverOverflow:
    """REVIEW2-C1 (F-PI-01): ``float(10**400)`` raised ``OverflowError`` out of
    the ``min_value``/``max_value`` checks, so a 400-digit number from the model
    ended the turn with ``FSMError``. A huge value is out of range: an invalid
    extraction, never an exception."""

    @pytest.mark.parametrize("field_type", ["int", "any"])
    @pytest.mark.parametrize("huge", _HUGE_VALUES)
    @pytest.mark.parametrize("rule", ["min_value", "max_value"])
    def test_a_huge_value_with_a_range_rule_is_decided_not_raised(
        self, field_type, huge, rule
    ):
        out = _validate_with_rules(field_type, huge, {rule: 10})
        positive = isinstance(huge, str) or huge > 0
        out_of_range = positive if rule == "max_value" else not positive
        assert out.is_valid is (not out_of_range)
        if out_of_range:
            assert out.validation_error

    @pytest.mark.parametrize("huge", _HUGE_VALUES)
    def test_a_huge_value_is_rejected_by_the_matching_bound_of_a_closed_range(
        self, huge
    ):
        positive = isinstance(huge, str) or huge > 0
        out = _validate_with_rules("int", huge, {"min_value": 0, "max_value": 10})
        assert out.is_valid is False
        assert ("exceeds maximum" if positive else "below minimum") in (
            out.validation_error or ""
        )

    def test_a_huge_value_below_an_enormous_max_still_passes(self):
        """The check is numeric, not skipped: 10**308 fits a float bound of
        1e308 * 1.5 only as an exact comparison, and 10**309 does not."""
        rules = {"max_value": 1.7e308}
        assert _validate_with_rules("int", 10**308, rules).is_valid is True
        assert _validate_with_rules("int", 10**309, rules).is_valid is False
        assert _validate_with_rules("int", 10**400, rules).is_valid is False
        low = {"min_value": -1.7e308}
        assert _validate_with_rules("int", -(10**308), low).is_valid is True
        assert _validate_with_rules("int", -(10**309), low).is_valid is False

    @pytest.mark.parametrize("field_type", ["int", "float", "any"])
    @pytest.mark.parametrize(
        ("value", "valid"), [(5, True), (0, True), (10, True), (11, False), (-1, False)]
    )
    def test_a_normal_value_keeps_its_verdict(self, field_type, value, valid):
        out = _validate_with_rules(field_type, value, {"min_value": 0, "max_value": 10})
        assert out.is_valid is valid

    def test_a_float_field_with_a_huge_value_is_invalid_with_a_range_rule(self):
        for huge in (10**400, -(10**400), "1" * 400):
            out = _validate_with_rules("float", huge, {"min_value": 0, "max_value": 10})
            assert out.is_valid is False

    @pytest.mark.parametrize("field_type", ["int", "any"])
    @pytest.mark.parametrize("huge", _HUGE_VALUES)
    def test_converse_survives_a_huge_number_and_stays_in_the_state(
        self, field_type, huge
    ):
        machine = _range_machine(field_type, {"min_value": 0, "max_value": 10})
        llm = MockLLM2Interface(extraction_data={"n": huge})
        api = API.from_definition(machine, llm_interface=llm)
        cid, _ = api.start_conversation()
        api.converse("a very big number", cid)
        assert api.get_current_state(cid) == "ask"
        assert "n" not in api.get_data(cid)

    def test_converse_still_accepts_an_in_range_number(self):
        machine = _range_machine("int", {"min_value": 0, "max_value": 10})
        llm = MockLLM2Interface(extraction_data={"n": 7})
        api = API.from_definition(machine, llm_interface=llm)
        cid, _ = api.start_conversation()
        api.converse("seven", cid)
        assert api.get_current_state(cid) == "done"
        assert api.get_data(cid)["n"] == 7


class TestOtherFloatSitesDoNotOverflow:
    """REVIEW2-C1 sweep: the same ``float()``/``math.isfinite`` on an integer
    too large for a float, at the two other model/provider-fed sites."""

    @pytest.mark.parametrize("huge", [10**400, -(10**400)], ids=["pos", "neg"])
    def test_a_huge_confidence_reads_as_the_default_like_infinity(self, huge):
        assert coerce_confidence(huge, 0.5) == 0.5
        assert coerce_confidence(math.inf, 0.5) == 0.5

    @pytest.mark.parametrize("huge", [10**400, -(10**400)], ids=["pos", "neg"])
    def test_an_embedding_with_a_huge_integer_is_a_refused_reply(self, huge):
        from fsm_llm.definitions import LLMResponseError

        reply = {"data": [{"index": 0, "embedding": [0.5, huge]}]}
        with pytest.raises(LLMResponseError, match="non-finite"):
            _embedding_vectors(reply, 1)


class TestAdvanceOnATerminalStateIsRefusedBeforeAnyWork:
    """F-PI-05: sync ``advance`` used to run the whole turn, then raise."""

    def _terminal(self):
        llm = MockLLM2Interface(response_text="hello")
        api = API.from_definition(_TWO_STATE, llm_interface=llm)
        cid, _ = api.start_conversation()
        api.converse("go", cid)
        assert api.get_current_state(cid) == "e"
        return api, llm, cid

    def test_sync_makes_no_llm_call_and_leaves_history_alone(self):
        api, llm, cid = self._terminal()
        pipeline = api.fsm_manager._pipeline
        instance = api.fsm_manager.instances[cid]
        calls = len(llm.call_history)
        exchanges = len(instance.context.conversation.exchanges)
        with pytest.raises(FSMError, match="Cannot advance from terminal state 'e'"):
            pipeline.advance(instance, cid)
        assert len(llm.call_history) == calls
        assert len(instance.context.conversation.exchanges) == exchanges

    def test_stream_twin_agrees(self):
        api, llm, cid = self._terminal()
        pipeline = api.fsm_manager._pipeline
        instance = api.fsm_manager.instances[cid]
        calls = len(llm.call_history)
        with pytest.raises(FSMError, match="Cannot advance from terminal state 'e'"):
            list(pipeline.advance_stream(instance, cid))
        assert len(llm.call_history) == calls


class TestGreetingHonoursTheOutputResponseFormat:
    """F-PI-09: a terminal-initial FSM's only reply carries the format."""

    _FORMAT: ClassVar[dict[str, str]] = {"type": "json_object"}

    def _greeting_request(self, context):
        llm = MockLLM2Interface(response_text="hello")
        api = API.from_definition(_ONE_STATE, llm_interface=llm)
        api.start_conversation(initial_context=context)
        (request,) = [r for kind, r in llm.call_history if kind == "generate_response"]
        return request

    def test_terminal_initial_state_sends_the_format(self):
        request = self._greeting_request({"_output_response_format": self._FORMAT})
        assert request.response_format == self._FORMAT

    def test_no_format_in_context_sends_none(self):
        assert self._greeting_request(None).response_format is None

    def test_non_terminal_initial_state_still_sends_none(self):
        llm = MockLLM2Interface(response_text="hello")
        api = API.from_definition(_TWO_STATE, llm_interface=llm)
        api.start_conversation(
            initial_context={"_output_response_format": self._FORMAT}
        )
        (request,) = [r for kind, r in llm.call_history if kind == "generate_response"]
        assert request.response_format is None


class TestHistoryAndLengthLimitsAreValidatedAtConstruction:
    """F-AF-04: a bad limit used to fail every start_conversation instead."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_history_size": -1},
            {"max_history_size": 1001},
            {"max_message_length": 0},
            {"max_message_length": 50001},
        ],
    )
    def test_api_constructor_refuses_the_value(self, kwargs):
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            API.from_definition(_TWO_STATE, llm_interface=MockLLM2Interface(), **kwargs)

    @pytest.mark.parametrize(
        "kwargs", [{"max_history_size": -1}, {"max_message_length": 0}]
    )
    def test_manager_constructor_refuses_the_value(self, kwargs):
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            FSMManager(llm_interface=MockLLM2Interface(), **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_history_size": 0, "max_message_length": 1},
            {"max_history_size": 1000, "max_message_length": 50000},
        ],
    )
    def test_the_bounds_themselves_are_accepted(self, kwargs):
        api = API.from_definition(
            _TWO_STATE, llm_interface=MockLLM2Interface(), **kwargs
        )
        api.start_conversation()


class TestStartConversationErrorContract:
    """F-AF-05 (type) and F-AF-12 (falsy ``initial_context``)."""

    def test_a_value_error_passes_through_like_push_and_pop(self, monkeypatch):
        api = API.from_definition(_TWO_STATE, llm_interface=MockLLM2Interface())

        def boom(*args, **kwargs):
            raise ValueError("bad value")

        monkeypatch.setattr(api.fsm_manager, "start_conversation", boom)
        with pytest.raises(ValueError, match="bad value") as caught:
            api.start_conversation()
        assert not isinstance(caught.value, FSMError)

    def test_an_unexpected_error_is_still_wrapped(self, monkeypatch):
        api = API.from_definition(_TWO_STATE, llm_interface=MockLLM2Interface())

        def boom(*args, **kwargs):
            raise RuntimeError("kaput")

        monkeypatch.setattr(api.fsm_manager, "start_conversation", boom)
        with pytest.raises(FSMError, match="Failed to start conversation: kaput"):
            api.start_conversation()

    @pytest.mark.parametrize("bad", [[], "", 0, [1], "x"])
    def test_a_non_dict_initial_context_is_refused_whatever_its_truthiness(self, bad):
        api = API.from_definition(_TWO_STATE, llm_interface=MockLLM2Interface())
        with pytest.raises(FSMError, match="initial_context must be a dict"):
            api.start_conversation(initial_context=bad)

    @pytest.mark.parametrize("fine", [None, {}, {"k": "v"}])
    def test_none_and_dicts_still_start(self, fine):
        api = API.from_definition(_TWO_STATE, llm_interface=MockLLM2Interface())
        cid, _ = api.start_conversation(initial_context=fine)
        assert cid


class TestContextCompactorUsesClearKeysDelta:
    """F-AF-02: same validation and same clear rule as ``clear_keys_delta``."""

    @pytest.mark.parametrize("bad", ["abc", b"abc", 5, [1, 2]])
    def test_transient_keys_must_be_a_collection_of_names(self, bad):
        with pytest.raises(ValueError, match="transient_keys"):
            ContextCompactor(transient_keys=bad)

    @pytest.mark.parametrize("bad", ["ab", [1], 7])
    def test_prune_keys_must_be_a_collection_of_names(self, bad):
        with pytest.raises(ValueError, match="prune_on_entry"):
            ContextCompactor(prune_on_entry={"review": bad})

    def test_none_arguments_still_mean_nothing_to_clear(self):
        compactor = ContextCompactor(transient_keys=None, prune_on_entry=None)
        assert compactor.compact({"a": 1}) == {}
        assert compactor.prune({"_current_state": "s", "a": 1}) == {}

    def test_a_key_holding_none_is_not_cleared(self):
        compactor = ContextCompactor(
            transient_keys={"x", "y"}, prune_on_entry={"s": {"x", "y"}}
        )
        context = {"x": None, "y": 1, "_current_state": "s"}
        assert compactor.compact(context) == {"y": None}
        assert compactor.prune(context) == {"y": None}


def _instance(exchanges: list[dict[str, str]]) -> FSMInstance:
    conv = Conversation(exchanges=exchanges, max_history_size=10)
    return FSMInstance(
        fsm_id="f", current_state="s", context=FSMContext(data={}, conversation=conv)
    )


_FIELD = FieldExtractionConfig(
    field_name="x", field_type="str", extraction_instructions="get x"
)


class TestFieldPromptContextGuard:
    """F-SU-04: an unserialisable context value drops the section, not the turn."""

    @pytest.mark.parametrize(
        "context", [{"a": {1: "x", "b": "y"}}, {"a": 10**5000}], ids=["keys", "int"]
    )
    def test_prompt_is_built_without_the_context_section(self, context):
        prompt = FieldExtractionPromptBuilder().build_field_extraction_prompt(
            _instance([]), _FIELD, "hi", dynamic_context=context
        )
        assert "get x" in prompt
        assert "10000000" not in prompt

    def test_a_good_context_still_reaches_the_prompt(self):
        prompt = FieldExtractionPromptBuilder().build_field_extraction_prompt(
            _instance([]), _FIELD, "hi", dynamic_context={"colour": "teal"}
        )
        assert "teal" in prompt


class TestOneHistoryRoleRule:
    """F-SU-16: the three history renderers share one role rule."""

    def test_the_pass_two_history_keeps_every_message_of_a_multi_key_entry(self):
        instance = _instance(
            [{"assistant": "A-msg", "system": "S-msg"}, {"user": "U1"}]
        )
        text = "\n".join(
            DataExtractionPromptBuilder()._build_enhanced_history_section(instance)
        )
        assert "A-msg" in text
        assert "S-msg" in text
        assert "U1" in text

    def test_role_case_is_ignored_by_the_field_prompt_like_the_others(self):
        instance = _instance([{"User": "U-cap"}, {"system": "S"}])
        prompt = FieldExtractionPromptBuilder().build_field_extraction_prompt(
            instance, _FIELD, "hi"
        )
        assert "User: U-cap" in prompt
        assert "Assistant: U-cap" not in prompt

    def test_pass_two_history_marks_a_capitalised_role_as_the_user(self):
        instance = _instance([{"User": "U-cap"}])
        text = "\n".join(
            DataExtractionPromptBuilder()._build_enhanced_history_section(instance)
        )
        assert '"user": "U-cap"' in text


class TestRejectedExtractionReasonIsLogged:
    """F-XC-12: ``validation_error`` reaches the DEBUG line."""

    def test_the_reason_is_in_the_log(self):
        fsm = {
            "name": "f",
            "description": "d",
            "initial_state": "a",
            "states": {
                "a": {
                    "id": "a",
                    "description": "d",
                    "purpose": "p",
                    "response_instructions": "hi",
                    "field_extractions": [
                        {
                            "field_name": "n",
                            "field_type": "int",
                            "extraction_instructions": "get n",
                        }
                    ],
                    "transitions": [
                        {"target_state": "b", "description": "go", "priority": 1}
                    ],
                },
                "b": {"id": "b", "description": "d", "purpose": "p"},
            },
        }
        llm = MockLLM2Interface(extraction_data={"n": 3.7})
        api = API.from_definition(fsm, llm_interface=llm)
        cid, _ = api.start_conversation()
        with _debug_records() as records:
            api.converse("my number is 3.7", cid)
        joined = "\n".join(records)
        assert "Type coercion to int failed" in joined


class TestStreamFailureIsLoggedLikeTheSyncTurn:
    """F-AF-09: a failing stream turn left no ERROR line, the sync turn did."""

    def _failing_api(self, monkeypatch):
        api = API.from_definition(
            _TWO_STATE, llm_interface=MockLLM2Interface(response_text="x")
        )
        cid, _ = api.start_conversation()

        def boom(*args, **kwargs):
            raise RuntimeError("pipeline exploded")

        monkeypatch.setattr(api.fsm_manager._pipeline, "process", boom)
        monkeypatch.setattr(api.fsm_manager._pipeline, "process_stream", boom)
        return api, cid

    def _error_records(self, run):
        records: list[str] = []
        logger.enable("fsm_llm")
        sink_id = logger.add(lambda msg: records.append(str(msg)), level="ERROR")
        try:
            with pytest.raises(FSMError, match="Failed to process message"):
                run()
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")
        return records

    def test_sync_turn_logs_the_failure(self, monkeypatch):
        api, cid = self._failing_api(monkeypatch)
        records = self._error_records(lambda: api.converse("hi", cid))
        assert any("pipeline exploded" in r for r in records)

    def test_stream_turn_logs_the_failure(self, monkeypatch):
        api, cid = self._failing_api(monkeypatch)
        records = self._error_records(lambda: list(api.converse_stream("hi", cid)))
        assert any("pipeline exploded" in r for r in records)


class TestPathDefinitionIsAcceptedEverywhere:
    """F-AF-19: ``API(Path)`` was refused while ``API.from_file(Path)`` worked."""

    @pytest.fixture
    def fsm_path(self, tmp_path):
        path = tmp_path / "fsm.json"
        path.write_text(json.dumps(_TWO_STATE), encoding="utf-8")
        return path

    def test_constructor_takes_a_path(self, fsm_path):
        api = API(fsm_definition=fsm_path, llm_interface=MockLLM2Interface())
        assert api.fsm_definition.name == "b"

    def test_process_fsm_definition_takes_a_path(self, fsm_path):
        definition, _ = API.process_fsm_definition(fsm_path)
        assert definition.initial_state == "s"

    def test_from_file_and_builder_still_take_a_path(self, fsm_path):
        assert API.from_file(fsm_path, llm_interface=MockLLM2Interface()).fsm_id
        built = (
            APIBuilder().set_definition(fsm_path).set_llm_interface(MockLLM2Interface())
        )
        assert built.build().fsm_definition.name == "b"


class TestExecuteHandlersTakesItsStateArgumentsByKeyword:
    """REVIEW-C6: ``MessagePipeline.execute_handlers`` lost its third positional
    parameter ``conversation_id``; an old positional call bound the id to
    ``current_state`` with no error."""

    def test_an_old_positional_call_raises(self):
        from fsm_llm.handlers import HandlerTiming

        api = API.from_definition(_TWO_STATE, llm_interface=MockLLM2Interface())
        cid, _ = api.start_conversation()
        pipeline = api.fsm_manager._pipeline
        instance = api.fsm_manager.instances[cid]
        with pytest.raises(TypeError):
            pipeline.execute_handlers(instance, HandlerTiming.PRE_PROCESSING, cid, "s")
        pipeline.execute_handlers(
            instance, HandlerTiming.PRE_PROCESSING, current_state="s"
        )
