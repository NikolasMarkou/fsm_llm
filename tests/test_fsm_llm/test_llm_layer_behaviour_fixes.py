"""LLM layer and classifier behaviour fixes (plan 5980fbb7, iteration 1, step 5).

Every test here fails on the parent commit 0bf6eb5.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litellm.types.utils import Message

import fsm_llm.classification as classification_module
from fsm_llm.classification import HierarchicalClassifier
from fsm_llm.definitions import (
    ClassificationSchema,
    CompletionRequest,
    FieldExtractionRequest,
    HierarchicalSchema,
    IntentDefinition,
)
from fsm_llm.llm import LiteLLMInterface
from tests.test_fsm_llm.test_llm_complete import _ADD_TOOL, _call, _complete, _reply

_PRETTY = json.dumps(
    {"intent": "buy", "items": [{"a": 1}], "note": "use } and { here"}, indent=2
)


def _recover(trace: str) -> str | None:
    """The content ``_extract_content_from_thinking`` takes from a trace."""
    return LiteLLMInterface._extract_content_from_thinking(
        Message(content="", reasoning_content=trace)
    )


class TestTraceRecoveryReadsTheLastWholeObject:
    """F-LL-01: the pretty-printed outer object, not an inner fragment."""

    def test_pretty_printed_object_with_a_nested_one_line_object(self):
        trace = (
            'thinking...\n{\n  "intent": "buy",\n  "items": [\n    {"a": 1}\n  ]\n}\n'
        )
        assert json.loads(_recover(trace) or "") == {
            "intent": "buy",
            "items": [{"a": 1}],
        }

    def test_an_early_one_line_draft_loses_to_the_later_final_object(self):
        trace = 'draft:\n{"value": "old"}\nfinal:\n{\n  "value": "new"\n}\n'
        assert json.loads(_recover(trace) or "") == {"value": "new"}

    def test_a_brace_inside_a_string_does_not_end_the_object(self):
        recovered = _recover("x\n" + _PRETTY + "\n")
        assert json.loads(recovered or "")["note"] == "use } and { here"

    def test_no_object_falls_back_to_the_last_line(self):
        assert _recover("first\nsecond\n") == "second"

    def test_a_one_line_object_is_still_recovered(self):
        assert _recover('noise\n{"message": "Hi"}\n') == '{"message": "Hi"}'


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


class _CountingInterface(LiteLLMInterface):
    built = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        type(self).built += 1
        super().__init__(*args, **kwargs)


class TestHierarchicalClassifierBuildsOneInterface:
    """F-LL-10: one model, one interface, one usage meter."""

    def _schema(self) -> HierarchicalSchema:
        def schema(*names: str) -> ClassificationSchema:
            return ClassificationSchema(
                intents=[IntentDefinition(name=n, description=n) for n in names],
                fallback_intent=names[-1],
            )

        return HierarchicalSchema(
            domain_schema=schema("shop", "help", "other"),
            intent_schemas={
                "shop": schema("buy", "browse"),
                "help": schema("refund", "faq"),
            },
        )

    def test_three_stages_share_one_constructed_interface(self):
        _CountingInterface.built = 0
        with patch.object(
            classification_module, "LiteLLMInterface", _CountingInterface
        ):
            clf = HierarchicalClassifier(self._schema(), model="gpt-4o-mini")
        assert _CountingInterface.built == 1
        stages = [clf._domain_classifier, *clf._intent_classifiers.values()]
        assert len({id(stage._llm) for stage in stages}) == 1
        assert {stage.model for stage in stages} == {"gpt-4o-mini"}

    def test_the_default_model_is_unchanged(self):
        _CountingInterface.built = 0
        with patch.object(
            classification_module, "LiteLLMInterface", _CountingInterface
        ):
            clf = HierarchicalClassifier(self._schema())
        assert _CountingInterface.built == 1
        assert clf._domain_classifier.model == clf._intent_classifiers["shop"].model


class TestBalancedJsonObjects:
    """The scanner behind the trace recovery (F-LL-01)."""

    def test_only_outermost_parsing_spans_are_returned_in_order(self):
        from fsm_llm.utilities import balanced_json_objects

        text = 'a {"x": {"y": 1}} b {"z": "}"} c'
        assert balanced_json_objects(text) == ['{"x": {"y": 1}}', '{"z": "}"}']

    def test_an_unparseable_outer_span_hides_the_objects_inside_it(self):
        from fsm_llm.utilities import balanced_json_objects

        assert balanced_json_objects('{ oops {"a": 1} }') == []

    @pytest.mark.parametrize("value", [None, 5, b"{}", ["{}"]])
    def test_a_non_str_gives_an_empty_list(self, value):
        from fsm_llm.utilities import balanced_json_objects

        assert balanced_json_objects(value) == []


class TestUnstructuredNumberRungSharesThePipelineRules:
    """The raw-text rung of field extraction reads numbers like the validator."""

    def _extract(self, text: str, field_type: str):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = text
        request = FieldExtractionRequest(
            system_prompt="extract",
            user_message="x",
            field_name="n",
            field_type=field_type,
        )
        return LiteLLMInterface(model="test-model")._parse_field_extraction_response(
            response, request
        )

    @pytest.mark.parametrize(
        ("text", "field_type", "value"),
        [("7", "int", 7), ("2.5", "float", 2.5), ("3", "float", 3.0)],
    )
    def test_plain_numbers_are_kept(self, text, field_type, value):
        out = self._extract(text, field_type)
        assert out.is_valid and out.value == value

    @pytest.mark.parametrize(
        ("text", "field_type"),
        [("nan", "float"), ("inf", "float"), ("1e999", "float"), ("3.7", "int")],
    )
    def test_non_finite_and_non_integral_text_gives_no_value(self, text, field_type):
        out = self._extract(text, field_type)
        assert out.value is None and out.is_valid is False
