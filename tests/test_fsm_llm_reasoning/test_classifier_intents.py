"""The strategy choice is a core classification with closed intents (plan baa37f40 step 5b).

Decision D-019 (owner order D-017, roadmap E3.4 slice b): ``recommend_strategy`` of the
classifier envelope declares one ``classification_extractions`` entry whose intents are the nine
strategies, so the choice is ONE structured ``complete`` call with a closed ``enum`` in place of a
free-text per-field call. Everything here is scripted: no network, no clock.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.core.verify import FSMValidator
from fsm_llm.logging import logger
from fsm_llm.reasoning import ReasoningEngine
from fsm_llm.reasoning.constants import ContextKeys, ReasoningType
from fsm_llm.reasoning.engine import _classification_from
from fsm_llm.reasoning.strategy_files import (
    STRATEGY_ORDER,
    load_document,
    load_fsm,
    strategy_facts,
)
from fsm_llm.testing import PromptGroundedLLM
from fsm_llm.text import capped_line
from tests.test_fsm_llm_reasoning import classifier_file
from tests.test_fsm_llm_reasoning.test_engine_scripted import (
    _PROBLEM,
    _VALID_SCRIPT,
    _engine,
    _ScriptedLLM,
)

REPO = Path(__file__).resolve().parents[2]
_NO_VALUE = "No recommended reasoning type was produced"


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")


def _warnings(run: Any) -> list[str]:
    """The WARNING messages the library logged while ``run()`` ran."""
    records: list[str] = []
    logger.enable("fsm_llm")
    sink = logger.add(
        lambda message: records.append(message.record["message"]), level="WARNING"
    )
    try:
        run()
    finally:
        logger.remove(sink)
        logger.disable("fsm_llm")
    return records


class TestTheFile:
    """The file is the derivation, and the derivation is the strategy facts."""

    def test_the_file_equals_its_derivation(self) -> None:
        text = classifier_file.CLASSIFIER_PATH.read_text(encoding="utf-8")
        derived = classifier_file.render(classifier_file.derive(json.loads(text)))
        assert derived == text

    def test_the_derivation_is_idempotent(self) -> None:
        once = classifier_file.derive(load_document("classifier"))
        assert classifier_file.derive(once) == once

    def test_one_entry_carries_the_nine_strategies_in_order(self) -> None:
        state = load_fsm("classifier")["states"]["recommend_strategy"]
        (entry,) = state["classification_extractions"]
        assert entry["field_name"] == ContextKeys.RECOMMENDED_REASONING_TYPE
        assert [i["name"] for i in entry["intents"]] == list(STRATEGY_ORDER)
        assert {i["name"] for i in entry["intents"]} == {t.value for t in ReasoningType}
        for intent in entry["intents"]:
            description = strategy_facts(intent["name"])["description"]
            assert intent["description"] == capped_line(description, 300)
            assert intent["description"] and "\n" not in intent["description"]
        assert entry["fallback_intent"] == "analytical"
        assert entry["confidence_threshold"] == 0.6
        assert entry["required"] is False
        assert entry["prompt_config"] == {"include_entities": False}

    def test_the_list_of_strategies_is_not_typed_twice(self) -> None:
        state = load_fsm("classifier")["states"]["recommend_strategy"]
        text = state["extraction_instructions"] + state["purpose"]
        assert "available strategies" not in text
        assert not any(f"- {name}:" in text for name in STRATEGY_ORDER)

    def test_the_definition_loads_and_validates(self) -> None:
        raw = load_fsm("classifier")
        FSMDefinition(**raw)
        report = FSMValidator(raw).validate()
        assert report.is_valid, report.errors

    def test_a_fallback_outside_the_list_is_refused_at_load(self) -> None:
        raw = load_fsm("classifier")
        entry = raw["states"]["recommend_strategy"]["classification_extractions"][0]
        entry["fallback_intent"] = "no_such_strategy"
        with pytest.raises(ValueError, match=r"(?i)fallback intent"):
            FSMDefinition(**raw)

    def test_only_the_recommend_state_classifies(self) -> None:
        states = load_fsm("classifier")["states"]
        classifying = [
            sid for sid, st in states.items() if st.get("classification_extractions")
        ]
        assert classifying == ["recommend_strategy"]

    def test_every_other_fsm_file_declares_no_classification(self) -> None:
        for name in ("orchestrator", *STRATEGY_ORDER):
            states = load_fsm(name)["states"].values()
            assert not any(st.get("classification_extractions") for st in states), name


class TestOneClassificationRequest:
    """The choice is one structured request, never a per-field one."""

    def test_a_solve_sends_one_closed_enum_request(self) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        _solution, trace_info = _engine(llm).solve_problem(_PROBLEM)

        (request,) = llm.completions
        assert request.call_type == "classification"
        assert request.tools is None
        schema = request.response_format["json_schema"]["schema"]
        assert schema["properties"]["intent"]["enum"] == list(STRATEGY_ORDER)
        system = request.messages[0]["content"]
        assert _PROBLEM in system
        assert request.messages[-1]["content"] is None
        assert trace_info["final_context"][ContextKeys.CLASSIFIED_PROBLEM_TYPE] == (
            "deductive"
        )

    def test_the_intent_prompt_carries_every_description_and_no_second_list(
        self,
    ) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        _engine(llm).solve_problem(_PROBLEM)

        system = llm.completions[0].messages[0]["content"]
        for name in STRATEGY_ORDER:
            description = strategy_facts(name)["description"]
            assert f"- {name}: {description}" in system
            assert system.count(f"- {name}:") == 1

    def test_no_per_field_or_bulk_request_asks_for_the_choice(self) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        _engine(llm).solve_problem(_PROBLEM)

        assert llm.field_requests(ContextKeys.RECOMMENDED_REASONING_TYPE) == []
        bulk = [r for r in llm.requests if type(r).__name__ == "BulkExtractionRequest"]
        assert bulk
        assert not any(
            ContextKeys.RECOMMENDED_REASONING_TYPE in r.system_prompt for r in bulk
        )

    def test_a_preferred_type_still_runs_the_one_classification(self) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        _engine(llm).solve_problem(
            _PROBLEM, {ContextKeys.PREFERRED_REASONING_TYPE: "deductive"}
        )

        # the classifier still runs: the preferred type beats its answer in
        # `_prepare_reasoning_execution`, the call count is the one of today
        assert [r.call_type for r in llm.completions] == ["classification"]


class TestEveryUnusableChoiceEndsOnAnalytical:
    @pytest.mark.parametrize(
        ("script", "warns"),
        [
            ({ContextKeys.RECOMMENDED_REASONING_TYPE: "analytical"}, False),
            ({ContextKeys.RECOMMENDED_REASONING_TYPE: "not_a_strategy"}, False),
            (
                {
                    ContextKeys.RECOMMENDED_REASONING_TYPE: "deductive",
                    "classification_confidence": 0.3,
                },
                True,
            ),
        ],
        ids=["fallback-intent", "unknown-intent", "low-confidence"],
    )
    def test_the_solve_answers_on_analytical(
        self, script: dict[str, Any], warns: bool
    ) -> None:
        llm = _ScriptedLLM({**_VALID_SCRIPT, **script})
        result: list[Any] = []
        messages = _warnings(
            lambda: result.append(_engine(llm).solve_problem(_PROBLEM))
        )

        solution, trace_info = result[0]
        context = trace_info["final_context"]
        assert solution == _VALID_SCRIPT["final_solution"]
        assert context[ContextKeys.CLASSIFIED_PROBLEM_TYPE] == "analytical"
        assert context[ContextKeys.REASONING_TYPE_SELECTED] == "analytical"
        assert len(llm.completions) == 1
        assert len([m for m in messages if m.startswith(_NO_VALUE)]) == int(warns)

    def test_a_provider_outage_ends_on_analytical_with_the_warning(self) -> None:
        llm = _ScriptedLLM(
            {**_VALID_SCRIPT, ContextKeys.RECOMMENDED_REASONING_TYPE: None}
        )
        result: list[Any] = []
        messages = _warnings(
            lambda: result.append(_engine(llm).solve_problem(_PROBLEM))
        )

        solution, trace_info = result[0]
        assert solution == _VALID_SCRIPT["final_solution"]
        context = trace_info["final_context"]
        assert context[ContextKeys.CLASSIFIED_PROBLEM_TYPE] == "analytical"
        assert len(llm.completions) == 1
        assert len([m for m in messages if m.startswith(_NO_VALUE)]) == 1

    def test_a_bug_in_the_hosts_complete_is_a_non_fatal_post_transition_failure(
        self,
    ) -> None:
        """The classifier runs as the post-transition extraction of its last state, and
        core logs any failure there as non-fatal (D-012 of plan 80b0bd4d keeps only a
        handler failure fatal): a bug in a host's ``complete`` ends on analytical with
        two WARNINGs, as a bug in its ``extract_field`` did before the intents. The
        explorer's map said the handler would be critical; the code path says not."""

        class _Broken(_ScriptedLLM):
            def complete(self, request: Any) -> Any:
                raise KeyError("host bug")

        engine_ = _engine(_Broken(_VALID_SCRIPT))
        result: list[Any] = []
        messages = _warnings(lambda: result.append(engine_.solve_problem(_PROBLEM)))

        context = result[0][1]["final_context"]
        assert context[ContextKeys.CLASSIFIED_PROBLEM_TYPE] == "analytical"
        assert [m for m in messages if "Post-transition extraction failed" in m] == [
            "Post-transition extraction failed (non-fatal): 'host bug'"
        ]
        assert len([m for m in messages if m.startswith(_NO_VALUE)]) == 1
        assert engine_.classifier.list_active_conversations() == []

    def test_the_warning_is_one_line_per_missing_value(self) -> None:
        result = {ContextKeys.STRATEGY_JUSTIFICATION: "j"}
        messages = _warnings(lambda: _classification_from(result))
        assert len(messages) == 1
        assert messages[0].startswith(_NO_VALUE)

    def test_a_present_value_logs_nothing(self) -> None:
        result = {ContextKeys.RECOMMENDED_REASONING_TYPE: "deductive"}
        assert _warnings(lambda: _classification_from(result)) == []


class TestTheInterfaceRule:
    """An injected interface must implement ``complete`` (a BREAKING change)."""

    def test_an_interface_without_complete_is_refused_at_construction(self) -> None:
        from fsm_llm.llm import LLMInterface

        class _SyncOnly(LLMInterface):
            def generate_response(self, request: Any) -> Any:  # pragma: no cover
                raise AssertionError

            def extract_field(self, request: Any) -> Any:  # pragma: no cover
                raise AssertionError

            def extract_bulk_data(self, request: Any) -> Any:  # pragma: no cover
                raise AssertionError

        with pytest.raises(ValueError, match="recommend_strategy"):
            ReasoningEngine(model="mock", llm_interface=_SyncOnly())

    def test_the_public_prompt_grounded_double_is_accepted(self) -> None:
        built = ReasoningEngine(model="mock", llm_interface=PromptGroundedLLM())
        assert built.classifier is not None

    def test_a_scripted_interface_with_complete_is_accepted(self) -> None:
        assert _engine(_ScriptedLLM({})).classifier is not None


def test_the_derivation_does_not_alias_the_loaded_document() -> None:
    document = load_document("classifier")
    before = copy.deepcopy(document)
    classifier_file.derive(document)
    assert document == before
