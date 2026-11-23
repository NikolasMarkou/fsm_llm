"""The verifier hooks of the call kind (plan baa37f40, step 8 = K1.2, roadmap E3.1).

G3 sees a call state's inputs and its choice key as READS on arrival and its `returns` parent keys
as WRITES; G4 needs no new rule; the validator knows the result key of a call state and lints a
choice key nothing writes, as a WARNING (never stricter than the loader). The loader rules of the
kind are `test_call_state`; the shared edge rule is `test_kind_call`.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.core.state_kinds import CALL, COMPLETION, CONVERSATION, SYNTHESIZE
from fsm_llm.core.verify import FSMValidator
from fsm_llm.core.verify.flow import (
    analyze_data_flow,
    analyze_liveness,
    declared_available,
)

_RESULT = "call_result"


def _gate_edges(
    key: str = _RESULT, ok: str = "done", failed: str = "sorry"
) -> list[dict[str, Any]]:
    return [
        {
            "target_state": ok,
            "description": "ok",
            "priority": 1,
            "conditions": [
                {
                    "description": "ok",
                    "requires_context_keys": [key],
                    "logic": {"==": [{"var": f"{key}.status"}, "ok"]},
                }
            ],
        },
        {
            "target_state": failed,
            "description": "failed or aborted",
            "priority": 2,
            "conditions": [
                {
                    "description": "not ok",
                    "requires_context_keys": [key],
                    "logic": {"in": [{"var": f"{key}.status"}, ["failed", "aborted"]]},
                }
            ],
        },
    ]


def _call(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "target": "strategy.alpha",
        "inputs": ["topic"],
        "returns": {"answer": "solution"},
        "on_failure": "sorry",
    }
    config.update(overrides)
    return config


def _definition(
    call: dict[str, Any] | None = None,
    *,
    ask: bool = True,
    handler_only: tuple[str, ...] = (),
    after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """start (asks for `topic` when `ask`) -> work (a call state) -> done | sorry."""
    start: dict[str, Any] = {
        "id": "start",
        "description": "ask",
        "purpose": "ask",
        "response_instructions": "Ask for the topic.",
        "transitions": [{"target_state": "work", "description": "go"}],
    }
    if ask:
        start["field_extractions"] = [
            {"field_name": "topic", "field_type": "str", "extraction_instructions": "t"}
        ]
    data: dict[str, Any] = {
        "name": "CallFlow",
        "description": "d",
        "initial_state": "start",
        "states": {
            "start": start,
            "work": {
                "id": "work",
                "description": "call",
                "purpose": "call",
                "call": _call() if call is None else call,
                "transitions": _gate_edges(),
            },
            "done": {
                "id": "done",
                "description": "done",
                "purpose": "done",
                "response_instructions": "Done.",
            },
            "sorry": {
                "id": "sorry",
                "description": "sorry",
                "purpose": "sorry",
                "response_instructions": "Sorry.",
            },
        },
    }
    if handler_only:
        data["handler_only_keys"] = list(handler_only)
    if after is not None:
        data["states"]["done"] = after
    return data


def _load(data: dict[str, Any]) -> FSMDefinition:
    return FSMDefinition(**copy.deepcopy(data))


def _codes(findings) -> list[tuple[str, str]]:
    return [(f.code, f.state) for f in findings]


class TestTheReadHook:
    def test_a_conversation_and_a_completion_state_read_no_declared_input(self):
        for spec in (CONVERSATION, COMPLETION):
            assert spec.read_keys(object()) == ()

    def test_a_call_state_reads_its_inputs_then_its_choice_key(self):
        config = _call(
            choose={"key": "picked", "targets": ["a.b", "c.d"]}, inputs=["x", "y"]
        )
        del config["target"]
        state = _load(_definition(config)).states["work"]
        assert CALL.read_keys(state) == ("x", "y", "picked")

    def test_a_target_call_reads_only_its_inputs(self):
        state = _load(_definition()).states["work"]
        assert CALL.read_keys(state) == ("topic",)

    def test_a_synthesize_state_reads_no_declared_input_yet(self):
        """D-016: G3 does not judge a synthesize state's inputs (it would move five
        recorded `data_flow` findings and the gates of every compiled procedure); the
        rule is carried to the step that moves synthesize onto the call executor."""
        state = _load(
            {
                "name": "S",
                "description": "d",
                "initial_state": "s",
                "states": {
                    "s": {
                        "id": "s",
                        "description": "d",
                        "purpose": "p",
                        "synthesize": {
                            "goal": "g",
                            "inputs": ["a", "b"],
                            "return_keys": ["r"],
                            "on_failure": "e",
                        },
                        "transitions": _gate_edges("synthesis_result", "e", "e"),
                    },
                    "e": {"id": "e", "description": "d", "purpose": "p"},
                },
            }
        )
        assert SYNTHESIZE.read_keys(state.states["s"]) == ()
        assert [
            f for f in analyze_data_flow(state) if f.code == "undeclared-input"
        ] == []


class TestAnInputNoStateWritesIsUndeclared:
    def test_an_input_nothing_writes_is_found_at_the_call_state(self):
        findings = analyze_data_flow(_load(_definition(ask=False)))
        assert ("undeclared-input", "work") in _codes(findings)
        message = next(f.message for f in findings if f.code == "undeclared-input")
        assert "'work'" in message and "'topic'" in message
        assert all(f.analysis == "data-flow" for f in findings)

    def test_a_driver_input_declares_it(self):
        findings = analyze_data_flow(_load(_definition(ask=False)), inputs=["topic"])
        assert [f for f in findings if f.code == "undeclared-input"] == []

    def test_an_earlier_extraction_declares_it(self):
        findings = analyze_data_flow(_load(_definition(ask=True)))
        assert [f for f in findings if f.code == "undeclared-input"] == []

    def test_a_key_written_on_one_path_only_is_not_declared_on_every_path(self):
        data = _definition(ask=False)
        data["states"]["start"]["transitions"] = [
            {"target_state": "left", "description": "l", "priority": 1},
            {"target_state": "work", "description": "w", "priority": 2},
        ]
        data["states"]["left"] = {
            "id": "left",
            "description": "l",
            "purpose": "l",
            "response_instructions": "x",
            "field_extractions": [
                {
                    "field_name": "topic",
                    "field_type": "str",
                    "extraction_instructions": "t",
                }
            ],
            "transitions": [{"target_state": "work", "description": "w"}],
        }
        findings = analyze_data_flow(_load(data))
        assert ("undeclared-input", "work") in _codes(findings)

    def test_the_call_states_own_result_is_not_an_arrival_key(self):
        """Arrival keys, not `declared_available` (which holds the state's own production)."""
        data = _definition(_call(inputs=[]), ask=False)
        data["states"]["work"]["call"]["inputs"] = []
        definition = _load(data)
        assert _RESULT in declared_available(definition)["work"]
        assert [
            f for f in analyze_data_flow(definition) if f.code == "undeclared-input"
        ] == []

    def test_each_missing_input_is_one_finding_in_sorted_order(self):
        data = _definition(_call(inputs=["zeta", "alpha", "alpha"]), ask=False)
        findings = [
            f for f in analyze_data_flow(_load(data)) if f.code == "undeclared-input"
        ]
        assert [("'alpha'" in f.message, "'zeta'" in f.message) for f in findings] == [
            (True, False),
            (False, True),
        ]


class TestTheChoiceKeyIsARead:
    def _choose(self, **extra: Any) -> dict[str, Any]:
        config = _call(
            choose={"key": "picked", "targets": ["strategy.alpha", "strategy.beta"]},
            **extra,
        )
        del config["target"]
        return config

    def test_an_unwritten_choice_key_is_undeclared(self):
        findings = analyze_data_flow(_load(_definition(self._choose())))
        assert any(
            f.code == "undeclared-input"
            and f.state == "work"
            and "'picked'" in f.message
            for f in findings
        )

    def test_a_handler_only_choice_key_unproduced_is_undeclared_until_listed_as_an_input(
        self,
    ):
        definition = _load(_definition(self._choose(), handler_only=("picked",)))
        assert any(
            f.code == "undeclared-input" and "'picked'" in f.message
            for f in analyze_data_flow(definition)
        )
        quiet = analyze_data_flow(definition, inputs=["picked"])
        assert [f for f in quiet if f.code == "undeclared-input"] == []

    def test_a_key_an_earlier_state_extracts_is_declared(self):
        data = _definition(self._choose())
        data["states"]["start"]["field_extractions"].append(
            {
                "field_name": "picked",
                "field_type": "str",
                "extraction_instructions": "which one",
            }
        )
        findings = analyze_data_flow(_load(data))
        assert [f for f in findings if f.code == "undeclared-input"] == []


class TestReturnsAreWrites:
    def test_a_returned_key_is_produced_by_the_call_state(self):
        definition = _load(_definition())
        assert "answer" in declared_available(definition)["work"]
        assert "answer" not in declared_available(definition)["start"]

    def test_a_return_key_counts_for_the_terminal(self):
        findings = analyze_data_flow(_load(_definition()), return_keys=["answer"])
        assert [f for f in findings if f.code == "return-key-undeclared"] == []

    def test_a_key_no_call_returns_is_still_undeclared_for_the_terminal(self):
        findings = analyze_data_flow(_load(_definition()), return_keys=["nothing"])
        assert ("return-key-undeclared", "done") in _codes(findings)

    def test_a_later_gate_reading_a_returned_key_is_declared(self):
        after = {
            "id": "done",
            "description": "done",
            "purpose": "done",
            "response_instructions": "Done.",
            "transitions": [
                {
                    "target_state": "end",
                    "description": "has answer",
                    "conditions": [
                        {
                            "description": "answer present",
                            "requires_context_keys": ["answer"],
                            "logic": {"var": "answer"},
                        }
                    ],
                }
            ],
        }
        data = _definition(after=after)
        data["states"]["end"] = {"id": "end", "description": "e", "purpose": "e"}
        findings = analyze_data_flow(_load(data))
        assert [f for f in findings if f.code == "undeclared-read"] == []

    def test_g4_needs_nothing(self):
        assert analyze_liveness(_load(_definition())) == []


class TestTheValidatorStaysNeverStricter:
    def test_a_loader_valid_call_definition_validates_with_at_most_warnings(self):
        for data in (_definition(), _definition(ask=False)):
            _load(data)
            result = FSMValidator(copy.deepcopy(data)).validate()
            assert result.is_valid, result.errors
            assert result.errors == []

    def test_an_extraction_naming_the_result_key_warns_with_the_call_writer(self):
        data = _definition()
        data["states"]["start"]["field_extractions"].append(
            {
                "field_name": _RESULT,
                "field_type": "str",
                "extraction_instructions": "x",
            }
        )
        result = FSMValidator(data).validate()
        assert any(
            f"names '{_RESULT}', a key only handlers or the call executor write" in w
            for w in result.warnings
        ), result.warnings

    def test_a_custom_result_key_is_the_one_the_validator_reads(self):
        data = _definition(_call(result_key="outcome"))
        data["states"]["work"]["transitions"] = _gate_edges("outcome")
        data["states"]["start"]["field_extractions"].append(
            {
                "field_name": "outcome",
                "field_type": "str",
                "extraction_instructions": "x",
            }
        )
        result = FSMValidator(data).validate()
        assert any(
            "names 'outcome', a key only handlers or the call executor" in w
            for w in result.warnings
        )

    def test_a_choice_key_nothing_can_write_is_a_warning_not_an_error(self):
        config = _call(choose={"key": "picked", "targets": ["a.b", "c.d"]})
        del config["target"]
        data = _definition(config)
        result = FSMValidator(data).validate()
        assert result.is_valid
        warnings = [w for w in result.warnings if "chooses its target by 'picked'" in w]
        assert len(warnings) == 1
        assert "'work'" in warnings[0]

    @pytest.mark.parametrize("how", ["handler_only", "extraction"])
    def test_a_choice_key_somebody_writes_is_not_warned(self, how):
        config = _call(choose={"key": "picked", "targets": ["a.b", "c.d"]})
        del config["target"]
        data = _definition(
            config, handler_only=("picked",) if how == "handler_only" else ()
        )
        if how == "extraction":
            data["states"]["start"]["field_extractions"].append(
                {
                    "field_name": "picked",
                    "field_type": "str",
                    "extraction_instructions": "x",
                }
            )
        result = FSMValidator(data).validate()
        assert [w for w in result.warnings if "chooses its target by" in w] == []

    @pytest.mark.parametrize(
        "bad",
        [None, 5, [], "", {"key": 5}, {"key": ""}, {"targets": []}, {}],
    )
    def test_a_malformed_raw_choose_never_raises_the_lint(self, bad):
        data = _definition()
        data["states"]["work"]["call"] = {"choose": bad, "on_failure": "sorry"}
        result = FSMValidator(data).validate()
        assert not result.is_valid  # the loader refuses it, the lint stays quiet
        assert [w for w in result.warnings if "chooses its target by" in w] == []

    def test_a_non_string_result_key_never_raises_the_validator(self):
        data = _definition()
        data["states"]["work"]["call"]["result_key"] = ["call_result"]
        result = FSMValidator(data).validate()
        assert not result.is_valid
