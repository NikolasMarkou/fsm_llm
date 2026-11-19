"""
Tests for the classification rule of the compile prompt (plan 51fde95b D-027,
part 4a, D-030): ``policy.allow_classification`` decides whether the prompt
names ``classification_extractions``; G5 reads nothing of it.
"""

import pytest

from fsm_llm.synthesis import SynthesisError, SynthesisPolicy, run_gates
from fsm_llm.synthesis.policy import _CLASSIFICATION_KEYS, _MODEL_SETTING_KEYS
from fsm_llm.synthesis.prompts import (
    CLASSIFICATION_RULE,
    compile_messages,
    system_prompt,
)

OFF, ON = SynthesisPolicy(), SynthesisPolicy(allow_classification=True)


def _routing_fsm() -> dict:
    """The shape the rule describes: classify, then route on the intent name."""
    ask = {
        "id": "ask",
        "description": "ask",
        "purpose": "ask",
        "response_instructions": "Ask what the user wants.",
        "classification_extractions": [
            {
                "field_name": "intent",
                "intents": [
                    {"name": "buy", "description": "wants to buy"},
                    {"name": "browse", "description": "is looking"},
                ],
                "fallback_intent": "browse",
                "confidence_threshold": 0.6,
            }
        ],
        "transitions": [
            {
                "target_state": "done",
                "description": "an intent is known",
                "priority": 1,
                "conditions": [
                    {
                        "description": "an intent is known",
                        "requires_context_keys": ["intent"],
                        "logic": {"in": [{"var": "intent"}, ["buy", "browse"]]},
                    }
                ],
            }
        ],
    }
    done = {
        "id": "done",
        "description": "d",
        "purpose": "d",
        "response_instructions": "",
    }
    return {
        "name": "Route",
        "description": "route",
        "initial_state": "ask",
        "states": {"ask": ask, "done": done},
    }


class TestThePromptRule:
    def test_off_by_default_and_the_text_is_the_same_as_before(self):
        assert SynthesisPolicy().allow_classification is False
        assert "classification" not in system_prompt(OFF).lower()

    def test_on_adds_exactly_the_one_rule_to_the_list(self):
        on, off = system_prompt(ON), system_prompt(OFF)
        assert CLASSIFICATION_RULE in on and CLASSIFICATION_RULE not in off
        assert on.replace("\n" + CLASSIFICATION_RULE, "") == off

    def test_a_compile_request_carries_it_in_the_system_message_only(self):
        sent = compile_messages("book a table", ["venue"], ["answer"], ON)
        assert CLASSIFICATION_RULE in sent[0]["content"]
        assert CLASSIFICATION_RULE not in sent[1]["content"]

    def test_the_rule_names_keys_g5_accepts_and_refuses_the_model_settings(self):
        keys = {"field_name", "intents", "fallback_intent", "confidence_threshold"}
        assert keys <= _CLASSIFICATION_KEYS
        assert all(k in CLASSIFICATION_RULE for k in keys)
        assert _MODEL_SETTING_KEYS == {"model", "prompt_config"}
        assert "Never set model or prompt_config" in CLASSIFICATION_RULE

    @pytest.mark.parametrize("value", [1, "yes", None])
    def test_the_flag_must_be_a_bool(self, value):
        with pytest.raises(SynthesisError, match="allow_classification"):
            SynthesisPolicy(allow_classification=value)


class TestG5ReadsNothingOfIt:
    @pytest.mark.parametrize("policy", [OFF, ON], ids=["off", "on"])
    def test_the_described_shape_passes_under_either_setting(self, policy):
        report = run_gates(_routing_fsm(), policy, return_keys=("intent",))
        assert report.passed, report.repair_text()

    @pytest.mark.parametrize("policy", [OFF, ON], ids=["off", "on"])
    def test_a_model_setting_is_refused_under_either_setting(self, policy):
        fsm = _routing_fsm()
        fsm["states"]["ask"]["classification_extractions"][0]["model"] = "other-model"
        report = run_gates(fsm, policy, return_keys=("intent",))
        assert not report.passed
        assert "model-setting" in {f.code for f in report.findings}
