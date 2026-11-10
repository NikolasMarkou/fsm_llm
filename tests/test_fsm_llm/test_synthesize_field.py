"""
Tests for the optional ``State.synthesize`` field (``SynthesizeConfig``).

Covers the load and refusal rules, the pipeline mint change (a synthesize state
mints no extraction and its result key is handler-only), the unchanged
``fsm_id`` of an existing FSM, the validator's known-key check, and a spike of
the push-from-``before_step`` seam on the real ``API`` (no synthesis code: a
local hook and a hand-written child).
"""

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from fsm_llm import API
from fsm_llm.api import ContextMergeStrategy
from fsm_llm.definitions import FSMDefinition, State, SynthesizeConfig
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.validator import FSMValidator
from tests.conftest import MockLLM2Interface

# fsm_id of ``_golden()``, captured on the parent commit before the field existed.
GOLDEN_FSM_ID = "fsm_Golden_043181d0"


def _golden() -> dict[str, Any]:
    return {
        "name": "Golden",
        "description": "d",
        "initial_state": "a",
        "states": {
            "a": {
                "id": "a",
                "description": "x",
                "purpose": "p",
                "response_instructions": "hi",
                "transitions": [{"target_state": "b", "description": "go"}],
            },
            "b": {"id": "b", "description": "y", "purpose": "q"},
        },
    }


def _synth(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "goal": "Book a table for two",
        "inputs": ["topic"],
        "return_keys": ["answer"],
        "on_failure": "apology",
    }
    config.update(overrides)
    return config


def _synth_state(
    config: dict[str, Any] | None = None, **state_overrides: Any
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "id": "solve",
        "description": "Obtain and run a procedure",
        "purpose": "Run the synthesized procedure",
        "synthesize": _synth() if config is None else config,
        "transitions": [
            {
                "target_state": "done",
                "description": "procedure ok",
                "priority": 1,
                "conditions": [
                    {
                        "description": "status ok",
                        "requires_context_keys": ["synthesis_result"],
                        "logic": {"==": [{"var": "synthesis_result.status"}, "ok"]},
                    }
                ],
            },
            {
                "target_state": "apology",
                "description": "procedure failed",
                "priority": 2,
                "conditions": [
                    {
                        "description": "status failed",
                        "requires_context_keys": ["synthesis_result"],
                        "logic": {"==": [{"var": "synthesis_result.status"}, "failed"]},
                    }
                ],
            },
        ],
    }
    state.update(state_overrides)
    return state


def _parent(config: dict[str, Any] | None = None, **state_overrides: Any):
    return {
        "name": "Concierge",
        "description": "Host conversation with one synthesize state",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "greet",
                "purpose": "greet the user",
                "response_instructions": "Say hello",
                "transitions": [{"target_state": "solve", "description": "go"}],
            },
            "solve": _synth_state(config, **state_overrides),
            "done": {
                "id": "done",
                "description": "finished",
                "purpose": "wrap up",
                "response_instructions": "Say goodbye",
            },
            "apology": {
                "id": "apology",
                "description": "failed",
                "purpose": "apologise",
                "response_instructions": "Apologise",
            },
        },
    }


def _bad_values() -> list[Any]:
    return [None, 5, 1.5, True, "", "   ", [], {}, "x" * 100_000]


class TestSynthesizeLoads:
    def test_loads_and_defaults(self):
        state = FSMDefinition(**_parent()).states["solve"]
        config = state.synthesize
        assert isinstance(config, SynthesizeConfig)
        assert config.goal == "Book a table for two"
        assert config.inputs == ("topic",)
        assert config.return_keys == ("answer",)
        assert config.on_failure == "apology"
        assert config.result_key == "synthesis_result"
        assert config.min_lifecycle == "verified"

    def test_absent_by_default(self):
        assert FSMDefinition(**_golden()).states["a"].synthesize is None

    def test_inputs_may_be_empty_and_duplicates_are_dropped(self):
        config = SynthesizeConfig(**_synth(inputs=[], return_keys=["a", "a", "b"]))
        assert config.inputs == ()
        assert config.return_keys == ("a", "b")

    def test_custom_result_key_and_trusted(self):
        config = SynthesizeConfig(**_synth(result_key="proc", min_lifecycle="trusted"))
        assert (config.result_key, config.min_lifecycle) == ("proc", "trusted")

    def test_goal_length_bounds(self):
        assert SynthesizeConfig(**_synth(goal="g" * 2000)).goal == "g" * 2000
        with pytest.raises(ValidationError):
            SynthesizeConfig(**_synth(goal="g" * 2001))

    def test_frozen_and_extra_forbidden(self):
        config = SynthesizeConfig(**_synth())
        with pytest.raises(ValidationError):
            config.goal = "other"
        with pytest.raises(ValidationError):
            SynthesizeConfig(**_synth(persona="a pirate"))


class TestSynthesizeRefusals:
    @pytest.mark.parametrize("field", ["goal", "on_failure", "result_key"])
    @pytest.mark.parametrize("bad", _bad_values(), ids=repr)
    def test_bad_scalar_values(self, field, bad):
        with pytest.raises(ValidationError):
            SynthesizeConfig(**_synth(**{field: bad}))

    @pytest.mark.parametrize("bad", [None, 5, 1.5, True, "", "abc", {}, {"a": 1}])
    @pytest.mark.parametrize("field", ["inputs", "return_keys"])
    def test_bad_key_collections(self, field, bad):
        with pytest.raises(ValidationError):
            SynthesizeConfig(**_synth(**{field: bad}))

    @pytest.mark.parametrize(
        "bad_name",
        [
            *[None, 5, "", "   ", "x" * 100_000, ["a"], {"a": 1}],
            *["_hidden", "system_x", "internal_x", "__dunder", "_conversation_id"],
            *["agent_trace", "password", "api_key", "session_token"],
        ],
    )
    @pytest.mark.parametrize("field", ["inputs", "return_keys"])
    def test_bad_key_names(self, field, bad_name):
        with pytest.raises(ValidationError):
            SynthesizeConfig(**_synth(**{field: ["fine", bad_name]}))

    def test_return_keys_need_one(self):
        with pytest.raises(ValidationError):
            SynthesizeConfig(**_synth(return_keys=[]))

    @pytest.mark.parametrize("bad", [None, 5, "", "[]", "VERIFIED", "draft", "any"])
    def test_min_lifecycle_vocabulary(self, bad):
        with pytest.raises(ValidationError):
            SynthesizeConfig(**_synth(min_lifecycle=bad))

    @pytest.mark.parametrize(
        "bad", ["_private", "system_x", "agent_trace", "_conversation_id", "  "]
    )
    def test_result_key_is_public_and_not_reserved(self, bad):
        with pytest.raises(ValidationError):
            SynthesizeConfig(**_synth(result_key=bad))

    @pytest.mark.parametrize("field", ["inputs", "return_keys"])
    def test_result_key_cannot_be_a_data_key(self, field):
        with pytest.raises(ValidationError):
            SynthesizeConfig(**_synth(**{field: ["synthesis_result"]}))


class TestSynthesizeStateRules:
    @pytest.mark.parametrize(
        "conflict",
        [
            {"required_context_keys": ["k"]},
            {"extraction_instructions": "extract something"},
            {
                "field_extractions": [
                    {"field_name": "f", "extraction_instructions": "get f"}
                ]
            },
            {
                "classification_extractions": [
                    {
                        "field_name": "c",
                        "intents": [
                            {"name": "a", "description": "A"},
                            {"name": "b", "description": "B"},
                        ],
                        "fallback_intent": "a",
                    }
                ]
            },
            {
                "completion": {
                    "response_format_key": "_fmt",
                }
            },
        ],
        ids=lambda c: next(iter(c)),
    )
    def test_excluded_fields(self, conflict):
        with pytest.raises(ValidationError, match="synthesize state cannot also"):
            FSMDefinition(**_parent(**conflict))

    def test_needs_transitions(self):
        with pytest.raises(ValidationError, match="transition targets"):
            State(**_synth_state(transitions=[]))

    def test_on_failure_must_be_a_transition_target(self):
        with pytest.raises(ValidationError, match="on_failure"):
            FSMDefinition(**_parent(_synth(on_failure="done_elsewhere")))

    def test_on_failure_may_be_any_transition_target(self):
        definition = FSMDefinition(**_parent(_synth(on_failure="done")))
        assert definition.states["solve"].synthesize.on_failure == "done"

    def test_existing_fsm_id_is_byte_identical(self):
        assert API.process_fsm_definition(_golden())[1] == GOLDEN_FSM_ID

    def test_a_synthesize_state_changes_only_its_own_fsm_id(self):
        _, with_field = API.process_fsm_definition(_parent())
        without = _parent()
        del without["states"]["solve"]["synthesize"]
        assert with_field != API.process_fsm_definition(without)[1]


class TestValidatorKnowsTheField:
    def test_no_unknown_key_warning(self):
        result = FSMValidator(_parent()).validate()
        assert result.is_valid, result.errors
        assert not [w for w in result.warnings if "synthesize" in w]

    def test_refusal_is_a_validator_error(self):
        result = FSMValidator(_parent(_synth(on_failure="nowhere"))).validate()
        assert not result.is_valid


class TestPipelineMintsNothing:
    def test_no_field_configs(self):
        state = FSMDefinition(**_parent()).states["solve"]
        assert MessagePipeline._build_field_configs_from_state(state) == []

    @pytest.mark.parametrize("result_key", ["synthesis_result", "proc_status"])
    def test_a_mock_cannot_forge_the_result(self, result_key):
        definition = _parent(_synth(result_key=result_key))
        for transition in definition["states"]["solve"]["transitions"]:
            condition = transition["conditions"][0]
            condition["requires_context_keys"] = [result_key]
            condition["logic"]["=="][0]["var"] = f"{result_key}.status"
        llm = MockLLM2Interface(
            extraction_data={result_key: {"status": "ok"}, "answer": "forged"}
        )
        api = API.from_definition(definition=definition, llm_interface=llm)
        conv_id, _ = api.start_conversation()
        api.converse("book it", conv_id)
        # The bare converse reached `solve` and stays there (BLOCKED, by design).
        assert api.get_current_state(conv_id) == "solve"
        api.converse("book it now", conv_id)
        assert api.get_current_state(conv_id) == "solve"
        data = api.get_data(conv_id)
        assert result_key not in data and "answer" not in data
        assert [call for call, _ in llm.call_history if call == "extract_field"] == []


_CHILD: dict[str, Any] = {
    "name": "Child",
    "description": "Hand-written procedure",
    "initial_state": "work",
    "states": {
        "work": {
            "id": "work",
            "description": "do the work",
            "purpose": "produce the answer",
            "required_context_keys": ["answer"],
            "extraction_instructions": "Extract the answer",
            "response_instructions": "Working",
            "transitions": [
                {
                    "target_state": "finished",
                    "description": "answer ready",
                    "conditions": [
                        {
                            "description": "has answer",
                            "requires_context_keys": ["answer"],
                        }
                    ],
                }
            ],
        },
        "finished": {
            "id": "finished",
            "description": "end",
            "purpose": "end of the child",
            "response_instructions": "Done",
        },
    },
}


class _SpikeHook:
    """A local ``before_step`` hook: push a child on a synthesize state, pop it."""

    def __init__(self, api: API, conv_id: str) -> None:
        self.api, self.conv_id = api, conv_id
        self.obtained = 0
        self.popped = 0
        self.child_data: dict[str, Any] = {}
        self.depths: list[int] = []

    def __call__(self, step: int) -> None:
        api, conv_id = self.api, self.conv_id
        self.depths.append(api.get_stack_depth(conv_id))
        if api.get_stack_depth(conv_id) > 1:
            if api.has_conversation_ended(conv_id):
                self.child_data = api.get_data(conv_id)
                api.pop_fsm(
                    conv_id,
                    context_to_return={"answer": self.child_data["answer"]},
                    merge_strategy=ContextMergeStrategy.UPDATE,
                )
                api.update_context(conv_id, {"synthesis_result": {"status": "ok"}})
                self.popped += 1
            return
        if api.get_current_state(
            conv_id
        ) == "solve" and "synthesis_result" not in api.get_data(conv_id):
            self.obtained += 1
            api.update_context(conv_id, {"synthesis_result": {"status": "running"}})
            api.push_fsm(
                conv_id,
                copy.deepcopy(_CHILD),
                inherit_context=False,
                context_to_pass={"topic": "tables"},
            )


class TestHookPushSpike:
    def test_push_run_pop_from_before_step(self):
        llm = MockLLM2Interface(extraction_data={"answer": "42"})
        api = API.from_definition(definition=_parent(), llm_interface=llm)
        conv_id, _ = api.start_conversation(
            {"noise": "parent only", "db_password": "hunter2"}
        )
        hook = _SpikeHook(api, conv_id)

        results = api.run_until_terminal(conv_id, max_steps=20, before_step=hook)

        assert results[-1].ended
        assert api.get_current_state(conv_id) == "done"
        assert api.get_stack_depth(conv_id) == 1
        data = api.get_data(conv_id)
        assert data["answer"] == "42"
        assert data["synthesis_result"] == {"status": "ok"}
        assert (hook.obtained, hook.popped) == (1, 1)
        assert max(hook.depths) == 2
        # inherit_context=False: only the passed input reached the child.
        assert hook.child_data["topic"] == "tables"
        assert "noise" not in hook.child_data
        assert "db_password" not in hook.child_data
