"""The optional ``State.call`` field (``CallStateConfig``, ``CallChoose``): the load rules.

Plan baa37f40, step 7 (K1.1, D-015, roadmap E3.1). Core only LOADS a call state: its models, the
State rules (no extraction, an edge for the failure), the loader's cross-state edge rule (the one
body of ``kind_call.check_result_edges``) and the version tier. The executor is step 11 and the
verifier hooks are step 8. Offline: no model, no network.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from fsm_llm import API
from fsm_llm.core.definitions import (
    DEFAULT_CALL_RESULT_KEY,
    CallChoose,
    CallStateConfig,
    FSMDefinition,
    State,
)
from fsm_llm.core.definitions.schema import SCHEMA_VERSION_DEFAULT
from fsm_llm.core.state_kinds import KINDS, get_kind, kind_of
from tests.conftest import compact_param_id

_STATUS = {"var": "call_result.status"}


def _call(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "target": "reasoning.analytical",
        "inputs": ["topic"],
        "returns": {"answer": "solution"},
        "on_failure": "apology",
    }
    config.update(overrides)
    return config


def _choose(**overrides: Any) -> dict[str, Any]:
    choose: dict[str, Any] = {
        "key": "picked",
        "targets": ["reasoning.analytical", "reasoning.deductive"],
    }
    choose.update(overrides)
    return choose


def _edge(
    target: str, priority: int, logic: Any, requires: tuple[str, ...] = ("call_result",)
) -> dict[str, Any]:
    return {
        "target_state": target,
        "description": f"to {target}",
        "priority": priority,
        "conditions": [
            {
                "description": "gate",
                "requires_context_keys": list(requires),
                "logic": logic,
            }
        ],
    }


def _ok_edge() -> dict[str, Any]:
    return _edge("done", 1, {"==": [_STATUS, "ok"]})


def _failure_edge() -> dict[str, Any]:
    return _edge("apology", 2, {"in": [_STATUS, ["failed", "aborted"]]})


def _call_state(
    config: dict[str, Any] | None = None, **state_overrides: Any
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "id": "work",
        "description": "Call a target",
        "purpose": "Run the target",
        "call": _call() if config is None else config,
        "transitions": [_ok_edge(), _failure_edge()],
    }
    state.update(state_overrides)
    return state


def _parent(
    config: dict[str, Any] | None = None,
    *,
    handler_only: tuple[str, ...] = (),
    extra_states: dict[str, Any] | None = None,
    **state_overrides: Any,
) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "name": "Caller",
        "description": "Host conversation with one call state",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "greet",
                "purpose": "greet the user",
                "response_instructions": "Say hello",
                "transitions": [{"target_state": "work", "description": "go"}],
            },
            "work": _call_state(config, **state_overrides),
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
            **(extra_states or {}),
        },
    }
    for extra_id in extra_states or {}:  # reachable, or the loader refuses an orphan
        definition["states"]["start"]["transitions"].append(
            {"target_state": extra_id, "description": f"to {extra_id}"}
        )
    if handler_only:
        definition["handler_only_keys"] = list(handler_only)
    return definition


def _refusal(definition: dict[str, Any]) -> str:
    with pytest.raises(ValidationError) as caught:
        FSMDefinition(**definition)
    return str(caught.value)


class TestCallLoads:
    def test_the_target_form_loads_with_its_defaults(self):
        state = FSMDefinition(**_parent()).states["work"]
        config = state.call
        assert isinstance(config, CallStateConfig)
        assert config.target == "reasoning.analytical"
        assert config.choose is None
        assert config.inputs == ("topic",)
        assert config.returns == {"answer": "solution"}
        assert config.on_failure == "apology"
        assert config.result_key == DEFAULT_CALL_RESULT_KEY == "call_result"
        assert (config.max_steps, config.max_seconds) == (None, None)

    def test_the_choose_form_loads(self):
        config = _call(target=None, choose=_choose(fallback="reasoning.deductive"))
        del config["target"]
        state = FSMDefinition(**_parent(config)).states["work"]
        choose = state.call.choose
        assert isinstance(choose, CallChoose)
        assert (choose.key, choose.fallback) == ("picked", "reasoning.deductive")
        assert choose.targets == ("reasoning.analytical", "reasoning.deductive")
        assert choose.targets_from is None

    def test_targets_from_is_a_declaration_kept_beside_the_closed_list(self):
        config = _call(choose=_choose(targets_from={"capability": "strategy"}))
        del config["target"]
        choose = FSMDefinition(**_parent(config)).states["work"].call.choose
        assert choose.targets_from.capability == "strategy"
        assert choose.targets == ("reasoning.analytical", "reasoning.deductive")

    def test_the_budgets_load(self):
        config = CallStateConfig(**_call(max_steps=30, max_seconds=2.5))
        assert (config.max_steps, config.max_seconds) == (30, 2.5)

    def test_a_state_without_it_is_unchanged(self):
        state = FSMDefinition(**_parent()).states["done"]
        assert state.call is None
        assert state.kind == "conversation"

    def test_the_state_kind_is_call(self):
        state = FSMDefinition(**_parent()).states["work"]
        assert state.kind == "call"
        assert kind_of(state) is get_kind("call")
        assert KINDS[-1].name == "call"

    def test_choose_key_may_be_a_handler_only_key(self):
        """D-015 (3): the orchestrator's choice key is written by a handler."""
        config = _call(choose=_choose(key="classified_problem_type"))
        del config["target"]
        FSMDefinition(**_parent(config, handler_only=("classified_problem_type",)))

    def test_duplicates_in_the_lists_are_dropped(self):
        config = CallStateConfig(**_call(inputs=["a", "a", "b"]))
        assert config.inputs == ("a", "b")
        choose = CallChoose(**_choose(targets=["x", "x", "y"]))
        assert choose.targets == ("x", "y")

    def test_frozen_and_extra_forbidden(self):
        config = CallStateConfig(**_call())
        with pytest.raises(ValidationError):
            config.target = "other"
        with pytest.raises(ValidationError):
            CallStateConfig(**_call(persona="a pirate"))
        with pytest.raises(ValidationError):
            CallChoose(**_choose(persona="a pirate"))


class TestTheDefinitionVersion:
    def test_the_default_is_4_3(self):
        assert SCHEMA_VERSION_DEFAULT == "4.3"
        assert FSMDefinition(**_parent()).version == "4.3"

    def test_an_omitted_version_moves_no_fsm_id(self):
        """The hash is `model_dump(exclude_defaults=True)`: the default is not in it."""
        plain = {
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
        assert API.process_fsm_definition(plain)[1] == "fsm_Golden_043181d0"

    def test_a_call_state_changes_only_its_own_fsm_id(self):
        _, with_field = API.process_fsm_definition(_parent())
        without = _parent()
        del without["states"]["work"]["call"]
        assert with_field != API.process_fsm_definition(without)[1]


def _bad_values() -> list[Any]:
    return [None, 5, 1.5, True, "", "   ", [], {}, "x" * 100_000]


class TestCallRefusals:
    @pytest.mark.parametrize("field", ["target", "on_failure", "result_key"])
    @pytest.mark.parametrize(
        "bad",
        [value for value in _bad_values() if value is not None],
        ids=lambda v: compact_param_id(v, "bad") or repr(v),
    )
    def test_bad_scalar_values(self, field, bad):
        with pytest.raises(ValidationError):
            CallStateConfig(**_call(**{field: bad}))

    @pytest.mark.parametrize("field", ["on_failure", "result_key"])
    def test_a_missing_or_none_required_scalar_is_refused(self, field):
        with pytest.raises(ValidationError):
            CallStateConfig(**_call(**{field: None}))

    def test_both_target_and_choose_is_refused(self):
        with pytest.raises(ValidationError, match="exactly one of target or choose"):
            CallStateConfig(**_call(choose=_choose()))

    def test_neither_target_nor_choose_is_refused(self):
        config = _call()
        del config["target"]
        with pytest.raises(ValidationError, match="exactly one of target or choose"):
            CallStateConfig(**config)

    def test_a_choose_with_no_targets_is_refused(self):
        for choose in (
            {"key": "picked", "targets": []},
            {"key": "picked"},
            {"key": "picked", "targets_from": {"capability": "strategy"}},
        ):
            config = _call(choose=choose)
            del config["target"]
            with pytest.raises(ValidationError, match="targets"):
                CallStateConfig(**config)

    def test_a_fallback_outside_the_closed_list_is_refused(self):
        with pytest.raises(ValidationError, match="fallback"):
            CallChoose(**_choose(fallback="somewhere.else"))

    @pytest.mark.parametrize(
        "bad", ["", "  ", "has space", ".lead", "a\nb", "x" * 129, 5, None]
    )
    def test_a_target_id_has_the_executable_id_shape(self, bad):
        with pytest.raises(ValidationError):
            CallStateConfig(**_call(target=bad))
        with pytest.raises(ValidationError):
            CallChoose(**_choose(targets=["fine.id", bad]))

    @pytest.mark.parametrize("bad", ["", "Upper", "1x", "has space", None, 5])
    def test_a_targets_from_capability_has_a_name_shape(self, bad):
        with pytest.raises(ValidationError):
            CallChoose(**_choose(targets_from={"capability": bad}))

    @pytest.mark.parametrize("bad", [None, 5, 1.5, True, "", "abc", {}, {"a": 1}])
    def test_bad_input_collections(self, bad):
        with pytest.raises(ValidationError):
            CallStateConfig(**_call(inputs=bad))

    @pytest.mark.parametrize("bad", [None, 5, "", "abc", ["a"], {"a": 5}, {"a": None}])
    def test_bad_returns_shapes(self, bad):
        with pytest.raises(ValidationError):
            CallStateConfig(**_call(returns=bad))

    @pytest.mark.parametrize(
        "bad_name",
        [
            *[None, 5, "", "   ", "x" * 100_000, ["a"], {"a": 1}],
            *["_hidden", "system_x", "internal_x", "__dunder", "_conversation_id"],
            *["agent_trace", "password", "api_key", "session_token"],
        ],
    )
    def test_bad_input_names(self, bad_name):
        with pytest.raises(ValidationError, match=r"public key names|must hold names"):
            CallStateConfig(**_call(inputs=["fine", bad_name]))

    @pytest.mark.parametrize(
        "bad_name",
        [
            *["", "   ", "x" * 100_000],
            *["_hidden", "system_x", "internal_x", "__dunder", "_approval_granted"],
            *["agent_trace", "password", "api_key", "session_token"],
        ],
    )
    def test_bad_return_names_on_both_sides(self, bad_name):
        """D2: the parent key and the child output are both public, non-secret names."""
        with pytest.raises(ValidationError, match="public key names"):
            CallStateConfig(**_call(returns={"fine": "ok", bad_name: "ok"}))
        with pytest.raises(ValidationError, match="public key names"):
            CallStateConfig(**_call(returns={"fine": "ok", "also": bad_name}))

    @pytest.mark.parametrize(
        "bad_name", ["_hidden", "system_x", "agent_trace", "password", "", "  "]
    )
    def test_the_choose_key_is_a_public_name(self, bad_name):
        with pytest.raises(ValidationError):
            CallChoose(**_choose(key=bad_name))

    @pytest.mark.parametrize(
        "bad", ["_private", "system_x", "agent_trace", "_conversation_id", "  "]
    )
    def test_result_key_is_public_and_not_reserved(self, bad):
        with pytest.raises(ValidationError):
            CallStateConfig(**_call(result_key=bad))

    def test_the_result_key_cannot_be_an_input(self):
        with pytest.raises(ValidationError, match="result_key cannot be"):
            CallStateConfig(**_call(inputs=["call_result"]))

    def test_the_result_key_cannot_be_a_return_key(self):
        with pytest.raises(ValidationError, match="result_key cannot be"):
            CallStateConfig(**_call(returns={"call_result": "x"}))

    def test_the_result_key_cannot_be_the_choose_key(self):
        config = _call(choose=_choose(key="call_result"))
        del config["target"]
        with pytest.raises(ValidationError, match="result_key cannot be"):
            CallStateConfig(**config)

    @pytest.mark.parametrize("field", ["max_steps", "max_seconds"])
    @pytest.mark.parametrize("bad", [0, -1, True, "5", float("nan"), float("inf")])
    def test_the_budgets_are_positive_numbers(self, field, bad):
        with pytest.raises(ValidationError):
            CallStateConfig(**_call(**{field: bad}))

    def test_a_bool_is_not_a_step_count(self):
        with pytest.raises(ValidationError):
            CallStateConfig(**_call(max_steps=True))


class TestCallStateRules:
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
            {"completion": {"response_format_key": "_fmt"}},
            {
                "synthesize": {
                    "goal": "g",
                    "return_keys": ["z"],
                    "on_failure": "apology",
                }
            },
        ],
        ids=lambda c: next(iter(c)),
    )
    def test_excluded_fields(self, conflict):
        with pytest.raises(ValidationError, match="call state cannot also"):
            FSMDefinition(**_parent(**conflict))

    def test_needs_transitions(self):
        with pytest.raises(ValidationError, match="transition targets"):
            State(**_call_state(transitions=[]))

    def test_on_failure_must_be_a_transition_target(self):
        with pytest.raises(ValidationError, match="on_failure"):
            FSMDefinition(**_parent(_call(on_failure="done_elsewhere")))


def _gate(*, requires: list[str] | None = None, logic: Any = None) -> dict[str, Any]:
    condition: dict[str, Any] = {"description": "gate"}
    if requires is not None:
        condition["requires_context_keys"] = requires
    if logic is not None:
        condition["logic"] = logic
    return {"target_state": "done", "description": "go", "conditions": [condition]}


class TestTheFailureEdgeMustExist:
    """The loader rule of E3.1: the edges read the result and a failure edge fires."""

    @pytest.mark.parametrize(
        "edge",
        [
            {"target_state": "done", "description": "no conditions"},
            {"target_state": "done", "description": "empty", "conditions": []},
            _gate(requires=["topic"]),
            _gate(logic={"==": [{"var": "topic"}, 1]}),
        ],
        ids=["bare", "empty-conditions", "other-key", "other-var"],
    )
    def test_an_edge_that_ignores_the_result_is_refused(self, edge):
        message = _refusal(_parent(transitions=[edge, _failure_edge()]))
        assert "the transition to 'done' must read the call result key" in message
        assert "before the target has produced a result" in message

    def test_a_missing_failure_edge_is_refused(self):
        definition = _parent(
            transitions=[
                _ok_edge(),
                _edge("apology", 2, {"==": [_STATUS, "failed"]}),
            ]
        )
        message = _refusal(definition)
        assert "no transition to 'apology' fires when the call failed or aborted" in (
            message
        )
        assert "'call_result' is aborted" in message

    def test_a_failure_edge_that_never_passes_is_refused(self):
        definition = _parent(
            transitions=[_ok_edge(), _edge("apology", 2, {"==": [_STATUS, "never"]})]
        )
        assert "no transition to 'apology' fires" in _refusal(definition)

    def test_an_edge_that_passes_on_running_is_refused(self):
        edge = _edge("done", 1, {"!=": [_STATUS, "ok"]})
        message = _refusal(_parent(transitions=[edge, _failure_edge()]))
        assert "the transition to 'done' fires before the target has produced" in (
            message
        )
        assert "'call_result' is running" in message

    def test_an_edge_that_passes_on_an_absent_result_is_refused(self):
        edge = _edge("done", 1, {"!": [{"var": "call_result"}]}, ())
        message = _refusal(_parent(transitions=[edge, _failure_edge()]))
        assert "'call_result' is absent" in message

    def test_the_in_failed_aborted_form_is_accepted(self):
        FSMDefinition(**_parent())

    def test_an_edge_gated_on_another_key_too_is_judged_with_that_key_present(self):
        failure = _edge(
            "apology",
            2,
            {
                "and": [
                    {"in": [_STATUS, ["failed", "aborted"]]},
                    {"==": [{"var": "note"}, "x"]},
                ]
            },
            ("call_result", "note"),
        )
        FSMDefinition(**_parent(transitions=[_ok_edge(), failure]))


class TestTheFailureEdgeMustWin:
    """Plan 8525c797 step 3 (D-004), end to end through the loader."""

    @staticmethod
    def _rival(priority: int) -> dict[str, Any]:
        return _edge("done", priority, {"in": [_STATUS, ["ok", "failed", "aborted"]]})

    def test_an_earlier_edge_that_also_passes_the_failure_is_refused(self):
        definition = _parent(transitions=[self._rival(1), _failure_edge()])
        message = _refusal(definition)
        assert "the transition to 'done' wins over the failure edge to 'apology'" in (
            message
        )

    def test_a_tie_with_the_failure_edge_is_refused(self):
        definition = _parent(transitions=[self._rival(2), _failure_edge()])
        message = _refusal(definition)
        assert "the transition to 'apology' ties with the transition to 'done'" in (
            message
        )

    def test_the_failure_edge_with_the_lowest_number_loads(self):
        failure = _edge("apology", 1, {"in": [_STATUS, ["failed", "aborted"]]})
        FSMDefinition(**_parent(transitions=[failure, self._rival(2)]))


class TestHostKeysMayNotBeResultOwned:
    def test_a_return_key_that_is_handler_only_is_refused(self):
        message = _refusal(
            _parent(_call(returns={"flag": "ok"}), handler_only=("flag",))
        )
        assert "call returns entry 'flag' collides with a handler_only_keys entry" in (
            message
        )

    def test_an_input_that_is_handler_only_is_accepted(self):
        FSMDefinition(**_parent(_call(inputs=["flag"]), handler_only=("flag",)))

    def test_an_input_that_is_the_result_of_another_call_state_is_refused(self):
        other = _call_state(
            _call(result_key="other_result"),
            id="nested",
            transitions=[
                _edge(
                    "done",
                    1,
                    {"==": [{"var": "other_result.status"}, "ok"]},
                    ("other_result",),
                ),
                _edge(
                    "apology",
                    2,
                    {"in": [{"var": "other_result.status"}, ["failed", "aborted"]]},
                    ("other_result",),
                ),
            ],
        )
        message = _refusal(
            _parent(_call(inputs=["other_result"]), extra_states={"nested": other})
        )
        assert (
            "call inputs entry 'other_result' collides with result_key of a call state"
            in message
        )

    def test_a_return_key_that_is_a_completion_result_is_refused(self):
        completion = {
            "id": "answer_state",
            "description": "a completion turn",
            "purpose": "p",
            "completion": {"response_format_key": "_fmt", "result_key": "answer_slot"},
            "transitions": [{"target_state": "done", "description": "go"}],
        }
        message = _refusal(
            _parent(
                _call(returns={"answer_slot": "x"}),
                extra_states={"answer_state": completion},
            )
        )
        assert "collides with result_key of a completion state" in message


class TestTheModelsAreOneSourceOfTruth:
    def test_the_refusals_come_from_one_checked_names_rule(self):
        """Mutant guard (D2): inputs and both sides of returns go through one rule.

        A name the SynthesizeConfig refuses is refused here with the same words."""
        for name in ("_hidden", "password", "agent_trace"):
            with pytest.raises(ValidationError) as caught:
                CallStateConfig(**_call(inputs=[name]))
            assert "may hold only public key names" in str(caught.value)
            with pytest.raises(ValidationError) as caught:
                CallStateConfig(**_call(returns={name: "x"}))
            assert "may hold only public key names" in str(caught.value)
            with pytest.raises(ValidationError) as caught:
                CallStateConfig(**_call(returns={"x": name}))
            assert "may hold only public key names" in str(caught.value)

    def test_a_loaded_definition_round_trips_through_its_dump(self):
        definition = FSMDefinition(**_parent())
        again = FSMDefinition(**copy.deepcopy(definition.model_dump()))
        assert again.states["work"].call == definition.states["work"].call
