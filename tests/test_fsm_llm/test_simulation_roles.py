"""The distinctness rule of ``fsm_llm.simulation.roles`` (plan c720601d, D-009).

A judge that is the executor or the compiler, or whose name cannot be told, is refused
unless the caller says ``allow_same_model=True``. The rule compares names only.
"""

from __future__ import annotations

import dataclasses

import pytest

from fsm_llm.simulation import (
    CODE_JUDGE_NOT_DISTINCT,
    Roles,
    SimulationError,
    Simulator,
)
from fsm_llm.testing import ScriptedLLM


def _llm(model: str | None) -> ScriptedLLM:
    return ScriptedLLM([], model=model)


class TestRefusals:
    def test_the_same_object_is_refused(self):
        same = _llm("m")
        with pytest.raises(SimulationError) as info:
            Simulator(same, same)
        assert info.value.code == CODE_JUDGE_NOT_DISTINCT == "judge_not_distinct"

    def test_a_judge_that_defaults_to_the_executor_is_refused(self):
        with pytest.raises(SimulationError, match="judge_not_distinct"):
            Simulator(_llm("m"))

    def test_equal_names_are_refused(self):
        with pytest.raises(SimulationError):
            Simulator(_llm("m"), _llm("m"))

    def test_names_equal_after_strip_are_refused(self):
        with pytest.raises(SimulationError):
            Simulator(_llm("m"), _llm("  m\n"))

    def test_an_unnamed_judge_fails_closed(self):
        with pytest.raises(SimulationError):
            Simulator(_llm("m"), _llm(None))

    def test_an_unnamed_executor_fails_closed(self):
        with pytest.raises(SimulationError):
            Simulator(_llm(None), _llm("j"))

    def test_a_blank_name_is_unnamed(self):
        with pytest.raises(SimulationError):
            Simulator(_llm("   "), _llm("j"))

    def test_a_model_attribute_that_is_not_a_string_is_unnamed(self):
        from unittest.mock import MagicMock

        seven = _llm("j")
        seven.model = 7  # type: ignore[assignment]
        with pytest.raises(SimulationError):
            Simulator(seven, _llm("j2"))
        one, two = _llm("a"), _llm("b")
        one.model, two.model = MagicMock(), MagicMock()  # type: ignore[assignment]
        with pytest.raises(SimulationError):  # a mock attribute is not a name
            Simulator(one, two)

    def test_the_message_names_the_code_and_no_model(self):
        with pytest.raises(SimulationError) as info:
            Simulator(_llm("secret-model-7"), _llm("secret-model-7"))
        assert "judge_not_distinct" in str(info.value)
        assert "secret-model-7" not in str(info.value)

    def test_the_compiler_is_checked_against_the_judge(self):
        sim = Simulator(_llm("runner"), _llm("judge"))
        with pytest.raises(SimulationError):
            sim.roles(_llm("judge"))
        with pytest.raises(SimulationError):
            sim.roles(_llm(None))  # an unnamed compiler cannot be told apart

    def test_the_compiler_as_the_same_object_is_refused(self):
        judge = _llm("judge")
        with pytest.raises(SimulationError):
            Simulator(_llm("runner"), judge).roles(judge)


class TestAllowed:
    def test_distinct_names_pass_and_are_named(self):
        sim = Simulator(_llm("runner"), _llm(" judge "))
        assert sim.roles(_llm("compiler")) == Roles(
            "compiler", "runner", "judge", False
        )

    def test_the_compiler_may_equal_the_executor(self):
        sim = Simulator(_llm("runner"), _llm("judge"))
        assert sim.roles(_llm("runner")).compiler == "runner"

    def test_without_a_compiler_it_is_none(self):
        assert Simulator(_llm("r"), _llm("j")).roles().compiler is None

    def test_the_flag_waives_the_check_and_is_carried(self):
        same = _llm(None)
        sim = Simulator(same, allow_same_model=True)
        roles = sim.roles(same)
        assert roles == Roles(None, None, None, True) and roles.same_model_allowed

    def test_roles_are_a_frozen_value(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Roles("a", "b", "c").judge = "x"  # type: ignore[misc]
