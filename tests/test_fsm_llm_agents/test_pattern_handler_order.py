"""The handlers each built-in pattern registers, in execution order (plan a555dcfe, step 8).

E4.2 turns the pattern builders into files and their handler registration into library
factories (steps 9 to 18). A handler that moves, changes priority or timing, or drops out reorders
a turn without any FSM text moving, so this golden pins the tuple of every registered handler of a
scripted run per pattern, read through the run's ``HandlerSystem`` (``api.handler_system.handlers``,
the sorted list the pipeline executes): name, timings, states, target states, watched keys and
priority, in order. The lifecycle handlers (error, compactor, summarizer) are in the list: they
stay in the class and their place among the pattern's own is part of the order.

The golden is ``tests/fixtures/executables/pattern_handler_order.json``, written by
``python -m tests.executables_corpus --write`` on the PARENT tree of the conversion (nothing
converted yet at step 8). Do NOT regenerate it to make a failure go away: a difference is a pattern
that registers something else (census invariants D1 and D4, hazard 4 of the plan).
"""

from __future__ import annotations

import json

import pytest

from tests import pattern_runs as runs
from tests.executables_corpus import HANDLER_ORDER_PATH

_GOLDEN = json.loads(HANDLER_ORDER_PATH.read_text(encoding="utf-8"))
_APPROVAL_NAMES = {"AgentHITLGate", "AgentApprovalFreshKeys"}


def _attempts(handlers: list) -> list[list]:
    """The handler lists of a run: one, or one per attempt of a retried run (verified_react)."""
    return [handlers] if handlers and isinstance(handlers[0][0], str) else handlers


def _names(handlers: list) -> list[str]:
    return [row[0] for row in handlers]


class TestGoldenFixture:
    def test_the_fixture_covers_exactly_the_cases(self):
        assert sorted(_GOLDEN) == sorted(runs.CASES())

    def test_every_case_registers_the_error_and_compactor_handlers(self):
        for name, golden in _GOLDEN.items():
            for handlers in _attempts(golden):
                names = _names(handlers)
                assert "AgentError" in names, name
                assert "AgentContextCompactor" in names, name

    def test_execution_order_is_priority_order(self):
        for name, golden in _GOLDEN.items():
            for handlers in _attempts(golden):
                priorities = [row[5] for row in handlers]
                assert priorities == sorted(priorities), name

    def test_only_a_run_with_an_approval_policy_registers_the_approval_handlers(self):
        for name, golden in _GOLDEN.items():
            for handlers in _attempts(golden):
                has = bool(_APPROVAL_NAMES & set(_names(handlers)))
                assert has == name.endswith("_approval"), name

    def test_the_gate_runs_before_everything_else_of_its_pattern(self):
        for name in ("react_approval", "reflexion_approval"):
            first = _GOLDEN[name][0]
            assert first[0] == "AgentHITLGate", name
            assert first[5] == min(row[5] for row in _GOLDEN[name]), name


class TestGoldenHandlers:
    @pytest.mark.parametrize("name", sorted(_GOLDEN))
    def test_the_registered_handlers_equal_the_golden(self, name):
        got = runs.run_case(name)["handlers"]
        assert got == _GOLDEN[name], f"{name}: the registered handlers moved"

    @pytest.mark.parametrize("name", sorted(_GOLDEN))
    def test_a_second_run_registers_the_same_handlers(self, name):
        first = runs.run_case(name)["handlers"]
        assert runs.run_case(name)["handlers"] == first

    def test_one_agent_runs_twice_with_the_same_handlers(self, monkeypatch):
        """Per-run state (D-014 of plan 089d0ec7): a second run of ONE instance registers
        a fresh, equal handler list, never the first run's objects."""
        from fsm_llm.agents.base.agent import BaseAgent

        seen: list[list] = []
        original = BaseAgent._prepare_run_api

        def spy(self, fsm_def, agent_type, handlers):
            api = original(self, fsm_def, agent_type, handlers)
            seen.append(list(api.handler_system.handlers))
            return api

        model = runs.ScriptedModel()
        agent = runs._cases()["react"].build(model)
        monkeypatch.setattr(BaseAgent, "_prepare_run_api", spy)
        agent.run("first task")
        model.counts.clear()
        agent.run("second task")
        assert len(seen) == 2
        assert [runs.handler_tuple(h) for h in seen[0]] == [
            runs.handler_tuple(h) for h in seen[1]
        ]
        assert not {id(h) for h in seen[0]} & {id(h) for h in seen[1]}


class TestARetriedRunRegistersTheSameHandlersEveryAttempt:
    """``verified_react`` runs the loop again on a rejected answer (step 28.5): each attempt gets its own,
    equal handler list, never the first attempt's objects."""

    @pytest.mark.parametrize(
        "name", ["verified_react_retry", "verified_react_rejected"]
    )
    def test_every_attempt_has_the_same_handlers(self, name):
        attempts = _attempts(_GOLDEN[name])

        assert len(attempts) == 2
        assert attempts[0] == attempts[1]
