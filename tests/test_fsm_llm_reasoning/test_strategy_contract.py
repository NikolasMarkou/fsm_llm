"""The contract of a strategy envelope is the problem in, every mapped key out (plan baa37f40, step 14).

D-030 (planned decision P6): a strategy must be a valid ``call`` target for the orchestrator
(step 15) and keep its standalone behaviour. The ``contract`` block of each of the nine files
is ``inputs [problem_statement]`` and ``outputs []``; the bodies (``body.fsm``, ``body.strategy``)
stay byte equal, so a refused calculator expression returns its error key instead of failing
the run, and a caller with no ``constraints`` runs. Offline: a scripted model, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import fsm_llm
from fsm_llm.core.codes import Status
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import ExecutableRegistry, FsmBodyExecutable
from fsm_llm.reasoning import executables as module
from fsm_llm.reasoning.executables import builtin_executables, strategy_id
from fsm_llm.reasoning.strategy_files import (
    STRATEGY_ORDER,
    load_document,
    load_fsm,
    strategy_facts,
)

from .test_executables import _llm, _registry

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "executables" / "reasoning"
PROBLEM = {"problem_statement": "Is Socrates mortal?"}


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")
    assert Path(module.__file__).resolve().is_relative_to(REPO / "src")


@pytest.mark.parametrize("name", STRATEGY_ORDER)
class TestTheEnvelope:
    def test_the_contract_is_the_problem_in_and_no_declared_output(self, name):
        contract = load_document(name)["contract"]
        assert contract == {"inputs": ["problem_statement"], "outputs": []}

    def test_the_body_is_the_fixture_as_decoded_text(self, name):
        """The bodies do not move: the E0.1 fixture is the byte proof (D-030, STOP IF)."""
        text = (FIXTURES / f"{name}.json").read_text(encoding="utf-8")
        assert json.dumps(load_fsm(name), indent=2, ensure_ascii=True) + "\n" == text

    def test_the_declaration_is_unchanged(self, name):
        facts = strategy_facts(name)
        assert set(facts) == {"description", "aliases", "results", "solution_key"}
        assert facts["solution_key"] in facts["results"]

    def test_the_document_loads_with_no_error(self, name):
        envelope, report = load_executable(load_document(name))
        assert envelope is not None and report.is_valid, report.errors
        assert list(envelope.contract.inputs) == ["problem_statement"]
        assert list(envelope.contract.outputs) == []


@pytest.mark.parametrize("name", STRATEGY_ORDER)
class TestARunWithOnlyTheProblem:
    def test_the_run_ends_ok_with_every_non_none_mapped_key(self, name):
        """Absent ``constraints`` (and components, and type) run: the parent contract refused
        them as ``inputs_missing``."""
        exe = _registry().get(strategy_id(name))
        result = exe.run(PROBLEM, context=RunContext(llm_interface=_llm(name)))
        facts = strategy_facts(name)
        assert result.status is Status.OK, (result.reason, result.detail)
        assert facts["solution_key"] in result.outputs
        assert set(result.outputs) <= set(facts["results"])
        assert all(value is not None for value in result.outputs.values())

    def test_a_key_outside_the_contract_is_refused(self, name):
        exe = _registry().get(strategy_id(name))
        with pytest.raises(ValueError, match="declares no input"):
            exe.run({**PROBLEM, "constraints": []})


class TestARefusedExpressionReturnsInsteadOfFailing:
    def test_division_by_zero_ends_ok_with_the_error_key(self):
        """The old contract required ``calculation_result``: a refused expression left it None
        and the run was ``failed/return_keys_missing`` with ``calculation_error_details`` lost."""
        exe = _registry().get("reasoning.simple_calculator")
        llm = _llm("simple_calculator", calculation_expression="1 / 0")
        result = exe.run(PROBLEM, context=RunContext(llm_interface=llm))
        assert result.status is Status.OK, (result.reason, result.detail)
        assert list(result.outputs) == ["calculation_error_details"]
        assert "division" in result.outputs["calculation_error_details"].lower()

    def test_a_good_expression_returns_the_result_and_the_proposed_solution(self):
        exe = _registry().get("reasoning.simple_calculator")
        result = exe.run(
            PROBLEM, context=RunContext(llm_interface=_llm("simple_calculator"))
        )
        assert result.outputs == {"calculation_result": 36, "proposed_solution": 36}


class TestTheResolverCarriesTheOutputMap:
    def test_a_strategy_resolves_with_its_results_map(self):
        resolved = _registry().resolve_target("reasoning.analytical")
        assert resolved is not None
        assert dict(resolved.returns) == strategy_facts("analytical")["results"]

    def test_every_strategy_carries_exactly_its_own_map(self):
        registry = _registry()
        for name in STRATEGY_ORDER:
            resolved = registry.resolve_target(strategy_id(name))
            assert resolved is not None
            assert dict(resolved.returns) == strategy_facts(name)["results"]

    def test_a_target_with_no_map_carries_none(self):
        """An ``fsm`` body names no outputs: ``returns`` stays ``None`` (the call's own
        ``returns`` is then the whole rule)."""
        registry = ExecutableRegistry()
        analytical = next(
            e for e in builtin_executables() if e.id == "reasoning.analytical"
        )
        assert isinstance(analytical, FsmBodyExecutable)
        document = load_document("analytical")
        document.update(
            kind="fsm",
            id="probe.plain",
            name="plain",
            body={"fsm": load_fsm("analytical")},
        )
        document["capabilities"] = []
        registry.add_document(document)
        resolved = registry.resolve_target("probe.plain")
        assert resolved is not None and resolved.returns is None

    def test_the_map_cannot_be_changed_through_the_resolved_target(self):
        resolved = _registry().resolve_target("reasoning.analytical")
        assert resolved is not None
        with pytest.raises(TypeError):
            resolved.returns["injected"] = "x"  # type: ignore[index]
