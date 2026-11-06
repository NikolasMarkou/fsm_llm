"""The harness timeout table against the examples' own agent budgets (plan
0068b055 step 12.21, live finding D5).

An example whose agent runs on the default ``AgentConfig.timeout_seconds``
(300 s) used to share its 300 s harness timeout: the agent's own timeout and the
harness kill fired together, scored 4 (clean exit with an ``Error:`` line) or 1
by chance. The table now gives such an example budget + 60 s.

The scan is static (AST over ``examples/**/run*.py``, read only): every call to
``AgentConfig(...)`` is one budget, the literal ``timeout_seconds=`` keyword when
there is one, else ``AgentConfig``'s default. A call whose ``timeout_seconds``
is not a literal is skipped (counted in ``_scan().skipped``).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from fsm_llm.agents.constants import Defaults
from fsm_llm.eval.config import EvalConfig
from fsm_llm.eval.constants import (
    CATEGORY_TIMEOUTS,
    EXAMPLE_SCRIPT_NAMES,
    EXAMPLE_TIMEOUTS,
)
from fsm_llm.eval.examples import get_timeout

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_MARGIN = 60
_DEFAULT_TIMEOUT = EvalConfig().timeout


@dataclass(frozen=True)
class _Budget:
    example: str
    category: str
    seconds: float
    literal: bool  # False: AgentConfig's default, no timeout_seconds in the call


@dataclass(frozen=True)
class _Scan:
    budgets: list[_Budget]
    skipped: list[str]  # "<example>:<line>" of a non-literal timeout_seconds


def _scan() -> _Scan:
    budgets: dict[tuple[str, float, bool], _Budget] = {}
    skipped: list[str] = []
    for script in EXAMPLE_SCRIPT_NAMES:
        for path in sorted(_EXAMPLES.glob(f"*/*/{script}")):
            category, directory = path.parts[-3], path.parts[-2]
            example = f"{category}/{directory}" + (
                "_manual" if script == "run_manual.py" else ""
            )
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else None
                if name != "AgentConfig":
                    continue
                value = next(
                    (k.value for k in node.keywords if k.arg == "timeout_seconds"),
                    None,
                )
                if value is None:
                    seconds, literal = Defaults.TIMEOUT_SECONDS, False
                elif isinstance(value, ast.Constant) and isinstance(
                    value.value, (int, float)
                ):
                    seconds, literal = float(value.value), True
                else:
                    skipped.append(f"{example}:{node.lineno}")
                    continue
                budgets[(example, seconds, literal)] = _Budget(
                    example, category, seconds, literal
                )
    return _Scan(list(budgets.values()), skipped)


def _harness(budget: _Budget) -> int:
    return get_timeout(
        budget.example,
        budget.category,
        _DEFAULT_TIMEOUT,
        EXAMPLE_TIMEOUTS,
        CATEGORY_TIMEOUTS,
    )


_SCAN = _scan()
_LITERAL = [b for b in _SCAN.budgets if b.literal]
_DEFAULTED = [b for b in _SCAN.budgets if not b.literal]


class TestScanFindsTheExamples:
    def test_examples_are_there(self):
        assert _EXAMPLES.is_dir()

    def test_scan_sees_literal_and_default_budgets(self):
        assert any(b.example == "agents/react_hitl_combined" for b in _LITERAL)
        assert len(_DEFAULTED) >= 20

    def test_default_budget_is_the_agent_config_default(self):
        assert Defaults.TIMEOUT_SECONDS == 300.0


class TestHarnessOutlastsTheAgentBudget:
    @pytest.mark.parametrize("budget", _LITERAL, ids=lambda b: b.example)
    def test_literal_budget_is_below_the_harness_timeout(self, budget):
        assert _harness(budget) > budget.seconds

    @pytest.mark.parametrize("budget", _DEFAULTED, ids=lambda b: b.example)
    def test_default_budget_never_ties_the_harness_timeout(self, budget):
        """Below the budget the harness is the hang detector and always fires
        first; at or above it the agent's timeout must fire first by a margin."""
        harness = _harness(budget)
        assert harness < budget.seconds or harness >= budget.seconds + _MARGIN

    def test_the_named_example_has_the_margin(self):
        assert EXAMPLE_TIMEOUTS["agents/supply_chain_optimizer"] == 300 + _MARGIN

    def test_no_example_ties_its_budget_exactly(self):
        ties = [b.example for b in _SCAN.budgets if _harness(b) == b.seconds]
        assert ties == []
