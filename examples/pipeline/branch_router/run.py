"""
Branch Router — boolean branch via stdlib.workflows.branch_term
==================================================================

Demonstrates ``fsm_llm.stdlib.workflows.branch_term``: a host-callable
predicate routes the input to either a calculator branch or a factual
branch. Theorem-2 runtime arm-only: oracle_calls == leaves(taken_arm)
== 1 for either arm in this demo.

Predicate is a pure-Python heuristic (contains digits + arithmetic
operator → "true" → calculator branch; else "false" → factual branch).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline
from fsm_llm.lam import leaf
from fsm_llm.stdlib.workflows import branch_term

SCHEMA_CALC = "examples.pipeline.branch_router.schemas.CalcAnswer"
SCHEMA_FACT = "examples.pipeline.branch_router.schemas.FactAnswer"

QUESTION = "What is 17 plus 25 multiplied by 2?"

CALC_TEMPLATE = (
    "You are a calculator. Compute the answer to: {question}\n\n"
    "Return JSON with:\n"
    "- answer: the numeric answer as a string\n"
    "- expression: the parsed arithmetic expression\n"
)

FACT_TEMPLATE = (
    "You are a factual Q&A assistant.\n"
    "Question: {question}\n\n"
    "Return JSON with:\n"
    "- answer: a concise factual answer\n"
    "- confidence: 0..1 score\n"
)


_ARITH_RE = re.compile(
    r"\d.*[\+\-\*/×÷]|"
    r"(plus|minus|times|multipl|divid|sum|product|equals)",
    re.IGNORECASE,
)


def is_arithmetic(question: str) -> str:
    """Predicate returning 'true' if the input looks arithmetic, else 'false'."""
    if isinstance(question, str) and _ARITH_RE.search(question):
        return "true"
    return "false"


def build_term():
    calc_leaf = leaf(
        template=CALC_TEMPLATE,
        input_vars=("question",),
        schema_ref=SCHEMA_CALC,
    )
    fact_leaf = leaf(
        template=FACT_TEMPLATE,
        input_vars=("question",),
        schema_ref=SCHEMA_FACT,
    )
    return branch_term(
        cond_var="is_arith",
        then_term=calc_leaf,
        else_term=fact_leaf,
        input_var="question",
    )


def checks(result, error, oracle_calls):
    answer = ""
    if isinstance(result, dict):
        answer = result.get("answer", "")
    return {
        "answer_present": (
            error is None and isinstance(answer, str) and len(answer) > 0
        ),
        "exactly_one_oracle_call": oracle_calls == 1,
        "no_error": error is None,
    }


def main() -> int:
    return run_pipeline(
        build_term(),
        env={"question": QUESTION, "is_arith": is_arithmetic},
        checks_fn=checks,
        title="Branch Router (stdlib.workflows.branch_term)",
    )


if __name__ == "__main__":
    sys.exit(main())
