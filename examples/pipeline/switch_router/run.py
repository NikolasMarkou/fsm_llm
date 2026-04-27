"""
Switch Router — N-way branch via stdlib.workflows.switch_term
================================================================

Demonstrates ``fsm_llm.stdlib.workflows.switch_term`` with 3 named
branches (math, history, geography) plus a default fallback. A
heuristic classifier keyword-matches the question and dispatches to one
arm. Theorem-2 runtime arm-only: oracle_calls == leaves(taken_arm) == 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline
from fsm_llm.lam import leaf
from fsm_llm.stdlib.workflows import switch_term

SCHEMA_ANS = "examples.pipeline.switch_router.schemas.DomainAnswer"

QUESTION = "Who was the first emperor of Rome?"

MATH_TEMPLATE = (
    "You are a math tutor. Answer: {question}\n"
    "Return JSON with 'answer' (numeric) and 'domain'='math'."
)
HISTORY_TEMPLATE = (
    "You are a history teacher. Answer: {question}\n"
    "Return JSON with 'answer' (one sentence) and 'domain'='history'."
)
GEO_TEMPLATE = (
    "You are a geography expert. Answer: {question}\n"
    "Return JSON with 'answer' (one sentence) and 'domain'='geography'."
)
DEFAULT_TEMPLATE = (
    "You are a general assistant. Answer: {question}\n"
    "Return JSON with 'answer' (one sentence) and 'domain'='other'."
)


_MATH_KW = ("plus", "minus", "times", "multiply", "divide", "sum", "+", "-", "*", "/")
_HIST_KW = (
    "emperor",
    "king",
    "queen",
    "war",
    "century",
    "ancient",
    "history",
    "rome",
    "egypt",
)
_GEO_KW = ("country", "capital", "river", "mountain", "continent", "ocean")


def classify(question: str) -> str:
    q = question.lower() if isinstance(question, str) else ""
    if any(k in q for k in _MATH_KW):
        return "math"
    if any(k in q for k in _HIST_KW):
        return "history"
    if any(k in q for k in _GEO_KW):
        return "geography"
    return "other"


def build_term():
    math_leaf = leaf(
        template=MATH_TEMPLATE, input_vars=("question",), schema_ref=SCHEMA_ANS
    )
    hist_leaf = leaf(
        template=HISTORY_TEMPLATE, input_vars=("question",), schema_ref=SCHEMA_ANS
    )
    geo_leaf = leaf(
        template=GEO_TEMPLATE, input_vars=("question",), schema_ref=SCHEMA_ANS
    )
    default_leaf = leaf(
        template=DEFAULT_TEMPLATE, input_vars=("question",), schema_ref=SCHEMA_ANS
    )
    return switch_term(
        classifier_var="classify",
        branches={"math": math_leaf, "history": hist_leaf, "geography": geo_leaf},
        default_term=default_leaf,
        input_var="question",
    )


def checks(result, error, oracle_calls):
    has_answer = isinstance(result, dict) and "answer" in result
    return {
        "answer_present": has_answer,
        "exactly_one_oracle_call": oracle_calls == 1,
        "no_error": error is None,
    }


def main() -> int:
    return run_pipeline(
        build_term(),
        env={"question": QUESTION, "classify": classify},
        checks_fn=checks,
        title="Switch Router (stdlib.workflows.switch_term)",
    )


if __name__ == "__main__":
    sys.exit(main())
