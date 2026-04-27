"""
Case-Routed Reasoning — inline kernel ``case_`` routes between two reasoning factories
=========================================================================================

Demonstrates the kernel ``case_`` primitive at the inline-DSL level
mixing with stdlib factories. A host-callable classifier returns
"math" or "narrative"; ``case_`` dispatches to either ``calculator_term``
(2 leaves) or ``analytical_term`` (3 leaves).

Theorem-2 runtime arm-only:
- math arm: 2 oracle calls
- narrative arm: 3 oracle calls
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline
from fsm_llm.lam import app, case_, var
from fsm_llm.stdlib.reasoning import (
    analytical_term,
    calculator_term,
)

SCHEMA_PARSED = "examples.pipeline.case_routed_reasoning.schemas.Parsed"
SCHEMA_COMPUTED = "examples.pipeline.case_routed_reasoning.schemas.Computed"
SCHEMA_DECOMP = "examples.pipeline.case_routed_reasoning.schemas.Decomposition"
SCHEMA_ANALYSIS = "examples.pipeline.case_routed_reasoning.schemas.Analysis"
SCHEMA_INTEGRATION = "examples.pipeline.case_routed_reasoning.schemas.Integration"

# Switch QUESTION between these to exercise either arm.
QUESTION = "What is 8 times 9, minus 5?"  # math arm — answer = 67

PARSE_PROMPT = (
    "Parse the arithmetic expression from the problem.\n"
    "Problem: {problem}\n"
    "Return JSON with 'expression' and 'operands'."
)
COMPUTE_PROMPT = (
    "Compute the answer to the parsed expression.\n"
    "Problem: {problem}\nParsed (JSON): {parsed}\n"
    "Return JSON with 'answer' (numeric string) and 'work'."
)

DECOMPOSE_PROMPT = (
    "Decompose the problem.\nProblem: {problem}\nReturn JSON with 'parts' and 'domain'."
)
ANALYZE_PROMPT = (
    "Analyse.\nProblem: {problem}\nDecomposition (JSON): {decomposition}\n"
    "Return JSON with 'findings' and 'key_insight'."
)
INTEGRATE_PROMPT = (
    "Integrate.\nProblem: {problem}\nAnalysis (JSON): {analysis}\n"
    "Return JSON with 'answer' and 'confidence'."
)


_ARITH_RE = re.compile(
    r"\d.*[\+\-\*/×÷]|"
    r"(plus|minus|times|multipl|divid|sum|product|equals)",
    re.IGNORECASE,
)


def classify(question: str) -> str:
    if isinstance(question, str) and _ARITH_RE.search(question):
        return "math"
    return "narrative"


def build_term():
    calc = calculator_term(
        PARSE_PROMPT,
        COMPUTE_PROMPT,
        parse_schema_ref=SCHEMA_PARSED,
        compute_schema_ref=SCHEMA_COMPUTED,
    )
    analyse = analytical_term(
        DECOMPOSE_PROMPT,
        ANALYZE_PROMPT,
        INTEGRATE_PROMPT,
        input_vars_a=("problem",),
        input_vars_b=("problem", "decomposition"),
        input_vars_c=("problem", "analysis"),
        schema_ref_a=SCHEMA_DECOMP,
        schema_ref_b=SCHEMA_ANALYSIS,
        schema_ref_c=SCHEMA_INTEGRATION,
    )
    # case_ scrutinee = classify(problem); branches keyed by classifier output.
    return case_(
        app(var("classify"), var("problem")),
        {"math": calc, "narrative": analyse},
        default=analyse,
    )


def checks(result, error, oracle_calls):
    has_answer = isinstance(result, dict) and "answer" in result
    # Math arm fires for the default question → 2 oracle calls.
    arm_consistent = oracle_calls in (2, 3)
    return {
        "result_is_dict": isinstance(result, dict),
        "answer_key_present": has_answer,
        "arm_oracle_calls_consistent": arm_consistent,
        "no_error": error is None,
    }


def main() -> int:
    return run_pipeline(
        build_term(),
        env={"problem": QUESTION, "classify": classify},
        checks_fn=checks,
        title="Case-Routed Reasoning (inline case_ + stdlib factories)",
    )


if __name__ == "__main__":
    sys.exit(main())
