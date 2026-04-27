"""
Calculator Reasoning — parse → compute (2 leaves)
=====================================================

Demonstrates ``fsm_llm.stdlib.reasoning.calculator_term`` (M3 slice 2):
2-leaf let-chain. Theorem-2 strict: 2 oracle calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline
from fsm_llm.stdlib.reasoning import calculator_term

SCHEMA_PARSED = "examples.pipeline.calculator_reasoning.schemas.Parsed"
SCHEMA_COMPUTED = "examples.pipeline.calculator_reasoning.schemas.Computed"

PROBLEM = "What is 17 multiplied by 23, plus 4?"

PARSE_PROMPT = (
    "Parse the arithmetic expression from the problem.\n"
    "Problem: {problem}\n\n"
    "Return JSON with 'expression' (e.g. '17*23+4') and 'operands' "
    "(list of numeric strings)."
)

COMPUTE_PROMPT = (
    "Compute the answer for the parsed expression. Show one step of work.\n"
    "Problem: {problem}\nParsed (JSON): {parsed}\n\n"
    "Return JSON with 'answer' (numeric string) and 'work' (one-line)."
)


def build_term():
    return calculator_term(
        PARSE_PROMPT,
        COMPUTE_PROMPT,
        parse_schema_ref=SCHEMA_PARSED,
        compute_schema_ref=SCHEMA_COMPUTED,
    )


def checks(result, error, oracle_calls):
    has_answer = isinstance(result, dict) and "answer" in result
    answer_is_correct = False
    if isinstance(result, dict):
        ans = str(result.get("answer", ""))
        # 17*23 + 4 = 391 + 4 = 395
        answer_is_correct = "395" in ans
    return {
        "result_is_dict": isinstance(result, dict),
        "answer_key_present": has_answer,
        "answer_is_395": answer_is_correct,
        "two_oracle_calls": oracle_calls == 2,
        "no_error": error is None,
    }


def main() -> int:
    return run_pipeline(
        build_term(),
        env={"problem": PROBLEM},
        checks_fn=checks,
        title="Calculator Reasoning (stdlib.reasoning.calculator_term)",
    )


if __name__ == "__main__":
    sys.exit(main())
