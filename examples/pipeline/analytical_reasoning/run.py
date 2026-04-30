"""
Analytical Reasoning — first standalone demo of stdlib.reasoning.analytical_term
====================================================================================

Demonstrates ``fsm_llm.stdlib.reasoning.analytical_term`` (M3 slice 2):
3-leaf let-chain — decompose → analyse → integrate. Theorem-2 strict:
oracle_calls == 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline
from fsm_llm.stdlib.reasoning import analytical_term

SCHEMA_DECOMP = "examples.pipeline.analytical_reasoning.schemas.Decomposition"
SCHEMA_ANALYSIS = "examples.pipeline.analytical_reasoning.schemas.Analysis"
SCHEMA_INTEGRATION = "examples.pipeline.analytical_reasoning.schemas.Integration"

PROBLEM = "Why does ice float on water?"

DECOMPOSE_PROMPT = (
    "Decompose the problem into 2-4 component sub-questions.\n"
    "Problem: {problem}\n\n"
    "Return JSON with 'parts' (list of sub-questions) and 'domain'."
)

ANALYZE_PROMPT = (
    "Analyse each sub-question and surface key findings.\n"
    "Problem: {problem}\nDecomposition (JSON): {decomposition}\n\n"
    "Return JSON with 'findings' (list of bullet strings) and "
    "'key_insight' (one sentence)."
)

INTEGRATE_PROMPT = (
    "Integrate the findings into a final answer.\n"
    "Problem: {problem}\nAnalysis (JSON): {analysis}\n\n"
    "Return JSON with 'answer' (1-2 sentences) and 'confidence' 0..1."
)


def build_term():
    return analytical_term(
        DECOMPOSE_PROMPT,
        ANALYZE_PROMPT,
        INTEGRATE_PROMPT,
        decomposition_schema_ref=SCHEMA_DECOMP,
        analysis_schema_ref=SCHEMA_ANALYSIS,
        integration_schema_ref=SCHEMA_INTEGRATION,
    )


def checks(result, error, oracle_calls):
    has_answer_key = isinstance(result, dict) and "answer" in result
    return {
        "result_is_dict": isinstance(result, dict),
        "answer_key_present": has_answer_key,
        "three_oracle_calls": oracle_calls == 3,
        "no_error": error is None,
    }


def main() -> int:
    return run_pipeline(
        build_term(),
        env={"problem": PROBLEM},
        checks_fn=checks,
        title="Analytical Reasoning (stdlib.reasoning.analytical_term)",
    )


if __name__ == "__main__":
    sys.exit(main())
