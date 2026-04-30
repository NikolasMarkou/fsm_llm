"""
Deductive Reasoning — premises → infer → conclude
=====================================================

Demonstrates ``fsm_llm.stdlib.reasoning.deductive_term`` (M3 slice 2):
3-leaf let-chain. Classic syllogism input. Theorem-2 strict: 3 calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline
from fsm_llm.stdlib.reasoning import deductive_term

SCHEMA_PREMISES = "examples.pipeline.deductive_reasoning.schemas.Premises"
SCHEMA_INFERENCE = "examples.pipeline.deductive_reasoning.schemas.Inference"
SCHEMA_CONCLUSION = "examples.pipeline.deductive_reasoning.schemas.Conclusion"

PROBLEM = "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?"

PREMISES_PROMPT = (
    "Extract the explicit and implicit premises from the problem.\n"
    "Problem: {problem}\n\n"
    "Return JSON with 'premises' (list of statements)."
)

INFER_PROMPT = (
    "Apply deductive inference rules to the premises.\n"
    "Problem: {problem}\nPremises (JSON): {premises}\n\n"
    "Return JSON with 'inference_steps' (list) and 'derived_fact'."
)

CONCLUDE_PROMPT = (
    "Form the final conclusion answering the problem.\n"
    "Problem: {problem}\nInference (JSON): {inference}\n\n"
    "Return JSON with 'answer' (yes/no/sentence) and 'is_valid' (bool)."
)


def build_term():
    return deductive_term(
        PREMISES_PROMPT,
        INFER_PROMPT,
        CONCLUDE_PROMPT,
        premises_schema_ref=SCHEMA_PREMISES,
        inference_schema_ref=SCHEMA_INFERENCE,
        conclusion_schema_ref=SCHEMA_CONCLUSION,
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
        title="Deductive Reasoning (stdlib.reasoning.deductive_term)",
    )


if __name__ == "__main__":
    sys.exit(main())
