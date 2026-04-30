"""
Abductive Reasoning — observe → hypothesize → infer best
============================================================

Demonstrates ``fsm_llm.stdlib.reasoning.abductive_term`` (M3 slice 2):
3-leaf let-chain. Theorem-2 strict: 3 oracle calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline
from fsm_llm.stdlib.reasoning import abductive_term

SCHEMA_OBS = "examples.pipeline.abductive_reasoning.schemas.Observation"
SCHEMA_HYP = "examples.pipeline.abductive_reasoning.schemas.Hypothesis"
SCHEMA_BEST = "examples.pipeline.abductive_reasoning.schemas.BestExplanation"

PROBLEM = (
    "A patient has a fever, sore throat, and white spots on the tonsils. "
    "What is the most likely diagnosis among common conditions?"
)

OBSERVE_PROMPT = (
    "List the observable facts from the case description.\n"
    "Problem: {problem}\n\n"
    "Return JSON with 'facts' (list) and 'pattern' (one phrase)."
)

HYPOTHESIZE_PROMPT = (
    "Generate 2-4 candidate hypotheses that could explain the observations.\n"
    "Problem: {problem}\nObservation (JSON): {observation}\n\n"
    "Return JSON with 'candidates' (list) and 'primary' (most likely)."
)

INFER_PROMPT = (
    "Pick the best explanation from the hypotheses.\n"
    "Problem: {problem}\nHypothesis (JSON): {hypothesis}\n\n"
    "Return JSON with 'answer' (final diagnosis) and 'rationale'."
)


def build_term():
    return abductive_term(
        OBSERVE_PROMPT,
        HYPOTHESIZE_PROMPT,
        INFER_PROMPT,
        observation_schema_ref=SCHEMA_OBS,
        hypothesis_schema_ref=SCHEMA_HYP,
        selection_schema_ref=SCHEMA_BEST,
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
        title="Abductive Reasoning (stdlib.reasoning.abductive_term)",
    )


if __name__ == "__main__":
    sys.exit(main())
