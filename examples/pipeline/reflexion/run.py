"""Reflexion — λ-DSL twin (S3 Reflexion shape).

λ-twin: solve → eval → reflect → re-solve. 4 oracle calls. Depth-1 retry
flatten of the original Reflexion fix-loop. Original at examples/agents/reflexion/
untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.pipeline._helpers import run_pipeline
from fsm_llm.lam import leaf, let_

SCHEMA_ATTEMPT = "examples.pipeline.reflexion.schemas.Attempt"
SCHEMA_EVAL = "examples.pipeline.reflexion.schemas.Evaluation"
SCHEMA_REFLECT = "examples.pipeline.reflexion.schemas.Reflection"
SCHEMA_FINAL = "examples.pipeline.reflexion.schemas.Final"

TASK = 'What is the population density of France?'


def build_term():
    solve = leaf(
        template="Solve this task. Provide an answer and brief rationale.\nTask: {task}",
        input_vars=("task",),
        schema_ref=SCHEMA_ATTEMPT,
    )
    evaluate = leaf(
        template=(
            "Evaluate the attempt. Provide quality_score 0..1, feedback, "
            "and set passed=true iff quality >= 0.7.\nTask: {task}\n"
            "Attempt: {attempt1}"
        ),
        input_vars=("task", "attempt1"),
        schema_ref=SCHEMA_EVAL,
    )
    reflect = leaf(
        template=(
            "Reflect on the attempt and feedback. Provide a lesson and strategy.\n"
            "Task: {task}\nAttempt: {attempt1}\nEvaluation: {evaluation}"
        ),
        input_vars=("task", "attempt1", "evaluation"),
        schema_ref=SCHEMA_REFLECT,
    )
    re_solve = leaf(
        template=(
            "Solve the task again, applying the lesson and strategy.\n"
            "Task: {task}\nReflection: {reflection}"
        ),
        input_vars=("task", "reflection"),
        schema_ref=SCHEMA_FINAL,
    )
    return let_(
        "attempt1", solve,
        let_("evaluation", evaluate,
            let_("reflection", reflect, re_solve),
        ),
    )


def checks(result, error, oracle_calls):
    ok = isinstance(result, dict) and len(str(result.get("answer", ""))) > 1
    return {
        "answer_present": ok,
        "iterations_ok": oracle_calls >= 1,
        "reflexion_loop_completed": error is None and oracle_calls >= 4,
    }


def main():
    return run_pipeline(build_term(), {"task": TASK}, checks_fn=checks, title='Reflexion (λ-DSL)')


if __name__ == "__main__":
    main()
