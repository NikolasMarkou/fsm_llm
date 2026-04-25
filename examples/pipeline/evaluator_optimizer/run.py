"""
Evaluator-Optimizer Pipeline — λ-DSL twin
==========================================

3-leaf pipeline: candidate → eval → refined.

Note: original ``EvaluatorOptimizerAgent`` runs a fix-loop until passed
or max_refinements. λ-DSL twin runs a single refinement pass; preserves
UX. Migrating to ``fix(...)`` is a future M5 task.

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/evaluator_optimizer/run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fsm_llm.lam import Executor, LiteLLMOracle, leaf, let_
from fsm_llm.llm import LiteLLMInterface

SCHEMA_C = "examples.pipeline.evaluator_optimizer.schemas.Candidate"
SCHEMA_E = "examples.pipeline.evaluator_optimizer.schemas.Eval"
SCHEMA_R = "examples.pipeline.evaluator_optimizer.schemas.Refined"

TASK = "Write a haiku about autumn leaves falling from trees."
EVAL_RUBRIC = (
    "Evaluate the haiku for: 5-7-5 syllable structure, vivid imagery, "
    "and emotional resonance. Score 0..1; passed=true iff score >= 0.7."
)


def build_term() -> Any:
    cand = leaf(
        template=(
            "You are a poet.\nTask: {task}\n"
            "Return JSON matching Candidate schema (text)."
        ),
        input_vars=("task",),
        schema_ref=SCHEMA_C,
    )
    ev = leaf(
        template=(
            f"Rubric: {EVAL_RUBRIC}\n"
            "Candidate (JSON): {candidate}\n"
            "Return JSON matching Eval schema."
        ),
        input_vars=("candidate",),
        schema_ref=SCHEMA_E,
    )
    refine = leaf(
        template=(
            "You refine the candidate based on the evaluator's feedback.\n"
            "Task: {task}\n"
            "Candidate (JSON): {candidate}\n"
            "Evaluation (JSON): {evaluation}\n"
            "Return JSON matching Refined schema (text, changes, score)."
        ),
        input_vars=("task", "candidate", "evaluation"),
        schema_ref=SCHEMA_R,
    )
    return let_(
        "candidate", cand,
        let_("evaluation", ev, refine),
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print(f"Task: {TASK}\nModel: {model}\nMax refinements: 1 (λ-twin approximation)")
    print("-" * 60)

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    error: Exception | None = None
    final: dict[str, Any] | None = None
    try:
        final = ex.run(build_term(), {"task": TASK})
    except Exception as e:
        error = e
        print(f"Error: {e}")

    if final is not None:
        print(f"\nFinal:\n{final.get('text', '')}")
        print(f"Score: {final.get('score')}")
        print(f"Oracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    text = (final or {}).get("text", "") if isinstance(final, dict) else ""
    checks = {
        "answer_present": (
            error is None and isinstance(text, str) and len(text) > 5
        ),
        "iterations_ok": ex.oracle_calls >= 1,
        "pipeline_completed": ex.oracle_calls >= 3,
    }
    extracted = sum(1 for v in checks.values() if v)
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        print(f"  {key:25s}: {passed!s:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} "
        f"({100 * extracted // len(checks)}%)"
    )


if __name__ == "__main__":
    main()
