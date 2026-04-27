"""
Eval-Opt Structured Pipeline — λ-DSL twin
==========================================

3-leaf pipeline producing a structured recipe with one refinement pass.

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/eval_opt_structured/run.py
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

SCHEMA_R = "examples.pipeline.eval_opt_structured.schemas.Recipe"
SCHEMA_E = "examples.pipeline.eval_opt_structured.schemas.Eval"
SCHEMA_F = "examples.pipeline.eval_opt_structured.schemas.RefinedRecipe"

TASK = (
    "Create a detailed recipe for homemade pasta carbonara. "
    "Be specific with quantities and timing."
)


def build_term() -> Any:
    candidate = leaf(
        template=(
            "You are a chef.\nTask: {task}\n"
            "Return JSON matching Recipe (name, cuisine, prep_time_minutes, "
            "cook_time_minutes, servings, difficulty, ingredients, steps, "
            "nutritional_notes)."
        ),
        input_vars=("task",),
        schema_ref=SCHEMA_R,
    )
    ev = leaf(
        template=(
            "Evaluate the recipe for completeness, accuracy, and clarity. "
            "Score 0..1; passed=true iff score >= 0.7.\n"
            "Recipe (JSON): {recipe}\n"
            "Return JSON matching Eval."
        ),
        input_vars=("recipe",),
        schema_ref=SCHEMA_E,
    )
    refine = leaf(
        template=(
            "Refine the recipe based on the evaluation.\n"
            "Task: {task}\n"
            "Recipe (JSON): {recipe}\n"
            "Eval (JSON): {evaluation}\n"
            "Return JSON matching RefinedRecipe (all Recipe fields plus a "
            "'changes' list)."
        ),
        input_vars=("task", "recipe", "evaluation"),
        schema_ref=SCHEMA_F,
    )
    return let_(
        "recipe",
        candidate,
        let_("evaluation", ev, refine),
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("=" * 60)
    print("Evaluator-Optimizer with Structured Output (λ-DSL)")
    print("=" * 60)
    print(f"Model: {model}\nTask: {TASK[:80]}...")
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
        print(f"\nName: {final.get('name')}")
        print(f"Ingredients: {len(final.get('ingredients', []))}")
        print(f"Steps: {len(final.get('steps', []))}")
        print(f"Oracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    name = (final or {}).get("name", "") if isinstance(final, dict) else ""
    ingredients = (
        (final or {}).get("ingredients", []) if isinstance(final, dict) else []
    )
    checks = {
        "answer_present": (error is None and isinstance(name, str) and len(name) > 0),
        "ingredients_present": isinstance(ingredients, list) and len(ingredients) >= 2,
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
