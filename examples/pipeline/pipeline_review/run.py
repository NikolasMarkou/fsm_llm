"""
Pipeline-Review — λ-DSL twin (3-stage chain + 2-stage review = 5 leaves).

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/pipeline_review/run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fsm_llm.llm import LiteLLMInterface
from fsm_llm.runtime import Executor, LiteLLMOracle, leaf, let

S_O = "examples.pipeline.pipeline_review.schemas.Outline"
S_D = "examples.pipeline.pipeline_review.schemas.Draft"
S_P = "examples.pipeline.pipeline_review.schemas.Polished"
S_R = "examples.pipeline.pipeline_review.schemas.Review"
S_F = "examples.pipeline.pipeline_review.schemas.Final"

TASK = (
    "Write API documentation for a REST API that manages a todo list "
    "application with CRUD operations, user authentication via JWT tokens, "
    "and pagination support."
)


def build_term() -> Any:
    outline = leaf(
        template=(
            "You are a technical writer.\nTask: {task}\n"
            "Produce an outline. Return JSON matching Outline (sections, summary)."
        ),
        input_vars=("task",),
        schema_ref=S_O,
    )
    draft = leaf(
        template=(
            "Write a draft from the outline.\nTask: {task}\n"
            "Outline (JSON): {outline}\n"
            "Return JSON matching Draft (text, word_count)."
        ),
        input_vars=("task", "outline"),
        schema_ref=S_D,
    )
    polish = leaf(
        template=(
            "Polish the draft for clarity and structure.\n"
            "Draft (JSON): {draft}\n"
            "Return JSON matching Polished (text, improvements)."
        ),
        input_vars=("draft",),
        schema_ref=S_P,
    )
    review = leaf(
        template=(
            "Review the polished document. Score 0..1; checker_passed=true "
            "iff quality_score >= 0.7.\n"
            "Polished (JSON): {polished}\n"
            "Return JSON matching Review."
        ),
        input_vars=("polished",),
        schema_ref=S_R,
    )
    final = leaf(
        template=(
            "Produce the final document incorporating any review feedback.\n"
            "Polished (JSON): {polished}\nReview (JSON): {review}\n"
            "Return JSON matching Final (text, review)."
        ),
        input_vars=("polished", "review"),
        schema_ref=S_F,
    )
    return let(
        "outline",
        outline,
        let("draft", draft, let("polished", polish, let("review", review, final))),
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("=" * 60)
    print("Pipeline + Review (λ-DSL)")
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
        text = final.get("text", "")
        print(f"\nFinal:\n{text[:600]}")
        print(f"\nOracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    text = (final or {}).get("text", "") if isinstance(final, dict) else ""
    checks = {
        "answer_present": (error is None and isinstance(text, str) and len(text) > 50),
        "iterations_ok": ex.oracle_calls >= 1,
        "pipeline_completed": ex.oracle_calls >= 5,
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
