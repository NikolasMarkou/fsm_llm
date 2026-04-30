"""
Maker-Checker Pipeline — λ-DSL twin
=====================================

3-leaf pipeline: make → check → revise.

Note: the original ``MakerCheckerAgent`` runs a fix-loop (revise up to N
times until quality_threshold). The λ-DSL twin runs a single revision pass
unconditionally; this preserves UX (3 oracle calls, structured output) at
the cost of ignoring the fix-loop early-exit. A future M5 slice can use
``fix(...)`` for the bounded recursion.

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/maker_checker/run.py
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

SCHEMA_DRAFT = "examples.pipeline.maker_checker.schemas.Draft"
SCHEMA_REVIEW = "examples.pipeline.maker_checker.schemas.Review"
SCHEMA_FINAL = "examples.pipeline.maker_checker.schemas.Final"

TASK = (
    "Write an email to a client apologizing for a 2-day delay in "
    "delivering their project and proposing a new timeline."
)
MAKE_INSTR = (
    "Write a professional email that is concise (under 150 words), "
    "uses a warm but professional tone, includes a clear subject line, "
    "and ends with a specific call to action."
)
CHECK_INSTR = (
    "Evaluate the email against these criteria: professional tone, "
    "conciseness (under 150 words), clear call to action, proper greeting "
    "and sign-off, no spelling/grammar issues. Score 0..1; checker_passed=true "
    "iff quality_score >= 0.7."
)


def build_term() -> Any:
    make_leaf = leaf(
        template=(
            f"Instructions: {MAKE_INSTR}\n"
            "Task: {task}\n"
            "Return JSON matching Draft (text, rationale)."
        ),
        input_vars=("task",),
        schema_ref=SCHEMA_DRAFT,
    )
    check_leaf = leaf(
        template=(
            f"Instructions: {CHECK_INSTR}\n"
            "Draft to evaluate (JSON): {draft}\n"
            "Return JSON matching Review (quality_score, feedback, "
            "checker_passed)."
        ),
        input_vars=("draft",),
        schema_ref=SCHEMA_REVIEW,
    )
    revise_leaf = leaf(
        template=(
            "You revise the draft using the reviewer's feedback.\n"
            "Original task: {task}\n"
            "Draft (JSON): {draft}\n"
            "Review (JSON): {review}\n"
            "Return JSON matching Final (text, revisions_made, quality_score)."
        ),
        input_vars=("task", "draft", "review"),
        schema_ref=SCHEMA_FINAL,
    )
    return let(
        "draft",
        make_leaf,
        let("review", check_leaf, revise_leaf),
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print(f"Task: {TASK}\nModel: {model}")
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
        print(f"\nFinal:\n{final.get('text', '')[:500]}")
        print(f"Quality score: {final.get('quality_score')}")
        print(f"Oracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    text = (final or {}).get("text", "") if isinstance(final, dict) else ""
    checks = {
        "answer_present": (error is None and isinstance(text, str) and len(text) > 10),
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
