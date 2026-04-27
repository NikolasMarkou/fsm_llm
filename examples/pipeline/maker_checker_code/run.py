"""
Maker-Checker-Code Pipeline — λ-DSL twin (3-leaf: write code, review, revise).

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/maker_checker_code/run.py
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

SCHEMA_D = "examples.pipeline.maker_checker_code.schemas.CodeDraft"
SCHEMA_R = "examples.pipeline.maker_checker_code.schemas.CodeReview"
SCHEMA_F = "examples.pipeline.maker_checker_code.schemas.CodeFinal"

TASK = (
    "Implement a thread-safe LRU (Least Recently Used) cache class in Python. "
    "It should support get and put operations with O(1) time complexity, "
    "automatically evict the least recently used item when capacity is reached, "
    "and be safe for concurrent access from multiple threads."
)
MAKE_INSTR = (
    "Write a Python class that implements a thread-safe LRU cache. Use "
    "OrderedDict + threading.Lock. Include type hints and docstring. "
    "Keep it under 40 lines."
)
CHECK_INSTR = (
    "Review the code: correctness, thread safety, type hints, docstring, "
    "code quality, edge cases. Score 0..1; checker_passed=true iff "
    "quality_score >= 0.7."
)


def build_term() -> Any:
    make = leaf(
        template=(
            f"Instructions: {MAKE_INSTR}\nTask: {{task}}\n"
            "Return JSON matching CodeDraft (code, rationale)."
        ),
        input_vars=("task",),
        schema_ref=SCHEMA_D,
    )
    check = leaf(
        template=(
            f"Instructions: {CHECK_INSTR}\nDraft (JSON): {{draft}}\n"
            "Return JSON matching CodeReview."
        ),
        input_vars=("draft",),
        schema_ref=SCHEMA_R,
    )
    revise = leaf(
        template=(
            "Revise the code based on the review.\nTask: {task}\n"
            "Draft (JSON): {draft}\nReview (JSON): {review}\n"
            "Return JSON matching CodeFinal."
        ),
        input_vars=("task", "draft", "review"),
        schema_ref=SCHEMA_F,
    )
    return let_("draft", make, let_("review", check, revise))


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print(f"Task: {TASK[:80]}...\nModel: {model}")
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
        code = final.get("code", "")
        print(f"\nFinal code ({len(code)} chars):\n{code[:400]}")
        print(f"Quality score: {final.get('quality_score')}")
        print(f"Oracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    code = (final or {}).get("code", "") if isinstance(final, dict) else ""
    checks = {
        "answer_present": (error is None and isinstance(code, str) and len(code) > 50),
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
