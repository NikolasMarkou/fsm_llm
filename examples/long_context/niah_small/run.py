"""
NIAH Small — Smaller (τ·k^2) variant of niah_demo
====================================================

Demonstrates ``fsm_llm.stdlib.long_context.niah`` (M5 slice 1) at a smaller
scale than ``niah_demo`` (which uses τ·k^3 = 2048): here τ=256, k=2, d=2,
so n = τ·k^d = 1024 and predicted_calls = k^d = 4.

Verifies:

1. ``needle_found`` — best-effort heuristic.
2. ``oracle_calls_match_planner`` — strict Theorem-2 gate.
"""

import os
import sys

from fsm_llm.runtime import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.long_context import best_answer_op, make_size_bucket, niah

NEEDLE = "ACCESS_CODE: ALPHA-1024"
DOC_LEN = 1024
TAU = 256
K = 2
NEEDLE_OFFSET = 512  # chunk-aligned


def build_haystack() -> str:
    filler = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse. "
    )
    body = (filler * ((DOC_LEN // len(filler)) + 2))[:DOC_LEN]
    needle_padded = " " + NEEDLE + " "
    doc = (
        body[:NEEDLE_OFFSET]
        + needle_padded
        + body[NEEDLE_OFFSET + len(needle_padded) :]
    )[:DOC_LEN]
    assert len(doc) == DOC_LEN
    assert NEEDLE in doc
    return doc


def main() -> int:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return 1

    print(f"Model: {model}")
    print(f"Doc length: {DOC_LEN} chars; τ={TAU}; k={K}")
    print(f"Needle: {NEEDLE!r} at offset {NEEDLE_OFFSET}")
    print("-" * 60)

    haystack = build_haystack()
    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    program = niah(
        "What is the access code? Reply with just the code value, e.g. 'ALPHA-XXXX'.",
        tau=TAU,
        k=K,
    )

    try:
        result = ex.run(
            program,
            {
                "document": haystack,
                "size_bucket": make_size_bucket(TAU),
                "best": best_answer_op(),
            },
        )
    except Exception as e:
        print(f"Oracle error: {e}")
        return 1

    predicted = plan(PlanInputs(n=DOC_LEN, K=10_000, tau=TAU, alpha=1.0, max_k=K))

    print(f"\nFinal answer: {result!r}")
    print(f"Oracle calls (actual): {ex.oracle_calls}")
    print(f"Oracle calls (predicted): {predicted.predicted_calls}")
    print(f"Plan: k*={predicted.k_star}, d={predicted.d}")

    needle_found = isinstance(result, str) and "ALPHA-1024" in result
    calls_match = ex.oracle_calls == predicted.predicted_calls

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "needle_found": needle_found,
        "oracle_calls_match_planner": calls_match,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:30s}: {passed!s:10s} [{status}]")
    pct = 100 * extracted / len(checks)
    print(f"\nExtraction rate: {extracted}/{len(checks)} ({pct:.0f}%)")

    return 0 if calls_match else 1


if __name__ == "__main__":
    sys.exit(main())
