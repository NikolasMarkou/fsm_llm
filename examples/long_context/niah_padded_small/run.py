"""
NIAH Padded Small — non-aligned input at smaller τ
=====================================================

Demonstrates ``fsm_llm.stdlib.long_context.niah_padded`` (M5 slice 4) at
a smaller scale than niah_padded_demo: τ=128 (vs 256), k=2, raw n=400
(non-aligned, between 256=τ·k and 512=τ·k^2). Padded N*=512, d=2 →
predicted_calls = k^d = 4.

Theorem-2 holds against the PADDED size N*, not the raw n.
"""

import os
import sys

from fsm_llm.lam import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.long_context import (
    aligned_size,
    best_answer_op,
    make_pad_callable,
    make_size_bucket,
    niah_padded,
)

NEEDLE = "ACCESS_CODE: CHARLIE-400"
DOC_LEN = 400  # deliberately UN-aligned (between τ·k=256 and τ·k^2=512)
TAU = 128
K = 2
NEEDLE_OFFSET = 128  # chunk-aligned in the padded doc


def build_haystack() -> str:
    filler = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
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

    n_star = aligned_size(DOC_LEN, TAU, K)
    print(f"Model: {model}")
    print(f"Raw doc length: {DOC_LEN} chars (UNALIGNED)")
    print(f"Padded length N* = {n_star}")
    print(f"τ={TAU}; k={K}")
    print("-" * 60)

    haystack = build_haystack()
    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    program = niah_padded(
        "What is the access code? Reply with just the code value.",
        tau=TAU,
        k=K,
    )

    try:
        result = ex.run(
            program,
            {
                "raw_document": haystack,
                "pad_to_aligned": make_pad_callable(TAU, K),
                "size_bucket": make_size_bucket(TAU),
                "best": best_answer_op(),
            },
        )
    except Exception as e:
        print(f"Oracle error: {e}")
        return 1

    predicted = plan(PlanInputs(n=n_star, K=10_000, tau=TAU, alpha=1.0, max_k=K))

    print(f"\nFinal answer: {result!r}")
    print(f"Oracle calls (actual): {ex.oracle_calls}")
    print(f"Oracle calls (predicted on N*): {predicted.predicted_calls}")
    print(f"Plan: k*={predicted.k_star}, d={predicted.d}")

    needle_found = isinstance(result, str) and "CHARLIE-400" in result
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
