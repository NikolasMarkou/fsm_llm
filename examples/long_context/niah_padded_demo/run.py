"""
NIAH-Padded Demo — Needle-in-Haystack on non-τ·k^d-aligned input
=================================================================

Demonstrates ``fsm_llm.stdlib.long_context.niah_padded`` (M5 slice 4 of
``docs/lambda.md``): the padded variant of ``niah`` that pre-pads the raw
document up to ``N* = aligned_size(n, τ, k)`` so the planner-executor
cost-equality contract holds for ARBITRARY user input length ``n``.

The demo synthesises a 2000-character haystack (deliberately unaligned —
1024 < 2000 < 2048 = τ·k^3 with τ=256, k=2) with a single needle at
offset 1024, runs ``niah_padded`` (which internally pads to 2048), and
verifies:

1. ``needle_found`` — the model recovered the secret from the right
   chunk. (Best-effort: padded chunks are whitespace; small models may
   emit prose instead of NOT_FOUND. This is an empirical risk; theorem-2
   below is mechanical.)
2. ``oracle_calls_match_planner`` — ``ex.oracle_calls`` exactly equals
   ``plan(PlanInputs(n=N*, ...)).predicted_calls`` against the PADDED
   size N*=2048.

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/long_context/niah_padded_demo/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/long_context/niah_padded_demo/run.py
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

NEEDLE = "ACCESS_CODE: SECRET-7421"
DOC_LEN = 2000  # deliberately UN-aligned (between τ·k^2=1024 and τ·k^3=2048)
TAU = 256
K = 2
NEEDLE_OFFSET = 1024


def build_haystack() -> str:
    """Synthesise a 2000-char (unaligned) document with the needle at offset 1024."""
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
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return 1

    n_star = aligned_size(DOC_LEN, TAU, K)
    print(f"Model: {model}")
    print(f"Raw doc length: {DOC_LEN} chars (UNALIGNED)")
    print(f"Padded length N* = aligned_size({DOC_LEN}, {TAU}, {K}) = {n_star}")
    print(f"τ={TAU}; k={K}")
    print(f"Needle: {NEEDLE!r} at offset {NEEDLE_OFFSET}")
    print("-" * 60)

    haystack = build_haystack()

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    program = niah_padded(
        "What is the access code? Reply with just the code value, e.g. 'SECRET-XXXX'.",
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

    # Note: planner takes n=N* (padded size), NOT the raw DOC_LEN.
    predicted = plan(PlanInputs(n=n_star, K=10_000, tau=TAU, alpha=1.0, max_k=K))

    print(f"\nFinal answer: {result!r}")
    print(f"Oracle calls (actual): {ex.oracle_calls}")
    print(f"Oracle calls (predicted on N*): {predicted.predicted_calls}")
    print(
        f"Plan: k*={predicted.k_star}, d={predicted.d}, "
        f"accuracy_floor={predicted.accuracy_floor:.3f}"
    )
    print(
        f"Padding overhead: N*/n = {n_star / DOC_LEN:.3f}x "
        f"({n_star - DOC_LEN} pad chars)"
    )

    needle_found = isinstance(result, str) and "SECRET-7421" in result
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

    # Theorem-2 (cost-equality) is the hard contract; needle_found is
    # best-effort (small models may return prose on whitespace chunks).
    return 0 if calls_match else 1


if __name__ == "__main__":
    sys.exit(main())
