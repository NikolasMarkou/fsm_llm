"""
NIAH Demo — Needle-in-Haystack via the λ-substrate
===================================================

Demonstrates ``fsm_llm.stdlib.long_context.niah`` (M5 slice 1 of
``docs/lambda.md``): a recursive ``fix + split + fmap(self) + reduce_(best)``
λ-term that finds a "needle" answer in a large document by chunking it
into tractable pieces and reducing the per-chunk answers.

The demo synthesises a 2048-character haystack with a single needle
("ACCESS_CODE: SECRET-7421") at a known offset, runs the ``niah`` factory
with τ=256 (8 leaves under k=2), and verifies two invariants:

1. ``needle_found`` — the model recovered the secret from the right chunk.
2. ``oracle_calls_match_planner`` — ``ex.oracle_calls`` exactly equals
   ``plan(PlanInputs(...)).predicted_calls`` (paper Theorem 2 — pre-computable
   cost).

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/long_context/niah_demo/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/long_context/niah_demo/run.py
"""

import os
import sys

from fsm_llm.runtime import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.long_context import best_answer_op, make_size_bucket, niah

NEEDLE = "ACCESS_CODE: SECRET-7421"
DOC_LEN = 2048
TAU = 256
K = 2
NEEDLE_OFFSET = 1024  # chunk-aligned: doc[1024:1280] under recursive halving


def build_haystack() -> str:
    """Synthesise a 2048-char document with the needle at offset 1024."""
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

    print(f"Model: {model}")
    print(f"Doc length: {DOC_LEN} chars; τ={TAU}; k={K}")
    print(f"Needle: {NEEDLE!r} at offset {NEEDLE_OFFSET}")
    print("-" * 60)

    haystack = build_haystack()

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    program = niah(
        "What is the access code? Reply with just the code value, e.g. 'SECRET-XXXX'.",
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
    print(
        f"Plan: k*={predicted.k_star}, d={predicted.d}, "
        f"accuracy_floor={predicted.accuracy_floor:.3f}"
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

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
