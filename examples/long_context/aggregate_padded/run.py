"""
Aggregate Padded — Compose pad_to_aligned helper with aggregate factory
=========================================================================

Demonstrates composing the niah_padded slice-4 padding helper (a host
callable bound via env) with the slice-2 ``aggregate`` factory using
inline kernel DSL: ``let_("document", app(var("pad_to_aligned"),
var("raw_document")), aggregate_term)``.

This shows the substrate's combinator chain: a host callable lifts a
non-aligned input to N* before passing it to an aligned-only factory,
all expressed in the λ-DSL without modifying either factory.

τ=512, k=2, raw n=1500 → N*=2048, d=2 → predicted_calls = k^d = 4.
"""

import os
import sys

from fsm_llm.llm import LiteLLMInterface
from fsm_llm.runtime import Executor, LiteLLMOracle, PlanInputs, app, let_, plan, var
from fsm_llm.stdlib.long_context import (
    aggregate,
    aggregate_op,
    aligned_size,
    make_pad_callable,
    make_size_bucket,
)

DOC_LEN = 1500  # UNALIGNED (between τ·k=1024 and τ·k^2=2048)
TAU = 512
K = 2

# Three topical sentences placed in the raw doc.
TOPICS = [
    "The product codename is HALYARD.",
    "The owning team is the Spire division.",
    "The first deployment was in Q2 2026.",
]
TOPIC_OFFSETS = [50, 600, 1100]


def build_haystack() -> str:
    filler = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse. "
    )
    body = list((filler * ((DOC_LEN // len(filler)) + 2))[:DOC_LEN])
    for offset, sentence in zip(TOPIC_OFFSETS, TOPICS, strict=True):
        sentence_padded = " " + sentence + " "
        for i, ch in enumerate(sentence_padded):
            if offset + i < DOC_LEN:
                body[offset + i] = ch
    doc = "".join(body)[:DOC_LEN]
    assert len(doc) == DOC_LEN
    for sentence in TOPICS:
        assert sentence in doc, f"missing topic: {sentence}"
    return doc


def main() -> int:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return 1

    n_star = aligned_size(DOC_LEN, TAU, K)
    print(f"Model: {model}")
    print(f"Raw doc: {DOC_LEN} chars (UNALIGNED) → padded N*={n_star}")
    print(f"τ={TAU}; k={K}")
    print(f"Topics seeded: {len(TOPICS)} at offsets {TOPIC_OFFSETS}")
    print("-" * 60)

    haystack = build_haystack()
    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    # Compose: bind 'document' = pad_to_aligned(raw_document), then run aggregate.
    aggregate_term = aggregate(
        "What facts does this document state about the product? "
        "(name, team, deployment, etc.)",
        tau=TAU,
        k=K,
    )
    program = let_(
        "document",
        app(var("pad_to_aligned"), var("raw_document")),
        aggregate_term,
    )

    try:
        result = ex.run(
            program,
            {
                "raw_document": haystack,
                "pad_to_aligned": make_pad_callable(TAU, K),
                "size_bucket": make_size_bucket(TAU),
                "merge": aggregate_op(),
            },
        )
    except Exception as e:
        print(f"Oracle error: {e}")
        return 1

    predicted = plan(PlanInputs(n=n_star, K=10_000, tau=TAU, alpha=1.0, max_k=K))

    print(f"\nFinal aggregate output:\n{result}\n")
    print(f"Oracle calls (actual): {ex.oracle_calls}")
    print(f"Oracle calls (predicted on N*): {predicted.predicted_calls}")
    print(f"Plan: k*={predicted.k_star}, d={predicted.d}")

    output_nontrivial = (
        isinstance(result, str) and len(result) > 30 and result != "NOT_FOUND"
    )
    calls_match = ex.oracle_calls == predicted.predicted_calls

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "oracle_calls_match_planner": calls_match,
        "output_nontrivial": output_nontrivial,
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
