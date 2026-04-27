"""
Multi-Hop 3 — 3-hop retrieval over a long document
=====================================================

Demonstrates ``fsm_llm.stdlib.long_context.multi_hop`` with hops=3
(existing multi_hop_demo uses hops=2). Each hop is a niah-shaped Fix
call; per Theorem-2: oracle_calls == hops · k^d.

τ=256, k=2, n=1024 → d=2, single-hop predicted=4, 3 hops → 12 total.
"""

import os
import sys

from fsm_llm.lam import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.long_context import (
    best_answer_op,
    make_size_bucket,
    multi_hop,
)

DOC_LEN = 1024
TAU = 256
K = 2
HOPS = 3

# Three planted facts to support a 3-hop chain.
ENTITY_NAME = "Project Vesper"
TEAM_NAME = "the Astrolab team"
LAUNCH_DATE = "March 7, 2026"

CHUNKS: list[str] = [
    " Internal note: The flagship initiative is "
    + ENTITY_NAME
    + ", a 2026 strategic effort. ",
    " Org chart: " + ENTITY_NAME + " is led by " + TEAM_NAME + ". ",
    " Release log: " + TEAM_NAME + " launched the program on "
    + LAUNCH_DATE + " after final review. ",
    " Filler: Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
]


def build_haystack() -> str:
    """Synthesise a 1024-char doc with 4 chunk-aligned 256-byte chunks."""
    assert len(CHUNKS) == DOC_LEN // TAU == 4
    chunks: list[str] = []
    for c in CHUNKS:
        if len(c) >= TAU:
            chunk = c[:TAU]
        else:
            chunk = c + " " * (TAU - len(c))
        chunks.append(chunk)
    doc = "".join(chunks)
    assert len(doc) == DOC_LEN
    assert ENTITY_NAME in doc
    assert TEAM_NAME in doc
    assert LAUNCH_DATE in doc
    return doc


def main() -> int:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return 1

    print(f"Model: {model}")
    print(f"Doc length: {DOC_LEN} chars; τ={TAU}; k={K}; hops={HOPS}")
    print("-" * 60)

    haystack = build_haystack()
    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    program = multi_hop(
        question="What is the launch date of the flagship initiative?",
        hops=HOPS,
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
    expected_calls = HOPS * predicted.predicted_calls

    print(f"\nFinal result: {result!r}")
    print(f"Oracle calls (actual): {ex.oracle_calls}")
    print(f"Oracle calls (per hop predicted): {predicted.predicted_calls}")
    print(f"Oracle calls (total expected = hops*per-hop): {expected_calls}")

    result_str = result if isinstance(result, str) else ""
    launch_date_found = "March 7" in result_str or "03-07" in result_str
    calls_match = ex.oracle_calls == expected_calls

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "oracle_calls_match_3hop_planner": calls_match,
        "launch_date_found": launch_date_found,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:32s}: {passed!s:10s} [{status}]")
    pct = 100 * extracted / len(checks)
    print(f"\nExtraction rate: {extracted}/{len(checks)} ({pct:.0f}%)")

    return 0 if calls_match else 1


if __name__ == "__main__":
    sys.exit(main())
