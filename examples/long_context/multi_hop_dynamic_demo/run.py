"""
Multi-Hop Dynamic — confidence-gated dynamic-hop retrieval (standalone)
=========================================================================

Demonstrates ``fsm_llm.stdlib.long_context.multi_hop_dynamic`` (M5 slice 6)
as a standalone example. Existing multi_hop_demo invokes dynamic mode
behind a ``--dynamic`` flag; this is the zero-arg version.

τ=256, k=2, n=1024 → d=2, single-hop predicted=4. With max_hops=4 and
the default ``not_found_gate``, the runner stops as soon as a hop returns
non-NOT_FOUND. Strict T2: oracle_calls == actual_hops · k^d. Loose
upper bound: ≤ max_hops · k^d.
"""

import os
import sys

from fsm_llm.llm import LiteLLMInterface
from fsm_llm.runtime import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.stdlib.long_context import (
    best_answer_op,
    make_dynamic_hop_runner,
    make_size_bucket,
    multi_hop_dynamic,
)

DOC_LEN = 1024
TAU = 256
K = 2
MAX_HOPS = 4

ANSWER_PHRASE = "the password is OPAL-4421"


def build_haystack() -> str:
    """1024-char doc with the answer planted in chunk 1 (offset 256-512)."""
    chunks: list[str] = [
        " Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        + "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ",
        " Internal note: "
        + ANSWER_PHRASE
        + " for the staging environment. "
        + "Lorem ipsum dolor sit amet. ",
        " Filler: Sed do eiusmod tempor. " + "Ut enim ad minim veniam. " * 3,
        " Filler: Duis aute irure dolor in reprehenderit. "
        + "Excepteur sint occaecat. " * 4,
    ]
    out = []
    for c in chunks:
        if len(c) >= TAU:
            out.append(c[:TAU])
        else:
            out.append(c + " " * (TAU - len(c)))
    doc = "".join(out)
    assert len(doc) == DOC_LEN
    assert ANSWER_PHRASE in doc
    return doc


def main() -> int:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return 1

    print(f"Model: {model}")
    print(f"Doc length: {DOC_LEN} chars; τ={TAU}; k={K}; max_hops={MAX_HOPS}")
    print("-" * 60)

    haystack = build_haystack()
    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    actual_hops_cell = [0]
    question = "What is the staging-environment password?"
    runner = make_dynamic_hop_runner(
        ex,
        question,
        max_hops=MAX_HOPS,
        peer_env={
            "size_bucket": make_size_bucket(TAU),
            "best": best_answer_op(),
        },
        tau=TAU,
        k=K,
        actual_hops_cell=actual_hops_cell,
    )
    program = multi_hop_dynamic(question, max_hops=MAX_HOPS)
    env = {"document": haystack, "dynamic_hop_runner": runner}

    try:
        result = ex.run(program, env)
    except Exception as e:
        print(f"Oracle error: {e}")
        return 1

    predicted = plan(PlanInputs(n=DOC_LEN, K=10_000, tau=TAU, alpha=1.0, max_k=K))
    actual_hops = actual_hops_cell[0]
    expected_calls = actual_hops * predicted.predicted_calls
    upper_bound = MAX_HOPS * predicted.predicted_calls

    print(f"\nFinal result: {result!r}")
    print(f"Oracle calls (actual): {ex.oracle_calls}")
    print(f"Oracle calls (predicted, per hop): {predicted.predicted_calls}")
    print(f"Actual hops run: {actual_hops} / max_hops={MAX_HOPS}")
    print(f"Strict expected (actual_hops*per-hop): {expected_calls}")
    print(f"Upper bound (max_hops*per-hop): {upper_bound}")

    result_str = result if isinstance(result, str) else ""
    answer_found = "OPAL-4421" in result_str
    calls_match = ex.oracle_calls == expected_calls
    upper_bound_holds = ex.oracle_calls <= upper_bound

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "oracle_calls_match_dynamic_planner": calls_match,
        "oracle_calls_within_upper_bound": upper_bound_holds,
        "answer_found": answer_found,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:35s}: {passed!s:10s} [{status}]")
    pct = 100 * extracted / len(checks)
    print(f"\nExtraction rate: {extracted}/{len(checks)} ({pct:.0f}%)")

    # Strict T2 is the hard gate; upper-bound is a sanity check.
    return 0 if calls_match else 1


if __name__ == "__main__":
    sys.exit(main())
