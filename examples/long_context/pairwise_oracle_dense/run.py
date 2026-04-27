"""
Pairwise Oracle (Dense) — Standalone oracle-mediated tournament demo
======================================================================

Demonstrates ``fsm_llm.stdlib.long_context.pairwise`` (M5 slice 3) with
``oracle_compare_op`` (M5 slice 5) at τ=256, k=2, n=1024, d=2 →
predicted_calls = 2*k^d - 1 = 7 (4 leaf calls + 3 oracle-mediated
reduce comparisons).

The existing pairwise_demo invokes oracle mode behind a ``--mode oracle``
flag. This example is the standalone/zero-arg version of the slice-5
contract: every chunk carries a distinct factual sentence so no reduce
arm short-circuits on a sentinel (D-S5-001 caveat).
"""

import os
import sys

from fsm_llm.lam import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.long_context import (
    make_size_bucket,
    oracle_compare_op,
    pairwise,
)

DOC_LEN = 1024
TAU = 256
K = 2

# Four distinct factual sentences — one per chunk-aligned 256-byte chunk.
# Topic A (target) is the marine-biology fact; the other three are decoys
# from unrelated domains so the leaf returns a non-sentinel for each.
TOPIC_A_PHRASE = "deep-sea hydrothermal vents host chemosynthetic tubeworms"

DENSE_CHUNKS: tuple[str, ...] = (
    " Astronomy fact: red giant stars expand dramatically as their "
    "hydrogen cores deplete and helium fusion begins. ",
    # Topic A — chunk 1 (target).
    " Marine biology fact: " + TOPIC_A_PHRASE + " that derive energy "
    "from sulfide-rich fluids vented at oceanic spreading ridges. ",
    " Linguistics fact: the Indo-European language family spans most of "
    "Europe and parts of southern and central Asia. ",
    " Cryptography fact: elliptic-curve signatures rest on the discrete "
    "logarithm problem over additively-written groups. ",
)


def build_haystack() -> str:
    """Synthesise a 1024-char doc with 4 distinct topical chunks."""
    assert len(DENSE_CHUNKS) == DOC_LEN // TAU == 4
    chunks: list[str] = []
    for topic in DENSE_CHUNKS:
        if len(topic) >= TAU:
            chunk = topic[:TAU]
        else:
            chunk = topic + " " * (TAU - len(topic))
        chunks.append(chunk)
    doc = "".join(chunks)
    assert len(doc) == DOC_LEN
    assert TOPIC_A_PHRASE in doc, "missing topic A phrase"
    return doc


def main() -> int:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return 1

    print(f"Model: {model}")
    print(f"Doc length: {DOC_LEN} chars; τ={TAU}; k={K}; mode=oracle (slice 5)")
    print(f"Target phrase: {TOPIC_A_PHRASE!r}")
    print("-" * 60)

    haystack = build_haystack()
    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    leaf_question = "What single factual statement is asserted in this passage?"
    compare_question = (
        "Which segment discusses Topic A (deep-sea hydrothermal vents "
        "and chemosynthetic marine biology) more directly?"
    )

    program = pairwise(question=leaf_question, tau=TAU, k=K)
    compare = oracle_compare_op(compare_question, ex)

    try:
        result = ex.run(
            program,
            {
                "document": haystack,
                "size_bucket": make_size_bucket(TAU),
                "compare": compare,
            },
        )
    except Exception as e:
        print(f"Oracle error: {e}")
        return 1

    predicted = plan(
        PlanInputs(
            n=DOC_LEN,
            K=10_000,
            tau=TAU,
            alpha=1.0,
            max_k=K,
            reduce_calls_per_node=1,
        )
    )

    print(f"\nSelected segment: {result!r}")
    print(f"Oracle calls (actual): {ex.oracle_calls}")
    print(f"Oracle calls (predicted): {predicted.predicted_calls}")
    print(f"Plan: k*={predicted.k_star}, d={predicted.d}")

    topic_a_selected = isinstance(result, str) and TOPIC_A_PHRASE in result
    calls_match = ex.oracle_calls == predicted.predicted_calls

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "oracle_calls_match_planner": calls_match,
        "topic_a_selected": topic_a_selected,
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
