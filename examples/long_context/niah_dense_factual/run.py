"""
NIAH Dense Factual — Needle in a dense fact-rich haystack
============================================================

Demonstrates ``fsm_llm.stdlib.long_context.niah`` (M5 slice 1) over a
dense haystack where every chunk carries its own factual sentence (no
filler/whitespace). The "needle" question targets one specific fact;
the other 7 facts are decoys. Strict T2 still holds because best_answer_op
prefers longer non-sentinel content — when leaves return prose for every
chunk, the reduce tree picks the longest matching answer (the planted
needle phrase is the longest).

τ=256, k=2, n=2048, d=3 → predicted_calls = k^d = 8.
"""

import os
import sys

from fsm_llm.llm import LiteLLMInterface
from fsm_llm.runtime import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.stdlib.long_context import best_answer_op, make_size_bucket, niah

DOC_LEN = 2048
TAU = 256
K = 2

# 8 distinct factual sentences — one per chunk-aligned 256-byte chunk.
# Chunk 4 holds the answer to "What is the access code?" — the others
# are unrelated facts. The leaf prompt makes each chunk return either
# its own factual statement or NOT_FOUND, depending on relevance.
NEEDLE_PHRASE = (
    "ACCESS_CODE: DELTA-2048 is the production deployment key "
    "for the platform-services tier"
)

DENSE_TOPICS: tuple[str, ...] = (
    " Astronomy fact: red giant stars expand dramatically as their "
    "hydrogen cores deplete and helium fusion begins. ",
    " Linguistics fact: the Indo-European language family spans most "
    "of Europe and parts of southern and central Asia. ",
    " Cryptography fact: elliptic-curve signatures rest on the discrete "
    "logarithm problem over additively-written groups. ",
    " Architecture fact: Roman concrete remained durable across millennia "
    "thanks to volcanic ash binders that healed micro-cracks. ",
    # Chunk 4 — the planted needle.
    " Internal note: " + NEEDLE_PHRASE + ". ",
    " Geology fact: zircon crystals preserve uranium-lead isotopic "
    "ratios and date the oldest known terrestrial materials. ",
    " Botany fact: mycorrhizal fungi exchange phosphorus for plant "
    "carbon across vast underground hyphal networks. ",
    " Marine biology fact: deep-sea hydrothermal vents host "
    "chemosynthetic tubeworms thriving in sulfide-rich plumes. ",
)


def build_haystack() -> str:
    assert len(DENSE_TOPICS) == DOC_LEN // TAU == 8
    chunks: list[str] = []
    for topic in DENSE_TOPICS:
        if len(topic) >= TAU:
            chunk = topic[:TAU]
        else:
            chunk = topic + " " * (TAU - len(topic))
        chunks.append(chunk)
    doc = "".join(chunks)
    assert len(doc) == DOC_LEN
    assert "DELTA-2048" in doc
    return doc


def main() -> int:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return 1

    print(f"Model: {model}")
    print(f"Doc length: {DOC_LEN} chars; τ={TAU}; k={K}")
    print("Mode: dense haystack — every chunk is a distinct factual sentence")
    print("-" * 60)

    haystack = build_haystack()
    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    program = niah(
        "What is the access code for the production deployment? "
        "Reply with the access code value.",
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

    needle_found = isinstance(result, str) and "DELTA-2048" in result
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
