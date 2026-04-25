"""
Pairwise Demo — Pick the segment most relevant to a question
=============================================================

Demonstrates ``fsm_llm.stdlib.long_context.pairwise`` (M5 slice 3 of
``docs/lambda.md``): a recursive ``fix + split + fmap(self) + reduce_(compare)``
λ-term that picks the most-relevant segment of a long document for a given
question via k-ary tournament reduction.

Synthesises a 2048-character haystack containing two competing topical
segments — one about Topic A (target) and one about Topic B (decoy) —
plus filler chunks. Asks "Which segment discusses Topic A in detail?";
pairwise should return the Topic A segment.

Two verification checks (parsed by scripts/eval.py):

1. ``oracle_calls_match_planner`` — Theorem 2 (predicted_calls == actual).
   This is the hard correctness gate.
2. ``topic_a_selected`` — heuristic: the selected segment contains the
   Topic A signature phrase. Best-effort, model-behaviour dependent.

Per D-S3-001, the slice-3 ``compare_op`` is "longer-non-sentinel-wins";
pairwise differentiation lives in the leaf prompt + demo content.

Run::

    export OPENAI_API_KEY=your-key-here
    python examples/long_context/pairwise_demo/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/long_context/pairwise_demo/run.py
"""

import os
import sys

from fsm_llm.lam import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.long_context import compare_op, make_size_bucket, pairwise

DOC_LEN = 2048
TAU = 256
K = 2

# Two competing topical segments. The pairwise factory is asked to pick
# the Topic A segment given a question that targets Topic A.
TOPIC_A_PHRASE = "deep-sea hydrothermal vents host chemosynthetic tubeworms"
TOPIC_B_PHRASE = "medieval guilds regulated apprenticeships in walled cities"
TOPIC_A_OFFSET = 512  # chunk-aligned: doc[512:768]
TOPIC_B_OFFSET = 1280  # chunk-aligned: doc[1280:1536]

TOPIC_A_SEGMENT = (
    " Marine biology fact: " + TOPIC_A_PHRASE + " that derive energy from "
    "sulfide-rich fluids vented at oceanic spreading ridges. "
)
TOPIC_B_SEGMENT = (
    " Economic history fact: " + TOPIC_B_PHRASE + " and controlled the "
    "transmission of trade craft skills across generations. "
)


def build_haystack() -> str:
    """Synthesise a 2048-char doc with two competing topical segments.

    Eight 256-char chunks. Topic A planted at chunk-aligned offset 512
    (chunk index 2); Topic B at offset 1280 (chunk index 5). Remaining
    six chunks are filler.
    """
    filler = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse. "
    )
    body = list((filler * ((DOC_LEN // len(filler)) + 2))[:DOC_LEN])
    for offset, segment in (
        (TOPIC_A_OFFSET, TOPIC_A_SEGMENT),
        (TOPIC_B_OFFSET, TOPIC_B_SEGMENT),
    ):
        for i, ch in enumerate(segment):
            if offset + i < DOC_LEN:
                body[offset + i] = ch
    doc = "".join(body)[:DOC_LEN]
    assert len(doc) == DOC_LEN
    assert TOPIC_A_PHRASE in doc, "missing topic A phrase"
    assert TOPIC_B_PHRASE in doc, "missing topic B phrase"
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
    print(f"Topic A planted at offset {TOPIC_A_OFFSET}: {TOPIC_A_PHRASE!r}")
    print(f"Topic B planted at offset {TOPIC_B_OFFSET}: {TOPIC_B_PHRASE!r}")
    print("-" * 60)

    haystack = build_haystack()

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    program = pairwise(
        question="Which segment discusses Topic A (deep-sea hydrothermal "
        "vents and chemosynthetic marine biology) in detail?",
        tau=TAU,
        k=K,
    )

    try:
        result = ex.run(
            program,
            {
                "document": haystack,
                "size_bucket": make_size_bucket(TAU),
                "compare": compare_op(),
            },
        )
    except Exception as e:
        print(f"Oracle error: {e}")
        return 1

    predicted = plan(PlanInputs(n=DOC_LEN, K=10_000, tau=TAU, alpha=1.0, max_k=K))

    print(f"\nSelected segment: {result!r}")
    print(f"Oracle calls (actual): {ex.oracle_calls}")
    print(f"Oracle calls (predicted): {predicted.predicted_calls}")
    print(
        f"Plan: k*={predicted.k_star}, d={predicted.d}, "
        f"accuracy_floor={predicted.accuracy_floor:.3f}"
    )

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

    # Theorem-2 (oracle_calls_match_planner) is the hard gate. Topic
    # selection is heuristic and model-behaviour dependent (per
    # LESSONS.md line 107). Exit 0 iff theorem-2 holds.
    return 0 if calls_match else 1


if __name__ == "__main__":
    sys.exit(main())
