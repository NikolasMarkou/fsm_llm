"""
Aggregate Demo — Synthesise across all chunks of a long document
=================================================================

Demonstrates ``fsm_llm.stdlib.long_context.aggregate`` (M5 slice 2 of
``docs/lambda.md``): a recursive ``fix + split + fmap(self) + reduce_(merge)``
λ-term that extracts per-chunk findings and joins them with a pure-Python
reduce op (no extra oracle calls — Theorem 2 holds identically to niah).

Synthesises a 2048-character document with 4 topical sentences placed at
chunk-aligned offsets. Aggregate asks "What facts does this document state
about the project?" and bullet-joins the per-chunk answers.

Two verification checks (parsed by scripts/eval.py):
1. ``oracle_calls_match_planner`` — Theorem 2 (predicted_calls == actual).
2. ``output_nontrivial`` — len(result) > 50 and result != "NOT_FOUND".

Run::

    export OPENAI_API_KEY=your-key-here
    python examples/long_context/aggregate_demo/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/long_context/aggregate_demo/run.py
"""

import os
import sys

from fsm_llm.lam import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.long_context import aggregate, aggregate_op, make_size_bucket

DOC_LEN = 2048
TAU = 256
K = 2

# Four topical sentences placed at chunk-aligned offsets (0, 512, 1024, 1536).
TOPICS = [
    "The project codename is ATLAS.",
    "The lead engineer is Dr. Vega.",
    "The deadline is set for Q3 2026.",
    "The total budget is approximately 4.2 million dollars.",
]
TOPIC_OFFSETS = [0, 512, 1024, 1536]


def build_haystack() -> str:
    """Synthesise a 2048-char doc with 4 topical sentences at known offsets."""
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
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return 1

    print(f"Model: {model}")
    print(f"Doc length: {DOC_LEN} chars; τ={TAU}; k={K}")
    print(f"Topics seeded: {len(TOPICS)} at offsets {TOPIC_OFFSETS}")
    print("-" * 60)

    haystack = build_haystack()

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    program = aggregate(
        "What facts does this document state about the project? "
        "(name, lead, deadline, budget, etc.)",
        tau=TAU,
        k=K,
    )

    try:
        result = ex.run(
            program,
            {
                "document": haystack,
                "size_bucket": make_size_bucket(TAU),
                "merge": aggregate_op(),
            },
        )
    except Exception as e:
        print(f"Oracle error: {e}")
        return 1

    predicted = plan(PlanInputs(n=DOC_LEN, K=10_000, tau=TAU, alpha=1.0, max_k=K))

    print(f"\nFinal aggregate output:\n{result}\n")
    print(f"Oracle calls (actual): {ex.oracle_calls}")
    print(f"Oracle calls (predicted): {predicted.predicted_calls}")
    print(
        f"Plan: k*={predicted.k_star}, d={predicted.d}, "
        f"accuracy_floor={predicted.accuracy_floor:.3f}"
    )

    output_nontrivial = (
        isinstance(result, str) and len(result) > 50 and result != "NOT_FOUND"
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

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
