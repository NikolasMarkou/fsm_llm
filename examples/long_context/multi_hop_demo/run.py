"""
Multi-Hop Demo — 2-hop retrieval over a long document
======================================================

Demonstrates ``fsm_llm.stdlib.long_context.multi_hop`` (M5 slice 3 of
``docs/lambda.md``): a ``Let`` chain of ``hops`` independent niah-style
``fix + split + fmap(self) + reduce_(best)`` λ-terms. Each hop's leaf
prompt closes over the prior hop's result via a ``Var`` reference threaded
by ``Let``.

Synthesises a 2048-character haystack containing two planted chunks:
hop-0 finds an entity name (``Project Helix``); hop-1 finds a fact about
that entity (its launch date). Filler chunks elsewhere.

Two verification checks (parsed by scripts/eval.py):

1. ``oracle_calls_match_2hop_planner`` — Theorem 2 across both hops:
   ``ex.oracle_calls == hops * plan(...).predicted_calls``. Hard
   correctness gate (per D-S3-002, each hop is an independent Fix
   call within Theorem-2 individually).
2. ``launch_date_found`` — heuristic: result string mentions
   ``March 15`` or ``2026-03-15``. Best-effort, model-behaviour dependent.

Per D-S3-002, multi_hop is shaped as a Let chain of independent Fix calls
rather than a single Fix with query-state monad — this keeps each hop
within the per-Fix Theorem-2 cost equality.

Run::

    export OPENAI_API_KEY=your-key-here
    python examples/long_context/multi_hop_demo/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/long_context/multi_hop_demo/run.py
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

DOC_LEN = 2048
TAU = 256
K = 2
HOPS = 2

ENTITY_NAME = "Project Helix"
LAUNCH_DATE_PHRASE = "March 15, 2026"

ENTITY_OFFSET = 512  # chunk-aligned: doc[512:768]
FACT_OFFSET = 1280  # chunk-aligned: doc[1280:1536]

ENTITY_CHUNK = (
    " Internal note: The flagship product is "
    + ENTITY_NAME
    + ", our 2026 strategic initiative led by the platform team. "
)
FACT_CHUNK = (
    " Release log: "
    + ENTITY_NAME
    + " was launched on "
    + LAUNCH_DATE_PHRASE
    + " by the platform team after a multi-quarter rollout. "
)


def build_haystack() -> str:
    """Synthesise a 2048-char doc with two planted chunks for 2-hop retrieval.

    Eight 256-char chunks. Entity name planted at chunk-aligned offset 512
    (chunk index 2); fact about the entity at offset 1280 (chunk index 5).
    Remaining six chunks are filler.
    """
    filler = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse. "
    )
    body = list((filler * ((DOC_LEN // len(filler)) + 2))[:DOC_LEN])
    for offset, segment in (
        (ENTITY_OFFSET, ENTITY_CHUNK),
        (FACT_OFFSET, FACT_CHUNK),
    ):
        for i, ch in enumerate(segment):
            if offset + i < DOC_LEN:
                body[offset + i] = ch
    doc = "".join(body)[:DOC_LEN]
    assert len(doc) == DOC_LEN
    assert ENTITY_NAME in doc, "missing entity name"
    assert LAUNCH_DATE_PHRASE in doc, "missing launch date phrase"
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
    print(f"Doc length: {DOC_LEN} chars; τ={TAU}; k={K}; hops={HOPS}")
    print(f"Entity planted at offset {ENTITY_OFFSET}: {ENTITY_NAME!r}")
    print(f"Fact planted at offset {FACT_OFFSET}: {LAUNCH_DATE_PHRASE!r}")
    print("-" * 60)

    haystack = build_haystack()

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    program = multi_hop(
        question="What is the launch date of the flagship product?",
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
    print(f"Oracle calls (predicted, single hop): {predicted.predicted_calls}")
    print(f"Oracle calls (predicted, total = hops*single): {expected_calls}")
    print(
        f"Plan: k*={predicted.k_star}, d={predicted.d}, "
        f"accuracy_floor={predicted.accuracy_floor:.3f}"
    )

    result_str = result if isinstance(result, str) else ""
    launch_date_found = (
        "March 15" in result_str or "2026-03-15" in result_str or "03-15" in result_str
    )
    calls_match = ex.oracle_calls == expected_calls

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "oracle_calls_match_2hop_planner": calls_match,
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

    # Theorem-2 (oracle_calls_match_2hop_planner) is the hard gate.
    # launch_date_found is heuristic and model-behaviour dependent
    # (per LESSONS.md line 107). Exit 0 iff theorem-2 holds.
    return 0 if calls_match else 1


if __name__ == "__main__":
    sys.exit(main())
