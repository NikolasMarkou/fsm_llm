"""
Consistency-with-Tools Pipeline — λ-DSL twin
==============================================

Self-consistency over a multi-step math problem with 3 samples.
Aggregate via majority vote on the rounded answer.

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/consistency_with_tools/run.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fsm_llm.llm import LiteLLMInterface
from fsm_llm.runtime import (
    Executor,
    LiteLLMOracle,
    ReduceOp,
    abs_,
    fmap,
    leaf,
    reduce,
    var,
)

NUM_SAMPLES = 3
SCHEMA_REF = "examples.pipeline.consistency_with_tools.schemas.CalcOut"
TASK = (
    "A car travels at 100 km/h for 2.5 hours. How far does it go in miles? "
    "Show your work step by step. "
    "Hint: 1 km = 0.621371 miles. "
    "First calculate total km, then convert to miles. "
    "Give the final answer as a number rounded to 1 decimal place."
)

TEMPLATE = (
    "You are a careful step-by-step reasoner.\n"
    "Question: {question}\n"
    "Sample index: {seed}\n"
    "Solve carefully, then return JSON matching CalcOut: "
    '{{"answer": <float>, "steps": ["...", ...], "confidence": <0..1>}}.'
)


def _majority_step(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Majority vote on rounded answer."""
    counts: Counter[float] = Counter()
    counts[round(float(a.get("answer", 0.0)), 1)] += 1
    counts[round(float(b.get("answer", 0.0)), 1)] += 1
    winner_key, _ = counts.most_common(1)[0]
    return a if round(float(a.get("answer", 0.0)), 1) == winner_key else b


MAJORITY_OP = ReduceOp(name="majority", fn=_majority_step, associative=True, unit=None)


def build_term() -> Any:
    sample_leaf = leaf(
        template=TEMPLATE,
        input_vars=("question", "seed"),
        schema_ref=SCHEMA_REF,
    )
    return reduce(var("majority"), fmap(abs_("seed", sample_leaf), var("seeds")))


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("=" * 60)
    print("Self-Consistency — Multi-Step Calculation (λ-DSL)")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Task: {TASK[:80]}...")
    print("-" * 60)

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    error: Exception | None = None
    final: dict[str, Any] | None = None
    try:
        final = ex.run(
            build_term(),
            {
                "question": TASK,
                "seeds": list(range(NUM_SAMPLES)),
                "majority": MAJORITY_OP,
            },
        )
    except Exception as e:
        error = e
        print(f"Error: {e}")

    if final is not None:
        print(f"\nAggregated answer: {final.get('answer')} mi")
        print(f"Confidence: {final.get('confidence')}")
        print(f"Oracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    answer_val = (final or {}).get("answer", 0.0) if isinstance(final, dict) else 0.0
    checks = {
        "answer_present": (
            error is None and final is not None and isinstance(answer_val, int | float)
        ),
        "samples_generated": ex.oracle_calls >= 2,
    }
    extracted = sum(1 for v in checks.values() if v)
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        print(f"  {key:25s}: {passed!s:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} "
        f"({100 * extracted // len(checks)}%)"
    )


if __name__ == "__main__":
    main()
