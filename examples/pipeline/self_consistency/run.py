"""
Self-Consistency Pipeline Example — λ-DSL twin of ``examples/agents/self_consistency``
=======================================================================================

Demonstrates docs/lambda.md M4: a pure λ-term that draws N independent samples
from a structured leaf and folds them with a majority-vote ``ReduceOp``. No
``fsm_llm_agents`` Agent classes — the term is built inline from
``fsm_llm.lam`` primitives.

Pipeline shape::

    reduce(majority, fmap(λ seed. leaf(prompt[seed], schema=AnswerOut), seeds))

Oracle-call equivalence: 5 leaf invocations, matching
``examples/agents/self_consistency/run.py``'s ``num_samples=5``.

Run::

    export OPENAI_API_KEY=your-key-here
    python examples/pipeline/self_consistency/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/self_consistency/run.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Make the project root importable so ``schema_ref`` strings of the form
# ``"examples.pipeline.self_consistency.schemas.AnswerOut"`` resolve via
# ``importlib.import_module`` regardless of the cwd from which this script
# is invoked. See plan decision D-006.
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

NUM_SAMPLES = 5
SCHEMA_REF = "examples.pipeline.self_consistency.schemas.AnswerOut"
TASK = "What is the capital of Australia?"

# Each sample is parameterised by a ``seed`` (int) so the prompts differ
# textually — encourages diversity of completions even at fixed temperature.
SAMPLE_TEMPLATE = (
    "You are a careful factual assistant.\n"
    "Question: {question}\n"
    "Sample index: {seed}\n"
    "Reply with the JSON object literal (NOT a string): "
    '{{"answer": "<short noun phrase>", '
    '"confidence": <number 0..1 or null>}}.\n'
    "When asked to set a 'value' field, set it to this JSON object directly, "
    "not to a string containing the JSON.\n"
    "Be concise; do not add prose outside the JSON."
)


def _majority_step(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Binary fold step for majority vote.

    Each fold accumulator is the most-frequent answer dict observed so far.
    REDUCE folds left-to-right via functools.reduce, so we treat ``a`` as
    the running winner and update only if ``b``'s answer text is more
    frequent in the {a, b} pair under a Counter view.
    """
    # The frozen pair: pick whichever has higher count when both are
    # collapsed over their .answer string. Ties resolve to ``a`` (first-seen).
    counts: Counter[str] = Counter()
    counts[a.get("answer", "")] += 1
    counts[b.get("answer", "")] += 1
    winner_answer, _ = counts.most_common(1)[0]
    return a if a.get("answer", "") == winner_answer else b


MAJORITY_OP = ReduceOp(name="majority", fn=_majority_step, associative=True, unit=None)


def build_term() -> Any:
    """Build the λ-term: reduce(majority, fmap(λ seed. leaf(...), seeds))."""
    sample_leaf = leaf(
        template=SAMPLE_TEMPLATE,
        input_vars=("question", "seed"),
        schema_ref=SCHEMA_REF,
    )
    sampler = abs_("seed", sample_leaf)
    samples_term = fmap(sampler, var("seeds"))
    return reduce(var("majority"), samples_term)


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print(f"Task: {TASK}")
    print(f"Model: {model}")
    print(f"Samples: {NUM_SAMPLES}")
    print("-" * 60)

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    term = build_term()
    env = {
        "question": TASK,
        "seeds": list(range(NUM_SAMPLES)),
        "majority": MAJORITY_OP,
    }

    error: Exception | None = None
    aggregated: dict[str, Any] | None = None
    try:
        aggregated = ex.run(term, env)
    except Exception as e:
        error = e
        print(f"Error: {e}")

    if aggregated is not None:
        print(f"\nAggregated answer: {aggregated.get('answer')}")
        print(f"Confidence: {aggregated.get('confidence')}")
        print(f"Oracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    answer_text = (aggregated or {}).get("answer", "")
    checks = {
        "answer_present": (
            error is None
            and aggregated is not None
            and isinstance(answer_text, str)
            and len(answer_text) > 0
        ),
        "iterations_ok": ex.oracle_calls >= 1,
        "completed": error is None and aggregated is not None,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {passed!s:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} "
        f"({100 * extracted // len(checks)}%)"
    )


if __name__ == "__main__":
    main()
