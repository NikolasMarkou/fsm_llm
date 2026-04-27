"""
Multi-Hop Dynamic Capped — Demonstrate dynamic-stop upper bound
=================================================================

Demonstrates ``fsm_llm.stdlib.long_context.multi_hop_dynamic`` with a
high ``max_hops`` cap and a custom confidence gate that lets the runner
take exactly 2 hops before stopping. Strict T2: actual_hops·k^d=2·4=8.
Loose upper bound: max_hops·k^d=5·4=20.

τ=256, k=2, n=1024 → d=2, single-hop predicted=4.
"""

import os
import sys
from typing import Any

from fsm_llm.lam import Executor, LiteLLMOracle, PlanInputs, plan
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.stdlib.long_context import (
    best_answer_op,
    make_dynamic_hop_runner,
    make_size_bucket,
    multi_hop_dynamic,
)

DOC_LEN = 1024
TAU = 256
K = 2
MAX_HOPS = 5
TARGET_HOPS = 2  # custom gate stops after this many hops

ENTITY_NAME = "Project Lantern"
LAUNCH_DATE = "April 12, 2026"


def build_haystack() -> str:
    """1024-char doc with entity name + launch date planted in 2 chunks."""
    chunks: list[str] = [
        " Internal note: The flagship initiative is "
        + ENTITY_NAME
        + ", a 2026 strategic effort. ",
        " Release log: " + ENTITY_NAME + " was launched on "
        + LAUNCH_DATE + " by the platform team. ",
        " Filler: Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        + "Sed do eiusmod tempor. ",
        " Filler: Duis aute irure dolor in reprehenderit. "
        + "Excepteur sint occaecat cupidatat. ",
    ]
    out = []
    for c in chunks:
        if len(c) >= TAU:
            out.append(c[:TAU])
        else:
            out.append(c + " " * (TAU - len(c)))
    doc = "".join(out)
    assert len(doc) == DOC_LEN
    assert ENTITY_NAME in doc
    assert LAUNCH_DATE in doc
    return doc


def make_capped_gate(target_hops: int):
    """Continue until ``target_hops`` hops have run, then stop on next call."""

    def _gate(result: Any, hop_index: int) -> bool:
        # hop_index is 0-based, fired AFTER the hop completes.
        # Return True (STOP) once we've completed `target_hops` hops.
        return hop_index + 1 >= target_hops

    return _gate


def main() -> int:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return 1

    print(f"Model: {model}")
    print(f"Doc length: {DOC_LEN}; τ={TAU}; k={K}; max_hops={MAX_HOPS}; "
          f"target_hops={TARGET_HOPS}")
    print("-" * 60)

    haystack = build_haystack()
    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    actual_hops_cell = [0]
    question = "What is the launch date of the flagship initiative?"
    runner = make_dynamic_hop_runner(
        ex,
        question,
        max_hops=MAX_HOPS,
        peer_env={
            "size_bucket": make_size_bucket(TAU),
            "best": best_answer_op(),
        },
        confidence_gate=make_capped_gate(TARGET_HOPS),
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
    print(f"Oracle calls (per hop predicted): {predicted.predicted_calls}")
    print(f"Actual hops run: {actual_hops} / max_hops={MAX_HOPS}")
    print(f"Strict expected (actual_hops*per-hop): {expected_calls}")
    print(f"Upper bound (max_hops*per-hop): {upper_bound}")

    calls_match = ex.oracle_calls == expected_calls
    upper_bound_holds = ex.oracle_calls <= upper_bound
    actual_under_max = actual_hops < MAX_HOPS  # demonstrates early stop

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "oracle_calls_match_dynamic_planner": calls_match,
        "oracle_calls_within_upper_bound": upper_bound_holds,
        "early_stop_under_max_hops": actual_under_max,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:35s}: {passed!s:10s} [{status}]")
    pct = 100 * extracted / len(checks)
    print(f"\nExtraction rate: {extracted}/{len(checks)} ({pct:.0f}%)")

    return 0 if calls_match else 1


if __name__ == "__main__":
    sys.exit(main())
