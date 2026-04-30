"""
Debate Pipeline Example — λ-DSL twin of ``examples/agents/debate``
====================================================================

Two rounds of (proposer → critic → judge), each leaf schema-validated.
Total: 6 oracle calls, matching the original ``num_rounds=2`` debate.

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/debate/run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fsm_llm.runtime import Executor, LiteLLMOracle, leaf, let_
from fsm_llm.llm import LiteLLMInterface

SCHEMA_TURN = "examples.pipeline.debate.schemas.Turn"
SCHEMA_VERDICT = "examples.pipeline.debate.schemas.Verdict"

TASK = (
    "Should AI be used to make hiring decisions in companies? "
    "Consider fairness, efficiency, legal implications, and human dignity."
)
PROPOSER_PERSONA = (
    "You are a technology optimist who believes AI will significantly "
    "benefit humanity. Support arguments with concrete examples."
)
CRITIC_PERSONA = (
    "You are a technology skeptic who identifies risks of AI. "
    "Challenge assumptions and point out overlooked dangers."
)
JUDGE_PERSONA = (
    "You are a balanced technology policy analyst. Evaluate both arguments "
    "fairly, identify the strongest points, synthesize a nuanced conclusion."
)


def _proposer(round_label: str) -> Any:
    template = (
        f"Persona: {PROPOSER_PERSONA}\n"
        f"Topic: {{topic}}\n"
        f"Round: {round_label}\n"
        "Make your strongest case for the affirmative side. Return JSON "
        "matching the Turn schema (argument, key_point)."
    )
    return leaf(
        template=template,
        input_vars=("topic",),
        schema_ref=SCHEMA_TURN,
    )


def _critic(round_label: str, prop_var: str) -> Any:
    template = (
        f"Persona: {CRITIC_PERSONA}\n"
        f"Topic: {{topic}}\n"
        f"Proposer's argument (JSON): {{{prop_var}}}\n"
        f"Round: {round_label}\n"
        "Counter the proposer with the strongest critique. Return JSON "
        "matching the Turn schema."
    )
    return leaf(
        template=template,
        input_vars=("topic", prop_var),
        schema_ref=SCHEMA_TURN,
    )


def _judge(round_label: str, prop_var: str, crit_var: str) -> Any:
    template = (
        f"Persona: {JUDGE_PERSONA}\n"
        f"Topic: {{topic}}\n"
        f"Proposer (JSON): {{{prop_var}}}\n"
        f"Critic (JSON): {{{crit_var}}}\n"
        f"Round: {round_label}\n"
        "Evaluate both. Return JSON matching the Verdict schema "
        "(verdict, winner, confidence)."
    )
    return leaf(
        template=template,
        input_vars=("topic", prop_var, crit_var),
        schema_ref=SCHEMA_VERDICT,
    )


def build_term() -> Any:
    """Two rounds: prop1 → crit1 → judge1 → prop2 (sees prior verdict) → crit2 → judge2."""
    return let_(
        "prop1",
        _proposer("1"),
        let_(
            "crit1",
            _critic("1", "prop1"),
            let_(
                "verdict1",
                _judge("1", "prop1", "crit1"),
                let_(
                    "prop2",
                    _proposer("2"),
                    let_("crit2", _critic("2", "prop2"), _judge("2", "prop2", "crit2")),
                ),
            ),
        ),
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Please set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print(f"Topic: {TASK}")
    print(f"Model: {model}")
    print("Rounds: 2")
    print("-" * 60)

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    error: Exception | None = None
    final: dict[str, Any] | None = None
    try:
        final = ex.run(build_term(), {"topic": TASK})
    except Exception as e:
        error = e
        print(f"Error: {e}")

    if final is not None:
        print(f"\nVerdict: {final.get('verdict', '')[:300]}")
        print(f"Winner: {final.get('winner')}")
        print(f"Confidence: {final.get('confidence')}")
        print(f"Oracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    verdict_text = (final or {}).get("verdict", "") if isinstance(final, dict) else ""
    checks = {
        "answer_present": (
            error is None
            and final is not None
            and isinstance(verdict_text, str)
            and len(verdict_text) > 10
        ),
        "iterations_ok": ex.oracle_calls >= 1,
        "rounds_completed": ex.oracle_calls >= 6,
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
