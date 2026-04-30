"""
Debate-with-Tools Pipeline — λ-DSL twin (evidence-based 2-round debate).

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/debate_with_tools/run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fsm_llm.llm import LiteLLMInterface
from fsm_llm.runtime import Executor, LiteLLMOracle, leaf, let_

SCHEMA_TURN = "examples.pipeline.debate_with_tools.schemas.Turn"
SCHEMA_VERDICT = "examples.pipeline.debate_with_tools.schemas.Verdict"

TASK = (
    "Should companies adopt permanent remote-first work policies? "
    "Consider productivity, employee wellbeing, collaboration, "
    "cost savings, and company culture."
)
PROPOSER = (
    "You are a future-of-work advocate. Cite evidence (Stanford 13% more "
    "productive, GitLab 45% YoY growth, Buffer 98% want remote). Argue "
    "remote-first with specific numbers."
)
CRITIC = (
    "You are an organizational psychologist citing risks (Gallup -20% "
    "engagement, WHO +41% stress, IBM reversal 2017). Challenge with "
    "behavioral research."
)
JUDGE = (
    "You are a management consultant. Hybrid workers report 83% satisfaction. "
    "Synthesize both sides; recommend a practical hybrid approach."
)


def _turn(persona: str, label: str, prior_var: str | None = None) -> Any:
    if prior_var is None:
        template = (
            f"Persona: {persona}\nTopic: {{topic}}\nLabel: {label}\n"
            "Make your strongest case. Return JSON matching Turn schema."
        )
        return leaf(template, ("topic",), schema_ref=SCHEMA_TURN)
    template = (
        f"Persona: {persona}\nTopic: {{topic}}\nLabel: {label}\n"
        f"Prior turn (JSON): {{{prior_var}}}\n"
        "Counter or extend. Return JSON matching Turn schema."
    )
    return leaf(template, ("topic", prior_var), schema_ref=SCHEMA_TURN)


def _judge(label: str, prop_var: str, crit_var: str) -> Any:
    template = (
        f"Persona: {JUDGE}\nTopic: {{topic}}\nLabel: {label}\n"
        f"Proposer (JSON): {{{prop_var}}}\nCritic (JSON): {{{crit_var}}}\n"
        "Return JSON matching Verdict schema."
    )
    return leaf(template, ("topic", prop_var, crit_var), schema_ref=SCHEMA_VERDICT)


def build_term() -> Any:
    return let_(
        "p1",
        _turn(PROPOSER, "round-1-proposer"),
        let_(
            "c1",
            _turn(CRITIC, "round-1-critic", "p1"),
            let_(
                "v1",
                _judge("round-1", "p1", "c1"),
                let_(
                    "p2",
                    _turn(PROPOSER, "round-2-proposer"),
                    let_(
                        "c2",
                        _turn(CRITIC, "round-2-critic", "p2"),
                        _judge("round-2", "p2", "c2"),
                    ),
                ),
            ),
        ),
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("=" * 60)
    print("Debate Agent — Evidence-Based Analysis (λ-DSL)")
    print("=" * 60)
    print(f"Model: {model}\nRounds: 2\nTopic: {TASK}")
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
        print(f"Oracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    verdict_text = (final or {}).get("verdict", "") if isinstance(final, dict) else ""
    checks = {
        "answer_present": (
            error is None and isinstance(verdict_text, str) and len(verdict_text) > 10
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
