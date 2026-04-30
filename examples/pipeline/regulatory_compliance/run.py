"""
Regulatory-Compliance Pipeline — λ-DSL twin (3-leaf assess→review→finalize).

Run::

    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/pipeline/regulatory_compliance/run.py
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
from fsm_llm.runtime import Executor, LiteLLMOracle, leaf, let

S_A = "examples.pipeline.regulatory_compliance.schemas.Assessment"
S_R = "examples.pipeline.regulatory_compliance.schemas.Review"
S_F = "examples.pipeline.regulatory_compliance.schemas.FinalReport"

TASK = (
    "Conduct a regulatory compliance assessment for PayStream Financial "
    "Technologies, a Series C fintech ($85M raised) operating in 12 US states "
    "and 5 EU countries, processing $2.1B in annual transaction volume with "
    "instant P2P payments and AI-driven credit scoring. Cover key regimes "
    "(PCI-DSS, GDPR, US state money-transmitter laws, EU PSD2, AI Act)."
)


def build_term() -> Any:
    assess = leaf(
        template=(
            "You are a senior compliance officer.\nTask: {task}\n"
            "Produce an assessment. Return JSON matching Assessment "
            "(summary, findings, risk_level: low/medium/high)."
        ),
        input_vars=("task",),
        schema_ref=S_A,
    )
    review = leaf(
        template=(
            "Audit the assessment for completeness and accuracy. "
            "Score 0..1; checker_passed=true iff score >= 0.7.\n"
            "Assessment (JSON): {assessment}\n"
            "Return JSON matching Review."
        ),
        input_vars=("assessment",),
        schema_ref=S_R,
    )
    finalize = leaf(
        template=(
            "Produce the final report incorporating audit feedback.\n"
            "Task: {task}\n"
            "Assessment (JSON): {assessment}\n"
            "Review (JSON): {review}\n"
            "Return JSON matching FinalReport."
        ),
        input_vars=("task", "assessment", "review"),
        schema_ref=S_F,
    )
    return let(
        "assessment",
        assess,
        let("review", review, finalize),
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("=" * 60)
    print("Regulatory Compliance (λ-DSL)")
    print("=" * 60)
    print(f"Model: {model}\nTask: {TASK[:80]}...")
    print("-" * 60)

    llm = LiteLLMInterface(model=model)
    oracle = LiteLLMOracle(llm, context_window_tokens=8192)
    ex = Executor(oracle=oracle)

    error: Exception | None = None
    final: dict[str, Any] | None = None
    try:
        final = ex.run(build_term(), {"task": TASK})
    except Exception as e:
        error = e
        print(f"Error: {e}")

    if final is not None:
        print(f"\nSummary: {final.get('summary', '')[:300]}")
        print(f"Findings: {len(final.get('findings', []))}")
        print(f"Risk level: {final.get('risk_level')}")
        print(f"Oracle calls: {ex.oracle_calls}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    summary = (final or {}).get("summary", "") if isinstance(final, dict) else ""
    findings = (final or {}).get("findings", []) if isinstance(final, dict) else []
    checks = {
        "answer_present": (
            error is None and isinstance(summary, str) and len(summary) > 20
        ),
        "findings_present": isinstance(findings, list) and len(findings) >= 1,
        "pipeline_completed": ex.oracle_calls >= 3,
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
