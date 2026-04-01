"""
Self-Consistency with Domain Knowledge
========================================

Demonstrates SelfConsistency for a complex multi-step reasoning task.
The agent generates N independent solutions to a unit conversion and
calculation problem, then aggregates via majority vote.

This dramatically improves reliability for multi-step math tasks
where a single generation might make arithmetic errors.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/consistency_with_tools/run.py
"""

import os

from fsm_llm_agents import AgentConfig, SelfConsistencyAgent


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    config = AgentConfig(
        model=model,
        max_iterations=8,
    )

    agent = SelfConsistencyAgent(
        config=config,
        num_samples=3,
    )

    task = (
        "A car travels at 100 km/h for 2.5 hours. How far does it go in miles? "
        "Show your work step by step. "
        "Hint: 1 km = 0.621371 miles. "
        "First calculate total km, then convert to miles. "
        "Give the final answer as a number rounded to 1 decimal place."
    )

    print("=" * 60)
    print("Self-Consistency — Multi-Step Calculation")
    print("=" * 60)
    print(f"Model: {model}")
    print("Samples: 3")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nAggregated answer: {result.answer}")
        print(f"Success: {result.success}")

        samples = result.final_context.get("samples", [])
        if samples:
            print(f"\nIndividual samples ({len(samples)}):")
            for i, s in enumerate(samples, 1):
                display = str(s)[:120] + "..." if len(str(s)) > 120 else str(s)
                print(f"  Sample {i}: {display}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    samples = result.final_context.get("samples", [])
    checks = {
        "answer_present": result.answer is not None and len(str(result.answer)) > 10,
        "samples_generated": len(samples) >= 2,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {str(passed):40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
