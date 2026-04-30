"""
Self-Consistency Agent Example — Reliable Answers via Sampling
==============================================================

Demonstrates the Self-Consistency pattern: generates multiple
independent answers at varying temperatures, then aggregates
via majority vote. This reduces hallucination and improves
reliability on factual questions.

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/self_consistency/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/self_consistency/run.py
"""

import os

from fsm_llm.stdlib.agents import AgentConfig, SelfConsistencyAgent


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:9b")
        return

    config = AgentConfig(
        model=model,
        max_iterations=10,
    )

    # Create Self-Consistency agent with 5 samples
    agent = SelfConsistencyAgent(
        config=config,
        num_samples=5,
    )

    task = "What is the capital of Australia?"
    print(f"Task: {task}")
    print(f"Model: {model}")
    print("Samples: 5")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nAggregated answer: {result.answer}")
        print(f"Success: {result.success}")

        # Show individual samples from context
        samples = result.final_context.get("samples", [])
        if samples:
            print(f"\nIndividual samples ({len(samples)}):")
            for i, s in enumerate(samples, 1):
                display = str(s)[:80] + "..." if len(str(s)) > 80 else str(s)
                print(f"  Sample {i}: {display}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": result.answer is not None and len(str(result.answer)) > 10,
        "iterations_ok": result.iterations_used >= 1,
        "completed": result.success,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {passed!s:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
