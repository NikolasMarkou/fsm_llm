"""
Debate Agent Example — Multi-Perspective Analysis
===================================================

Demonstrates the Debate pattern: two AI personas argue opposing
viewpoints while a judge evaluates the exchange. This produces
more nuanced answers than single-perspective generation.

Personas:
  - Proposer: Advocates for a position
  - Critic: Challenges and finds weaknesses
  - Judge: Evaluates arguments and reaches a verdict

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/debate/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/debate/run.py
"""

import os

from fsm_llm_agents import AgentConfig, DebateAgent

# ──────────────────────────────────────────────
# Agent Setup and Execution
# ──────────────────────────────────────────────


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:9b")
        return

    # Create Debate agent with custom personas
    config = AgentConfig(
        model=model,
        max_iterations=25,
        temperature=0.7,
    )

    agent = DebateAgent(
        config=config,
        num_rounds=2,
        proposer_persona=(
            "You are a technology optimist who believes AI will significantly "
            "benefit humanity. Support your arguments with concrete examples "
            "and evidence."
        ),
        critic_persona=(
            "You are a technology skeptic who identifies risks and unintended "
            "consequences of AI. Challenge assumptions and point out overlooked "
            "dangers with specific examples."
        ),
        judge_persona=(
            "You are a balanced technology policy analyst. Evaluate both "
            "arguments fairly, identify the strongest points from each side, "
            "and synthesize a nuanced conclusion."
        ),
    )

    # Run the debate
    task = (
        "Should AI be used to make hiring decisions in companies? "
        "Consider fairness, efficiency, legal implications, and human dignity."
    )
    print(f"Topic: {task}")
    print(f"Model: {model}")
    print("Rounds: 2")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nVerdict: {result.answer}")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.iterations_used}")

        # Show debate rounds
        rounds = result.final_context.get("debate_rounds", [])
        if rounds:
            print(f"\nDebate rounds ({len(rounds)}):")
            for r in rounds:
                if isinstance(r, dict):
                    num = r.get("round_num", "?")
                    verdict = r.get("judge_verdict", "N/A")
                    print(f"  Round {num}: {verdict[:100]}...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
