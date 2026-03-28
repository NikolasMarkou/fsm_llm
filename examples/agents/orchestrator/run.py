"""
Orchestrator Agent Example -- Multi-Agent Coordination
======================================================

Demonstrates the orchestrator-workers pattern: the orchestrator
decomposes a task, delegates subtasks to worker agents, and
synthesizes the results.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/orchestrator/run.py
"""

import os

from fsm_llm_agents import AgentConfig, AgentResult, OrchestratorAgent


def simple_worker(subtask: str) -> AgentResult:
    """A mock worker that handles subtasks with canned responses."""
    s = subtask.lower()

    responses = {
        "benefit": (
            "Key benefits of renewable energy: reduces carbon emissions, "
            "creates jobs, decreases dependence on fossil fuels, and provides "
            "long-term cost savings as technology improves."
        ),
        "challenge": (
            "Main challenges: intermittency (sun/wind aren't constant), "
            "high upfront infrastructure costs, energy storage limitations, "
            "and land use / environmental impact concerns."
        ),
        "economic": (
            "Economic impact: renewables sector employs over 12 million globally, "
            "solar costs dropped 89% since 2010, wind is now cheaper than coal "
            "in most markets. Transition costs are offset by fuel savings."
        ),
        "policy": (
            "Policy recommendations: carbon pricing, renewable portfolio standards, "
            "grid modernization investment, R&D tax credits, and international "
            "cooperation frameworks like the Paris Agreement."
        ),
    }

    for key, response in responses.items():
        if key in s:
            return AgentResult(answer=response, success=True)

    return AgentResult(
        answer=f"Researched: {subtask}. Found general information on the topic.",
        success=True,
    )


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    config = AgentConfig(model=model, max_iterations=15, temperature=0.7)
    agent = OrchestratorAgent(
        worker_factory=simple_worker,
        config=config,
        max_workers=4,
    )

    task = (
        "Analyze the global transition to renewable energy. "
        "Cover benefits, challenges, economic impact, and policy recommendations."
    )

    print("=" * 60)
    print("Orchestrator Agent -- Multi-Agent Coordination")
    print("=" * 60)
    print(f"Model: {model}")
    print("Max workers: 4")
    print(f"Task: {task}\n")

    try:
        result = agent.run(task)
        print(f"\nSynthesized Answer:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
