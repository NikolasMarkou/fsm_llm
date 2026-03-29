"""
Multi-Debate Panel — Parallel Debates with Synthesis
=====================================================

Demonstrates running multiple independent Debate agents in parallel,
each analyzing a different aspect of a complex topic, then synthesizing
all verdicts into a comprehensive analysis.

Architecture:
  ┌─ Debate 1: Economic perspective
  ├─ Debate 2: Social/ethical perspective
  └─ Debate 3: Technical feasibility
  → Final synthesis of all three verdicts

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/multi_debate_panel/run.py
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from fsm_llm_agents import AgentConfig, DebateAgent


def create_debate(model: str, proposer: str, critic: str, judge: str) -> DebateAgent:
    """Create a configured DebateAgent with custom personas."""
    config = AgentConfig(model=model, max_iterations=20, temperature=0.7)
    return DebateAgent(
        config=config,
        num_rounds=1,
        proposer_persona=proposer,
        critic_persona=critic,
        judge_persona=judge,
    )


def run_debate(agent: DebateAgent, task: str, label: str) -> dict:
    """Run a single debate and return results."""
    try:
        result = agent.run(task)
        return {
            "label": label,
            "verdict": result.answer,
            "success": result.success,
            "iterations": result.iterations_used,
            "rounds": len(result.final_context.get("debate_rounds", [])),
        }
    except Exception as e:
        return {
            "label": label,
            "verdict": f"Debate failed: {e}",
            "success": False,
            "iterations": 0,
            "rounds": 0,
        }


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    topic = "Should governments implement Universal Basic Income (UBI)?"

    # Define three debate panels with different perspectives
    panels = [
        {
            "label": "Economic Panel",
            "task": f"{topic} Focus on: economic impact, inflation risk, labor market effects, and fiscal sustainability.",
            "proposer": "You are a progressive economist who advocates for UBI as an economic stimulus. Cite macroeconomic benefits and pilot study results.",
            "critic": "You are a fiscal conservative economist who sees UBI as inflationary and unsustainable. Cite cost projections and labor supply concerns.",
            "judge": "You are a central bank policy analyst. Evaluate the economic arguments objectively and give a balanced verdict on fiscal feasibility.",
        },
        {
            "label": "Social/Ethics Panel",
            "task": f"{topic} Focus on: poverty reduction, human dignity, social cohesion, and potential dependency.",
            "proposer": "You are a social justice advocate who sees UBI as essential for human dignity and poverty elimination. Cite inequality data.",
            "critic": "You are a community development expert who worries UBI undermines work ethic and community ties. Cite behavioral research.",
            "judge": "You are an ethics professor. Evaluate both the rights-based and consequentialist arguments for a nuanced ethical verdict.",
        },
        {
            "label": "Feasibility Panel",
            "task": f"{topic} Focus on: implementation complexity, technology requirements, pilot study outcomes, and political viability.",
            "proposer": "You are a policy implementation expert who sees UBI as technically straightforward. Cite successful pilots (Finland, Kenya, Stockton).",
            "critic": "You are a public administration specialist who sees immense logistical challenges. Cite bureaucratic complexity and political obstacles.",
            "judge": "You are a governance researcher. Evaluate practical feasibility and recommend a realistic implementation timeline if warranted.",
        },
    ]

    print("=" * 60)
    print("Multi-Debate Panel — Parallel Perspective Analysis")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Topic: {topic}")
    print(f"Panels: {len(panels)}")
    print("-" * 60)

    # Run debates in parallel
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for panel in panels:
            agent = create_debate(
                model,
                panel["proposer"],
                panel["critic"],
                panel["judge"],
            )
            future = executor.submit(run_debate, agent, panel["task"], panel["label"])
            futures[future] = panel["label"]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"\n  [{result['label']}] Complete (success={result['success']}, rounds={result['rounds']})"
            )

    # Sort by original order
    label_order = [p["label"] for p in panels]
    results.sort(key=lambda r: label_order.index(r["label"]))

    # Display verdicts
    print(f"\n{'=' * 60}")
    print("PANEL VERDICTS")
    print("=" * 60)
    for r in results:
        print(f"\n--- {r['label']} ---")
        print(f"Verdict: {str(r['verdict'])[:300]}")
        print(f"Iterations: {r['iterations']}")

    # Synthesize (simple concatenation — in production, use another agent)
    print(f"\n{'=' * 60}")
    print("SYNTHESIS")
    print("=" * 60)

    all_verdicts = " | ".join(
        f"[{r['label']}]: {str(r['verdict'])[:150]}" for r in results
    )
    print(f"\nCombined perspectives: {all_verdicts[:500]}")

    successes = sum(1 for r in results if r["success"])
    total_iterations = sum(r["iterations"] for r in results)
    print(f"\nPanels succeeded: {successes}/{len(results)}")
    print(f"Total iterations: {total_iterations}")


if __name__ == "__main__":
    main()
