"""
Debate Agent — Evidence-Based Multi-Perspective Analysis
=========================================================

Demonstrates the Debate pattern with domain-expert personas that
bring specific data and case studies to the discussion. Each debater
argues from a different evidence base, producing nuanced analysis.

Personas:
  - Proposer: Future-of-work advocate with productivity & cost data
  - Critic: Organizational psychologist with wellbeing & collaboration data
  - Judge: Management consultant synthesizing evidence-based recommendation

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/debate_with_tools/run.py
"""

import os

from fsm_llm_agents import AgentConfig, DebateAgent


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    config = AgentConfig(
        model=model,
        max_iterations=12,
        temperature=0.7,
    )

    agent = DebateAgent(
        config=config,
        num_rounds=2,
        proposer_persona=(
            "You are a future-of-work advocate. Cite evidence: "
            "Stanford (2023): remote workers 13% more productive. "
            "Employers save $11k/yr per remote worker; employees save $4k/yr commuting. "
            "Buffer survey: 98% want to continue remotely. "
            "Remote options = 25% lower turnover. "
            "Case studies: GitLab (all-remote, 45% YoY growth), "
            "Automattic ($500M+ revenue), Spotify (satisfaction +15%, attrition -20%). "
            "Argue remote-first is the future with specific numbers."
        ),
        critic_persona=(
            "You are an organizational psychologist studying remote work risks. Cite evidence: "
            "Gallup: remote workers 20% lower engagement. "
            "WHO: 41% higher stress. Sedentary behavior +32%. "
            "Microsoft: remote teams 25% slower decisions, collaboration -50%. "
            "21% report loneliness as biggest challenge. "
            "Case studies: IBM reversed remote work 2017 (innovation), "
            "Yahoo banned remote 2013. "
            "Challenge with counterevidence and behavioral research."
        ),
        judge_persona=(
            "You are a management consultant synthesizing both sides with nuance. "
            "Note: hybrid workers show highest satisfaction at 83%. "
            "Weigh the evidence presented, identify where each side is strongest, "
            "and recommend a practical hybrid approach with specific guidelines. "
            "Consider that office vacancy rates hit 18% in 2024."
        ),
    )

    task = (
        "Should companies adopt permanent remote-first work policies? "
        "Consider productivity, employee wellbeing, collaboration, "
        "cost savings, and company culture."
    )

    print("=" * 60)
    print("Debate Agent — Evidence-Based Analysis")
    print("=" * 60)
    print(f"Model: {model}")
    print("Rounds: 2")
    print(f"Topic: {task}")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nVerdict: {result.answer}")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.iterations_used}")

        rounds = result.final_context.get("debate_rounds", [])
        if rounds:
            print(f"\nDebate rounds ({len(rounds)}):")
            for r in rounds:
                if isinstance(r, dict):
                    num = r.get("round_num", "?")
                    verdict = r.get("judge_verdict", "N/A")
                    print(f"  Round {num}: {str(verdict)[:120]}...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
