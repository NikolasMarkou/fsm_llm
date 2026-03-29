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
        max_iterations=30,
        temperature=0.7,
    )

    agent = DebateAgent(
        config=config,
        num_rounds=2,
        proposer_persona=(
            "You are a future-of-work advocate with deep knowledge of remote work research. "
            "Key evidence you should cite:\n"
            "- Stanford study (2023): remote workers are 13% more productive\n"
            "- Global Workplace Analytics: employers save $11,000/year per remote worker\n"
            "- Employees save $4,000/year on commuting costs\n"
            "- Buffer survey: 98% of remote workers want to continue remotely\n"
            "- Companies offering remote options see 25% lower turnover\n"
            "- Case studies: GitLab (2,000+ all-remote, 45% YoY revenue growth), "
            "Automattic/WordPress (2,000 remote, $500M+ revenue), "
            "Spotify ('Work From Anywhere' — satisfaction up 15%, attrition down 20%)\n"
            "Argue that remote-first is the future with specific numbers."
        ),
        critic_persona=(
            "You are an organizational psychologist who studies the risks of remote work. "
            "Key evidence you should cite:\n"
            "- Gallup: fully remote workers report 20% lower engagement\n"
            "- WHO report: remote workers report 41% higher stress levels\n"
            "- Sedentary behavior increased 32% among full-time remote workers\n"
            "- Microsoft Work Trend Index: remote teams take 25% longer to reach decisions\n"
            "- Spontaneous collaboration dropped 50% in remote settings\n"
            "- 21% of remote workers report loneliness as biggest challenge\n"
            "- Case studies: IBM reversed remote work in 2017 (citing innovation), "
            "Yahoo banned remote work in 2013 under Marissa Mayer\n"
            "Challenge the proposer with specific counterevidence and behavioral research."
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
