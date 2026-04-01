"""
SkillLoader Example -- Directory-Based Skill Discovery
======================================================

Demonstrates loading tools from a skills directory using
SkillLoader.from_directory(), grouping by category, and
building a ToolRegistry automatically.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/skill_loader/run.py
"""

import os

from fsm_llm_agents import AgentConfig, ReactAgent
from fsm_llm_agents.skills import SkillLoader


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Load skills from the skills/ directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    skills_dir = os.path.join(current_dir, "skills")

    print("=" * 60)
    print("SkillLoader -- Directory-Based Tool Discovery")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Skills directory: {skills_dir}\n")

    # Load all skills
    skills = SkillLoader.from_directory(skills_dir)
    print(f"Loaded {len(skills)} skills:")
    for skill in skills:
        print(f"  - {skill.name} [{skill.category}]: {skill.description}")

    # Group by category
    by_category = SkillLoader.by_category(skills)
    print(f"\nCategories: {list(by_category.keys())}")

    # Convert to ToolRegistry
    registry = SkillLoader.to_tool_registry(skills)
    print(f"Registry tools: {', '.join(registry.tool_names)}\n")

    # Create agent with loaded tools
    config = AgentConfig(model=model, max_iterations=8, temperature=0.7)
    agent = ReactAgent(tools=registry, config=config)

    task = "What is the weather like in Paris right now?"
    print(f"Task: {task}")
    print("-" * 40)

    try:
        result = agent.run(task)
        print(f"Answer: {result.answer}")
        print(f"Tools used: {result.tools_used}")
        print(f"Iterations: {result.iterations_used}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": result.answer is not None and len(str(result.answer)) > 10,
        "iterations_ok": result.iterations_used >= 1,
        "completed": result.iterations_used < config.max_iterations,
        "tools_called": len(result.tools_used) > 0,
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
