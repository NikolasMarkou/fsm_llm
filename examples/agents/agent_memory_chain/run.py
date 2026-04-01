"""
Agent Memory Chain — Multi-Task Continuity via Working Memory
==============================================================

Demonstrates how a single agent with WorkingMemory maintains context
across multiple independent task runs. Each task builds on knowledge
accumulated in previous tasks, showing persistent learning.

Sequence:
  Task 1: Research a company → store findings in memory
  Task 2: Recall research → generate recommendation

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/agent_memory_chain/run.py
"""

import os

from fsm_llm.memory import WorkingMemory
from fsm_llm_agents import (
    AgentConfig,
    ReactAgent,
    ToolRegistry,
    create_memory_tools,
    tool,
)


@tool
def company_info(name: str) -> str:
    """Look up company information."""
    companies = {
        "acme": (
            "Acme Corp: Founded 2015, B2B SaaS, 500 employees. "
            "Products: project management and team collaboration tools. "
            "Markets: North America (60%), Europe (30%), Asia (10%). "
            "CEO: Jane Smith. Headquarters: San Francisco."
        ),
        "techstart": (
            "TechStart Inc: Founded 2019, AI/ML platform, 120 employees. "
            "Products: automated data pipeline and model deployment. "
            "Revenue: $15M ARR, growing 80% YoY. Series B funded ($40M)."
        ),
    }
    for key, value in companies.items():
        if key in name.lower():
            return value
    return f"Company info for {name}: mid-size technology company."


@tool
def financial_data(company: str) -> str:
    """Get financial data for a company."""
    financials = {
        "acme": (
            "Acme Corp financials: Revenue $85M (2024), +22% YoY. "
            "Gross margin 72%. Net income $8.5M. Burn rate: cash-flow positive. "
            "ARR: $78M. Churn rate: 5.2% annual. LTV/CAC ratio: 3.8x. "
            "Cash reserves: $45M. Debt: $10M credit facility."
        ),
    }
    for key, value in financials.items():
        if key in company.lower():
            return value
    return f"Financial data for {company}: healthy growth metrics."


@tool
def market_position(sector: str) -> str:
    """Assess market position in a sector."""
    sectors = {
        "project management": (
            "Project management SaaS: $7.5B market (2024). "
            "Leaders: Asana (15%), Monday.com (12%), Jira (20%). "
            "Acme Corp estimated at 3% market share. "
            "Growth opportunity: mid-market segment underserved."
        ),
        "collaboration": (
            "Team collaboration: $18B market. Dominated by Microsoft Teams, Slack. "
            "Niche players carving specialized segments. "
            "AI-powered features becoming table stakes."
        ),
    }
    for key, value in sectors.items():
        if key in sector.lower():
            return value
    return (
        f"Market position in {sector}: competitive landscape with growth opportunities."
    )


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Shared memory across all tasks
    memory = WorkingMemory()
    memory_tools = create_memory_tools(memory)

    # Build registry with research + memory tools
    registry = ToolRegistry()
    registry.register(company_info._tool_definition)
    registry.register(financial_data._tool_definition)
    registry.register(market_position._tool_definition)
    for tool_def in memory_tools:
        registry.register(tool_def)

    config = AgentConfig(model=model, max_iterations=4, temperature=0.5)

    print("=" * 60)
    print("Agent Memory Chain — Multi-Task Continuity")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print()

    # ── Task 1: Research + Store ──
    task1 = (
        "Use company_info tool for Acme Corp, then use remember tool "
        "to store the key facts (name, revenue, employees)."
    )
    print(f"[Task 1 - Research] {task1[:80]}...")
    print("-" * 40)
    agent1 = ReactAgent(tools=registry, config=config)
    try:
        result1 = agent1.run(task1)
        print(f"  Answer: {result1.answer[:200]}")
        print(f"  Tools used: {result1.tools_used}")
    except Exception as e:
        print(f"  Error: {e}")

    print()

    # ── Task 2: Recall + Recommend ──
    task2 = (
        "Use recall tool to retrieve Acme Corp data from memory. "
        "Give a Buy/Hold/Pass recommendation."
    )
    print(f"[Task 2 - Recommendation] {task2[:80]}...")
    print("-" * 40)
    agent2 = ReactAgent(tools=registry, config=config)
    try:
        result2 = agent2.run(task2)
        print(f"  Answer: {result2.answer[:300]}")
        print(f"  Tools used: {result2.tools_used}")
    except Exception as e:
        print(f"  Error: {e}")

    # ── Final Memory State ──
    print(f"\n{'=' * 60}")
    print("Final Memory State (accumulated across 2 tasks):")
    print("=" * 60)
    for buffer_name in memory.list_buffers():
        data = memory.get_buffer(buffer_name)
        if data:
            print(f"\n  [{buffer_name}]")
            for key, value in data.items():
                print(f"    {key} = {str(value)[:80]}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    try:
        r1_answer = result1.answer is not None and len(str(result1.answer)) > 10
        r1_tools = len(result1.tools_used) > 0
    except Exception:
        r1_answer = False
        r1_tools = False
    try:
        r2_answer = result2.answer is not None and len(str(result2.answer)) > 10
        r2_tools = len(result2.tools_used) > 0
    except Exception:
        r2_answer = False
        r2_tools = False
    checks = {
        "task1_answer_present": r1_answer,
        "task1_tools_called": r1_tools,
        "task2_answer_present": r2_answer,
        "task2_tools_called": r2_tools,
        "memory_populated": len(memory.list_buffers()) > 0,
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
