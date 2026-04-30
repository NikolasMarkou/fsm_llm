"""
Plan-Execute with Recovery — Replanning on Tool Failure
========================================================

Demonstrates PlanExecuteAgent's ability to recover from failures:
when a planned tool call fails, the agent detects the failure,
replans the remaining steps, and continues execution.

This simulates a data pipeline where some API calls randomly fail,
forcing the agent to adapt its strategy mid-execution.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/plan_execute_recovery/run.py
"""

import os
import random

from fsm_llm.stdlib.agents import AgentConfig, PlanExecuteAgent, ToolRegistry

# Use a fixed seed for reproducible "failures"
_rng = random.Random(42)
_call_count = {"fetch_sales": 0, "fetch_inventory": 0}


def fetch_sales(params: dict) -> str:
    """Fetch sales data for a product category."""
    category = params.get("category", "unknown")
    _call_count["fetch_sales"] += 1

    # First call to certain categories "fails" to demonstrate recovery
    if category.lower() == "electronics" and _call_count["fetch_sales"] == 1:
        return (
            "ERROR: Sales API timeout for electronics. Service temporarily unavailable."
        )

    data = {
        "electronics": "Electronics sales: $2.4M (Q4), +15% YoY. Top: laptops ($800K), phones ($600K), tablets ($400K).",
        "clothing": "Clothing sales: $1.8M (Q4), +8% YoY. Top: winter coats ($500K), shoes ($400K), accessories ($300K).",
        "food": "Food & grocery: $3.1M (Q4), +5% YoY. Top: fresh produce ($900K), dairy ($700K), snacks ($500K).",
        "home": "Home & garden: $1.2M (Q4), +12% YoY. Top: furniture ($500K), decor ($350K), tools ($200K).",
    }
    for key, value in data.items():
        if key in category.lower():
            return value
    return f"Sales data for {category}: $1.5M average, steady growth."


def fetch_inventory(params: dict) -> str:
    """Check inventory levels for a product category."""
    category = params.get("category", "unknown")

    inventory = {
        "electronics": "Electronics inventory: 15,000 units. Stock level: 72% (reorder needed for laptops). Lead time: 14 days.",
        "clothing": "Clothing inventory: 28,000 units. Stock level: 85% (healthy). Seasonal items being cleared.",
        "food": "Food inventory: 45,000 units. Stock level: 91% (well-stocked). Perishables rotation: 3-day cycle.",
        "home": "Home inventory: 8,500 units. Stock level: 65% (low on furniture). Supplier backlog: 21 days.",
    }
    for key, value in inventory.items():
        if key in category.lower():
            return value
    return f"Inventory for {category}: moderate levels, standard reorder cycles."


def analyze_trends(params: dict) -> str:
    """Analyze sales trends and generate insights."""
    data = params.get("data", "")
    d = data.lower()

    if "electronics" in d and "clothing" in d:
        return (
            "Cross-category analysis: Electronics outperforming clothing by 33% in revenue. "
            "Electronics shows stronger growth (+15% vs +8%). Recommend: increase electronics "
            "marketing budget by 20%, reduce clothing inventory by 10%."
        )
    if "inventory" in d and "sales" in d:
        return (
            "Inventory-Sales correlation: Home & garden has lowest stock (65%) but strong growth (+12%). "
            "Risk: stockout in 2 weeks without reorder. Food is overstocked relative to growth rate."
        )
    return "Trend analysis: data shows mixed signals, recommend deeper investigation."


def generate_report(params: dict) -> str:
    """Generate a summary report from collected data."""
    findings = params.get("findings", "")
    return (
        f"QUARTERLY BUSINESS REPORT\n"
        f"========================\n"
        f"Based on analysis: {findings[:200]}\n\n"
        f"Key recommendations:\n"
        f"1. Prioritize electronics restocking (14-day lead time)\n"
        f"2. Address home & garden supply chain risk\n"
        f"3. Reallocate marketing budget toward high-growth categories"
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    registry = ToolRegistry()
    registry.register_function(
        fetch_sales,
        name="fetch_sales",
        description="Fetch quarterly sales data for a product category (electronics, clothing, food, home)",
        parameter_schema={
            "properties": {
                "category": {"type": "string", "description": "Product category"}
            }
        },
    )
    registry.register_function(
        fetch_inventory,
        name="fetch_inventory",
        description="Check current inventory levels for a product category",
        parameter_schema={
            "properties": {
                "category": {"type": "string", "description": "Product category"}
            }
        },
    )
    registry.register_function(
        analyze_trends,
        name="analyze_trends",
        description="Analyze trends from collected sales and inventory data",
        parameter_schema={
            "properties": {
                "data": {"type": "string", "description": "Collected data to analyze"}
            }
        },
    )
    registry.register_function(
        generate_report,
        name="generate_report",
        description="Generate a summary report from findings",
        parameter_schema={
            "properties": {
                "findings": {"type": "string", "description": "Key findings to include"}
            }
        },
    )

    config = AgentConfig(model=model, max_iterations=20, temperature=0.3)
    agent = PlanExecuteAgent(
        tools=registry,
        config=config,
        max_replans=2,
    )

    task = (
        "Generate a quarterly business report. Steps needed: "
        "1) Fetch sales data for electronics and clothing, "
        "2) Check inventory for both categories, "
        "3) Analyze the combined data for trends, "
        "4) Generate a summary report with recommendations. "
        "Note: some API calls may fail temporarily — retry if needed."
    )

    print("=" * 60)
    print("Plan-Execute with Recovery")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print("Max replans: 2")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nReport:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")
        print(f"Tools used: {result.tools_used}")

        steps = result.final_context.get("plan_steps", [])
        if steps:
            print(f"\nPlan execution ({len(steps)} steps):")
            for s in steps:
                if isinstance(s, dict):
                    status = s.get("status", "?")
                    desc = s.get("description", "N/A")
                    print(f"  [{status}] {desc[:80]}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": result.answer is not None and len(str(result.answer)) > 10,
        "iterations_ok": result.iterations_used >= 1,
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
