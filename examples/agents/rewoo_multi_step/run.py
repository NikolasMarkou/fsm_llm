"""
REWOO Multi-Step — Complex Dependency Planning
================================================

Demonstrates REWOO (Reasoning Without Observation) with a complex
multi-step plan where later steps depend on earlier evidence (#E1, #E2).

This example solves a multi-part travel planning problem requiring:
  1. Weather lookup for candidate cities
  2. Flight price comparison
  3. Hotel availability check
  4. Budget calculation combining all data

The key advantage of REWOO is that all planning happens in one LLM call,
then execution is pure tool invocation — token-efficient for complex plans.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/rewoo_multi_step/run.py
"""

import os

from fsm_llm.stdlib.agents import AgentConfig, REWOOAgent, ToolRegistry


def check_weather(params: dict) -> str:
    """Check weather forecast for a city."""
    city = str(params.get("city", "")).lower()
    forecasts = {
        "barcelona": "Barcelona (next week): 24°C, sunny, 5% rain chance. Perfect beach weather.",
        "london": "London (next week): 14°C, overcast, 65% rain chance. Bring an umbrella.",
        "tokyo": "Tokyo (next week): 28°C, humid, 30% rain chance. Cherry blossom season ending.",
        "rome": "Rome (next week): 22°C, partly cloudy, 15% rain chance. Pleasant for sightseeing.",
        "paris": "Paris (next week): 18°C, mild, 25% rain chance. Good for walking tours.",
        "new york": "New York (next week): 20°C, clear skies, 10% rain chance. Great outdoor weather.",
    }
    for key, value in forecasts.items():
        if key in city:
            return value
    return f"Weather for {city}: typical seasonal conditions, moderate temperatures."


def search_flights(params: dict) -> str:
    """Search for flight prices to a destination."""
    destination = str(params.get("destination", "")).lower()
    flights = {
        "barcelona": "Flights to Barcelona: $380 (economy), $890 (business). Duration: 2h30m. 5 daily flights.",
        "london": "Flights to London: $320 (economy), $750 (business). Duration: 1h45m. 12 daily flights.",
        "tokyo": "Flights to Tokyo: $890 (economy), $2,400 (business). Duration: 12h. 3 daily flights.",
        "rome": "Flights to Rome: $410 (economy), $950 (business). Duration: 2h15m. 4 daily flights.",
        "paris": "Flights to Paris: $290 (economy), $680 (business). Duration: 1h30m. 8 daily flights.",
        "new york": "Flights to New York: $450 (economy), $1,200 (business). Duration: 8h. 6 daily flights.",
    }
    for key, value in flights.items():
        if key in destination:
            return value
    return f"Flights to {destination}: estimated $500 economy, varies by season."


def check_hotels(params: dict) -> str:
    """Check hotel availability and prices."""
    city = str(params.get("city", "")).lower()
    nights = params.get("nights", 5)
    hotels = {
        "barcelona": f"Barcelona hotels ({nights} nights): Budget $75/night, Mid-range $150/night, Luxury $320/night. 85% available.",
        "london": f"London hotels ({nights} nights): Budget $120/night, Mid-range $220/night, Luxury $450/night. 70% available.",
        "tokyo": f"Tokyo hotels ({nights} nights): Budget $60/night, Mid-range $130/night, Luxury $350/night. 90% available.",
        "rome": f"Rome hotels ({nights} nights): Budget $80/night, Mid-range $160/night, Luxury $380/night. 75% available.",
        "paris": f"Paris hotels ({nights} nights): Budget $110/night, Mid-range $200/night, Luxury $420/night. 65% available.",
    }
    for key, value in hotels.items():
        if key in city:
            return value
    return f"Hotels in {city} ({nights} nights): estimated $100-300/night average."


def calculate_budget(params: dict) -> str:
    """Calculate total trip budget from components."""
    import ast
    import operator

    expression = str(params.get("expression", ""))
    if not expression.strip():
        return "Error: No expression provided."

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def _safe_eval(node):
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_safe_eval(node.operand))
        raise ValueError(f"Unsupported: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return f"Budget calculation: {expression} = ${result:,.2f}"
    except Exception as e:
        return f"Calculation error: {e}"


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    registry = ToolRegistry()
    registry.register_function(
        check_weather,
        name="check_weather",
        description="Check weather forecast for a city",
        parameter_schema={
            "properties": {"city": {"type": "string", "description": "City name"}}
        },
    )
    registry.register_function(
        search_flights,
        name="search_flights",
        description="Search flight prices to a destination city",
        parameter_schema={
            "properties": {
                "destination": {"type": "string", "description": "Destination city"}
            }
        },
    )
    registry.register_function(
        check_hotels,
        name="check_hotels",
        description="Check hotel availability and prices in a city",
        parameter_schema={
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "nights": {"type": "integer", "description": "Number of nights"},
            }
        },
    )
    registry.register_function(
        calculate_budget,
        name="calculate_budget",
        description="Calculate total budget from a math expression (e.g., '380 + 150 * 5')",
        parameter_schema={
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            }
        },
    )

    config = AgentConfig(model=model, max_iterations=12, temperature=0.3)
    agent = REWOOAgent(tools=registry, config=config)

    task = (
        "Plan a 5-night vacation on a $2000 budget. Compare Barcelona and Rome: "
        "check the weather for both, find flight prices, check mid-range hotel costs, "
        "calculate total cost for each city (flights + hotel), and recommend "
        "which city offers the best value within budget."
    )

    print("=" * 60)
    print("REWOO Multi-Step — Complex Travel Planning")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nRecommendation:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Tools used: {result.tools_used}")

        evidence = result.final_context.get("evidence", {})
        if evidence:
            print(f"\nEvidence chain ({len(evidence)} steps):")
            for key, value in evidence.items():
                print(f"  {key}: {str(value)[:100]}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": result.answer is not None and len(str(result.answer)) > 10,
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
