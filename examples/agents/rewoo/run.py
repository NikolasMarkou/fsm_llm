"""
REWOO Agent Example — Plan-First Tool Use
==========================================

Demonstrates the REWOO (Reasoning Without Observation) pattern:
the LLM plans ALL tool calls upfront in a single pass, then
executes them sequentially with evidence substitution (#E1, #E2).

This is more token-efficient than ReAct (2 LLM calls instead of
2*N for N tools) but less adaptive to intermediate results.

Tools:
  - search: Simulates web search
  - calculate: Performs arithmetic
  - lookup: Retrieves specific facts

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/rewoo/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/rewoo/run.py
"""

import os

from fsm_llm_agents import AgentConfig, REWOOAgent, ToolRegistry


def search(params: dict) -> str:
    """Simulate a web search engine."""
    query = str(params.get("query", "")).lower()
    if "population" in query and "france" in query:
        return "France population: approximately 67.8 million (2024)."
    if "population" in query and "germany" in query:
        return "Germany population: approximately 84.4 million (2024)."
    if "gdp" in query and "france" in query:
        return "France GDP: approximately $3.05 trillion (2024)."
    return f"Search results for: {params.get('query', '')} — data found."


def calculate(params: dict) -> str:
    """Perform arithmetic calculations safely using AST parsing."""
    import ast
    import operator

    expression = str(params.get("expression", ""))
    if not expression.strip():
        return "Error: Empty expression"

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
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
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return f"Result: {_safe_eval(tree)}"
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
        return f"Could not evaluate: {expression} ({e})"


def lookup(params: dict) -> str:
    """Look up specific facts."""
    topic = str(params.get("topic", "")).lower()
    facts = {
        "france capital": "Paris",
        "germany capital": "Berlin",
        "france area": "640,679 km²",
        "germany area": "357,022 km²",
    }
    for key, value in facts.items():
        if key in topic:
            return value
    return f"Lookup for '{params.get('topic', '')}': data available."


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:9b")
        return

    registry = ToolRegistry()
    registry.register_function(
        search,
        name="search",
        description="Search the web for information",
        parameter_schema={"query": "search query string"},
    )
    registry.register_function(
        calculate,
        name="calculate",
        description="Perform arithmetic calculations",
        parameter_schema={"expression": "math expression to evaluate"},
    )
    registry.register_function(
        lookup,
        name="lookup",
        description="Look up specific facts about a topic",
        parameter_schema={"topic": "topic to look up"},
    )

    config = AgentConfig(
        model=model,
        max_iterations=10,
        temperature=0.3,
    )

    agent = REWOOAgent(tools=registry, config=config)

    task = (
        "Find the population of France and Germany, calculate the "
        "difference, and tell me which country is more populous."
    )
    print(f"Task: {task}")
    print(f"Model: {model}")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nAnswer: {result.answer}")
        print(f"Success: {result.success}")
        print(f"Tools used: {result.tools_used}")

        # Show execution evidence
        evidence = result.final_context.get("evidence", {})
        if evidence:
            print(f"\nEvidence chain ({len(evidence)} steps):")
            for key, value in evidence.items():
                display = str(value)[:80]
                print(f"  {key}: {display}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
