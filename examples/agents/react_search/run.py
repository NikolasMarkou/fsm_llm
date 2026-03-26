"""
ReAct Agent Example — Tool-Use Search Agent
=============================================

Demonstrates the ReAct (Reasoning + Acting) pattern using FSM-LLM.
The agent is given a task and uses tools step-by-step to gather
information before producing a final answer.

Tools:
  - search: Simulates a web search
  - calculate: Evaluates math expressions
  - lookup: Looks up facts from a mock knowledge base

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/react_search/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/react_search/run.py
"""

import os

from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry

# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────


def search(params: dict) -> str:
    """Simulate a web search engine."""
    query = params.get("query", "")
    q = query.lower()

    results = {
        "population of france": "France has a population of approximately 68.4 million people (2024 estimate).",
        "capital of japan": "The capital of Japan is Tokyo, with a metropolitan population of about 14 million.",
        "speed of light": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "height of mount everest": "Mount Everest stands at 8,849 meters (29,032 feet) above sea level.",
        "python programming": "Python is a high-level, general-purpose programming language created by Guido van Rossum.",
    }

    for key, value in results.items():
        if key in q:
            return value

    return f"Search results for '{query}': No specific results found. Try a more specific query."


def calculate(params: dict) -> str:
    """Evaluate a mathematical expression safely using AST parsing."""
    import ast
    import operator

    expression = params.get("expression", "")
    if not expression or not expression.strip():
        return "Error: Empty expression"

    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def _safe_eval(node):
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_safe_eval(node.operand))
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return f"{expression} = {result}"
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
        return f"Error evaluating '{expression}': {e}"


def lookup(params: dict) -> str:
    """Look up a fact from a mock knowledge base."""
    topic = params.get("topic", "").lower()

    facts = {
        "france": "France is a country in Western Europe. Official language: French. Currency: Euro. Area: 551,695 km2.",
        "japan": "Japan is an island country in East Asia. Official language: Japanese. Currency: Yen. Area: 377,975 km2.",
        "everest": "Mount Everest is the highest mountain above sea level, located in the Himalayas on the China-Nepal border.",
        "python": "Python was first released in 1991. It emphasizes code readability and supports multiple programming paradigms.",
    }

    for key, value in facts.items():
        if key in topic:
            return value

    return f"No facts found for topic: '{topic}'"


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:9b")
        return

    # Build tool registry
    registry = ToolRegistry()
    registry.register_function(
        search,
        name="search",
        description="Search the web for information about a topic",
        parameter_schema={
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            }
        },
    )
    registry.register_function(
        calculate,
        name="calculate",
        description="Evaluate a mathematical expression (e.g., '68.4 / 2')",
        parameter_schema={
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate",
                },
            }
        },
    )
    registry.register_function(
        lookup,
        name="lookup",
        description="Look up background facts about a topic from the knowledge base",
        parameter_schema={
            "properties": {
                "topic": {"type": "string", "description": "Topic to look up"},
            }
        },
    )

    # Create the agent
    config = AgentConfig(
        model=model,
        max_iterations=8,
        temperature=0.7,
    )
    agent = ReactAgent(tools=registry, config=config)

    print("=" * 60)
    print("ReAct Search Agent")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print("Type a question or 'quit' to exit.\n")

    while True:
        task = input("Task: ").strip()
        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not task:
            continue

        print(f"\nAgent working on: {task}")
        print("-" * 40)

        try:
            result = agent.run(task)
            print(f"\nAnswer: {result.answer}")
            print(f"Tools used: {result.tools_used}")
            print(f"Iterations: {result.iterations_used}")
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
