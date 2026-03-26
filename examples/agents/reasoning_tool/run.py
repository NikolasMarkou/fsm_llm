"""
Method 1: Reasoning as Agent Tool
===================================

Wraps ``ReasoningEngine.solve_problem()`` as a regular tool in a
ReactAgent. This is the simplest integration path -- zero library
changes, pure application-level composition.

Tools:
  - search: Simulates a web search
  - calculate: Evaluates math expressions
  - reason: Invokes structured reasoning via ReasoningEngine

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/reasoning_tool/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/reasoning_tool/run.py
"""

import os

from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry

# Try to import reasoning engine
try:
    from fsm_llm_reasoning import ReasoningEngine

    _HAS_REASONING = True
except ImportError:
    _HAS_REASONING = False

# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────


def search(params: dict) -> str:
    """Simulate a web search engine."""
    query = params.get("query", "")
    q = query.lower()

    results = {
        "population of france": "France: ~68.4 million people (2024).",
        "gdp of france": "France GDP: ~$3.05 trillion (2024 estimate).",
        "prime number": "A prime number is a natural number greater than 1 with no positive divisors other than 1 and itself.",
        "fibonacci": "The Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89...",
    }

    for key, value in results.items():
        if key in q:
            return value

    return f"Search results for '{query}': No specific results found."


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


def _build_reason_tool(model: str) -> callable:
    """Build a reasoning tool function that wraps ReasoningEngine."""
    engine = ReasoningEngine(model=model)

    def reason(params: dict) -> str:
        """Use structured reasoning to analyze a complex problem."""
        problem = params.get("problem", "")
        if not problem:
            return "Error: No problem statement provided."

        try:
            solution, trace_info = engine.solve_problem(problem)
            reasoning_types = trace_info.get("reasoning_trace", {}).get(
                "reasoning_types_used", ["unknown"]
            )
            return f"[Reasoning type: {reasoning_types}] {solution}"
        except Exception as e:
            return f"Reasoning failed: {e}"

    return reason


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set OPENAI_API_KEY or use Ollama (LLM_MODEL=ollama_chat/...)")
        return

    if not _HAS_REASONING:
        print("This example requires fsm_llm_reasoning.")
        print("Install with: pip install fsm-llm[reasoning]")
        return

    # Build tool registry
    registry = ToolRegistry()
    registry.register_function(
        search,
        name="search",
        description="Search the web for information",
        parameter_schema={
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            }
        },
    )
    registry.register_function(
        calculate,
        name="calculate",
        description="Evaluate a mathematical expression",
        parameter_schema={
            "properties": {
                "expression": {"type": "string", "description": "Math expression"},
            }
        },
    )
    registry.register_function(
        _build_reason_tool(model),
        name="reason",
        description=(
            "Use structured reasoning (analytical, deductive, critical) "
            "to analyze complex problems requiring multi-step logic"
        ),
        parameter_schema={
            "properties": {
                "problem": {"type": "string", "description": "Problem to reason about"},
            }
        },
    )

    config = AgentConfig(model=model, max_iterations=10, temperature=0.7)
    agent = ReactAgent(tools=registry, config=config)

    print("=" * 60)
    print("Method 1: Reasoning as Agent Tool")
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
