"""
Reflexion Agent Example — Self-Improving Search Agent
=====================================================

Demonstrates the Reflexion pattern: extends ReAct with evaluation and
verbal self-critique. When the agent's answer doesn't pass evaluation,
it reflects on what went wrong, stores the lesson in episodic memory,
and retries with improved strategy.

Tools:
  - search: Simulates a web search
  - calculate: Evaluates math expressions

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/reflexion/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/reflexion/run.py
"""

import os

from fsm_llm_agents import (
    AgentConfig,
    EvaluationResult,
    ReflexionAgent,
    ToolRegistry,
)

# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────


def search(params: dict) -> str:
    """Simulate a web search engine."""
    query = params.get("query", "")
    q = query.lower()

    results = {
        "population of france": (
            "France has a population of approximately 68.4 million people."
        ),
        "area of france": "France has an area of approximately 643,801 km².",
        "population density": (
            "Population density is calculated as population divided by area."
        ),
    }

    for key, value in results.items():
        if key in q:
            return value
    return f"No specific results for: {query}"


def calculate(params: dict) -> str:
    """Evaluate a math expression safely using AST parsing."""
    import ast
    import operator

    expression = params.get("expression", "")
    if not expression or not expression.strip():
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
        return str(_safe_eval(tree))
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
        return f"Calculation error: {e}"


# ──────────────────────────────────────────────
# Evaluation Function (external validation)
# ──────────────────────────────────────────────


def evaluate_answer(context: dict) -> EvaluationResult:
    """Check if the answer mentions a specific number (not just text)."""
    answer = context.get("final_answer", "")
    observations = context.get("observations", [])

    # Check if we have numeric data in observations
    has_numeric = any(char.isdigit() for obs in observations for char in str(obs))

    if has_numeric and any(char.isdigit() for char in answer):
        return EvaluationResult(
            passed=True,
            score=0.9,
            feedback="Answer includes specific numeric data.",
        )

    return EvaluationResult(
        passed=False,
        score=0.3,
        feedback="Answer lacks specific numbers. Search for quantitative data.",
    )


# ──────────────────────────────────────────────
# Agent Setup and Execution
# ──────────────────────────────────────────────


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    # Register tools
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
        description="Evaluate a math expression",
        parameter_schema={"expression": "math expression to evaluate"},
    )

    # Create Reflexion agent with external evaluation
    config = AgentConfig(
        model=model,
        max_iterations=8,
        temperature=0.5,
    )

    agent = ReflexionAgent(
        tools=registry,
        config=config,
        evaluation_fn=evaluate_answer,
        max_reflections=3,
    )

    # Run the agent
    task = "What is the population density of France in people per square kilometer?"
    print(f"Task: {task}")
    print(f"Model: {model}")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nAnswer: {result.answer}")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.iterations_used}")
        print(f"Tools used: {result.tools_used}")

        # Show episodic memory (reflections)
        memory = result.final_context.get("episodic_memory", [])
        if memory:
            print(f"\nReflections ({len(memory)}):")
            for m in memory:
                if isinstance(m, dict):
                    print(f"  - {m.get('reflection', 'N/A')}")
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
