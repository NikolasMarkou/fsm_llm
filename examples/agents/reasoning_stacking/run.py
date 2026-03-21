"""
Method 4: Reasoning via FSM Stacking (ReasoningReactAgent)
============================================================

Uses the new ``ReasoningReactAgent`` class which auto-registers a
``reason`` pseudo-tool. When the LLM selects ``reason``, the agent
invokes ``ReasoningEngine`` via FSM stacking (push_fsm / pop_fsm).
Results are stored under namespaced ``ReasoningIntegrationKeys``.

The agent autonomously decides when to use structured reasoning
versus regular tools based on task complexity.

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/reasoning_stacking/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/reasoning_stacking/run.py
"""

import os

# Try to import ReasoningReactAgent
try:
    from fsm_llm_agents import AgentConfig, ReasoningReactAgent, ToolRegistry

    _HAS_AGENT = True
except ImportError:
    _HAS_AGENT = False

# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────


def search(params: dict) -> str:
    """Search for factual information."""
    query = params.get("query", "")
    q = query.lower()

    results = {
        "population of france": "France: ~68.4 million people (2024).",
        "speed of light": "Speed of light: 299,792,458 m/s.",
        "boiling point": "Water boils at 100C (212F) at sea level.",
        "prime number": "A prime is a natural number > 1 divisible only by 1 and itself.",
    }

    for key, value in results.items():
        if key in q:
            return value
    return f"No results for '{query}'."


def calculate(params: dict) -> str:
    """Evaluate a math expression."""
    expression = params.get("expression", "")
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters: {expression}"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set OPENAI_API_KEY or use Ollama (LLM_MODEL=ollama_chat/...)")
        return

    if not _HAS_AGENT:
        print("This example requires fsm_llm_agents and fsm_llm_reasoning.")
        print("Install with: pip install fsm-llm[agents,reasoning]")
        return

    # Build tool registry (reason is auto-registered)
    registry = ToolRegistry()
    registry.register_function(
        search,
        name="search",
        description="Search the web for factual information",
        parameter_schema={
            "properties": {"query": {"type": "string", "description": "Search query"}}
        },
    )
    registry.register_function(
        calculate,
        name="calculate",
        description="Evaluate a mathematical expression",
        parameter_schema={
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            }
        },
    )

    config = AgentConfig(model=model, max_iterations=12, temperature=0.7)
    agent = ReasoningReactAgent(tools=registry, config=config)

    print("=" * 60)
    print("Method 4: Reasoning via FSM Stacking")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print("  (reason tool auto-registered by ReasoningReactAgent)")
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

            # Show reasoning integration results if present
            r_result = result.final_context.get("reasoning_integration_result")
            if r_result:
                print(f"Reasoning result: {r_result[:200]}...")
                print(
                    f"Reasoning type: {result.final_context.get('reasoning_integration_type_used')}"
                )
        except Exception as e:
            print(f"Error: {e}")
        print()


if __name__ == "__main__":
    main()
