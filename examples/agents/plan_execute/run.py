"""
Plan-and-Execute Agent Example — Research Planner
==================================================

Demonstrates the Plan-and-Execute pattern: separates strategic planning
from tactical execution. The planner LLM generates a full decomposition
upfront, then the executor handles each step sequentially. If a step
fails, the agent replans.

Tools:
  - search: Simulates a web search
  - summarize: Simulates text summarization

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/plan_execute/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/plan_execute/run.py
"""

import os

from fsm_llm.stdlib.agents import AgentConfig, PlanExecuteAgent, ToolRegistry

# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────


def search(params: dict) -> str:
    """Simulate a web search engine."""
    query = params.get("query", "")
    q = query.lower()

    results = {
        "python web frameworks": (
            "Popular Python web frameworks: Django (full-featured, ORM, admin), "
            "Flask (lightweight, extensible), FastAPI (async, OpenAPI, type hints), "
            "Starlette (ASGI, lightweight)."
        ),
        "django vs flask": (
            "Django: batteries-included, best for complex apps. "
            "Flask: minimalist, best for microservices and APIs."
        ),
        "fastapi performance": (
            "FastAPI benchmarks show ~15,000 req/s for simple JSON endpoints, "
            "comparable to Node.js and Go web frameworks."
        ),
    }

    for key, value in results.items():
        if key in q:
            return value
    return f"Search results for: {query} — Various articles found."


def summarize(params: dict) -> str:
    """Simulate text summarization."""
    text = params.get("text", "")
    if len(text) > 100:
        return f"Summary: {text[:100]}..."
    return f"Summary: {text}"


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
        description="Search the web for information about a topic",
        parameter_schema={"query": "search query string"},
    )
    registry.register_function(
        summarize,
        name="summarize",
        description="Summarize a piece of text",
        parameter_schema={"text": "text to summarize"},
    )

    # Create Plan-and-Execute agent
    config = AgentConfig(
        model=model,
        max_iterations=10,
        temperature=0.3,
    )

    agent = PlanExecuteAgent(
        tools=registry,
        config=config,
        max_replans=2,
    )

    # Run the agent
    task = (
        "Compare the top 3 Python web frameworks (Django, Flask, FastAPI) "
        "and recommend one for building a REST API microservice."
    )
    print(f"Task: {task}")
    print(f"Model: {model}")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nAnswer: {result.answer}")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.iterations_used}")
        print(f"Tools used: {result.tools_used}")

        # Show plan steps
        steps = result.final_context.get("plan_steps", [])
        if steps:
            print(f"\nPlan steps ({len(steps)}):")
            for s in steps:
                if isinstance(s, dict):
                    status = s.get("status", "?")
                    desc = s.get("description", "N/A")
                    print(f"  [{status}] {desc}")
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
