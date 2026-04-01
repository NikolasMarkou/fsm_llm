"""
Multi-Tool Recovery Example -- Error Handling and Retry
=======================================================

Tests agent resilience when tools fail. Some tools are designed
to fail on first call, testing whether the agent can recover
and try alternative approaches.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/multi_tool_recovery/run.py
"""

import os

from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry, tool

_call_counts: dict[str, int] = {}


@tool
def flaky_search(query: str) -> str:
    """Search for information (may fail intermittently)."""
    _call_counts["flaky_search"] = _call_counts.get("flaky_search", 0) + 1
    if _call_counts["flaky_search"] == 1:
        raise ConnectionError("Search service temporarily unavailable")
    return f"Search results for '{query}': Found 3 relevant articles about {query}."


@tool
def reliable_lookup(topic: str) -> str:
    """Look up facts from a reliable knowledge base."""
    facts = {
        "python": "Python 3.12 released Oct 2023. Key features: f-strings improvements, type parameter syntax.",
        "machine learning": "ML is a subset of AI. Common frameworks: TensorFlow, PyTorch, scikit-learn.",
        "web development": "Modern web dev uses frameworks like React, Vue, Django, FastAPI.",
    }
    for key, value in facts.items():
        if key in topic.lower():
            return value
    return f"No specific facts found for: {topic}"


@tool
def always_fails(input_text: str) -> str:
    """A broken tool that always raises an error."""
    raise RuntimeError("This tool is permanently broken and should not be retried")


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    registry = ToolRegistry()
    registry.register(flaky_search._tool_definition)
    registry.register(reliable_lookup._tool_definition)
    registry.register(always_fails._tool_definition)

    config = AgentConfig(model=model, max_iterations=5, temperature=0.7)
    agent = ReactAgent(tools=registry, config=config)

    task = "Find information about Python programming and its latest features."

    print("=" * 60)
    print("Multi-Tool Recovery -- Error Handling Test")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print("Note: flaky_search fails on first call, always_fails always errors")
    print(f"Task: {task}\n")

    try:
        result = agent.run(task)
        print(f"\nAnswer: {result.answer}")
        print(f"Success: {result.success}")
        print(f"Tools used: {result.tools_used}")
        print(f"Iterations: {result.iterations_used}")
        print(f"\nTool call counts: {_call_counts}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Tool call counts: {_call_counts}")


if __name__ == "__main__":
    main()
