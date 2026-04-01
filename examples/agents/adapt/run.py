"""
ADaPT Agent Example -- Adaptive Decomposition and Planning
==========================================================

Demonstrates the ADaPT pattern: attempt a task directly, and if
it's too complex, decompose it into subtasks solved recursively.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/adapt/run.py
"""

import os

from fsm_llm_agents import ADaPTAgent, AgentConfig, ToolRegistry, tool


@tool
def search(query: str) -> str:
    """Search the web for information."""
    q = query.lower()
    facts = {
        "python": "Python is a high-level language. Latest stable: 3.12. Created by Guido van Rossum.",
        "rust": "Rust is a systems language focused on safety. Created by Mozilla Research.",
        "javascript": "JavaScript is the language of the web. Runs in browsers and Node.js.",
        "go": "Go (Golang) is a statically typed language by Google. Known for concurrency.",
        "java": "Java is a class-based OOP language. Runs on the JVM. Used in enterprise.",
    }
    for key, value in facts.items():
        if key in q:
            return value
    return f"No results for: {query}"


@tool
def summarize(text: str) -> str:
    """Summarize a block of text into key points."""
    words = text.split()
    if len(words) > 20:
        return f"Summary ({len(words)} words): {' '.join(words[:20])}..."
    return f"Summary: {text}"


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    registry = ToolRegistry()
    registry.register(search._tool_definition)
    registry.register(summarize._tool_definition)

    config = AgentConfig(model=model, max_iterations=7, temperature=0.7)
    agent = ADaPTAgent(tools=registry, config=config, max_depth=2)

    task = (
        "Summarize the key differences between Python and Rust for building web APIs."
    )

    print("=" * 60)
    print("ADaPT Agent -- Adaptive Decomposition")
    print("=" * 60)
    print(f"Model: {model}")
    print("Max depth: 2")
    print(f"Task: {task}\n")

    try:
        result = agent.run(task)
        print(f"\nAnswer: {result.answer}")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.iterations_used}")
        print(f"Tools used: {result.tools_used}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
