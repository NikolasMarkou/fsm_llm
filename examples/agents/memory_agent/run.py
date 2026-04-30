"""
Memory Agent Example -- WorkingMemory with ReAct
=================================================

Demonstrates an agent that uses WorkingMemory tools (remember,
recall, forget, list_memories) to persist information across
reasoning steps.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/memory_agent/run.py
"""

import os

from fsm_llm.memory import WorkingMemory
from fsm_llm.stdlib.agents import (
    AgentConfig,
    ReactAgent,
    ToolRegistry,
    create_memory_tools,
)


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Create shared working memory
    memory = WorkingMemory()

    # Create memory management tools
    memory_tools = create_memory_tools(memory)

    # Build registry with memory tools
    registry = ToolRegistry()
    for tool_def in memory_tools:
        registry.register(tool_def)

    config = AgentConfig(model=model, max_iterations=5, temperature=0.7)
    agent = ReactAgent(tools=registry, config=config)

    print("=" * 60)
    print("Memory Agent -- WorkingMemory + ReAct")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print()

    # Task 1: Remember some facts
    task1 = (
        "Use the remember tool to store: deadline is March 15th "
        "and client is Acme Corp."
    )
    print(f"Task 1: {task1}")
    print("-" * 40)
    try:
        result1 = agent.run(task1)
        print(f"Answer: {result1.answer}")
        print(f"Tools used: {result1.tools_used}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Task 2: Recall from memory
    task2 = "Use recall tool to retrieve what you stored about the project."
    print(f"Task 2: {task2}")
    print("-" * 40)
    try:
        result2 = agent.run(task2)
        print(f"Answer: {result2.answer}")
        print(f"Tools used: {result2.tools_used}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Show final memory state
    print("=" * 60)
    print("Final Memory State:")
    print("=" * 60)
    for buffer_name in memory.list_buffers():
        data = memory.get_buffer(buffer_name)
        if data:
            print(f"  [{buffer_name}]")
            for key, value in data.items():
                print(f"    {key} = {value}")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    try:
        r1_answer = result1.answer is not None and len(str(result1.answer)) > 10
        r1_tools = len(result1.tools_used) > 0
    except Exception:
        r1_answer = False
        r1_tools = False
    try:
        r2_answer = result2.answer is not None and len(str(result2.answer)) > 10
        r2_tools = len(result2.tools_used) > 0
    except Exception:
        r2_answer = False
        r2_tools = False
    checks = {
        "task1_answer_present": r1_answer,
        "task1_tools_called": r1_tools,
        "task2_answer_present": r2_answer,
        "task2_tools_called": r2_tools,
        "memory_populated": len(memory.list_buffers()) > 0,
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
