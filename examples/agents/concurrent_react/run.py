"""
Concurrent ReAct Example -- Multiple Independent Agent Runs
============================================================

Demonstrates running multiple ReAct agents concurrently using
threading, each solving a different task with the same tools.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/concurrent_react/run.py
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fsm_llm_agents import AgentConfig, ReactAgent, ToolRegistry, tool


@tool
def lookup(topic: str) -> str:
    """Look up facts about a topic."""
    facts = {
        "python": "Python: high-level language, dynamic typing, huge ecosystem, great for ML/AI.",
        "rust": "Rust: systems language, memory safety without GC, fast as C/C++.",
        "javascript": "JavaScript: web language, event-driven, runs everywhere (browser + server).",
        "solar": "Solar energy: photovoltaic cells, 20-25% efficiency, costs dropped 89% since 2010.",
        "wind": "Wind energy: turbines convert kinetic energy, onshore and offshore, 45% capacity factor.",
        "nuclear": "Nuclear energy: fission reactors, high energy density, low carbon, safety concerns.",
    }
    for key, value in facts.items():
        if key in topic.lower():
            return value
    return (
        f"General info on {topic}: a notable subject with various aspects to explore."
    )


def run_agent_task(agent: ReactAgent, task: str, task_id: int) -> dict:
    """Run a single agent task and return results."""
    start = time.monotonic()
    try:
        result = agent.run(task)
        elapsed = time.monotonic() - start
        return {
            "task_id": task_id,
            "task": task,
            "answer": result.answer[:200],
            "success": result.success,
            "tools_used": result.tools_used,
            "iterations": result.iterations_used,
            "time": elapsed,
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "task_id": task_id,
            "task": task,
            "answer": None,
            "success": False,
            "error": str(e),
            "time": elapsed,
        }


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    registry = ToolRegistry()
    registry.register(lookup._tool_definition)

    tasks = [
        "What are the key features of Python as a programming language?",
        "Compare solar and wind energy sources.",
        "What makes Rust unique among programming languages?",
    ]

    print("=" * 60)
    print("Concurrent ReAct -- Parallel Agent Execution")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tasks: {len(tasks)}")
    print()

    overall_start = time.monotonic()

    # Each task gets its own agent instance (thread-safe)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for i, task in enumerate(tasks):
            config = AgentConfig(model=model, max_iterations=6, temperature=0.7)
            agent = ReactAgent(tools=registry, config=config)
            future = executor.submit(run_agent_task, agent, task, i)
            futures[future] = i

        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    overall_time = time.monotonic() - overall_start

    # Sort by task_id for display
    results.sort(key=lambda r: r["task_id"])

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    for r in results:
        print(f"\nTask {r['task_id'] + 1}: {r['task']}")
        if r["success"]:
            print(f"  Answer: {r['answer']}")
            print(f"  Tools: {r.get('tools_used', [])}")
            print(f"  Iterations: {r.get('iterations', '?')}")
        else:
            print(f"  Error: {r.get('error', 'Unknown')}")
        print(f"  Time: {r['time']:.2f}s")

    print(f"\nTotal wall-clock time: {overall_time:.2f}s")
    sum_time = sum(r["time"] for r in results)
    print(f"Sum of individual times: {sum_time:.2f}s")
    if sum_time > 0:
        print(f"Parallelism speedup: {sum_time / overall_time:.1f}x")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    successes = sum(1 for r in results if r["success"])
    answers_present = sum(
        1 for r in results if r.get("answer") and len(str(r["answer"])) > 10
    )
    checks = {
        "all_tasks_succeeded": successes == len(tasks),
        "all_answers_present": answers_present == len(tasks),
        "parallelism_achieved": len(results) == len(tasks),
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {str(passed):40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
