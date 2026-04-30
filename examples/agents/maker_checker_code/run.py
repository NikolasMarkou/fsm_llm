"""
Maker-Checker for Code Review
===============================

Demonstrates the MakerChecker pattern for code generation and review.
The "maker" writes a Python class implementation while the "checker"
reviews it against coding standards and correctness criteria.

This simulates a real-world code review workflow where the checker
acts as a senior developer reviewing a pull request.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/maker_checker_code/run.py
"""

import os

from fsm_llm.stdlib.agents import AgentConfig, MakerCheckerAgent


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    config = AgentConfig(
        model=model,
        max_iterations=10,
        temperature=0.3,
    )

    agent = MakerCheckerAgent(
        maker_instructions=(
            "Write a Python class that implements a thread-safe LRU cache. "
            "Requirements:\n"
            "- Use `collections.OrderedDict` for O(1) operations\n"
            "- Thread-safe using `threading.Lock`\n"
            "- `get(key)` returns value or None, moves to most recent\n"
            "- `put(key, value)` adds/updates entry, evicts oldest if at capacity\n"
            "- `size` property returns current number of entries\n"
            "- Include type hints and a docstring\n"
            "- Keep it under 40 lines of code"
        ),
        checker_instructions=(
            "Review the Python code as a senior developer. Score 0.0-1.0:\n"
            "- Correctness: Does it implement LRU eviction properly? (+0.2)\n"
            "- Thread safety: Are all shared state accesses protected by lock? (+0.2)\n"
            "- Type hints: Are parameters and return types annotated? (+0.15)\n"
            "- Docstring: Does the class have a clear docstring? (+0.15)\n"
            "- Code quality: Clean, readable, follows PEP 8 conventions? (+0.15)\n"
            "- Edge cases: Handles zero capacity, None values, duplicate keys? (+0.15)\n"
            "Set checker_passed=true only if quality_score >= 0.7. "
            "Provide specific feedback on any issues found."
        ),
        config=config,
        max_revisions=3,
        quality_threshold=0.7,
    )

    task = (
        "Implement a thread-safe LRU (Least Recently Used) cache class in Python. "
        "It should support get and put operations with O(1) time complexity, "
        "automatically evict the least recently used item when capacity is reached, "
        "and be safe for concurrent access from multiple threads."
    )

    print("=" * 60)
    print("Maker-Checker — Code Review Pattern")
    print("=" * 60)
    print(f"Model: {model}")
    print("Max revisions: 3")
    print("Quality threshold: 0.7")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nGenerated Code:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")

        revision_count = result.final_context.get("revision_count", 0)
        checker_passed = result.final_context.get("checker_passed", False)
        quality_score = result.final_context.get("quality_score", 0)

        print(f"Revisions: {revision_count}")
        print(f"Checker passed: {checker_passed}")
        print(f"Quality score: {quality_score}")

        feedback = result.final_context.get("checker_feedback", "")
        if feedback:
            print(f"Reviewer feedback: {str(feedback)[:300]}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": result.answer is not None and len(str(result.answer)) > 10,
        "iterations_ok": result.iterations_used >= 1,
        "quality_score": float(result.final_context.get("quality_score", 0) or 0) > 0,
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
