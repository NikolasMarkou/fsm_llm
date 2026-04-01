"""
Reflexion Agent — Self-Improving Code Generator
=================================================

Demonstrates the Reflexion pattern for code generation: the agent writes
a Python function, an external evaluator tests it, and if tests fail
the agent reflects on the error and retries with lessons learned.

This shows how Reflexion's episodic memory enables progressive improvement
through self-critique and accumulated learning.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/reflexion_code_gen/run.py
"""

import os

from fsm_llm_agents import AgentConfig, EvaluationResult, ReflexionAgent, ToolRegistry


def run_tests(params: dict) -> str:
    """Run test cases against the generated function code."""
    code = params.get("code", "")
    if not code.strip():
        return "Error: No code provided to test."

    # Extract function definition
    try:
        namespace = {}
        exec(code, namespace)
    except Exception as e:
        return f"Syntax/runtime error: {e}"

    # Find the function
    fn = None
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("_"):
            fn = obj
            break

    if fn is None:
        return "Error: No function found in the code."

    # Test cases for fibonacci
    test_cases = [
        (0, 0),
        (1, 1),
        (2, 1),
        (5, 5),
        (10, 55),
        (15, 610),
    ]

    passed = 0
    failures = []
    for inp, expected in test_cases:
        try:
            result = fn(inp)
            if result == expected:
                passed += 1
            else:
                failures.append(f"fibonacci({inp}): expected {expected}, got {result}")
        except Exception as e:
            failures.append(f"fibonacci({inp}): raised {type(e).__name__}: {e}")

    total = len(test_cases)
    if passed == total:
        return f"All {total} tests passed!"
    else:
        report = f"Passed {passed}/{total}.\nFailures:\n" + "\n".join(failures)
        return report


def evaluate_code_quality(context: dict) -> EvaluationResult:
    """Evaluate the generated code for correctness and quality."""
    answer = context.get("final_answer", "")
    observations = context.get("observations", [])

    # Check if tests passed in the observations
    all_passed = any("All" in str(obs) and "passed" in str(obs) for obs in observations)

    # Check code is present
    has_code = "def " in answer or "def " in str(observations)

    # Check for common issues
    has_recursion_limit = "RecursionError" in str(observations)

    if all_passed and has_code:
        return EvaluationResult(
            passed=True,
            score=1.0,
            feedback="Code passes all test cases.",
        )

    feedback_parts = []
    if not has_code:
        feedback_parts.append("No function definition found in the answer.")
    if has_recursion_limit:
        feedback_parts.append(
            "RecursionError detected — add memoization or use iteration."
        )
    if not all_passed:
        # Extract failure details from observations
        for obs in observations:
            if "Failures" in str(obs) or "error" in str(obs).lower():
                feedback_parts.append(f"Test results: {str(obs)[:200]}")
                break

    score = 0.3 if has_code else 0.0
    if any("Passed" in str(obs) for obs in observations):
        score = 0.5

    return EvaluationResult(
        passed=False,
        score=score,
        feedback=" ".join(feedback_parts)
        if feedback_parts
        else "Tests did not all pass.",
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    registry = ToolRegistry()
    registry.register_function(
        run_tests,
        name="run_tests",
        description=(
            "Run test cases against a Python function. Pass the complete "
            "function code as the 'code' parameter. The function should be "
            "named 'fibonacci' and take an integer n, returning the nth "
            "Fibonacci number (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1)."
        ),
        parameter_schema={
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python function code to test",
                },
            }
        },
    )

    config = AgentConfig(
        model=model,
        max_iterations=8,
        temperature=0.5,
    )

    agent = ReflexionAgent(
        tools=registry,
        config=config,
        evaluation_fn=evaluate_code_quality,
        max_reflections=3,
    )

    task = (
        "Write a Python function called 'fibonacci' that takes an integer n "
        "and returns the nth Fibonacci number (0-indexed). fibonacci(0)=0, "
        "fibonacci(1)=1, fibonacci(2)=1, fibonacci(5)=5, fibonacci(10)=55. "
        "Use the run_tests tool to verify your implementation. "
        "Make sure it handles n=0 correctly and is efficient (no naive recursion)."
    )

    print("=" * 60)
    print("Reflexion Agent — Self-Improving Code Generator")
    print("=" * 60)
    print(f"Model: {model}")
    print("Max reflections: 3")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nFinal answer:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")
        print(f"Tools used: {result.tools_used}")

        memory = result.final_context.get("episodic_memory", [])
        if memory:
            print(f"\nReflections ({len(memory)}):")
            for m in memory:
                if isinstance(m, dict):
                    print(f"  - {str(m.get('reflection', 'N/A'))[:100]}")
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
        print(f"  {key:25s}: {str(passed):40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
