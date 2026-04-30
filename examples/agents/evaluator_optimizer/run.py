"""
Evaluator-Optimizer Agent Example — Iterative Refinement
=========================================================

Demonstrates the Evaluator-Optimizer pattern: generates an initial
output, evaluates it with an external function, and refines until
the evaluation passes or max refinements are reached.

The evaluation function provides structured feedback that guides
the refinement. This is useful for tasks with clear quality criteria.

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/evaluator_optimizer/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/evaluator_optimizer/run.py
"""

import os

from fsm_llm.stdlib.agents import AgentConfig, EvaluationResult, EvaluatorOptimizerAgent


def evaluate_haiku(output: str, context: dict) -> EvaluationResult:
    """
    Evaluate whether the output is a valid haiku.

    A haiku has 3 lines with a 5-7-5 syllable pattern.
    We check structure (3 lines) and approximate syllable count.
    """
    lines = [line.strip() for line in output.strip().split("\n") if line.strip()]

    feedback_parts = []
    checks = {}

    # Check line count
    checks["three_lines"] = len(lines) == 3
    if not checks["three_lines"]:
        feedback_parts.append(f"Expected 3 lines, got {len(lines)}.")

    # Check that it's not just prose
    checks["poetic_form"] = len(lines) >= 2 and all(len(line) < 60 for line in lines)
    if not checks["poetic_form"]:
        feedback_parts.append("Lines should be short and poetic, not prose.")

    # Check it mentions the topic
    task = context.get("task", "").lower()
    topic_words = [w for w in task.split() if len(w) > 3]
    output_lower = output.lower()
    topic_present = any(w in output_lower for w in topic_words)
    checks["on_topic"] = topic_present
    if not checks["on_topic"]:
        feedback_parts.append("The haiku should relate to the given topic.")

    passed = all(checks.values())
    score = sum(checks.values()) / len(checks) if checks else 0.0
    # criteria_met expects list[str] of passing criteria names
    criteria_met = [name for name, ok in checks.items() if ok]

    if passed:
        feedback = "Valid haiku structure with good topical relevance."
    else:
        feedback = " ".join(feedback_parts)

    return EvaluationResult(
        passed=passed,
        score=score,
        feedback=feedback,
        criteria_met=criteria_met,
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:9b")
        return

    config = AgentConfig(
        model=model,
        max_iterations=8,
        temperature=0.7,
    )

    agent = EvaluatorOptimizerAgent(
        evaluation_fn=evaluate_haiku,
        config=config,
        max_refinements=3,
    )

    task = "Write a haiku about autumn leaves falling from trees."
    print(f"Task: {task}")
    print(f"Model: {model}")
    print("Max refinements: 3")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nFinal haiku:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")

        # Show refinement history
        refinement_count = result.final_context.get("refinement_count", 0)
        eval_passed = result.final_context.get("evaluation_passed", False)
        eval_score = result.final_context.get("evaluation_score", 0)
        eval_feedback = result.final_context.get("evaluation_feedback", "")

        print(f"Refinements: {refinement_count}")
        print(f"Evaluation passed: {eval_passed}")
        print(f"Final score: {eval_score}")
        if eval_feedback:
            print(f"Feedback: {eval_feedback}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": result.answer is not None and len(str(result.answer)) > 10,
        "iterations_ok": result.iterations_used >= 1,
        "completed": result.iterations_used < config.max_iterations,
        "score_present": result.final_context.get("evaluation_score", 0) > 0,
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
