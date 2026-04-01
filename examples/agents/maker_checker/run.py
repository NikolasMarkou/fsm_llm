"""
Maker-Checker Agent Example — Quality-Gated Content Generation
===============================================================

Demonstrates the Maker-Checker pattern: a "maker" persona generates
content while a "checker" persona evaluates it against quality criteria.
The loop continues until the checker approves or max revisions are hit.

This is useful for tasks requiring quality assurance, compliance checks,
or multi-perspective review of generated content.

Run:
    export OPENAI_API_KEY=your-key-here
    python examples/agents/maker_checker/run.py

    # Or with Ollama:
    export LLM_MODEL=ollama_chat/qwen3.5:9b
    python examples/agents/maker_checker/run.py
"""

import os

from fsm_llm_agents import AgentConfig, MakerCheckerAgent


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
        max_iterations=10,
        temperature=0.5,
    )

    agent = MakerCheckerAgent(
        maker_instructions=(
            "Write a professional email that is concise (under 150 words), "
            "uses a warm but professional tone, includes a clear subject line, "
            "and ends with a specific call to action."
        ),
        checker_instructions=(
            "Evaluate the email against these criteria and assign a quality "
            "score from 0.0 to 1.0:\n"
            "- Professional tone (not too casual, not too stiff)\n"
            "- Conciseness (under 150 words)\n"
            "- Clear call to action\n"
            "- Proper greeting and sign-off\n"
            "- No spelling or grammar issues\n"
            "Set checker_passed=true only if quality_score >= 0.7"
        ),
        config=config,
        max_revisions=3,
        quality_threshold=0.7,
    )

    task = (
        "Write an email to a client apologizing for a 2-day delay in "
        "delivering their project and proposing a new timeline."
    )
    print(f"Task: {task}")
    print(f"Model: {model}")
    print("Max revisions: 3")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nFinal email:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")

        # Show revision history
        revision_count = result.final_context.get("revision_count", 0)
        checker_passed = result.final_context.get("checker_passed", False)
        quality_score = result.final_context.get("quality_score", "N/A")

        print(f"Revisions: {revision_count}")
        print(f"Checker passed: {checker_passed}")
        print(f"Quality score: {quality_score}")

        # Show checker feedback if available
        feedback = result.final_context.get("checker_feedback", "")
        if feedback:
            preview = str(feedback)[:200]
            print(f"Checker feedback: {preview}")
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
        print(f"  {key:25s}: {str(passed):40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
