"""
Pipeline + Review — PromptChain Generation with MakerChecker QA
================================================================

Demonstrates combining two agent patterns: PromptChain generates a
technical document through a multi-stage pipeline (outline → draft →
format), then MakerChecker reviews it against quality criteria.

This shows how to compose agent patterns sequentially for production
content generation workflows.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/pipeline_review/run.py
"""

import os

from fsm_llm_agents import (
    AgentConfig,
    ChainStep,
    MakerCheckerAgent,
    PromptChainAgent,
)


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # ── Stage 1: PromptChain generates the document ──
    chain = [
        ChainStep(
            step_id="outline",
            name="Outline",
            extraction_instructions=(
                "Extract as JSON:\n"
                '- "sections": list of section titles\n'
                '- "key_topics": list of main topics to cover\n'
                '- "target_audience": who this document is for'
            ),
            response_instructions=(
                "Create a detailed outline for an API documentation page. "
                "Include sections for: Overview, Authentication, Endpoints, "
                "Error Handling, Rate Limits, and Examples. Each section should "
                "have 2-3 bullet points of what to cover."
            ),
        ),
        ChainStep(
            step_id="draft",
            name="Draft",
            extraction_instructions=(
                "Extract as JSON:\n"
                '- "draft_text": the full draft documentation\n'
                '- "word_count": estimated word count\n'
                '- "code_examples_count": number of code examples included'
            ),
            response_instructions=(
                "Using the outline from the previous step, write the full API "
                "documentation draft. Include realistic code examples for each "
                "endpoint. Use clear, technical language appropriate for developers. "
                "Include request/response examples."
            ),
        ),
        ChainStep(
            step_id="format",
            name="Format & Polish",
            extraction_instructions=(
                "Extract as JSON:\n"
                '- "final_doc": the formatted documentation\n'
                '- "improvements": list of improvements made\n'
                '- "readability_score": self-assessed readability 0-1'
            ),
            response_instructions=(
                "Polish the draft documentation. Ensure consistent formatting, "
                "add helpful notes and warnings where appropriate, verify code "
                "examples are complete and correct, and add a table of contents "
                "reference. Output the final polished documentation."
            ),
        ),
    ]

    chain_config = AgentConfig(model=model, max_iterations=20, temperature=0.5)
    chain_agent = PromptChainAgent(chain=chain, config=chain_config)

    task = (
        "Write API documentation for a REST API that manages a todo list "
        "application with CRUD operations, user authentication via JWT tokens, "
        "and pagination support."
    )

    print("=" * 60)
    print("Pipeline + Review — PromptChain + MakerChecker")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Pipeline: {' -> '.join(s.name for s in chain)} -> Review")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    # Run the generation pipeline
    print("\n[Stage 1] Generating documentation via PromptChain...")
    try:
        gen_result = chain_agent.run(task)
        generated_doc = gen_result.answer
        print(f"  Pipeline complete: {gen_result.success}")
        print(f"  Iterations: {gen_result.iterations_used}")
        chain_results = gen_result.final_context.get("chain_results", [])
        print(f"  Steps completed: {len(chain_results)}")
    except Exception as e:
        print(f"  Pipeline error: {e}")
        return

    if not generated_doc or len(generated_doc.strip()) < 50:
        print("  Generated document too short, skipping review.")
        print(f"  Output: {generated_doc}")
        return

    # ── Stage 2: MakerChecker reviews the document ──
    print("\n[Stage 2] Quality review via MakerChecker...")

    review_config = AgentConfig(model=model, max_iterations=20, temperature=0.3)
    reviewer = MakerCheckerAgent(
        maker_instructions=(
            "Present the API documentation provided in the task. "
            "The documentation has already been written — your job is to "
            "present it clearly and make minor formatting improvements only."
        ),
        checker_instructions=(
            "Review the API documentation against these criteria:\n"
            "- Has authentication section with JWT examples\n"
            "- Has at least 3 endpoint descriptions\n"
            "- Includes error handling guidance\n"
            "- Uses consistent formatting\n"
            "- Includes code examples\n"
            "Score 0.0-1.0 and set checker_passed=true if score >= 0.6"
        ),
        config=review_config,
        max_revisions=2,
        quality_threshold=0.6,
    )

    review_task = (
        f"Review and present this API documentation:\n\n{generated_doc[:2000]}"
    )

    try:
        review_result = reviewer.run(review_task)

        print(f"\n{'=' * 60}")
        print("FINAL REVIEWED DOCUMENT")
        print("=" * 60)
        print(f"\n{review_result.answer}")
        print(
            f"\nReview passed: {review_result.final_context.get('checker_passed', False)}"
        )
        print(f"Quality score: {review_result.final_context.get('quality_score', 0)}")
        print(f"Revisions: {review_result.final_context.get('revision_count', 0)}")
        print(
            f"Total iterations (pipeline + review): {gen_result.iterations_used + review_result.iterations_used}"
        )
    except Exception as e:
        print(f"  Review error: {e}")
        print(f"\nFalling back to unreviewed document:\n{generated_doc[:500]}")


if __name__ == "__main__":
    main()
