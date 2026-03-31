"""
Meta-Builder with Quality Review Loop
=======================================

Demonstrates composing the MetaBuilderAgent with a MakerCheckerAgent
to automatically review and refine generated FSM artifacts.

Pipeline:
  1. MetaBuilder generates an FSM definition programmatically
  2. MakerChecker reviews it against quality criteria
  3. If review fails, the MakerChecker suggests improvements

This shows how to use the programmatic (non-interactive) meta builder
API combined with quality assurance patterns.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/meta/meta_review_loop/run.py
"""

import json
import os

from fsm_llm_agents import (
    AgentConfig,
    MakerCheckerAgent,
    MetaBuilderAgent,
)
from fsm_llm_agents.definitions import MetaBuilderConfig


def main():
    model = os.environ.get("LLM_MODEL", "")
    if not model:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = "gpt-4o-mini" if api_key else "ollama_chat/qwen3.5:4b"

    print("=" * 60)
    print("Meta-Builder + MakerChecker Quality Review")
    print("=" * 60)
    print(f"Model: {model}")
    print("-" * 60)

    # ── Stage 1: Programmatic FSM Generation ──
    print("\n[Stage 1] Generating FSM definition via MetaBuilderAgent...")

    build_config = MetaBuilderConfig(model=model, max_iterations=20, temperature=0.5)

    try:
        builder = MetaBuilderAgent(config=build_config)
        spec = (
            "Build an FSM for a customer feedback chatbot. States: "
            "greeting (welcome, ask about purchase), "
            "collect_rating (get 1-5 rating), "
            "handle_complaint (if rating<=3, gather complaints), "
            "ask_testimonial (if rating>=4, collect testimonial), "
            "thank_you (thank and end)."
        )

        build_result = builder.run(spec)

        # Use the MetaBuilderResult.artifact field (dict) directly
        artifact = getattr(build_result, "artifact", None)
        if artifact and isinstance(artifact, dict) and artifact.get("states"):
            fsm_result = artifact
            fsm_json = json.dumps(fsm_result, indent=2)
            states = fsm_result.get("states", {})
            state_count = len(states) if isinstance(states, dict) else len(states)
            print(f"  FSM generated: {state_count} states")
            print(f"  Name: {fsm_result.get('name', 'unnamed')}")
        else:
            fsm_json = str(build_result.answer)
            print("  FSM generated (raw output)")
            raise ValueError("No structured artifact returned")
    except Exception as e:
        print(f"  Builder error: {e}")
        # Create a fallback FSM for the review stage
        fsm_result = {
            "name": "FeedbackBot",
            "description": "Customer feedback collection chatbot",
            "initial_state": "greeting",
            "persona": "A friendly customer service representative collecting feedback",
            "states": {
                "greeting": {
                    "id": "greeting",
                    "description": "Welcome the customer",
                    "purpose": "Greet and ask about recent purchase",
                    "extraction_instructions": "Extract customer_name if mentioned",
                    "response_instructions": "Greet warmly and ask about their recent purchase experience",
                    "transitions": [
                        {
                            "target_state": "collect_rating",
                            "description": "Customer responds about purchase",
                            "priority": 100,
                            "conditions": [],
                        }
                    ],
                },
                "collect_rating": {
                    "id": "collect_rating",
                    "description": "Get satisfaction rating",
                    "purpose": "Collect 1-5 rating",
                    "extraction_instructions": "Extract satisfaction_rating as integer 1-5",
                    "response_instructions": "Ask for a rating from 1 to 5",
                    "transitions": [
                        {
                            "target_state": "handle_complaint",
                            "description": "Low rating",
                            "priority": 100,
                            "conditions": [
                                {
                                    "description": "Rating is 3 or below",
                                    "requires_context_keys": ["satisfaction_rating"],
                                    "logic": {
                                        "<=": [{"var": "satisfaction_rating"}, 3]
                                    },
                                }
                            ],
                        },
                        {
                            "target_state": "ask_testimonial",
                            "description": "High rating",
                            "priority": 90,
                            "conditions": [
                                {
                                    "description": "Rating is 4 or above",
                                    "requires_context_keys": ["satisfaction_rating"],
                                    "logic": {
                                        ">=": [{"var": "satisfaction_rating"}, 4]
                                    },
                                }
                            ],
                        },
                    ],
                },
                "handle_complaint": {
                    "id": "handle_complaint",
                    "description": "Address complaints",
                    "purpose": "Gather complaint details for escalation",
                    "extraction_instructions": "Extract complaint_details",
                    "response_instructions": "Empathize and ask for specific feedback about what went wrong",
                    "transitions": [
                        {
                            "target_state": "thank_you",
                            "description": "Complaint recorded",
                            "priority": 100,
                            "conditions": [],
                        }
                    ],
                },
                "ask_testimonial": {
                    "id": "ask_testimonial",
                    "description": "Request testimonial",
                    "purpose": "Collect positive testimonial",
                    "extraction_instructions": "Extract testimonial_text",
                    "response_instructions": "Thank them and ask if they'd share a brief testimonial",
                    "transitions": [
                        {
                            "target_state": "thank_you",
                            "description": "Testimonial collected or declined",
                            "priority": 100,
                            "conditions": [],
                        }
                    ],
                },
                "thank_you": {
                    "id": "thank_you",
                    "description": "End conversation",
                    "purpose": "Thank customer and close",
                    "extraction_instructions": "No extraction needed",
                    "response_instructions": "Thank the customer warmly for their feedback",
                    "transitions": [],
                },
            },
        }
        fsm_json = json.dumps(fsm_result, indent=2)
        print(f"  Using fallback FSM: {len(fsm_result['states'])} states")

    # ── Stage 2: MakerChecker Review ──
    print("\n[Stage 2] Reviewing FSM quality via MakerChecker...")

    review_config = AgentConfig(model=model, max_iterations=20, temperature=0.3)
    reviewer = MakerCheckerAgent(
        maker_instructions=(
            "Present the FSM definition provided in the task. "
            "If there are obvious issues (missing fields, broken transitions), "
            "fix them. Otherwise present the FSM as-is."
        ),
        checker_instructions=(
            "Review the FSM definition against these quality criteria:\n"
            "- Has at least 3 states (score +0.2)\n"
            "- Has an initial_state that exists in states (score +0.2)\n"
            "- All transitions reference valid target states (score +0.2)\n"
            "- States have extraction_instructions and response_instructions (score +0.2)\n"
            "- Has a terminal state with no transitions (score +0.2)\n"
            "Set checker_passed=true if quality_score >= 0.6"
        ),
        config=review_config,
        max_revisions=2,
        quality_threshold=0.6,
    )

    review_task = f"Review this FSM definition for quality:\n\n{fsm_json[:3000]}"

    try:
        review_result = reviewer.run(review_task)

        print(f"\n{'=' * 60}")
        print("REVIEWED FSM DEFINITION")
        print("=" * 60)
        print(f"\n{review_result.answer[:1000]}")
        print(
            f"\nReview passed: {review_result.final_context.get('checker_passed', False)}"
        )
        print(f"Quality score: {review_result.final_context.get('quality_score', 0)}")
        print(f"Revisions: {review_result.final_context.get('revision_count', 0)}")
        print(f"Total iterations: {review_result.iterations_used}")

        feedback = review_result.final_context.get("checker_feedback", "")
        if feedback:
            print(f"Reviewer feedback: {str(feedback)[:300]}")
    except Exception as e:
        print(f"  Review error: {e}")
        print(f"\nOriginal FSM (unreviewed):\n{fsm_json[:500]}")


if __name__ == "__main__":
    main()
