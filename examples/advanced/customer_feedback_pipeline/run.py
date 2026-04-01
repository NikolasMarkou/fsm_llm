"""
Customer Feedback Pipeline -- Multi-Stage Feedback with Handler Analytics
=========================================================================

Demonstrates a multi-stage customer feedback collection pipeline with
handler hooks tracking sentiment progression, response times, and
engagement metrics across a detailed product feedback scenario.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/customer_feedback_pipeline/run.py
"""

import os
import time
from collections import Counter

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming

# Metrics tracked by handlers
metrics: dict = {
    "state_visits": [],
    "extractions_per_state": {},
    "processing_times": [],
    "total_context_keys": [],
}


def build_fsm() -> dict:
    return {
        "name": "CustomerFeedbackPipeline",
        "description": "Multi-stage customer feedback pipeline with analytics",
        "initial_state": "product_rating",
        "persona": (
            "You are a senior customer experience analyst at NovaTech Electronics, "
            "a consumer electronics company that manufactures smartphones, tablets, "
            "laptops, and smart home devices. Be appreciative of feedback and probe "
            "for specific, actionable details."
        ),
        "states": {
            "product_rating": {
                "id": "product_rating",
                "description": "Collect initial product rating",
                "purpose": "Get overall satisfaction score and product identification",
                "extraction_instructions": "Extract 'product_name' (which NovaTech product) and 'overall_rating' (1-5 score).",
                "response_instructions": "Thank the customer for taking time to provide feedback. NovaTech sold 4.2 million units last quarter across 12 product lines, including the Nova Pro X smartphone ($899), Nova Tab Ultra tablet ($649), and Nova Home Hub smart display ($249). Customer satisfaction is trending at 4.1/5.0 with a 4.5/5.0 year-end goal. Common feedback themes: battery life (34% of reviews), camera quality (28%), software updates (22%), build quality (16%). The product team reviews feedback weekly. NovaTech offers a 2-year warranty, 30-day satisfaction guarantee, and a loyalty program with points for accessories and early product access (returns down 15% since launch). Ask which NovaTech product they're reviewing and their overall satisfaction rating (1-5).",
                "transitions": [
                    {
                        "target_state": "detailed_feedback",
                        "description": "Rating received",
                        "conditions": [
                            {
                                "description": "Product and rating provided",
                                "requires_context_keys": ["product_name"],
                                "logic": {"has_context": "product_name"},
                            }
                        ],
                    }
                ],
            },
            "detailed_feedback": {
                "id": "detailed_feedback",
                "description": "Collect detailed feedback on specific aspects",
                "purpose": "Gather specific praise and criticism",
                "extraction_instructions": "Extract 'best_feature' (what they like most) and 'biggest_issue' (main complaint or area for improvement).",
                "response_instructions": "Ask what they like best about the product and what could be improved. Reference the common feedback areas: battery, camera, software, build quality.",
                "transitions": [
                    {
                        "target_state": "improvement_suggestions",
                        "description": "Details collected",
                        "conditions": [
                            {
                                "description": "Feedback provided",
                                "requires_context_keys": ["best_feature"],
                                "logic": {"has_context": "best_feature"},
                            }
                        ],
                    }
                ],
            },
            "improvement_suggestions": {
                "id": "improvement_suggestions",
                "description": "Collect specific improvement suggestions",
                "purpose": "Get actionable product improvement ideas",
                "extraction_instructions": "Extract 'suggestion' (specific improvement idea) and 'would_recommend' (yes or no).",
                "response_instructions": "Ask for a specific suggestion for improvement. Also ask if they would recommend this product to others.",
                "transitions": [
                    {
                        "target_state": "loyalty_check",
                        "description": "Suggestions collected",
                        "conditions": [
                            {
                                "description": "Suggestion provided",
                                "requires_context_keys": ["suggestion"],
                                "logic": {"has_context": "suggestion"},
                            }
                        ],
                    }
                ],
            },
            "loyalty_check": {
                "id": "loyalty_check",
                "description": "Check loyalty program status",
                "purpose": "Offer loyalty program benefits",
                "extraction_instructions": "Extract 'loyalty_member' (yes or no) and 'purchase_frequency' (how often they buy NovaTech products).",
                "response_instructions": "Ask if they're a NovaTech loyalty program member. Ask how often they purchase NovaTech products. Mention loyalty benefits: points, early access, accessories.",
                "transitions": [
                    {
                        "target_state": "summary",
                        "description": "Loyalty checked",
                        "conditions": [
                            {
                                "description": "Loyalty status known",
                                "requires_context_keys": ["loyalty_member"],
                                "logic": {"has_context": "loyalty_member"},
                            }
                        ],
                    }
                ],
            },
            "summary": {
                "id": "summary",
                "description": "Summarize feedback session",
                "purpose": "Present feedback summary and thank customer",
                "extraction_instructions": "None",
                "response_instructions": "Summarize all feedback: product, rating, best feature, biggest issue, suggestion, recommendation status, and loyalty info. Thank them for their valuable input.",
                "transitions": [],
            },
        },
    }


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("=" * 60)
    print("Customer Feedback Pipeline -- Handler Analytics")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    # Register handlers
    fsm.create_handler(
        name="state_tracker",
        timing=HandlerTiming.POST_TRANSITION,
        action=lambda ctx: metrics["state_visits"].append(
            ctx.get("_current_state", "?")
        ),
    )

    fsm.create_handler(
        name="extraction_counter",
        timing=HandlerTiming.POST_PROCESSING,
        action=lambda ctx: metrics["extractions_per_state"].update(
            {
                ctx.get("_current_state", "?"): len(
                    [k for k in ctx if not k.startswith("_")]
                )
            }
        ),
    )

    fsm.create_handler(
        name="context_monitor",
        timing=HandlerTiming.CONTEXT_UPDATE,
        action=lambda ctx: metrics["total_context_keys"].append(
            len([k for k in ctx if not k.startswith("_")])
        ),
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "I'm reviewing the Nova Pro X smartphone. I'd give it a solid 4 out of 5",
        "The camera quality is outstanding, best I've owned. But the battery only lasts about 6 hours with heavy use",
        "I'd love to see a battery optimization mode for power users. And yes, I would definitely recommend it",
        "Yes, I'm a loyalty member. I buy a new NovaTech product about once a year",
    ]

    expected_keys = [
        "product_name",
        "overall_rating",
        "best_feature",
        "biggest_issue",
        "suggestion",
        "would_recommend",
        "loyalty_member",
        "purchase_frequency",
    ]

    for msg in messages:
        print(f"\nYou: {msg}")
        t0 = time.time()
        response = fsm.converse(msg, conv_id)
        elapsed = time.time() - t0
        metrics["processing_times"].append(elapsed)
        print(f"Bot: {response}")
        state = fsm.get_current_state(conv_id)
        print(f"  State: {state} ({elapsed:.1f}s)")

        if fsm.has_conversation_ended(conv_id):
            break

    # Final summary
    print("\n" + "=" * 60)
    print("FEEDBACK SUMMARY")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    extracted = 0
    for key in expected_keys:
        value = data.get(key)
        status = "EXTRACTED" if value is not None else "MISSING"
        if value is not None:
            extracted += 1
        print(f"  {key:25s}: {str(value)[:40]:40s} [{status}]")

    print(
        f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)"
    )

    # Handler metrics
    print("\n" + "=" * 60)
    print("HANDLER ANALYTICS")
    print("=" * 60)
    print(f"  State visits: {metrics['state_visits']}")
    print(f"  Processing times: {[f'{t:.1f}s' for t in metrics['processing_times']]}")
    print(
        f"  Avg processing time: {sum(metrics['processing_times']) / max(len(metrics['processing_times']), 1):.1f}s"
    )
    print(f"  Context growth: {metrics['total_context_keys']}")
    state_counts = Counter(metrics["state_visits"])
    print(f"  State distribution: {dict(state_counts)}")

    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
