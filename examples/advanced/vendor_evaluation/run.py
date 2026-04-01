"""
Vendor Evaluation -- Scoring Pipeline with Context Tracking
============================================================

Demonstrates a vendor evaluation process with handler hooks tracking
scoring criteria, comparison metrics, and decision progression
across a detailed procurement scenario.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/vendor_evaluation/run.py
"""

import os
import time
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming


metrics: dict[str, Any] = {
    "evaluation_stages": [],
    "scores_tracked": [],
    "context_size": [],
}


def build_fsm() -> dict:
    return {
        "name": "VendorEvaluationBot",
        "description": "Structured vendor evaluation with scoring and tracking",
        "initial_state": "vendor_profile",
        "persona": (
            "You are a senior procurement analyst at Horizon Manufacturing Corporation, "
            "a mid-size industrial manufacturer with $380 million in annual revenue and "
            "operations in 8 countries. Be analytical and fair when evaluating vendors."
        ),
        "states": {
            "vendor_profile": {
                "id": "vendor_profile",
                "description": "Collect vendor identification",
                "purpose": "Identify the vendor being evaluated",
                "extraction_instructions": "Extract 'vendor_name' (company name) and 'vendor_category' (raw materials, components, logistics, or services).",
                "response_instructions": "Begin the vendor evaluation. The company manages 240 active vendor relationships across raw materials, components, logistics, and professional services. Evaluation uses a standardized scorecard with 5 categories: Quality (30%), Price Competitiveness (25%), Delivery Reliability (20%), Financial Stability (15%), and Innovation Capability (10%). Each scored 1-5. Minimum qualifying score is 3.0 weighted average; preferred vendor requires 4.0+; strategic partners (4.5+) get multi-year contracts and early project access. Scorecards reviewed quarterly. New vendor onboarding takes 4-6 weeks (site visits, sample testing, financial review). Supplier diversity goal: 15% spend with minority/women-owned businesses (currently 11.3%). Ask for the vendor's company name and which category they supply.",
                "transitions": [
                    {
                        "target_state": "quality_assessment",
                        "description": "Vendor identified",
                        "conditions": [
                            {
                                "description": "Vendor named",
                                "requires_context_keys": ["vendor_name"],
                                "logic": {"has_context": "vendor_name"},
                            }
                        ],
                    }
                ],
            },
            "quality_assessment": {
                "id": "quality_assessment",
                "description": "Assess quality and delivery",
                "purpose": "Score quality and reliability metrics",
                "extraction_instructions": "Extract 'quality_score' (1-5 rating) and 'delivery_score' (1-5 rating for on-time delivery).",
                "response_instructions": "Ask about the vendor's quality track record and on-time delivery performance. Request scores from 1-5 for each. Quality is weighted 30%, delivery 20%.",
                "transitions": [
                    {
                        "target_state": "financial_review",
                        "description": "Quality assessed",
                        "conditions": [
                            {
                                "description": "Quality scored",
                                "requires_context_keys": ["quality_score"],
                                "logic": {"has_context": "quality_score"},
                            }
                        ],
                    }
                ],
            },
            "financial_review": {
                "id": "financial_review",
                "description": "Review pricing and financial stability",
                "purpose": "Evaluate cost competitiveness and financial health",
                "extraction_instructions": "Extract 'price_score' (1-5 competitiveness rating) and 'financial_stability_score' (1-5 rating).",
                "response_instructions": "Ask about price competitiveness compared to market rates and the vendor's financial stability. Price is weighted 25%, stability 15%.",
                "transitions": [
                    {
                        "target_state": "innovation_check",
                        "description": "Financials reviewed",
                        "conditions": [
                            {
                                "description": "Price scored",
                                "requires_context_keys": ["price_score"],
                                "logic": {"has_context": "price_score"},
                            }
                        ],
                    }
                ],
            },
            "innovation_check": {
                "id": "innovation_check",
                "description": "Evaluate innovation and diversity",
                "purpose": "Assess innovation capability and diversity status",
                "extraction_instructions": "Extract 'innovation_score' (1-5 rating) and 'diversity_certified' (yes or no).",
                "response_instructions": "Ask about the vendor's innovation capabilities and R&D investment. Ask if they hold any diversity certifications (minority-owned, women-owned). Innovation is weighted 10%.",
                "transitions": [
                    {
                        "target_state": "evaluation_summary",
                        "description": "Innovation assessed",
                        "conditions": [
                            {
                                "description": "Innovation scored",
                                "requires_context_keys": ["innovation_score"],
                                "logic": {"has_context": "innovation_score"},
                            }
                        ],
                    }
                ],
            },
            "evaluation_summary": {
                "id": "evaluation_summary",
                "description": "Present evaluation results",
                "purpose": "Calculate weighted score and classify vendor",
                "extraction_instructions": "None",
                "response_instructions": "Calculate the weighted average score and present the vendor evaluation: name, category, all 5 scores, weighted average, diversity status. Classify as: Below threshold (<3.0), Qualified (3.0-3.9), Preferred (4.0-4.4), or Strategic Partner (4.5+).",
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
    print("Vendor Evaluation -- Scoring with Context Tracking")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    # Register handlers
    fsm.create_handler(
        name="stage_tracker",
        timing=HandlerTiming.POST_TRANSITION,
        action=lambda ctx: metrics["evaluation_stages"].append(ctx.get("_current_state", "?")),
    )

    fsm.create_handler(
        name="score_tracker",
        timing=HandlerTiming.CONTEXT_UPDATE,
        action=lambda ctx: metrics["scores_tracked"].append(
            {k: v for k, v in ctx.items() if k.endswith("_score") and not k.startswith("_")}
        ),
    )

    fsm.create_handler(
        name="context_monitor",
        timing=HandlerTiming.POST_PROCESSING,
        action=lambda ctx: metrics["context_size"].append(
            len([k for k in ctx if not k.startswith("_")])
        ),
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "We're evaluating Precision Components Inc., they supply electronic components",
        "Quality is excellent, I'd rate them a 4. Delivery reliability is solid too, a 4",
        "Their pricing is competitive, I'd say a 3. Financial stability looks strong, a 5",
        "Innovation is moderate, maybe a 3. They are a certified women-owned business",
    ]

    expected_keys = [
        "vendor_name", "vendor_category", "quality_score", "delivery_score",
        "price_score", "financial_stability_score", "innovation_score", "diversity_certified",
    ]

    for msg in messages:
        print(f"\nYou: {msg}")
        t0 = time.time()
        response = fsm.converse(msg, conv_id)
        elapsed = time.time() - t0
        print(f"Bot: {response}")
        state = fsm.get_current_state(conv_id)
        print(f"  State: {state} ({elapsed:.1f}s)")

        if fsm.has_conversation_ended(conv_id):
            break

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    extracted = 0
    for key in expected_keys:
        value = data.get(key)
        status = "EXTRACTED" if value is not None else "MISSING"
        if value is not None:
            extracted += 1
        print(f"  {key:30s}: {str(value)[:30]:30s} [{status}]")

    print(f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)")

    print("\n" + "=" * 60)
    print("HANDLER ANALYTICS")
    print("=" * 60)
    print(f"  Evaluation stages: {metrics['evaluation_stages']}")
    print(f"  Scores tracked: {len(metrics['scores_tracked'])} updates")
    print(f"  Context growth: {metrics['context_size']}")

    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
