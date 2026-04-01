"""
Budget Review -- Financial Review with Handler Analytics
========================================================

Demonstrates a departmental budget review process with handler hooks
tracking financial data collection, variance analysis, and approval
workflow across a detailed corporate finance scenario.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/budget_review/run.py
"""

import os
import time
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming


metrics: dict[str, Any] = {
    "review_stages": [],
    "financial_data": [],
    "context_growth": [],
}


def build_fsm() -> dict:
    return {
        "name": "BudgetReviewBot",
        "description": "Departmental budget review with variance tracking",
        "initial_state": "department_info",
        "persona": (
            "You are the Director of Financial Planning and Analysis (FP&A) at Cascade "
            "Industries, a diversified manufacturing conglomerate with $1.2 billion in "
            "annual revenue and 4,200 employees across 12 facilities. Be analytical, "
            "precise with numbers, and focused on variance explanations."
        ),
        "states": {
            "department_info": {
                "id": "department_info",
                "description": "Identify the department under review",
                "purpose": "Set the context for the budget review",
                "extraction_instructions": "Extract 'department_name' (which department) and 'business_unit' (which business unit it belongs to).",
                "response_instructions": "Begin the Q1 budget review. Cascade Industries has 4 business units: Aerospace Components (35% of revenue), Medical Devices (28%), Industrial Equipment (22%), and Consumer Products (15%). Annual budget cycle runs October-December with quarterly reviews. Key metrics: revenue variance (target <5%), COGS (<62% of revenue), SG&A (<18% of revenue), EBITDA margin (>20%), capex adherence (within 10%), working capital DSO (<45 days). Approval thresholds: $500K+ requires VP, $2M+ requires CFO, $5M+ requires board. Inter-department transfers up to $100K approved at director level. Systems: SAP for ERP, Adaptive Insights for budgeting. Ask which department is being reviewed and which business unit it belongs to.",
                "transitions": [
                    {
                        "target_state": "revenue_review",
                        "description": "Department identified",
                        "conditions": [
                            {
                                "description": "Department known",
                                "requires_context_keys": ["department_name"],
                                "logic": {"has_context": "department_name"},
                            }
                        ],
                    }
                ],
            },
            "revenue_review": {
                "id": "revenue_review",
                "description": "Review revenue performance",
                "purpose": "Assess revenue vs plan",
                "extraction_instructions": "Extract 'q1_revenue_actual' (actual Q1 revenue) and 'revenue_variance' (percentage over or under plan).",
                "response_instructions": "Ask about Q1 revenue actual vs plan. What's the variance percentage? Target is within 5% of plan. Ask about the main drivers of any variance.",
                "transitions": [
                    {
                        "target_state": "expense_review",
                        "description": "Revenue reviewed",
                        "conditions": [
                            {
                                "description": "Revenue data provided",
                                "requires_context_keys": ["q1_revenue_actual"],
                                "logic": {"has_context": "q1_revenue_actual"},
                            }
                        ],
                    }
                ],
            },
            "expense_review": {
                "id": "expense_review",
                "description": "Review expense categories",
                "purpose": "Assess spending vs budget",
                "extraction_instructions": "Extract 'total_expenses' (Q1 total spend) and 'largest_variance_category' (expense category with biggest variance).",
                "response_instructions": "Ask about total Q1 expenses vs budget. Which expense category had the largest variance? Track COGS (<62% of revenue) and SG&A (<18% of revenue) targets.",
                "transitions": [
                    {
                        "target_state": "forecast_adjustment",
                        "description": "Expenses reviewed",
                        "conditions": [
                            {
                                "description": "Expenses reported",
                                "requires_context_keys": ["total_expenses"],
                                "logic": {"has_context": "total_expenses"},
                            }
                        ],
                    }
                ],
            },
            "forecast_adjustment": {
                "id": "forecast_adjustment",
                "description": "Discuss forecast adjustments",
                "purpose": "Determine any needed budget changes",
                "extraction_instructions": "Extract 'forecast_change' (increase, decrease, or maintain current forecast) and 'action_items' (key follow-up actions).",
                "response_instructions": "Based on Q1 results, should the full-year forecast be adjusted? Ask about recommended changes and key action items. Mention approval thresholds for budget changes.",
                "transitions": [
                    {
                        "target_state": "review_summary",
                        "description": "Forecast discussed",
                        "conditions": [
                            {
                                "description": "Forecast decision made",
                                "requires_context_keys": ["forecast_change"],
                                "logic": {"has_context": "forecast_change"},
                            }
                        ],
                    }
                ],
            },
            "review_summary": {
                "id": "review_summary",
                "description": "Summarize budget review",
                "purpose": "Present complete review findings",
                "extraction_instructions": "None",
                "response_instructions": "Summarize the Q1 budget review: department, business unit, revenue actual and variance, total expenses, largest variance category, forecast recommendation, and action items. Include overall assessment of financial health.",
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
    print("Budget Review -- Financial Review with Handler Analytics")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    # Register handlers
    fsm.create_handler(
        name="review_stage_tracker",
        timing=HandlerTiming.POST_TRANSITION,
        action=lambda ctx: metrics["review_stages"].append(ctx.get("_current_state", "?")),
    )

    fsm.create_handler(
        name="financial_data_monitor",
        timing=HandlerTiming.CONTEXT_UPDATE,
        action=lambda ctx: metrics["financial_data"].append(
            {k: v for k, v in ctx.items() if any(term in k for term in ["revenue", "expense", "forecast"]) and not k.startswith("_")}
        ),
    )

    fsm.create_handler(
        name="context_tracker",
        timing=HandlerTiming.POST_PROCESSING,
        action=lambda ctx: metrics["context_growth"].append(
            len([k for k in ctx if not k.startswith("_")])
        ),
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "We're reviewing the Engineering department budget under the Aerospace Components business unit",
        "Q1 actual revenue was $48.2 million against a plan of $45 million. That's about 7% over plan due to a new defense contract",
        "Total Q1 expenses were $38.5 million versus budget of $36 million. The largest variance was in R&D spending, up 12% due to prototype development",
        "We recommend increasing the full-year forecast by 5%. Key action item is to reallocate $800K from general SG&A to R&D for Q2",
    ]

    expected_keys = [
        "department_name", "business_unit", "q1_revenue_actual", "revenue_variance",
        "total_expenses", "largest_variance_category", "forecast_change", "action_items",
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
    print("BUDGET REVIEW SUMMARY")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    extracted = 0
    for key in expected_keys:
        value = data.get(key)
        status = "EXTRACTED" if value is not None else "MISSING"
        if value is not None:
            extracted += 1
        print(f"  {key:30s}: {str(value)[:35]:35s} [{status}]")

    print(f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)")

    print("\n" + "=" * 60)
    print("HANDLER ANALYTICS")
    print("=" * 60)
    print(f"  Review stages: {metrics['review_stages']}")
    print(f"  Financial data updates: {len(metrics['financial_data'])}")
    print(f"  Context growth: {metrics['context_growth']}")

    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
