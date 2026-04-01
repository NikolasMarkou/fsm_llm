"""
Loan Assessment -- Context Tracking with Handler Analytics
==========================================================

Demonstrates a comprehensive loan assessment process with handler
hooks monitoring risk scoring, document verification progress,
and decision tracking across a detailed financial scenario.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/loan_assessment/run.py
"""

import os
import time
from typing import Any

from fsm_llm import API
from fsm_llm.handlers import HandlerTiming

metrics: dict[str, Any] = {
    "risk_signals": [],
    "state_transitions": [],
    "extraction_counts": [],
}


def build_fsm() -> dict:
    return {
        "name": "LoanAssessmentBot",
        "description": "Comprehensive loan assessment with risk tracking",
        "initial_state": "applicant_profile",
        "persona": (
            "You are a senior loan officer at Pacific Coast Federal Credit Union, "
            "a member-owned financial institution with $4.8 billion in assets and "
            "180,000 members across Washington, Oregon, and California. Be professional, "
            "transparent about rates, and helpful with financial guidance."
        ),
        "states": {
            "applicant_profile": {
                "id": "applicant_profile",
                "description": "Collect applicant identity and loan type",
                "purpose": "Identify the applicant and desired loan product",
                "extraction_instructions": "Extract 'applicant_name' (full name) and 'loan_type' (personal, auto, heloc, or mortgage).",
                "response_instructions": "Welcome the member to Pacific Coast FCU loan services. The credit union offers personal loans ($1,000-$50,000, 7.9%-15.9% APR), auto loans (new: 4.9%-8.9%, used: 5.9%-10.9%), HELOCs ($25,000-$500,000, prime + 0.5%-2.0%), and mortgages (15/20/30 year terms). Approval criteria: minimum credit score 640 personal, 620 auto, 680 HELOC, 700 mortgage. DTI must be below 43%. Employment verification requires 2+ years at current employer or 5+ years in same field. Rate discounts: 0.25% for direct deposit, 0.25% for auto-pay, 0.50% for members with 3+ accounts. Processing: 1-3 days personal, 2-5 auto, 30-45 mortgage. All applications include a complimentary credit review. NCUA-insured up to $250,000. Ask for their name and which loan product interests them.",
                "transitions": [
                    {
                        "target_state": "financial_info",
                        "description": "Profile collected",
                        "conditions": [
                            {
                                "description": "Name and loan type known",
                                "requires_context_keys": ["applicant_name"],
                                "logic": {"has_context": "applicant_name"},
                            }
                        ],
                    }
                ],
            },
            "financial_info": {
                "id": "financial_info",
                "description": "Collect financial details",
                "purpose": "Assess income and credit standing",
                "extraction_instructions": "Extract 'annual_income' (yearly income amount) and 'credit_score_range' (excellent, good, fair, or poor).",
                "response_instructions": "Ask about their annual income and approximate credit score range. Mention minimum score requirements for their loan type.",
                "transitions": [
                    {
                        "target_state": "loan_details",
                        "description": "Financial info collected",
                        "conditions": [
                            {
                                "description": "Income provided",
                                "requires_context_keys": ["annual_income"],
                                "logic": {"has_context": "annual_income"},
                            }
                        ],
                    }
                ],
            },
            "loan_details": {
                "id": "loan_details",
                "description": "Collect specific loan parameters",
                "purpose": "Determine loan amount and term",
                "extraction_instructions": "Extract 'loan_amount' (desired amount) and 'loan_term' (desired repayment period).",
                "response_instructions": "Ask for the desired loan amount and preferred repayment term. Provide rate ranges for their loan type and mention available discounts.",
                "transitions": [
                    {
                        "target_state": "employment",
                        "description": "Loan details set",
                        "conditions": [
                            {
                                "description": "Amount provided",
                                "requires_context_keys": ["loan_amount"],
                                "logic": {"has_context": "loan_amount"},
                            }
                        ],
                    }
                ],
            },
            "employment": {
                "id": "employment",
                "description": "Verify employment",
                "purpose": "Confirm employment stability",
                "extraction_instructions": "Extract 'employer_name' (current employer) and 'years_employed' (years at current job).",
                "response_instructions": "Ask about their current employer and how long they've been there. Mention the 2+ year requirement or 5+ years in same field alternative.",
                "transitions": [
                    {
                        "target_state": "assessment_result",
                        "description": "Employment verified",
                        "conditions": [
                            {
                                "description": "Employer known",
                                "requires_context_keys": ["employer_name"],
                                "logic": {"has_context": "employer_name"},
                            }
                        ],
                    }
                ],
            },
            "assessment_result": {
                "id": "assessment_result",
                "description": "Present preliminary assessment",
                "purpose": "Summarize application and provide initial assessment",
                "extraction_instructions": "None",
                "response_instructions": "Summarize the loan application: name, loan type, income, credit range, amount, term, employer, and tenure. Provide a preliminary assessment based on the information and estimated rate range. Mention processing timeline.",
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
    print("Loan Assessment -- Context Tracking with Handlers")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    # Register handlers
    fsm.create_handler(
        name="transition_logger",
        timing=HandlerTiming.POST_TRANSITION,
        action=lambda ctx: metrics["state_transitions"].append(
            ctx.get("_current_state", "?")
        ),
    )

    fsm.create_handler(
        name="extraction_monitor",
        timing=HandlerTiming.POST_PROCESSING,
        action=lambda ctx: metrics["extraction_counts"].append(
            {
                "state": ctx.get("_current_state", "?"),
                "keys": len([k for k in ctx if not k.startswith("_")]),
            }
        ),
    )

    fsm.create_handler(
        name="risk_monitor",
        timing=HandlerTiming.CONTEXT_UPDATE,
        action=lambda ctx: metrics["risk_signals"].append(
            {
                "credit": ctx.get("credit_score_range", "unknown"),
                "income": ctx.get("annual_income", "unknown"),
            }
        ),
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "I'm Michael Chen and I'd like to apply for an auto loan for a new vehicle",
        "My annual income is $95,000 and my credit score is around 740, so I'd say excellent",
        "I'm looking to borrow $35,000 over 5 years",
        "I work at Amazon Web Services as a solutions architect, been there 4 years",
    ]

    expected_keys = [
        "applicant_name",
        "loan_type",
        "annual_income",
        "credit_score_range",
        "loan_amount",
        "loan_term",
        "employer_name",
        "years_employed",
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
    print("LOAN ASSESSMENT SUMMARY")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    extracted = 0
    for key in expected_keys:
        value = data.get(key)
        status = "EXTRACTED" if value is not None else "MISSING"
        if value is not None:
            extracted += 1
        print(f"  {key:25s}: {str(value)[:35]:35s} [{status}]")

    print(
        f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)"
    )

    print("\n" + "=" * 60)
    print("HANDLER ANALYTICS")
    print("=" * 60)
    print(f"  State transitions: {metrics['state_transitions']}")
    print(f"  Extraction progression: {metrics['extraction_counts']}")
    print(f"  Risk signals captured: {len(metrics['risk_signals'])}")

    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
