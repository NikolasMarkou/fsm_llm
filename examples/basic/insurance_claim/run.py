"""
Insurance Claim Filing -- Large Context Multi-Turn Extraction
=============================================================

Tests FSM extraction for a detailed insurance claim process with
rich contextual information about policy types, coverage, and
claims procedures.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/basic/insurance_claim/run.py
"""

import os

from fsm_llm import API


def build_fsm() -> dict:
    return {
        "name": "InsuranceClaimBot",
        "description": "Processes insurance claims with detailed policy context",
        "initial_state": "policyholder",
        "persona": (
            "You are a claims specialist at Guardian Shield Insurance Company, "
            "handling claims for homeowner's, auto, and renter's insurance policies. "
            "Be compassionate and efficient when processing claims."
        ),
        "states": {
            "policyholder": {
                "id": "policyholder",
                "description": "Verify policyholder identity",
                "purpose": "Collect name and policy number",
                "extraction_instructions": "Extract 'policyholder_name' (full name) and 'policy_number' (policy ID).",
                "response_instructions": "Welcome the caller to Guardian Shield Insurance claims. Guardian Shield is one of the top 20 property and casualty insurers in the US, serving over 3 million policyholders across all 50 states. Standard homeowner deductibles range from $500 to $2,500 and auto policy deductibles from $250 to $1,000. Claims under $5,000 are processed within 5-7 business days. Larger claims require an adjuster visit within 48 hours. Emergency claims for burst pipes or fire damage receive same-day acknowledgment. The company has a 96% claim satisfaction rating. Ask for their full name and policy number to begin the claim.",
                "transitions": [
                    {
                        "target_state": "incident",
                        "description": "Policyholder verified",
                        "conditions": [
                            {
                                "description": "Name and policy provided",
                                "requires_context_keys": ["policyholder_name"],
                                "logic": {"has_context": "policyholder_name"},
                            }
                        ],
                    }
                ],
            },
            "incident": {
                "id": "incident",
                "description": "Collect incident details",
                "purpose": "Document what happened",
                "extraction_instructions": "Extract 'incident_type' (fire, water damage, theft, accident, etc.) and 'incident_date' (when it happened).",
                "response_instructions": "Ask about the incident: what happened, when it occurred, and the type of damage or loss.",
                "transitions": [
                    {
                        "target_state": "damage",
                        "description": "Incident documented",
                        "conditions": [
                            {
                                "description": "Incident described",
                                "requires_context_keys": ["incident_type"],
                                "logic": {"has_context": "incident_type"},
                            }
                        ],
                    }
                ],
            },
            "damage": {
                "id": "damage",
                "description": "Assess damage and estimate",
                "purpose": "Document damage extent and estimated cost",
                "extraction_instructions": "Extract 'damage_description' (what was damaged) and 'estimated_cost' (dollar amount).",
                "response_instructions": "Ask for details about what was damaged and their estimate of the cost. Mention that claims under $5,000 process in 5-7 days, larger claims get an adjuster visit within 48 hours.",
                "transitions": [
                    {
                        "target_state": "summary",
                        "description": "Damage assessed",
                        "conditions": [
                            {
                                "description": "Damage documented",
                                "requires_context_keys": ["damage_description"],
                                "logic": {"has_context": "damage_description"},
                            }
                        ],
                    }
                ],
            },
            "summary": {
                "id": "summary",
                "description": "Summarize and file the claim",
                "purpose": "Present claim summary and next steps",
                "extraction_instructions": "None",
                "response_instructions": "Summarize the claim: policyholder, policy number, incident type, date, damage description, and estimated cost. Provide the expected timeline and next steps.",
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
    print("Insurance Claim Filing -- Large Context Extraction")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "I'm David Wilson, policy number GS-2024-88431",
        "We had water damage from a burst pipe last Tuesday, January 14th",
        "The kitchen flooring and lower cabinets are damaged. I estimate about $8,500 in repairs",
    ]

    expected_keys = ["policyholder_name", "policy_number", "incident_type", "incident_date", "damage_description", "estimated_cost"]

    for msg in messages:
        print(f"\nYou: {msg}")
        response = fsm.converse(msg, conv_id)
        print(f"Bot: {response}")
        state = fsm.get_current_state(conv_id)
        print(f"  State: {state}")

        if fsm.has_conversation_ended(conv_id):
            break

    print("\n" + "=" * 60)
    print("CLAIM SUMMARY")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    extracted = 0
    for key in expected_keys:
        value = data.get(key)
        status = "EXTRACTED" if value is not None else "MISSING"
        if value is not None:
            extracted += 1
        print(f"  {key:25s}: {str(value):30s} [{status}]")

    print(f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)")
    print(f"Final state: {fsm.get_current_state(conv_id)}")
    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
