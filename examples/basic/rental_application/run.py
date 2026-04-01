"""
Rental Application -- Large Context Multi-Turn Extraction
=========================================================

Tests FSM extraction for an apartment rental application with
detailed property context, screening requirements, and lease terms.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/basic/rental_application/run.py
"""

import os

from fsm_llm import API


def build_fsm() -> dict:
    return {
        "name": "RentalApplicationBot",
        "description": "Processes rental applications with detailed property context",
        "initial_state": "applicant_info",
        "persona": (
            "You are a leasing agent at Oakwood Heights Apartments, a 250-unit luxury "
            "apartment community in Charlotte, North Carolina. "
            "Be professional and helpful when collecting application details."
        ),
        "states": {
            "applicant_info": {
                "id": "applicant_info",
                "description": "Collect applicant personal information",
                "purpose": "Gather name and employment details",
                "extraction_instructions": "Extract 'applicant_name' (full name) and 'employer' (current employer name).",
                "response_instructions": "Welcome the prospective tenant to Oakwood Heights Apartments. The property features studio ($1,100/mo), one-bedroom ($1,450/mo), two-bedroom ($1,850/mo), and three-bedroom ($2,400/mo) floor plans. All units include stainless steel appliances, granite countertops, in-unit washer/dryer, and a private balcony. Amenities include a rooftop pool, 24-hour fitness center, co-working space, package lockers, and covered parking ($75/mo extra). Pets welcome with $300 deposit and $35/mo rent (limit 2, breed restrictions apply). Lease terms: 6, 12, or 18 months; 12-month leases get one month free. Move-in requires first month's rent plus security deposit. Credit minimum 620, income 3x rent. Applications processed within 48 hours. Ask for their full name and current employer.",
                "transitions": [
                    {
                        "target_state": "unit_preference",
                        "description": "Applicant info collected",
                        "conditions": [
                            {
                                "description": "Name provided",
                                "requires_context_keys": ["applicant_name"],
                                "logic": {"has_context": "applicant_name"},
                            }
                        ],
                    }
                ],
            },
            "unit_preference": {
                "id": "unit_preference",
                "description": "Collect unit and lease preferences",
                "purpose": "Determine desired floor plan and lease term",
                "extraction_instructions": "Extract 'unit_type' (studio, one-bedroom, two-bedroom, or three-bedroom) and 'lease_term' (6, 12, or 18 months).",
                "response_instructions": "Present available floor plans with pricing. Ask which unit type they prefer and desired lease length. Mention 12-month leases get one month free.",
                "transitions": [
                    {
                        "target_state": "move_in_details",
                        "description": "Preferences set",
                        "conditions": [
                            {
                                "description": "Unit type chosen",
                                "requires_context_keys": ["unit_type"],
                                "logic": {"has_context": "unit_type"},
                            }
                        ],
                    }
                ],
            },
            "move_in_details": {
                "id": "move_in_details",
                "description": "Collect move-in details",
                "purpose": "Determine timeline and special needs",
                "extraction_instructions": "Extract 'move_in_date' (desired date) and 'has_pets' (yes or no).",
                "response_instructions": "Ask about their desired move-in date and whether they have any pets. Mention pet policy: $300 deposit, $35/mo rent, limit 2.",
                "transitions": [
                    {
                        "target_state": "summary",
                        "description": "Details collected",
                        "conditions": [
                            {
                                "description": "Move-in date known",
                                "requires_context_keys": ["move_in_date"],
                                "logic": {"has_context": "move_in_date"},
                            }
                        ],
                    }
                ],
            },
            "summary": {
                "id": "summary",
                "description": "Summarize application",
                "purpose": "Present application summary and costs",
                "extraction_instructions": "None",
                "response_instructions": "Summarize: applicant name, employer, unit type, lease term, move-in date, and pet status. Calculate estimated move-in cost (first month + security deposit + any pet deposit). Mention 48-hour processing time.",
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
    print("Rental Application -- Large Context Extraction")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "Hi, I'm Marcus Johnson. I work at Bank of America",
        "I'm looking for a two-bedroom on a 12-month lease",
        "I'd like to move in around February 1st. I have one small dog",
    ]

    expected_keys = [
        "applicant_name",
        "employer",
        "unit_type",
        "lease_term",
        "move_in_date",
        "has_pets",
    ]

    for msg in messages:
        print(f"\nYou: {msg}")
        response = fsm.converse(msg, conv_id)
        print(f"Bot: {response}")
        state = fsm.get_current_state(conv_id)
        print(f"  State: {state}")

        if fsm.has_conversation_ended(conv_id):
            break

    print("\n" + "=" * 60)
    print("APPLICATION SUMMARY")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    extracted = 0
    for key in expected_keys:
        value = data.get(key)
        status = "EXTRACTED" if value is not None else "MISSING"
        if value is not None:
            extracted += 1
        print(f"  {key:25s}: {value!s:30s} [{status}]")

    print(
        f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)"
    )
    print(f"Final state: {fsm.get_current_state(conv_id)}")
    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
