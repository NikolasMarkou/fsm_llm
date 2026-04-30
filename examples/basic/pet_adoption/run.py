from fsm_llm.dialog.api import API
"""
Pet Adoption Application -- Large Context Multi-Turn Extraction
===============================================================

Tests FSM extraction for a pet adoption application process with
detailed shelter context, animal information, and screening.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/basic/pet_adoption/run.py
"""

import os



def build_fsm() -> dict:
    return {
        "name": "PetAdoptionBot",
        "description": "Processes pet adoption applications with detailed shelter context",
        "initial_state": "applicant_info",
        "persona": (
            "You are an adoption counselor at Paws & Hearts Animal Shelter, a non-profit "
            "no-kill shelter in Portland, Oregon. "
            "Be warm and encouraging while collecting application information."
        ),
        "states": {
            "applicant_info": {
                "id": "applicant_info",
                "description": "Collect applicant information",
                "purpose": "Gather name and housing situation",
                "extraction_instructions": "Extract 'applicant_name' (full name) and 'housing_type' (house, apartment, condo, etc.).",
                "response_instructions": "Welcome the potential adopter to Paws & Hearts Animal Shelter, which has been operating for 15 years. The shelter currently houses 85 dogs and 120 cats, plus small animals like rabbits and guinea pigs. Adoption fees are $150 for dogs, $100 for cats, and $50 for small animals, including spay/neuter, microchipping, first vaccinations, and a vet health check. Adopters must be at least 21 and provide proof of pet-friendly housing. For dogs, a home visit may be scheduled within 7 days. There is a 2-week trial period. Foster-to-adopt is available for first-time owners. Open Tuesday-Sunday 10am-5pm. Ask for their name and what type of housing they have.",
                "transitions": [
                    {
                        "target_state": "pet_preference",
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
            "pet_preference": {
                "id": "pet_preference",
                "description": "Collect pet preferences",
                "purpose": "Understand what type of pet they want",
                "extraction_instructions": "Extract 'pet_type' (dog, cat, or small animal) and 'size_preference' (small, medium, or large).",
                "response_instructions": "Ask what type of pet they're interested in and any size preferences. Mention current availability: 85 dogs, 120 cats, plus small animals.",
                "transitions": [
                    {
                        "target_state": "experience",
                        "description": "Preferences collected",
                        "conditions": [
                            {
                                "description": "Pet type known",
                                "requires_context_keys": ["pet_type"],
                                "logic": {"has_context": "pet_type"},
                            }
                        ],
                    }
                ],
            },
            "experience": {
                "id": "experience",
                "description": "Assess pet experience",
                "purpose": "Evaluate adopter's experience level",
                "extraction_instructions": "Extract 'pet_experience' (first-time owner or experienced) and 'other_pets' (any current pets at home).",
                "response_instructions": "Ask about their experience with pets and whether they currently have other animals at home. Mention foster-to-adopt for first-time owners.",
                "transitions": [
                    {
                        "target_state": "summary",
                        "description": "Experience assessed",
                        "conditions": [
                            {
                                "description": "Experience known",
                                "requires_context_keys": ["pet_experience"],
                                "logic": {"has_context": "pet_experience"},
                            }
                        ],
                    }
                ],
            },
            "summary": {
                "id": "summary",
                "description": "Summarize application",
                "purpose": "Present application summary and next steps",
                "extraction_instructions": "None",
                "response_instructions": "Summarize the application: applicant name, housing, pet type preference, size, experience level, and other pets. Explain next steps including the adoption fee and any home visit requirements.",
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
    print("Pet Adoption Application -- Large Context Extraction")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "I'm Amanda Torres. I live in a house with a large fenced yard",
        "I'd love to adopt a medium-sized dog",
        "I grew up with dogs and currently have one older cat at home",
    ]

    expected_keys = [
        "applicant_name",
        "housing_type",
        "pet_type",
        "size_preference",
        "pet_experience",
        "other_pets",
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
