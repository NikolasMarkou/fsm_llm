"""
Medical Intake -- Large Context Multi-Turn Extraction
=====================================================

Tests FSM extraction for a patient intake process with detailed
medical context, symptoms collection, and history gathering.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/basic/medical_intake/run.py
"""

import os

from fsm_llm import API


def build_fsm() -> dict:
    return {
        "name": "MedicalIntakeBot",
        "description": "Collects patient intake information for a medical appointment",
        "initial_state": "patient_info",
        "persona": (
            "You are a medical intake coordinator at Riverside Family Health Center, "
            "a multi-specialty clinic collecting pre-visit information for new patient "
            "appointments. Be empathetic and professional when collecting patient information."
        ),
        "states": {
            "patient_info": {
                "id": "patient_info",
                "description": "Collect basic patient information",
                "purpose": "Gather name and date of birth",
                "extraction_instructions": "Extract 'patient_name' (full name) and 'date_of_birth' from the message.",
                "response_instructions": "Welcome the new patient to Riverside Family Health Center. The clinic has departments in family medicine, internal medicine, orthopedics, and dermatology, serving approximately 2,500 patients monthly with 15 physicians on staff. The clinic requires patient name, date of birth, and insurance information before scheduling. Accepted insurance plans include BlueCross, Aetna, UnitedHealth, Cigna, and Medicare/Medicaid. Self-pay patients receive a 20% discount. Average wait times are 15 minutes for scheduled appointments. Same-day urgent appointments are available Monday-Friday 8am-6pm and Saturday 9am-1pm. Ask for their full name and date of birth.",
                "transitions": [
                    {
                        "target_state": "symptoms",
                        "description": "Patient info collected",
                        "conditions": [
                            {
                                "description": "Name and DOB provided",
                                "requires_context_keys": ["patient_name"],
                                "logic": {"has_context": "patient_name"},
                            }
                        ],
                    }
                ],
            },
            "symptoms": {
                "id": "symptoms",
                "description": "Collect primary symptoms",
                "purpose": "Understand reason for visit",
                "extraction_instructions": "Extract 'primary_symptom' (main complaint) and 'symptom_duration' (how long).",
                "response_instructions": "Ask about their primary reason for visiting. What symptoms are they experiencing and for how long?",
                "transitions": [
                    {
                        "target_state": "insurance",
                        "description": "Symptoms collected",
                        "conditions": [
                            {
                                "description": "Symptoms described",
                                "requires_context_keys": ["primary_symptom"],
                                "logic": {"has_context": "primary_symptom"},
                            }
                        ],
                    }
                ],
            },
            "insurance": {
                "id": "insurance",
                "description": "Collect insurance information",
                "purpose": "Verify insurance coverage",
                "extraction_instructions": "Extract 'insurance_provider' (company name) and 'member_id' (insurance ID number).",
                "response_instructions": "Ask for their insurance provider and member ID. Mention accepted plans: BlueCross, Aetna, UnitedHealth, Cigna, Medicare/Medicaid. Self-pay gets 20% discount.",
                "transitions": [
                    {
                        "target_state": "confirmation",
                        "description": "Insurance info collected",
                        "conditions": [
                            {
                                "description": "Insurance provided",
                                "requires_context_keys": ["insurance_provider"],
                                "logic": {"has_context": "insurance_provider"},
                            }
                        ],
                    }
                ],
            },
            "confirmation": {
                "id": "confirmation",
                "description": "Confirm intake information",
                "purpose": "Summarize and confirm all collected data",
                "extraction_instructions": "None",
                "response_instructions": "Summarize all collected information: patient name, DOB, symptoms, duration, insurance. Confirm the appointment at Riverside Family Health Center.",
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
    print("Medical Intake -- Large Context Extraction")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "I'm Robert Martinez, born March 15, 1985",
        "I've been having persistent lower back pain for about 3 weeks now",
        "I have BlueCross insurance, member ID BC-7742918",
    ]

    expected_keys = [
        "patient_name",
        "date_of_birth",
        "primary_symptom",
        "symptom_duration",
        "insurance_provider",
        "member_id",
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
    print("INTAKE SUMMARY")
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
