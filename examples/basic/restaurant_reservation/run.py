from fsm_llm.dialog.api import API

"""
Restaurant Reservation -- Large Context Multi-Turn Extraction
=============================================================

Tests FSM extraction for a detailed restaurant reservation process
with rich contextual information about the venue, menu, and dining
preferences.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/basic/restaurant_reservation/run.py
"""

import os


def build_fsm() -> dict:
    return {
        "name": "RestaurantReservationBot",
        "description": "Manages restaurant reservation with detailed venue context",
        "initial_state": "guest_info",
        "persona": (
            "You are the reservation manager at La Maison Dorée, an upscale "
            "French-Japanese fusion restaurant in downtown San Francisco. "
            "Be warm and refined when assisting with reservations."
        ),
        "states": {
            "guest_info": {
                "id": "guest_info",
                "description": "Collect guest name and party size",
                "purpose": "Identify the guest and party",
                "extraction_instructions": "Extract 'guest_name' (name for reservation) and 'party_size' (number of guests).",
                "response_instructions": "Welcome the caller to La Maison Dorée. The restaurant has 60 seats across three dining areas: the Main Hall (30 seats, lively atmosphere), the Garden Room (20 seats, enclosed patio with herb garden views), and the Chef's Counter (10 seats, front-row kitchen experience with tasting menu only at $185 per person). Regular menu prices range from $45-$85 per entree. Open Tuesday-Saturday 5:30pm-10:30pm. Weekend reservations should be made at least one week in advance. Dietary restrictions are accommodated including gluten-free, vegan, and common allergies. Corkage fee is $35/bottle. Valet parking available for $20. Ask for their name and party size.",
                "transitions": [
                    {
                        "target_state": "date_time",
                        "description": "Guest info collected",
                        "conditions": [
                            {
                                "description": "Name and size provided",
                                "requires_context_keys": ["guest_name"],
                                "logic": {"has_context": "guest_name"},
                            }
                        ],
                    }
                ],
            },
            "date_time": {
                "id": "date_time",
                "description": "Collect date and time preference",
                "purpose": "Schedule the reservation",
                "extraction_instructions": "Extract 'reservation_date' (date) and 'preferred_time' (time).",
                "response_instructions": "Ask for their preferred date and time. Mention hours are Tuesday-Saturday, 5:30pm-10:30pm. Weekend bookings need one week advance notice.",
                "transitions": [
                    {
                        "target_state": "seating",
                        "description": "Date and time set",
                        "conditions": [
                            {
                                "description": "Date provided",
                                "requires_context_keys": ["reservation_date"],
                                "logic": {"has_context": "reservation_date"},
                            }
                        ],
                    }
                ],
            },
            "seating": {
                "id": "seating",
                "description": "Collect seating and dietary preferences",
                "purpose": "Finalize dining preferences",
                "extraction_instructions": "Extract 'seating_preference' (main hall, garden room, or chef's counter) and 'dietary_restrictions' (any allergies or restrictions).",
                "response_instructions": "Describe the three dining areas: Main Hall (lively), Garden Room (patio), Chef's Counter ($185 tasting menu). Ask for seating preference and any dietary restrictions.",
                "transitions": [
                    {
                        "target_state": "confirmation",
                        "description": "Preferences set",
                        "conditions": [
                            {
                                "description": "Seating chosen",
                                "requires_context_keys": ["seating_preference"],
                                "logic": {"has_context": "seating_preference"},
                            }
                        ],
                    }
                ],
            },
            "confirmation": {
                "id": "confirmation",
                "description": "Confirm reservation details",
                "purpose": "Summarize and confirm the reservation",
                "extraction_instructions": "None",
                "response_instructions": "Summarize the reservation: guest name, party size, date, time, seating area, and dietary notes. Confirm the booking at La Maison Dorée.",
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
    print("Restaurant Reservation -- Large Context Extraction")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "Hi, reservation for Thompson, party of 4 please",
        "We'd like next Saturday evening, around 7:30pm",
        "The Garden Room sounds lovely. One guest is gluten-free",
    ]

    expected_keys = [
        "guest_name",
        "party_size",
        "reservation_date",
        "preferred_time",
        "seating_preference",
        "dietary_restrictions",
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
    print("RESERVATION SUMMARY")
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
