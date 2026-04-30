from fsm_llm.dialog.api import API
"""
Travel Booking -- Large Context Multi-Turn Extraction
=====================================================

Tests FSM extraction for a detailed travel reservation process
with rich contextual information about destinations, preferences,
and travel logistics.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/basic/travel_booking/run.py
"""

import os



def build_fsm() -> dict:
    return {
        "name": "TravelBookingBot",
        "description": "Collects detailed travel booking information",
        "initial_state": "destination",
        "persona": (
            "You are a senior travel consultant at Wanderlust Premium Travel Agency, "
            "specializing in customized international vacation packages. "
            "Be enthusiastic and knowledgeable when helping plan trips."
        ),
        "states": {
            "destination": {
                "id": "destination",
                "description": "Collect travel destination preferences",
                "purpose": "Understand where the client wants to go",
                "extraction_instructions": "Extract 'destination' (city or country) and 'travel_dates' (when they want to travel).",
                "response_instructions": "Welcome the traveler to Wanderlust Premium Travel Agency. Your agency partners with over 200 luxury hotels across 45 countries and has exclusive airline deals with Emirates, Singapore Airlines, and Lufthansa for premium cabin upgrades. Three package tiers are available: Explorer ($2,000-$5,000), Premium ($5,000-$12,000), and Ultra ($12,000-$30,000) per person. All packages include travel insurance, airport transfers, and 24/7 concierge support. Popular destinations this season: Kyoto for cherry blossoms, Amalfi Coast for Mediterranean escapes, Patagonia for adventure, and the Maldives for beach retreats. Group bookings of 4+ get 15% off. Ask where and when they'd like to travel.",
                "transitions": [
                    {
                        "target_state": "preferences",
                        "description": "Destination selected",
                        "conditions": [
                            {
                                "description": "Destination provided",
                                "requires_context_keys": ["destination"],
                                "logic": {"has_context": "destination"},
                            }
                        ],
                    }
                ],
            },
            "preferences": {
                "id": "preferences",
                "description": "Collect travel preferences and budget",
                "purpose": "Understand accommodation and budget needs",
                "extraction_instructions": "Extract 'num_travelers' (number of people) and 'budget_tier' (explorer, premium, or ultra).",
                "response_instructions": "Ask about the number of travelers and preferred budget tier. Describe the three package tiers available.",
                "transitions": [
                    {
                        "target_state": "activities",
                        "description": "Preferences collected",
                        "conditions": [
                            {
                                "description": "Travelers and budget known",
                                "requires_context_keys": ["num_travelers"],
                                "logic": {"has_context": "num_travelers"},
                            }
                        ],
                    }
                ],
            },
            "activities": {
                "id": "activities",
                "description": "Collect activity and interest preferences",
                "purpose": "Plan activities based on interests",
                "extraction_instructions": "Extract 'interests' (what activities they enjoy) and 'special_requests' (any special needs).",
                "response_instructions": "Ask about preferred activities and any special requirements like dietary needs or accessibility.",
                "transitions": [
                    {
                        "target_state": "confirmation",
                        "description": "Activities selected",
                        "conditions": [
                            {
                                "description": "Interests provided",
                                "requires_context_keys": ["interests"],
                                "logic": {"has_context": "interests"},
                            }
                        ],
                    }
                ],
            },
            "confirmation": {
                "id": "confirmation",
                "description": "Confirm and summarize booking",
                "purpose": "Present the complete travel plan",
                "extraction_instructions": "None",
                "response_instructions": "Summarize the complete travel plan: destination, dates, number of travelers, budget tier, interests, and special requests. Provide an estimated price range.",
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
    print("Travel Booking -- Large Context Extraction")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "We'd love to visit Kyoto, Japan in mid-April for the cherry blossoms",
        "There will be 2 of us and we're thinking premium tier, around $8,000 each",
        "We enjoy temple visits, tea ceremonies, and local food tours. No special dietary needs",
    ]

    expected_keys = [
        "destination",
        "travel_dates",
        "num_travelers",
        "budget_tier",
        "interests",
        "special_requests",
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
    print("BOOKING SUMMARY")
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
