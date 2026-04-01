"""
Event Registration -- Large Context Multi-Turn Extraction
=========================================================

Tests FSM extraction for a conference registration process with
detailed event context, session preferences, and logistics.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/basic/event_registration/run.py
"""

import os

from fsm_llm import API


def build_fsm() -> dict:
    return {
        "name": "EventRegistrationBot",
        "description": "Handles conference registration with detailed event context",
        "initial_state": "attendee_info",
        "persona": (
            "You are the registration coordinator for TechSummit 2026, the premier "
            "annual technology conference at the Moscone Center in San Francisco, "
            "June 15-18, 2026. Be enthusiastic when registering attendees."
        ),
        "states": {
            "attendee_info": {
                "id": "attendee_info",
                "description": "Collect attendee personal information",
                "purpose": "Gather name and company details",
                "extraction_instructions": "Extract 'attendee_name' (full name) and 'company' (organization name).",
                "response_instructions": "Welcome the attendee to TechSummit 2026 registration. The conference features 4 tracks: AI and Machine Learning, Cloud Native Infrastructure, Cybersecurity, and Developer Experience. There are 120+ sessions, 50 workshops, and 3 keynote speakers from major tech companies. Registration tiers: Community ($299, sessions only), Professional ($799, sessions + workshops + lunch), and VIP ($1,499, all access + speaker dinner + priority seating). Early bird pricing ends May 1st with 25% discount. Group rates of 10+ get an additional 15% off. The venue is accessible with live captioning for keynotes. Hotel partner rates start at $189/night at the Marriott Marquis connected via skybridge. Ask for their full name and company/organization.",
                "transitions": [
                    {
                        "target_state": "ticket_selection",
                        "description": "Attendee info collected",
                        "conditions": [
                            {
                                "description": "Name provided",
                                "requires_context_keys": ["attendee_name"],
                                "logic": {"has_context": "attendee_name"},
                            }
                        ],
                    }
                ],
            },
            "ticket_selection": {
                "id": "ticket_selection",
                "description": "Select registration tier",
                "purpose": "Choose ticket type and track preferences",
                "extraction_instructions": "Extract 'ticket_tier' (community, professional, or vip) and 'preferred_track' (AI, cloud, security, or devex).",
                "response_instructions": "Present the three registration tiers with pricing. Ask which tier they prefer and which track interests them most.",
                "transitions": [
                    {
                        "target_state": "logistics",
                        "description": "Ticket selected",
                        "conditions": [
                            {
                                "description": "Tier chosen",
                                "requires_context_keys": ["ticket_tier"],
                                "logic": {"has_context": "ticket_tier"},
                            }
                        ],
                    }
                ],
            },
            "logistics": {
                "id": "logistics",
                "description": "Collect logistics preferences",
                "purpose": "Handle travel and accommodation needs",
                "extraction_instructions": "Extract 'needs_hotel' (yes or no) and 'dietary_preference' (any dietary needs for meals).",
                "response_instructions": "Ask about hotel needs (partner rate $189/night at Marriott Marquis) and any dietary preferences for conference meals.",
                "transitions": [
                    {
                        "target_state": "confirmation",
                        "description": "Logistics set",
                        "conditions": [
                            {
                                "description": "Hotel preference stated",
                                "requires_context_keys": ["needs_hotel"],
                                "logic": {"has_context": "needs_hotel"},
                            }
                        ],
                    }
                ],
            },
            "confirmation": {
                "id": "confirmation",
                "description": "Confirm registration",
                "purpose": "Summarize and confirm all details",
                "extraction_instructions": "None",
                "response_instructions": "Summarize the registration: attendee name, company, ticket tier, preferred track, hotel needs, and dietary preferences. Confirm the TechSummit 2026 registration.",
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
    print("Event Registration -- Large Context Extraction")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    messages = [
        "Hi, I'm Lisa Park from Quantum Computing Labs",
        "I'd like the Professional tier please. I'm most interested in the AI and Machine Learning track",
        "Yes, I'll need a hotel room. No dietary restrictions",
    ]

    expected_keys = [
        "attendee_name",
        "company",
        "ticket_tier",
        "preferred_track",
        "needs_hotel",
        "dietary_preference",
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
    print("REGISTRATION SUMMARY")
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
