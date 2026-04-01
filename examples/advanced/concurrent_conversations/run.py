"""
Concurrent Conversations Example -- Multiple Simultaneous Sessions
=================================================================

Demonstrates running multiple conversations on the same API
instance simultaneously, verifying context isolation between them.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/concurrent_conversations/run.py
"""

import os

from fsm_llm import API


def build_fsm() -> dict:
    """Simple profile collection FSM."""
    return {
        "name": "ProfileBot",
        "description": "Collects user profile info",
        "initial_state": "collect",
        "persona": "A friendly data collector",
        "states": {
            "collect": {
                "id": "collect",
                "description": "Collect user information",
                "purpose": "Gather name and role from the user",
                "extraction_instructions": "Extract user_name and user_role from the message",
                "response_instructions": "Greet the user and ask for their name and role if not yet provided. If both are provided, confirm them.",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "All info collected",
                        "conditions": [
                            {
                                "description": "Both name and role available",
                                "requires_context_keys": ["user_name", "user_role"],
                                "logic": {
                                    "and": [
                                        {"has_context": "user_name"},
                                        {"has_context": "user_role"},
                                    ]
                                },
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "Profile complete",
                "purpose": "Confirm the collected profile",
                "extraction_instructions": "None",
                "response_instructions": "Confirm the user's profile with their name and role, then say goodbye",
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
    print("Concurrent Conversations -- Context Isolation Test")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.7
    )

    # Start 3 conversations simultaneously
    users = [
        {"name": "Alice", "role": "engineer"},
        {"name": "Bob", "role": "designer"},
        {"name": "Carol", "role": "manager"},
    ]

    conversations = {}
    for user in users:
        conv_id, response = fsm.start_conversation()
        conversations[conv_id] = user
        print(f"[{user['name']}] Started conv {conv_id[:8]}...")
        print(f"  Bot: {response}")

    # List active conversations
    active = fsm.list_active_conversations()
    print(f"\nActive conversations: {len(active)}")

    # Each user introduces themselves (interleaved)
    print("\n--- Round 1: Introductions ---")
    for conv_id, user in conversations.items():
        msg = f"Hi, I'm {user['name']} and I work as a {user['role']}"
        print(f"\n[{user['name']}] {msg}")
        try:
            response = fsm.converse(msg, conv_id)
            print(f"  Bot: {response}")
            state = fsm.get_current_state(conv_id)
            print(f"  State: {state}")
        except Exception as e:
            print(f"  Error: {e}")

    # Verify context isolation
    print("\n--- Context Isolation Check ---")
    all_isolated = True
    for conv_id, user in conversations.items():
        data = fsm.get_data(conv_id)
        extracted_name = data.get("user_name", "?")
        extracted_role = data.get("user_role", "?")
        name_match = (
            user["name"].lower() in str(extracted_name).lower()
            if extracted_name != "?"
            else False
        )
        role_match = (
            user["role"].lower() in str(extracted_role).lower()
            if extracted_role != "?"
            else False
        )

        status = "OK" if (name_match or role_match) else "LEAK?"
        if not (name_match or role_match):
            all_isolated = False

        print(
            f"  [{user['name']}] name={extracted_name}, role={extracted_role} -- {status}"
        )

    # Per-conversation verification
    expected_keys = ["user_name", "user_role"]
    overall_extracted = 0
    overall_total = 0

    for conv_id, user in conversations.items():
        print(f"\n{'=' * 60}")
        print(f"VERIFICATION -- {user['name']}")
        print("=" * 60)
        data = fsm.get_data(conv_id)
        extracted = 0
        for key in expected_keys:
            value = data.get(key)
            status = "EXTRACTED" if value is not None else "MISSING"
            if value is not None:
                extracted += 1
            print(f"  {key:25s}: {str(value)[:40]:40s} [{status}]")
        print(
            f"\nExtraction rate: {extracted}/{len(expected_keys)} ({100 * extracted / len(expected_keys):.0f}%)"
        )
        print(f"Final state: {fsm.get_current_state(conv_id)}")
        overall_extracted += extracted
        overall_total += len(expected_keys)

    print(f"\n{'=' * 60}")
    print("OVERALL VERIFICATION")
    print("=" * 60)
    print(
        f"Total extraction rate: {overall_extracted}/{overall_total} ({100 * overall_extracted / overall_total:.0f}%)"
    )
    print(f"Context isolation: {'PASSED' if all_isolated else 'FAILED'}")

    # Clean up
    for conv_id in conversations:
        fsm.end_conversation(conv_id)

    remaining = fsm.list_active_conversations()
    print(f"\nAfter cleanup: {len(remaining)} active conversations")


if __name__ == "__main__":
    main()
