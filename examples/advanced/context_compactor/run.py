"""
Context Compactor Example -- Managing Context Growth
====================================================

Demonstrates how FSM context grows over a long conversation
and how ContextCompactor mechanisms (transient keys, pruning)
help manage memory. Also tests context_length JsonLogic operator.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/advanced/context_compactor/run.py
"""

import os

from fsm_llm import API


def build_fsm() -> dict:
    """FSM that collects multiple items then summarizes."""
    return {
        "name": "ItemCollector",
        "description": "Collects items from the user and summarizes them",
        "initial_state": "collecting",
        "persona": "A helpful list builder",
        "states": {
            "collecting": {
                "id": "collecting",
                "description": "Collect items from the user",
                "purpose": "Add items to a growing list",
                "extraction_instructions": "Extract any new item mentioned by the user as 'new_item'. Also count total items mentioned so far as 'item_count'.",
                "response_instructions": "Acknowledge the item and ask if they have more to add. List all items collected so far.",
                "transitions": [
                    {
                        "target_state": "summarize",
                        "description": "User says they are done",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "User indicates completion",
                                "requires_context_keys": ["user_done"],
                                "logic": {"==": [{"var": "user_done"}, true]},
                            }
                        ],
                    },
                    {
                        "target_state": "summarize",
                        "description": "Enough items collected (3+)",
                        "priority": 50,
                        "conditions": [
                            {
                                "description": "At least 3 items",
                                "requires_context_keys": ["item_count"],
                                "logic": {">=": [{"var": "item_count"}, 3]},
                            }
                        ],
                    },
                ],
            },
            "summarize": {
                "id": "summarize",
                "description": "Summarize collected items",
                "purpose": "Present a final summary of all items",
                "extraction_instructions": "None",
                "response_instructions": "Present a nice summary of all the items collected, organized in a list",
                "transitions": [],
            },
        },
    }


true = True


def main():
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    print("=" * 60)
    print("Context Compactor -- Managing Context Growth")
    print("=" * 60)
    print(f"Model: {model}\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.7
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    items = [
        "Add milk to my list",
        "Also eggs and bread",
        "Don't forget butter",
        "And some cheese",
        "That's everything, I'm done",
    ]

    for msg in items:
        print(f"\nYou: {msg}")

        # Show context size before
        data = fsm.get_data(conv_id)
        context_keys = [k for k in data if not k.startswith("_")]
        print(f"  [Context before: {len(context_keys)} keys, {len(str(data))} chars]")

        try:
            response = fsm.converse(msg, conv_id)
            print(f"Bot: {response}")
            print(f"  State: {fsm.get_current_state(conv_id)}")
        except Exception as e:
            print(f"Error: {e}")

        # Show context size after
        data = fsm.get_data(conv_id)
        context_keys = [k for k in data if not k.startswith("_")]
        state = fsm.get_current_state(conv_id)
        print(f"  [Context after: {len(context_keys)} keys, state={state}]")

        if fsm.has_conversation_ended(conv_id):
            break

    # Final context dump
    print("\n" + "=" * 60)
    print("FINAL CONTEXT")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    for key, value in sorted(data.items()):
        if not key.startswith("_"):
            val_str = str(value)[:80]
            print(f"  {key}: {val_str}")

    history = fsm.get_conversation_history(conv_id)
    print(f"\nConversation history: {len(history)} messages")

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    data = fsm.get_data(conv_id)
    expected_keys = ["new_item", "item_count", "user_done"]
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

    fsm.end_conversation(conv_id)


if __name__ == "__main__":
    main()
