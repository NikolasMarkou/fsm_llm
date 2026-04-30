from fsm_llm.dialog.api import API
"""
Multi-Intent Classification Example -- Multiple Intents Per Message
===================================================================

Tests classification when a single user message contains multiple
intents (e.g., "I want to buy a phone AND check my order status").
Uses classification_extractions on states.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/classification/multi_intent/run.py
"""

import os



def build_fsm() -> dict:
    """FSM with classification_extractions for multi-intent detection."""
    return {
        "name": "MultiIntentBot",
        "description": "Detects and handles multiple intents in one message",
        "initial_state": "listen",
        "persona": "A smart assistant that can detect when users ask multiple things at once",
        "states": {
            "listen": {
                "id": "listen",
                "description": "Listen for user intents",
                "purpose": "Classify user intent(s) and route appropriately",
                "extraction_instructions": "Identify the user's primary intent. Set intent to 'question' for questions, 'complaint' for complaints, 'purchase' for buying intent, 'greeting' for greetings",
                "response_instructions": "Acknowledge the user's request and address their primary need. If they mentioned multiple things, acknowledge all of them.",
                "classification_extractions": [
                    {
                        "field_name": "primary_intent",
                        "intents": [
                            {
                                "name": "question",
                                "description": "User is asking a question or seeking information",
                            },
                            {
                                "name": "complaint",
                                "description": "User is complaining about something or reporting a problem",
                            },
                            {
                                "name": "purchase",
                                "description": "User wants to buy something or make a transaction",
                            },
                            {
                                "name": "greeting",
                                "description": "User is greeting or starting a conversation",
                            },
                        ],
                        "fallback_intent": "question",
                        "confidence_threshold": 0.5,
                        "required": False,
                    }
                ],
                "transitions": [
                    {
                        "target_state": "handle_complaint",
                        "description": "Complaint detected",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "User is complaining",
                                "requires_context_keys": ["primary_intent"],
                                "logic": {
                                    "==": [{"var": "primary_intent"}, "complaint"]
                                },
                            }
                        ],
                    },
                    {
                        "target_state": "handle_purchase",
                        "description": "Purchase intent detected",
                        "priority": 90,
                        "conditions": [
                            {
                                "description": "User wants to buy",
                                "requires_context_keys": ["primary_intent"],
                                "logic": {
                                    "==": [{"var": "primary_intent"}, "purchase"]
                                },
                            }
                        ],
                    },
                    {
                        "target_state": "handle_question",
                        "description": "Question detected",
                        "priority": 50,
                        "conditions": [
                            {
                                "description": "User has a question",
                                "requires_context_keys": ["primary_intent"],
                                "logic": {
                                    "==": [{"var": "primary_intent"}, "question"]
                                },
                            }
                        ],
                    },
                    {
                        "target_state": "done",
                        "description": "User greets — end conversation",
                        "priority": 10,
                        "conditions": [
                            {
                                "description": "Greeting intent detected",
                                "requires_context_keys": ["primary_intent"],
                                "logic": {
                                    "==": [{"var": "primary_intent"}, "greeting"]
                                },
                            }
                        ],
                    },
                ],
            },
            "handle_complaint": {
                "id": "handle_complaint",
                "description": "Handle customer complaint",
                "purpose": "Address the complaint with empathy and solutions",
                "extraction_instructions": "Extract the specific complaint topic",
                "response_instructions": "Empathize with the customer and offer to resolve their issue. Ask for details if needed.",
                "transitions": [
                    {
                        "target_state": "listen",
                        "description": "Return to listening",
                        "conditions": [
                            {
                                "description": "Complaint addressed",
                                "requires_context_keys": ["complaint_acknowledged"],
                                "logic": {"has_context": "complaint_acknowledged"},
                            }
                        ],
                    }
                ],
            },
            "handle_purchase": {
                "id": "handle_purchase",
                "description": "Handle purchase intent",
                "purpose": "Guide the user through a purchase",
                "extraction_instructions": "Extract what the user wants to buy as 'desired_product'",
                "response_instructions": "Help the user with their purchase. Ask about preferences and budget.",
                "transitions": [
                    {
                        "target_state": "listen",
                        "description": "Return to listening",
                        "conditions": [
                            {
                                "description": "Purchase discussed",
                                "requires_context_keys": ["desired_product"],
                                "logic": {"has_context": "desired_product"},
                            }
                        ],
                    }
                ],
            },
            "handle_question": {
                "id": "handle_question",
                "description": "Answer user question",
                "purpose": "Provide helpful answers",
                "extraction_instructions": "Extract the question topic",
                "response_instructions": "Answer the user's question clearly and helpfully",
                "transitions": [
                    {
                        "target_state": "listen",
                        "description": "Return to listening",
                        "conditions": [
                            {
                                "description": "Question answered",
                                "requires_context_keys": ["question_topic"],
                                "logic": {"has_context": "question_topic"},
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "Conversation complete",
                "purpose": "End the conversation",
                "extraction_instructions": "No extraction needed",
                "response_instructions": "Thank the user and say goodbye",
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
    print("Multi-Intent Classification")
    print("=" * 60)
    print(f"Model: {model}")
    print("Intents: question, complaint, purchase, greeting\n")

    fsm = API.from_definition(
        definition=build_fsm(), model=model, api_key=api_key, temperature=0.5
    )

    conv_id, response = fsm.start_conversation()
    print(f"Bot: {response}")

    test_messages = [
        ("I want to buy a new phone", "purchase"),
        ("Your service was terrible last time!", "complaint"),
        ("What are your store hours?", "question"),
        (
            "Hi there, I need help with a return and also want to order something new",
            "complaint+purchase",
        ),
    ]

    results = []
    for msg, expected_intent in test_messages:
        print(f"\nYou: {msg}")
        print(f"  Expected intent: {expected_intent}")

        try:
            response = fsm.converse(msg, conv_id)
            state = fsm.get_current_state(conv_id)
            data = fsm.get_data(conv_id)
            detected = data.get("primary_intent", "none")

            print(f"  Detected intent: {detected}")
            print(f"  State: {state}")
            print(f"  Bot: {response}")

            # For multi-intent cases like "complaint+purchase", accept either
            expected_options = expected_intent.split("+")
            matched = detected in expected_options
            results.append((expected_intent, detected, matched))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((expected_intent, "error", False))

    fsm.end_conversation(conv_id)

    # Verification summary
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    correct = 0
    total = len(results)
    for i, (expected, actual, matched) in enumerate(results):
        status = "EXTRACTED" if matched else "MISSING"
        if matched:
            correct += 1
        print(f"  test_{i}_{expected:20s}: {actual!s:40s} [{status}]")
    print(f"\nExtraction rate: {correct}/{total} ({100 * correct / total:.0f}%)")


if __name__ == "__main__":
    main()
