"""
Smart Helpdesk Example — Classification + Core FSM
====================================================

Demonstrates using fsm_llm_classification to classify the user's initial
message, then routing them into the appropriate FSM conversation flow.

Combines:
    - fsm_llm_classification: Classifier + IntentRouter for initial triage
    - fsm_llm: Core FSM for guided conversation flows

Key Concepts:
    - Intent classification to select the right FSM
    - Dynamic FSM loading based on classification result
    - Seamless handoff from classifier to conversation flow
    - Multiple FSM definitions for different support domains

Usage:
    export OPENAI_API_KEY="your-key-here"
    python run.py

    # Or with a local Ollama model:
    export LLM_MODEL="ollama_chat/qwen3.5:9b"
    python run.py
"""

import os

from fsm_llm import API
from fsm_llm_classification import (
    ClassificationPromptConfig,
    ClassificationSchema,
    Classifier,
    IntentDefinition,
)

# ------------------------------------------------------------------
# Classification schema for initial triage
# ------------------------------------------------------------------

schema = ClassificationSchema(
    intents=[
        IntentDefinition(
            name="technical_support",
            description="User has a technical issue: device problems, software bugs, "
            "connectivity issues, performance problems, error messages",
        ),
        IntentDefinition(
            name="account_management",
            description="User needs help with their account: password reset, profile "
            "update, billing changes, account deletion, login issues",
        ),
        IntentDefinition(
            name="general_inquiry",
            description="General questions about products, services, pricing, or "
            "anything that doesn't fit technical support or account management",
        ),
    ],
    fallback_intent="general_inquiry",
    confidence_threshold=0.5,
)


# ------------------------------------------------------------------
# FSM paths for each intent
# ------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FSM_MAP = {
    "technical_support": os.path.join(CURRENT_DIR, "fsm_troubleshooting.json"),
    "account_management": os.path.join(CURRENT_DIR, "fsm_account.json"),
}


def handle_general_inquiry(message: str) -> str:
    """Handle general inquiries without an FSM — simple response."""
    return (
        "Thanks for your question! For general inquiries, here are some quick answers:\n"
        "- Pricing: Visit our website at example.com/pricing\n"
        "- Hours: We're available Mon-Fri, 9am-6pm\n"
        "- Returns: 30-day return policy on all items\n\n"
        "Is there anything specific you'd like to know?"
    )


def run_fsm_conversation(
    fsm_path: str,
    model: str,
    api_key: str | None,
    first_message: str,
    intent: str,
    entities: dict,
):
    """Run a guided FSM conversation after classification."""
    fsm = API.from_file(path=fsm_path, model=model, api_key=api_key, temperature=0.7)

    # Pass classification results as initial context
    initial_context = {
        "classified_intent": intent,
        "initial_message": first_message,
    }
    # Pass any extracted entities
    initial_context.update(entities or {})

    conversation_id, response = fsm.start_conversation(initial_context=initial_context)
    print(f"\nSpecialist: {response}")

    # Send the original message as the first turn (the FSM already has context)
    response = fsm.converse(first_message, conversation_id)
    print(f"\nSpecialist: {response}")

    # Continue the conversation
    while not fsm.has_conversation_ended(conversation_id):
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("exit", "quit"):
            break

        response = fsm.converse(user_input, conversation_id)

        state = fsm.get_current_state(conversation_id)
        print(f"  [{state}]")
        print(f"\nSpecialist: {response}")

    # Show collected context
    ctx = fsm.get_data(conversation_id)
    relevant = {
        k: v
        for k, v in ctx.items()
        if not k.startswith("_") and k not in ("initial_message",)
    }
    if relevant:
        print("\n--- Collected Information ---")
        for k, v in relevant.items():
            print(f"  {k}: {v}")

    fsm.end_conversation(conversation_id)


def main():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Error: OPENAI_API_KEY not set. Export it or use Ollama.")
        print("       export LLM_MODEL='ollama_chat/qwen3.5:9b'")
        return

    # Build classifier
    classifier = Classifier(
        schema,
        model=model,
        config=ClassificationPromptConfig(temperature=0.0),
    )

    print("Smart Helpdesk (type 'quit' to exit)")
    print("=" * 50)
    print("Describe your issue and we'll route you to the right specialist.\n")

    user_input = input("You: ").strip()
    if not user_input or user_input.lower() in ("quit", "exit"):
        return

    # Classify the user's message
    try:
        result = classifier.classify(user_input)
    except Exception as e:
        print(f"  Classification error: {e}")
        print(f"\nBot: {handle_general_inquiry(user_input)}")
        return

    print(f"\n  Detected intent:  {result.intent}")
    print(f"  Confidence:       {result.confidence:.2f}")
    if result.entities:
        print(f"  Entities:         {result.entities}")

    # Route to the appropriate handler
    if result.intent in FSM_MAP:
        print(f"\n  Routing to {result.intent.replace('_', ' ')} specialist...")
        print("-" * 50)
        try:
            run_fsm_conversation(
                fsm_path=FSM_MAP[result.intent],
                model=model,
                api_key=api_key,
                first_message=user_input,
                intent=result.intent,
                entities=result.entities or {},
            )
        except Exception as e:
            print(f"  Error in specialist conversation: {e}")
    else:
        # General inquiry — no FSM needed
        print(f"\nBot: {handle_general_inquiry(user_input)}")


if __name__ == "__main__":
    main()
