"""
Classified Transitions Example — Classification-Aware FSM Routing
=================================================================

Demonstrates ``transition_classification``, a State-level feature that uses
fsm_llm_classification's Classifier to resolve ambiguous transitions instead
of a raw LLM prompt.

Key Concepts:
    - ``transition_classification=null`` in FSM JSON auto-generates a
      ClassificationSchema from the transition descriptions
    - When the TransitionEvaluator finds multiple valid transitions (AMBIGUOUS),
      the Classifier decides which path to take with structured confidence scoring
    - Graceful degradation: if fsm_llm_classification is not installed,
      falls back to the default LLM-based transition resolution

Contrast with smart_helpdesk:
    - smart_helpdesk: Classification happens BEFORE the FSM, selecting which FSM to load
    - This example: Classification happens INSIDE the FSM, at the transition level

Usage:
    export OPENAI_API_KEY="your-key-here"
    python run.py

    # Or with a local Ollama model:
    export LLM_MODEL="ollama_chat/qwen3.5:9b"
    python run.py
"""

import os

from fsm_llm import API

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FSM_PATH = os.path.join(CURRENT_DIR, "fsm.json")


def main():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Error: OPENAI_API_KEY not set. Export it or use Ollama.")
        print("       export LLM_MODEL='ollama_chat/qwen3.5:9b'")
        return

    # Load FSM — the 'welcome' state has transition_classification=null,
    # so ambiguous transitions from that state will use the Classifier
    fsm = API.from_file(path=FSM_PATH, model=model, api_key=api_key, temperature=0.7)

    print("\nClassified Transitions Demo (type 'quit' to exit)")
    print("=" * 55)
    print("This FSM uses classification to route your message to the")
    print("right support path: billing, technical, or account.\n")

    conversation_id, response = fsm.start_conversation()
    print(f"Support: {response}")

    while not fsm.has_conversation_ended(conversation_id):
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        try:
            response = fsm.converse(user_input, conversation_id)

            state = fsm.get_current_state(conversation_id)
            print(f"  [state: {state}]")
            print(f"\nSupport: {response}")
        except Exception as e:
            print(f"  Error: {e}")

    # Show collected context
    ctx = fsm.get_data(conversation_id)
    relevant = {k: v for k, v in ctx.items() if not k.startswith("_") and k != "system"}
    if relevant:
        print("\n--- Collected Information ---")
        for k, v in relevant.items():
            print(f"  {k}: {v}")

    fsm.end_conversation(conversation_id)
    print("\nConversation ended.")


if __name__ == "__main__":
    main()
