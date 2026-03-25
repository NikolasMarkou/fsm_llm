"""
Classified Transitions — Manual Mode Example
=============================================

Demonstrates ``transition_classification`` with a dict config (manual mode),
where the FSM author provides custom intent descriptions for each transition
target. This gives more control over how the Classifier interprets each path.

Compare with run.py which uses auto-mode (``transition_classification=true``).

Key Differences from Auto-Mode:
    - Custom intent descriptions (more detailed than transition descriptions)
    - Custom confidence threshold per state
    - Better classification accuracy for domain-specific routing

Usage:
    export OPENAI_API_KEY="your-key-here"
    python run_manual.py

    # Or with a local Ollama model:
    export LLM_MODEL="ollama_chat/qwen3.5:9b"
    python run_manual.py
"""

import os

from fsm_llm import API

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FSM_PATH = os.path.join(CURRENT_DIR, "fsm_manual.json")


def main():
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Error: OPENAI_API_KEY not set. Export it or use Ollama.")
        print("       export LLM_MODEL='ollama_chat/qwen3.5:9b'")
        return

    # Check if classification extension is available
    try:
        import fsm_llm_classification  # noqa: F401

        print("  [fsm_llm_classification installed — using classified transitions]")
    except ImportError:
        print(
            "  [fsm_llm_classification not installed — using LLM fallback for transitions]"
        )
        print("  Install with: pip install fsm-llm[classification]")

    # Load FSM — the 'greeting' state has transition_classification with a dict
    # config that provides custom descriptions and confidence_threshold=0.7
    fsm = API.from_file(path=FSM_PATH, model=model, api_key=api_key, temperature=0.7)

    print("\nProduct Advisor — Manual Classification Mode (type 'quit' to exit)")
    print("=" * 65)
    print("This FSM uses custom classification descriptions to route you to")
    print("the right product category: electronics, clothing, or home & garden.\n")

    conversation_id, response = fsm.start_conversation()
    print(f"Advisor: {response}")

    while not fsm.has_conversation_ended(conversation_id):
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        response = fsm.converse(user_input, conversation_id)

        state = fsm.get_current_state(conversation_id)
        print(f"  [state: {state}]")
        print(f"\nAdvisor: {response}")

    # Show collected context
    ctx = fsm.get_data(conversation_id)
    relevant = {k: v for k, v in ctx.items() if not k.startswith("_") and k != "system"}
    if relevant:
        print("\n--- Collected Preferences ---")
        for k, v in relevant.items():
            print(f"  {k}: {v}")

    fsm.end_conversation(conversation_id)
    print("\nConversation ended.")


if __name__ == "__main__":
    main()
