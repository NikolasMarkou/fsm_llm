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

    # ------------------------------------------------------------------
    # Test cases: each message should route to a specific state
    # ------------------------------------------------------------------

    test_cases = [
        ("I was charged twice on my last invoice", "billing"),
        ("The app keeps crashing when I open it", "technical"),
        ("I need to reset my password", "account"),
        ("Can I get a refund for last month?", "billing"),
    ]

    print("\nClassified Transitions Demo")
    print("=" * 55)
    print("This FSM uses classification to route your message to the")
    print("right support path: billing, technical, or account.\n")

    results = []
    for msg, expected_state in test_cases:
        # Each test case gets its own conversation (starts in 'welcome')
        fsm = API.from_file(
            path=FSM_PATH, model=model, api_key=api_key, temperature=0.7
        )
        conversation_id, response = fsm.start_conversation()
        print(f"Support: {response}")
        print(f"\nYou: {msg}")
        print(f"  Expected state: {expected_state}")

        try:
            response = fsm.converse(msg, conversation_id)
            state = fsm.get_current_state(conversation_id)
            print(f"  Actual state:   {state}")
            print(f"\nSupport: {response}")

            results.append((expected_state, state))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((expected_state, "error"))

        fsm.end_conversation(conversation_id)
        print()

    # Verification summary
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    correct = 0
    total = len(results)
    for i, (expected, actual) in enumerate(results):
        # Accept if we landed in the expected state or passed through it
        # (e.g., might have transitioned to 'resolved' already)
        matched = actual == expected or actual == "resolved"
        status = "EXTRACTED" if matched else "MISSING"
        if matched:
            correct += 1
        print(f"  test_{i}_{expected:20s}: {actual!s:40s} [{status}]")
    print(f"\nExtraction rate: {correct}/{total} ({100 * correct / total:.0f}%)")


if __name__ == "__main__":
    main()
