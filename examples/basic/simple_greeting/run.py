from fsm_llm.dialog.api import API

"""
Simple Greeting Example for FSM-LLM

This script demonstrates a minimal implementation of FSM-LLM for a basic
greeting and farewell conversation using the simplified API.
"""

import os

# --------------------------------------------------------------


def main():
    # Get model and API key from environment
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and "ollama" not in model.lower():
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        print("Or use Ollama: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Load the FSM definition from the JSON file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fsm_path = os.path.join(current_dir, "fsm.json")

    try:
        # Create the FSM-LLM instance using the simplified API
        fsm = API.from_file(
            path=fsm_path,
            model=model,
            api_key=api_key,
            temperature=0.7,  # Higher temperature for more variety in responses
        )

        # Start a new conversation with an empty message
        # This will trigger the initial greeting
        conversation_id, response = fsm.start_conversation()
        print(f"System: {response}")

        # Main conversation loop
        while not fsm.has_conversation_ended(conversation_id):
            # Get user input
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Check for manual exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting conversation.")
                break

            # Process the user input
            try:
                response = fsm.converse(user_input, conversation_id)
                print(f"System: {response}")
                print(f"  State: {fsm.get_current_state(conversation_id)}")

            except Exception as e:
                print(f"Error: {e!s}")

        # Verification summary
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        data = fsm.get_data(conversation_id)
        expected_keys = ["mood", "intent"]
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
        print(f"Final state: {fsm.get_current_state(conversation_id)}")

        # Clean up
        fsm.end_conversation(conversation_id)

    except FileNotFoundError:
        print(f"Error: Could not find FSM definition at {fsm_path}")
    except Exception as e:
        print(f"Error: {e!s}")


# --------------------------------------------------------------


if __name__ == "__main__":
    main()
