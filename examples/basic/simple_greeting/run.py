"""
Simple Greeting Example for LLM-FSM

This script demonstrates a minimal implementation of LLM-FSM for a basic
greeting and farewell conversation using the simplified API.
"""

import os
from llm_fsm import API

# --------------------------------------------------------------


def main():
    # Get API key from environment or set it directly
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        return

    # Load the FSM definition from the JSON file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fsm_path = os.path.join(current_dir, "fsm.json")

    try:
        # Create the LLM-FSM instance using the simplified API
        fsm = API.from_file(
            path=fsm_path,
            model="gpt-4o-mini",  # You can change to another model
            api_key=api_key,
            temperature=0.7  # Higher temperature for more variety in responses
        )

        # Start a new conversation with an empty message
        # This will trigger the initial greeting
        conversation_id, response = fsm.start_conversation()
        print(f"System: {response}")

        # Main conversation loop
        while not fsm.has_conversation_ended(conversation_id):
            # Get user input
            user_input = input("You: ")

            # Check for manual exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting conversation.")
                break

            # Process the user input
            try:
                response = fsm.converse(user_input, conversation_id)
                print(f"System: {response}")

            except Exception as e:
                print(f"Error: {str(e)}")

        # Clean up
        fsm.end_conversation(conversation_id)

    except FileNotFoundError:
        print(f"Error: Could not find FSM definition at {fsm_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

# --------------------------------------------------------------


if __name__ == "__main__":
    main()