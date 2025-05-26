"""
Form Filling Example for LLM-FSM

This script demonstrates using LLM-FSM to implement a structured form-filling
conversation to collect user information using the simplified API.
"""

import os
import json
from llm_fsm import API


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
        # This will trigger the initial welcome message
        conversation_id, response = fsm.converse("")
        print(f"System: {response}")

        # Main conversation loop
        while not fsm.has_conversation_ended(conversation_id):
            # Get user input
            user_input = input("You: ")

            # Check for manual exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting conversation.")
                break

            # Store the user message in context for logic conditions
            context = fsm.get_data(conversation_id)
            context["user_message"] = user_input.lower()

            # Process the user input
            try:
                _, response = fsm.converse(user_input, conversation_id)
                print(f"System: {response}")

                # Get the current state for debugging (optional)
                current_state = fsm.get_current_state(conversation_id)

                # Check if we've reached a terminal state
                if fsm.has_conversation_ended(conversation_id):
                    print("Conversation has ended.")

                    # Display the final collected data
                    data = fsm.get_data(conversation_id)
                    print("\nCollected Form Data:")

                    # Filter out system keys (those starting with _)
                    form_data = {k: v for k, v in data.items() if not k.startswith('_') and k != 'user_message'}

                    for key, value in form_data.items():
                        print(f"- {key.capitalize()}: {value}")

            except Exception as e:
                print(f"Error: {str(e)}")

        # Clean up
        fsm.end_conversation(conversation_id)

    except FileNotFoundError:
        print(f"Error: Could not find FSM definition at {fsm_path}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()