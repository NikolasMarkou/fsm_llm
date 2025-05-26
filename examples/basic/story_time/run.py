"""
Simple CLI-based implementation of the Three Little Pigs Interactive Story
using the LLM-FSM framework.
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
            model="gpt-4o",  # Recommended for rich storytelling
            api_key=api_key,
            temperature=0.9  # Higher temperature for more creative storytelling
        )

        # Track user interactions for summary
        user_interactions = {}

        # Print welcome banner
        print("\n" + "=" * 60)
        print("📚 THREE LITTLE PIGS INTERACTIVE STORY 📚".center(60))
        print("=" * 60)

        # Start a new conversation with an empty message
        # This will trigger the initial introduction
        conversation_id, response = fsm.converse("")
        print(f"\nStoryteller: {response}\n")

        # Main conversation loop
        while not fsm.has_conversation_ended(conversation_id):
            # Get user input
            user_input = input("You: ")

            # Check for manual exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting story.")
                break

            # Process the user input
            try:
                _, response = fsm.converse(user_input, conversation_id)
                print(f"\nStoryteller: {response}\n")

                # Get the current state and context for tracking
                current_state = fsm.get_current_state(conversation_id)
                context = fsm.get_data(conversation_id)

                # Store user interactions for story summary
                for key in context:
                    if key.startswith("user_") and key not in user_interactions:
                        user_interactions[key] = context[key]

                # Check if we've reached the end
                if fsm.has_conversation_ended(conversation_id):
                    print("\n" + "=" * 60)
                    print("📖 STORY COMPLETE! 📖".center(60))
                    print("=" * 60)

                    # Print story summary
                    print("\nYour Journey Through the Story:")

                    if user_interactions:
                        for key, value in user_interactions.items():
                            # Format the key for display
                            display_key = key.replace('user_', '').replace('_', ' ').title()
                            print(f"• {display_key}: {value}")

                    print("\nThank you for experiencing this interactive tale!")

            except Exception as e:
                print(f"Error: {str(e)}")

        # Clean up
        fsm.end_conversation(conversation_id)

    except FileNotFoundError:
        print(f"Error: Could not find FSM definition at {fsm_path}")
        print("Make sure to create fsm.json with the Three Little Pigs story definition")
    except json.JSONDecodeError:
        print(f"Error: The FSM definition file contains invalid JSON")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()