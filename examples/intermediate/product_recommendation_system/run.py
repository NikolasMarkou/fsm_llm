"""
Product Recommendation System using LLM-FSM

This script implements a tech product recommendation system that helps users
find the right smartphone or laptop based on their preferences and budget.
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
            model="gpt-4o",  # You can change to another model
            api_key=api_key,
            temperature=0.7  # Balanced temperature for recommendations
        )

        # Store user preferences for summary
        user_preferences = {}

        print("\n🛒 TECH PRODUCT RECOMMENDATION SYSTEM 🛒")
        print("=" * 60)

        # Start a new conversation with an empty message
        # This will trigger the initial welcome message
        conversation_id, response = fsm.converse("")
        print(f"Advisor: {response}")

        # Main conversation loop
        while not fsm.has_conversation_ended(conversation_id):
            # Get user input
            user_input = input("\nYou: ")

            # Check for manual exit command
            if user_input.lower() in ["exit", "quit", "stop"]:
                print("Exiting conversation.")
                break

            # Process the user input
            try:
                _, response = fsm.converse(user_input, conversation_id)
                print(f"\nAdvisor: {response}")

                # Track user preferences for summary
                context = fsm.get_data(conversation_id)
                current_state = fsm.get_current_state(conversation_id)

                # Save key preferences
                if "device_type" in context and "device_type" not in user_preferences:
                    user_preferences["device_type"] = context["device_type"]
                if "budget" in context and "budget" not in user_preferences:
                    user_preferences["budget"] = context["budget"]

                # Track current state for the recommendation
                if current_state.startswith("recommend_") and "recommendation_state" not in user_preferences:
                    user_preferences["recommendation_state"] = current_state

                # Check if we've reached the end state
                if fsm.has_conversation_ended(conversation_id):
                    print("\n" + "=" * 60)
                    print("🏁 RECOMMENDATION COMPLETE 🏁")

                    # Display summary of preferences and recommendation
                    print("\nYour Technology Preferences:")

                    if "device_type" in user_preferences:
                        device = user_preferences["device_type"].capitalize()
                        print(f"• Device Type: {device}")

                    if "budget" in user_preferences:
                        budget_tier = user_preferences["budget"].capitalize()
                        print(f"• Budget Range: {budget_tier}")

                    print("\nThank you for using our Product Recommendation System!")

            except Exception as e:
                print(f"Error: {str(e)}")

        # Clean up
        fsm.end_conversation(conversation_id)

    except FileNotFoundError:
        print(f"Error: Could not find FSM definition at {fsm_path}")
        print("Make sure to create fsm.json with the Product Recommendation System definition")
    except json.JSONDecodeError:
        print(f"Error: The FSM definition file contains invalid JSON")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()