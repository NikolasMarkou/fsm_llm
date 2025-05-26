"""
Adaptive Yoga Instruction Example for LLM-FSM

This script demonstrates using LLM-FSM to create a yoga instruction flow
that adapts to the user's engagement level.
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
            model="gpt-4-0125-preview",  # You can change to another model
            api_key=api_key,
            temperature=0.7  # Higher temperature for more variety in responses
        )

        # Initialize conversation with engagement tracking
        initial_context = {
            "engagement_level": "medium",  # Start with neutral assumption
            "completed_poses": [],  # Track which poses have been completed
            "session_feedback": []  # Store feedback from each step
        }

        # Start a new conversation with the initial context
        conversation_id, response = fsm.converse("", initial_context=initial_context)
        print(f"System: {response}")

        # Main conversation loop
        while not fsm.has_conversation_ended(conversation_id):
            # Get user input
            user_input = input("You: ")

            # Check for manual exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting conversation.")
                break

            # Get current context
            context = fsm.get_data(conversation_id)

            # Store the user message in context for sentiment analysis in transitions
            context["user_message"] = user_input.lower()

            # Basic sentiment analysis for engagement level
            engagement_words = {
                'high': ['yes', 'sure', 'great', 'love', 'enjoy', 'amazing', 'good', 'like', 'more', 'next'],
                'low': ['no', 'not', 'boring', 'difficult', 'hard', 'tired', 'stop', 'enough', 'bored', 'confused',
                        'meh']
            }

            # Simple word-based sentiment detection
            high_matches = sum(1 for word in engagement_words['high'] if word in user_input.lower())
            low_matches = sum(1 for word in engagement_words['low'] if word in user_input.lower())

            # Set engagement level based on response
            # Also consider response length as an engagement signal
            if len(user_input) < 5:  # Very short response
                context["engagement_level"] = "low"
            elif high_matches > low_matches:
                context["engagement_level"] = "high"
            elif low_matches > high_matches:
                context["engagement_level"] = "low"
            # If neutral or ambiguous, keep previous engagement level

            # Store this interaction for analysis
            context["session_feedback"].append({
                "user_input": user_input,
                "engagement_level": context["engagement_level"],
                "current_state": fsm.get_current_state(conversation_id)
            })

            # Process the user input
            try:
                _, response = fsm.converse(user_input, conversation_id)
                print(f"System: {response}")

                # Get the current state for debugging (optional)
                current_state = fsm.get_current_state(conversation_id)

                # Print debug info if you want to see the state transitions
                # print(f"[DEBUG] Current state: {current_state}, Engagement: {context['engagement_level']}")

                # Check if we've reached a terminal state
                if fsm.has_conversation_ended(conversation_id):
                    print("\nYoga session has concluded.")

                    # Analyze the session (for demonstration purposes)
                    print("\nSession Analysis:")
                    if "completed_poses" in context:
                        poses_completed = len(context["completed_poses"])
                        print(f"- Poses completed: {poses_completed}")
                        print(f"- Final engagement level: {context.get('engagement_level', 'unknown')}")

                        if poses_completed >= 3:
                            print("- Session completed successfully!")
                        else:
                            print("- Session ended early, possibly due to low engagement.")

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