"""
Adaptive Yoga Instruction Example for FSM-LLM

This script demonstrates using FSM-LLM to create a yoga instruction flow
that adapts to the user's engagement level.
"""

import os
from fsm_llm import API


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
            temperature=0.7  # Higher temperature for more variety in responses
        )

        # Initialize conversation with engagement tracking
        initial_context = {
            "engagement_level": "medium",  # Start with neutral assumption
            "completed_poses": [],  # Track which poses have been completed
            "session_feedback": []  # Store feedback from each step
        }

        # Start a new conversation with the initial context
        conversation_id, response = fsm.start_conversation(initial_context=initial_context)
        print(f"System: {response}")

        # Main conversation loop
        while not fsm.has_conversation_ended(conversation_id):
            # Get user input
            user_input = input("You: ")

            # Check for manual exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting conversation.")
                break

            # Store the user message in context for JsonLogic transition conditions
            context = fsm.get_data(conversation_id)
            context["user_message"] = user_input.lower()

            # Track session interactions
            session_feedback = context.get("session_feedback", [])
            session_feedback.append({
                "user_input": user_input,
                "engagement_level": context.get("engagement_level", "medium"),
                "current_state": fsm.get_current_state(conversation_id),
            })
            context["session_feedback"] = session_feedback

            # Process the user input
            try:
                response = fsm.converse(user_input, conversation_id)
                print(f"System: {response}")

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
