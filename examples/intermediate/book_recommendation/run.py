"""
Enhanced Book Recommendation System using LLM-FSM

This script extends the basic book recommendation system with additional features:
1. Keeps track of all recommended books
2. Allows customizing the user's preferences
3. Provides a summary of recommendations at the end
4. Includes basic error handling and logging
"""

import os
import json
import logging
from datetime import datetime
from llm_fsm import API

# --------------------------------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("book_recommender.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("BookRecommender")


class BookRecommendationSystem:
    def __init__(self, fsm_path, api_key, model="gpt-4o", temperature=0.7):
        """Initialize the book recommendation system."""
        self.fsm_path = fsm_path
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.fsm = None
        self.conversation_id = None
        self.recommended_books = []
        self.session_start_time = datetime.now()

        # Initialize the FSM
        self._initialize_fsm()

    def _initialize_fsm(self):
        """Initialize the FSM with the provided configuration."""
        try:
            logger.info(f"Initializing FSM from {self.fsm_path}")
            self.fsm = API.from_file(
                path=self.fsm_path,
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature
            )
            logger.info("FSM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FSM: {str(e)}")
            raise

    def start_conversation(self, initial_context=None):
        """Start a new conversation with the book recommendation system."""
        try:
            logger.info("Starting new conversation")
            self.conversation_id, response = self.fsm.converse(
                user_message="",  # Empty for initial greeting
                initial_context=initial_context or {}
            )
            logger.info(f"Conversation started with ID: {self.conversation_id}")
            return response
        except Exception as e:
            logger.error(f"Failed to start conversation: {str(e)}")
            raise

    def process_message(self, user_input):
        """Process a user message and get the system's response."""
        if not self.conversation_id:
            logger.error("No active conversation")
            raise ValueError("No active conversation. Call start_conversation() first.")

        try:
            logger.info(f"Processing user message: {user_input}")

            # Process the user input
            _, response = self.fsm.converse(user_input, self.conversation_id)

            # Track the current state
            current_state = self.fsm.get_current_state(self.conversation_id)
            logger.info(f"Current state: {current_state}")

            # Track books (more advanced extraction could be implemented)
            if current_state == "recommend_book" or (
                current_state == "check_engagement" and
                self.fsm.get_data(self.conversation_id).get("_prev_state") == "recommend_book"
            ):
                # Simple extraction - this is a basic heuristic
                # A more robust system would store book details in the context
                for line in response.split('\n'):
                    if "title" in line.lower() or "book" in line.lower():
                        if ":" in line:
                            possible_title = line.split(":", 1)[1].strip()
                            if possible_title and possible_title not in self.recommended_books:
                                self.recommended_books.append(possible_title)
                                logger.info(f"Extracted book: {possible_title}")
                        break

            return response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise

    def is_conversation_ended(self):
        """Check if the conversation has ended."""
        if not self.conversation_id:
            return True
        return self.fsm.has_conversation_ended(self.conversation_id)

    def get_recommended_books(self):
        """Get the list of recommended books."""
        return self.recommended_books

    def get_user_preferences(self):
        """Get the user's genre preferences."""
        if not self.conversation_id:
            return None

        context = self.fsm.get_data(self.conversation_id)
        return context.get("genres", None)

    def end_conversation(self):
        """End the current conversation."""
        if self.conversation_id:
            logger.info(f"Ending conversation: {self.conversation_id}")
            self.fsm.end_conversation(self.conversation_id)
            self.conversation_id = None

    def get_session_summary(self):
        """Get a summary of the current session."""
        duration = datetime.now() - self.session_start_time
        minutes = int(duration.total_seconds() // 60)
        seconds = int(duration.total_seconds() % 60)

        return {
            "duration": f"{minutes} minutes, {seconds} seconds",
            "books_recommended": len(self.recommended_books),
            "book_list": self.recommended_books,
            "user_preferences": self.get_user_preferences()
        }


def main():
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=your-api-key-here")
        return

    # Load the FSM definition from the JSON file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fsm_path = os.path.join(current_dir, "fsm.json")

    try:
        # Create the book recommendation system
        recommender = BookRecommendationSystem(
            fsm_path=fsm_path,
            api_key=api_key,
            model="gpt-4o",  # Change as needed
            temperature=0.8   # Higher for more creativity in recommendations
        )

        # Start the conversation
        print("Starting Book Recommendation System...")
        response = recommender.start_conversation()
        print(f"System: {response}")

        # Main conversation loop
        while not recommender.is_conversation_ended():
            # Get user input
            user_input = input("You: ")

            # Check for manual exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting conversation.")
                break

            # Process the message
            try:
                response = recommender.process_message(user_input)
                print(f"System: {response}")

                # Check if conversation has ended
                if recommender.is_conversation_ended():
                    print("Conversation has ended.")

                    # Show session summary
                    summary = recommender.get_session_summary()

                    print("\n--- Session Summary ---")
                    print(f"Duration: {summary['duration']}")

                    if summary["user_preferences"]:
                        print(f"Your genre preferences: {summary['user_preferences']}")

                    if summary["books_recommended"] > 0:
                        print(f"\nBooks recommended ({summary['books_recommended']}):")
                        for i, book in enumerate(summary["book_list"], 1):
                            print(f"{i}. {book}")
                    else:
                        print("\nNo books were recommended in this session.")

                    print("\nThank you for using the Book Recommendation System!")

            except Exception as e:
                print(f"Error: {str(e)}")
                logger.error(f"Error in main loop: {str(e)}")

        # Clean up
        recommender.end_conversation()

    except FileNotFoundError:
        print(f"Error: Could not find FSM definition at {fsm_path}")
        print("Make sure to create fsm.json with the Book Recommendation System definition")
    except json.JSONDecodeError:
        print(f"Error: The FSM definition file at {fsm_path} contains invalid JSON")
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Fatal error: {str(e)}")

# --------------------------------------------------------------


if __name__ == "__main__":
    main()

# --------------------------------------------------------------
