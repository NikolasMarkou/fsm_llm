"""
Generic FSM Loader for LLM-FSM Framework

This module provides functionality to load any JSON-based FSM definition
into the LLM-FSM framework and interact with it.
"""

import os
import json
from typing import Dict, Any, Optional, List

from .fsm import (
    FSMDefinition, LLMInterface, LLMRequest, LLMResponse,
    FSMManager, StateTransition
)


def load_fsm(file_path: str) -> FSMDefinition:
    """
    Load an FSM definition from a JSON file.

    Args:
        file_path:

    Returns:
        An FSMDefinition object

    Raises:
        FileNotFoundError: If the FSM file doesn't exist
        ValueError: If the FSM definition is invalid
    """
    # Load and parse the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            fsm_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"FSM definition file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in FSM definition: {e}")

    # Parse into an FSMDefinition
    try:
        fsm_definition = FSMDefinition.model_validate(fsm_data)
    except Exception as e:
        raise ValueError(f"Error parsing FSM definition: {e}")

    return fsm_definition


class SimpleLLMInterface(LLMInterface):
    """
    Simple implementation of the LLM interface that uses a provided LLM function.
    """

    def __init__(self, llm_function):
        """
        Initialize the LLM interface.

        Args:
            llm_function: A function that takes a prompt and returns a response
        """
        self.llm_function = llm_function

    def send_request(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to the LLM and parse the response.

        Args:
            request: The LLM request

        Returns:
            The LLM's response
        """
        # Construct the prompt
        prompt = f"""
SYSTEM: {request.system_prompt}

USER: {request.user_message}
"""

        # Send to the LLM
        response_text = self.llm_function(prompt)

        # Extract and parse the JSON response
        response_json = self._extract_json(response_text)
        if not response_json:
            # If JSON extraction fails, create a default response staying in the same state
            return LLMResponse(
                transition=StateTransition(
                    target_state=request.context.get("current_state", ""),
                    context_update={}
                ),
                message=response_text,
                reasoning="Failed to extract structured response"
            )

        # Parse into an LLMResponse
        try:
            # Extract the transition data
            transition_data = response_json.get("transition", {})
            target_state = transition_data.get("target_state", "")
            context_update = transition_data.get("context_update", {})

            # Create the response object
            return LLMResponse(
                transition=StateTransition(
                    target_state=target_state,
                    context_update=context_update
                ),
                message=response_json.get("message", ""),
                reasoning=response_json.get("reasoning", "")
            )
        except Exception as e:
            # If parsing fails, create a default response
            return LLMResponse(
                transition=StateTransition(
                    target_state=request.context.get("current_state", ""),
                    context_update={}
                ),
                message=response_text,
                reasoning=f"Error parsing response: {str(e)}"
            )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from the LLM's response text.

        Args:
            text: The LLM's response text

        Returns:
            Extracted JSON as a dictionary, or None if extraction fails
        """
        import re
        import json

        # Try to find JSON between code blocks first
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object in the text
        json_pattern = r'{[\s\S]*}'
        json_match = re.search(json_pattern, text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None


class FSMRunner:
    """
    Class to run an FSM conversation.
    """

    def __init__(
            self,
            fsm_directory: str,
            llm_function
    ):
        """
        Initialize the FSM runner.

        Args:
            fsm_directory: Directory containing FSM JSON files
            llm_function: Function that interfaces with an LLM
        """
        # Set up the FSM loader
        self.fsm_loader = FSMLoader(fsm_directory)

        # Set up the LLM interface
        self.llm_interface = SimpleLLMInterface(llm_function)

        # Set up the FSM manager
        self.fsm_manager = FSMManager(
            fsm_loader=self.fsm_loader.load_fsm,
            llm_interface=self.llm_interface
        )

    def create_conversation(self, fsm_id: str) -> Any:
        """
        Create a new conversation with the specified FSM.

        Args:
            fsm_id: Identifier for the FSM (filename without extension)

        Returns:
            An FSM instance
        """
        return self.fsm_manager.create_instance(fsm_id)

    def process_message(
            self,
            instance: Any,
            user_message: str
    ) -> str:
        """
        Process a user message and get the response.

        Args:
            instance: The FSM instance
            user_message: The user's message

        Returns:
            The response to send to the user
        """
        try:
            updated_instance, response = self.fsm_manager.process_user_input(
                instance, user_message
            )
            return response
        except Exception as e:
            return f"Error processing message: {str(e)}"

    def get_conversation_history(self, instance: Any) -> List[Dict[str, str]]:
        """
        Get the conversation history.

        Args:
            instance: The FSM instance

        Returns:
            The conversation history
        """
        return instance.context.conversation.exchanges


# Example usage
def example_usage():
    # Directory containing FSM JSON files
    fsm_directory = "path/to/fsm/definitions"

    # Example LLM function - replace with actual implementation
    def example_llm_function(prompt):
        # This would call your actual LLM API
        print("Sending to LLM:", prompt)
        # Mock response - in a real implementation this would come from the LLM
        return """
        I'll help you with that!

        ```json
        {
          "transition": {
            "target_state": "collect_email",
            "context_update": {
              "name": "John Doe"
            }
          },
          "message": "Thanks for providing your name, John. Could you please share your email address?",
          "reasoning": "The user provided their full name, so I've stored it and am moving to collect their email."
        }
        ```
        """

    # Create the FSM runner
    runner = FSMRunner(fsm_directory, example_llm_function)

    # Create a new conversation
    instance = runner.create_conversation("linear_information_gathering")

    # Process user messages
    response = runner.process_message(instance, "Hello, I'm John Doe.")
    print("System:", response)

    # Get conversation history
    history = runner.get_conversation_history(instance)
    print("Conversation history:", history)


if __name__ == "__main__":
    example_usage()