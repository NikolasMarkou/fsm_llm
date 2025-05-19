"""
LLM-FSM Simplified API: A user-friendly entry point for working with Finite State Machines
powered by Large Language Models.

This module provides a simplified interface for common operations with LLM-FSM,
making it easier to create, manage, and interact with conversational FSMs.
"""

import os
from functools import wraps
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .fsm import FSMManager
from .logging import logger
from .llm import LiteLLMInterface
from .definitions import FSMDefinition, FSMError

# --------------------------------------------------------------


def handle_conversation_errors(method_or_error_msg: Union[Callable, str] = None):
    """
    Decorator for handling common conversation-related errors in LLM_FSM methods.

    Can be used in two ways:
    1. @handle_conversation_errors - uses the method name in error messages
    2. @handle_conversation_errors("Custom error message") - uses the provided message
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self, conversation_id, *args, **kwargs):
            error_message = (
                method_or_error_msg if isinstance(method_or_error_msg, str)
                else f"Failed in {method.__name__}"
            )

            try:
                return method(self, conversation_id, *args, **kwargs)
            except ValueError:
                logger.error(f"Invalid conversation ID: {conversation_id}")
                raise ValueError(f"Conversation not found: {conversation_id}")
            except Exception as e:
                logger.error(f"Error in {method.__name__}: {str(e)}")
                raise FSMError(f"{error_message}: {str(e)}")
        return wrapper

    # Handle both @handle_conversation_errors and @handle_conversation_errors("msg")
    if callable(method_or_error_msg):
        return decorator(method_or_error_msg)
    return decorator

# --------------------------------------------------------------


class LLM_FSM:
    """
    Main entry point for working with LLM-FSM.

    This class provides a simplified interface for using the LLM-FSM framework,
    hiding the complexity of the underlying components while maintaining all
    the power and flexibility of the full system.

    Example usage:
    ```python
    # Create from a JSON file
    fsm = LLM_FSM.from_file("my_fsm.json", model="gpt-4")

    # Start a conversation
    conversation_id, response = fsm.start_conversation()
    print(f"System: {response}")

    # or use converse to start and send a message in one step
    conversation_id, response = fsm.converse("Hello!")
    print(f"System: {response}")

    # Continue the conversation
    _, response = fsm.converse("My name is Alice", conversation_id)
    print(f"System: {response}")

    # Get collected data
    data = fsm.get_data(conversation_id)
    print(f"Collected data: {data}")

    # When done, end or delete the conversation
    fsm.end_conversation(conversation_id)  # Marks as completed but keeps data
    fsm.delete_conversation(conversation_id)  # Completely removes all data
    ```
    """

    def __init__(self,
                 fsm_definition: Union[FSMDefinition, Dict[str, Any], str],
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 temperature: float = 0.5,
                 max_tokens: int = 1000,
                 max_history_size: int = 5,
                 max_message_length: int = 1000):
        """
        Initialize an LLM-FSM instance.

        Args:
            fsm_definition: FSM definition as an object, dictionary, or path to file
            model: The LLM model to use (e.g., "gpt-4", "claude-3-opus")
            api_key: Optional API key (will use environment variables if not provided)
            temperature: LLM temperature parameter (0.0-1.0)
            max_tokens: Maximum tokens for LLM responses
            max_history_size: Maximum number of exchanges to keep in conversation history
            max_message_length: Maximum length of messages in characters
        """
        # Initialize LLM interface
        self.llm_interface = LiteLLMInterface(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Initialize FSM manager
        self.fsm_manager = FSMManager(
            llm_interface=self.llm_interface,
            max_history_size=max_history_size,
            max_message_length=max_message_length
        )

        # Store the FSM definition
        self.fsm_definition = fsm_definition

        # Store active conversations
        self.active_conversations: Dict[str, bool] = {}

        logger.info(f"LLM_FSM initialized with model={model}, max_history_size={max_history_size}")

    @classmethod
    def from_file(cls, path: str, **kwargs) -> 'LLM_FSM':
        """
        Create an LLM-FSM instance from a JSON file.

        Args:
            path: Path to the FSM definition JSON file
            **kwargs: Additional arguments to pass to the constructor (model, api_key, etc.)

        Returns:
            An initialized LLM_FSM instance

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"FSM definition file not found: {path}")

        return cls(fsm_definition=path, **kwargs)

    def start_conversation(self, initial_context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Start a new conversation with the FSM.

        This method initializes a new conversation and returns the initial response from the FSM.

        Args:
            initial_context: Optional initial context data for user personalization

        Returns:
            A tuple of (conversation_id, initial_response)

        Raises:
            FSMError: If there's an error starting the conversation
        """
        try:
            # Start new conversation through the FSM manager
            conversation_id, response = self.fsm_manager.start_conversation(
                self.fsm_definition,
                initial_context=initial_context
            )

            # Track the conversation
            self.active_conversations[conversation_id] = True

            return conversation_id, response

        except Exception as e:
            logger.error(f"Error starting conversation: {str(e)}")
            raise FSMError(f"Failed to start conversation: {str(e)}")

    def converse(self,
                 user_message: str,
                 conversation_id: Optional[str] = None,
                 initial_context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Process a message and return the response.

        If conversation_id is None, starts a new conversation.
        If conversation_id is provided, continues that conversation.

        Args:
            user_message: The user's message
            conversation_id: Optional ID for an existing conversation
            initial_context: Optional initial context for new conversations

        Returns:
            A tuple of (conversation_id, response)

        Raises:
            ValueError: If the conversation ID is not found or invalid
            FSMError: If there's an error in the FSM processing
        """
        # Starting a new conversation
        if conversation_id is None:
            conversation_id, response = self.start_conversation(initial_context)

            # Process initial message if provided and non-empty
            if user_message.strip():
                try:
                    response = self.fsm_manager.process_message(conversation_id, user_message)
                except Exception as e:
                    logger.error(f"Error processing initial message: {str(e)}")
                    raise FSMError(f"Failed to process initial message: {str(e)}")

            return conversation_id, response

        # Continuing an existing conversation
        try:
            response = self.fsm_manager.process_message(conversation_id, user_message)
            return conversation_id, response
        except ValueError:
            logger.error(f"Invalid conversation ID: {conversation_id}")
            raise ValueError(f"Conversation not found: {conversation_id}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise FSMError(f"Failed to process message: {str(e)}")

    @handle_conversation_errors
    def get_data(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get collected data from a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            The context data collected during the conversation

        Raises:
            ValueError: If the conversation ID is not found
        """
        return self.fsm_manager.get_conversation_data(conversation_id)

    @handle_conversation_errors
    def is_conversation_ended(self, conversation_id: str) -> bool:
        """
        Check if a conversation has ended (reached a terminal state).

        Args:
            conversation_id: The conversation ID

        Returns:
            True if the conversation has ended, False otherwise

        Raises:
            ValueError: If the conversation ID is not found
        """
        return self.fsm_manager.is_conversation_ended(conversation_id)

    @handle_conversation_errors
    def get_current_state(self, conversation_id: str) -> str:
        """
        Get the current state of a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            The current state ID

        Raises:
            ValueError: If the conversation ID is not found
        """
        return self.fsm_manager.get_conversation_state(conversation_id)

    @handle_conversation_errors("Failed to end conversation")
    def end_conversation(self, conversation_id: str) -> None:
        """
        Explicitly end a conversation and clean up resources.

        This marks the conversation as completed but retains the data.
        Use delete_conversation to completely remove all data.

        Args:
            conversation_id: The conversation ID

        Raises:
            ValueError: If the conversation ID is not found
        """
        self.fsm_manager.end_conversation(conversation_id)

        # Remove from active conversations
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]

    @handle_conversation_errors("Failed to delete conversation")
    def delete_conversation(self, conversation_id: str) -> None:
        """
        Completely delete a conversation and remove all associated data.

        Unlike end_conversation which just marks a conversation as ended,
        this method completely removes all data related to the conversation
        from memory.

        Args:
            conversation_id: The conversation ID

        Raises:
            ValueError: If the conversation ID is not found
        """
        # First end the conversation (which validates the ID exists)
        self.fsm_manager.end_conversation(conversation_id)

        # Then remove the conversation from the FSM manager's instances
        if conversation_id in self.fsm_manager.instances:
            del self.fsm_manager.instances[conversation_id]

        # Remove from active conversations tracking
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]

        logger.info(f"Conversation {conversation_id} completely deleted")

    @handle_conversation_errors
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Get the conversation history for a specific conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            A list of conversation exchanges (user and system messages)

        Raises:
            ValueError: If the conversation ID is not found
        """
        complete_data = self.fsm_manager.get_complete_conversation(conversation_id)
        return complete_data.get("conversation_history", [])

    @handle_conversation_errors("Failed to save conversation")
    def save_conversation(self, conversation_id: str, path: str) -> None:
        """
        Save a conversation's state to a file for later resumption.

        Args:
            conversation_id: The conversation ID
            path: The file path to save to

        Raises:
            ValueError: If the conversation ID is not found
            IOError: If the file cannot be written
        """
        try:
            import json

            # Get complete conversation data
            complete_data = self.fsm_manager.get_complete_conversation(conversation_id)

            # Save to file
            with open(path, 'w') as f:
                json.dump(complete_data, f, indent=2)

            logger.info(f"Conversation {conversation_id} saved to {path}")

        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            raise IOError(f"Failed to save conversation: {str(e)}")

    def list_active_conversations(self) -> List[str]:
        """
        List all active conversation IDs.

        Returns:
            A list of active conversation IDs
        """
        return list(self.active_conversations.keys())

    def close(self) -> None:
        """
        Clean up all active conversations and resources.

        This method can be called explicitly to clean up when done
        using the LLM_FSM instance.
        """
        for conversation_id in list(self.active_conversations.keys()):
            try:
                self.end_conversation(conversation_id)
            except Exception as e:
                logger.warning(f"Error ending conversation {conversation_id} during cleanup: {str(e)}")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - clean up all active conversations.

        This ensures all conversations are properly ended when using
        the LLM_FSM class in a 'with' statement.
        """
        self.close()
