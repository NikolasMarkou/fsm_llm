"""
LLM-FSM Simplified API: A user-friendly entry point for working with Finite State Machines
powered by Large Language Models.

This module provides a simplified interface for common operations with LLM-FSM,
making it easier to create, manage, and interact with conversational FSMs.
"""

import os
from typing import Dict, Any, Optional, Tuple, List, Union

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .fsm import FSMManager
from .llm import LiteLLMInterface, LLMInterface
from .definitions import FSMDefinition, FSMError
from .logging import logger, handle_conversation_errors

# --------------------------------------------------------------


class API:
    """
    Main entry point for working with LLM-FSM.

    This class provides a simplified interface for using the LLM-FSM framework,
    hiding the complexity of the underlying components while maintaining all
    the power and flexibility of the full system.

    It can handle multiple conversations, each one accessed through the conversation_id
    """

    def __init__(self,
                 fsm_definition: Union[FSMDefinition, Dict[str, Any], str],
                 llm_interface: Optional[LLMInterface] = None,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 max_history_size: int = 5,
                 max_message_length: int = 1000,
                 **llm_kwargs):
        """
        Initialize an LLM-FSM instance.

        Args:
            fsm_definition: FSM definition as an object, dictionary, or path to file
            llm_interface: Optional custom LLM interface instance. If provided, other LLM parameters are ignored.
            model: The LLM model to use (e.g., "gpt-4o", "claude-3-opus"). Used only if llm_interface is None.
            api_key: Optional API key (will use environment variables if not provided). Used only if llm_interface is None.
            temperature: LLM temperature parameter (0.0-1.0). Used only if llm_interface is None.
            max_tokens: Maximum tokens for LLM responses. Used only if llm_interface is None.
            max_history_size: Maximum number of exchanges to keep in conversation history
            max_message_length: Maximum length of messages in characters
            **llm_kwargs: Additional keyword arguments to pass to LiteLLMInterface constructor if llm_interface is None

        Raises:
            ValueError: If neither llm_interface nor model is provided
        """
        # Handle LLM interface initialization
        if llm_interface is not None:
            # Use provided custom LLM interface
            if not isinstance(llm_interface, LLMInterface):
                raise ValueError("llm_interface must be an instance of LLMInterface")
            self.llm_interface = llm_interface
            logger.info(f"LLM_FSM initialized with custom LLM interface: {type(llm_interface).__name__}")
        else:
            # Create default LiteLLMInterface with provided parameters
            # Set defaults for LiteLLMInterface parameters
            model = model or "gpt-4o"
            temperature = temperature if temperature is not None else 0.5
            max_tokens = max_tokens if max_tokens is not None else 1000

            self.llm_interface = LiteLLMInterface(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **llm_kwargs
            )
            logger.info(f"LLM_FSM initialized with default LiteLLM interface, model={model}")

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

        logger.info(f"LLM_FSM fully initialized with max_history_size={max_history_size}")

    @classmethod
    def from_file(cls, path: str, **kwargs) -> 'API':
        """
        Create an LLM-FSM instance from a JSON file.

        Args:
            path: Path to the FSM definition JSON file
            **kwargs: Additional arguments to pass to the constructor (llm_interface, model, api_key, etc.)

        Returns:
            An initialized API instance

        Raises:
            FileNotFoundError: If the file doesn't exist

        Examples:
            >>> # Using default LiteLLM interface
            >>> api = API.from_file("fsm.json", model="gpt-4o", temperature=0.7)

            >>> # Using custom LLM interface
            >>> custom_llm = MyCustomLLMInterface()
            >>> api = API.from_file("fsm.json", llm_interface=custom_llm)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"FSM definition file not found: {path}")

        return cls(fsm_definition=path, **kwargs)

    @classmethod
    def from_definition(cls,
                       fsm_definition: Union[FSMDefinition, Dict[str, Any]],
                       **kwargs) -> 'API':
        """
        Create an LLM-FSM instance from an FSM definition object or dictionary.

        Args:
            fsm_definition: FSM definition as an object or dictionary
            **kwargs: Additional arguments to pass to the constructor (llm_interface, model, api_key, etc.)

        Returns:
            An initialized API instance

        Examples:
            >>> # Using default LiteLLM interface
            >>> api = API.from_definition(my_fsm_dict, model="claude-3-opus")

            >>> # Using custom LLM interface
            >>> custom_llm = MyCustomLLMInterface()
            >>> api = API.from_definition(my_fsm_definition, llm_interface=custom_llm)
        """
        return cls(fsm_definition=fsm_definition, **kwargs)

    def start_conversation(
            self,
            initial_context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
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
                 conversation_id: str) -> str:
        """
        Process a message and return the response,
        this must be an existing already started conversation

        Args:
            user_message: The user's message
            conversation_id: ID for an existing conversation

        Returns:
            response

        Raises:
            ValueError: If the conversation ID is not found or invalid
            FSMError: If there's an error in the FSM processing
        """
        try:
            response = self.fsm_manager.process_message(conversation_id, user_message)
            return response
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
    def has_conversation_ended(self, conversation_id: str) -> bool:
        """
        Check if a conversation has ended (reached a terminal state).

        Args:
            conversation_id: The conversation ID

        Returns:
            True if the conversation has ended, False otherwise

        Raises:
            ValueError: If the conversation ID is not found
        """
        return self.fsm_manager.has_conversation_ended(conversation_id)

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

    def get_llm_interface(self) -> LLMInterface:
        """
        Get the currently used LLM interface.

        Returns:
            The LLM interface instance being used by this API instance

        Examples:
            >>> api = API.from_file("fsm.json", model="gpt-4o")
            >>> llm_interface = api.get_llm_interface()
            >>> print(f"Using model: {llm_interface.model}")
        """
        return self.llm_interface

    def close(self) -> None:
        """
        Clean up all active conversations and resources.

        This method can be called explicitly to clean up when done
        using the API instance.
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
        the API class in a 'with' statement.
        """
        self.close()

# --------------------------------------------------------------