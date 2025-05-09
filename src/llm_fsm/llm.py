import os
import time
import json
import litellm
from typing import Optional
from litellm import completion, get_supported_openai_params


from .logging import logger
from .definitions import LLMRequest, LLMResponse, LLMResponseSchema, LLMResponseError, StateTransition, FSMInstance, State

class LLMInterface:
    """
    Interface for communicating with LLMs.
    """

    def send_request(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to the LLM and get the response.

        Args:
            request: The LLM request

        Returns:
            The LLM's response

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement send_request")


class LiteLLMInterface(LLMInterface):
    """
    Implementation of LLMInterface using LiteLLM.

    This class uses LiteLLM to send requests to various LLM providers
    while maintaining a consistent interface.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        enable_json_validation: bool = True,
        **kwargs
    ):
        """
        Initialize the LiteLLM interface.

        Args:
            model: The model to use (e.g., "gpt-4", "claude-3-opus")
            api_key: Optional API key (will use environment variables if not provided)
            enable_json_validation: Whether to enable JSON schema validation
            **kwargs: Additional arguments to pass to LiteLLM
        """
        self.model = model
        self.kwargs = kwargs

        logger.info(f"Initializing LiteLLMInterface with model: {model}")

        # Extract provider from model name for API key setting
        if api_key:
            # Simple provider detection, can be expanded
            if "gpt" in model.lower() or "openai" in model.lower():
                os.environ["OPENAI_API_KEY"] = api_key
                logger.debug("Setting OPENAI_API_KEY environment variable")
            elif "claude" in model.lower() or "anthropic" in model.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key
                logger.debug("Setting ANTHROPIC_API_KEY environment variable")
            else:
                # For other providers, we'll need to determine the right env var
                # or pass it directly to LiteLLM
                self.kwargs["api_key"] = api_key
                logger.debug("Using API key directly in LiteLLM kwargs")
        else:
            logger.debug("No API key provided, assuming it's set in environment variables")

        # Enable JSON schema validation if needed
        if enable_json_validation:
            litellm.enable_json_schema_validation = True
            logger.debug("Enabled JSON schema validation in LiteLLM")

    def send_request(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to the LLM using LiteLLM and get the response.

        Args:
            request: The LLM request

        Returns:
            The LLM's response

        Raises:
            LLMResponseError: If there's an error processing the LLM response
        """
        try:
            start_time = time.time()

            # Log the request (truncated for brevity)
            logger.info(f"Sending request to {self.model}")
            logger.debug(f"User message: {request.user_message[:50]}{'...' if len(request.user_message) > 50 else ''}")

            # Prepare messages for LiteLLM
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_message}
            ]

            # Check if the model supports structured output (response_format or json_schema)
            supported_params = get_supported_openai_params(model=self.model)
            logger.debug(f"Supported parameters for {self.model}: {', '.join(supported_params)}")

            # Decide on the response format approach
            if "response_format" in supported_params:
                # The model supports the OpenAI-style response_format
                logger.debug(f"Using response_format for {self.model}")
                response = completion(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    **self.kwargs
                )
            else:
                # For other models, try to use json_schema if possible
                # or fall back to parsing from unstructured output
                try:
                    if litellm.supports_response_schema(model=self.model):
                        logger.debug(f"Using json_schema for {self.model}")
                        response = completion(
                            model=self.model,
                            messages=messages,
                            response_format=LLMResponseSchema,
                            **self.kwargs
                        )
                    else:
                        # Fall back to unstructured output
                        # We'll add instruction to return JSON in the system prompt
                        logger.debug(f"Using enhanced prompt with JSON instructions for {self.model}")
                        enhanced_prompt = (
                            f"{request.system_prompt}\n\n"
                            "IMPORTANT: You must respond with a valid JSON object that follows this schema:\n"
                            "{\n"
                            '  "transition": {\n'
                            '    "target_state": "state_id",\n'
                            '    "context_update": {"key1": "value1", "key2": "value2"}\n'
                            '  },\n'
                            '  "message": "Your message to the user",\n'
                            '  "reasoning": "Your reasoning for this decision"\n'
                            "}\n"
                        )

                        # Update system message with enhanced prompt
                        messages[0] = {"role": "system", "content": enhanced_prompt}

                        response = completion(
                            model=self.model,
                            messages=messages,
                            **self.kwargs
                        )
                except Exception as schema_error:
                    # If schema approach fails, fall back to unstructured output
                    # with manual JSON instructions
                    logger.warning(f"JSON schema approach failed: {str(schema_error)}")
                    logger.debug("Falling back to basic approach with JSON instructions")

                    enhanced_prompt = (
                        f"{request.system_prompt}\n\n"
                        "IMPORTANT: You must respond with a valid JSON object that follows this schema:\n"
                        "{\n"
                        '  "transition": {\n'
                        '    "target_state": "state_id",\n'
                        '    "context_update": {"key1": "value1", "key2": "value2"}\n'
                        '  },\n'
                        '  "message": "Your message to the user",\n'
                        '  "reasoning": "Your reasoning for this decision"\n'
                        "}\n"
                    )

                    # Update system message with enhanced prompt
                    messages[0] = {"role": "system", "content": enhanced_prompt}

                    response = completion(
                        model=self.model,
                        messages=messages,
                        **self.kwargs
                    )

            # Calculate response time
            response_time = time.time() - start_time
            logger.info(f"Received response from {self.model} in {response_time:.2f}s")

            # Extract the response content
            content = response.choices[0].message.content

            # Handle different response types
            if hasattr(content, "model_dump"):
                # This is already a Pydantic model (likely from json_schema)
                logger.debug("Response is a Pydantic model")
                response_data = content.model_dump()
            else:
                # This is a string, try to parse as JSON
                try:
                    logger.debug("Parsing response as JSON")
                    response_data = json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract JSON from the text
                    logger.warning("Response is not valid JSON, attempting to extract JSON from text")
                    from .utilities import extract_json_from_text
                    extracted_json = extract_json_from_text(content)
                    if not extracted_json:
                        error_msg = f"Could not parse JSON from LLM response: {content[:100]}..."
                        logger.error(error_msg)
                        raise LLMResponseError(error_msg)
                    response_data = extracted_json
                    logger.debug("Successfully extracted JSON from text")

            # Create a StateTransition from the response
            transition_data = response_data.get("transition", {})
            transition = StateTransition(
                target_state=transition_data.get("target_state", ""),
                context_update=transition_data.get("context_update", {})
            )

            # Log the transition
            logger.info(f"Transition to: {transition.target_state}")
            if transition.context_update:
                logger.debug(f"Context updates: {json.dumps(transition.context_update)}")

            # Create and return the LLMResponse
            return LLMResponse(
                transition=transition,
                message=response_data.get("message", ""),
                reasoning=response_data.get("reasoning", None)
            )

        except Exception as e:
            # Handle exceptions
            error_msg = f"Error processing LLM response: {str(e)}"
            logger.error(error_msg)
            raise LLMResponseError(error_msg)


class PromptBuilder:
    """
    Builder for creating prompts for the LLM.
    """

    def build_system_prompt(self, instance: FSMInstance, state: State) -> str:
        """
        Build a system prompt for the current state with clearer instructions about valid transitions.

        Args:
            instance: The FSM instance
            state: The current state

        Returns:
            A system prompt string
        """
        logger.debug(f"Building system prompt for state: {state.id}")

        # Start with the basic prompt
        prompt_parts = [
            f"# {instance.fsm_id}",
            f"## Current State: {state.id}",
            f"Description: {state.description}",
            f"Purpose: {state.purpose}"
        ]

        # Add instructions if available
        if state.instructions:
            prompt_parts.append(f"\n## Instructions:\n{state.instructions}")

        # Add required context keys with enhanced instructions
        if state.required_context_keys:
            keys_str = ", ".join(state.required_context_keys)
            prompt_parts.append(f"\n## Information to collect:\n{keys_str}")

            # Special handling for name collection
            if "name" in state.required_context_keys:
                prompt_parts.append("\n## EXTRACTION INSTRUCTIONS:")
                prompt_parts.append("When the user provides their name, you MUST:")
                prompt_parts.append("1. Extract the name explicitly mentioned (e.g., 'My name is John' → 'John')")
                prompt_parts.append("2. Extract implicit name mentions (e.g., 'Call me John' → 'John')")
                prompt_parts.append(
                    "3. Store the extracted name in the context_update field as: {\"name\": \"ExtractedName\"}")
                prompt_parts.append(
                    "4. Only transition to the next state if you have successfully extracted and stored the name")
                prompt_parts.append(
                    "\nIncorrect: Responding with 'Nice to meet you, John!' but not adding {\"name\": \"John\"} to context_update")
                prompt_parts.append(
                    "Correct: Adding {\"name\": \"John\"} to context_update AND responding with 'Nice to meet you, John!'")

        # Add available transitions with explicit instructions
        available_states = [t.target_state for t in state.transitions]
        prompt_parts.append("\n## Available Transitions:")

        if available_states:
            for i, transition in enumerate(state.transitions):
                prompt_parts.append(f"{i + 1}. To '{transition.target_state}': {transition.description}")

            # Add EXPLICIT instructions about valid transitions
            prompt_parts.append("\n## IMPORTANT TRANSITION RULES:")
            prompt_parts.append("1. You MUST ONLY choose from the following valid target states:")
            prompt_parts.append("   " + ", ".join([f"'{state}'" for state in available_states]))
            prompt_parts.append("2. Do NOT invent or create new states that are not in the above list.")
            prompt_parts.append("3. If you're unsure which state to transition to, stay in the current state.")
            prompt_parts.append(f"4. The current state is '{state.id}' - you can choose to stay here if needed.")
        else:
            prompt_parts.append("This state has no outgoing transitions. Stay in the current state.")

        # Add current context with clearer formatting
        if instance.context.data:
            prompt_parts.append("\n## Current Context:")
            for key, value in instance.context.data.items():
                prompt_parts.append(f"- {key}: {value}")
        else:
            prompt_parts.append("\n## Current Context: None (empty)")

        # Add conversation history
        recent_exchanges = instance.context.conversation.get_recent(5)
        if recent_exchanges:
            prompt_parts.append("\n## Recent conversation history:")
            for exchange in recent_exchanges:
                for role, text in exchange.items():
                    prompt_parts.append(f"{role.capitalize()}: {text}")

        # Add example dialogue if available
        if state.example_dialogue:
            prompt_parts.append("\n## Example dialogue for this state:")
            for exchange in state.example_dialogue:
                for role, text in exchange.items():
                    prompt_parts.append(f"{role.capitalize()}: {text}")

        available_states = ', '.join([f"{state}" for state in available_states])
        # Add response format instructions
        prompt_parts.extend([
            "\n## Response Format:",
            "Respond with a JSON object with the following structure:",
            "```json",
            "{",
            '  "transition": {',
            '    "target_state": "state_id",',
            '    "context_update": {"key1": "value1", "key2": "value2"}',
            '  },',
            '  "message": "Your message to the user",',
            '  "reasoning": "Your reasoning for this decision"',
            "}",
            "```",
            "\nImportant:",
            "1. Collect all required information from the user's message",
            "2. Only transition to a new state if all required information is collected or another transition is appropriate",
            "3. Your message should be conversational and natural",
            "4. Don't mention states, transitions, or context keys to the user",
            f"5. Remember, you can ONLY choose from these valid target states: {available_states}"
        ])

        prompt = "\n".join(prompt_parts)
        logger.debug(f"System prompt length: {len(prompt)} characters")

        return prompt


class JSONPromptBuilder:
    """
    Builder for creating JSON-formatted prompts for the LLM (no Markdown).
    """

    def build_system_prompt(self, instance: FSMInstance, state: State) -> str:
        """
        Build a system prompt for the current state as a JSON object.

        Args:
            instance: The FSM instance
            state: The current state

        Returns:
            A JSON-formatted system prompt string
        """
        logger.debug(f"Building JSON system prompt for state: {state.id}")

        # Create base JSON structure
        prompt_data = {
            "fsm_name": instance.fsm_id,
            "current_state": {
                "id": state.id,
                "description": state.description,
                "purpose": state.purpose
            },
            "instructions": state.instructions if state.instructions else None,
            "information_to_collect": state.required_context_keys if state.required_context_keys else [],
            "extraction_guides": {},
            "transitions": [],
            "valid_target_states": [t.target_state for t in state.transitions],
            "transition_rules": [
                "You MUST ONLY choose from valid target states",
                "Do NOT invent or create new states",
                "If you're unsure which state to transition to, stay in the current state",
                f"The current state is '{state.id}' - you can choose to stay here if needed"
            ],
            "current_context": instance.context.data if instance.context.data else {},
            "conversation_history": [],
            "example_dialogue": state.example_dialogue if state.example_dialogue else [],
            "response_format": {
                "structure": {
                    "transition": {
                        "target_state": "state_id",
                        "context_update": {"key1": "value1", "key2": "value2"}
                    },
                    "message": "Your message to the user",
                    "reasoning": "Your reasoning for this decision"
                },
                "rules": [
                    "Collect all required information from the user's message",
                    "Only transition to a new state if all required information is collected",
                    "Your message should be conversational and natural",
                    "Don't mention states, transitions, or context keys to the user",
                    f"Remember, you can ONLY choose from these valid target states: {', '.join([f'{t}' for t in [t.target_state for t in state.transitions]])}"
                ]
            }
        }

        # Add special extraction guides for specific fields
        if state.required_context_keys and "name" in state.required_context_keys:
            prompt_data["extraction_guides"]["name"] = [
                "Extract names explicitly mentioned (e.g., 'My name is John' → 'John')",
                "Extract implicit name mentions (e.g., 'Call me John' → 'John')",
                "Store the extracted name in the context_update field as: {\"name\": \"ExtractedName\"}",
                "Only transition to the next state if you have successfully extracted and stored the name",
                "Incorrect: Responding with 'Nice to meet you, John!' but not adding {\"name\": \"John\"} to context_update",
                "Correct: Adding {\"name\": \"John\"} to context_update AND responding with 'Nice to meet you, John!'"
            ]

        # Add transitions with descriptions
        for transition in state.transitions:
            transition_data = {
                "target_state": transition.target_state,
                "description": transition.description,
                "priority": transition.priority
            }

            # Add condition information if available
            if transition.conditions:
                condition_data = []
                for condition in transition.conditions:
                    condition_data.append({
                        "description": condition.description,
                        "requires_context_keys": condition.requires_context_keys if condition.requires_context_keys else []
                    })
                transition_data["conditions"] = condition_data

            prompt_data["transitions"].append(transition_data)

        # Add recent conversation history
        recent_exchanges = instance.context.conversation.get_recent(5)
        if recent_exchanges:
            for exchange in recent_exchanges:
                for role, text in exchange.items():
                    prompt_data["conversation_history"].append({
                        "role": role,
                        "content": text
                    })

        # Convert to JSON string
        prompt_json = json.dumps(prompt_data, indent=2)

        logger.debug(f"JSON system prompt length: {len(prompt_json)} characters")

        return prompt_json