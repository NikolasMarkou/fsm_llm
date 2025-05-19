"""
PromptBuilder module for LLM-FSM.

This module contains the PromptBuilder class responsible for creating
structured prompts that guide the LLM's behavior within the finite state machine.
"""

import json
from typing import List, Dict, Any

# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .logging import logger
from .definitions import FSMInstance, State
from .constants import DEFAULT_MAX_HISTORY_SIZE

# --------------------------------------------------------------


class PromptBuilder:
    """
    Builder for creating prompts for the LLM using JSON and markdown.

    The PromptBuilder creates structured prompts that provide the LLM with:
    - Current state information and purpose
    - Available transitions and their conditions
    - Current context data
    - Conversation history
    - Response format requirements

    These prompts help the LLM understand its role within the FSM and make
    appropriate state transition decisions.
    """

    def __init__(self, max_history_size: int = DEFAULT_MAX_HISTORY_SIZE):
        """
        Initialize the PromptBuilder.

        Args:
            max_history_size: Default number of conversation history exchanges to include in prompts
        """
        self.max_history_size = max_history_size
        logger.debug(f"PromptBuilder initialized with max_history_size={max_history_size}")

    def build_system_prompt(self, instance: FSMInstance, state: State) -> str:
        """
        Build a system prompt for the current state with instructions about valid transitions.

        Args:
            instance: The FSM instance containing context and conversation history
            state: The current state definition

        Returns:
            A system prompt string using markdown and JSON formatting
        """
        logger.debug(f"Building system prompt for state: {state.id}")

        # Get available states for transitions
        available_states = [t.target_state for t in state.transitions]
        available_states_str = ", ".join([f"'{s}'" for s in available_states])

        # Build the markdown prompt structure
        prompt_parts = [
            "<task>",
            "You are operating as part of a Finite State Machine (FSM) for conversational AI.",
            "In this process, you move between defined states based on input, extracting information and making appropriate state transitions while maintaining conversation context.",
            "All FSM-specific information will be enclosed in <fsm> and </fsm> tags below. It defines your current operational context.",
            "You MUST follow information in <state>, <purpose>, and <instructions> tags to understand your role.",
            "You MUST extract information specified in <information_to_collect> tags following the <information_extraction_instructions>.",
            "You MUST only use transitions defined in <transition_instructions> and adhere to <transition_rules>.",
            "You MUST reference <current_context> to maintain conversation continuity and use <conversation_history> for context.",
            "You MUST respond in the exact JSON format specified in the <response_format> tag and follow all <instructions>.",
            "</task>"
        ]

        # Log important context for debugging
        logger.debug(f"Building prompt for state: {state.id}")
        logger.debug(f"Available transitions: {available_states}")
        logger.debug(f"Required context keys: {state.required_context_keys if state.required_context_keys else 'None'}")

        # FSM Header
        prompt_parts.append("<fsm>")
        prompt_parts.append(f"<current_state>")
        prompt_parts.append(f"<name>{state.id}</name>")
        prompt_parts.append(f"<description>{state.description}</description>")
        prompt_parts.append(f"<purpose>{state.purpose}</purpose>")
        # Add instructions if available
        if state.instructions:
            prompt_parts.append("<state_instructions>")
            prompt_parts.append(state.instructions)
            prompt_parts.append("</state_instructions>")

        # Add required context keys and extraction instructions
        if state.required_context_keys:
            prompt_parts.append("<information_to_collect>")
            prompt_parts.append(", ".join(state.required_context_keys))
            prompt_parts.append("</information_to_collect>")

            prompt_parts.append("<information_extraction_instructions>")
            prompt_parts.append("- Extract all required information explicitly mentioned by the user.")
            prompt_parts.append("- If information is ambiguous or unclear, ask for clarification.")
            prompt_parts.append("- Store extracted information in the `context_update` field of your response.")
            prompt_parts.append("- Only transition to a new state when all required information is collected.")
            prompt_parts.append("</information_extraction_instructions>")
        prompt_parts.append(f"</current_state>")
        prompt_parts.append("")

        # Add persona if available - place this early in the prompt for maximum impact
        if instance.persona:
            prompt_parts.append("<persona>")
            prompt_parts.append(instance.persona)
            prompt_parts.append("</persona>")

        # Add current context as JSON
        prompt_parts.append("<current_context>")
        if instance.context.data:
            prompt_parts.append(json.dumps(instance.context.data, indent=2))
        else:
            prompt_parts.append("No context data available")
        prompt_parts.append("</current_context>")

        # Get conversation history using a configured history size
        max_history = getattr(instance.context.conversation, 'max_history_size', self.max_history_size)
        recent_exchanges = instance.context.conversation.get_recent(max_history)

        # Add conversation history as JSON
        if recent_exchanges:
            prompt_parts.append("<conversation_history>")

            # Convert exchanges to a standard format
            formatted_exchanges = []
            for exchange in recent_exchanges:
                formatted_exchange = {}
                for role, text in exchange.items():
                    role_lower = role.lower()
                    # Standardize role names
                    if role_lower == "user":
                        formatted_exchange["user"] = text
                    else:
                        formatted_exchange["system"] = text
                formatted_exchanges.append(formatted_exchange)

            # Add the conversation history as JSON
            prompt_parts.append(json.dumps(formatted_exchanges, indent=2))

            prompt_parts.append("</conversation_history>")
            prompt_parts.append("")

        # Add response format instructions with JSON schema
        prompt_parts.append("<response>")
        prompt_parts.append("Respond with a JSON object with the following schema inside the <response_format> tags:")
        prompt_parts.append("<response_format>")
        prompt_parts.append(json.dumps({
            "transition": {
                "target_state": "state_id",
                "context_update": {"key1": "value1", "key2": "value2"}
            },
            "message": "Your message to the user",
            "reasoning": "Your reasoning for the state transition decision"
        }, indent=2))
        prompt_parts.append("</response_format>")
        prompt_parts.append("<response_details>")
        prompt_parts.append("- `transition.target_state`: Must be one of the valid states listed in <transition_instructions>.")
        prompt_parts.append("- `transition.context_update`: Should contain key-value pairs of information extracted from the user message.")
        prompt_parts.append("- `message`: The text that will be shown to the user. Should be conversational and natural.")
        prompt_parts.append("- `reasoning`: Optional field explaining your decision-making process. Not shown to the user.")
        prompt_parts.append("- The `transition` field is used internally by the system and should not be mentioned to the user.")
        prompt_parts.append("- The `message` field will be shown directly to the user.")
        prompt_parts.append("- Do NOT mention information from <transition_instructions> in the message.")
        prompt_parts.append("- The `reasoning` field is optional and helps explain your decision process.")
        prompt_parts.append("- When extracting information, ensure keys in `context_update` match exactly what was specified in <information_to_collect>.")
        prompt_parts.append("- If you're uncertain about information, stay in the current state and ask for clarification.")
        prompt_parts.append("- Your message should acknowledge the information provided by the user when appropriate.")
        prompt_parts.append("- Never transition to a state that isn't listed in the available transitions.")
        prompt_parts.append("</response_details>")
        prompt_parts.append("</response>")

        # Add available transitions as JSON
        prompt_parts.append("<transition_instructions>")
        prompt_parts.append("- This is a list of valid transitions for the current state.")
        prompt_parts.append("- Each transition has a target state and a description.")
        prompt_parts.append("- The description should be informative and provide a detailed explanation of the transition's purpose.")
        prompt_parts.append("- Do not use any of this information when forming the message for the user")
        if state.transitions:
            transitions_data = []

            for transition in state.transitions:
                transition_info = {
                    "target_state": transition.target_state,
                    "description": transition.description,
                    "priority": transition.priority
                }

                # Add conditions if any
                if transition.conditions:
                    conditions = []
                    for condition in transition.conditions:
                        condition_info = {
                            "description": condition.description
                        }
                        if condition.requires_context_keys:
                            condition_info["required_keys"] = condition.requires_context_keys
                        conditions.append(condition_info)
                    transition_info["conditions"] = conditions

                transitions_data.append(transition_info)

            # Format transitions as JSON
            prompt_parts.append(json.dumps(transitions_data, indent=2))
        else:
            prompt_parts.append("This state has no outgoing transitions.")

        prompt_parts.append("</transition_instructions>")

        # Add transition rules
        if available_states:
            prompt_parts.append("<transition_rules>")
            prompt_parts.append(f"- You MUST ONLY choose from the following valid target states: [{available_states_str}]")
            prompt_parts.append(f"- Do NOT invent or create new states that are not in the list above.")
            prompt_parts.append(f"- Do NOT mention information from <transition_instructions> in the message")
            prompt_parts.append(f"- If you're unsure which state to transition to, stay in the current state.")
            prompt_parts.append(f"- The current state is '{state.id}' - you can choose to stay here if needed.")
            prompt_parts.append("</transition_rules>")

        # Add important guidelines
        prompt_parts.append("<instructions>")
        prompt_parts.append("- Collect all required information from the user's message")
        prompt_parts.append("- Only transition to a new state if all required information is collected or another transition is appropriate")
        prompt_parts.append("- Maintain context continuity across the conversation")
        prompt_parts.append("- If user provides information relevant to a different state, still collect and store it")
        prompt_parts.append("- Utilize the current context in your conversation message")

        # Add persona guideline if a persona is specified
        if instance.persona:
            prompt_parts.append("- Maintain the specified persona and tone in all your responses")
            prompt_parts.append("- Adapt your language style and formality based on the persona description")
        else:
            prompt_parts.append("- Your message should be conversational and natural.")

        # Add final reminder about valid target states
        prompt_parts.append(f"- Remember, you can ONLY choose from these valid target states: [{available_states_str}]")
        prompt_parts.append("</instructions>")
        prompt_parts.append("</fsm>")

        # Join all parts with newlines
        prompt = "\n".join(prompt_parts)
        logger.debug(f"System prompt length: {len(prompt)} characters")

        return prompt