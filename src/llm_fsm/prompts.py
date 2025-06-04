"""
PromptBuilder: Structured System Prompt Generation for LLM-FSM

The PromptBuilder is responsible for generating structured prompts that instruct
Large Language Models (LLMs) how to operate within a Finite State Machine (FSM).
It converts FSM state definitions, context data, and conversation history into
carefully formatted prompts that enable consistent state transitions and natural
language understanding.

Key Responsibilities:
--------------------
1. Creating structured XML-like prompts for LLMs to follow
2. Ensuring consistent state transitions according to FSM definitions
3. Maintaining conversation context and history
4. Sanitizing all text to prevent prompt structure corruption
5. Managing token budgets to avoid exceeding LLM context limits

Security Considerations:
-----------------------
This PromptBuilder implements comprehensive security measures to prevent prompt
injection attacks and structure corruption:

- All user and LLM-generated text is sanitized before inclusion in prompts
- XML-like tags are escaped to prevent structure breaks
- CDATA sections protect JSON data from interference
- Robust regex patterns handle various tag formats including self-closing tags
- Error handling for non-serializable objects prevents runtime failures

Token Management:
----------------
The builder implements token budget management to prevent exceeding LLM context limits:
- Dynamic conversation history trimming based on token estimation
- Token overhead accounting for sanitization and CDATA wrapping
- Warning thresholds to identify potential context window issues

Usage Example:
-------------
    from llm_fsm.prompt_builder import PromptBuilder
    from llm_fsm.definitions import FSMInstance, State

    # Initialize the builder
    builder = PromptBuilder(max_token_budget=3000)

    # Build a prompt for the current state
    prompt = builder.build_system_prompt(fsm_instance, current_state)

    # Send the prompt to an LLM provider
    llm_response = llm_client.generate(prompt)

Design Decisions:
----------------
- XML-like Structure: Provides clear section boundaries for LLMs to follow
- CDATA Protection: Ensures valid JSON can be embedded without escaping issues
- Comprehensive Sanitization: Prevents any tag-like structures from breaking prompts
- Deterministic Output: Ensures consistent, testable prompt generation
- Token Optimization: Minimizes whitespace in embedded JSON

The structure of generated prompts is as follows:
<task>...</task>
<fsm>
  <current_state>...</current_state>
  <current_context>...</current_context>
  <conversation_history>...</conversation_history>
  <valid_states>...</valid_states>
  <transitions>...</transitions>
  <response>...</response>
  <examples>...</examples>
  <guidelines>...</guidelines>
  <format_rules>...</format_rules>
</fsm>

This design allows the LLM to understand its current state, valid transitions,
available context, and expected response format - enabling predictable and
reliable state machine behavior while leveraging the LLM's natural language
capabilities.
"""

import re
import json
import html
from typing import List, Dict
from dataclasses import dataclass


# --------------------------------------------------------------
# local imports
# --------------------------------------------------------------

from .logging import logger
from .definitions import FSMInstance, State
from .constants import DEFAULT_MAX_HISTORY_SIZE

# --------------------------------------------------------------

# Few-shot examples to improve LLM understanding
FEW_SHOT_EXAMPLES = """
Example 1:
User message: "My name is John Smith"
Current state: collect_name
Required information: name

Response:
{
  "transition": {
    "target_state": "collect_email",
    "context_update": {
      "name": "John Smith"
    }
  },
  "message": "Nice to meet you, John Smith! Could you please provide your email address?",
  "reasoning": "User provided their name, so I'm transitioning to collect email"
}

Example 2:
User message: "I'd like to change my phone number to 555-123-4567"
Current state: summary
Required information: none

Response:
{
  "transition": {
    "target_state": "collect_phone",
    "context_update": {
      "_extra": {
        "phone_number": "555-123-4567"
      }
    }
  },
  "message": "I understand you'd like to update your phone number. Let me help you with that.",
  "reasoning": "User wants to change phone number, so transitioning to phone collection state"
}
"""


@dataclass(frozen=True)
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

    max_history_size: int = DEFAULT_MAX_HISTORY_SIZE
    max_token_budget: int = 3000  # Approximate token budget for history
    token_warning_threshold_delta: int = 500  # Delta added to max_token_budget for warning threshold
    include_examples: bool = False  # Whether to include few-shot examples

    def __post_init__(self):
        """Log the effective max_history_size after initialization."""
        logger.debug(f"PromptBuilder initialized with effective max_history_size={self.max_history_size}")

    def _sanitize_text_for_prompt(self, text: str) -> str:
        """
        Sanitize text to prevent XML tag confusion in prompts.

        Escapes both opening and closing tags (with or without attributes)
        that might interfere with prompt structure.

        Args:
            text: The text to sanitize

        Returns:
            Sanitized text that won't break XML-like structure
        """
        if text is None:
            return ""

        # Critical tags that could break the prompt structure
        critical_tags = ["fsm", "task", "current_state", "transitions",
                         "response", "guidelines", "format_rules", "persona",
                         "valid_states", "information_to_collect", "examples",
                         "state_instructions", "id", "description", "purpose",
                         "conversation_history", "current_context"]

        # Create pattern for tags with potential attributes, whitespace, or self-closing notation
        # Using [^>]* to avoid over-greedy matches that could span multiple tags
        tag_pattern = r'</?(?:' + '|'.join(critical_tags) + r')(?:[^>]*)?/?>'
        return re.sub(tag_pattern, lambda m: html.escape(m.group(0)), text, flags=re.IGNORECASE)

    def _escape_cdata(self, text: str) -> str:
        """
        Escape CDATA end sequences in text to prevent breaking CDATA blocks.

        Args:
            text: Text that will be placed inside a CDATA block

        Returns:
            Text with CDATA end markers safely escaped
        """
        # Replace any CDATA end markers with a safe equivalent
        return text.replace("]]>", "]]&gt;")

    def _clip_history(self, exchanges: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Clip history to stay within token budget.

        Fixed to use more conservative token estimation.

        Args:
            exchanges: List of conversation exchanges

        Returns:
            Trimmed list of exchanges that fits within token budget
        """
        if not exchanges:
            return exchanges

        # Make a copy to avoid modifying the original
        result = list(exchanges)

        # FIXED: More conservative token estimation
        # Assume average of 2.5 chars per token (more conservative than 4)
        # Account for JSON overhead, CDATA wrapping, and potential UTF-8 expansion
        cdata_overhead = 100  # Increased overhead for safety
        json_overhead_factor = 1.3  # JSON serialization overhead
        utf8_expansion_factor = 1.5  # Potential UTF-8 multi-byte characters
        chars_per_token = 2.5  # More conservative estimate

        adjusted_max = max(self.max_token_budget - cdata_overhead, 1)

        initial_count = len(result)

        while result:
            # Serialize and check both raw size and potential escaped size
            serialized = json.dumps(result, separators=(",", ": "))

            # Calculate conservative token estimate
            char_count = len(serialized)
            adjusted_char_count = char_count * json_overhead_factor * utf8_expansion_factor
            token_estimate = int(adjusted_char_count / chars_per_token)

            if token_estimate <= adjusted_max:
                break

            # Remove oldest exchange and try again
            result.pop(0)

        # If we had to trim, log a warning with token estimates
        if len(result) < initial_count:
            logger.warning(
                f"Clipped conversation history from {initial_count} to {len(result)} exchanges "
                f"(estimated {token_estimate} tokens, adjusted max {adjusted_max})"
            )

        return result

    def _validate_state(self, state: State) -> None:
        """
        Validate state properties to ensure prompt building will succeed.

        Args:
            state: The state to validate

        Raises:
            ValueError: If state has invalid properties that would break prompt building
        """
        if not state.id:
            raise ValueError("State ID cannot be empty")

        # Additional validations could be added here as needed

    def _build_prompt_parts(self, instance: FSMInstance, state: State) -> List[str]:
        """
        Build the individual parts of the system prompt.

        This method is exposed to facilitate unit testing of prompt sections.

        Args:
            instance: The FSM instance containing context and conversation history
            state: The current state definition

        Returns:
            A list of prompt parts before joining

        Raises:
            ValueError: If state properties are invalid
        """
        # Validate state properties first
        self._validate_state(state)

        # Build the markdown prompt structure
        prompt_parts = [
            "<task>",
            "You are the Natural Language Understanding component in a Finite State Machine (FSM) based conversational system.",
            "Your responsibilities:",
            "- Process user input based on current state (<current_state>)",
            "- Collect required information from input to `context_update`",
            "- Select appropriate transitions from <transitions>",
            "- Generate messages based on the instructions",
            "- Follow the <response> instructions to generate valid JSON output",
            "</task>"
        ]

        # FSM Header
        prompt_parts.append("<fsm>")
        # Add persona if available - place this early in the prompt for maximum impact
        if instance.persona:
            prompt_parts.append("<persona>")
            prompt_parts.append(self._sanitize_text_for_prompt(instance.persona))
            prompt_parts.append("</persona>")

        # Current state information - sanitize ALL fields
        safe_state_id = self._sanitize_text_for_prompt(state.id)
        safe_state_desc = self._sanitize_text_for_prompt(state.description)
        safe_state_purpose = self._sanitize_text_for_prompt(state.purpose)

        prompt_parts.append("<current_state>")
        prompt_parts.append(f"<id>{safe_state_id}</id>")
        prompt_parts.append(f"<description>{safe_state_desc}</description>")
        prompt_parts.append(f"<purpose>{safe_state_purpose}</purpose>")

        # Add instructions if available - sanitize
        if state.instructions:
            prompt_parts.append("<state_instructions>")
            prompt_parts.append(self._sanitize_text_for_prompt(state.instructions))
            prompt_parts.append("</state_instructions>")

        # Add required context keys and extraction instructions
        if state.required_context_keys:
            # Sanitize each key to be safe
            safe_keys = [self._sanitize_text_for_prompt(key) for key in state.required_context_keys]
            prompt_parts.append("<information_to_collect>")
            for safe_key in safe_keys:
                prompt_parts.append(f"\t<collect>{safe_key}</collect>")
            #prompt_parts.append(", ".join(safe_keys))
            prompt_parts.append("</information_to_collect>")

            prompt_parts.append("<information_collection_instructions>")
            prompt_parts.append("- Collect all required information explicitly mentioned by the user.")
            prompt_parts.append("- If information is ambiguous or unclear, ask for clarification.")
            prompt_parts.append("- Store collected information in the `context_update` field of your response.")
            prompt_parts.append("- Only transition to a new state when all required information is collected.")
            prompt_parts.append("- If extra information seems relevant but the key is not declared, nest it under `_extra`.")
            prompt_parts.append("- Do not create or populate undeclared context keys, except within the `_extra` object.")
            prompt_parts.append("</information_collection_instructions>")
        prompt_parts.append("</current_state>")

        # Add current context as JSON with CDATA protection
        safe_context_json = "{}"
        if instance.context.data:
            try:
                # Use compact JSON serialization with minimal indentation
                raw_json = json.dumps(instance.context.data,
                                      indent=1,
                                      separators=(",", ": "))
                # Escape any CDATA end markers
                safe_context_json = self._escape_cdata(raw_json)
            except (TypeError, ValueError) as e:
                # Handle non-serializable objects
                logger.warning(f"Context data contains non-serializable objects: {e}")
                # Try with default=str to convert non-serializable objects
                try:
                    raw_json = json.dumps(instance.context.data,
                                         indent=1,
                                         separators=(",", ": "),
                                         default=str)
                    safe_context_json = self._escape_cdata(raw_json)
                except Exception as e2:
                    logger.error(f"Failed to serialize context data even with str conversion: {e2}")
                    # Fall back to empty object
                    safe_context_json = "{}"

        prompt_parts.append("<current_context><![CDATA[")
        prompt_parts.append(safe_context_json)
        prompt_parts.append("]]></current_context>")

        # Get conversation history using a configured history size
        max_history = getattr(instance.context.conversation, 'max_history_size', None) or self.max_history_size
        recent_exchanges = instance.context.conversation.get_recent(max_history)

        # Add conversation history as JSON with CDATA protection and token budget
        if recent_exchanges:
            # Convert exchanges to a standard format with proper role separation and sanitization
            formatted_exchanges = []
            for exchange in recent_exchanges:
                formatted_exchange = {}
                for role, text in exchange.items():
                    role_lower = role.lower()
                    # Sanitize ALL text to prevent tag issues
                    sanitized_text = self._sanitize_text_for_prompt(text)

                    # Properly distinguish between roles
                    if role_lower == "user":
                        formatted_exchange["user"] = sanitized_text
                    elif role_lower == "assistant":
                        formatted_exchange["assistant"] = sanitized_text
                    else:
                        formatted_exchange["system"] = sanitized_text
                formatted_exchanges.append(formatted_exchange)

            # Clip history to fit token budget
            trimmed_exchanges = self._clip_history(formatted_exchanges)

            # Serialize history and escape any CDATA end markers
            history_json = json.dumps(trimmed_exchanges,
                               indent=1,
                               separators=(",", ": "))
            safe_history_json = self._escape_cdata(history_json)

            prompt_parts.append("<conversation_history><![CDATA[")
            prompt_parts.append(safe_history_json)
            prompt_parts.append("]]></conversation_history>")

        # Get transition targets from state
        transition_targets = set()
        if state.transitions:
            transition_targets = set(t.target_state for t in state.transitions)
        available_states = list(transition_targets | {state.id})
        available_states.sort()  # Make ordering deterministic for testing

        # Add valid states as a separate section - sanitize each state ID
        safe_states = [self._sanitize_text_for_prompt(state_id) for state_id in available_states]
        prompt_parts.append("<valid_states>")
        # doing xml here on purpose
        for safe_state in safe_states:
            prompt_parts.append(f"\t<state>{safe_state}</state>")
        # prompt_parts.append(", ".join(safe_states))  # Removed quotes to match schema
        prompt_parts.append("</valid_states>")

        # Add available transitions as JSON
        transitions_data = []
        # Find the maximum priority for self-transition
        max_priority = 0

        # Guard against None transitions
        if state.transitions:
            max_priority = max((t.priority for t in state.transitions), default=0)

            # Process transitions - sanitize all text fields
            for transition in state.transitions:
                transition_info = {
                    "to": self._sanitize_text_for_prompt(transition.target_state),
                    "desc": self._sanitize_text_for_prompt(transition.description),
                    "priority": transition.priority
                }

                # Add conditions if any - sanitize all text fields
                if transition.conditions:
                    conditions = []
                    for condition in transition.conditions:
                        condition_info = {
                            "desc": self._sanitize_text_for_prompt(condition.description),
                            "keys": []
                        }

                        # Sanitize each key
                        if condition.requires_context_keys:
                            condition_info["keys"] = [
                                self._sanitize_text_for_prompt(key)
                                for key in condition.requires_context_keys
                            ]

                        conditions.append(condition_info)
                    transition_info["conditions"] = conditions

                transitions_data.append(transition_info)

        # Add a "self" transition explicitly if not already defined
        if state.id not in transition_targets:
            transitions_data.append({
                "to": self._sanitize_text_for_prompt(state.id),
                "desc": "Remain in current state if needed",
                "priority": max_priority + 100  # Ensure lowest priority
            })

        # Sort transitions for deterministic output in tests
        transitions_data.sort(key=lambda t: (t["priority"], t["to"]))

        # Format transitions as JSON with CDATA protection
        transitions_json = json.dumps(transitions_data,
                            indent=1,
                            separators=(",", ": "))
        safe_transitions_json = self._escape_cdata(transitions_json)

        prompt_parts.append("<transitions><![CDATA[")
        prompt_parts.append(safe_transitions_json)
        prompt_parts.append("]]></transitions>")

        # Add response format instructions with JSON schema and _extra example
        prompt_parts.append("<response>")
        prompt_parts.append("Your response must be valid JSON with the following structure:")
        prompt_parts.append(
'''{
    "transition": {
        "target_state": "state_id",
        "context_update": {
            "key1": "value1",
            "_extra": {}
        }
    },
    "message": "Your message",
    "reasoning": "Your reasoning"
}''')
        prompt_parts.append("Where:")
        prompt_parts.append("\t- `transition.target_state` is REQUIRED and must be one of the valid states.")
        prompt_parts.append("\t- `transition.context_update` is REQUIRED, containing any extracted information.")
        prompt_parts.append("\t- `message` is REQUIRED and contains the user-facing text.")
        prompt_parts.append("\t- `reasoning` is OPTIONAL and explains your decision (not shown to user).")
        prompt_parts.append("\t- `_extra` is for storing relevant information not explicitly requested.")
        prompt_parts.append("</response>")

        # Add few-shot examples if enabled - wrap in CDATA for safety and sanitize
        if self.include_examples:
            # First sanitize the examples to handle potential XML tag issues
            sanitized_examples = self._sanitize_text_for_prompt(FEW_SHOT_EXAMPLES)
            # Then escape CDATA markers
            safe_examples = self._escape_cdata(sanitized_examples)
            prompt_parts.append("<examples><![CDATA[")
            prompt_parts.append(safe_examples)
            prompt_parts.append("]]></examples>")

        # Add important guidelines
        prompt_parts.append("<guidelines>")
        prompt_parts.append("\t- Extract all required information from user input")
        prompt_parts.append("\t- Store relevant information even if unexpected (using `_extra`)")
        prompt_parts.append("\t- Reference current context for continuity")
        prompt_parts.append("\t- Only transition when conditions are met")

        # Persona instruction condensed to a single line
        if instance.persona:
            prompt_parts.append("\t- Maintain the specified persona consistently")
        else:
            prompt_parts.append("\t- Keep messages conversational and natural")

        prompt_parts.append("</guidelines>")

        # Add format rules for reliability
        prompt_parts.append("<format_rules>")
        prompt_parts.append("\t- Return ONLY valid JSON - no markdown code fences, no additional explanations, no comments.")
        prompt_parts.append("\t- Do not add keys not specified in the schema.")
        prompt_parts.append("\t- Ensure all values are properly quoted and formatted according to JSON standards.")
        prompt_parts.append("\t- Do not mention any of the above to the user. You can use the context, but never show it to the user")
        prompt_parts.append("</format_rules>")

        prompt_parts.append("</fsm>")

        return prompt_parts

    def build_system_prompt(self, instance: FSMInstance, state: State) -> str:
        """
        Build a system prompt for the current state with instructions about valid transitions.

        Args:
            instance: The FSM instance containing context and conversation history
            state: The current state definition

        Returns:
            A system prompt string using markdown and JSON formatting

        Raises:
            ValueError: If state properties are invalid for prompt building
        """
        logger.debug(f"Building system prompt for state: {state.id}")

        # Get prompt parts and join with newlines
        prompt_parts = self._build_prompt_parts(instance, state)
        prompt = "\n".join(prompt_parts)

        # Token budget safety check (approximate) - computed from max_token_budget
        approx_tokens = len(prompt.split())
        warning_threshold = self.max_token_budget + self.token_warning_threshold_delta

        if approx_tokens > warning_threshold:
            logger.warning(
                f"Prompt may be approaching token limit: ~{approx_tokens} tokens "
                f"(warning threshold: {warning_threshold})"
            )

        return prompt

    def get_joined_prompt_parts(self, instance: FSMInstance, state: State) -> str:
        """
        Helper method for unit tests to get the full prompt string.

        Args:
            instance: The FSM instance containing context and conversation history
            state: The current state definition

        Returns:
            The full prompt string formed by joining the prompt parts
        """
        return "\n".join(self._build_prompt_parts(instance, state))

    # --------------------------------------------------------------