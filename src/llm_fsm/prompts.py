"""
PromptBuilder: Configurable Structured System Prompt Generation for LLM-FSM

The PromptBuilder is responsible for generating structured prompts that instruct
Large Language Models (LLMs) how to operate within a Finite State Machine (FSM).
It converts FSM state definitions, context data, and conversation history into
carefully formatted prompts that enable consistent state transitions and natural
language understanding.

Enhanced with comprehensive configuration options for history management,
token budgets, and prompt customization.

Key Features:
-------------
1. Configurable conversation history limits (by message count and token budget)
2. Flexible prompt structure with optional sections
3. Comprehensive text sanitization and security measures
4. Token budget management with conservative estimation
5. Extensible configuration system

Security Considerations:
-----------------------
This PromptBuilder implements comprehensive security measures to prevent prompt
injection attacks and structure corruption:

- All user and LLM-generated text is sanitized before inclusion in prompts
- XML-like tags are escaped to prevent structure breaks
- CDATA sections protect JSON data from interference
- Robust regex patterns handle various tag formats including self-closing tags
- Error handling for non-serializable objects prevents runtime failures

"""

import re
import json
import html
import textwrap
from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .logging import logger
from .definitions import FSMInstance, State
from .constants import DEFAULT_MAX_HISTORY_SIZE


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

class HistoryManagementStrategy(str, Enum):
    """Strategy for managing conversation history."""
    MESSAGE_COUNT = "message_count"  # Limit by number of messages
    TOKEN_BUDGET = "token_budget"    # Limit by estimated token count
    HYBRID = "hybrid"                # Use both limits (whichever is hit first)


@dataclass(frozen=True)
class PromptConfig:
    """
    Comprehensive configuration for PromptBuilder behavior.

    This configuration class allows fine-tuning of all aspects of prompt
    generation, from history management to content inclusion.
    """

    # History Management
    max_history_messages: int = DEFAULT_MAX_HISTORY_SIZE
    max_token_budget: int = 3000
    history_strategy: HistoryManagementStrategy = HistoryManagementStrategy.HYBRID
    enable_history_compression: bool = False  # Future feature for compressing old messages

    # Token Estimation
    chars_per_token: float = 2.5  # Conservative estimate
    token_estimation_factor: float = 1.3  # Safety factor for overhead
    json_overhead_factor: float = 1.3  # JSON serialization overhead
    utf8_expansion_factor: float = 1.5  # UTF-8 multi-byte expansion
    cdata_overhead_tokens: int = 50  # Overhead for CDATA wrapping

    # Content Inclusion Flags
    include_examples: bool = False
    include_persona: bool = True
    include_state_instructions: bool = True
    include_conversation_history: bool = True
    include_context_data: bool = True

    # Prompt Structure
    enable_detailed_guidelines: bool = True
    enable_format_rules: bool = True
    enable_information_collection_instructions: bool = True

    # Warning Thresholds
    token_warning_threshold_delta: int = 500
    enable_token_warnings: bool = True

    # Development/Debug Options
    deterministic_output: bool = True  # Sort transitions for consistent testing
    verbose_logging: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_history_messages < 0:
            raise ValueError("max_history_messages must be non-negative")
        if self.max_token_budget < 100:
            raise ValueError("max_token_budget must be at least 100")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")

        if self.verbose_logging:
            logger.debug(f"PromptConfig initialized: {self}")


# ============================================================================
# FEW-SHOT EXAMPLES
# ============================================================================

FEW_SHOT_EXAMPLES = textwrap.dedent("""
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
    """).strip()


# ============================================================================
# MAIN PROMPT BUILDER CLASS
# ============================================================================

@dataclass(frozen=True)
class PromptBuilder:
    """
    Configurable builder for creating structured prompts for LLMs in FSM contexts.

    The PromptBuilder creates structured prompts that provide the LLM with:
    - Current state information and purpose
    - Available transitions and their conditions
    - Current context data (configurable)
    - Conversation history (with flexible management)
    - Response format requirements
    - Optional few-shot examples and guidelines

    All aspects of prompt generation can be configured through PromptConfig.
    """

    config: PromptConfig = field(default_factory=PromptConfig)

    def __post_init__(self):
        """Log configuration after initialization."""
        if self.config.verbose_logging:
            logger.debug(f"PromptBuilder initialized with config: {self.config}")
        else:
            logger.debug(
                f"PromptBuilder initialized - history: {self.config.max_history_messages} messages, "
                f"token budget: {self.config.max_token_budget}, strategy: {self.config.history_strategy.value}"
            )

    # ========================================================================
    # TEXT SANITIZATION AND SECURITY
    # ========================================================================

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
        critical_tags = [
            "fsm", "task", "current_state", "transitions", "response", "guidelines",
            "format_rules", "persona", "valid_states", "information_to_collect",
            "examples", "state_instructions", "id", "description", "purpose",
            "conversation_history", "current_context"
        ]

        # Create pattern for tags with potential attributes, whitespace, or self-closing notation
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
        return text.replace("]]>", "]]&gt;")

    # ========================================================================
    # HISTORY MANAGEMENT
    # ========================================================================

    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for a given text using conservative estimates.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        char_count = len(text)
        adjusted_count = (
            char_count *
            self.config.json_overhead_factor *
            self.config.utf8_expansion_factor *
            self.config.token_estimation_factor
        )
        return int(adjusted_count / self.config.chars_per_token)

    def _limit_history_by_message_count(self, exchanges: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Limit history by maximum number of messages.

        Args:
            exchanges: List of conversation exchanges

        Returns:
            Limited list of exchanges
        """
        if len(exchanges) <= self.config.max_history_messages:
            return exchanges

        result = exchanges[-self.config.max_history_messages:]

        if self.config.verbose_logging:
            logger.debug(
                f"Limited history by message count: {len(exchanges)} -> {len(result)} exchanges"
            )

        return result

    def _limit_history_by_token_budget(self, exchanges: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Limit history by estimated token budget.

        Args:
            exchanges: List of conversation exchanges

        Returns:
            Limited list of exchanges that fit within token budget
        """
        if not exchanges:
            return exchanges

        # Account for CDATA and JSON overhead
        available_tokens = max(self.config.max_token_budget - self.config.cdata_overhead_tokens, 1)

        # Start from the most recent and work backwards
        result = []
        current_tokens = 0

        for exchange in reversed(exchanges):
            # Estimate tokens for this exchange
            exchange_json = json.dumps(exchange, separators=(",", ": "))
            exchange_tokens = self._estimate_token_count(exchange_json)

            # Check if adding this exchange would exceed budget
            if current_tokens + exchange_tokens > available_tokens and result:
                break

            result.insert(0, exchange)  # Insert at beginning to maintain order
            current_tokens += exchange_tokens

        if len(result) < len(exchanges) and self.config.verbose_logging:
            logger.debug(
                f"Limited history by token budget: {len(exchanges)} -> {len(result)} exchanges "
                f"(estimated {current_tokens} tokens, budget: {available_tokens})"
            )

        return result

    def _manage_conversation_history(self, exchanges: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Apply the configured history management strategy.

        Args:
            exchanges: Original list of exchanges

        Returns:
            Managed list of exchanges according to configuration
        """
        if not exchanges:
            return exchanges

        if self.config.history_strategy == HistoryManagementStrategy.MESSAGE_COUNT:
            return self._limit_history_by_message_count(exchanges)

        elif self.config.history_strategy == HistoryManagementStrategy.TOKEN_BUDGET:
            return self._limit_history_by_token_budget(exchanges)

        elif self.config.history_strategy == HistoryManagementStrategy.HYBRID:
            # Apply both limits and use the more restrictive result
            by_count = self._limit_history_by_message_count(exchanges)
            by_tokens = self._limit_history_by_token_budget(exchanges)

            # Return the shorter result (more restrictive)
            result = by_count if len(by_count) <= len(by_tokens) else by_tokens

            if len(result) < len(exchanges) and self.config.verbose_logging:
                logger.debug(
                    f"Applied hybrid history management: {len(exchanges)} -> {len(result)} exchanges "
                    f"(count limit: {len(by_count)}, token limit: {len(by_tokens)})"
                )

            return result

        else:
            logger.warning(f"Unknown history strategy: {self.config.history_strategy}")
            return exchanges

    # ========================================================================
    # PROMPT BUILDING METHODS
    # ========================================================================

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

    def _build_task_section(self) -> List[str]:
        """Build the task section of the prompt."""
        task_description = textwrap.dedent("""
            You are the Natural Language Understanding component in a Finite State Machine (FSM) based conversational system.
            Your responsibilities:
            - Process user input based on current state (<current_state>)
            - Collect required information from input to `context_update`
            - Select appropriate transitions from <transitions>
            - Generate messages based on the instructions
            - Follow the <response> instructions to generate valid JSON output
            """).strip()

        return [
            "<task>",
            task_description,
            "</task>"
        ]

    def _build_persona_section(self, instance: FSMInstance) -> List[str]:
        """Build the persona section if enabled and available."""
        parts = []
        if self.config.include_persona and instance.persona:
            parts.extend([
                "<persona>",
                textwrap.dedent(self._sanitize_text_for_prompt(instance.persona)).strip(),
                "</persona>"
            ])
        return parts

    def _build_current_state_section(self, state: State) -> List[str]:
        """Build the current state section."""
        parts = [
            "<current_state>",
            f"<id>{self._sanitize_text_for_prompt(state.id)}</id>",
            f"<description>{self._sanitize_text_for_prompt(state.description)}</description>",
            f"<purpose>{self._sanitize_text_for_prompt(state.purpose)}</purpose>"
        ]

        # Add instructions if enabled and available
        if self.config.include_state_instructions and state.instructions:
            parts.extend([
                "<state_instructions>",
                textwrap.dedent(self._sanitize_text_for_prompt(state.instructions)).strip(),
                "</state_instructions>"
            ])

        # Add required context keys
        if state.required_context_keys:
            safe_keys = [self._sanitize_text_for_prompt(key) for key in state.required_context_keys]
            parts.append("<information_to_collect>")
            for safe_key in safe_keys:
                parts.append(f"<collect>{safe_key}</collect>")
            parts.append("</information_to_collect>")

            # Add collection instructions if enabled
            if self.config.enable_information_collection_instructions:
                instructions = textwrap.dedent("""
                    - Collect all required information explicitly mentioned by the user.
                    - If information is ambiguous or unclear, ask for clarification.
                    - Store collected information in the `context_update` field of your response.
                    - Only transition to a new state when all required information is collected.
                    - If extra information seems relevant but the key is not declared, nest it under `_extra`.
                    - Do not create or populate undeclared context keys, except within the `_extra` object.
                    """).strip()

                parts.extend([
                    "<information_collection_instructions>",
                    textwrap.dedent(instructions).strip(),
                    "</information_collection_instructions>"
                ])

        parts.append("</current_state>")
        return parts

    def _build_context_section(self, instance: FSMInstance) -> List[str]:
        """Build the current context section if enabled."""
        parts = []

        if not self.config.include_context_data:
            return parts

        safe_context_json = "{}"
        if instance.context.data:
            try:
                raw_json = json.dumps(instance.context.data, indent=1, separators=(",", ": "))
                safe_context_json = self._escape_cdata(raw_json)
            except (TypeError, ValueError) as e:
                logger.warning(f"Context data contains non-serializable objects: {e}")
                try:
                    raw_json = json.dumps(instance.context.data, indent=1, separators=(",", ": "), default=str)
                    safe_context_json = self._escape_cdata(raw_json)
                except Exception as e2:
                    logger.error(f"Failed to serialize context data: {e2}")
                    safe_context_json = "{}"

        parts.extend([
            "<current_context><![CDATA[",
            safe_context_json,
            "]]></current_context>"
        ])

        return parts

    def _build_history_section(self, instance: FSMInstance) -> List[str]:
        """Build the conversation history section if enabled."""
        parts = []

        if not self.config.include_conversation_history:
            return parts

        # Get recent exchanges
        recent_exchanges = instance.context.conversation.get_recent(self.config.max_history_messages * 2)  # Get more initially

        if not recent_exchanges:
            return parts

        # Format and sanitize exchanges
        formatted_exchanges = []
        for exchange in recent_exchanges:
            formatted_exchange = {}
            for role, text in exchange.items():
                role_lower = role.lower()
                sanitized_text = self._sanitize_text_for_prompt(text)

                if role_lower == "user":
                    formatted_exchange["user"] = sanitized_text
                elif role_lower == "assistant":
                    formatted_exchange["assistant"] = sanitized_text
                else:
                    formatted_exchange["system"] = sanitized_text
            formatted_exchanges.append(formatted_exchange)

        # Apply history management strategy
        managed_exchanges = self._manage_conversation_history(formatted_exchanges)

        if managed_exchanges:
            history_json = json.dumps(managed_exchanges, indent=1, separators=(",", ": "))
            safe_history_json = self._escape_cdata(history_json)

            parts.extend([
                "<conversation_history><![CDATA[",
                safe_history_json,
                "]]></conversation_history>"
            ])

        return parts

    def _build_transitions_section(self, state: State) -> List[str]:
        """Build the valid states and transitions section."""
        parts = []

        # Get transition targets
        transition_targets = set()
        if state.transitions:
            transition_targets = set(t.target_state for t in state.transitions)

        available_states = list(transition_targets | {state.id})
        if self.config.deterministic_output:
            available_states.sort()

        # Add valid states
        safe_states = [self._sanitize_text_for_prompt(state_id) for state_id in available_states]
        parts.append("<valid_states>")
        for safe_state in safe_states:
            parts.append(f"\t<state>{safe_state}</state>")
        parts.append("</valid_states>")

        # Build transitions data
        transitions_data = []
        max_priority = 0

        if state.transitions:
            max_priority = max((t.priority for t in state.transitions), default=0)

            for transition in state.transitions:
                transition_info = {
                    "to": self._sanitize_text_for_prompt(transition.target_state),
                    "desc": self._sanitize_text_for_prompt(transition.description),
                    "priority": transition.priority
                }

                if transition.conditions:
                    conditions = []
                    for condition in transition.conditions:
                        condition_info = {
                            "desc": self._sanitize_text_for_prompt(condition.description),
                            "keys": []
                        }

                        if condition.requires_context_keys:
                            condition_info["keys"] = [
                                self._sanitize_text_for_prompt(key)
                                for key in condition.requires_context_keys
                            ]
                        conditions.append(condition_info)
                    transition_info["conditions"] = conditions

                transitions_data.append(transition_info)

        # Add self-transition if not present
        if state.id not in transition_targets:
            transitions_data.append({
                "to": self._sanitize_text_for_prompt(state.id),
                "desc": "Remain in current state if needed",
                "priority": max_priority + 100
            })

        # Sort for deterministic output
        if self.config.deterministic_output:
            transitions_data.sort(key=lambda t: (t["priority"], t["to"]))

        # Add transitions as JSON
        transitions_json = json.dumps(transitions_data, indent=1, separators=(",", ": "))
        safe_transitions_json = self._escape_cdata(transitions_json)

        parts.extend([
            "<transitions><![CDATA[",
            safe_transitions_json,
            "]]></transitions>"
        ])

        return parts

    def _build_response_section(self) -> List[str]:
        """Build the response format section."""
        json_schema = textwrap.dedent("""
            {
                "transition": {
                    "target_state": "state_id",
                    "context_update": {
                        "key1": "value1",
                        "_extra": {}
                    }
                },
                "message": "Your message",
                "reasoning": "Your reasoning"
            }""").strip()

        return [
            "<response>",
            "Your response must be valid JSON with the following structure:",
            json_schema,
            "Where:",
            "\t- `transition.target_state` is REQUIRED and must be one of the valid states.",
            "\t- `transition.context_update` is REQUIRED, containing any extracted information.",
            "\t- `message` is REQUIRED and contains the user-facing text.",
            "\t- `reasoning` is OPTIONAL and explains your decision (not shown to user).",
            "\t- `_extra` is for storing relevant information not explicitly requested.",
            "</response>"
        ]

    def _build_examples_section(self) -> List[str]:
        """Build the examples section if enabled."""
        parts = []
        if self.config.include_examples:
            sanitized_examples = self._sanitize_text_for_prompt(FEW_SHOT_EXAMPLES)
            safe_examples = self._escape_cdata(sanitized_examples)
            parts.extend([
                "<examples><![CDATA[",
                safe_examples,
                "]]></examples>"
            ])
        return parts

    def _build_guidelines_section(self, instance: FSMInstance) -> List[str]:
        """Build the guidelines section if enabled."""
        parts = []

        if not self.config.enable_detailed_guidelines:
            return parts

        guidelines = textwrap.dedent("""
            - Extract all required information from user input
            - Store relevant information even if unexpected (using `_extra`)
            - Reference current context for continuity
            - Only transition when conditions are met
            """).strip()

        parts.extend([
            "<guidelines>",
            guidelines
        ])

        if self.config.include_persona and instance.persona:
            parts.append("- Maintain the specified persona consistently")
        else:
            parts.append("- Keep messages conversational and natural")

        parts.append("</guidelines>")
        return parts

    def _build_format_rules_section(self) -> List[str]:
        """Build the format rules section if enabled."""
        parts = []

        if self.config.enable_format_rules:
            rules = textwrap.dedent("""
                - Return ONLY valid JSON - no markdown code fences, no additional explanations, no comments.
                - Do not add keys not specified in the schema.
                - Ensure all values are properly quoted and formatted according to JSON standards.
                - Do not mention any of the above to the user. You can use the context, but never show it to the user
                """).strip()

            parts.extend([
                "<format_rules>",
                rules,
                "</format_rules>"
            ])

        return parts

    def _build_prompt_parts(self, instance: FSMInstance, state: State) -> List[str]:
        """
        Build the individual parts of the system prompt.

        Args:
            instance: The FSM instance containing context and conversation history
            state: The current state definition

        Returns:
            A list of prompt parts before joining

        Raises:
            ValueError: If state properties are invalid
        """
        self._validate_state(state)

        # Build all sections
        prompt_parts = []

        # Task section
        prompt_parts.extend(self._build_task_section())

        # FSM wrapper start
        prompt_parts.append("<fsm>")

        # Persona (early in prompt for maximum impact)
        prompt_parts.extend(self._build_persona_section(instance))

        # Current state
        prompt_parts.extend(self._build_current_state_section(state))

        # Context
        prompt_parts.extend(self._build_context_section(instance))

        # History
        prompt_parts.extend(self._build_history_section(instance))

        # Transitions
        prompt_parts.extend(self._build_transitions_section(state))

        # Response format
        prompt_parts.extend(self._build_response_section())

        # Examples
        prompt_parts.extend(self._build_examples_section())

        # Guidelines
        prompt_parts.extend(self._build_guidelines_section(instance))

        # Format rules
        prompt_parts.extend(self._build_format_rules_section())

        # FSM wrapper end
        prompt_parts.append("</fsm>")

        return prompt_parts

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def build_system_prompt(self, instance: FSMInstance, state: State) -> str:
        """
        Build a system prompt for the current state with all configured options.

        Args:
            instance: The FSM instance containing context and conversation history
            state: The current state definition

        Returns:
            A system prompt string using XML-like structure with JSON data

        Raises:
            ValueError: If state properties are invalid for prompt building
        """
        if self.config.verbose_logging:
            logger.debug(f"Building system prompt for state: {state.id}")

        # Get prompt parts and join
        prompt_parts = self._build_prompt_parts(instance, state)
        prompt = "\n".join(prompt_parts)

        # Token budget safety check if warnings are enabled
        if self.config.enable_token_warnings:
            approx_tokens = self._estimate_token_count(prompt)
            warning_threshold = self.config.max_token_budget + self.config.token_warning_threshold_delta

            if approx_tokens > warning_threshold:
                logger.warning(
                    f"Prompt may be approaching token limit: ~{approx_tokens} tokens "
                    f"(warning threshold: {warning_threshold})"
                )

        return prompt

    def get_prompt_parts(self, instance: FSMInstance, state: State) -> List[str]:
        """
        Get the individual prompt parts for testing or inspection.

        Args:
            instance: The FSM instance containing context and conversation history
            state: The current state definition

        Returns:
            List of prompt parts before joining
        """
        return self._build_prompt_parts(instance, state)

    def estimate_prompt_tokens(self, instance: FSMInstance, state: State) -> int:
        """
        Estimate the token count for a prompt that would be generated.

        Args:
            instance: The FSM instance containing context and conversation history
            state: The current state definition

        Returns:
            Estimated token count
        """
        prompt = self.build_system_prompt(instance, state)
        return self._estimate_token_count(prompt)

    def get_history_info(self, instance: FSMInstance) -> Dict[str, Any]:
        """
        Get information about how history would be managed for this instance.

        Args:
            instance: The FSM instance to analyze

        Returns:
            Dictionary with history management information
        """
        total_exchanges = len(instance.context.conversation.get_recent(1000))  # Get all
        recent_exchanges = instance.context.conversation.get_recent(self.config.max_history_messages * 2)

        # Apply management strategy to see results
        managed_exchanges = self._manage_conversation_history(recent_exchanges) if recent_exchanges else []

        return {
            "total_exchanges": total_exchanges,
            "available_exchanges": len(recent_exchanges),
            "managed_exchanges": len(managed_exchanges),
            "strategy": self.config.history_strategy.value,
            "max_messages_config": self.config.max_history_messages,
            "max_tokens_config": self.config.max_token_budget,
            "estimated_tokens": self._estimate_token_count(json.dumps(managed_exchanges)) if managed_exchanges else 0
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_prompt_builder(
    max_history_messages: Optional[int] = None,
    max_token_budget: Optional[int] = None,
    include_examples: bool = False,
    history_strategy: Optional[HistoryManagementStrategy] = None,
    **kwargs
) -> PromptBuilder:
    """
    Convenience function to create a PromptBuilder with common configurations.

    Args:
        max_history_messages: Maximum number of history messages to include
        max_token_budget: Maximum token budget for history
        include_examples: Whether to include few-shot examples
        history_strategy: Strategy for managing conversation history
        **kwargs: Additional configuration options

    Returns:
        Configured PromptBuilder instance
    """
    config_kwargs = {}

    if max_history_messages is not None:
        config_kwargs['max_history_messages'] = max_history_messages
    if max_token_budget is not None:
        config_kwargs['max_token_budget'] = max_token_budget
    if history_strategy is not None:
        config_kwargs['history_strategy'] = history_strategy

    config_kwargs['include_examples'] = include_examples
    config_kwargs.update(kwargs)

    config = PromptConfig(**config_kwargs)
    return PromptBuilder(config)