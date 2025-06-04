"""
Enhanced 2-Pass Prompt Builders for LLM-FSM Architecture.

This module provides specialized prompt builders that separate:
1. Content generation prompts (focused on current state and natural conversation)
2. Transition decision prompts (focused on choosing between transition options)

The architecture prevents FSM structure leakage while maintaining the rich prompting
style that produces better LLM responses. Incorporates comprehensive configuration,
token management, and security features from the original single-pass system.

Key Features:
- Information isolation prevents FSM structure leakage
- Rich XML-like prompt structure with comprehensive sections
- Configurable token budget and history management
- Comprehensive text sanitization and security measures
- Few-shot examples and detailed guidelines
- CDATA protection for JSON data
"""

import re
import json
import html
import textwrap
from enum import Enum
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------

from .logging import logger
from .definitions import (
    FSMInstance,
    State,
    TransitionOption,
    FSMDefinition
)
from .constants import DEFAULT_MAX_HISTORY_SIZE


# ============================================================================
# SHARED CONFIGURATION AND UTILITIES
# ============================================================================

class HistoryManagementStrategy(str, Enum):
    """Strategy for managing conversation history."""
    MESSAGE_COUNT = "message_count"  # Limit by number of messages
    TOKEN_BUDGET = "token_budget"  # Limit by estimated token count
    HYBRID = "hybrid"  # Use both limits (whichever is hit first)


@dataclass(frozen=True)
class BasePromptConfig:
    """Base configuration shared between both prompt builders."""

    # History Management
    max_history_messages: int = DEFAULT_MAX_HISTORY_SIZE
    max_token_budget: int = 3000
    history_strategy: HistoryManagementStrategy = HistoryManagementStrategy.HYBRID

    # Token Estimation
    chars_per_token: float = 2.5  # Conservative estimate
    token_estimation_factor: float = 1.3  # Safety factor for overhead
    json_overhead_factor: float = 1.3  # JSON serialization overhead
    utf8_expansion_factor: float = 1.5  # UTF-8 multi-byte expansion
    cdata_overhead_tokens: int = 50  # Overhead for CDATA wrapping

    # Content inclusion
    include_conversation_history: bool = True
    include_persona: bool = True

    # Security
    filter_internal_context: bool = True
    internal_key_prefixes: List[str] = field(default_factory=lambda: ['_', 'system_'])

    # Development/Debug Options
    deterministic_output: bool = True  # Sort for consistent testing
    verbose_logging: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_history_messages < 0:
            raise ValueError("max_history_messages must be non-negative")
        if self.max_token_budget < 100:
            raise ValueError("max_token_budget must be at least 100")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")


class BasePromptBuilder:
    """Base class with shared functionality for both prompt builders."""

    def __init__(self, config: BasePromptConfig = None):
        """Initialize with configuration."""
        self.config = config or BasePromptConfig()
        if self.config.verbose_logging:
            logger.debug(f"{self.__class__.__name__} initialized with config: {self.config}")

    # ========================================================================
    # TEXT SANITIZATION AND SECURITY (Enhanced from old version)
    # ========================================================================

    def _sanitize_text_for_prompt(self, text: str) -> str:
        """
        Sanitize text to prevent XML tag confusion in prompts.
        Enhanced version with more comprehensive tag protection.
        """
        if text is None:
            return ""

        # Critical tags that could break the prompt structure
        critical_tags = [
            "task", "fsm", "content_generation", "transition_decision",
            "current_state", "current_objective", "current_situation",
            "persona", "purpose", "instructions", "information_needed",
            "conversation_history", "current_context", "context_summary",
            "response_format", "examples", "guidelines", "format_rules",
            "transitions", "available_options", "option", "target", "when",
            "priority", "valid_states", "state", "information_to_collect"
        ]

        # Create pattern for tags with potential attributes, whitespace, or self-closing notation
        tag_pattern = r'</?(?:' + '|'.join(critical_tags) + r')(?:[^>]*)?/?>'
        return re.sub(tag_pattern, lambda m: html.escape(m.group(0)), text, flags=re.IGNORECASE)

    def _escape_cdata(self, text: str) -> str:
        """Escape CDATA end sequences in text to prevent breaking CDATA blocks."""
        return text.replace("]]>", "]]&gt;")

    # ========================================================================
    # HISTORY MANAGEMENT (From old version)
    # ========================================================================

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count using conservative estimates."""
        char_count = len(text)
        adjusted_count = (
                char_count *
                self.config.json_overhead_factor *
                self.config.utf8_expansion_factor *
                self.config.token_estimation_factor
        )
        return int(adjusted_count / self.config.chars_per_token)

    def _limit_history_by_message_count(self, exchanges: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Limit history by maximum number of messages."""
        if len(exchanges) <= self.config.max_history_messages:
            return exchanges

        result = exchanges[-self.config.max_history_messages:]
        if self.config.verbose_logging:
            logger.debug(f"Limited history by message count: {len(exchanges)} -> {len(result)} exchanges")
        return result

    def _limit_history_by_token_budget(self, exchanges: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Limit history by estimated token budget."""
        if not exchanges:
            return exchanges

        available_tokens = max(self.config.max_token_budget - self.config.cdata_overhead_tokens, 1)
        result = []
        current_tokens = 0

        for exchange in reversed(exchanges):
            exchange_json = json.dumps(exchange, separators=(",", ": "))
            exchange_tokens = self._estimate_token_count(exchange_json)

            if current_tokens + exchange_tokens > available_tokens and result:
                break

            result.insert(0, exchange)
            current_tokens += exchange_tokens

        if len(result) < len(exchanges) and self.config.verbose_logging:
            logger.debug(f"Limited history by token budget: {len(exchanges)} -> {len(result)} exchanges")

        return result

    def _manage_conversation_history(self, exchanges: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply the configured history management strategy."""
        if not exchanges:
            return exchanges

        if self.config.history_strategy == HistoryManagementStrategy.MESSAGE_COUNT:
            return self._limit_history_by_message_count(exchanges)
        elif self.config.history_strategy == HistoryManagementStrategy.TOKEN_BUDGET:
            return self._limit_history_by_token_budget(exchanges)
        elif self.config.history_strategy == HistoryManagementStrategy.HYBRID:
            by_count = self._limit_history_by_message_count(exchanges)
            by_tokens = self._limit_history_by_token_budget(exchanges)
            return by_count if len(by_count) <= len(by_tokens) else by_tokens
        else:
            logger.warning(f"Unknown history strategy: {self.config.history_strategy}")
            return exchanges

    def _filter_context_for_security(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter context data for security purposes."""
        if not self.config.filter_internal_context:
            return context_data

        filtered = {}
        for key, value in context_data.items():
            if not any(key.startswith(prefix) for prefix in self.config.internal_key_prefixes):
                filtered[key] = value

        return filtered

    def _humanize_key(self, key: str) -> str:
        """Convert technical context keys to human-readable descriptions."""
        key_mappings = {
            "email": "email address",
            "phone": "phone number",
            "name": "full name",
            "address": "address",
            "preferences": "preferences",
            "user_id": "user information",
            "date_of_birth": "date of birth",
            "payment_method": "payment information"
        }
        return key_mappings.get(key, key.replace("_", " "))


# ============================================================================
# CONTENT GENERATION PROMPT BUILDER
# ============================================================================

@dataclass(frozen=True)
class ContentPromptConfig(BasePromptConfig):
    """Configuration for content generation prompts."""

    # Content inclusion
    include_context_data: bool = True
    include_examples: bool = False
    include_state_instructions: bool = True

    # Prompt structure
    enable_detailed_guidelines: bool = True
    enable_format_rules: bool = True
    enable_data_extraction_guidance: bool = True


class ContentPromptBuilder(BasePromptBuilder):
    """
    Builds prompts for content generation focused on current state only.

    Enhanced with comprehensive prompting style from the original system
    while maintaining information isolation principles.
    """

    def __init__(self, config: ContentPromptConfig = None):
        """Initialize content prompt builder with configuration."""
        super().__init__(config or ContentPromptConfig())

    def build_content_prompt(
            self,
            instance: FSMInstance,
            state: State,
            fsm_definition: FSMDefinition
    ) -> str:
        """
        Build comprehensive system prompt for content generation.

        Args:
            instance: FSM instance with context and history
            state: Current state definition
            fsm_definition: FSM definition for persona

        Returns:
            System prompt focused on content generation with rich structure
        """
        logger.debug(f"Building content prompt for state: {state.id}")

        # Build comprehensive prompt sections
        sections = []

        # Task definition (enhanced from old version)
        sections.extend(self._build_enhanced_task_section())

        # Content wrapper
        sections.append("<content_generation>")

        # Persona (if available)
        if self.config.include_persona:
            sections.extend(self._build_persona_section(instance, fsm_definition))

        # Current state context (enhanced)
        sections.extend(self._build_enhanced_state_context_section(state))

        # Conversation history (with advanced management)
        if self.config.include_conversation_history:
            sections.extend(self._build_enhanced_history_section(instance))

        # Current context data (filtered and enhanced)
        if self.config.include_context_data:
            sections.extend(self._build_enhanced_context_section(instance))

        # Response format (comprehensive from old version)
        sections.extend(self._build_comprehensive_response_format())

        # Examples (if enabled)
        if self.config.include_examples:
            sections.extend(self._build_content_examples_section())

        # Guidelines (detailed from old version)
        if self.config.enable_detailed_guidelines:
            sections.extend(self._build_content_guidelines_section())

        # Format rules (from old version)
        if self.config.enable_format_rules:
            sections.extend(self._build_format_rules_section())

        sections.append("</content_generation>")

        prompt = "\n".join(sections)
        logger.debug(f"Content prompt built: {len(prompt)} characters")

        return prompt

    def _build_enhanced_task_section(self) -> List[str]:
        """Build enhanced task definition section."""
        task_description = textwrap.dedent("""
            You are the Natural Language Understanding and Generation component in a conversational AI system.
            Your responsibilities:
            - Understand user input and respond naturally and appropriately
            - Extract relevant information from user responses and store it properly
            - Maintain consistent conversation flow based on current objectives
            - Generate helpful, contextually appropriate responses

            IMPORTANT: Focus on natural conversation. Do not reference system architecture,
            states, or technical implementation details in your responses to users.
            """).strip()

        return [
            "<task>",
            task_description,
            "</task>",
            ""
        ]

    def _build_persona_section(
            self,
            instance: FSMInstance,
            fsm_definition: FSMDefinition
    ) -> List[str]:
        """Build persona section for consistent character."""
        persona = instance.persona or fsm_definition.persona
        if not persona:
            return []

        return [
            "<persona>",
            self._sanitize_text_for_prompt(persona),
            "</persona>",
            ""
        ]

    def _build_enhanced_state_context_section(self, state: State) -> List[str]:
        """Build enhanced current state context section."""
        sections = [
            "<current_objective>",
            f"<purpose>{self._sanitize_text_for_prompt(state.purpose)}</purpose>"
        ]

        # Add state-specific instructions
        if self.config.include_state_instructions and state.instructions:
            sections.extend([
                "<instructions>",
                textwrap.dedent(self._sanitize_text_for_prompt(state.instructions)).strip(),
                "</instructions>"
            ])

        # Add required information collection guidance
        if state.required_context_keys:
            sections.append("<information_needed>")
            sections.append("Try to naturally collect the following information when relevant:")
            for key in state.required_context_keys:
                natural_key = self._humanize_key(key)
                sections.append(f"- {natural_key}")
            sections.append("</information_needed>")

            # Add detailed collection instructions if enabled
            if self.config.enable_data_extraction_guidance:
                instructions = textwrap.dedent("""
                    Information Collection Guidelines:
                    - Collect information explicitly mentioned by the user
                    - If information is ambiguous, ask for clarification naturally
                    - Store collected information in the `extracted_data` field
                    - Don't force information collection if user hasn't provided it
                    - If extra relevant information is provided, include it as well
                    """).strip()

                sections.extend([
                    "<collection_guidance>",
                    instructions,
                    "</collection_guidance>"
                ])

        sections.extend([
            "</current_objective>",
            ""
        ])

        return sections

    def _build_enhanced_history_section(self, instance: FSMInstance) -> List[str]:
        """Build enhanced conversation history section."""
        recent_exchanges = instance.context.conversation.get_recent(
            self.config.max_history_messages * 2
        )

        if not recent_exchanges:
            return []

        # Format and manage exchanges
        formatted_exchanges = []
        for exchange in recent_exchanges:
            safe_exchange = {}
            for role, message in exchange.items():
                role_lower = role.lower()
                sanitized_text = self._sanitize_text_for_prompt(message)

                if role_lower == "user":
                    safe_exchange["user"] = sanitized_text
                elif role_lower == "assistant":
                    safe_exchange["assistant"] = sanitized_text
                else:
                    safe_exchange["system"] = sanitized_text
            formatted_exchanges.append(safe_exchange)

        # Apply history management strategy
        managed_exchanges = self._manage_conversation_history(formatted_exchanges)

        if managed_exchanges:
            try:
                history_json = json.dumps(managed_exchanges, indent=1, separators=(",", ": "))
                safe_history_json = self._escape_cdata(history_json)
                return [
                    "<conversation_history><![CDATA[",
                    safe_history_json,
                    "]]></conversation_history>",
                    ""
                ]
            except Exception as e:
                logger.warning(f"Failed to serialize conversation history: {e}")

        return []

    def _build_enhanced_context_section(self, instance: FSMInstance) -> List[str]:
        """Build enhanced context section with security filtering."""
        if not instance.context.data:
            return []

        # Filter context for security and content generation
        user_context = self._filter_context_for_security(instance.context.data)

        if not user_context:
            return []

        try:
            context_json = json.dumps(user_context, indent=1, separators=(",", ": "))
            safe_context_json = self._escape_cdata(context_json)
            return [
                "<current_context><![CDATA[",
                safe_context_json,
                "]]></current_context>",
                ""
            ]
        except Exception as e:
            logger.warning(f"Failed to serialize context: {e}")
            return []

    def _build_comprehensive_response_format(self) -> List[str]:
        """Build comprehensive response format section (from old version style)."""
        json_schema = textwrap.dedent("""
            {
                "extracted_data": {
                    "key1": "value1",
                    "key2": "value2"
                },
                "message": "Your natural response to the user",
                "reasoning": "Brief internal reasoning (optional)"
            }""").strip()

        return [
            "<response_format>",
            "Your response must be valid JSON with the following structure:",
            json_schema,
            "",
            "Where:",
            "\t- `extracted_data` is REQUIRED, containing any information extracted from user input.",
            "\t- `message` is REQUIRED and contains the natural, user-facing response text.",
            "\t- `reasoning` is OPTIONAL and explains your decision (not shown to user).",
            "",
            "Important:",
            "\t- Return ONLY valid JSON - no markdown code fences, no additional text",
            "\t- All JSON strings must be properly quoted and escaped",
            "\t- Include empty object {} for extracted_data if no information was extracted",
            "</response_format>",
            ""
        ]

    def _build_content_examples_section(self) -> List[str]:
        """Build few-shot examples for content generation."""
        examples = textwrap.dedent("""
            Example 1:
            User: "My name is Sarah and I'm 25 years old"
            Current purpose: Collect user information

            Response:
            {
              "extracted_data": {
                "name": "Sarah",
                "age": 25
              },
              "message": "Nice to meet you, Sarah! Thanks for sharing that information."
            }

            Example 2:
            User: "I need help but I'm not sure what with"
            Current purpose: Understand user needs

            Response:
            {
              "extracted_data": {},
              "message": "I'm here to help! Could you tell me a bit more about what's on your mind?"
            }

            Example 3:
            User: "Can you change my email to john@example.com?"
            Current purpose: Handle account updates

            Response:
            {
              "extracted_data": {
                "email": "john@example.com",
                "request_type": "email_update"
              },
              "message": "I can help you update your email address. Let me process that change for you."
            }
            """).strip()

        safe_examples = self._escape_cdata(examples)
        return [
            "<examples><![CDATA[",
            safe_examples,
            "]]></examples>",
            ""
        ]

    def _build_content_guidelines_section(self) -> List[str]:
        """Build detailed guidelines section for content generation."""
        guidelines = textwrap.dedent("""
            Content Generation Guidelines:
            - Maintain a natural, conversational tone appropriate to the context
            - Extract all relevant information the user provides, even if unexpected
            - Ask for clarification when user input is ambiguous or incomplete
            - Reference previous context when it helps maintain conversation flow
            - Be helpful and responsive to user needs and requests
            - Keep responses concise but complete
            - Don't mention technical system details or internal states to users
            """).strip()

        return [
            "<guidelines>",
            guidelines,
            "</guidelines>",
            ""
        ]

    def _build_format_rules_section(self) -> List[str]:
        """Build format rules section for proper JSON output."""
        rules = textwrap.dedent("""
            Critical Format Rules:
            - Return ONLY valid JSON - no markdown code fences, no additional explanations
            - Do not add keys not specified in the schema
            - Ensure all values are properly quoted and formatted according to JSON standards  
            - Use double quotes for all strings, not single quotes
            - Do not include trailing commas in JSON objects or arrays
            - Escape special characters in strings (quotes, backslashes, newlines)
            """).strip()

        return [
            "<format_rules>",
            rules,
            "</format_rules>",
            ""
        ]


# ============================================================================
# TRANSITION DECISION PROMPT BUILDER  
# ============================================================================

@dataclass(frozen=True)
class TransitionPromptConfig(BasePromptConfig):
    """Configuration for transition decision prompts."""

    # Decision guidance
    include_context_summary: bool = True
    include_transition_descriptions: bool = True
    require_reasoning: bool = True

    # Security
    limit_transition_details: bool = True


class TransitionPromptBuilder(BasePromptBuilder):
    """
    Builds prompts for transition decision making.

    Enhanced with comprehensive prompting style while maintaining focus
    on transition selection without exposing unnecessary FSM details.
    """

    def __init__(self, config: TransitionPromptConfig = None):
        """Initialize transition prompt builder."""
        super().__init__(config or TransitionPromptConfig())

    def build_transition_prompt(
            self,
            current_state: str,
            available_transitions: List[TransitionOption],
            context: Dict[str, Any],
            user_message: str,
            extracted_data: Dict[str, Any] = None
    ) -> str:
        """
        Build comprehensive system prompt for transition decision.

        Args:
            current_state: Current state identifier
            available_transitions: Available transition options
            context: Relevant context for decision
            user_message: Original user message
            extracted_data: Data extracted from content generation phase

        Returns:
            System prompt for transition decision with rich structure
        """
        logger.debug(f"Building transition prompt for {len(available_transitions)} options")

        sections = []

        # Enhanced task definition
        sections.extend(self._build_enhanced_transition_task_section())

        # Decision wrapper
        sections.append("<transition_decision>")

        # Current situation (enhanced)
        sections.extend(self._build_enhanced_situation_section(current_state, user_message, extracted_data))

        # Available options (comprehensive)
        sections.extend(self._build_comprehensive_options_section(available_transitions))

        # Context summary
        if self.config.include_context_summary:
            sections.extend(self._build_enhanced_context_summary_section(context))

        # Response format (detailed)
        sections.extend(self._build_comprehensive_transition_response_format())

        # Decision guidelines (detailed)
        sections.extend(self._build_detailed_decision_guidelines())

        # Format rules
        sections.extend(self._build_transition_format_rules())

        sections.append("</transition_decision>")

        prompt = "\n".join(sections)
        logger.debug(f"Transition prompt built: {len(prompt)} characters")

        return prompt

    def _build_enhanced_transition_task_section(self) -> List[str]:
        """Build enhanced task definition for transition decision."""
        task_description = textwrap.dedent("""
            You are the decision-making component for a conversational AI system.
            Your role is to analyze the current conversation state and select the most
            appropriate next step based on:
            - Current conversation context and objectives
            - User's latest message and extracted information
            - Available transition options and their priorities
            - Overall conversation flow and user needs

            Choose the option that best serves the user's intent and maintains natural conversation flow.
            """).strip()

        return [
            "<task>",
            task_description,
            "</task>",
            ""
        ]

    def _build_enhanced_situation_section(
            self,
            current_state: str,
            user_message: str,
            extracted_data: Dict[str, Any] = None
    ) -> List[str]:
        """Build enhanced current situation context."""
        sections = [
            "<current_situation>",
            f"<current_step>{self._sanitize_text_for_prompt(current_state)}</current_step>",
            f"<user_message>{self._sanitize_text_for_prompt(user_message)}</user_message>"
        ]

        # Add extracted data if available
        if extracted_data:
            try:
                data_json = json.dumps(extracted_data, indent=1, separators=(",", ": "))
                safe_data_json = self._escape_cdata(data_json)
                sections.extend([
                    "<extracted_information><![CDATA[",
                    safe_data_json,
                    "]]></extracted_information>"
                ])
            except Exception as e:
                logger.warning(f"Failed to serialize extracted data: {e}")

        sections.extend([
            "</current_situation>",
            ""
        ])

        return sections

    def _build_comprehensive_options_section(self, transitions: List[TransitionOption]) -> List[str]:
        """Build comprehensive available options section."""
        sections = ["<available_options>"]

        # Sort by priority for consistent presentation
        sorted_transitions = sorted(transitions,
                                    key=lambda t: t.priority) if self.config.deterministic_output else transitions

        for i, transition in enumerate(sorted_transitions, 1):
            sections.append(f"<option id=\"{i}\">")
            sections.append(f"  <target>{self._sanitize_text_for_prompt(transition.target_state)}</target>")

            if self.config.include_transition_descriptions and transition.description:
                sections.append(f"  <when>{self._sanitize_text_for_prompt(transition.description)}</when>")

            sections.append(f"  <priority>{transition.priority}</priority>")
            sections.append("</option>")

        sections.extend([
            "</available_options>",
            ""
        ])

        return sections

    def _build_enhanced_context_summary_section(self, context: Dict[str, Any]) -> List[str]:
        """Build enhanced relevant context summary."""
        if not context:
            return []

        # Filter context for transition decisions (more permissive than content generation)
        filtered_context = self._filter_transition_context(context)

        if not filtered_context:
            return []

        try:
            context_json = json.dumps(filtered_context, indent=1, separators=(",", ": "))
            safe_context_json = self._escape_cdata(context_json)
            return [
                "<context_summary><![CDATA[",
                safe_context_json,
                "]]></context_summary>",
                ""
            ]
        except Exception as e:
            logger.warning(f"Failed to serialize transition context: {e}")
            return []

    def _build_comprehensive_transition_response_format(self) -> List[str]:
        """Build comprehensive response format for transition decisions."""
        json_schema = textwrap.dedent("""
            {
                "selected_transition": "target_state_name",
                "reasoning": "Brief explanation of why this transition was chosen"
            }""").strip()

        sections = [
            "<response_format>",
            "Your response must be valid JSON with the following structure:",
            json_schema,
            "",
            "Requirements:",
            "\t- `selected_transition` is REQUIRED and must exactly match one of the target values",
            "\t- Choose the transition that best fits the user's intent and conversation flow"
        ]

        if self.config.require_reasoning:
            sections.append("\t- `reasoning` is REQUIRED and should briefly explain your choice")

        sections.extend([
            "",
            "Important:",
            "\t- Return ONLY valid JSON - no markdown code fences, no additional text",
            "\t- The selected_transition value must match exactly (case-sensitive)",
            "</response_format>",
            ""
        ])

        return sections

    def _build_detailed_decision_guidelines(self) -> List[str]:
        """Build detailed decision-making guidelines."""
        guidelines = textwrap.dedent("""
            Decision-Making Guidelines:
            - Prioritize the user's explicit intent and stated needs
            - Consider the natural flow of the conversation
            - Lower priority numbers indicate higher importance when appropriate
            - Choose the most specific transition that applies to the situation
            - If multiple transitions seem appropriate, prefer the one with lower priority number
            - Consider the context and previously collected information
            - Default to staying in current state only if no other transition clearly applies
            - Focus on what best serves the user's goals and needs
            """).strip()

        return [
            "<guidelines>",
            guidelines,
            "</guidelines>",
            ""
        ]

    def _build_transition_format_rules(self) -> List[str]:
        """Build format rules for transition responses."""
        rules = textwrap.dedent("""
            Critical Format Rules:
            - Return ONLY valid JSON - no markdown, no explanations outside the JSON
            - Use exact target state names as provided in the options
            - Ensure proper JSON formatting with double quotes for strings
            - Do not include any additional fields beyond those specified
            - Keep reasoning concise but informative
            """).strip()

        return [
            "<format_rules>",
            rules,
            "</format_rules>",
            ""
        ]

    def _filter_transition_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Filter context data relevant for transition decisions."""
        # For transition decisions, include more context but filter sensitive system info
        filtered = {}

        system_keys = {
            '_conversation_id', '_timestamp', '_fsm_id', 'system_handlers',
            '__internal__', '_system_state'
        }

        for key, value in context.items():
            if key not in system_keys and not key.startswith('__'):
                # Still apply basic internal filtering
                if not any(key.startswith(prefix) for prefix in self.config.internal_key_prefixes):
                    filtered[key] = value

        return filtered


# ============================================================================
# CONVENIENCE FUNCTIONS AND FACTORY METHODS
# ============================================================================

def create_content_prompt_builder(
        max_history_messages: Optional[int] = None,
        max_token_budget: Optional[int] = None,
        include_examples: bool = False,
        include_context_data: bool = True,
        **kwargs
) -> ContentPromptBuilder:
    """
    Convenience function to create a ContentPromptBuilder with common configurations.

    Args:
        max_history_messages: Maximum number of history messages to include
        max_token_budget: Maximum token budget for history
        include_examples: Whether to include few-shot examples
        include_context_data: Whether to include context data
        **kwargs: Additional configuration options

    Returns:
        Configured ContentPromptBuilder instance
    """
    config_kwargs = {}

    if max_history_messages is not None:
        config_kwargs['max_history_messages'] = max_history_messages
    if max_token_budget is not None:
        config_kwargs['max_token_budget'] = max_token_budget

    config_kwargs['include_examples'] = include_examples
    config_kwargs['include_context_data'] = include_context_data
    config_kwargs.update(kwargs)

    config = ContentPromptConfig(**config_kwargs)
    return ContentPromptBuilder(config)


def create_transition_prompt_builder(
        include_context_summary: bool = True,
        require_reasoning: bool = True,
        **kwargs
) -> TransitionPromptBuilder:
    """
    Convenience function to create a TransitionPromptBuilder with common configurations.

    Args:
        include_context_summary: Whether to include context summary
        require_reasoning: Whether to require reasoning in responses
        **kwargs: Additional configuration options

    Returns:
        Configured TransitionPromptBuilder instance
    """
    config_kwargs = {
        'include_context_summary': include_context_summary,
        'require_reasoning': require_reasoning
    }
    config_kwargs.update(kwargs)

    config = TransitionPromptConfig(**config_kwargs)
    return TransitionPromptBuilder(config)


# ============================================================================
# BACKWARD COMPATIBILITY (Optional)
# ============================================================================

# For users who might expect a single PromptBuilder class, provide an adapter
class PromptBuilder:
    """
    Backward compatibility adapter that provides both content and transition building.

    This class maintains the interface of the original PromptBuilder while
    internally using the new 2-pass architecture.
    """

    def __init__(self, content_config: ContentPromptConfig = None, transition_config: TransitionPromptConfig = None):
        """Initialize with configurations for both builders."""
        self.content_builder = ContentPromptBuilder(content_config)
        self.transition_builder = TransitionPromptBuilder(transition_config)

    def build_content_prompt(self, instance: FSMInstance, state: State, fsm_definition: FSMDefinition) -> str:
        """Build content generation prompt."""
        return self.content_builder.build_content_prompt(instance, state, fsm_definition)

    def build_transition_prompt(self, current_state: str, available_transitions: List[TransitionOption],
                                context: Dict[str, Any], user_message: str,
                                extracted_data: Dict[str, Any] = None) -> str:
        """Build transition decision prompt."""
        return self.transition_builder.build_transition_prompt(
            current_state, available_transitions, context, user_message, extracted_data
        )