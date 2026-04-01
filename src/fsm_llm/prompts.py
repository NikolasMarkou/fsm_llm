"""
Prompt Builders FSM-LLM Architecture.

This module provides specialized prompt builders that support:
1. Data extraction prompts (focused on understanding and extracting information)
2. Transition decision prompts (focused on choosing between transition options)
3. Response generation prompts (focused on generating user-facing messages)

The architecture prevents FSM structure leakage while maintaining rich prompting
capabilities and ensuring responses are generated with full context of the final state.

Key Features:
- Information isolation prevents FSM structure leakage
- Rich XML-like prompt structure with comprehensive sections
- Configurable token budget and history management
- Comprehensive text sanitization and security measures
- Few-shot examples and detailed guidelines
- CDATA protection for JSON data
"""

import html
import json
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from .constants import (
    COMPILED_FORBIDDEN_CONTEXT_PATTERNS,
    DEFAULT_MAX_HISTORY_SIZE,
    INTERNAL_KEY_PREFIXES,
)
from .definitions import (
    ClassificationSchema,
    FieldExtractionConfig,
    FSMDefinition,
    FSMInstance,
    State,
)

# --------------------------------------------------------------
# Local imports
# --------------------------------------------------------------
from .logging import logger

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
    """Base configuration shared between all prompt builders."""

    # History Management
    max_history_messages: int = DEFAULT_MAX_HISTORY_SIZE
    max_token_budget: int = 3000
    history_strategy: HistoryManagementStrategy = HistoryManagementStrategy.HYBRID

    # Token Estimation
    chars_per_token: float = 2.5  # Conservative estimate
    token_estimation_factor: float = 1.3  # Safety factor for overhead
    utf8_expansion_factor: float = 1.5  # UTF-8 multi-byte expansion
    cdata_overhead_tokens: int = 50  # Overhead for CDATA wrapping

    # Content inclusion
    include_conversation_history: bool = True
    include_persona: bool = True

    # Security
    filter_internal_context: bool = True
    internal_key_prefixes: list[str] = field(
        default_factory=lambda: list(INTERNAL_KEY_PREFIXES)
    )

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
    """Base class with shared functionality for all prompt builders."""

    # Critical tags that could break the prompt structure — compiled once
    _CRITICAL_TAGS: ClassVar[list[str]] = [
        "task",
        "fsm",
        "data_extraction",
        "response_generation",
        "transition_decision",
        "current_state",
        "current_objective",
        "current_situation",
        "persona",
        "purpose",
        "instructions",
        "information_needed",
        "conversation_history",
        "current_context",
        "context_summary",
        "response_format",
        "examples",
        "guidelines",
        "format_rules",
        "transitions",
        "available_options",
        "option",
        "target",
        "when",
        "priority",
        "valid_states",
        "state",
        "information_to_extract",
        "extraction_focus",
        "final_state_context",
        "user_message",
        "original_input",
        "extracted_data",
        "extracted_information",
        "response_instructions",
        "information_still_needed",
        "extraction_instructions",
        "extraction_guidance",
        "collect",
        "current_step",
        "transition_info",
        "system",
        "instruction",
        "role",
        "message",
        "assistant",
        "human",
    ]
    _SANITIZE_PATTERN = re.compile(
        r"</?(?:" + "|".join(_CRITICAL_TAGS) + r")(?:[^>]*)?/?>", re.IGNORECASE
    )

    def __init__(self, config: BasePromptConfig | None = None):
        """Initialize with configuration."""
        self.config = config or BasePromptConfig()
        if self.config.verbose_logging:
            logger.debug(
                f"{self.__class__.__name__} initialized with config: {self.config}"
            )

    # ========================================================================
    # TEXT SANITIZATION AND SECURITY (Enhanced from old version)
    # ========================================================================

    def _sanitize_text_for_prompt(self, text: str | None) -> str:
        """
        Sanitize text to prevent XML tag confusion in prompts.
        Enhanced version with more comprehensive tag protection.
        Escapes XML-like tags and strips newlines to prevent prompt injection.
        """
        if text is None:
            return ""

        sanitized = self._SANITIZE_PATTERN.sub(lambda m: html.escape(m.group(0)), text)
        # Strip newlines to prevent prompt injection via field names
        return sanitized.replace("\n", " ").replace("\r", " ")

    def _escape_cdata(self, text: str) -> str:
        """Escape CDATA end sequences and XML closing tags to prevent breaking CDATA/XML blocks."""
        # Escape CDATA end sequences (including doubled variants like ]]]]>)
        text = text.replace("]]>", "]]&gt;")
        # Escape XML closing tags that could break our prompt structure
        text = text.replace("</", "&lt;/")
        return text

    # ========================================================================
    # HISTORY MANAGEMENT (From old version)
    # ========================================================================

    def _estimate_token_count(self, text: str, is_json: bool = False) -> int:
        """Estimate token count using conservative estimates."""
        char_count = len(text)
        adjusted_count = (
            char_count
            * self.config.utf8_expansion_factor
            * self.config.token_estimation_factor
        )
        return int(adjusted_count / self.config.chars_per_token)

    def _limit_history_by_message_count(
        self, exchanges: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Limit history by maximum number of messages."""
        if self.config.max_history_messages <= 0:
            return []
        if len(exchanges) <= self.config.max_history_messages:
            return exchanges

        result = exchanges[-self.config.max_history_messages :]
        if self.config.verbose_logging:
            logger.debug(
                f"Limited history by message count: {len(exchanges)} -> {len(result)} exchanges"
            )
        return result

    def _limit_history_by_token_budget(
        self, exchanges: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Limit history by estimated token budget."""
        if not exchanges:
            return exchanges

        available_tokens = max(
            self.config.max_token_budget - self.config.cdata_overhead_tokens, 1
        )
        result: list[dict[str, str]] = []
        current_tokens = 0

        for exchange in reversed(exchanges):
            exchange_json = json.dumps(exchange, separators=(",", ": "))
            exchange_tokens = self._estimate_token_count(exchange_json, is_json=True)

            if current_tokens + exchange_tokens > available_tokens and result:
                break

            result.insert(0, exchange)
            current_tokens += exchange_tokens

        if len(result) < len(exchanges) and self.config.verbose_logging:
            logger.debug(
                f"Limited history by token budget: {len(exchanges)} -> {len(result)} exchanges"
            )

        return result

    def _manage_conversation_history(
        self, exchanges: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Apply the configured history management strategy."""
        if not exchanges:
            return exchanges

        if self.config.history_strategy == HistoryManagementStrategy.MESSAGE_COUNT:
            return self._limit_history_by_message_count(exchanges)
        elif self.config.history_strategy == HistoryManagementStrategy.TOKEN_BUDGET:
            return self._limit_history_by_token_budget(exchanges)
        elif self.config.history_strategy == HistoryManagementStrategy.HYBRID:
            by_count = self._limit_history_by_message_count(exchanges)
            return self._limit_history_by_token_budget(by_count)
        else:
            logger.warning(f"Unknown history strategy: {self.config.history_strategy}")
            return exchanges

    def _filter_context_for_security(
        self, context_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Filter context data for security purposes.

        Removes internal keys (prefixed with _, system_, etc.) and keys
        matching forbidden security patterns (password, secret, token, api_key).
        """
        if not self.config.filter_internal_context:
            return context_data

        filtered = {}
        for key, value in context_data.items():
            # Skip internal-prefixed keys
            if any(
                key.startswith(prefix) for prefix in self.config.internal_key_prefixes
            ):
                continue
            # Skip keys matching forbidden security patterns
            if any(p.match(key) for p in COMPILED_FORBIDDEN_CONTEXT_PATTERNS):
                continue
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
            "payment_method": "payment information",
        }
        return key_mappings.get(key, key.replace("_", " "))

    def _build_enhanced_context_section(self, instance: FSMInstance) -> list[str]:
        """Build enhanced context section with security filtering."""
        if not instance.context.data:
            return []

        # Filter context for security and extraction
        user_context = self._filter_context_for_security(instance.context.data)

        if not user_context:
            return []

        try:
            context_json = json.dumps(
                user_context, indent=1, separators=(",", ": "), default=str
            )
            safe_context_json = self._escape_cdata(context_json)
            return [
                "<current_context><![CDATA[",
                safe_context_json,
                "]]></current_context>",
                "",
            ]
        except (TypeError, ValueError, OverflowError) as e:
            logger.warning(f"Failed to serialize context: {e}")
            return []

    # ========================================================================
    # SHARED SECTION FACTORIES (used by all prompt builders)
    # ========================================================================

    @staticmethod
    def _build_task_section(task_description: str) -> list[str]:
        """Build a ``<task>`` section wrapping the given description."""
        return ["<task>", textwrap.dedent(task_description).strip(), "</task>", ""]

    @staticmethod
    def _build_response_format(
        json_schema: str,
        field_descriptions: list[str],
        notes: list[str] | None = None,
        field_heading: str = "Where:",
    ) -> list[str]:
        """Build a ``<response_format>`` section with schema + field docs.

        Args:
            json_schema: The JSON schema example (will be dedented).
            field_descriptions: Lines describing each field (tab-prefixed).
            notes: Optional extra notes appended after the field descriptions.
            field_heading: Heading before field descriptions (default "Where:").
        """
        sections = [
            "<response_format>",
            "Your response must be valid JSON with the following structure:",
            textwrap.dedent(json_schema).strip(),
            "",
            field_heading,
        ]
        sections.extend(f"\t- {d}" for d in field_descriptions)
        if notes:
            sections.append("")
            heading = notes[0]
            sections.append(heading)
            sections.extend(f"\t- {n}" for n in notes[1:])
        sections.extend(["</response_format>", ""])
        return sections

    @staticmethod
    def _build_guidelines(guidelines_text: str) -> list[str]:
        """Build a ``<guidelines>`` section wrapping the given text."""
        return [
            "<guidelines>",
            textwrap.dedent(guidelines_text).strip(),
            "</guidelines>",
            "",
        ]

    @staticmethod
    def _build_format_rules(rules_text: str) -> list[str]:
        """Build a ``<format_rules>`` section wrapping the given text."""
        return [
            "<format_rules>",
            textwrap.dedent(rules_text).strip(),
            "</format_rules>",
            "",
        ]

    def _build_enhanced_history_section(self, instance: FSMInstance) -> list[str]:
        """Build enhanced conversation history section."""
        recent_exchanges = instance.context.conversation.get_recent(
            self.config.max_history_messages
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
                elif role_lower == "system":
                    safe_exchange["system"] = sanitized_text
                else:
                    logger.warning(
                        f"Unknown role '{role}' in conversation history, treating as system"
                    )
                    safe_exchange["system"] = sanitized_text
            formatted_exchanges.append(safe_exchange)

        # Apply history management strategy
        managed_exchanges = self._manage_conversation_history(formatted_exchanges)

        if managed_exchanges:
            try:
                history_json = json.dumps(
                    managed_exchanges, indent=1, separators=(",", ": "), default=str
                )
                safe_history_json = self._escape_cdata(history_json)
                return [
                    "<conversation_history><![CDATA[",
                    safe_history_json,
                    "]]></conversation_history>",
                    "",
                ]
            except (TypeError, ValueError, OverflowError) as e:
                logger.warning(f"Failed to serialize conversation history: {e}")

        return []


# ============================================================================
# DATA EXTRACTION PROMPT BUILDER (Pass 1)
# ============================================================================


@dataclass(frozen=True)
class DataExtractionPromptConfig(BasePromptConfig):
    """Configuration for data extraction prompts."""

    # Content inclusion
    include_context_data: bool = True
    include_state_instructions: bool = True

    # Prompt structure
    enable_detailed_guidelines: bool = True
    enable_format_rules: bool = True
    enable_extraction_guidance: bool = True


class DataExtractionPromptBuilder(BasePromptBuilder):
    """
    Builds prompts for data extraction focused on understanding user input.

    This builder creates prompts that focus purely on extracting and understanding
    information from user input without generating any user-facing responses.
    """

    config: DataExtractionPromptConfig

    def __init__(self, config: DataExtractionPromptConfig | None = None):
        """Initialize data extraction prompt builder with configuration."""
        super().__init__(config or DataExtractionPromptConfig())

    def build_extraction_prompt(
        self, instance: FSMInstance, state: State, fsm_definition: FSMDefinition
    ) -> str:
        """
        Build comprehensive system prompt for data extraction.

        Args:
            instance: FSM instance with context and history
            state: Current state definition
            fsm_definition: FSM definition for persona

        Returns:
            System prompt focused on data extraction
        """
        logger.debug(f"Building data extraction prompt for state: {state.id}")

        # Build comprehensive prompt sections
        sections = []

        # Task definition (enhanced for extraction)
        sections.extend(self._build_extraction_task_section())

        # Data extraction wrapper
        sections.append("<data_extraction>")

        # Current state context (enhanced for extraction)
        sections.extend(self._build_extraction_state_context_section(state))

        # Conversation history (with advanced management)
        if self.config.include_conversation_history:
            sections.extend(self._build_enhanced_history_section(instance))

        # Current context data (filtered and enhanced)
        if self.config.include_context_data:
            sections.extend(self._build_enhanced_context_section(instance))

        # Response format (comprehensive for extraction)
        sections.extend(self._build_extraction_response_format())

        # Guidelines (detailed for extraction)
        if self.config.enable_detailed_guidelines:
            sections.extend(self._build_extraction_guidelines_section())

        # Format rules
        if self.config.enable_format_rules:
            sections.extend(self._build_format_rules_section())

        sections.append("</data_extraction>")

        prompt = "\n".join(sections)
        logger.debug(f"Data Extraction prompt built ({len(prompt)} characters)")

        return prompt

    def _build_extraction_task_section(self) -> list[str]:
        """Build enhanced task definition section for data extraction."""
        return self._build_task_section("""
            You are the data extraction component.
            Instructions:
            - Analyze and understand user input thoroughly.
            - Extract relevant information and data from the user input.
            - Provide confidence ratings for extracted information.
            - The required values to extract from the context and the user input are in <information_to_extract>
            """)

    def _build_extraction_state_context_section(self, state: State) -> list[str]:
        """Build enhanced current state context section for extraction."""
        sections = [
            "<extraction_focus>",
            f"<purpose>{self._sanitize_text_for_prompt(state.purpose)}</purpose>",
        ]

        # Add state-specific extraction instructions
        if self.config.include_state_instructions and state.extraction_instructions:
            sections.extend(
                [
                    "<extraction_instructions>",
                    textwrap.dedent(
                        self._sanitize_text_for_prompt(state.extraction_instructions)
                    ).strip(),
                    "</extraction_instructions>",
                ]
            )

        # Add required information collection guidance
        if state.required_context_keys:
            sections.append("<information_to_extract>")
            for key in state.required_context_keys:
                sections.append(f"<collect>{key}</collect>")
            sections.append("</information_to_extract>")

            # Add detailed extraction instructions
            if self.config.enable_extraction_guidance:
                instructions = textwrap.dedent("""
                    Extraction Guidelines:
                    - Extract information explicitly mentioned by the user.
                    - If information is ambiguous, note the ambiguity in your reasoning and confidence.
                    - Don't make assumptions about information not provided.
                    - If the user provides extra relevant information, include it as well in the extra field.
                    - Rate your confidence in each piece of extracted information.
                    """).strip()

                sections.extend(
                    ["<extraction_guidance>", instructions, "</extraction_guidance>"]
                )

        sections.extend(["</extraction_focus>", ""])

        return sections

    def _build_extraction_response_format(self) -> list[str]:
        """Build comprehensive response format section for data extraction."""
        return self._build_response_format(
            json_schema="""
            {
                "extracted_data": {
                    "key1": "value1",
                    "key2": "value2",
                    "extra": {}
                },
                "confidence": 0.95,
                "reasoning": "Brief explanation of extraction decisions"
            }""",
            field_descriptions=[
                "`extracted_data` is REQUIRED, containing information extracted from user input.",
                "key names can be found in <information_to_extract>",
                "`extra` is for storing relevant information not explicitly requested.",
                "`confidence` is REQUIRED (0.0 to 1.0) representing your confidence in the extraction.",
                "`reasoning` is OPTIONAL, explaining your extraction decisions (not shown to user).",
            ],
            notes=[
                "Critical Points:",
                "Return ONLY valid JSON - no markdown code fences, no additional text",
                "Include empty object {} for extracted_data if no information was extracted",
                "Set `additional_info_needed` to true if more information is required from the user.",
                "Do NOT generate any other messages",
            ],
        )

    def _build_extraction_guidelines_section(self) -> list[str]:
        """Build detailed guidelines section for data extraction."""
        return self._build_guidelines("""
            Data Extraction Guidelines:
            - Focus on explicit information provided by the user.
            - Do not create or populate keys with no values or empty strings.
            - Extract implied information only when confidence is high.
            - Always provide confidence ratings for your extractions.
            - Don't make assumptions about missing information.
            - Extract all relevant information, even if unexpected.
            - Note ambiguities and unclear statements in reasoning.
            - Consider context from previous conversation exchanges.
            """)

    def _build_format_rules_section(self) -> list[str]:
        """Build format rules section for proper JSON output."""
        return self._build_format_rules("""
            Critical Format Rules:
            - Return ONLY valid JSON - no markdown code fences, no additional explanations.
            - Do not add keys not specified in the schema.
            - Do not create or populate keys with no values or empty strings.
            - Ensure all values are properly quoted and formatted according to JSON standards.
            - Use double quotes for all strings, not single quotes.
            - Do not include trailing commas in JSON objects or arrays.
            - Escape special characters in strings (quotes, backslashes, newlines).
            - Confidence must be a number between 0.0 and 1.0.
            """)

    def build_refinement_prompt(
        self,
        instance: FSMInstance,
        state: State,
        fsm_definition: FSMDefinition,
        previous_extraction: dict[str, Any],
        missing_keys: list[str],
    ) -> str:
        """
        Build a focused re-extraction prompt for a refinement pass.

        Used when the first extraction pass had low confidence or did not
        capture all ``required_context_keys``.  The refinement prompt
        tells the LLM what was already found and asks it to focus on the
        missing fields.

        Args:
            instance: FSM instance with context and history
            state: Current state definition
            fsm_definition: FSM definition for persona
            previous_extraction: Data already extracted in the first pass
            missing_keys: Context keys still missing after the first pass

        Returns:
            System prompt focused on extracting the missing data
        """
        logger.debug(
            f"Building refinement prompt for state {state.id}: missing={missing_keys}"
        )

        sections: list[str] = []

        # Task definition — emphasise refinement
        sections.extend(
            self._build_task_section(
                """
            You are the data extraction component performing a REFINEMENT pass.
            A previous extraction already captured some information.  Your job
            is to re-examine the user input and extract ONLY the fields that
            are still missing.
            """
            )
        )

        sections.append("<data_extraction_refinement>")

        # What was already found
        sections.append("<previously_extracted>")
        if previous_extraction:
            for key, value in previous_extraction.items():
                sections.append(
                    f'  <found key="{self._sanitize_text_for_prompt(str(key))}">'
                    f"{self._sanitize_text_for_prompt(str(value))}</found>"
                )
        else:
            sections.append("  (nothing was extracted in the first pass)")
        sections.append("</previously_extracted>")

        # What is still missing
        sections.append("<still_missing>")
        for key in missing_keys:
            sections.append(
                f"  <collect>{self._sanitize_text_for_prompt(key)}</collect>"
            )
        sections.append("</still_missing>")

        # State context — purpose and extraction instructions
        sections.append("<extraction_focus>")
        sections.append(
            f"<purpose>{self._sanitize_text_for_prompt(state.purpose)}</purpose>"
        )
        if state.extraction_instructions:
            sections.extend(
                [
                    "<extraction_instructions>",
                    textwrap.dedent(
                        self._sanitize_text_for_prompt(state.extraction_instructions)
                    ).strip(),
                    "</extraction_instructions>",
                ]
            )
        sections.append("</extraction_focus>")

        # Conversation history (limited)
        if self.config.include_conversation_history:
            sections.extend(self._build_enhanced_history_section(instance))

        # Response format
        sections.extend(self._build_extraction_response_format())

        # Focused guidelines
        sections.extend(
            self._build_guidelines(
                """
            Refinement Guidelines:
            - Focus ONLY on the fields listed in <still_missing>.
            - Do NOT re-extract fields already listed in <previously_extracted>.
            - Re-read the full user message carefully for any overlooked details.
            - If a missing field genuinely cannot be found, omit it from extracted_data.
            - Set confidence based on how certain you are about the NEW extractions.
            """
            )
        )

        sections.append("</data_extraction_refinement>")

        prompt = "\n".join(sections)
        logger.debug(f"Refinement prompt built ({len(prompt)} characters)")
        return prompt


# ============================================================================
# RESPONSE GENERATION PROMPT BUILDER (Pass 2)
# ============================================================================


@dataclass(frozen=True)
class ResponsePromptConfig(BasePromptConfig):
    """Configuration for response generation prompts."""

    # Content inclusion
    include_extracted_data: bool = True

    # Response customization
    enable_response_guidelines: bool = True
    enable_persona_guidance: bool = True


class ResponseGenerationPromptBuilder(BasePromptBuilder):
    """
    Builds prompts for generating user-facing responses after state transitions.

    This builder creates prompts that focus on generating appropriate responses
    based on the final state context and extracted user data.
    """

    config: ResponsePromptConfig

    def __init__(self, config: ResponsePromptConfig | None = None):
        """Initialize response generation prompt builder."""
        super().__init__(config or ResponsePromptConfig())

    def build_response_prompt(
        self,
        instance: FSMInstance,
        state: State,
        fsm_definition: FSMDefinition,
        extracted_data: dict[str, Any] | None = None,
        transition_occurred: bool = False,
        previous_state: str | None = None,
        user_message: str = "",
    ) -> str:
        """
        Build comprehensive system prompt for response generation.

        Args:
            instance: FSM instance with context and history
            state: Current (final) state definition
            fsm_definition: FSM definition for persona
            extracted_data: Data extracted in Pass 1
            transition_occurred: Whether a state transition occurred
            previous_state: Previous state if transition occurred
            user_message: Original user message

        Returns:
            System prompt focused on response generation
        """
        logger.debug(f"Building response generation prompt for state: {state.id}")

        # Build comprehensive prompt sections
        sections = []

        # Task definition (enhanced for response generation)
        sections.extend(self._build_response_task_section())

        # Response generation wrapper
        sections.append("<response_generation>")

        # Persona (if available and enabled)
        if self.config.include_persona and self.config.enable_persona_guidance:
            sections.extend(self._build_persona_section(instance, fsm_definition))

        # Final state context
        sections.extend(
            self._build_final_state_context_section(
                state, transition_occurred, previous_state
            )
        )

        # User message context
        sections.extend(self._build_user_message_section(user_message))

        # Extracted data context
        if self.config.include_extracted_data and extracted_data:
            sections.extend(self._build_extracted_data_section(extracted_data))

        # Conversation history
        if self.config.include_conversation_history:
            sections.extend(self._build_enhanced_history_section(instance))

        # Current context data
        sections.extend(self._build_enhanced_context_section(instance))

        # Response format
        sections.extend(self._build_response_format_section())

        # Guidelines
        if self.config.enable_response_guidelines:
            sections.extend(self._build_response_guidelines_section())

        sections.append("</response_generation>")

        prompt = "\n".join(sections)
        logger.debug(f"Response generation prompt built ({len(prompt)} characters)")

        return prompt

    def _build_response_task_section(self) -> list[str]:
        """Build enhanced task definition section for response generation."""
        return self._build_task_section("""
            You are the Response Generation component in a conversational AI system.
            Your responsibility is to:
            - Generate appropriate user-facing responses based on the <persona>,
            - Respond based on the current conversation state and context,
            - Acknowledge any new information that was extracted from user input,
            - Guide the conversation naturally toward the current state's purpose.
            - Maintain consistent persona and conversational flow.
            """)

    def _build_persona_section(
        self, instance: FSMInstance, fsm_definition: FSMDefinition
    ) -> list[str]:
        """Build persona section for consistent response generation."""
        persona = instance.persona or fsm_definition.persona
        if not persona:
            return []

        return ["<persona>", self._sanitize_text_for_prompt(persona), "</persona>", ""]

    def _build_final_state_context_section(
        self, state: State, transition_occurred: bool, previous_state: str | None = None
    ) -> list[str]:
        """Build final state context section."""
        sections = [
            "<final_state_context>",
            f"<current_state>{state.id}</current_state>",
            f"<purpose>{self._sanitize_text_for_prompt(state.purpose)}</purpose>",
        ]

        # Add response-specific instructions
        if state.response_instructions:
            sections.extend(
                [
                    "<response_instructions>",
                    textwrap.dedent(
                        self._sanitize_text_for_prompt(state.response_instructions)
                    ).strip(),
                    "</response_instructions>",
                ]
            )

        # Add information still needed
        if state.required_context_keys:
            sections.append("<information_still_needed>")
            sections.append(
                "If any of these are still needed, naturally work toward collecting them:"
            )
            for key in state.required_context_keys:
                natural_key = self._humanize_key(key)
                sections.append(f"- {natural_key}")
            sections.append("</information_still_needed>")

        if transition_occurred and previous_state:
            sections.append(
                f"<transition_info>Just transitioned from '{previous_state}' to '{state.id}'. "
                f"Acknowledge this transition naturally.</transition_info>"
            )

        sections.extend(["</final_state_context>", ""])

        return sections

    def _build_user_message_section(self, user_message: str) -> list[str]:
        """Build user message context section."""
        if not user_message:
            return []

        return [
            "<user_message>",
            f"<original_input>{self._sanitize_text_for_prompt(user_message)}</original_input>",
            "</user_message>",
            "",
        ]

    def _build_extracted_data_section(
        self, extracted_data: dict[str, Any]
    ) -> list[str]:
        """Build extracted data context section."""
        if not extracted_data:
            return []

        try:
            data_json = json.dumps(extracted_data, indent=1, separators=(",", ": "))
            safe_data_json = self._escape_cdata(data_json)
            return [
                "<extracted_data><![CDATA[",
                safe_data_json,
                "]]></extracted_data>",
                "",
            ]
        except (TypeError, ValueError, OverflowError) as e:
            logger.warning(f"Failed to serialize extracted data: {e}")
            return []

    def _build_response_format_section(self) -> list[str]:
        """Build response format section."""
        return self._build_response_format(
            json_schema="""
            {
                "message": "Your natural response to the user",
                "reasoning": "Brief internal reasoning (optional)"
            }""",
            field_descriptions=[
                "`message` is REQUIRED and contains the natural, user-facing response text.",
                "`reasoning` is OPTIONAL and explains your response decisions (not shown to user).",
            ],
            notes=[
                "Important:",
                "Return ONLY valid JSON - no markdown code fences, no additional text",
                "Acknowledge new information when appropriate",
                "Guide toward the current state's purpose when needed",
            ],
        )

    def _build_response_guidelines_section(self) -> list[str]:
        """Build detailed guidelines section for response generation."""
        return self._build_guidelines("""
            Response Generation Guidelines:
            - Acknowledge new information the user has provided when it's significant.
            - Guide the conversation toward the current state's purpose when appropriate.
            - Ask follow-up questions if more information is needed.
            - Maintain consistent persona based on the <persona>.
            - Don't mention technical system details or internal states to users.
            """)


# ============================================================================
# CLASSIFICATION PROMPT BUILDERS
# ============================================================================


@dataclass(frozen=True)
class ClassificationPromptConfig:
    """Controls classification prompt generation behavior."""

    include_reasoning: bool = True
    max_tokens: int = 512
    temperature: float = 0.0
    include_entities: bool = True
    multi_intent: bool = False
    max_intents: int = 3


def build_classification_json_schema(
    schema: ClassificationSchema,
    *,
    multi_intent: bool = False,
    max_intents: int = 3,
    include_reasoning: bool = True,
    include_entities: bool = True,
) -> dict:
    """
    Build a JSON Schema dict from a ClassificationSchema.

    Ordering: reasoning (CoT) precedes intent to mitigate constrained-decoding
    probability distortion.
    """
    intent_enum = schema.intent_names

    if multi_intent:
        intent_score_props: dict = {
            "intent": {
                "type": "string",
                "enum": intent_enum,
                "description": "Classified intent name",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Model confidence in this classification",
            },
        }
        intent_score_required = ["intent", "confidence"]
        if include_entities:
            intent_score_props["entities"] = {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Extracted entities relevant to this intent",
            }
            intent_score_required.append("entities")

        properties: dict = {}
        required: list[str] = []

        if include_reasoning:
            properties["reasoning"] = {
                "type": "string",
                "description": "Chain-of-thought explanation before classification",
            }
            required.append("reasoning")

        properties["intents"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": intent_score_props,
                "required": intent_score_required,
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": max_intents,
            "description": "Ranked list of detected intents, most probable first",
        }
        required.append("intents")

    else:
        properties = {}
        required = []

        if include_reasoning:
            properties["reasoning"] = {
                "type": "string",
                "description": "Chain-of-thought explanation before classification",
            }
            required.append("reasoning")

        properties["intent"] = {
            "type": "string",
            "enum": intent_enum,
            "description": "The primary classified intent of the user input",
        }
        required.append("intent")

        properties["confidence"] = {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Model confidence in this classification",
        }
        required.append("confidence")

        if include_entities:
            properties["entities"] = {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Extracted entities relevant to the intent",
            }
            required.append("entities")

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_classification_system_prompt(
    schema: ClassificationSchema,
    config: ClassificationPromptConfig | None = None,
) -> str:
    """
    Build the system prompt for an intent classification task.

    Includes intent definitions and behavioral rules. The JSON schema
    is embedded so prompt-only approaches also get structure guidance.
    """
    if config is None:
        config = ClassificationPromptConfig()

    intent_lines = []
    for intent in schema.intents:
        intent_lines.append(f"- {intent.name}: {intent.description}")
    intent_block = "\n".join(intent_lines)

    json_schema = build_classification_json_schema(
        schema,
        multi_intent=config.multi_intent,
        max_intents=config.max_intents,
        include_reasoning=config.include_reasoning,
        include_entities=config.include_entities,
    )
    schema_str = json.dumps(json_schema, indent=2)

    rules = [
        "1. Output ONLY valid JSON matching the schema below.",
        "2. Set confidence between 0.0 and 1.0 based on how clear the intent is.",
    ]
    next_rule = 3
    if config.include_entities:
        rules.append(
            f"{next_rule}. Extract any relevant entities (e.g., order IDs, product names, amounts)."
        )
        next_rule += 1
    rules.append(
        f"{next_rule}. "
        f'Use "{schema.fallback_intent}" when intent is ambiguous or does not fit other categories.'
    )
    next_rule += 1
    if config.include_reasoning:
        rules.append(f"{next_rule}. Write your reasoning before selecting the intent.")
        next_rule += 1
    if config.multi_intent:
        rules.append(
            f"{next_rule}. "
            "If the message contains multiple intents, return them ranked by confidence."
        )

    rules_block = "\n".join(rules)

    if config.multi_intent:
        classify_instruction = (
            "classify it into one or more of the following intents, "
            "ranked by confidence"
        )
    else:
        classify_instruction = "classify it into exactly one of the following intents"

    return (
        f"You are an intent classification engine. Analyze the user's message and "
        f"{classify_instruction}:\n\n"
        f"{intent_block}\n\n"
        f"Rules:\n{rules_block}\n\n"
        f"Output JSON Schema:\n```json\n{schema_str}\n```"
    )


# ============================================================================
# FIELD EXTRACTION PROMPT BUILDER
# ============================================================================


@dataclass(frozen=True)
class FieldExtractionPromptConfig(BasePromptConfig):
    """Configuration for field extraction prompts."""

    include_context_data: bool = True
    include_conversation_history: bool = True


class FieldExtractionPromptBuilder(BasePromptBuilder):
    """Builds focused prompts for targeted single-field extraction.

    Each prompt asks the LLM to extract ONE specific field, given custom
    instructions and dynamic context.  The response schema is minimal:
    ``{"field_name": ..., "value": ..., "confidence": ..., "reasoning": ...}``.
    """

    config: FieldExtractionPromptConfig

    def __init__(self, config: FieldExtractionPromptConfig | None = None):
        super().__init__(config or FieldExtractionPromptConfig())

    def build_field_extraction_prompt(
        self,
        instance: FSMInstance,
        field_config: FieldExtractionConfig,
        user_message: str,
        dynamic_context: dict[str, Any] | None = None,
    ) -> str:
        """Build a focused prompt for extracting a single field.

        Args:
            instance: Current FSM instance (for conversation history).
            field_config: Configuration for the field to extract.
            user_message: The user input to extract from.
            dynamic_context: Subset of context keys relevant to this field.

        Returns:
            Complete system prompt string.
        """
        field_name = self._sanitize_text_for_prompt(field_config.field_name)
        field_type = self._sanitize_text_for_prompt(field_config.field_type)
        instructions = textwrap.dedent(
            self._sanitize_text_for_prompt(field_config.extraction_instructions)
        ).strip()

        # Build a concise, direct prompt optimized for small models.
        # Ollama JSON schema enforcement handles output formatting,
        # so we focus on WHAT to extract rather than HOW to format.
        sections: list[str] = [
            f"Extract the field '{field_name}' ({field_type}) from the user's message.",
            "",
            f"Instructions: {instructions}",
        ]

        # Validation rules (keep concise)
        if field_config.validation_rules:
            rules_json = json.dumps(
                field_config.validation_rules,
                sort_keys=self.config.deterministic_output,
            )
            sections.append(f"Validation: {rules_json}")

        # Dynamic context (previously extracted fields)
        if self.config.include_context_data and dynamic_context:
            filtered = self._filter_context_for_security(dynamic_context)
            if filtered:
                ctx_json = json.dumps(
                    filtered,
                    sort_keys=self.config.deterministic_output,
                    default=str,
                )
                sections.append(f"Already extracted: {ctx_json}")

        # Conversation history (compact)
        if self.config.include_conversation_history:
            recent = instance.context.conversation.get_recent(
                self.config.max_history_messages
            )
            if recent:
                sections.append("")
                sections.append("Recent conversation:")
                for entry in recent:
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                    if role == "user" and content:
                        sections.append(f"  User: {content}")
                    elif role in ("assistant", "system") and content:
                        msg = content
                        if len(msg) > 150:
                            msg = msg[:150] + "..."
                        sections.append(f"  Assistant: {msg}")

        # User message
        sections.append("")
        sections.append(f"User message: {user_message}")

        # Response format — concise since JSON schema is enforced
        sections.extend(
            [
                "",
                "Respond with JSON:",
                f'{{"field_name": "{field_name}", "value": <extracted {field_type} or null>, "confidence": <0.0-1.0>, "reasoning": "..."}}',
                "",
                f"Set value to null if '{field_name}' is not present in the user's message.",
            ]
        )

        return "\n".join(sections)
