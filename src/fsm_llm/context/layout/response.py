"""The Pass-2 response-generation prompt builder.

Layer ``context`` (plan d454fd62, step 74): split out of the old ``fsm_llm.prompts``.
"""

import textwrap
from dataclasses import dataclass
from typing import Any

from fsm_llm.core.definitions import FSMDefinition, FSMInstance, State
from fsm_llm.logging import logger

from .base import BasePromptBuilder, BasePromptConfig


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
        user_message: str | None = "",
        plain_text_response: bool = False,
        context: dict[str, Any] | None = None,
        rejected_corrections: dict[str, Any] | None = None,
        extraction_failed: bool = False,
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
            user_message: Original user message. ``None`` is a step without a
                user message (``API.advance``): the prompt then asks for the
                state's output as its response instructions describe it, and
                for no acknowledgement of a message, of new information or of
                a state change. A string, ``""`` included (the greeting,
                ``converse("")``), builds the conversational prompt unchanged.
            plain_text_response: Ask for plain user-facing text instead of the
                ``{"message", "reasoning"}`` JSON envelope. The streaming path
                sets this so the yielded tokens and the stored history are the
                same plain text. Default ``False`` keeps the JSON envelope, so
                the synchronous prompt is unchanged.
            context: Optional context view to render in ``<current_context>``
                (already scoped by ``context_scope.read_keys``). ``None`` uses
                the full ``instance.context.data``.
            rejected_corrections: Values the user asked for that the extraction
                step refused to apply (D-032). ``None`` or empty adds nothing, so
                a turn without a rejection builds the same prompt as before.
            extraction_failed: The bulk extraction call of this turn raised
                (D-050); adds one plain line saying a restated or corrected
                value may not have been stored. ``False`` adds nothing.

        Returns:
            System prompt focused on response generation
        """
        logger.debug(f"Building response generation prompt for state: {state.id}")

        # Build comprehensive prompt sections
        sections = []

        # Task definition (enhanced for response generation)
        sections.extend(self._build_response_task_section(user_message=user_message))

        # Response generation wrapper
        sections.append("<response_generation>")

        # Persona (if available and enabled)
        if self.config.include_persona and self.config.enable_persona_guidance:
            sections.extend(self._build_persona_section(instance, fsm_definition))

        # Final state context
        sections.extend(
            self._build_final_state_context_section(
                state, transition_occurred, previous_state, user_message=user_message
            )
        )

        # User message context
        sections.extend(self._build_user_message_section(user_message or ""))

        # Extracted data context
        if self.config.include_extracted_data and extracted_data:
            sections.extend(self._build_extracted_data_section(extracted_data))

        # Refused corrections (empty on every benign turn: no bytes added)
        if rejected_corrections or extraction_failed:
            sections.extend(
                self._build_rejected_corrections_section(
                    rejected_corrections or {}, extraction_failed
                )
            )

        # Conversation history
        if self.config.include_conversation_history:
            sections.extend(self._build_enhanced_history_section(instance))

        # Current context data (scoped view when the pipeline supplies one)
        sections.extend(self._build_enhanced_context_section(instance, context))

        # Response format
        sections.extend(
            self._build_response_format_section(
                plain_text_response, user_message=user_message
            )
        )

        # Guidelines
        if self.config.enable_response_guidelines:
            sections.extend(
                self._build_response_guidelines_section(user_message=user_message)
            )

        sections.append("</response_generation>")

        prompt = "\n".join(sections)
        logger.debug(f"Response generation prompt built ({len(prompt)} characters)")

        return prompt

    def _build_response_task_section(
        self, *, user_message: str | None = ""
    ) -> list[str]:
        """Build enhanced task definition section for response generation.

        ``user_message=None`` (a step without a user message) gets the
        no-message wording; any string keeps the conversational one.
        """
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-035
        # A step without a user message has nobody to answer: its reply is
        # the state's output. Do NOT send it the conversational wording below
        # ("acknowledge new information from user input", "acknowledge this
        # transition", "ask follow-up questions"): live, the model then opened
        # final answers with thanks for a message nobody sent. Do NOT key this
        # on an empty string either: `""` is the greeting and `converse("")`,
        # whose prompts must stay byte-identical. See decisions.md D-035.
        if user_message is None:
            return self._build_task_section("""
            You are the Response Generation component of a state machine run.
            No user message was sent on this step. Your responsibility is to:
            - Write the output of the current state, as its <response_instructions> describe it,
            - Base it on the context given below,
            - Open the output with its content: nobody has just spoken, so do not greet, thank or reply to anyone, and do not comment on a state change,
            - Maintain the <persona> when one is given.
            """)
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
        self,
        state: State,
        transition_occurred: bool,
        previous_state: str | None = None,
        *,
        user_message: str | None = "",
    ) -> list[str]:
        """Build final state context section.

        ``user_message=None`` (a step without a user message) leaves out the
        ``<transition_info>`` line: there is nobody to acknowledge it to.
        """
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

        if transition_occurred and previous_state and user_message is not None:
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

        # DECISION plan-2026-09-19T175721-21cd7f8e/D-039
        # Filter with the SAME security filter as <current_context>: a declared
        # `password` field extracted this turn must not be echoed to Pass 2.
        # Do NOT "simplify" by passing `extracted_data` through unfiltered (it is
        # shown even outside `read_keys`, D-005, but never unfiltered), and keep
        # `default=str` on the dumps below: without it one datetime/Decimal value
        # raises and drops the whole section.
        data = self._filter_context_for_security(extracted_data)
        if not data:
            return []

        return self._json_cdata_block("extracted_data", data, "extracted data")

    def _build_rejected_corrections_section(
        self, rejected_corrections: dict[str, Any], extraction_failed: bool = False
    ) -> list[str]:
        """Build the ``<rejected_corrections>`` block.

        DECISION plan-2026-09-19T175721-21cd7f8e/D-032: when the user asked for
        a change the store refused (provenance rule, D-015), Pass 2 is told so
        and the stored value stands; otherwise the reply says "updated" about
        data that never changed (LV3-01). Do NOT emit it for values the user
        did not ask for (the pipeline only passes grounded ones), and keep the
        same security filter and CDATA escaping as ``<extracted_data>``.

        D-050: ``extraction_failed`` adds ONE plain line (before the block, if
        any) because the automatic re-extraction of the user's latest message
        raised and a restated value may not have been stored; it is worded
        "may not", not "was not", since a per-field call of the same turn can
        have succeeded.
        """
        notice = (
            [
                "The automatic re-extraction of the user's latest message failed, "
                "so a value the user restated or corrected may NOT have been "
                "stored; do not claim it was updated.",
                "",
            ]
            if extraction_failed
            else []
        )
        data = self._filter_context_for_security(rejected_corrections)
        if not data:
            return notice
        data_json = self._dump_json(
            data, "rejected corrections", indent=1, separators=(",", ": ")
        )
        if data_json is None:
            return notice
        return [
            *notice,
            "<rejected_corrections>",
            "The user asked for these values but they were NOT applied; the "
            "stored value stands. Tell the user the value was not changed and do "
            "not claim it was updated.",
            "<![CDATA[",
            self._escape_cdata(data_json),
            "]]>",
            "</rejected_corrections>",
            "",
        ]

    def _build_response_format_section(
        self, plain_text: bool = False, *, user_message: str | None = ""
    ) -> list[str]:
        """Build response format section.

        ``plain_text=True`` replaces the JSON envelope with a plain-text
        instruction (streaming path, see ``build_response_prompt``).
        ``user_message=None`` (a step without a user message) keeps both
        formats and drops their conversational notes.
        """
        if user_message is None:
            if plain_text:
                return [
                    "<response_format>",
                    "Respond with plain text only: the output itself.",
                    "",
                    "Important:",
                    "\t- Do NOT use JSON, curly braces, or markdown code fences",
                    "\t- Do NOT include your reasoning or any field names",
                    "</response_format>",
                    "",
                ]
            return self._build_response_format(
                json_schema="""
            {
                "message": "The output text",
                "reasoning": "Brief internal reasoning (optional)"
            }""",
                field_descriptions=[
                    "`message` is REQUIRED and contains the complete output text.",
                    "`reasoning` is OPTIONAL and explains your decisions (not part of the output).",
                ],
                notes=[
                    "Important:",
                    "Return ONLY valid JSON - no markdown code fences, no additional text",
                    "The output goes inside `message`; never reply with the bare text",
                ],
            )
        if plain_text:
            # DECISION plan-2026-09-19T175721-21cd7f8e/D-003: the streaming
            # path yields raw deltas and persists their concatenation, so a
            # prompt that demands the {"message","reasoning"} envelope leaks it
            # to the user and into history (live: it then taught the model to
            # answer in envelopes). Do NOT "fix" this by stripping the envelope
            # from the token stream (holds back tokens, breaks time-to-first-
            # token) or by unwrapping only at persist time (the user still sees
            # it); that unwrap is the recorded fallback if the model ignores
            # this instruction. The sync prompt below is deliberately unchanged.
            return [
                "<response_format>",
                "Respond with plain text only: the natural, user-facing reply.",
                "",
                "Important:",
                "\t- Do NOT use JSON, curly braces, or markdown code fences",
                "\t- Do NOT include your reasoning or any field names",
                "\t- Acknowledge new information when appropriate",
                "\t- Guide toward the current state's purpose when needed",
                "</response_format>",
                "",
            ]
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

    def _build_response_guidelines_section(
        self, *, user_message: str | None = ""
    ) -> list[str]:
        """Build detailed guidelines section for response generation.

        ``user_message=None`` (a step without a user message) gets guidelines
        with no acknowledgement or follow-up question.
        """
        if user_message is None:
            return self._build_guidelines("""
            Response Generation Guidelines:
            - Follow the <response_instructions>: they say what the output is.
            - The output opens with content: no greeting, no thanks, no praise and no remark about the step itself.
            - Maintain consistent persona based on the <persona>.
            - Don't mention technical system details or internal states.
            """)
        return self._build_guidelines("""
            Response Generation Guidelines:
            - Acknowledge new information the user has provided when it's significant.
            - Guide the conversation toward the current state's purpose when appropriate.
            - Ask follow-up questions if more information is needed.
            - Maintain consistent persona based on the <persona>.
            - Don't mention technical system details or internal states to users.
            """)
