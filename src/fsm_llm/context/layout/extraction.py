"""Pass-1 prompt builders: the data-extraction builder and the per-field extraction builder.

Layer ``context`` (plan d454fd62, step 74): split out of the old ``fsm_llm.prompts``.
"""

import json
import textwrap
from dataclasses import dataclass
from typing import Any

from fsm_llm.constants import FIELD_PROMPT_CONTEXT_LABEL
from fsm_llm.core.definitions import FieldExtractionConfig, FSMInstance

from .base import (
    BasePromptBuilder,
    BasePromptConfig,
    _cap_history_line,
    _history_messages,
)


@dataclass(frozen=True)
class DataExtractionPromptConfig(BasePromptConfig):
    """Configuration for the Pass-1 builder (the shared base settings only)."""


class DataExtractionPromptBuilder(BasePromptBuilder):
    """
    The Pass-1 builder slot of ``FSMManager``/``MessagePipeline``.

    It carries the shared ``BasePromptBuilder`` machinery (sanitizer, security
    filter, history and context sections); the pipeline builds its bulk
    extraction prompt inline and calls this builder for sanitization and,
    on a turn with no user message, for the context and history sections
    the extraction reads (``build_extraction_source_sections``).
    """

    # DECISION plan-2026-09-21T203800-8a03483a/D-041
    # `build_extraction_prompt` and `build_refinement_prompt` were deleted: no
    # code in any package called them, and their response format promised
    # `extra`/`additional_info_needed` keys that no grammar or parser reads
    # (audit D10). Do NOT restore a second Pass-1 prompt here; change the one
    # the pipeline actually sends. See decisions.md D-041.

    config: DataExtractionPromptConfig

    def __init__(self, config: DataExtractionPromptConfig | None = None):
        """Initialize data extraction prompt builder with configuration."""
        super().__init__(config or DataExtractionPromptConfig())

    def build_extraction_source_sections(
        self, instance: FSMInstance, context: dict[str, Any]
    ) -> list[str]:
        """Render what a bulk extraction with no user message reads from.

        Contract: ``context`` is the context view the state may read (the
        pipeline passes user-visible data after ``context_scope.read_keys``).
        Returns prompt lines: the ``<current_context>`` section (security
        filter, then the key cap, exactly as Pass 2 renders it) followed by
        the conversation summary and ``<conversation_history>`` section. An
        empty ``context`` yields no context section; it never falls back to
        the full ``instance.context.data``. Returns ``[]`` when there is
        neither context nor history. Never raises.
        """
        # DECISION plan-2026-09-30T062855-07ad3f8c/D-023
        # These are SECTIONS of the one bulk prompt the pipeline builds, not a
        # second Pass-1 prompt (8a03483a/D-041 above still holds). Do NOT
        # render the context here with a bare json.dumps or a new filter: the
        # shared section methods are the only path that applies
        # has_internal_prefix / is_forbidden_context_entry and the caps.
        return [
            *self._build_enhanced_context_section(instance, context),
            *self._build_enhanced_history_section(instance),
        ]


@dataclass(frozen=True)
class FieldExtractionPromptConfig(BasePromptConfig):
    """Configuration for field extraction prompts (``include_conversation_history``
    is the base setting)."""

    include_context_data: bool = True


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
        user_message: str | None,
        dynamic_context: dict[str, Any] | None = None,
    ) -> str:
        """Build a focused prompt for extracting a single field.

        Args:
            instance: Current FSM instance (for conversation history).
            field_config: Configuration for the field to extract.
            user_message: The user input to extract from, or ``None`` when
                the turn has no user message. With ``None`` the prompt asks
                for the value the field's instructions describe, worked out
                from the context and the recent conversation and composed by
                the model when the instructions ask for something written,
                decided or chosen; it carries no "User message:" line. A
                string, the empty one included, gives the message prompt.
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
        from datetime import date as _date

        # DECISION plan-2026-09-30T062855-07ad3f8c/D-026
        # With no user message the prompt must NOT say "Extract ... from the
        # context" with "null if the information is completely absent": a
        # small model then answers null for every field whose value has to be
        # written (a summary, a plan, a decision), because that text is not
        # literally in the context (measured live, 2 of 10 runs). The
        # no-message wording says the value comes from following the field's
        # instructions and is composed when they ask for that. Do NOT fold
        # the two wordings into one: the message path is pinned byte for
        # byte. See decisions.md D-026.
        no_message = user_message is None
        sections: list[str] = [
            (
                f"Determine the value of the field '{field_name}' ({field_type}). "
                "There is no user message: follow the instructions below, "
                "using the context and recent conversation given."
                if no_message
                else f"Extract the field '{field_name}' ({field_type}) "
                "from the user's message."
            ),
            f"Today's date: {_date.today().isoformat()}",
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
            # DECISION plan-2026-07-18T162030-a02151fe/D-025 [STALE]
            # This builder assembles its OWN context section rather than calling
            # `_build_enhanced_context_section`, so it does NOT inherit the
            # `max_context_keys` cap applied there — measured at 500/500 keys
            # landing uncapped. It is the worst site to leave unbounded: this
            # runs once PER FIELD. Cap it here explicitly; if this section is
            # ever folded back into the shared base method, delete this call
            # rather than leaving it to double-apply. Order matches D-020:
            # security filtering FIRST, so filtered-out secret keys cannot
            # consume budget slots and evict legitimate ones.
            filtered = self._filter_context_for_security(dynamic_context)
            filtered = self._limit_context_by_key_count(filtered)
            ctx_json = None
            if filtered:
                # Same guard as every sibling context section (`_dump_json`): a
                # value that cannot be dumped (mixed-type keys under sort_keys,
                # a huge int) drops the section with a WARNING, keeps the prompt.
                ctx_json = self._dump_json(
                    filtered,
                    "field-prompt context",
                    sort_keys=self.config.deterministic_output,
                )
            if ctx_json is not None:
                # DECISION plan-2026-07-21T045419-9925aa3a/D-010
                # Sanitize the dumped JSON before emitting. These values were
                # extracted on a PRIOR turn -- the LLM wrote them -- so a value
                # like `<extraction_focus>x</extraction_focus>` or `</task>`
                # arrives here as an unescaped structural token on THIS turn's
                # per-field prompt. Do NOT drop this call back to a bare
                # `json.dumps` "because it's just our own extracted data": the
                # sibling context sites already sanitize, and `_TAG_PATTERN`
                # only touches `<tag>`-like substrings, never JSON `{`/`[`/`"`
                # structure, so extraction quality is unaffected. See D-010.
                sections.append(
                    f"{FIELD_PROMPT_CONTEXT_LABEL} "
                    f"{self._sanitize_text_for_prompt(ctx_json)}"
                )

        # Conversation history (compact)
        # DECISION plan-2026-07-19T191147-4b664252/D-012 [STALE]
        # This block assembles its own history text, so it must apply the SAME
        # capping the base builders get from `_manage_conversation_history` --
        # it previously took `get_recent()` and emitted it whole, so a
        # token-budget config that trimmed `DataExtractionPromptBuilder` to one
        # turn left this builder emitting all five, and it runs once PER FIELD.
        # Do NOT drop the `_manage_conversation_history` call, and do NOT cap
        # only the `Assistant:` lines: an unbounded `User:` line was the larger
        # half of the leak. Truncate BEFORE managing so the budget accounts for
        # the text actually emitted. See decisions.md D-012.
        if self.config.include_conversation_history:
            summary, recent = instance.context.conversation.get_summary_and_recent(
                self.config.max_history_messages
            )
            summary_block = self._build_summary_block(summary)
            if summary_block:
                sections.append("")
                sections.append(summary_block)
            # DECISION plan-2026-07-19T191147-4b664252/D-019 [STALE]
            # Emit ONE single-key dict per (role, message) pair. Do NOT collapse
            # an entry's roles into a single dict keyed by "user"/"system": the
            # role mapping is many-to-one (`"user" if role == "user" else
            # "system"`), so an entry carrying both `assistant` and `system`
            # collided into one slot and one message was SILENTLY LOST -- the
            # pre-D-012 loop emitted both. In-tree writers only ever append
            # single-key entries (`Conversation.add_user_message` /
            # `add_system_message`), but `Conversation.exchanges` is a plain
            # `list[dict[str, str]]` field, so restored/injected history can
            # carry multi-key entries -- `FSMManager.seed_restored_conversation`
            # handles that shape. Splitting also makes the budget below count
            # MESSAGES rather than entries, which is what D-012 asked for.
            # See decisions.md D-019.
            # The pairs come from `_history_messages`, the one role rule.
            formatted: list[dict[str, str]] = []
            for role, message in _history_messages(recent):
                if not message:
                    continue
                # Sanitize BEFORE the per-line cap so the cap still bounds
                # the text actually emitted into the prompt.
                msg = _cap_history_line(self._sanitize_text_for_prompt(message))
                formatted.append({role: msg})

            managed = self._manage_conversation_history(formatted)
            if managed:
                sections.append("")
                sections.append("Recent conversation:")
                for exchange in managed:
                    for role, msg in exchange.items():
                        label = "User" if role == "user" else "Assistant"
                        sections.append(f"  {label}: {msg}")

        # User message
        if user_message is not None:
            sections.append("")
            sections.append(
                f"User message: {self._sanitize_text_for_prompt(user_message)}"
            )

        # Response format — concise since JSON schema is enforced
        examples = (
            "(e.g., 'next Saturday' for a date, 'around 7pm' for a time, "
            "'a few' for a number). "
        )
        value_hint = field_type if no_message else f"extracted {field_type}"
        sections.extend(
            [
                "",
                "Respond with JSON:",
                f'{{"field_name": "{field_name}", "value": <{value_hint} or null>, "confidence": <0.0-1.0>, "reasoning": "..."}}',
                "",
                (
                    "IMPORTANT: If the instructions ask you to write, decide "
                    "or choose something, compose the value yourself from "
                    "what is given above. A partial or relative value is fine "
                    + examples
                    + "Only set null if the instructions cannot be followed "
                    "from what is given."
                    if no_message
                    else "IMPORTANT: Extract the value even if partial or relative "
                    + examples
                    + "If the value is not in the current "
                    "message, check the recent conversation above. "
                    "Only set null if the information is completely absent."
                ),
            ]
        )

        return "\n".join(sections)
