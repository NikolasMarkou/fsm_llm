"""
Unit tests for the FSM-LLM prompt builder system.

Tests BasePromptConfig validation, text sanitization, token estimation,
and the three prompt builders: DataExtraction, ResponseGeneration, Transition.
"""

from typing import ClassVar

import pytest

from fsm_llm.constants import MAX_CONTEXT_FILTER_DEPTH
from fsm_llm.definitions import (
    FieldExtractionConfig,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    State,
    Transition,
)
from fsm_llm.prompts import (
    BasePromptBuilder,
    BasePromptConfig,
    DataExtractionPromptBuilder,
    DataExtractionPromptConfig,
    FieldExtractionPromptBuilder,
    FieldExtractionPromptConfig,
    HistoryManagementStrategy,
    ResponseGenerationPromptBuilder,
)

# ============================================================================
# Helpers -- minimal domain objects for prompt builder tests
# ============================================================================


def _make_state(
    state_id: str = "greeting",
    purpose: str = "Greet the user",
    description: str = "Greeting state",
    required_context_keys=None,
    extraction_instructions=None,
    response_instructions=None,
    transitions=None,
):
    return State(
        id=state_id,
        description=description,
        purpose=purpose,
        required_context_keys=required_context_keys,
        extraction_instructions=extraction_instructions,
        response_instructions=response_instructions,
        transitions=transitions or [],
    )


def _make_instance(
    fsm_id: str = "test",
    current_state: str = "greeting",
    context_data=None,
    persona=None,
    exchanges=None,
):
    ctx = FSMContext(data=context_data or {})
    if exchanges:
        for ex in exchanges:
            for role, msg in ex.items():
                if role == "user":
                    ctx.conversation.add_user_message(msg)
                else:
                    ctx.conversation.add_system_message(msg)
    return FSMInstance(
        fsm_id=fsm_id,
        current_state=current_state,
        context=ctx,
        persona=persona,
    )


def _make_fsm_definition(persona=None):
    """Create a minimal valid FSMDefinition."""
    return FSMDefinition(
        name="test_fsm",
        description="Test FSM",
        initial_state="greeting",
        persona=persona,
        states={
            "greeting": State(
                id="greeting",
                description="Greeting",
                purpose="Greet the user",
                transitions=[
                    Transition(
                        target_state="end",
                        description="Finish",
                        priority=100,
                    )
                ],
            ),
            "end": State(
                id="end",
                description="End",
                purpose="End conversation",
                transitions=[],
            ),
        },
    )


# ============================================================================
# 1-3. BasePromptConfig -- defaults and validation
# ============================================================================


class TestBasePromptConfig:
    """Tests for BasePromptConfig default values and validation."""

    def test_default_values(self):
        cfg = BasePromptConfig()
        assert cfg.max_history_messages == 5
        assert cfg.max_token_budget == 3000
        assert cfg.chars_per_token == 2.5
        assert cfg.history_strategy == HistoryManagementStrategy.HYBRID
        assert cfg.include_conversation_history is True
        assert cfg.include_persona is True
        assert cfg.filter_internal_context is True
        assert cfg.deterministic_output is True

    def test_max_history_messages_non_negative(self):
        # zero is fine
        BasePromptConfig(max_history_messages=0)
        with pytest.raises(
            ValueError, match="max_history_messages must be non-negative"
        ):
            BasePromptConfig(max_history_messages=-1)

    def test_max_token_budget_minimum(self):
        BasePromptConfig(max_token_budget=100)
        with pytest.raises(ValueError, match="max_token_budget must be at least 100"):
            BasePromptConfig(max_token_budget=99)

    def test_chars_per_token_positive(self):
        BasePromptConfig(chars_per_token=0.1)
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            BasePromptConfig(chars_per_token=0)
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            BasePromptConfig(chars_per_token=-1.0)

    def test_frozen_immutability(self):
        cfg = BasePromptConfig()
        with pytest.raises(AttributeError):
            cfg.max_history_messages = 10


# ============================================================================
# 4. BasePromptBuilder._sanitize_text_for_prompt
# ============================================================================


class TestSanitizeText:
    """Tests for XML tag sanitization in prompt text."""

    def setup_method(self):
        self.builder = BasePromptBuilder()

    def test_removes_critical_xml_tags(self):
        text = "<task>do something</task>"
        result = self.builder._sanitize_text_for_prompt(text)
        assert "<task>" not in result
        assert "</task>" not in result
        # Content should survive
        assert "do something" in result

    def test_escapes_self_closing_tags(self):
        text = "<persona />"
        result = self.builder._sanitize_text_for_prompt(text)
        assert "<persona" not in result or "&lt;" in result

    def test_escapes_tags_with_attributes(self):
        text = '<option id="1">foo</option>'
        result = self.builder._sanitize_text_for_prompt(text)
        assert "<option" not in result or "&lt;" in result

    def test_handles_none_input(self):
        assert self.builder._sanitize_text_for_prompt(None) == ""

    def test_preserves_non_critical_tags(self):
        text = "<b>bold</b> <i>italic</i>"
        result = self.builder._sanitize_text_for_prompt(text)
        assert "<b>bold</b>" in result
        assert "<i>italic</i>" in result

    def test_case_insensitive_tag_removal(self):
        text = "<TASK>upper</TASK> <Task>mixed</Task>"
        result = self.builder._sanitize_text_for_prompt(text)
        assert "<TASK>" not in result
        assert "<Task>" not in result

    def test_handles_special_characters_in_text(self):
        text = "User said: 'hello & goodbye' with <persona> inside"
        result = self.builder._sanitize_text_for_prompt(text)
        # The ampersand in plain text should survive (only tags are escaped)
        assert "hello & goodbye" in result
        # The critical tag should be escaped
        assert "<persona>" not in result


# ============================================================================
# 5. BasePromptBuilder._estimate_token_count
# ============================================================================


class TestEstimateTokenCount:
    """Tests for token estimation logic."""

    def test_basic_calculation(self):
        builder = BasePromptBuilder()
        cfg = builder.config
        text = "hello"
        expected = int(
            len(text)
            * cfg.utf8_expansion_factor
            * cfg.token_estimation_factor
            / cfg.chars_per_token
        )
        assert builder._estimate_token_count(text) == expected

    def test_empty_string_returns_zero(self):
        builder = BasePromptBuilder()
        assert builder._estimate_token_count("") == 0

    def test_respects_custom_chars_per_token(self):
        cfg = BasePromptConfig(chars_per_token=5.0)
        builder = BasePromptBuilder(config=cfg)
        count_wide = builder._estimate_token_count("a" * 100)
        # With higher chars_per_token, token count should be lower
        default_builder = BasePromptBuilder()
        count_default = default_builder._estimate_token_count("a" * 100)
        assert count_wide < count_default


# ============================================================================
# 6-8. DataExtractionPromptBuilder
# ============================================================================


class TestDataExtractionPromptBuilder:
    """Tests for data extraction prompt building."""

    def setup_method(self):
        self.builder = DataExtractionPromptBuilder()
        self.fsm_def = _make_fsm_definition()

    def test_builds_extraction_prompt_with_state_context(self):
        state = _make_state(purpose="Collect user email")
        instance = _make_instance()
        prompt = self.builder.build_extraction_prompt(instance, state, self.fsm_def)

        assert "<data_extraction>" in prompt
        assert "</data_extraction>" in prompt
        assert "<task>" in prompt
        assert "data extraction component" in prompt.lower()
        assert "Collect user email" in prompt

    def test_includes_required_context_keys(self):
        state = _make_state(
            purpose="Collect info",
            required_context_keys=["email", "phone"],
        )
        instance = _make_instance()
        prompt = self.builder.build_extraction_prompt(instance, state, self.fsm_def)

        assert "<information_to_extract>" in prompt
        assert "<collect>email</collect>" in prompt
        assert "<collect>phone</collect>" in prompt

    def test_handles_empty_required_context_keys(self):
        state = _make_state(purpose="Greet user", required_context_keys=None)
        instance = _make_instance()
        prompt = self.builder.build_extraction_prompt(instance, state, self.fsm_def)

        # No <collect> items should be present when no keys are required
        assert "<collect>" not in prompt
        # Should still produce a valid prompt
        assert "<data_extraction>" in prompt

    def test_includes_extraction_instructions(self):
        state = _make_state(
            purpose="Collect data",
            extraction_instructions="Focus on extracting the user's preferred language.",
        )
        instance = _make_instance()
        prompt = self.builder.build_extraction_prompt(instance, state, self.fsm_def)

        assert "<extraction_instructions>" in prompt
        assert "preferred language" in prompt

    def test_handles_missing_extraction_instructions(self):
        state = _make_state(purpose="Greet", extraction_instructions=None)
        instance = _make_instance()
        prompt = self.builder.build_extraction_prompt(instance, state, self.fsm_def)
        assert "<extraction_instructions>" not in prompt

    def test_includes_conversation_history(self):
        instance = _make_instance(
            exchanges=[
                {"user": "Hi there"},
                {"system": "Hello! How can I help?"},
            ]
        )
        state = _make_state()
        prompt = self.builder.build_extraction_prompt(instance, state, self.fsm_def)
        assert "<conversation_history>" in prompt
        assert "Hi there" in prompt

    def test_includes_context_data(self):
        instance = _make_instance(context_data={"name": "Alice"})
        state = _make_state()
        prompt = self.builder.build_extraction_prompt(instance, state, self.fsm_def)

        assert "<current_context>" in prompt
        assert "Alice" in prompt

    def test_filters_internal_context_keys(self):
        instance = _make_instance(
            context_data={"name": "Alice", "_secret": "hidden", "system_flag": True}
        )
        state = _make_state()
        prompt = self.builder.build_extraction_prompt(instance, state, self.fsm_def)

        assert "Alice" in prompt
        assert "_secret" not in prompt
        assert "system_flag" not in prompt

    def test_response_format_section_present(self):
        state = _make_state()
        instance = _make_instance()
        prompt = self.builder.build_extraction_prompt(instance, state, self.fsm_def)

        assert "<response_format>" in prompt
        assert "extracted_data" in prompt
        assert "confidence" in prompt

    def test_disabling_guidelines_and_format_rules(self):
        cfg = DataExtractionPromptConfig(
            enable_detailed_guidelines=False,
            enable_format_rules=False,
        )
        builder = DataExtractionPromptBuilder(config=cfg)
        state = _make_state()
        instance = _make_instance()
        prompt = builder.build_extraction_prompt(instance, state, self.fsm_def)

        assert "<guidelines>" not in prompt
        assert "<format_rules>" not in prompt


# ============================================================================
# 9-11. ResponseGenerationPromptBuilder
# ============================================================================


class TestResponseGenerationPromptBuilder:
    """Tests for response generation prompt building."""

    def setup_method(self):
        self.builder = ResponseGenerationPromptBuilder()
        self.fsm_def = _make_fsm_definition(persona="Friendly assistant named Bob")

    def test_builds_response_prompt_with_persona(self):
        state = _make_state(purpose="Say goodbye")
        instance = _make_instance(persona="Friendly assistant named Bob")
        prompt = self.builder.build_response_prompt(
            instance, state, self.fsm_def, user_message="bye"
        )

        assert "<response_generation>" in prompt
        assert "</response_generation>" in prompt
        assert "<persona>" in prompt
        assert "Bob" in prompt

    def test_includes_conversation_history(self):
        instance = _make_instance(
            exchanges=[
                {"user": "What is your name?"},
                {"system": "I am Bob."},
            ]
        )
        state = _make_state()
        prompt = self.builder.build_response_prompt(
            instance, state, self.fsm_def, user_message="thanks"
        )

        assert "<conversation_history>" in prompt
        assert "What is your name?" in prompt

    def test_handles_missing_persona(self):
        fsm_no_persona = _make_fsm_definition(persona=None)
        instance = _make_instance(persona=None)
        state = _make_state()
        prompt = self.builder.build_response_prompt(
            instance, state, fsm_no_persona, user_message="hello"
        )

        # Should still produce valid prompt without a dedicated persona section
        assert "<response_generation>" in prompt
        # The persona block (with content between opening/closing tags) should not appear
        # Note: "<persona>" as literal text may appear inside the task description,
        # so we check that no actual persona section with closing tag is rendered.
        assert "</persona>" not in prompt

    def test_includes_extracted_data(self):
        state = _make_state()
        instance = _make_instance()
        extracted = {"email": "alice@example.com", "name": "Alice"}
        prompt = self.builder.build_response_prompt(
            instance,
            state,
            self.fsm_def,
            extracted_data=extracted,
            user_message="my info",
        )

        assert "<extracted_data>" in prompt
        assert "alice@example.com" in prompt
        assert "Alice" in prompt

    def test_handles_missing_extracted_data(self):
        state = _make_state()
        instance = _make_instance()
        prompt = self.builder.build_response_prompt(
            instance,
            state,
            self.fsm_def,
            extracted_data=None,
            user_message="hello",
        )
        assert "<extracted_data>" not in prompt

    def test_includes_transition_info_when_transition_occurred(self):
        state = _make_state(state_id="farewell", purpose="Say goodbye")
        instance = _make_instance(current_state="farewell")
        prompt = self.builder.build_response_prompt(
            instance,
            state,
            self.fsm_def,
            transition_occurred=True,
            previous_state="greeting",
            user_message="bye",
        )

        assert "<transition_info>" in prompt
        assert "greeting" in prompt
        assert "farewell" in prompt

    def test_no_transition_info_when_no_transition(self):
        state = _make_state()
        instance = _make_instance()
        prompt = self.builder.build_response_prompt(
            instance,
            state,
            self.fsm_def,
            transition_occurred=False,
            user_message="hello",
        )
        assert "<transition_info>" not in prompt

    def test_includes_response_instructions(self):
        state = _make_state(
            purpose="Collect data",
            response_instructions="Always ask for the user's name first.",
        )
        instance = _make_instance()
        prompt = self.builder.build_response_prompt(
            instance, state, self.fsm_def, user_message="hi"
        )

        assert "<response_instructions>" in prompt
        assert "user's name first" in prompt

    def test_includes_user_message_section(self):
        state = _make_state()
        instance = _make_instance()
        prompt = self.builder.build_response_prompt(
            instance,
            state,
            self.fsm_def,
            user_message="I want to book a flight",
        )

        assert "<user_message>" in prompt
        assert "book a flight" in prompt

    def test_empty_user_message_omits_section(self):
        state = _make_state()
        instance = _make_instance()
        prompt = self.builder.build_response_prompt(
            instance, state, self.fsm_def, user_message=""
        )
        assert "<user_message>" not in prompt

    def test_includes_required_context_keys_as_information_needed(self):
        state = _make_state(
            purpose="Collect info",
            required_context_keys=["email", "phone"],
        )
        instance = _make_instance()
        prompt = self.builder.build_response_prompt(
            instance, state, self.fsm_def, user_message="hi"
        )

        assert "<information_still_needed>" in prompt
        assert "email address" in prompt
        assert "phone number" in prompt


# ============================================================================
# 12-13. TransitionPromptBuilder (REMOVED)
# TransitionPromptBuilder and TransitionPromptConfig have been removed.
# Transition decisions now use classification-based resolution.
# ============================================================================


# ============================================================================
# 14-15. History management strategies
# ============================================================================


class TestHistoryManagement:
    """Tests for conversation history limiting strategies."""

    def test_message_count_strategy_limits_exchanges(self):
        cfg = BasePromptConfig(
            max_history_messages=2,
            history_strategy=HistoryManagementStrategy.MESSAGE_COUNT,
        )
        builder = BasePromptBuilder(config=cfg)

        exchanges = [
            {"user": "msg1"},
            {"user": "msg2"},
            {"user": "msg3"},
            {"user": "msg4"},
        ]
        result = builder._manage_conversation_history(exchanges)
        assert len(result) == 2
        assert result[0]["user"] == "msg3"
        assert result[1]["user"] == "msg4"

    def test_message_count_strategy_no_trimming_needed(self):
        cfg = BasePromptConfig(
            max_history_messages=10,
            history_strategy=HistoryManagementStrategy.MESSAGE_COUNT,
        )
        builder = BasePromptBuilder(config=cfg)
        exchanges = [{"user": "a"}, {"user": "b"}]
        result = builder._manage_conversation_history(exchanges)
        assert len(result) == 2

    def test_token_budget_strategy_limits_by_tokens(self):
        cfg = BasePromptConfig(
            max_token_budget=200,
            history_strategy=HistoryManagementStrategy.TOKEN_BUDGET,
        )
        builder = BasePromptBuilder(config=cfg)

        # Create exchanges large enough that not all fit
        long_msg = "word " * 200  # ~1000 chars
        exchanges = [
            {"user": long_msg},
            {"user": long_msg},
            {"user": "short"},
        ]
        result = builder._manage_conversation_history(exchanges)
        # Should keep fewer than all 3
        assert len(result) < len(exchanges)

    def test_token_budget_always_keeps_at_least_one(self):
        cfg = BasePromptConfig(
            max_token_budget=100,
            history_strategy=HistoryManagementStrategy.TOKEN_BUDGET,
        )
        builder = BasePromptBuilder(config=cfg)

        big = "x" * 5000
        exchanges = [{"user": big}]
        result = builder._manage_conversation_history(exchanges)
        # Should keep at least one even if it exceeds budget
        assert len(result) == 1

    def test_hybrid_strategy_picks_smaller_result(self):
        cfg = BasePromptConfig(
            max_history_messages=10,
            max_token_budget=200,
            history_strategy=HistoryManagementStrategy.HYBRID,
        )
        builder = BasePromptBuilder(config=cfg)

        long_msg = "word " * 200
        exchanges = [{"user": long_msg} for _ in range(5)]
        result = builder._manage_conversation_history(exchanges)
        # Message count allows 5 but token budget should reduce it
        assert len(result) < 5

    def test_empty_exchanges_returns_empty(self):
        builder = BasePromptBuilder()
        assert builder._manage_conversation_history([]) == []


# ============================================================================
# 16. Prompt output format -- returns string with expected XML sections
# ============================================================================


class TestPromptOutputFormat:
    """Tests that prompt output has expected structural sections."""

    def test_extraction_prompt_returns_string(self):
        builder = DataExtractionPromptBuilder()
        fsm_def = _make_fsm_definition()
        state = _make_state()
        instance = _make_instance()
        prompt = builder.build_extraction_prompt(instance, state, fsm_def)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_response_prompt_returns_string(self):
        builder = ResponseGenerationPromptBuilder()
        fsm_def = _make_fsm_definition()
        state = _make_state()
        instance = _make_instance()
        prompt = builder.build_response_prompt(
            instance, state, fsm_def, user_message="hello"
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_extraction_prompt_has_expected_sections(self):
        builder = DataExtractionPromptBuilder()
        fsm_def = _make_fsm_definition()
        state = _make_state(
            required_context_keys=["name"],
            extraction_instructions="Get name",
        )
        instance = _make_instance(
            context_data={"email": "a@b.com"},
            exchanges=[{"user": "hi"}, {"system": "hey"}],
        )
        prompt = builder.build_extraction_prompt(instance, state, fsm_def)

        expected_sections = [
            "<task>",
            "</task>",
            "<data_extraction>",
            "</data_extraction>",
            "<extraction_focus>",
            "</extraction_focus>",
            "<response_format>",
            "</response_format>",
            "<guidelines>",
            "</guidelines>",
            "<format_rules>",
            "</format_rules>",
            "<conversation_history>",
            "</conversation_history>",
            "<current_context>",
            "</current_context>",
        ]
        for section in expected_sections:
            assert section in prompt, f"Missing section: {section}"

    def test_response_prompt_has_expected_sections(self):
        builder = ResponseGenerationPromptBuilder()
        fsm_def = _make_fsm_definition(persona="Helper")
        state = _make_state(
            purpose="Help user",
            response_instructions="Be brief",
            required_context_keys=["email"],
        )
        instance = _make_instance(persona="Helper")
        prompt = builder.build_response_prompt(
            instance,
            state,
            fsm_def,
            extracted_data={"email": "a@b.com"},
            user_message="hi",
        )

        expected_sections = [
            "<task>",
            "</task>",
            "<response_generation>",
            "</response_generation>",
            "<persona>",
            "</persona>",
            "<final_state_context>",
            "</final_state_context>",
            "<user_message>",
            "</user_message>",
            "<extracted_data>",
            "</extracted_data>",
            "<response_format>",
            "</response_format>",
            "<guidelines>",
            "</guidelines>",
        ]
        for section in expected_sections:
            assert section in prompt, f"Missing section: {section}"


# ============================================================================
# Additional edge-case and security tests
# ============================================================================


class TestSecurityFiltering:
    """Tests for context security filtering."""

    def test_filter_context_for_security_removes_internal_keys(self):
        builder = BasePromptBuilder()
        data = {"name": "Alice", "_token": "secret", "system_log": "hidden", "age": 30}
        filtered = builder._filter_context_for_security(data)
        assert filtered == {"name": "Alice", "age": 30}

    def test_filter_context_disabled(self):
        cfg = BasePromptConfig(filter_internal_context=False)
        builder = BasePromptBuilder(config=cfg)
        data = {"name": "Alice", "_token": "secret"}
        filtered = builder._filter_context_for_security(data)
        assert filtered == data


class TestHumanizeKey:
    """Tests for key humanization."""

    def test_known_keys(self):
        builder = BasePromptBuilder()
        assert builder._humanize_key("email") == "email address"
        assert builder._humanize_key("phone") == "phone number"
        assert builder._humanize_key("name") == "full name"

    def test_unknown_key_underscores_to_spaces(self):
        builder = BasePromptBuilder()
        assert builder._humanize_key("favorite_color") == "favorite color"


class TestEscapeCdata:
    """Tests for CDATA escape handling."""

    def test_escapes_cdata_end_sequence(self):
        builder = BasePromptBuilder()
        result = builder._escape_cdata("data]]>more")
        assert "]]>" not in result
        assert "]]&gt;" in result

    def test_no_cdata_end_passes_through(self):
        builder = BasePromptBuilder()
        text = "normal data without issues"
        assert builder._escape_cdata(text) == text


class TestFieldExtractionContinueDeAnchor:
    """Step 2b: build_field_extraction_prompt de-anchors from the agent-loop
    sentinel 'Continue.' — when user_message == 'Continue.' it adds guidance to
    extract from task/context; a normal user message leaves the path unchanged."""

    _SENTINEL_MARKER = "agent-loop continuation signal"

    def _build(self, user_message: str) -> str:
        builder = FieldExtractionPromptBuilder()
        instance = _make_instance(current_state="plan")
        field_config = FieldExtractionConfig(
            field_name="plan_steps",
            field_type="any",
            extraction_instructions="Extract the list of plan steps.",
        )
        return builder.build_field_extraction_prompt(
            instance=instance,
            field_config=field_config,
            user_message=user_message,
            dynamic_context={"task": "compute the sum of 2 and 3"},
        )

    def test_continue_sentinel_adds_deanchor_instruction(self):
        prompt = self._build("Continue.")
        assert self._SENTINEL_MARKER in prompt
        # Still anchors on the task/context.
        assert "Already extracted" in prompt or "task" in prompt.lower()

    def test_normal_message_path_unchanged(self):
        prompt = self._build("I want to book a flight to Paris")
        assert self._SENTINEL_MARKER not in prompt

    def test_real_user_typing_continue_is_not_suppressed(self):
        # The de-anchor branch only ADDS guidance; it does not remove the
        # normal "User message:" line, so a real user typing 'Continue.' is
        # unharmed.
        prompt = self._build("Continue.")
        assert "User message: Continue." in prompt


# ============================================================================
# Step 10 / F-10 -- FieldExtractionPromptBuilder honours history capping
# ============================================================================


def _field_config():
    return FieldExtractionConfig(
        field_name="destination",
        field_type="str",
        extraction_instructions="Extract the destination city.",
    )


def _instance_with_big_history(turns: int = 5, words: int = 200):
    """Five large user/assistant exchanges, each far beyond any small budget."""
    exchanges = []
    for i in range(turns):
        exchanges.append({"user": f"U{i} " + "word " * words})
        exchanges.append({"system": f"A{i} " + "word " * words})
    return _make_instance(exchanges=exchanges)


class TestFieldExtractionHistoryCapping:
    """SC-12: the field-extraction builder assembles its own history block, so
    it must apply the same `history_strategy` / token-budget capping the base
    builders get. Before the fix it emitted `get_recent()` whole -- all five
    turns, uncapped `User:` lines -- under a budget that trimmed
    `DataExtractionPromptBuilder` to a single turn."""

    _TIGHT: ClassVar[dict] = {
        "history_strategy": HistoryManagementStrategy.TOKEN_BUDGET,
        "max_token_budget": 100,
    }

    def _field_prompt(self, instance, **cfg):
        builder = FieldExtractionPromptBuilder(FieldExtractionPromptConfig(**cfg))
        return builder.build_field_extraction_prompt(
            instance=instance,
            field_config=_field_config(),
            user_message="I want to go to Paris",
        )

    def _data_prompt(self, instance, **cfg):
        builder = DataExtractionPromptBuilder(DataExtractionPromptConfig(**cfg))
        return builder.build_extraction_prompt(
            instance, _make_state(), _make_fsm_definition()
        )

    def test_tight_token_budget_drops_the_oldest_turn(self):
        instance = _instance_with_big_history()
        prompt = self._field_prompt(instance, **self._TIGHT)

        # Oldest exchange gone, newest retained.
        assert "U0" not in prompt
        assert "A4" in prompt

    def test_generous_budget_keeps_every_turn(self):
        # Vacuity guard for the test above: the drop must come from the budget,
        # not from the history block having silently become unreachable.
        instance = _instance_with_big_history()
        prompt = self._field_prompt(
            instance,
            history_strategy=HistoryManagementStrategy.TOKEN_BUDGET,
            max_token_budget=100_000,
        )
        assert "Recent conversation:" in prompt
        for i in range(5):
            assert f"U{i}" in prompt
            assert f"A{i}" in prompt

    def test_multi_key_history_entry_loses_no_message(self):
        """D-019: the role map is many-to-one (`"user" if role == "user" else
        "system"`), so collapsing an entry into ONE dict silently dropped a
        message whenever an entry carried two non-`user` roles. In-tree writers
        emit single-key entries, but `Conversation.exchanges` is a plain
        `list[dict[str, str]]` -- restored/injected history can be multi-key,
        and `API._replay_history` explicitly handles that shape."""
        # NOTE: `_make_instance` routes through `add_user_message`/
        # `add_system_message`, which SPLIT a multi-key entry into two
        # single-key ones -- so it cannot build this shape. Assign
        # `exchanges` directly, which is what a restored session does.
        instance = _make_instance()
        instance.context.conversation.exchanges = [
            {"assistant": "AAA-assistant", "system": "SSS-system"}
        ]
        prompt = self._field_prompt(
            instance,
            history_strategy=HistoryManagementStrategy.TOKEN_BUDGET,
            max_token_budget=100_000,
        )
        assert "AAA-assistant" in prompt
        assert "SSS-system" in prompt

    def test_multi_key_user_and_system_entry_keeps_both(self):
        """The `{"user": ..., "system": ...}` shape `_replay_history` accepts."""
        instance = _make_instance()
        instance.context.conversation.exchanges = [
            {"user": "UUU-user", "system": "SSS-system"}
        ]
        prompt = self._field_prompt(
            instance,
            history_strategy=HistoryManagementStrategy.TOKEN_BUDGET,
            max_token_budget=100_000,
        )
        assert "User: UUU-user" in prompt
        assert "Assistant: SSS-system" in prompt

    def test_length_tracks_the_data_extraction_builder(self):
        instance = _instance_with_big_history()
        field_prompt = self._field_prompt(instance, **self._TIGHT)
        data_prompt = self._data_prompt(instance, **self._TIGHT)

        # Same order of magnitude as the builder that was already capped.
        # Un-fixed, the field prompt is the larger of the two.
        assert len(field_prompt) < len(data_prompt)

    def test_user_lines_are_capped_like_assistant_lines(self):
        instance = _make_instance(
            exchanges=[{"user": "x" * 5000}, {"system": "y" * 5000}]
        )
        prompt = self._field_prompt(
            instance,
            history_strategy=HistoryManagementStrategy.TOKEN_BUDGET,
            max_token_budget=100_000,
        )
        user_lines = [ln for ln in prompt.splitlines() if ln.startswith("  User: ")]
        assistant_lines = [
            ln for ln in prompt.splitlines() if ln.startswith("  Assistant: ")
        ]
        assert user_lines and assistant_lines
        for line in user_lines + assistant_lines:
            # 150-char cap + "..." + label prefix.
            assert len(line) <= 150 + 3 + len("  Assistant: ")

    def test_message_count_strategy_is_honoured(self):
        instance = _instance_with_big_history()
        prompt = self._field_prompt(
            instance,
            history_strategy=HistoryManagementStrategy.MESSAGE_COUNT,
            max_history_messages=1,
        )
        # get_recent(1) yields the final pair; the count strategy then keeps
        # only the last message of it.
        assert prompt.count("  User: ") + prompt.count("  Assistant: ") == 1

    def test_history_can_be_disabled(self):
        instance = _instance_with_big_history()
        prompt = self._field_prompt(instance, include_conversation_history=False)
        assert "Recent conversation:" not in prompt

    def test_empty_history_emits_no_section(self):
        prompt = self._field_prompt(_make_instance())
        assert "Recent conversation:" not in prompt


# ============================================================================
# Step 10 / F-11 completion -- _filter_context_for_security recurses
# ============================================================================


class TestNestedContextSecurityFiltering:
    """SC-11b: `_filter_context_for_security` is the LAST line of defense for
    `initial_context` / `API.update_context` writes (`clean_context_keys` is
    not on that path). Before the fix it walked only the top level, so
    `{"user": {"password": "hunter2"}}` reached the LLM prompt intact."""

    def test_nested_secret_removed_by_the_helper(self):
        builder = BasePromptBuilder()
        filtered = builder._filter_context_for_security(
            {"user": {"password": "hunter2", "name": "bob"}}
        )
        assert filtered == {"user": {"name": "bob"}}

    def test_nested_secret_absent_from_the_built_extraction_prompt(self):
        # The point of the fix: assert through the real prompt-building entry
        # point, not only the private helper.
        builder = DataExtractionPromptBuilder()
        instance = _make_instance(
            context_data={"user": {"password": "hunter2", "name": "bob"}}
        )
        prompt = builder.build_extraction_prompt(
            instance, _make_state(), _make_fsm_definition()
        )
        assert "hunter2" not in prompt
        assert "bob" in prompt

    def test_nested_secret_absent_from_the_built_response_prompt(self):
        builder = ResponseGenerationPromptBuilder()
        instance = _make_instance(
            context_data={"account": {"api_key": "sk-live-123", "tier": "gold"}}
        )
        prompt = builder.build_response_prompt(
            instance, _make_state(), _make_fsm_definition(), user_message="hi"
        )
        assert "sk-live-123" not in prompt
        assert "gold" in prompt

    def test_nested_secret_absent_from_the_field_extraction_prompt(self):
        builder = FieldExtractionPromptBuilder()
        prompt = builder.build_field_extraction_prompt(
            instance=_make_instance(),
            field_config=_field_config(),
            user_message="Paris",
            dynamic_context={"session": {"auth_token": "tok-abc", "city": "Paris"}},
        )
        assert "tok-abc" not in prompt

    def test_dicts_inside_lists_are_filtered(self):
        builder = BasePromptBuilder()
        filtered = builder._filter_context_for_security(
            {"users": [{"password": "x", "name": "ann"}, {"name": "bo"}]}
        )
        assert filtered == {"users": [{"name": "ann"}, {"name": "bo"}]}

    def test_internal_prefix_stripped_at_depth(self):
        builder = BasePromptBuilder()
        filtered = builder._filter_context_for_security(
            {"outer": {"_hidden": 1, "SYSTEM_log": 2, "kept": 3}}
        )
        assert filtered == {"outer": {"kept": 3}}

    def test_falsy_values_survive_at_every_depth(self):
        builder = BasePromptBuilder()
        payload = {
            "a": 0,
            "b": False,
            "c": "",
            "d": [],
            "nested": {"e": 0, "f": False, "g": "", "h": [], "i": {}},
        }
        assert builder._filter_context_for_security(payload) == payload

    def test_strings_are_not_traversed(self):
        builder = BasePromptBuilder()
        filtered = builder._filter_context_for_security({"note": "password: hunter2"})
        # The filter matches KEY names, not values -- unchanged by design.
        assert filtered == {"note": "password: hunter2"}

    def test_non_string_keys_do_not_crash_the_filter(self):
        builder = BasePromptBuilder()
        filtered = builder._filter_context_for_security({"outer": {1: "a", None: "b"}})
        assert filtered == {"outer": {1: "a", None: "b"}}

    def test_container_past_the_depth_bound_is_dropped_not_passed_through(self):
        builder = BasePromptBuilder()
        payload: dict = {"password": "hunter2"}
        for _ in range(MAX_CONTEXT_FILTER_DEPTH + 2):
            payload = {"n": payload}
        filtered = builder._filter_context_for_security(payload)
        assert "hunter2" not in repr(filtered)

    def test_payload_just_inside_the_depth_bound_is_still_filtered(self):
        # Vacuity guard: a bound of 0 would satisfy the test above alone.
        builder = BasePromptBuilder()
        payload: dict = {"password": "hunter2", "name": "bob"}
        for _ in range(MAX_CONTEXT_FILTER_DEPTH - 2):
            payload = {"n": payload}
        filtered = builder._filter_context_for_security(payload)
        assert "hunter2" not in repr(filtered)
        assert "bob" in repr(filtered)

    def test_self_referential_dict_does_not_recurse_forever(self):
        builder = BasePromptBuilder()
        payload: dict = {"name": "bob"}
        payload["self"] = payload
        builder._filter_context_for_security(payload)  # must terminate

    def test_filter_disabled_returns_payload_untouched(self):
        cfg = BasePromptConfig(filter_internal_context=False)
        builder = BasePromptBuilder(config=cfg)
        data = {"user": {"password": "hunter2"}}
        assert builder._filter_context_for_security(data) == data

    def test_security_filtering_runs_before_the_key_count_cap(self):
        # Ordering invariant: capping first would let the cap decide whether a
        # secret is seen. The nested secret must be removed even though its
        # holder key survives the cap.
        builder = DataExtractionPromptBuilder()
        context_data = {f"filler_{i}": i for i in range(250)}
        context_data["profile"] = {"password": "hunter2", "nickname": "bo"}
        instance = _make_instance(context_data=context_data)
        prompt = builder.build_extraction_prompt(
            instance, _make_state(), _make_fsm_definition()
        )
        # Holder key survived the 200-key cap (it is the newest key)...
        assert "nickname" in prompt
        # ...and the secret nested inside it was still removed.
        assert "hunter2" not in prompt
