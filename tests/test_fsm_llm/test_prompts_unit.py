"""
Unit tests for the FSM-LLM prompt builder system.

Tests BasePromptConfig validation, text sanitization, token estimation,
and the three prompt builders: DataExtraction, ResponseGeneration, Transition.
"""

import pytest

from fsm_llm.definitions import (
    FSMContext,
    FSMDefinition,
    FSMInstance,
    State,
    Transition,
    TransitionOption,
)
from fsm_llm.prompts import (
    BasePromptBuilder,
    BasePromptConfig,
    DataExtractionPromptBuilder,
    DataExtractionPromptConfig,
    HistoryManagementStrategy,
    ResponseGenerationPromptBuilder,
    TransitionPromptBuilder,
    TransitionPromptConfig,
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
# 12-13. TransitionPromptBuilder
# ============================================================================


class TestTransitionPromptBuilder:
    """Tests for transition decision prompt building."""

    def setup_method(self):
        self.builder = TransitionPromptBuilder()

    def _make_options(self, *specs):
        """Create TransitionOption list from (target, description, priority) tuples."""
        return [
            TransitionOption(target_state=t, description=d, priority=p)
            for t, d, p in specs
        ]

    def test_builds_transition_prompt_with_options(self):
        options = self._make_options(
            ("farewell", "User wants to leave", 100),
            ("help", "User needs help", 50),
        )
        prompt = self.builder.build_transition_prompt(
            current_state="greeting",
            available_transitions=options,
            context={"name": "Alice"},
            user_message="I need help",
        )

        assert "<transition_decision>" in prompt
        assert "</transition_decision>" in prompt
        assert "<available_options>" in prompt
        assert "farewell" in prompt
        assert "help" in prompt

    def test_single_transition_option(self):
        options = self._make_options(("end", "Conversation complete", 100))
        prompt = self.builder.build_transition_prompt(
            current_state="review",
            available_transitions=options,
            context={},
            user_message="done",
        )

        assert "<available_options>" in prompt
        assert "end" in prompt
        assert '<option id="1">' in prompt
        # Only one option
        assert '<option id="2">' not in prompt

    def test_multiple_transitions_sorted_by_priority(self):
        options = self._make_options(
            ("low_priority", "Low", 200),
            ("high_priority", "High", 10),
            ("mid_priority", "Mid", 100),
        )
        prompt = self.builder.build_transition_prompt(
            current_state="start",
            available_transitions=options,
            context={},
            user_message="go",
        )

        # With deterministic_output=True (default), options are sorted by priority
        high_pos = prompt.index("high_priority")
        mid_pos = prompt.index("mid_priority")
        low_pos = prompt.index("low_priority")
        assert high_pos < mid_pos < low_pos

    def test_includes_extracted_data(self):
        options = self._make_options(("next", "Move on", 100))
        extracted = {"name": "Alice", "intent": "booking"}
        prompt = self.builder.build_transition_prompt(
            current_state="greeting",
            available_transitions=options,
            context={},
            user_message="book a room",
            extracted_data=extracted,
        )

        assert "<extracted_information>" in prompt
        assert "Alice" in prompt
        assert "booking" in prompt

    def test_includes_context_summary(self):
        options = self._make_options(("next", "Proceed", 100))
        context = {"user_name": "Bob", "step": 2}
        prompt = self.builder.build_transition_prompt(
            current_state="collect",
            available_transitions=options,
            context=context,
            user_message="continue",
        )

        assert "<context_summary>" in prompt
        assert "Bob" in prompt

    def test_filters_internal_context_keys(self):
        options = self._make_options(("next", "Go", 100))
        context = {"name": "Visible", "_internal": "hidden", "system_debug": "nope"}
        prompt = self.builder.build_transition_prompt(
            current_state="s1",
            available_transitions=options,
            context=context,
            user_message="ok",
        )

        assert "Visible" in prompt
        assert "_internal" not in prompt
        assert "system_debug" not in prompt

    def test_empty_context_omits_section(self):
        options = self._make_options(("next", "Go", 100))
        prompt = self.builder.build_transition_prompt(
            current_state="s1",
            available_transitions=options,
            context={},
            user_message="ok",
        )
        assert "<context_summary>" not in prompt

    def test_response_format_section(self):
        options = self._make_options(("end", "Done", 100))
        prompt = self.builder.build_transition_prompt(
            current_state="s1",
            available_transitions=options,
            context={},
            user_message="finish",
        )

        assert "<response_format>" in prompt
        assert "selected_transition" in prompt

    def test_includes_transition_descriptions(self):
        options = self._make_options(
            ("farewell", "User explicitly says goodbye", 100),
        )
        prompt = self.builder.build_transition_prompt(
            current_state="greeting",
            available_transitions=options,
            context={},
            user_message="bye",
        )
        assert "<when>" in prompt
        assert "User explicitly says goodbye" in prompt

    def test_disable_transition_descriptions(self):
        cfg = TransitionPromptConfig(include_transition_descriptions=False)
        builder = TransitionPromptBuilder(config=cfg)
        options = self._make_options(("end", "Done", 100))
        prompt = builder.build_transition_prompt(
            current_state="s1",
            available_transitions=options,
            context={},
            user_message="ok",
        )
        assert "<when>" not in prompt


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

    def test_transition_prompt_returns_string(self):
        builder = TransitionPromptBuilder()
        options = [
            TransitionOption(target_state="end", description="Done", priority=100)
        ]
        prompt = builder.build_transition_prompt(
            current_state="start",
            available_transitions=options,
            context={},
            user_message="go",
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
