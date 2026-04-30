"""
Producer-level parity tests for R3 step 14 (narrowed) — the
``to_template_and_schema`` methods on the 3 PromptBuilder classes and the
``classification_template`` free function.

Per D-PLAN-09-RESOLUTION-step14-narrowed (2026-04-27, plan
plan_2026-04-27_a426f667), step 14 ships a template-producer surface only:
each new method/function emits ``(template_str, env, schema)`` such that
``template_str.format(**env)`` byte-equals the existing ``build_*_prompt`` /
``build_classification_system_prompt`` output for the same inputs. The
pipeline.py callbacks remain on ``build_*_prompt`` + LiteLLMInterface direct
calls until R6 (per-state Leaf specialisation), so wire-level T5 byte-equality
is trivially preserved by the unchanged pipeline; producer-level parity is the
gate.

The representative test matrix below draws from the same kinds of FSM/state
fixtures already covered in ``test_prompts_unit.py`` — minimal greeting state,
state with required_context_keys, state with extraction/response instructions,
state with persona, with/without history, etc. — to give the producer-level
parity gate meaningful coverage without rebuilding the full prompt unit-test
matrix.
"""

from __future__ import annotations

import pytest

from fsm_llm.dialog.definitions import (
    ClassificationSchema,
    FieldExtractionConfig,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    IntentDefinition,
    State,
    Transition,
)
from fsm_llm.dialog.prompts import (
    ClassificationPromptConfig,
    DataExtractionPromptBuilder,
    FieldExtractionPromptBuilder,
    ResponseGenerationPromptBuilder,
    _escape_format_braces,
    build_classification_system_prompt,
    classification_template,
)

# ============================================================================
# Helpers — small fixture matrix for parity tests
# ============================================================================


def _state_minimal() -> State:
    return State(
        id="greeting",
        description="Greeting",
        purpose="Greet the user",
        transitions=[],
    )


def _state_with_extraction() -> State:
    return State(
        id="collect_info",
        description="Collect user info",
        purpose="Collect the user's name and email",
        extraction_instructions=(
            "Extract the user's full name and email address. "
            "Respect formatting like {first} {last}."
        ),
        required_context_keys=["full_name", "email"],
        response_instructions="Acknowledge the info and ask for confirmation.",
        transitions=[
            Transition(target_state="confirm", description="Done", priority=100)
        ],
    )


def _instance_minimal(persona: str | None = None) -> FSMInstance:
    return FSMInstance(
        fsm_id="t",
        current_state="greeting",
        context=FSMContext(data={}),
        persona=persona,
    )


def _instance_with_history() -> FSMInstance:
    ctx = FSMContext(data={"prior": "value"})
    ctx.conversation.add_user_message("Hi, I'm Alice.")
    ctx.conversation.add_system_message("Hello Alice — what's your email?")
    return FSMInstance(
        fsm_id="t",
        current_state="collect_info",
        context=ctx,
        persona="A friendly assistant",
    )


def _fsm_definition(persona: str | None = None) -> FSMDefinition:
    return FSMDefinition(
        name="t",
        description="t",
        initial_state="greeting",
        persona=persona,
        states={
            "greeting": State(
                id="greeting",
                description="Greeting",
                purpose="Greet",
                transitions=[
                    Transition(target_state="end", description="Done", priority=100)
                ],
            ),
            "end": State(
                id="end",
                description="End",
                purpose="End",
                transitions=[],
            ),
        },
    )


# ============================================================================
# 1. Helper: _escape_format_braces invariant
# ============================================================================


class TestEscapeFormatBraces:
    """Round-trip invariant: escape then format(**{}) returns original."""

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "no braces here",
            "one { brace",
            "one } brace",
            "matched {pair}",
            "doubled {{already}}",
            'json: {"key": "value"}',
            'schema: {\n  "k": 1\n}',
            "weird }{ order",
            "deep {{{nested}}}",
        ],
    )
    def test_round_trip(self, raw: str) -> None:
        escaped = _escape_format_braces(raw)
        assert escaped.format() == raw


# ============================================================================
# 2. DataExtractionPromptBuilder.to_template_and_schema parity
# ============================================================================


class TestDataExtractionTemplateParity:
    """``template.format(**env)`` byte-equals ``build_extraction_prompt``."""

    def test_minimal_state(self) -> None:
        builder = DataExtractionPromptBuilder()
        instance = _instance_minimal()
        state = _state_minimal()
        fsm = _fsm_definition()

        rendered = builder.build_extraction_prompt(instance, state, fsm)
        template, env, schema = builder.to_template_and_schema(instance, state, fsm)

        assert template.format(**env) == rendered
        assert env == {}
        assert schema is None

    def test_state_with_required_keys_and_instructions(self) -> None:
        builder = DataExtractionPromptBuilder()
        instance = _instance_with_history()
        state = _state_with_extraction()
        fsm = _fsm_definition(persona="Friendly")

        rendered = builder.build_extraction_prompt(instance, state, fsm)
        template, env, _schema = builder.to_template_and_schema(instance, state, fsm)

        assert template.format(**env) == rendered
        # The original prompt embeds a JSON schema with literal braces;
        # the template must therefore contain doubled braces.
        assert "{{" in template
        assert "}}" in template


# ============================================================================
# 3. ResponseGenerationPromptBuilder.to_template_and_schema parity
# ============================================================================


class TestResponseGenerationTemplateParity:
    """``template.format(**env)`` byte-equals ``build_response_prompt``."""

    def test_no_extracted_data_no_transition(self) -> None:
        builder = ResponseGenerationPromptBuilder()
        instance = _instance_minimal()
        state = _state_minimal()
        fsm = _fsm_definition()

        rendered = builder.build_response_prompt(
            instance,
            state,
            fsm,
            extracted_data=None,
            transition_occurred=False,
            previous_state=None,
            user_message="",
        )
        template, env, schema = builder.to_template_and_schema(
            instance,
            state,
            fsm,
            extracted_data=None,
            transition_occurred=False,
            previous_state=None,
            user_message="",
        )

        assert template.format(**env) == rendered
        assert env == {}
        assert schema is None

    def test_with_extracted_data_and_transition(self) -> None:
        builder = ResponseGenerationPromptBuilder()
        instance = _instance_with_history()
        state = _state_with_extraction()
        fsm = _fsm_definition(persona="Friendly")
        extracted = {"full_name": "Alice", "email": "a@b.com"}

        rendered = builder.build_response_prompt(
            instance,
            state,
            fsm,
            extracted_data=extracted,
            transition_occurred=True,
            previous_state="greeting",
            user_message="My email is a@b.com",
        )
        template, env, _schema = builder.to_template_and_schema(
            instance,
            state,
            fsm,
            extracted_data=extracted,
            transition_occurred=True,
            previous_state="greeting",
            user_message="My email is a@b.com",
        )

        assert template.format(**env) == rendered

    def test_user_message_with_literal_braces(self) -> None:
        """A literal `{` in user content must round-trip via the template."""
        builder = ResponseGenerationPromptBuilder()
        instance = _instance_minimal(persona="A {curly} persona")
        state = _state_minimal()
        fsm = _fsm_definition(persona="A {curly} persona")

        rendered = builder.build_response_prompt(
            instance,
            state,
            fsm,
            extracted_data={"k": "v with { brace"},
            transition_occurred=False,
            previous_state=None,
            user_message="message with } brace",
        )
        template, env, _ = builder.to_template_and_schema(
            instance,
            state,
            fsm,
            extracted_data={"k": "v with { brace"},
            transition_occurred=False,
            previous_state=None,
            user_message="message with } brace",
        )

        assert template.format(**env) == rendered


# ============================================================================
# 4. FieldExtractionPromptBuilder.to_template_and_schema parity
# ============================================================================


class TestFieldExtractionTemplateParity:
    """``template.format(**env)`` byte-equals ``build_field_extraction_prompt``."""

    def _field_config(self) -> FieldExtractionConfig:
        return FieldExtractionConfig(
            field_name="email",
            field_type="str",
            extraction_instructions="Extract a valid email address.",
        )

    def test_no_dynamic_context(self) -> None:
        builder = FieldExtractionPromptBuilder()
        instance = _instance_minimal()
        cfg = self._field_config()

        rendered = builder.build_field_extraction_prompt(
            instance, cfg, "my email is a@b.com", dynamic_context=None
        )
        template, env, schema = builder.to_template_and_schema(
            instance, cfg, "my email is a@b.com", dynamic_context=None
        )

        assert template.format(**env) == rendered
        assert env == {}
        assert schema is None

    def test_with_dynamic_context_and_history(self) -> None:
        builder = FieldExtractionPromptBuilder()
        instance = _instance_with_history()
        cfg = self._field_config()

        rendered = builder.build_field_extraction_prompt(
            instance, cfg, "a@b.com", dynamic_context={"full_name": "Alice"}
        )
        template, env, _ = builder.to_template_and_schema(
            instance, cfg, "a@b.com", dynamic_context={"full_name": "Alice"}
        )

        assert template.format(**env) == rendered
        # The prompt embeds an inline JSON example with literal braces.
        assert "{{" in template

    def test_user_message_with_literal_braces(self) -> None:
        builder = FieldExtractionPromptBuilder()
        instance = _instance_minimal()
        cfg = self._field_config()
        msg = "they said: {price} and {qty}"

        rendered = builder.build_field_extraction_prompt(instance, cfg, msg)
        template, env, _ = builder.to_template_and_schema(instance, cfg, msg)

        assert template.format(**env) == rendered


# ============================================================================
# 5. classification_template parity
# ============================================================================


class TestClassificationTemplateParity:
    """``template.format(**env)`` byte-equals ``build_classification_system_prompt``."""

    def _schema(self) -> ClassificationSchema:
        return ClassificationSchema(
            intents=[
                IntentDefinition(name="buy", description="User wants to purchase"),
                IntentDefinition(name="browse", description="User is browsing"),
                IntentDefinition(name="other", description="Anything else"),
            ],
            fallback_intent="other",
        )

    def test_default_config(self) -> None:
        schema = self._schema()
        rendered = build_classification_system_prompt(schema)
        template, env, schema_out = classification_template(schema)

        assert template.format(**env) == rendered
        assert env == {}
        assert schema_out is None
        # Embedded JSON schema → at least one literal `{` was in the rendered
        # output, so the template must have doubled braces.
        assert "{{" in template

    def test_multi_intent_no_reasoning(self) -> None:
        schema = self._schema()
        cfg = ClassificationPromptConfig(
            multi_intent=True,
            include_reasoning=False,
            include_entities=False,
        )
        rendered = build_classification_system_prompt(schema, cfg)
        template, env, _ = classification_template(schema, cfg)

        assert template.format(**env) == rendered

    def test_intent_description_with_braces(self) -> None:
        """Intent descriptions with literal `{` must round-trip."""
        schema = ClassificationSchema(
            intents=[
                IntentDefinition(
                    name="buy", description="User wants to {purchase} something"
                ),
                IntentDefinition(name="other", description="Anything else"),
            ],
            fallback_intent="other",
        )
        rendered = build_classification_system_prompt(schema)
        template, env, _ = classification_template(schema)
        assert template.format(**env) == rendered


# ============================================================================
# 6. Smoke: return-shape contract
# ============================================================================


class TestReturnShapeContract:
    """All 4 producers return a 3-tuple ``(str, dict, type | None)``."""

    def test_data_extraction_shape(self) -> None:
        builder = DataExtractionPromptBuilder()
        out = builder.to_template_and_schema(
            _instance_minimal(), _state_minimal(), _fsm_definition()
        )
        assert isinstance(out, tuple) and len(out) == 3
        template, env, schema = out
        assert isinstance(template, str)
        assert isinstance(env, dict)
        assert schema is None

    def test_response_generation_shape(self) -> None:
        builder = ResponseGenerationPromptBuilder()
        out = builder.to_template_and_schema(
            _instance_minimal(), _state_minimal(), _fsm_definition()
        )
        assert isinstance(out, tuple) and len(out) == 3

    def test_field_extraction_shape(self) -> None:
        builder = FieldExtractionPromptBuilder()
        cfg = FieldExtractionConfig(
            field_name="x", field_type="str", extraction_instructions="extract x"
        )
        out = builder.to_template_and_schema(_instance_minimal(), cfg, "msg")
        assert isinstance(out, tuple) and len(out) == 3

    def test_classification_template_shape(self) -> None:
        schema = ClassificationSchema(
            intents=[
                IntentDefinition(name="a", description="aa"),
                IntentDefinition(name="other", description="oo"),
            ],
            fallback_intent="other",
        )
        out = classification_template(schema)
        assert isinstance(out, tuple) and len(out) == 3
