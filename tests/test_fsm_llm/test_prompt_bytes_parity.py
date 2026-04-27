"""
R6.1 byte-parity tests for `to_compile_time_template` producers.

Each cohort fixture builds (state, fsm_definition, instance, ...), retrieves the
compile-time triple via `ResponseGenerationPromptBuilder.to_compile_time_template`,
renders the legacy prompt via `build_response_prompt`, and asserts that
`template.format(**runtime_env)` byte-equals the legacy renderer output.

Under D-S1-03 (degenerate single-placeholder design), the substitution is the
identity on the rendered string — but the test still proves the invariant the
cohort Leaf path will rely on at step 3 (compile_fsm cohort emission) and step 8
(Theorem-2 invariant).
"""

from __future__ import annotations

import pytest

from fsm_llm.dialog.definitions import (
    ClassificationSchema,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    IntentDefinition,
    State,
)
from fsm_llm.dialog.prompts import (
    ResponseGenerationPromptBuilder,
    classification_compile_time_template,
    classification_template,
)


def _make_cohort_state(
    state_id: str = "cohort",
    purpose: str = "Cohort response state",
    response_instructions: str | None = None,
) -> State:
    """Build a cohort-eligible state — response-only, no extractions, no transitions."""
    kwargs: dict = {
        "id": state_id,
        "description": f"{state_id} description",
        "purpose": purpose,
    }
    if response_instructions is not None:
        kwargs["response_instructions"] = response_instructions
    return State(**kwargs)


def _make_fsm(state: State, persona: str | None = None) -> FSMDefinition:
    """Build a minimal FSMDefinition wrapping a single cohort state."""
    kwargs: dict = {
        "name": "ParityTestFSM",
        "description": "byte-parity fixture",
        "initial_state": state.id,
        "states": {state.id: state},
    }
    if persona is not None:
        kwargs["persona"] = persona
    return FSMDefinition(**kwargs)


# ---------------------------------------------------------------------------
# Fixtures (≥3 cohort variants per plan SC1)
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_cohort_fixture() -> tuple[State, FSMDefinition, FSMInstance]:
    """Cohort #1: bare-minimum response-only state, no persona, no instructions."""
    state = _make_cohort_state()
    defn = _make_fsm(state)
    instance = FSMInstance(
        fsm_id="parity_min", current_state=state.id, context=FSMContext()
    )
    return state, defn, instance


@pytest.fixture
def persona_cohort_fixture() -> tuple[State, FSMDefinition, FSMInstance]:
    """Cohort #2: cohort state under a personaful FSM."""
    state = _make_cohort_state(
        state_id="welcome",
        purpose="Greet the user warmly",
    )
    defn = _make_fsm(state, persona="A cheerful concierge with a dry sense of humour.")
    instance = FSMInstance(
        fsm_id="parity_persona", current_state=state.id, context=FSMContext()
    )
    return state, defn, instance


@pytest.fixture
def instructions_cohort_fixture() -> tuple[State, FSMDefinition, FSMInstance]:
    """Cohort #3: cohort state with response_instructions baked in."""
    state = _make_cohort_state(
        state_id="confirm",
        purpose="Confirm the order details",
        response_instructions="Use a numbered list. Mention exactly two next-step options.",
    )
    defn = _make_fsm(state, persona="An efficient ops bot.")
    instance = FSMInstance(
        fsm_id="parity_instructions",
        current_state=state.id,
        context=FSMContext(data={"order_id": "ABC-001"}),
    )
    return state, defn, instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResponseGenCompileTimeTriple:
    """`to_compile_time_template` returns the kernel-Leaf triple shape."""

    def test_triple_shape_matches_kernel_leaf(self, minimal_cohort_fixture):
        state, defn, _ = minimal_cohort_fixture
        builder = ResponseGenerationPromptBuilder()

        triple = builder.to_compile_time_template(state, defn)

        assert isinstance(triple, tuple)
        assert len(triple) == 3
        template, input_vars, schema_ref = triple
        assert isinstance(template, str)
        assert isinstance(input_vars, tuple)
        assert all(isinstance(v, str) for v in input_vars)
        assert schema_ref is None or isinstance(schema_ref, str)

    def test_template_has_the_expected_single_placeholder(self, minimal_cohort_fixture):
        state, defn, _ = minimal_cohort_fixture
        builder = ResponseGenerationPromptBuilder()

        template, input_vars, _ = builder.to_compile_time_template(state, defn)
        assert template == "{response_prompt_rendered}"
        assert input_vars == ("response_prompt_rendered",)

    def test_schema_ref_is_none_preserving_string_contract(self, minimal_cohort_fixture):
        """``schema_ref=None`` preserves CB_RESPOND's string-returning contract.

        Schema enforcement (Pydantic decode of ``ResponseGenerationResponse``)
        requires pipeline output-unwrap support, deferred to a future plan.
        For R6.2 v1 the cohort Leaf returns a string from the oracle, matching
        the legacy host path byte-for-byte.
        """
        state, defn, _ = minimal_cohort_fixture
        builder = ResponseGenerationPromptBuilder()

        _, _, schema_ref = builder.to_compile_time_template(state, defn)
        assert schema_ref is None

    def test_existing_to_template_and_schema_unchanged(self, minimal_cohort_fixture):
        """The R3 narrowed `to_template_and_schema` path still works.

        R6.1 is purely additive (D-S1-03); the existing producer is untouched.
        """
        state, defn, instance = minimal_cohort_fixture
        builder = ResponseGenerationPromptBuilder()

        template_legacy, env_legacy, schema_legacy = builder.to_template_and_schema(
            instance, state, defn, user_message="hello"
        )
        # R3-narrow shape: rendered prompt + brace-escapes + empty env + None schema.
        assert isinstance(template_legacy, str)
        assert env_legacy == {}
        assert schema_legacy is None
        # The compile-time path's triple has the documented shape.
        ct_template, ct_input_vars, ct_schema_ref = builder.to_compile_time_template(
            state, defn
        )
        assert ct_template == "{response_prompt_rendered}"
        assert ct_input_vars == ("response_prompt_rendered",)
        assert ct_schema_ref is None


@pytest.mark.parametrize(
    "fixture_name",
    ["minimal_cohort_fixture", "persona_cohort_fixture", "instructions_cohort_fixture"],
)
class TestResponseGenByteParity:
    """`template.format(**runtime_env)` byte-equals the legacy renderer output.

    Under D-S1-03 (degenerate single-placeholder design) the substitution is the
    identity on the rendered string — but this is the load-bearing invariant for
    step 3 (cohort Leaf emission) and step 8 (Theorem-2 strict equality).
    """

    def test_byte_parity_basic(self, fixture_name, request):
        state, defn, instance = request.getfixturevalue(fixture_name)
        builder = ResponseGenerationPromptBuilder()

        template, input_vars, _ = builder.to_compile_time_template(state, defn)
        rendered = builder.build_response_prompt(
            instance, state, defn, user_message="hello"
        )
        runtime_env = {input_vars[0]: rendered}

        assert template.format(**runtime_env) == rendered

    def test_byte_parity_with_extracted_data(self, fixture_name, request):
        state, defn, instance = request.getfixturevalue(fixture_name)
        builder = ResponseGenerationPromptBuilder()

        template, input_vars, _ = builder.to_compile_time_template(state, defn)
        rendered = builder.build_response_prompt(
            instance,
            state,
            defn,
            extracted_data={"name": "Alex", "qty": 3},
            user_message="three please",
        )
        runtime_env = {input_vars[0]: rendered}

        assert template.format(**runtime_env) == rendered

    def test_byte_parity_with_transition_metadata(self, fixture_name, request):
        state, defn, instance = request.getfixturevalue(fixture_name)
        builder = ResponseGenerationPromptBuilder()

        template, input_vars, _ = builder.to_compile_time_template(state, defn)
        rendered = builder.build_response_prompt(
            instance,
            state,
            defn,
            transition_occurred=True,
            previous_state="prior_state",
            user_message="ok",
        )
        runtime_env = {input_vars[0]: rendered}

        assert template.format(**runtime_env) == rendered


class TestClassificationCompileTimeWrapper:
    """`classification_compile_time_template` byte-equals the legacy free fn."""

    def _schema(self) -> ClassificationSchema:
        return ClassificationSchema(
            intents=[
                IntentDefinition(name="buy", description="user wants to purchase"),
                IntentDefinition(name="browse", description="user is just looking"),
            ],
            fallback_intent="browse",
        )

    def test_template_byte_equals_legacy(self):
        schema = self._schema()
        ct_template, _, _ = classification_compile_time_template(schema)
        legacy_template, _, _ = classification_template(schema)
        assert ct_template == legacy_template

    def test_input_vars_empty(self):
        schema = self._schema()
        _, ct_input_vars, _ = classification_compile_time_template(schema)
        assert ct_input_vars == ()

    def test_schema_ref_none_today(self):
        """Until a Pydantic ClassificationLLMResponse exists, schema_ref is None."""
        schema = self._schema()
        _, _, ct_schema_ref = classification_compile_time_template(schema)
        assert ct_schema_ref is None

    def test_template_format_roundtrips_to_unescaped_rendered(self):
        """`template.format()` un-doubles the brace-escapes back to the rendered prompt.

        The legacy `classification_template` escapes literal `{` / `}` (which
        appear in the embedded JSON schema) to `{{` / `}}` so that
        `.format(**env)` is safe. A no-arg `.format()` therefore un-doubles them
        and yields the un-escaped rendered prompt — i.e. exactly what
        `build_classification_system_prompt` produces directly.
        """
        from fsm_llm.dialog.prompts import build_classification_system_prompt

        schema = self._schema()
        ct_template, _, _ = classification_compile_time_template(schema)
        rendered_unescaped = build_classification_system_prompt(schema)
        assert ct_template.format() == rendered_unescaped
