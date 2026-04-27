"""R9b cohort-widening tests (plan_2026-04-27_32652286 step 4).

Covers plan E6: a state with transitions but no extractions, whose response
prompt does not reference extracted/classified fields, would *in principle*
become cohort-eligible. In practice (D-R9b worst-case), the
``_state_transitions_break_cohort_byte_equivalence`` STOP-IF guard
keeps any transition-bearing state on the legacy CB_RESPOND path because
the cohort env-build cannot resolve transition_occurred / previous_state /
target-state response_instructions before transition evaluation runs. These
tests pin both the helper behaviour AND the STOP-IF guard so future
per-target pre-rendering work has an audit trail.
"""

from __future__ import annotations

import pytest

from fsm_llm.dialog.compile_fsm import (
    _is_cohort_state,
    _response_prompt_uses_extracted_fields,
    _state_transitions_break_cohort_byte_equivalence,
)
from fsm_llm.dialog.definitions import (
    ClassificationExtractionConfig,
    FieldExtractionConfig,
    FSMDefinition,
    IntentDefinition,
    State,
    Transition,
)


@pytest.fixture(autouse=True)
def cohort_emission_on(monkeypatch):
    """All R9b widening tests assume the gate is ON (R9a default)."""
    monkeypatch.setenv("FSM_LLM_COHORT_EMISSION", "1")
    from fsm_llm.dialog.compile_fsm import _compile_fsm_by_id

    _compile_fsm_by_id.cache_clear()
    yield
    _compile_fsm_by_id.cache_clear()


# ---------------------------------------------------------------------------
# Helper construction
# ---------------------------------------------------------------------------


def _make_state(
    state_id: str,
    *,
    transitions: list[Transition] | None = None,
    response_instructions: str = "Say something nice.",
    field_extractions: list[FieldExtractionConfig] | None = None,
    classification_extractions: list[ClassificationExtractionConfig] | None = None,
) -> State:
    return State(
        id=state_id,
        description=f"{state_id} description",
        purpose=f"{state_id} purpose",
        response_instructions=response_instructions,
        transitions=transitions or [],
        field_extractions=field_extractions or [],
        classification_extractions=classification_extractions or [],
    )


def _make_fsm(states: dict[str, State], initial: str = "start") -> FSMDefinition:
    return FSMDefinition(
        name="test_widening",
        description="R9b widening fixture",
        initial_state=initial,
        persona="test",
        states=states,
    )


# ---------------------------------------------------------------------------
# Helper: _response_prompt_uses_extracted_fields
# ---------------------------------------------------------------------------


class TestResponsePromptUsesExtractedFields:
    def test_no_response_instructions_returns_false(self):
        state = _make_state("s", response_instructions="")
        assert _response_prompt_uses_extracted_fields(state) is False

    def test_no_extraction_fields_returns_false(self):
        state = _make_state(
            "s", response_instructions="Mention the user_name and timestamp."
        )
        # No field_extractions / classification_extractions configured.
        assert _response_prompt_uses_extracted_fields(state) is False

    def test_field_name_appearing_as_whole_word_returns_true(self):
        state = _make_state(
            "s",
            response_instructions="Greet the user by user_name.",
            field_extractions=[
                FieldExtractionConfig(
                    field_name="user_name",
                    field_type="str",
                    extraction_instructions="extract name",
                )
            ],
        )
        assert _response_prompt_uses_extracted_fields(state) is True

    def test_classification_field_name_match_returns_true(self):
        state = _make_state(
            "s",
            response_instructions="React to user_intent appropriately.",
            classification_extractions=[
                ClassificationExtractionConfig(
                    field_name="user_intent",
                    intents=[
                        IntentDefinition(name="buy", description="purchase"),
                        IntentDefinition(name="browse", description="look"),
                    ],
                    fallback_intent="browse",
                )
            ],
        )
        assert _response_prompt_uses_extracted_fields(state) is True

    def test_field_name_as_substring_does_not_match(self):
        # "user" is a substring of "user_name"; only whole-word matches count.
        state = _make_state(
            "s",
            response_instructions="Greet the user warmly.",  # "user", not "user_name"
            field_extractions=[
                FieldExtractionConfig(
                    field_name="user_name",
                    field_type="str",
                    extraction_instructions="extract name",
                )
            ],
        )
        assert _response_prompt_uses_extracted_fields(state) is False


# ---------------------------------------------------------------------------
# Helper: _state_transitions_break_cohort_byte_equivalence (STOP-IF guard)
# ---------------------------------------------------------------------------


class TestTransitionByteEquivalenceGuard:
    def test_terminal_state_passes_guard(self):
        state = _make_state("end", transitions=[])
        assert _state_transitions_break_cohort_byte_equivalence(state) is False

    def test_state_with_transitions_fails_guard(self):
        state = _make_state(
            "start",
            transitions=[
                Transition(
                    target_state="end",
                    description="t",
                    conditions=[],
                    priority=100,
                )
            ],
        )
        assert _state_transitions_break_cohort_byte_equivalence(state) is True


# ---------------------------------------------------------------------------
# Composite: _is_cohort_state with widening signals (D-R9b worst-case
# behaviour: transition-bearing states stay on legacy path).
# ---------------------------------------------------------------------------


class TestIsCohortStateWidened:
    def test_terminal_extraction_free_state_is_cohort(self):
        state = _make_state("end")
        fsm = _make_fsm({"end": state}, initial="end")
        assert _is_cohort_state(state, fsm) is True

    def test_state_with_transitions_blocked_by_byte_equivalence_guard(self):
        # E6 / D-R9b worst case: even though no extractions are declared and
        # the response_instructions doesn't reference extracted fields, the
        # transition byte-equivalence STOP-IF guard rejects this state until
        # per-target pre-rendering lands.
        start = _make_state(
            "start",
            transitions=[
                Transition(
                    target_state="end",
                    description="advance",
                    conditions=[],
                    priority=100,
                )
            ],
            response_instructions="Greet briefly.",
        )
        end = _make_state("end")
        fsm = _make_fsm({"start": start, "end": end})
        assert _is_cohort_state(start, fsm) is False  # blocked by STOP-IF
        assert _is_cohort_state(end, fsm) is True  # terminal still cohort

    def test_state_with_extracted_field_reference_blocked_by_helper(self):
        # Hypothetical: even if the byte-equivalence guard were lifted, the
        # field-reference helper would still reject this state.
        state = _make_state(
            "s",
            response_instructions="Use user_name in the reply.",
            field_extractions=[
                FieldExtractionConfig(
                    field_name="user_name",
                    field_type="str",
                    extraction_instructions="extract",
                )
            ],
        )
        fsm = _make_fsm({"s": state}, initial="s")
        # Already rejected by field_extractions presence (R6.2 hard-no), but
        # the helper would also flag it independently.
        assert _is_cohort_state(state, fsm) is False
        assert _response_prompt_uses_extracted_fields(state) is True
