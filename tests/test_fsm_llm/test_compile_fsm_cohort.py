"""
R6.2 cohort emission shape tests.

Cohort Leaf emission is gated on the ``FSM_LLM_COHORT_EMISSION`` env var. These
tests set it to ``"1"`` and assert the compiled term contains a ``Leaf`` for
cohort states (terminal response-only) and the legacy
``App(Var(CB_RESPOND), Var(VAR_INSTANCE))`` for non-cohort states.
"""

from __future__ import annotations

import pytest

from fsm_llm.dialog.compile_fsm import (
    CB_RESPOND,
    COHORT_RESPONSE_PROMPT_VAR,
    VAR_INSTANCE,
    _is_cohort_state,
    compile_fsm,
)
from fsm_llm.dialog.definitions import (
    ClassificationExtractionConfig,
    FieldExtractionConfig,
    FSMDefinition,
    IntentDefinition,
    State,
    Transition,
)
from fsm_llm.runtime.ast import App, Case, Leaf, Let, Var


@pytest.fixture
def cohort_emission_on(monkeypatch):
    """Activate the FSM_LLM_COHORT_EMISSION env-var gate for the duration."""
    monkeypatch.setenv("FSM_LLM_COHORT_EMISSION", "1")
    # Bust compile_fsm_cached's lru_cache so cohort-emitted terms aren't
    # masked by a previously-compiled non-cohort cached term.
    from fsm_llm.dialog.compile_fsm import _compile_fsm_by_id

    _compile_fsm_by_id.cache_clear()
    yield
    _compile_fsm_by_id.cache_clear()


@pytest.fixture
def cohort_emission_off(monkeypatch):
    """Explicit OFF state — confirms gate semantics for negative tests.

    Post-R9a (plan_2026-04-27_32652286 step 3): the gate defaults to ON, so
    asserting OFF requires an explicit falsy value rather than ``delenv``.
    The opt-out env reading is itself removed in R9c (step 5), at which
    point this fixture and the negative tests below are retired.
    """
    monkeypatch.setenv("FSM_LLM_COHORT_EMISSION", "0")
    from fsm_llm.dialog.compile_fsm import _compile_fsm_by_id

    _compile_fsm_by_id.cache_clear()
    yield
    _compile_fsm_by_id.cache_clear()


def _terminal_cohort_state(state_id: str = "end") -> State:
    return State(
        id=state_id,
        description=f"{state_id} description",
        purpose="Wrap up the conversation",
        response_instructions="Say goodbye.",
    )


def _unwrap_to_case(term) -> Case:
    body = term
    for _ in range(4):
        body = body.body
    assert isinstance(body, Case)
    return body


# ---------------------------------------------------------------------------
# Predicate tests — _is_cohort_state respects the gate
# ---------------------------------------------------------------------------


class TestCohortPredicateGate:
    def test_predicate_returns_false_when_gate_off(self, cohort_emission_off):
        state = _terminal_cohort_state()
        defn = FSMDefinition(
            name="F",
            description="d",
            initial_state=state.id,
            states={state.id: state},
        )
        assert _is_cohort_state(state, defn) is False

    def test_predicate_returns_true_for_cohort_when_gate_on(self, cohort_emission_on):
        state = _terminal_cohort_state()
        defn = FSMDefinition(
            name="F",
            description="d",
            initial_state=state.id,
            states={state.id: state},
        )
        assert _is_cohort_state(state, defn) is True

    def test_state_with_transitions_excluded(self, cohort_emission_on):
        sa = State(
            id="a",
            description="a",
            purpose="start",
            transitions=[Transition(target_state="b", description="go")],
        )
        sb = _terminal_cohort_state("b")
        defn = FSMDefinition(
            name="F",
            description="d",
            initial_state="a",
            states={"a": sa, "b": sb},
        )
        assert _is_cohort_state(sa, defn) is False
        assert _is_cohort_state(sb, defn) is True

    def test_state_with_field_extractions_excluded(self, cohort_emission_on):
        sx = State(
            id="x",
            description="x",
            purpose="extract",
            field_extractions=[
                FieldExtractionConfig(field_name="n", extraction_instructions="get")
            ],
        )
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states={"x": sx}
        )
        assert _is_cohort_state(sx, defn) is False

    def test_state_with_required_context_keys_excluded(self, cohort_emission_on):
        sy = State(id="y", description="y", purpose="need", required_context_keys=["k"])
        defn = FSMDefinition(
            name="F", description="d", initial_state="y", states={"y": sy}
        )
        assert _is_cohort_state(sy, defn) is False

    def test_state_with_classification_extractions_excluded(self, cohort_emission_on):
        sc = State(
            id="c",
            description="c",
            purpose="classify",
            classification_extractions=[
                ClassificationExtractionConfig(
                    field_name="intent",
                    intents=[
                        IntentDefinition(name="buy", description="buy"),
                        IntentDefinition(name="browse", description="browse"),
                    ],
                    fallback_intent="browse",
                )
            ],
        )
        defn = FSMDefinition(
            name="F", description="d", initial_state="c", states={"c": sc}
        )
        assert _is_cohort_state(sc, defn) is False

    def test_state_with_extraction_instructions_excluded(self, cohort_emission_on):
        se = State(
            id="e",
            description="e",
            purpose="extract",
            extraction_instructions="extract everything",
        )
        defn = FSMDefinition(
            name="F", description="d", initial_state="e", states={"e": se}
        )
        assert _is_cohort_state(se, defn) is False


# ---------------------------------------------------------------------------
# Compilation shape tests — compile_fsm emits Leaf for cohort, App for non-cohort
# ---------------------------------------------------------------------------


class TestCompileFsmCohortShape:
    def test_terminal_cohort_state_compiles_to_leaf(self, cohort_emission_on):
        state = _terminal_cohort_state("end")
        defn = FSMDefinition(
            name="F",
            description="d",
            initial_state=state.id,
            states={state.id: state},
        )

        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        cohort_branch = case_node.branches["end"]

        assert isinstance(cohort_branch, Leaf)
        assert cohort_branch.template == "{response_prompt_rendered}"
        assert cohort_branch.input_vars == ("response_prompt_rendered",)
        assert cohort_branch.schema_ref is None
        # The input_var is the canonical reserved name.
        assert cohort_branch.input_vars[0] == COHORT_RESPONSE_PROMPT_VAR

    def test_terminal_cohort_state_compiles_to_app_when_gate_off(
        self, cohort_emission_off
    ):
        """Default-OFF path preserves byte-equivalent legacy compile output."""
        state = _terminal_cohort_state("end")
        defn = FSMDefinition(
            name="F",
            description="d",
            initial_state=state.id,
            states={state.id: state},
        )

        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        legacy_branch = case_node.branches["end"]

        assert isinstance(legacy_branch, App)
        assert isinstance(legacy_branch.fn, Var)
        assert legacy_branch.fn.name == CB_RESPOND
        assert isinstance(legacy_branch.arg, Var)
        assert legacy_branch.arg.name == VAR_INSTANCE

    def test_non_cohort_state_unchanged(self, cohort_emission_on):
        """States with transitions retain the legacy Let+Case dispatch shape."""
        sa = State(
            id="a",
            description="a",
            purpose="start",
            transitions=[Transition(target_state="b", description="go")],
        )
        sb = _terminal_cohort_state("b")
        defn = FSMDefinition(
            name="F",
            description="d",
            initial_state="a",
            states={"a": sa, "b": sb},
        )

        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)

        # 'a' has transitions → Let(disc, eval_transit, Case(...)) shape.
        assert isinstance(case_node.branches["a"], Let)
        # 'b' is terminal cohort → Leaf.
        assert isinstance(case_node.branches["b"], Leaf)

    def test_cohort_leaf_carries_no_schema_today(self, cohort_emission_on):
        """``schema_ref=None`` preserves CB_RESPOND's str-returning contract."""
        state = _terminal_cohort_state()
        defn = FSMDefinition(
            name="F",
            description="d",
            initial_state=state.id,
            states={state.id: state},
        )

        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        leaf = case_node.branches[state.id]
        assert isinstance(leaf, Leaf)
        assert leaf.schema_ref is None


# ---------------------------------------------------------------------------
# Reserved-name contract test — COHORT_RESPONSE_PROMPT_VAR must not collide
# ---------------------------------------------------------------------------


class TestReservedNameContract:
    def test_cohort_var_does_not_collide_with_RESERVED_VARS(self):
        from fsm_llm.dialog.compile_fsm import RESERVED_VARS

        assert COHORT_RESPONSE_PROMPT_VAR not in RESERVED_VARS, (
            f"COHORT_RESPONSE_PROMPT_VAR={COHORT_RESPONSE_PROMPT_VAR!r} "
            f"collides with a RESERVED_VARS entry — pick a fresh name."
        )

    def test_cohort_var_is_the_documented_string(self):
        assert COHORT_RESPONSE_PROMPT_VAR == "response_prompt_rendered"
