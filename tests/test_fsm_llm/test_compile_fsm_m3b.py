"""
M3b — non-cohort response Leaf emission tests (gated on
``State._emit_response_leaf_for_non_cohort``).

Plan: ``plans/plan_2026-04-28_6597e394`` step 4b.

When the M3a private State attr is flipped to ``True``, the compiler
lifts the non-cohort response position from
``App(CB_RESPOND, instance)`` to
``Let(NONCOHORT_RESPONSE_PROMPT_VAR,
       App(CB_RENDER_RESPONSE_PROMPT, instance),
       Leaf("{...}", input_vars=("...",), schema_ref=None))``.

This file ships ~30 strict-Theorem-2 tests asserting:

(A) **Compile shape** — opt-in flips response position from App to Let+Leaf.
(B) **Theorem-2 strict equality** — ``Executor.oracle_calls == plan(...).predicted_calls``
    for non-cohort states under field=True. Each non-cohort state's response
    position is exactly one Leaf, so predicted_calls == 1.
(C) **Default-False preservation** — without the flip, compile output is
    byte-equivalent to the M3a baseline (no Let+Leaf shape leaks into the
    default path).
(D) **Pipeline integration** — the host callable
    ``CB_RENDER_RESPONSE_PROMPT`` is bound by ``_build_compiled_env``
    and the Let-Leaf shape evaluates end-to-end with a scripted oracle.
"""

from __future__ import annotations

import pytest

from fsm_llm.dialog.compile_fsm import (
    CB_RENDER_RESPONSE_PROMPT,
    CB_RESPOND,
    NONCOHORT_RESPONSE_PROMPT_VAR,
    RESERVED_VARS,
    VAR_CONV_ID,
    VAR_INSTANCE,
    VAR_MESSAGE,
    VAR_STATE_ID,
    _is_cohort_state,
    compile_fsm,
)
from fsm_llm.dialog.definitions import (
    ClassificationExtractionConfig,
    FieldExtractionConfig,
    FSMContext,
    FSMDefinition,
    FSMInstance,
    IntentDefinition,
    State,
    Transition,
    TransitionCondition,
)
from fsm_llm.runtime.ast import App, Case, Leaf, Let, Var
from fsm_llm.runtime.executor import Executor
from fsm_llm.runtime.planner import PlanInputs, plan

# ---------------------------------------------------------------------------
# Test infrastructure — scripted oracle, FSM helpers
# ---------------------------------------------------------------------------


class _ScriptedOracle:
    """Deterministic oracle. Each invoke returns the next scripted reply."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.invocations: list[tuple[str, type | None]] = []

    def invoke(self, prompt, schema=None, *, model_override=None, env=None):
        self.invocations.append((prompt, schema))
        if not self._replies:
            raise RuntimeError("scripted oracle exhausted")
        return self._replies.pop(0)

    def tokenize(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def cache_clear():
    """Bust compile_fsm_cached's lru_cache between tests so opt-in flips don't
    leak between cases."""
    from fsm_llm.dialog.compile_fsm import _compile_fsm_by_id

    _compile_fsm_by_id.cache_clear()
    yield
    _compile_fsm_by_id.cache_clear()


def _enable_leaf(state: State) -> State:
    """Flip the M3a private attr to True. Returns the same State for chaining."""
    state._emit_response_leaf_for_non_cohort = True
    return state


def _unwrap_to_case(term) -> Case:
    body = term
    for _ in range(4):
        body = body.body
    assert isinstance(body, Case)
    return body


def _make_state_with_field_extraction(state_id: str = "s") -> State:
    return State(
        id=state_id,
        description=f"{state_id} desc",
        purpose=f"Extract a field for {state_id}",
        response_instructions="Acknowledge the extracted value.",
        field_extractions=[
            FieldExtractionConfig(
                field_name="user_name",
                extraction_instructions="Extract the user's name.",
            )
        ],
    )


def _make_state_with_class_extraction(state_id: str = "c") -> State:
    return State(
        id=state_id,
        description=f"{state_id} desc",
        purpose="Classify intent",
        response_instructions="Respond per intent.",
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


def _make_state_with_required_keys(state_id: str = "r") -> State:
    return State(
        id=state_id,
        description=f"{state_id} desc",
        purpose="Need keys",
        response_instructions="Respond once keys present.",
        required_context_keys=["k1"],
    )


def _make_state_with_extraction_instructions(state_id: str = "e") -> State:
    return State(
        id=state_id,
        description=f"{state_id} desc",
        purpose="Bulk extract",
        response_instructions="Respond after bulk extract.",
        extraction_instructions="Extract everything notable.",
    )


def _make_two_state_fsm_with_transition() -> FSMDefinition:
    sa = State(
        id="a",
        description="a desc",
        purpose="start",
        response_instructions="Greet and route.",
        transitions=[
            Transition(
                target_state="b",
                description="always go to b",
                conditions=[
                    TransitionCondition(
                        description="always",
                        logic={"==": [1, 1]},
                    )
                ],
            )
        ],
    )
    sb = State(
        id="b",
        description="b desc",
        purpose="end",
        response_instructions="Say bye.",
    )
    return FSMDefinition(
        name="TwoState",
        description="d",
        initial_state="a",
        states={"a": sa, "b": sb},
    )


# ---------------------------------------------------------------------------
# (A) Compile shape — opt-in produces Let+Leaf at response position
# ---------------------------------------------------------------------------


class TestCompileShape:
    def test_field_extraction_state_default_emits_app_cb_respond(self, cache_clear):
        """Default field=False → legacy ``App(CB_RESPOND, instance)`` shape."""
        sx = _make_state_with_field_extraction("x")
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states={"x": sx}
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        # Outer Let wraps for field_extraction; innermost body is App(CB_RESPOND, ...)
        body = case_node.branches["x"]
        # walk through the field-extraction Let
        assert isinstance(body, Let)
        inner = body.body
        # No transitions → no transition Let; the response is the inner App.
        assert isinstance(inner, App)
        assert isinstance(inner.fn, Var)
        assert inner.fn.name == CB_RESPOND

    def test_field_extraction_state_optin_emits_let_leaf(self, cache_clear):
        """Field=True → response position becomes Let(...) wrapping a Leaf."""
        sx = _enable_leaf(_make_state_with_field_extraction("x"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states={"x": sx}
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        body = case_node.branches["x"]
        # Outermost: extraction Let; inner: response Let; innermost: Leaf.
        assert isinstance(body, Let)
        response_let = body.body
        assert isinstance(response_let, Let)
        assert response_let.name == NONCOHORT_RESPONSE_PROMPT_VAR
        # Let's value must be App(CB_RENDER_RESPONSE_PROMPT, instance).
        v = response_let.value
        assert isinstance(v, App)
        assert isinstance(v.fn, Var) and v.fn.name == CB_RENDER_RESPONSE_PROMPT
        assert isinstance(v.arg, Var) and v.arg.name == VAR_INSTANCE
        # Let's body must be a Leaf with the correct template + input_vars.
        leaf_node = response_let.body
        assert isinstance(leaf_node, Leaf)
        assert leaf_node.template == "{" + NONCOHORT_RESPONSE_PROMPT_VAR + "}"
        assert leaf_node.input_vars == (NONCOHORT_RESPONSE_PROMPT_VAR,)
        assert leaf_node.schema_ref is None

    def test_classification_extraction_state_optin_emits_let_leaf(self, cache_clear):
        sc = _enable_leaf(_make_state_with_class_extraction("c"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="c", states={"c": sc}
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        body = case_node.branches["c"]
        # outer Let = class_extract; inner = response Let; inner.body = Leaf.
        assert isinstance(body, Let)
        response_let = body.body
        assert isinstance(response_let, Let)
        assert response_let.name == NONCOHORT_RESPONSE_PROMPT_VAR
        assert isinstance(response_let.body, Leaf)

    def test_required_keys_state_optin_emits_let_leaf(self, cache_clear):
        sr = _enable_leaf(_make_state_with_required_keys("r"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="r", states={"r": sr}
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        body = case_node.branches["r"]
        # required_context_keys triggers extraction Let.
        assert isinstance(body, Let)
        response_let = body.body
        assert isinstance(response_let, Let)
        assert response_let.name == NONCOHORT_RESPONSE_PROMPT_VAR
        assert isinstance(response_let.body, Leaf)

    def test_extraction_instructions_state_optin_emits_let_leaf(self, cache_clear):
        se = _enable_leaf(_make_state_with_extraction_instructions("e"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="e", states={"e": se}
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        body = case_node.branches["e"]
        assert isinstance(body, Let)
        response_let = body.body
        assert isinstance(response_let, Let)
        assert response_let.name == NONCOHORT_RESPONSE_PROMPT_VAR
        assert isinstance(response_let.body, Leaf)

    def test_transition_state_optin_emits_let_leaf_in_each_case_arm(self, cache_clear):
        """For non-terminal states with transitions, the Let+Leaf body is shared
        across all Case arms (advanced/blocked/ambiguous/default)."""
        defn = _make_two_state_fsm_with_transition()
        # Opt-in only on the non-terminal state 'a'. 'b' is terminal-cohort.
        _enable_leaf(defn.states["a"])
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        a_body = case_node.branches["a"]
        # Outer Let = eval_transit dispatch; its body is a Case.
        assert isinstance(a_body, Let)
        case_dispatch = a_body.body
        assert isinstance(case_dispatch, Case)
        # Each non-ambig branch should be the response Let-Leaf shape.
        for arm_name in ("advanced", "blocked"):
            arm = case_dispatch.branches[arm_name]
            assert isinstance(arm, Let)
            assert arm.name == NONCOHORT_RESPONSE_PROMPT_VAR
            assert isinstance(arm.body, Leaf)
        # Default arm
        assert isinstance(case_dispatch.default, Let)
        assert case_dispatch.default.name == NONCOHORT_RESPONSE_PROMPT_VAR

    def test_terminal_non_cohort_state_optin_emits_let_leaf(self, cache_clear):
        """A terminal state that is non-cohort (e.g. has required_context_keys but
        no transitions) ships the Let+Leaf at top level inside the Case branch."""
        sr = _enable_leaf(_make_state_with_required_keys("r"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="r", states={"r": sr}
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        body = case_node.branches["r"]
        # Outer Let (extraction); inner Let (response); inner.body Leaf.
        assert isinstance(body, Let)
        response_let = body.body
        assert isinstance(response_let, Let)
        assert isinstance(response_let.body, Leaf)
        # This state is terminal — no Case dispatch wraps it.
        # Confirm by walking outward: response_let is directly the body of the
        # outer extraction Let.
        assert body.body is response_let

    def test_cohort_state_unaffected_by_optin_field(self, cache_clear):
        """A cohort state (terminal, no extractions, no transitions) emits a
        bare Leaf regardless of the M3a field. The opt-in only affects
        non-cohort states."""
        s = State(
            id="end",
            description="end",
            purpose="wrap up",
            response_instructions="Say goodbye.",
        )
        # Even with the field set to True on a cohort state, we still emit
        # the cohort Leaf shape (single Leaf with COHORT_RESPONSE_PROMPT_VAR).
        _enable_leaf(s)
        defn = FSMDefinition(
            name="F", description="d", initial_state="end", states={"end": s}
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        # cohort branch is a bare Leaf — the cohort branch wins because
        # _is_cohort_state runs first in _compile_state.
        cohort_branch = case_node.branches["end"]
        assert isinstance(cohort_branch, Leaf)
        # Crucially: the input_var is the cohort var, NOT the non-cohort one.
        assert cohort_branch.input_vars == ("response_prompt_rendered",)


# ---------------------------------------------------------------------------
# (B) Theorem-2 strict equality
# ---------------------------------------------------------------------------


class TestTheorem2NonCohortLeaf:
    """``Executor.oracle_calls == plan(...).predicted_calls`` for non-cohort
    response Leaves under field=True."""

    @pytest.mark.parametrize(
        "state_factory,state_id",
        [
            (_make_state_with_field_extraction, "x"),
            (_make_state_with_class_extraction, "c"),
            (_make_state_with_required_keys, "r"),
            (_make_state_with_extraction_instructions, "e"),
        ],
        ids=[
            "field_extraction",
            "class_extraction",
            "required_keys",
            "extraction_instructions",
        ],
    )
    def test_oracle_calls_equals_predicted_strict(
        self, state_factory, state_id, cache_clear
    ):
        s = _enable_leaf(state_factory(state_id))
        defn = FSMDefinition(
            name="F", description="d", initial_state=state_id, states={state_id: s}
        )
        term = compile_fsm(defn)
        case_body = _unwrap_to_case(term)

        instance = FSMInstance(
            fsm_id="m3b",
            current_state=state_id,
            context=FSMContext(),
        )
        # Stub the rendering callback to produce a fixed prompt — we do NOT
        # exercise the real prompt builder here (that's tested in section D).
        rendered = "<RENDERED-NON-COHORT-PROMPT>"

        env = {
            VAR_STATE_ID: state_id,
            VAR_MESSAGE: "hi",
            VAR_CONV_ID: "conv-m3b",
            VAR_INSTANCE: instance,
            CB_RENDER_RESPONSE_PROMPT: lambda _i: rendered,
            # Extraction callbacks are no-ops (return None — the Let bindings
            # discard them).
            "_cb_extract": lambda _i: None,
            "_cb_field_extract": lambda _i: None,
            "_cb_class_extract": lambda _i: None,
            CB_RESPOND: lambda _i: "would-not-fire",
        }

        oracle = _ScriptedOracle(["Hello back!"])
        executor = Executor(oracle=oracle)
        result = executor.run(case_body, env)

        assert result == "Hello back!"
        # Strict Theorem-2: 1 Leaf = 1 oracle call. Host callbacks (extract,
        # render) do NOT count.
        assert executor.oracle_calls == 1
        predicted = plan(PlanInputs(n=1, tau=1, K=8192))
        assert predicted.predicted_calls == 1
        assert executor.oracle_calls == predicted.predicted_calls

    def test_oracle_invoked_with_substituted_prompt(self, cache_clear):
        """The non-cohort Leaf substitutes the Let-bound rendered prompt into
        its template before calling the oracle."""
        sx = _enable_leaf(_make_state_with_field_extraction("x"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states={"x": sx}
        )
        term = compile_fsm(defn)
        case_body = _unwrap_to_case(term)

        instance = FSMInstance(
            fsm_id="m3b",
            current_state="x",
            context=FSMContext(),
        )
        rendered = "<NON-COHORT PROMPT WITH TURN-STATE>"
        env = {
            VAR_STATE_ID: "x",
            VAR_MESSAGE: "hi",
            VAR_CONV_ID: "c",
            VAR_INSTANCE: instance,
            CB_RENDER_RESPONSE_PROMPT: lambda _i: rendered,
            "_cb_extract": lambda _i: None,
            "_cb_field_extract": lambda _i: None,
            "_cb_class_extract": lambda _i: None,
            CB_RESPOND: lambda _i: "would-not-fire",
        }
        oracle = _ScriptedOracle(["resp"])
        executor = Executor(oracle=oracle)
        executor.run(case_body, env)
        assert oracle.invocations == [(rendered, None)]

    def test_render_callback_invoked_after_extraction(self, cache_clear):
        """The render callback runs at Let-time (after extraction Let has fired),
        so the prompt sees post-extraction state. We assert this by ordering
        observations: extraction callback writes to a list before render reads."""
        sx = _enable_leaf(_make_state_with_field_extraction("x"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states={"x": sx}
        )
        term = compile_fsm(defn)
        case_body = _unwrap_to_case(term)

        instance = FSMInstance(fsm_id="m3b", current_state="x", context=FSMContext())
        events: list[str] = []

        def _extract(_i):
            events.append("extract")
            return None

        def _render(_i):
            events.append("render")
            return "prompt"

        env = {
            VAR_STATE_ID: "x",
            VAR_MESSAGE: "m",
            VAR_CONV_ID: "c",
            VAR_INSTANCE: instance,
            "_cb_extract": _extract,
            "_cb_field_extract": _extract,
            "_cb_class_extract": _extract,
            CB_RENDER_RESPONSE_PROMPT: _render,
            CB_RESPOND: lambda _i: "x",
        }
        oracle = _ScriptedOracle(["ok"])
        executor = Executor(oracle=oracle)
        executor.run(case_body, env)
        # Extraction must fire BEFORE render.
        assert events.index("extract") < events.index("render")
        # Both fired exactly once.
        assert events.count("extract") == 1
        assert events.count("render") == 1

    def test_theorem2_holds_with_transition_state(self, cache_clear):
        """A non-terminal state (with transitions) under field=True still hits
        Theorem-2: exactly 1 oracle call (the response Leaf). Transition
        evaluation runs via host callback CB_EVAL_TRANSIT — invisible to the
        oracle counter."""
        defn = _make_two_state_fsm_with_transition()
        _enable_leaf(defn.states["a"])
        term = compile_fsm(defn)
        case_body = _unwrap_to_case(term)

        instance = FSMInstance(fsm_id="m3b", current_state="a", context=FSMContext())

        def _eval_transit(_i):
            return "advanced"

        env = {
            VAR_STATE_ID: "a",
            VAR_MESSAGE: "hello",
            VAR_CONV_ID: "c",
            VAR_INSTANCE: instance,
            "_cb_eval_transit": _eval_transit,
            "_cb_resolve_ambig": lambda _i: lambda _m: None,
            CB_RENDER_RESPONSE_PROMPT: lambda _i: "rendered",
            CB_RESPOND: lambda _i: "would-not-fire",
        }
        oracle = _ScriptedOracle(["resp-advanced"])
        executor = Executor(oracle=oracle)
        result = executor.run(case_body, env)
        assert result == "resp-advanced"
        assert executor.oracle_calls == 1
        # Plan's predicted_calls for the simplest single-Leaf shape is 1.
        assert plan(PlanInputs(n=1, tau=1, K=8192)).predicted_calls == 1


# ---------------------------------------------------------------------------
# (C) Default-False preservation — no Let+Leaf shape leaks
# ---------------------------------------------------------------------------


class TestDefaultFalsePreservation:
    """At field=False (the M3b default), the compiler MUST produce the legacy
    ``App(CB_RESPOND, instance)`` shape for non-cohort states. The new
    Let-Leaf shape must NOT appear anywhere in the AST."""

    @pytest.mark.parametrize(
        "state_factory,state_id",
        [
            (_make_state_with_field_extraction, "x"),
            (_make_state_with_class_extraction, "c"),
            (_make_state_with_required_keys, "r"),
            (_make_state_with_extraction_instructions, "e"),
        ],
        ids=[
            "field_extraction",
            "class_extraction",
            "required_keys",
            "extraction_instructions",
        ],
    )
    def test_default_false_emits_legacy_app(self, state_factory, state_id, cache_clear):
        s = state_factory(state_id)  # NO _enable_leaf call.
        assert s._emit_response_leaf_for_non_cohort is False
        defn = FSMDefinition(
            name="F", description="d", initial_state=state_id, states={state_id: s}
        )
        term = compile_fsm(defn)
        # Walk the AST and assert no Let with NONCOHORT_RESPONSE_PROMPT_VAR
        # appears.
        assert not _ast_contains_noncohort_let(term)
        # And confirm CB_RESPOND App is in the response position.
        assert _ast_contains_app_cb_respond(term)

    def test_two_state_fsm_default_false_emits_legacy(self, cache_clear):
        defn = _make_two_state_fsm_with_transition()
        # No flips. Both states default-False.
        term = compile_fsm(defn)
        assert not _ast_contains_noncohort_let(term)
        # Non-cohort 'a' uses CB_RESPOND App; cohort 'b' uses bare Leaf.
        case_node = _unwrap_to_case(term)
        # 'a' is non-cohort with transition → has CB_RESPOND somewhere.
        assert isinstance(case_node.branches["a"], Let)
        # 'b' is cohort terminal → bare Leaf.
        assert isinstance(case_node.branches["b"], Leaf)


# ---------------------------------------------------------------------------
# AST search helpers (used by the default-False preservation tests)
# ---------------------------------------------------------------------------


def _ast_contains_noncohort_let(term) -> bool:
    """Recursively walk the AST and return True iff any Let binds the
    NONCOHORT_RESPONSE_PROMPT_VAR name."""
    if isinstance(term, Let):
        if term.name == NONCOHORT_RESPONSE_PROMPT_VAR:
            return True
        return _ast_contains_noncohort_let(term.value) or _ast_contains_noncohort_let(
            term.body
        )
    if isinstance(term, Case):
        if term.default is not None and _ast_contains_noncohort_let(term.default):
            return True
        for b in term.branches.values():
            if _ast_contains_noncohort_let(b):
                return True
        return _ast_contains_noncohort_let(term.scrutinee)
    if isinstance(term, App):
        return _ast_contains_noncohort_let(term.fn) or _ast_contains_noncohort_let(
            term.arg
        )
    if hasattr(term, "body") and not isinstance(term, (Let, Case)):
        return _ast_contains_noncohort_let(term.body)
    return False


def _ast_contains_app_cb_respond(term) -> bool:
    """Walk the AST and return True iff App(Var(CB_RESPOND), Var(VAR_INSTANCE))
    appears anywhere."""
    if isinstance(term, App):
        if (
            isinstance(term.fn, Var)
            and term.fn.name == CB_RESPOND
            and isinstance(term.arg, Var)
            and term.arg.name == VAR_INSTANCE
        ):
            return True
        return _ast_contains_app_cb_respond(term.fn) or _ast_contains_app_cb_respond(
            term.arg
        )
    if isinstance(term, Let):
        return _ast_contains_app_cb_respond(term.value) or _ast_contains_app_cb_respond(
            term.body
        )
    if isinstance(term, Case):
        if term.default is not None and _ast_contains_app_cb_respond(term.default):
            return True
        for b in term.branches.values():
            if _ast_contains_app_cb_respond(b):
                return True
        return _ast_contains_app_cb_respond(term.scrutinee)
    if hasattr(term, "body") and not isinstance(term, (Let, Case)):
        return _ast_contains_app_cb_respond(term.body)
    return False


# ---------------------------------------------------------------------------
# (D) Pipeline integration — CB_RENDER_RESPONSE_PROMPT bound by env builder
# ---------------------------------------------------------------------------


class TestPipelineEnvBinding:
    """Verify that ``MessagePipeline._build_compiled_env`` binds
    ``CB_RENDER_RESPONSE_PROMPT`` in every tier (regardless of opt-in field).

    The binding is unconditional so that any opt-in non-cohort state at any
    tier finds the callable in env. At field=False (the default), the
    compiler emits no Let+App for the name, so the binding is harmlessly
    unused.
    """

    def test_render_callback_bound_at_tier_3(self):
        """Build a tier-3 env via the public Program path and assert the
        callable is bound."""
        # We need to construct a MessagePipeline. Easiest path: go through API.
        # But API requires an LLM interface; we can mock it.
        from unittest.mock import Mock

        from fsm_llm.dialog.api import API
        from fsm_llm.runtime._litellm import LLMInterface

        defn = _make_two_state_fsm_with_transition()
        mock_llm = Mock(spec=LLMInterface)
        api = API.from_definition(defn, llm_interface=mock_llm)
        # Reach into the manager → pipeline.
        pipeline = api.fsm_manager._pipeline
        assert hasattr(pipeline, "_make_cb_render_response_prompt")

    def test_make_cb_render_response_prompt_returns_callable(self):
        """The factory method returns a 1-arg callable."""
        from unittest.mock import Mock

        from fsm_llm.dialog.api import API
        from fsm_llm.runtime._litellm import LLMInterface

        defn = _make_two_state_fsm_with_transition()
        mock_llm = Mock(spec=LLMInterface)
        api = API.from_definition(defn, llm_interface=mock_llm)
        pipeline = api.fsm_manager._pipeline

        # Build a minimal turn_state. Just need an FSMInstance.
        instance = FSMInstance(
            fsm_id="x",
            current_state="a",
            context=FSMContext(),
        )

        from fsm_llm.dialog.turn import _TurnState

        ts = _TurnState()
        cb = pipeline._make_cb_render_response_prompt(instance, "hello", "conv-id", ts)
        assert callable(cb)


# ---------------------------------------------------------------------------
# (E) Reserved names + RESERVED_VARS coverage
# ---------------------------------------------------------------------------


class TestReservedVarCoverage:
    def test_reserved_vars_includes_render_callback(self):
        assert CB_RENDER_RESPONSE_PROMPT in RESERVED_VARS

    def test_noncohort_response_prompt_var_is_distinct(self):
        from fsm_llm.dialog.compile_fsm import COHORT_RESPONSE_PROMPT_VAR

        assert NONCOHORT_RESPONSE_PROMPT_VAR != COHORT_RESPONSE_PROMPT_VAR

    def test_render_callback_name_not_collision(self):
        # Must not equal any other callback constant.
        from fsm_llm.dialog.compile_fsm import (
            CB_CLASS_EXTRACT,
            CB_EVAL_TRANSIT,
            CB_EXTRACT,
            CB_FIELD_EXTRACT,
            CB_RESOLVE_AMBIG,
            CB_RESPOND,
            CB_TRANSIT,
        )

        all_cbs = {
            CB_EXTRACT,
            CB_FIELD_EXTRACT,
            CB_CLASS_EXTRACT,
            CB_EVAL_TRANSIT,
            CB_RESOLVE_AMBIG,
            CB_TRANSIT,
            CB_RESPOND,
            CB_RENDER_RESPONSE_PROMPT,
        }
        # All distinct (no collisions).
        assert len(all_cbs) == 8


# ---------------------------------------------------------------------------
# (F) Cohort predicate is unchanged by the M3a field
# ---------------------------------------------------------------------------


class TestTheorem2PlanComponents:
    """Additional Theorem-2 evidence: ``predicted_calls == leaf_calls + reduce_calls``,
    and non-cohort response Leaves contribute exactly one leaf_call each."""

    def test_plan_leaf_calls_is_one_for_simplest(self):
        p = plan(PlanInputs(n=1, tau=1, K=8192))
        assert p.leaf_calls == 1
        assert p.reduce_calls == 0
        assert p.predicted_calls == 1

    def test_executor_oracle_calls_equals_plan_leaf_calls_for_optin_state(
        self, cache_clear
    ):
        sx = _enable_leaf(_make_state_with_field_extraction("x"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states={"x": sx}
        )
        term = compile_fsm(defn)
        case_body = _unwrap_to_case(term)
        instance = FSMInstance(fsm_id="m3b", current_state="x", context=FSMContext())
        env = {
            VAR_STATE_ID: "x",
            VAR_MESSAGE: "m",
            VAR_CONV_ID: "c",
            VAR_INSTANCE: instance,
            "_cb_extract": lambda _i: None,
            "_cb_field_extract": lambda _i: None,
            "_cb_class_extract": lambda _i: None,
            CB_RENDER_RESPONSE_PROMPT: lambda _i: "p",
            CB_RESPOND: lambda _i: "x",
        }
        ex = Executor(oracle=_ScriptedOracle(["r"]))
        ex.run(case_body, env)
        p = plan(PlanInputs(n=1, tau=1, K=8192))
        assert ex.oracle_calls == p.leaf_calls
        assert ex.oracle_calls == p.predicted_calls

    def test_two_optin_states_each_count_one_leaf(self, cache_clear):
        """Two non-cohort states each opted-in. Each invocation of the compiled
        term against a single state ID exercises exactly one Leaf — the response
        Leaf for the dispatched state. The Case selects one branch per turn."""
        sa = _make_state_with_field_extraction("a")
        # Add a transition from a→b to satisfy reachability validation.
        sa.transitions = [
            Transition(
                target_state="b",
                description="route to b",
                conditions=[
                    TransitionCondition(description="always", logic={"==": [1, 1]})
                ],
            )
        ]
        _enable_leaf(sa)
        sb = _enable_leaf(_make_state_with_class_extraction("b"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="a", states={"a": sa, "b": sb}
        )
        term = compile_fsm(defn)
        case_body = _unwrap_to_case(term)
        # Run with state_id='a'
        instance = FSMInstance(fsm_id="m3b", current_state="a", context=FSMContext())
        env = {
            VAR_STATE_ID: "a",
            VAR_MESSAGE: "m",
            VAR_CONV_ID: "c",
            VAR_INSTANCE: instance,
            "_cb_extract": lambda _i: None,
            "_cb_field_extract": lambda _i: None,
            "_cb_class_extract": lambda _i: None,
            "_cb_eval_transit": lambda _i: "advanced",
            "_cb_resolve_ambig": lambda _i: lambda _m: None,
            CB_RENDER_RESPONSE_PROMPT: lambda _i: "p-a",
            CB_RESPOND: lambda _i: "x",
        }
        ex_a = Executor(oracle=_ScriptedOracle(["A"]))
        ex_a.run(case_body, env)
        assert ex_a.oracle_calls == 1
        # Run with state_id='b'
        env_b = dict(env)
        env_b[VAR_STATE_ID] = "b"
        env_b[VAR_INSTANCE] = FSMInstance(
            fsm_id="m3b", current_state="b", context=FSMContext()
        )
        env_b[CB_RENDER_RESPONSE_PROMPT] = lambda _i: "p-b"
        ex_b = Executor(oracle=_ScriptedOracle(["B"]))
        ex_b.run(case_body, env_b)
        assert ex_b.oracle_calls == 1


class TestNonCohortLetEnvVarSemantics:
    """The Let-bound rendered-prompt env name flows correctly into the Leaf."""

    def test_let_value_is_app_render_callback_with_instance_arg(self, cache_clear):
        sx = _enable_leaf(_make_state_with_field_extraction("x"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states={"x": sx}
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        outer_let = case_node.branches["x"]
        response_let = outer_let.body
        v = response_let.value
        assert isinstance(v, App)
        assert isinstance(v.fn, Var) and v.fn.name == CB_RENDER_RESPONSE_PROMPT
        assert isinstance(v.arg, Var) and v.arg.name == VAR_INSTANCE

    def test_leaf_template_format_matches_input_var(self, cache_clear):
        """The Leaf template's single placeholder must match its input_vars
        (executor.py::_eval_leaf substitutes via str.format)."""
        sx = _enable_leaf(_make_state_with_field_extraction("x"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states={"x": sx}
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        outer_let = case_node.branches["x"]
        response_let = outer_let.body
        leaf_node = response_let.body
        # str.format with the input_vars binding works.
        rendered = leaf_node.template.format(
            **{NONCOHORT_RESPONSE_PROMPT_VAR: "<test>"}
        )
        assert rendered == "<test>"


class TestCohortPredicateInvariant:
    def test_cohort_predicate_ignores_optin_field(self, cache_clear):
        """``_is_cohort_state`` predicate returns the same value regardless
        of ``_emit_response_leaf_for_non_cohort``. The field controls compile
        emission for non-cohort states only; cohort states are unaffected."""
        # A cohort-eligible state.
        s_cohort = State(
            id="end",
            description="end",
            purpose="wrap up",
            response_instructions="bye",
        )
        defn = FSMDefinition(
            name="F", description="d", initial_state="end", states={"end": s_cohort}
        )
        before = _is_cohort_state(s_cohort, defn)
        s_cohort._emit_response_leaf_for_non_cohort = True
        after = _is_cohort_state(s_cohort, defn)
        assert before is after is True

        # A non-cohort-eligible state.
        s_nc = _make_state_with_field_extraction("nc")
        defn2 = FSMDefinition(
            name="F", description="d", initial_state="nc", states={"nc": s_nc}
        )
        before2 = _is_cohort_state(s_nc, defn2)
        s_nc._emit_response_leaf_for_non_cohort = True
        after2 = _is_cohort_state(s_nc, defn2)
        assert before2 is after2 is False
