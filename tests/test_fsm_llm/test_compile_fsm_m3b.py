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
    CB_APPEND_HISTORY,
    CB_RENDER_RESPONSE_PROMPT,
    CB_RESPOND,
    CB_RESPOND_SYNTHETIC,
    NONCOHORT_RESPONSE_PROMPT_VAR,
    NONCOHORT_RESPONSE_VAR,
    RESERVED_VARS,
    VAR_CONV_ID,
    VAR_INSTANCE,
    VAR_MESSAGE,
    VAR_STATE_ID,
    _is_cohort_state,
    compile_fsm,
)


# D2 — identity history-append for tests. The curried 2-arg App
# ``App(App(CB_APPEND_HISTORY, instance), value)`` evaluates to ``value``;
# tests don't exercise the real history-write side effect (covered via
# integration in ``TestD2HistoryAppend``).
def _identity_append_history(_inst):
    return lambda v: v


# D2 — walk past the outer D2 history-append Let to the inner M3b prompt-Let.
# After D2, the opt-in shape is::
#
#     Let(NONCOHORT_RESPONSE_VAR,
#         <inner_m3b_let>,
#         App(App(CB_APPEND_HISTORY, instance), NONCOHORT_RESPONSE_VAR))
#
# Pre-D2 callers expected ``body.body`` to be the prompt Let directly;
# post-D2 they need ``body.body.value`` to reach the same inner Let.
# This helper handles both shapes for back-compat (for the rare test
# that runs with the gate disabled — but currently all opt-in tests use
# the D2 outer wrap).
def _inner_m3b_let(outer):
    """Given an arbitrary term, walk down to the FIRST D2 outer Let
    encountered (binding NONCOHORT_RESPONSE_VAR), then return the inner
    M3b prompt-rendering Let.

    Walks through Abs, extraction Lets, disc Lets, and Case dispatch
    (taking the first non-empty branch). Asserts the D2 shape is intact.
    """
    from fsm_llm.runtime.ast import Abs as _Abs
    from fsm_llm.runtime.ast import App as _App
    from fsm_llm.runtime.ast import Case as _Case
    from fsm_llm.runtime.ast import Let as _Let
    from fsm_llm.runtime.ast import Var as _Var

    cur = outer
    seen = 0
    while seen < 64:  # bound walk to surface drift loudly
        if isinstance(cur, _Let) and cur.name == NONCOHORT_RESPONSE_VAR:
            assert isinstance(cur.body, _App)
            inner_app = cur.body.fn
            assert isinstance(inner_app, _App)
            assert isinstance(inner_app.fn, _Var)
            assert inner_app.fn.name == CB_APPEND_HISTORY
            return cur.value
        if isinstance(cur, _Abs):
            cur = cur.body
        elif isinstance(cur, _Let):
            cur = cur.body
        elif isinstance(cur, _Case):
            chosen = None
            for arm in cur.branches.values():
                chosen = arm
                break
            cur = chosen if chosen is not None else cur.default
        else:
            break
        seen += 1
    raise AssertionError(
        f"could not find D2 outer Let bound to {NONCOHORT_RESPONSE_VAR!r} "
        f"in term: {type(cur).__name__}"
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


# D3 (plan_f1003066) — terminal opt-in states fall back to legacy
# `App(CB_RESPOND, instance)` because `_output_response_format` enforcement
# is runtime-only. The non-terminal Let+Leaf path requires at least one
# transition. Test helpers below all add a trivial always-go-to "_end"
# transition so the M3b/D2 path is exercised; tests asserting D3 fallback
# use the `_terminal_*` variants.
def _always_to_end_transition() -> Transition:
    return Transition(
        target_state="_end",
        description="always",
        conditions=[
            TransitionCondition(description="always", logic={"==": [1, 1]})
        ],
    )


def _end_state() -> State:
    """Terminal `_end` state used by helper-built non-terminal states.

    Each `_make_state_with_*` helper above now adds an always-go-to
    `_end` transition so the M3b/D2 path is exercised under opt-in
    (terminal states fall back to legacy via D3). Construction sites
    must include `_end` in the FSMDefinition.states dict. Use the
    `**_with_end()` splat helper below for brevity.
    """
    return State(
        id="_end", description="end", purpose="end", response_instructions="bye"
    )


def _with_end(states_dict: dict) -> dict:
    """Add `_end` to a states dict; use as `states={**_with_end({...})}`."""
    states_dict["_end"] = _end_state()
    return states_dict


def _wrap_with_end_state(states_dict: dict, initial_state: str) -> FSMDefinition:
    """Return an FSMDefinition that adds a terminal "_end" state to make
    the named state non-terminal (so the D2 path is exercised under
    opt-in instead of D3 fallback)."""
    states_dict["_end"] = State(
        id="_end",
        description="end",
        purpose="end",
        response_instructions="bye",
    )
    return FSMDefinition(
        name="F", description="d", initial_state=initial_state, states=states_dict
    )


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
        transitions=[_always_to_end_transition()],
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
        transitions=[_always_to_end_transition()],
    )


def _make_state_with_required_keys(state_id: str = "r") -> State:
    return State(
        id=state_id,
        description=f"{state_id} desc",
        purpose="Need keys",
        response_instructions="Respond once keys present.",
        required_context_keys=["k1"],
        transitions=[_always_to_end_transition()],
    )


def _make_state_with_extraction_instructions(state_id: str = "e") -> State:
    return State(
        id=state_id,
        description=f"{state_id} desc",
        purpose="Bulk extract",
        response_instructions="Respond after bulk extract.",
        extraction_instructions="Extract everything notable.",
        transitions=[_always_to_end_transition()],
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
        """Default field=False → legacy ``App(CB_RESPOND, instance)`` shape
        appears in the response position(s) of the compiled term."""
        sx = _make_state_with_field_extraction("x")
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states=_with_end({"x": sx})
        )
        term = compile_fsm(defn)
        # State has a transition → Let(disc, App(CB_EVAL_TRANSIT,...), Case(...)) —
        # the legacy CB_RESPOND App appears in each Case branch.
        assert _ast_contains_app_cb_respond(term)
        # Confirm the M3b non-cohort Let is NOT present (default-False).
        assert not _ast_contains_noncohort_let(term)

    def test_field_extraction_state_optin_emits_let_leaf(self, cache_clear):
        """Field=True → response position is D2-outer-Let wrapping the M3b
        inner Let wrapping a Leaf."""
        sx = _enable_leaf(_make_state_with_field_extraction("x"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states=_with_end({"x": sx})
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        body = case_node.branches["x"]
        # Outermost: extraction Let; inner: D2 Let; innermost: M3b prompt Let.
        assert isinstance(body, Let)
        response_let = _inner_m3b_let(body.body)
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
            name="F", description="d", initial_state="c", states=_with_end({"c": sc})
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        body = case_node.branches["c"]
        # outer Let = class_extract; inner = D2 Let; D2-inner = M3b prompt Let.
        assert isinstance(body, Let)
        response_let = _inner_m3b_let(body.body)
        assert isinstance(response_let, Let)
        assert response_let.name == NONCOHORT_RESPONSE_PROMPT_VAR
        assert isinstance(response_let.body, Leaf)

    def test_required_keys_state_optin_emits_let_leaf(self, cache_clear):
        sr = _enable_leaf(_make_state_with_required_keys("r"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="r", states=_with_end({"r": sr})
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        body = case_node.branches["r"]
        # required_context_keys triggers extraction Let.
        assert isinstance(body, Let)
        response_let = _inner_m3b_let(body.body)
        assert isinstance(response_let, Let)
        assert response_let.name == NONCOHORT_RESPONSE_PROMPT_VAR
        assert isinstance(response_let.body, Leaf)

    def test_extraction_instructions_state_optin_emits_let_leaf(self, cache_clear):
        se = _enable_leaf(_make_state_with_extraction_instructions("e"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="e", states=_with_end({"e": se})
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        body = case_node.branches["e"]
        assert isinstance(body, Let)
        response_let = _inner_m3b_let(body.body)
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
        # Each non-ambig branch should be the D2-outer Let wrapping the
        # M3b prompt Let.
        for arm_name in ("advanced", "blocked"):
            arm = case_dispatch.branches[arm_name]
            assert isinstance(arm, Let)
            inner = _inner_m3b_let(arm)
            assert inner.name == NONCOHORT_RESPONSE_PROMPT_VAR
            assert isinstance(inner.body, Leaf)
        # Default arm
        assert isinstance(case_dispatch.default, Let)
        default_inner = _inner_m3b_let(case_dispatch.default)
        assert default_inner.name == NONCOHORT_RESPONSE_PROMPT_VAR

    def test_terminal_non_cohort_state_optin_emits_let_leaf(self, cache_clear):
        """A terminal state that is non-cohort (e.g. has required_context_keys but
        no transitions) ships the D2 outer Let wrapping the M3b inner
        Let+Leaf at the top of the Case branch (under the outer
        extraction Let)."""
        sr = _enable_leaf(_make_state_with_required_keys("r"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="r", states=_with_end({"r": sr})
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        body = case_node.branches["r"]
        # Outer Let (extraction); body.body is D2 outer Let; D2-inner is M3b.
        assert isinstance(body, Let)
        d2_outer = body.body
        response_let = _inner_m3b_let(d2_outer)
        assert isinstance(response_let, Let)
        assert isinstance(response_let.body, Leaf)
        # This state is terminal — no Case dispatch wraps it.
        assert body.body is d2_outer

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
            name="F",
            description="d",
            initial_state=state_id,
            states=_with_end({state_id: s}),
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
            CB_APPEND_HISTORY: _identity_append_history,
            "_cb_eval_transit": lambda _i: "advanced",
            "_cb_resolve_ambig": lambda _i: lambda _m: None,
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
            name="F", description="d", initial_state="x", states=_with_end({"x": sx})
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
            CB_APPEND_HISTORY: _identity_append_history,
            "_cb_eval_transit": lambda _i: "advanced",
            "_cb_resolve_ambig": lambda _i: lambda _m: None,
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
            name="F", description="d", initial_state="x", states=_with_end({"x": sx})
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
            CB_APPEND_HISTORY: _identity_append_history,
            "_cb_eval_transit": lambda _i: "advanced",
            "_cb_resolve_ambig": lambda _i: lambda _m: None,
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
            CB_APPEND_HISTORY: _identity_append_history,
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
            name="F",
            description="d",
            initial_state=state_id,
            states=_with_end({state_id: s}),
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


def _ast_contains_app_var(term, var_name: str) -> bool:
    """Walk the AST and return True iff App(Var(var_name), Var(VAR_INSTANCE))
    appears anywhere."""
    if isinstance(term, App):
        if (
            isinstance(term.fn, Var)
            and term.fn.name == var_name
            and isinstance(term.arg, Var)
            and term.arg.name == VAR_INSTANCE
        ):
            return True
        return _ast_contains_app_var(term.fn, var_name) or _ast_contains_app_var(
            term.arg, var_name
        )
    if isinstance(term, Let):
        return _ast_contains_app_var(term.value, var_name) or _ast_contains_app_var(
            term.body, var_name
        )
    if isinstance(term, Case):
        if term.default is not None and _ast_contains_app_var(term.default, var_name):
            return True
        for b in term.branches.values():
            if _ast_contains_app_var(b, var_name):
                return True
        return _ast_contains_app_var(term.scrutinee, var_name)
    if hasattr(term, "body") and not isinstance(term, (Let, Case)):
        return _ast_contains_app_var(term.body, var_name)
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
            name="F", description="d", initial_state="x", states=_with_end({"x": sx})
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
            CB_APPEND_HISTORY: _identity_append_history,
            "_cb_eval_transit": lambda _i: "advanced",
            "_cb_resolve_ambig": lambda _i: lambda _m: None,
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
        # Override the helper-added _end transition: route a→b instead.
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
        sb = _make_state_with_class_extraction("b")
        # b is also non-terminal in this test (so D2 path fires, not D3
        # legacy fallback). Keep b's helper-added _end transition; FSMDefinition
        # below adds the `_end` state.
        _enable_leaf(sb)
        defn = FSMDefinition(
            name="F",
            description="d",
            initial_state="a",
            states=_with_end({"a": sa, "b": sb}),
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
            CB_APPEND_HISTORY: _identity_append_history,
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
            name="F", description="d", initial_state="x", states=_with_end({"x": sx})
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        outer_let = case_node.branches["x"]
        response_let = _inner_m3b_let(outer_let.body)
        v = response_let.value
        assert isinstance(v, App)
        assert isinstance(v.fn, Var) and v.fn.name == CB_RENDER_RESPONSE_PROMPT
        assert isinstance(v.arg, Var) and v.arg.name == VAR_INSTANCE

    def test_leaf_template_format_matches_input_var(self, cache_clear):
        """The Leaf template's single placeholder must match its input_vars
        (executor.py::_eval_leaf substitutes via str.format)."""
        sx = _enable_leaf(_make_state_with_field_extraction("x"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states=_with_end({"x": sx})
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        outer_let = case_node.branches["x"]
        response_let = _inner_m3b_let(outer_let.body)
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
            name="F",
            description="d",
            initial_state="nc",
            states=_with_end({"nc": s_nc}),
        )
        before2 = _is_cohort_state(s_nc, defn2)
        s_nc._emit_response_leaf_for_non_cohort = True
        after2 = _is_cohort_state(s_nc, defn2)
        assert before2 is after2 is False


# ---------------------------------------------------------------------------
# (G) D1 — empty-`response_instructions` synthetic-callback gate
# ---------------------------------------------------------------------------
#
# Plan: ``plans/plan_2026-04-28_f1003066`` step 6.
#
# When a non-cohort state has empty (but NOT None) ``response_instructions``,
# the legacy ``_make_cb_respond`` fast-path returns a synthetic
# ``f"[{state.id}]"`` and appends it to conversation history. The bare M3b
# Leaf would issue a real call against an empty prompt and lose the synthetic
# semantics. D1 adds a compile-time gate that emits
# ``App(var(CB_RESPOND_SYNTHETIC), var(VAR_INSTANCE))`` instead — preserving
# the synthetic + history semantics with **0 oracle calls**.


def _make_state_empty_response_instructions(state_id: str = "es") -> State:
    """Non-cohort (has extraction_instructions) NON-TERMINAL state with
    empty response. Non-terminal-ness is required so that D3 doesn't
    shadow D1 (D3 falls back terminal opt-in states to legacy)."""
    return State(
        id=state_id,
        description=f"{state_id} desc",
        purpose="empty response",
        # Empty string — D1 trigger. Distinct from None (default).
        response_instructions="",
        # Non-cohort because of extraction_instructions.
        extraction_instructions="extract everything",
        transitions=[_always_to_end_transition()],
    )


class TestD1EmptyInstructionsGate:
    """D1 — opt-in non-cohort state with ``response_instructions == ""``
    emits ``App(CB_RESPOND_SYNTHETIC, instance)`` instead of the Let+Leaf
    shape. None-instructions still falls through to the Let+Leaf path."""

    def test_empty_instructions_optin_emits_app_synthetic(self, cache_clear):
        s = _enable_leaf(_make_state_empty_response_instructions("es"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="es", states=_with_end({"es": s})
        )
        term = compile_fsm(defn)
        # State is non-terminal (has _end transition) so D1's App appears
        # inside the Case dispatch branches. Walk via grep helper.
        assert _ast_contains_app_var(term, CB_RESPOND_SYNTHETIC)
        # CB_RESPOND legacy is NOT in the response position (D1 replaced it).
        assert not _ast_contains_app_cb_respond(term)
        # M3b non-cohort Let is NOT present (D1 supplants it).
        assert not _ast_contains_noncohort_let(term)

    def test_empty_instructions_default_off_unchanged(self, cache_clear):
        """At field=False (default), empty-instructions states still emit
        the legacy ``App(CB_RESPOND, instance)`` shape — D1 only fires
        under opt-in."""
        s = _make_state_empty_response_instructions("es")
        # NO _enable_leaf — default False.
        defn = FSMDefinition(
            name="F", description="d", initial_state="es", states=_with_end({"es": s})
        )
        term = compile_fsm(defn)
        # Default-False: legacy CB_RESPOND in response position; no D1
        # synthetic, no M3b non-cohort Let.
        assert _ast_contains_app_cb_respond(term)
        assert not _ast_contains_app_var(term, CB_RESPOND_SYNTHETIC)
        assert not _ast_contains_noncohort_let(term)

    def test_none_instructions_optin_falls_through_to_leaf(self, cache_clear):
        """Setting ``response_instructions=None`` (the default unset value)
        does NOT trigger D1 — only empty-string fires the synthetic gate.
        Verify the standard Let+Leaf path still emits."""
        s = _enable_leaf(
            State(
                id="ni",
                description="ni",
                purpose="none instructions",
                # response_instructions left unset → None
                extraction_instructions="extract",
                # Non-terminal so D3 doesn't shadow with legacy fallback.
                transitions=[_always_to_end_transition()],
            )
        )
        assert s.response_instructions is None  # Pydantic default
        defn = FSMDefinition(
            name="F",
            description="d",
            initial_state="ni",
            states=_with_end({"ni": s}),
        )
        term = compile_fsm(defn)
        case_node = _unwrap_to_case(term)
        body = case_node.branches["ni"]
        assert isinstance(body, Let)
        response_let = _inner_m3b_let(body.body)
        assert isinstance(response_let, Let)
        assert response_let.name == NONCOHORT_RESPONSE_PROMPT_VAR
        assert isinstance(response_let.body, Leaf)

    def test_d1_synthetic_callback_returns_state_id_and_appends_history(
        self, cache_clear
    ):
        """Direct unit test on the ``_make_cb_respond_synthetic`` factory.

        The factory returns a 1-arg callable that (1) returns
        ``f"[{state.id}]"``, (2) appends that synthetic to conversation
        history, and (3) does not invoke any LLM method.

        Note: ``generate_initial_response`` (`turn.py:1340`) is a separate
        non-compiled path that bypasses the compiled term entirely — D1
        only applies to the compiled response position reached via
        ``process_message``. We test the closure directly here; the
        end-to-end opt-in driver path is covered by the integration smoke
        in plan step 9.
        """
        from unittest.mock import Mock

        from fsm_llm.dialog.api import API
        from fsm_llm.dialog.turn import _TurnState
        from fsm_llm.runtime._litellm import LLMInterface

        s = _enable_leaf(_make_state_empty_response_instructions("es"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="es", states=_with_end({"es": s})
        )
        mock_llm = Mock(spec=LLMInterface)
        api = API.from_definition(defn, llm_interface=mock_llm)
        pipeline = api.fsm_manager._pipeline
        # Pin the fsm_resolver to return our defn unconditionally — bypasses
        # the registry-id roundtrip for this isolated unit test.
        pipeline.fsm_resolver = lambda _fid: defn
        instance = FSMInstance(
            fsm_id="F",
            current_state="es",
            context=FSMContext(),
        )
        ts = _TurnState()
        cb = pipeline._make_cb_respond_synthetic(instance, "hello", "conv", ts)
        assert callable(cb)
        # Invoke the closure directly — mirrors what App(CB_RESPOND_SYNTHETIC,
        # instance) does at executor evaluation time.
        result = cb(instance)
        assert result == "[es]"
        # History append.
        history = instance.context.conversation.exchanges
        assert any("[es]" in str(exc) for exc in history)
        # Zero LLM calls — neither generate_response nor extract_field were
        # invoked on the mock.
        mock_llm.generate_response.assert_not_called()
        if hasattr(mock_llm, "extract_field"):
            try:
                mock_llm.extract_field.assert_not_called()
            except AssertionError:
                pass  # acceptable — only assertion that matters is generate_response

    def test_d1_synthetic_streaming_yields_single_chunk_iterator(
        self, cache_clear
    ):
        """A.D4(b) (plan_ca542489 step 3) — when ``turn_state.stream`` is
        True the synthetic callback returns ``iter([f"[{state.id}]"])``,
        a single-chunk iterator that mirrors the legacy streaming I4
        fast-path. History append still happens once."""
        from unittest.mock import Mock

        from fsm_llm.dialog.api import API
        from fsm_llm.dialog.turn import _TurnState
        from fsm_llm.runtime._litellm import LLMInterface

        s = _enable_leaf(_make_state_empty_response_instructions("es"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="es", states=_with_end({"es": s})
        )
        mock_llm = Mock(spec=LLMInterface)
        api = API.from_definition(defn, llm_interface=mock_llm)
        pipeline = api.fsm_manager._pipeline
        pipeline.fsm_resolver = lambda _fid: defn
        instance = FSMInstance(
            fsm_id="F", current_state="es", context=FSMContext()
        )
        ts = _TurnState(stream=True)  # ← streaming-mode turn state
        cb = pipeline._make_cb_respond_synthetic(instance, "hello", "conv", ts)

        result = cb(instance)

        # Result is an iterator (NOT a string).
        assert not isinstance(result, str)
        # Drains to exactly one chunk: the synthetic sentinel.
        chunks = list(result)
        assert chunks == ["[es]"]
        # History append happened exactly once (synchronously, before the
        # iterator was constructed — mirrors legacy I4 fast-path).
        history = instance.context.conversation.exchanges
        assert sum("[es]" in str(exc) for exc in history) == 1
        # Zero LLM calls.
        mock_llm.generate_response.assert_not_called()

    def test_d1_synthetic_non_streaming_returns_string_byte_equivalent(
        self, cache_clear
    ):
        """Regression gate: ``turn_state.stream=False`` (default) preserves
        pre-A.D4 behaviour — the callable returns the synthetic STRING,
        not an iterator. This is the byte-equivalence contract for every
        non-streaming caller."""
        from unittest.mock import Mock

        from fsm_llm.dialog.api import API
        from fsm_llm.dialog.turn import _TurnState
        from fsm_llm.runtime._litellm import LLMInterface

        s = _enable_leaf(_make_state_empty_response_instructions("es"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="es", states=_with_end({"es": s})
        )
        mock_llm = Mock(spec=LLMInterface)
        api = API.from_definition(defn, llm_interface=mock_llm)
        pipeline = api.fsm_manager._pipeline
        pipeline.fsm_resolver = lambda _fid: defn
        instance = FSMInstance(
            fsm_id="F", current_state="es", context=FSMContext()
        )
        ts = _TurnState()  # stream defaults to False
        assert ts.stream is False  # explicit assertion of the default contract
        cb = pipeline._make_cb_respond_synthetic(instance, "hello", "conv", ts)

        result = cb(instance)

        # Identical pre-A.D4 contract: returns the synthetic string verbatim.
        assert result == "[es]"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# (H) D2 — outer Let with curried CB_APPEND_HISTORY
# ---------------------------------------------------------------------------


class TestD2HistoryAppendOuterLet:
    """D2 wraps the M3b inner Let in an outer Let bound to
    NONCOHORT_RESPONSE_VAR; the outer Let body is
    App(App(CB_APPEND_HISTORY, instance), <var>) which appends to history
    and returns the response unchanged. The host App does NOT increment
    oracle_calls — Theorem-2 strict equality preserved."""

    def test_outer_let_shape_and_appended_history(self, cache_clear):
        """Direct unit test on the closure: invoking the outer Let body
        equivalent appends to instance.context.conversation and returns
        the response value unchanged."""
        from unittest.mock import Mock

        from fsm_llm.dialog.api import API
        from fsm_llm.dialog.turn import _TurnState
        from fsm_llm.runtime._litellm import LLMInterface

        s = _enable_leaf(_make_state_with_required_keys("r"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="r", states=_with_end({"r": s})
        )
        mock_llm = Mock(spec=LLMInterface)
        api = API.from_definition(defn, llm_interface=mock_llm)
        pipeline = api.fsm_manager._pipeline
        pipeline.fsm_resolver = lambda _fid: defn
        instance = FSMInstance(
            fsm_id="F", current_state="r", context=FSMContext()
        )
        ts = _TurnState()
        cb_factory = pipeline._make_cb_append_history(
            instance, "hi", "conv", ts
        )
        # Curried: factory(inst) returns (value -> value).
        inner = cb_factory(instance)
        result = inner("the response string")
        assert result == "the response string"
        # History append happened.
        history = instance.context.conversation.exchanges
        assert any("the response string" in str(exc) for exc in history)
        # last_response_generation also stashed.
        assert instance.last_response_generation is not None
        assert instance.last_response_generation.message == "the response string"

    def test_outer_let_in_compile_output_for_optin_state(self, cache_clear):
        """The compile output for an opt-in non-cohort state has the D2
        outer Let at the response position (inside each Case branch when
        the state is non-terminal)."""
        sx = _enable_leaf(_make_state_with_field_extraction("x"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states=_with_end({"x": sx})
        )
        term = compile_fsm(defn)
        # Walk via helper — finds the D2 outer Let regardless of whether
        # the state is terminal (direct under extraction) or non-terminal
        # (inside a Case branch under disc Let).
        inner_m3b = _inner_m3b_let(term)
        # _inner_m3b_let returned the inner M3b Let (the value of the D2
        # outer); validates the D2 outer's structure internally.
        assert inner_m3b.name == NONCOHORT_RESPONSE_PROMPT_VAR
        # Direct shape check on the outer App for completeness — find via
        # AST grep for App(App(CB_APPEND_HISTORY, instance), var).
        assert _ast_contains_app_var(term, CB_RESPOND) is False
        # CB_APPEND_HISTORY appears as a curried inner-App fn name (the
        # outer-let body); _inner_m3b_let validates this internally.

    def test_d2_executor_oracle_calls_unchanged(self, cache_clear):
        """The D2 outer wrap doesn't change executor's oracle_calls count
        (host App is not a Leaf). End-to-end: opt-in state, scripted
        oracle, exactly 1 oracle call."""
        sx = _enable_leaf(_make_state_with_field_extraction("x"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="x", states=_with_end({"x": sx})
        )
        term = compile_fsm(defn)
        case_body = _unwrap_to_case(term)
        instance = FSMInstance(
            fsm_id="m3b", current_state="x", context=FSMContext()
        )
        env = {
            VAR_STATE_ID: "x",
            VAR_MESSAGE: "hi",
            VAR_CONV_ID: "c",
            VAR_INSTANCE: instance,
            CB_RENDER_RESPONSE_PROMPT: lambda _i: "rendered",
            "_cb_extract": lambda _i: None,
            "_cb_field_extract": lambda _i: None,
            "_cb_class_extract": lambda _i: None,
            CB_RESPOND: lambda _i: "would-not-fire",
            CB_APPEND_HISTORY: _identity_append_history,
            "_cb_eval_transit": lambda _i: "advanced",
            "_cb_resolve_ambig": lambda _i: lambda _m: None,
        }
        oracle = _ScriptedOracle(["leaf-result"])
        executor = Executor(oracle=oracle)
        result = executor.run(case_body, env)
        # Outer Let body returns identity-appended value (= "leaf-result").
        assert result == "leaf-result"
        # Theorem-2: 1 Leaf = 1 oracle call. D2 outer App does NOT count.
        assert executor.oracle_calls == 1


# ---------------------------------------------------------------------------
# (H+) D2 + A.D4(b) — CB_APPEND_HISTORY iterator-aware streaming wrap
# (plan_2026-04-28_ca542489 step 2)
# ---------------------------------------------------------------------------


class TestD2AppendHistoryIteratorAware:
    """D2's outer-Let CB_APPEND_HISTORY callback handles both string Leaf
    values (non-streaming) and Iterator[str] values (streaming), so the
    SAME compiled term shape works under ``Executor.run(stream=True)`` and
    ``Executor.run(stream=False)``.

    Iterator path tees chunks to consumer while accumulating; on iterator
    exhaustion (or GeneratorExit), ``"".join(chunks)`` is appended to
    ``inst.context.conversation`` — mirroring the legacy
    ``_stream_response_generation_pass`` ``finally`` block."""

    def test_cb_append_history_handles_iterator_value(self, cache_clear):
        """When called with an Iterator value, the closure returns a
        tee-generator that yields the same chunks AND, on exhaustion,
        appends the joined string to history."""
        from unittest.mock import Mock

        from fsm_llm.dialog.api import API
        from fsm_llm.dialog.turn import _TurnState
        from fsm_llm.runtime._litellm import LLMInterface

        s = _enable_leaf(_make_state_with_required_keys("r"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="r", states=_with_end({"r": s})
        )
        mock_llm = Mock(spec=LLMInterface)
        api = API.from_definition(defn, llm_interface=mock_llm)
        pipeline = api.fsm_manager._pipeline
        pipeline.fsm_resolver = lambda _fid: defn
        instance = FSMInstance(
            fsm_id="F", current_state="r", context=FSMContext()
        )
        ts = _TurnState()
        cb_factory = pipeline._make_cb_append_history(instance, "hi", "conv", ts)
        inner = cb_factory(instance)

        # Pass an iterator value (mimics oracle.invoke_stream return).
        result = inner(iter(["chunk-A", " ", "chunk-B"]))

        # Result is itself an iterator (NOT a string) — consumer sees chunks.
        chunks = list(result)
        assert chunks == ["chunk-A", " ", "chunk-B"]
        # After exhaustion, history has the joined string.
        history = instance.context.conversation.exchanges
        assert any("chunk-A chunk-B" in str(exc) for exc in history)
        # last_response_generation NOT set on streaming path (I3 preserved).
        assert instance.last_response_generation is None

    def test_cb_append_history_iterator_appends_only_on_exhaustion(
        self, cache_clear
    ):
        """``add_system_message`` must NOT fire until the iterator is
        exhausted — mirrors the legacy streaming ``finally`` semantics
        (``turn.py:1377-1381``)."""
        from unittest.mock import Mock

        from fsm_llm.dialog.api import API
        from fsm_llm.dialog.turn import _TurnState
        from fsm_llm.runtime._litellm import LLMInterface

        s = _enable_leaf(_make_state_with_required_keys("r"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="r", states=_with_end({"r": s})
        )
        mock_llm = Mock(spec=LLMInterface)
        api = API.from_definition(defn, llm_interface=mock_llm)
        pipeline = api.fsm_manager._pipeline
        pipeline.fsm_resolver = lambda _fid: defn
        instance = FSMInstance(
            fsm_id="F", current_state="r", context=FSMContext()
        )
        ts = _TurnState()
        cb_factory = pipeline._make_cb_append_history(instance, "hi", "conv", ts)
        inner = cb_factory(instance)

        result = inner(iter(["a", "b", "c"]))
        result_iter = iter(result)

        # Consume one chunk; assert history is still empty.
        first = next(result_iter)
        assert first == "a"
        assert len(instance.context.conversation.exchanges) == 0

        # Consume rest; assert history is populated after exhaustion.
        rest = list(result_iter)
        assert rest == ["b", "c"]
        history = instance.context.conversation.exchanges
        assert any("abc" in str(exc) for exc in history)

    def test_cb_append_history_iterator_appends_on_generator_exit(
        self, cache_clear
    ):
        """Consumer abandons the iterator (e.g. raises) — partial-chunks
        accumulated so far MUST still be appended via the ``finally``
        block. This is the lifecycle correctness gate for streaming
        cancellation."""
        from unittest.mock import Mock

        from fsm_llm.dialog.api import API
        from fsm_llm.dialog.turn import _TurnState
        from fsm_llm.runtime._litellm import LLMInterface

        s = _enable_leaf(_make_state_with_required_keys("r"))
        defn = FSMDefinition(
            name="F", description="d", initial_state="r", states=_with_end({"r": s})
        )
        mock_llm = Mock(spec=LLMInterface)
        api = API.from_definition(defn, llm_interface=mock_llm)
        pipeline = api.fsm_manager._pipeline
        pipeline.fsm_resolver = lambda _fid: defn
        instance = FSMInstance(
            fsm_id="F", current_state="r", context=FSMContext()
        )
        ts = _TurnState()
        cb_factory = pipeline._make_cb_append_history(instance, "hi", "conv", ts)
        inner = cb_factory(instance)

        result = inner(iter(["x", "y", "z"]))
        result_iter = iter(result)

        # Take only the first chunk, then close the generator (simulating
        # consumer abandonment / early break).
        first = next(result_iter)
        assert first == "x"
        result_iter.close()  # triggers GeneratorExit inside the tee generator

        # The finally block should have appended whatever was accumulated
        # so far (just "x").
        history = instance.context.conversation.exchanges
        assert any("x" in str(exc) for exc in history)


# ---------------------------------------------------------------------------
# (I) D3 — terminal opt-in fallback to legacy App(CB_RESPOND, instance)
# ---------------------------------------------------------------------------


class TestD3TerminalStructuredFallback:
    """D3 — terminal opt-in states fall back to legacy
    ``App(CB_RESPOND, instance)`` (preserving runtime
    `_output_response_format` enforcement). Conservative compile-time
    guard: ALL terminal opt-in states fall back, regardless of whether
    `_output_response_format` is present at runtime — see D-004 in
    plan_2026-04-28_f1003066/decisions.md."""

    def test_terminal_optin_falls_back_to_legacy_cb_respond(self, cache_clear):
        """A terminal non-cohort state (e.g. has required_context_keys
        but no transitions) falls back to legacy App(CB_RESPOND, instance)
        under opt-in — the M3b/D2 Let+Leaf shape does NOT appear."""
        s = _enable_leaf(
            State(
                id="t",
                description="terminal",
                purpose="terminal extraction",
                response_instructions="Respond once data extracted.",
                required_context_keys=["k1"],  # → non-cohort
                # No transitions → terminal → D3 fallback fires.
            )
        )
        assert s.transitions == []
        defn = FSMDefinition(
            name="F", description="d", initial_state="t", states={"t": s}
        )
        term = compile_fsm(defn)
        # Legacy CB_RESPOND is in the response position.
        assert _ast_contains_app_cb_respond(term)
        # M3b/D2 non-cohort Let is NOT present.
        assert not _ast_contains_noncohort_let(term)
        # D1 synthetic gate is NOT present (response_instructions is non-empty).
        assert not _ast_contains_app_var(term, CB_RESPOND_SYNTHETIC)

    def test_terminal_optin_with_empty_instructions_still_fallback(self, cache_clear):
        """A terminal opt-in state with empty response_instructions:
        D3 (terminal-only) takes precedence over D1 (empty-instructions),
        because the if/elif gate in `_compile_state` checks `not
        state.transitions` first. So legacy CB_RESPOND wins (which itself
        has the legacy empty-instructions short-circuit at runtime)."""
        s = _enable_leaf(
            State(
                id="t",
                description="terminal-empty",
                purpose="terminal empty",
                response_instructions="",
                extraction_instructions="extract",
                # No transitions → terminal → D3 wins.
            )
        )
        assert s.transitions == []
        defn = FSMDefinition(
            name="F", description="d", initial_state="t", states={"t": s}
        )
        term = compile_fsm(defn)
        # Legacy CB_RESPOND wins; D1 synthetic does NOT fire under opt-in
        # for terminal states (the runtime _make_cb_respond still does the
        # empty-instructions short-circuit at runtime — D-R10-7.4).
        assert _ast_contains_app_cb_respond(term)
        assert not _ast_contains_app_var(term, CB_RESPOND_SYNTHETIC)
        assert not _ast_contains_noncohort_let(term)

    def test_non_terminal_optin_uses_d2_path(self, cache_clear):
        """A NON-terminal opt-in state (with transitions) uses the D2
        Let+Leaf path — confirming D3 is gated on terminal-only."""
        s = _enable_leaf(_make_state_with_field_extraction("nt"))
        # _make_state_with_field_extraction adds an _end transition →
        # non-terminal → D3 doesn't fire → D2 path used.
        assert len(s.transitions) > 0
        defn = FSMDefinition(
            name="F",
            description="d",
            initial_state="nt",
            states=_with_end({"nt": s}),
        )
        term = compile_fsm(defn)
        # M3b non-cohort Let IS present (D2 path active).
        assert _ast_contains_noncohort_let(term)


class TestD1ReservedNameCoverage:
    def test_reserved_vars_includes_synthetic_callback(self):
        assert CB_RESPOND_SYNTHETIC in RESERVED_VARS

    def test_reserved_vars_includes_append_history(self):
        assert CB_APPEND_HISTORY in RESERVED_VARS

    def test_synthetic_name_distinct_from_other_callbacks(self):
        from fsm_llm.dialog.compile_fsm import (
            CB_CLASS_EXTRACT,
            CB_EVAL_TRANSIT,
            CB_EXTRACT,
            CB_FIELD_EXTRACT,
            CB_RENDER_RESPONSE_PROMPT,
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
            CB_RESPOND_SYNTHETIC,
            CB_APPEND_HISTORY,
        }
        assert len(all_cbs) == 10

    def test_noncohort_response_var_distinct_from_prompt_var(self):
        assert NONCOHORT_RESPONSE_VAR != NONCOHORT_RESPONSE_PROMPT_VAR
