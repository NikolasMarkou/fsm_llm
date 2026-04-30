"""Tests for handlers.compose AST splicer (R5 step 5).

Per D-STEP-04-RESOLUTION (plan_2026-04-27_43d56276), R5 ships with
2 real term-side splices (PRE/POST_PROCESSING) and 6 identity splices
(START/END_CONVERSATION, ERROR — host-side lifecycle dispatch;
PRE/POST_TRANSITION, CONTEXT_UPDATE — host-side per-turn dispatch with
cardinality / conditional gating that the structural splicer cannot
match). All 8 timings dispatch through ``make_handler_runner`` for
execution-path uniformity, but only PRE/POST_PROCESSING actually
mutate the AST.

These tests cover:

1. The 8-timing splice table:
   * 2 real splices (PRE/POST_PROCESSING) — verify the term shape
     changes, the host_call Combinator nodes are emitted, and the
     runner is invoked at evaluation time.
   * 6 identity splices — verify ``compose`` returns the same Term
     object structure for handlers registered at those timings (i.e.
     the splicer for that timing is identity, even though the splicer
     for OTHER timings still wraps when any handler is present).

2. Composition order — when both PRE and POST handlers are present,
   the wrap order is ``Let(post_result, Let(pre_h, host_call, body),
   Let(post_h, host_call, Var(post_result)))`` so PRE fires before
   body, POST fires after.

3. Idempotency — ``compose(term, [])`` and ``compose(term, None)``
   return ``term`` unchanged (identity, ``is`` test).

4. Cache invalidation — ``FSMManager.register_handler`` increments
   ``_handlers_version``, causing ``get_composed_term`` to recompose.
"""

from __future__ import annotations

from typing import Any

from fsm_llm.handlers import (
    CONTEXT_DATA_VAR,
    CURRENT_STATE_VAR,
    HANDLER_RUNNER_VAR_NAME,
    TARGET_STATE_VAR,
    UPDATED_KEYS_VAR,
    HandlerBuilder,
    HandlerSystem,
    HandlerTiming,
    compose,
    make_handler_runner,
    required_env_bindings,
)
from fsm_llm.runtime.ast import Abs, Case, Combinator, CombinatorOp, Let, Var
from fsm_llm.runtime.dsl import abs_, case_, var
from fsm_llm.runtime.executor import Executor

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


def _fsm_term_shape() -> Abs:
    """A 4-deep Abs chain ending in Case — the shape compile_fsm emits."""
    return abs_(
        "msg",
        abs_(
            "state_id",
            abs_(
                "conv_id",
                abs_(
                    "instance",
                    case_(var("state_id"), {"init": var("instance")}, default=None),
                ),
            ),
        ),
    )


def _handler_at(timing: HandlerTiming, name: str = "h"):
    return HandlerBuilder(name).at(timing).do(lambda ctx: {})


def _innermost_body(term: Any) -> Any:
    """Walk through Abs layers and return the innermost body."""
    while isinstance(term, Abs):
        term = term.body
    return term


# ---------------------------------------------------------------
# 1. Idempotency
# ---------------------------------------------------------------


class TestIdempotency:
    """compose(term, []) and compose(term, None) return term unchanged."""

    def test_empty_list_is_identity(self) -> None:
        term = _fsm_term_shape()
        assert compose(term, []) is term

    def test_none_is_identity(self) -> None:
        term = _fsm_term_shape()
        assert compose(term, None) is term


# ---------------------------------------------------------------
# 2. Per-timing splice tests (8 — 2 real + 6 identity)
# ---------------------------------------------------------------


class TestSplicePerTiming:
    """For each of 8 HandlerTiming values, assert the splice behavior.

    PRE/POST_PROCESSING are real splices: term shape mutates (Case →
    Let-wrapped). The other 6 are identity splices: when a handler
    registered ONLY at that timing is composed, the splicer for that
    timing is a no-op — but other splicers (PRE/POST_PROCESSING) still
    fire because compose applies all 8 splicers to a non-empty handler
    list. The "identity" property here is that NO Case-branch wrap
    or Let-chain CONTEXT_UPDATE wrap is introduced (verified by
    inspecting term structure).
    """

    def test_pre_processing_emits_let_host_call_around_body(self) -> None:
        term = _fsm_term_shape()
        composed = compose(term, [_handler_at(HandlerTiming.PRE_PROCESSING)])
        inner = _innermost_body(composed)
        # Outer Let is POST_result wrap; inner Let is PRE host_call.
        assert isinstance(inner, Let)
        # The PRE side-effect Let lives inside `inner.value`.
        pre_let = inner.value
        assert isinstance(pre_let, Let)
        assert isinstance(pre_let.value, Combinator)
        assert pre_let.value.op == CombinatorOp.HOST_CALL
        # The PRE host_call's Var-head references the runner.
        head_var = pre_let.value.args[0]
        assert isinstance(head_var, Var)
        assert head_var.name == HANDLER_RUNNER_VAR_NAME

    def test_post_processing_emits_post_let_returning_result(self) -> None:
        term = _fsm_term_shape()
        composed = compose(term, [_handler_at(HandlerTiming.POST_PROCESSING)])
        inner = _innermost_body(composed)
        assert isinstance(inner, Let)
        # Outer is `Let(_..._result, body, Let(_h, host_call, Var(_..._result)))`.
        assert inner.name.endswith("_result")
        assert isinstance(inner.body, Let)
        # body's value is the POST host_call Combinator.
        post_h = inner.body.value
        assert isinstance(post_h, Combinator)
        assert post_h.op == CombinatorOp.HOST_CALL
        # body's body returns the result Var.
        result_ref = inner.body.body
        assert isinstance(result_ref, Var)
        assert result_ref.name == inner.name

    def test_pre_transition_is_identity(self) -> None:
        """PRE_TRANSITION splicer is identity — no Case-branch wrap."""
        term = _fsm_term_shape()
        composed = compose(term, [_handler_at(HandlerTiming.PRE_TRANSITION)])
        inner = _innermost_body(composed)
        # The Case is wrapped by PRE/POST_PROCESSING (because compose applies
        # ALL splicers when the handler list is non-empty), but the Case
        # itself is unchanged structurally — its branches are NOT individually
        # wrapped by PRE_TRANSITION.
        # Walk down to the Case and confirm branches are bare references.
        case_node = _find_first_case(inner)
        assert case_node is not None
        for branch_body in case_node.branches.values():
            # No Let wrapper introduced by PRE_TRANSITION splicer
            # (the splicer is identity per D-STEP-04-RESOLUTION).
            assert not isinstance(branch_body, Let)

    def test_post_transition_is_identity(self) -> None:
        term = _fsm_term_shape()
        composed = compose(term, [_handler_at(HandlerTiming.POST_TRANSITION)])
        inner = _innermost_body(composed)
        case_node = _find_first_case(inner)
        assert case_node is not None
        for branch_body in case_node.branches.values():
            assert not isinstance(branch_body, Let)

    def test_context_update_is_identity(self) -> None:
        """CONTEXT_UPDATE splicer does NOT inject per-Let wraps."""
        # Build a term shape with an inner Let chain inside a Case branch.
        term_with_let = abs_(
            "msg",
            abs_(
                "state_id",
                abs_(
                    "conv_id",
                    abs_(
                        "instance",
                        case_(
                            var("state_id"),
                            {
                                "init": _let_chain(
                                    [("a", var("instance")), ("b", var("a"))],
                                    var("b"),
                                )
                            },
                            default=None,
                        ),
                    ),
                ),
            ),
        )
        composed = compose(term_with_let, [_handler_at(HandlerTiming.CONTEXT_UPDATE)])
        inner = _innermost_body(composed)
        case_node = _find_first_case(inner)
        assert case_node is not None
        # Walk the Let chain in the "init" branch — count non-discard Lets.
        # Pre-step-4 splicer would inject a CONTEXT_UPDATE host_call after
        # each. Post-D-STEP-04-RESOLUTION it is identity.
        branch_body = case_node.branches["init"]
        non_discard_lets = _count_non_discard_lets(branch_body)
        # Original chain has exactly 2 non-discard Lets (a, b). Identity
        # splice keeps that count; a real splice would add 2 more.
        assert non_discard_lets == 2

    def test_start_conversation_is_identity(self) -> None:
        """START_CONVERSATION splicer does not modify the term."""
        term = _fsm_term_shape()
        # Bypass the PRE/POST_PROCESSING wrappers by composing with NO
        # other timing-handler — START_CONVERSATION-only.
        composed = compose(term, [_handler_at(HandlerTiming.START_CONVERSATION)])
        # PRE/POST_PROCESSING still wrap (they always fire when handler
        # list non-empty). But START_CONVERSATION itself injects no
        # additional wrappers — so the structure matches a PRE/POST_PROCESSING-
        # only compose.
        baseline = compose(term, [_handler_at(HandlerTiming.PRE_PROCESSING)])
        # Both have the same outer-Abs depth + same innermost shape type.
        assert _term_skeleton(composed) == _term_skeleton(baseline)

    def test_end_conversation_is_identity(self) -> None:
        term = _fsm_term_shape()
        composed = compose(term, [_handler_at(HandlerTiming.END_CONVERSATION)])
        baseline = compose(term, [_handler_at(HandlerTiming.PRE_PROCESSING)])
        assert _term_skeleton(composed) == _term_skeleton(baseline)

    def test_error_is_identity(self) -> None:
        term = _fsm_term_shape()
        composed = compose(term, [_handler_at(HandlerTiming.ERROR)])
        baseline = compose(term, [_handler_at(HandlerTiming.PRE_PROCESSING)])
        assert _term_skeleton(composed) == _term_skeleton(baseline)


# ---------------------------------------------------------------
# 3. Composition order
# ---------------------------------------------------------------


class TestCompositionOrder:
    """PRE fires before body; POST fires after; runner is invoked
    in the correct order at evaluation time."""

    def test_pre_runs_before_body_post_runs_after(self) -> None:
        term = _fsm_term_shape()
        hs = HandlerSystem(error_mode="raise")
        order: list[str] = []

        def pre_h(ctx: dict[str, Any], **kw: Any) -> dict[str, Any]:
            order.append("pre")
            return {}

        def post_h(ctx: dict[str, Any], **kw: Any) -> dict[str, Any]:
            order.append("post")
            return {}

        hs.register_handler(
            HandlerBuilder("p").at(HandlerTiming.PRE_PROCESSING).do(pre_h)
        )
        hs.register_handler(
            HandlerBuilder("q").at(HandlerTiming.POST_PROCESSING).do(post_h)
        )

        composed = compose(term, list(hs.handlers))
        # Strip 4 Abs layers; eval the inner body with a synthetic env.
        body = _innermost_body(composed)
        env: dict[str, Any] = {
            HANDLER_RUNNER_VAR_NAME: make_handler_runner(hs),
            CURRENT_STATE_VAR: "init",
            TARGET_STATE_VAR: None,
            CONTEXT_DATA_VAR: {},
            UPDATED_KEYS_VAR: None,
            "instance": "INSTANCE_VAL",
            "state_id": "init",
        }
        env.update(required_env_bindings())

        # Add a sentinel into 'pre' between order calls to capture body fire.
        # The body itself is a Case branching to var("instance"). Eval
        # returns "INSTANCE_VAL"; we inject a body sentinel by appending
        # 'body' to order via a wrapper handler-system call. The simplest
        # check is: pre and post both ran, and pre ran first.
        result = Executor().run(body, env)
        assert result == "INSTANCE_VAL"
        assert order == ["pre", "post"], f"expected pre-then-post, got {order}"


# ---------------------------------------------------------------
# 4. Cache invalidation
# ---------------------------------------------------------------


class TestCacheInvalidation:
    """FSMManager.register_handler increments _handlers_version, which
    invalidates the get_composed_term cache."""

    def test_register_handler_invalidates_composed_term_cache(self) -> None:
        from fsm_llm.dialog.definitions import FSMDefinition, State
        from fsm_llm.dialog.fsm import FSMManager
        from fsm_llm.runtime._litellm import LLMInterface

        # Minimal valid FSM that compile_fsm can compile.
        fsm = FSMDefinition(
            name="cache_test",
            description="cache test",
            initial_state="hello",
            states={
                "hello": State(
                    id="hello",
                    description="d",
                    purpose="p",
                    response_instructions="r",
                    transitions=[],
                ),
            },
        )

        from unittest.mock import MagicMock

        llm = MagicMock(spec=LLMInterface)
        mgr = FSMManager(fsm_loader=lambda _id: fsm, llm_interface=llm)

        composed_v0 = mgr.get_composed_term(fsm.name)
        composed_v0_again = mgr.get_composed_term(fsm.name)
        assert composed_v0 is composed_v0_again, "cache hit before register"

        # Register a handler — version increments — next call recomposes.
        mgr.register_handler(
            HandlerBuilder("h").at(HandlerTiming.PRE_PROCESSING).do(lambda ctx: {})
        )
        composed_v1 = mgr.get_composed_term(fsm.name)
        assert composed_v1 is not composed_v0, (
            "register_handler must invalidate the composed-term cache"
        )

        # Subsequent calls hit the v1 cache entry.
        composed_v1_again = mgr.get_composed_term(fsm.name)
        assert composed_v1 is composed_v1_again

    def test_empty_handler_list_returns_base_term_via_compose_identity(self) -> None:
        """When no handlers are registered, compose is identity — the
        composed-term cache returns the same Term as get_compiled_term."""
        from unittest.mock import MagicMock

        from fsm_llm.dialog.definitions import FSMDefinition, State
        from fsm_llm.dialog.fsm import FSMManager
        from fsm_llm.runtime._litellm import LLMInterface

        fsm = FSMDefinition(
            name="ident_test",
            description="d",
            initial_state="hello",
            states={
                "hello": State(
                    id="hello",
                    description="d",
                    purpose="p",
                    response_instructions="r",
                    transitions=[],
                ),
            },
        )

        llm = MagicMock(spec=LLMInterface)
        mgr = FSMManager(fsm_loader=lambda _id: fsm, llm_interface=llm)
        base = mgr.get_compiled_term(fsm.name)
        composed = mgr.get_composed_term(fsm.name)
        assert composed is base, (
            "compose([], term) must be identity; the composed-term cache "
            "should return the same Term object as get_compiled_term when "
            "no handlers are registered"
        )


# ---------------------------------------------------------------
# 5. required_env_bindings
# ---------------------------------------------------------------


class TestRequiredEnvBindings:
    """The static (timing-string) env bindings cover all 8 timings."""

    def test_covers_all_8_timings(self) -> None:
        bindings = required_env_bindings()
        assert len(bindings) == 8
        for timing in HandlerTiming:
            key = f"_handler_timing_{timing.value}"
            assert key in bindings
            assert bindings[key] == timing.value


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _find_first_case(term: Any) -> Case | None:
    """Recursively walk a term and return the first Case found."""
    if isinstance(term, Case):
        return term
    if isinstance(term, Abs):
        return _find_first_case(term.body)
    if isinstance(term, Let):
        # Search both value and body — PRE_PROCESSING wraps put Case in Let.value.
        found = _find_first_case(term.value)
        if found is not None:
            return found
        return _find_first_case(term.body)
    return None


def _let_chain(bindings: list[tuple[str, Any]], body: Any) -> Any:
    """Build Let(n1, v1, Let(n2, v2, ..., body))."""
    result = body
    for name, value in reversed(bindings):
        result = Let(name=name, value=value, body=result)
    return result


def _count_non_discard_lets(term: Any) -> int:
    """Count non-discard Lets in a Let-chain (walks Let.body only)."""
    count = 0
    while isinstance(term, Let):
        if not term.name.startswith("_fsm_handler_"):
            count += 1
        term = term.body
    return count


def _term_skeleton(term: Any) -> str:
    """Render a tag-only skeleton of a term for structural equality.

    Discard-prefixed names (``_fsm_handler_<n>``) and their result-name
    counterparts vary across compose calls due to a global counter. We
    normalize them to ``<DISCARD>`` and ``<RESULT>`` so structural
    equality holds across runs.
    """
    if isinstance(term, Abs):
        return f"Abs({_term_skeleton(term.body)})"
    if isinstance(term, Let):
        name = _norm_name(term.name)
        return f"Let[{name}]({_term_skeleton(term.value)};{_term_skeleton(term.body)})"
    if isinstance(term, Case):
        branches = ",".join(
            f"{k}->{_term_skeleton(v)}" for k, v in sorted(term.branches.items())
        )
        return f"Case({branches})"
    if isinstance(term, Combinator):
        return f"Combinator[{term.op}]"
    if isinstance(term, Var):
        return f"Var[{_norm_name(term.name)}]"
    return type(term).__name__


def _norm_name(name: str) -> str:
    """Normalize discard-counter-suffixed names to a stable token."""
    if name.startswith("_fsm_handler_"):
        if name.endswith("_result"):
            return "<RESULT>"
        return "<DISCARD>"
    return name
