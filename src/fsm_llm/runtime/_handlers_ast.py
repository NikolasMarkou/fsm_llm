from __future__ import annotations

"""Private AST-rewrite layer for handler splicing.

This module is the kernel-level term-rewrite half of the handler system.
It hosts:

* :func:`compose` — the public top-level entry that splices handler
  invocations into a compiled λ-term. Re-exported from
  :mod:`fsm_llm.handlers` for the public import path
  ``from fsm_llm.handlers import compose``.
* :func:`required_env_bindings` — pure-data helper that returns the
  static (timing-string) env bindings needed by spliced terms.
* The five reserved env-name string constants
  (:data:`HANDLER_RUNNER_VAR_NAME`, :data:`CURRENT_STATE_VAR`,
  :data:`TARGET_STATE_VAR`, :data:`CONTEXT_DATA_VAR`,
  :data:`UPDATED_KEYS_VAR`).
* Eight ``_splice_<timing>`` helpers (some real AST rewrites, some
  identity placeholders — see below) plus their shared building
  blocks (:func:`_handler_invocation`, :func:`_timing_var_name`,
  :func:`_wrap_innermost_abs_body`, :func:`_wrap_pre`,
  :func:`_wrap_post`, :func:`_fresh_discard_name`).

History
-------
This was originally a tail-end section of :mod:`fsm_llm.handlers`
(``handlers.py`` lines ~975-1426 pre-0.8.0). It moved here in 0.8.0 to
narrow ``handlers.py`` to its host-dispatch role (HandlerSystem,
HandlerBuilder, BaseHandler, FSMHandler, LambdaHandler,
make_handler_runner, HandlerTiming, error classes, create_handler) and
to put kernel-level AST transforms in ``runtime/`` where they belong.
The split is **pure code motion**: every public surface
(:func:`compose`, :func:`required_env_bindings`, the five env-name
constants) is re-exported byte-equivalent from :mod:`fsm_llm.handlers`,
so existing call sites — ``from fsm_llm.handlers import compose``,
``from fsm_llm import compose`` — continue to work unchanged.

Layering invariant
------------------
This module imports from :mod:`fsm_llm.runtime.ast` and
:mod:`fsm_llm.runtime.dsl` only. It does **not** import from
:mod:`fsm_llm.handlers`, :mod:`fsm_llm.dialog`, :mod:`fsm_llm.stdlib`,
or :mod:`fsm_llm.program` — that would breach the L1 (kernel) closure
rule enforced by ``tests/test_fsm_llm/test_layering.py``. The
``HandlerTiming`` enum lives in ``handlers.py`` for host-dispatch
reasons (the runner needs the enum for kw-arg dispatch into
``HandlerSystem.execute_handlers``); this module mirrors the eight
timing-string values as private string constants. The string values
are the wire-format contract — ``HandlerTiming(value)`` round-trips
on the host side.

R5 architectural notes
----------------------
Per ``plans/plan_2026-04-27_43d56276/plan.md`` and D-PLAN-02, R5
reframed handler execution: rather than the FSM dialog pipeline calling
``HandlerSystem.execute_handlers`` as Python middleware around
``Executor.run``, the handler dispatch is **spliced into the compiled
λ-term itself** at the appropriate structural seam, and runs inside
the executor as a ``Combinator(op=HOST_CALL, ...)`` invocation. The
host-callable bound at the env name :data:`HANDLER_RUNNER_VAR_NAME`
(returned by :func:`fsm_llm.handlers.make_handler_runner`) is the
bridge.

Of the eight ``HandlerTiming`` values, only PRE_PROCESSING and
POST_PROCESSING are real AST splices in this module. The other six
are identity transforms because:

* ``START_CONVERSATION`` / ``END_CONVERSATION`` — host-side. The
  pre-R5 dispatch sites are in ``dialog/fsm.py``; they fire once per
  conversation, not per turn. The splice function is identity; the
  host invokes the runner directly through ``make_handler_runner`` so
  the execution path is unified.
* ``ERROR`` — host-trapped. The executor cannot trap exceptions
  (would breach the kernel's purity invariants), so the host's
  exception boundary catches and dispatches ERROR handlers via the
  runner before re-raising. Splice function is identity.
* ``PRE_TRANSITION`` / ``POST_TRANSITION`` / ``CONTEXT_UPDATE`` — see
  the D-STEP-04-RESOLUTION note further down (cardinality and
  conditional-gating semantics that the structural splicer cannot
  match without dedicated test coverage). Identity for now; the call
  sites in ``dialog/turn.py`` keep calling
  ``MessagePipeline.execute_handlers(...)`` directly.
"""

import itertools
from collections.abc import Callable
from typing import Any

from .ast import Abs, Combinator, Term
from .dsl import host_call, let_, var

# --------------------------------------------------------------
# Reserved env-name string constants
# --------------------------------------------------------------

# Canonical env-binding name for the handler runner host-callable.
# The AST splicer emits ``host_call(HANDLER_RUNNER_VAR_NAME, <timing>, ...)``
# nodes. The dialog/runtime caller (Program/API in step 3, MessagePipeline
# in step 4) is responsible for binding this name in env to the value
# returned by :func:`fsm_llm.handlers.make_handler_runner`.
HANDLER_RUNNER_VAR_NAME: str = "__fsm_handlers__"

# Reserved env names referenced by the splicer (must be bound by the
# dialog-side caller before evaluating a spliced term):
#
# * ``HANDLER_RUNNER_VAR_NAME`` — the host-callable.
# * ``CURRENT_STATE_VAR`` — the current FSM state id (str).
# * ``TARGET_STATE_VAR`` — the next state id (str | None) — only meaningful
#   under PRE/POST_TRANSITION; bound to ``None`` elsewhere.
# * ``CONTEXT_DATA_VAR`` — the current context dict (dict[str, Any]).
# * ``UPDATED_KEYS_VAR`` — set[str] | None — only meaningful under
#   CONTEXT_UPDATE; bound to ``None`` elsewhere.
CURRENT_STATE_VAR: str = "current_state_id"
TARGET_STATE_VAR: str = "target_state_id"
CONTEXT_DATA_VAR: str = "context_data"
UPDATED_KEYS_VAR: str = "updated_keys"

# --------------------------------------------------------------
# HandlerTiming string-value mirror
# --------------------------------------------------------------
#
# The eight HandlerTiming enum values are mirrored here as private
# string constants so this module does not need to import the enum
# from ``fsm_llm.handlers`` (would breach the L1 layering rule). The
# string values ARE the wire-format contract — the host side does
# ``HandlerTiming(value)`` to round-trip back to the enum, and the
# splicer only ever needs the string form (encoded into a Var name
# via :func:`_timing_var_name`).
#
# Keep these in sync with ``HandlerTiming`` in ``fsm_llm.handlers``.
# A drift between this module and the enum would surface as a
# :func:`required_env_bindings` test failure
# (``tests/test_fsm_llm/test_handlers_ast.py::TestRequiredEnvBindings``).

_TIMING_START_CONVERSATION: str = "start_conversation"
_TIMING_PRE_PROCESSING: str = "pre_processing"
_TIMING_POST_PROCESSING: str = "post_processing"
_TIMING_PRE_TRANSITION: str = "pre_transition"
_TIMING_POST_TRANSITION: str = "post_transition"
_TIMING_CONTEXT_UPDATE: str = "context_update"
_TIMING_END_CONVERSATION: str = "end_conversation"
_TIMING_ERROR: str = "error"

_ALL_TIMING_VALUES: tuple[str, ...] = (
    _TIMING_START_CONVERSATION,
    _TIMING_PRE_PROCESSING,
    _TIMING_POST_PROCESSING,
    _TIMING_PRE_TRANSITION,
    _TIMING_POST_TRANSITION,
    _TIMING_CONTEXT_UPDATE,
    _TIMING_END_CONVERSATION,
    _TIMING_ERROR,
)


# --------------------------------------------------------------
# Fresh-name generation
# --------------------------------------------------------------

# Sentinel name used as the input_var for synthesized Let bindings that
# discard the handler runner's return value. The runner is invoked for its
# side-effect on the FSM context (the host-callable mutates ``instance``);
# its return value is therefore discarded by the surrounding Let.
_DISCARD_VAR_PREFIX: str = "_fsm_handler_"

# Internal counter for generating fresh _DISCARD_VAR_PREFIX names per
# splice. The names need only be unique within a single ``compose`` call.
# ``itertools.count()`` is thread-safe (its ``__next__`` holds the GIL for
# its single critical section), and replaces the pre-0.7.0
# ``_DISCARD_COUNTER: list[int] = [0]`` mutable-global pattern.
_DISCARD_COUNTER = itertools.count(1)


def _fresh_discard_name() -> str:
    """Generate a fresh discard-var name for a synthesized Let binding."""
    return f"{_DISCARD_VAR_PREFIX}{next(_DISCARD_COUNTER)}"


# --------------------------------------------------------------
# Public API: compose
# --------------------------------------------------------------


def compose(
    term: Term,
    handlers: list[Any] | None,
    *,
    handler_runner_var: str = HANDLER_RUNNER_VAR_NAME,
) -> Term:
    """Splice handler-invocation seams into a compiled λ-term.

    For each :class:`fsm_llm.handlers.HandlerTiming` value, the corresponding
    ``_splice_<timing>`` rewriter is applied to ``term`` in deterministic
    order (program-level outermost, then turn-level, then per-Case, then
    per-Let). The result is a new ``Term`` whose evaluation invokes the
    host-callable bound at ``handler_runner_var`` at every splice point.

    Idempotent for ``handlers in (None, [])``: returns ``term`` unchanged
    (zero AST mutation, zero new nodes). This is the back-compat path for
    FSMs registered with no handlers — the splicer must not perturb the
    AST shape.

    The ``handlers`` argument is **not** introspected to filter splice
    points; instead, every timing's splice is unconditionally applied,
    and the host-callable's
    :meth:`fsm_llm.handlers.HandlerSystem.execute_handlers`
    delegates to ``should_execute`` per handler at runtime. This mirrors
    the pre-R5 middleware semantics. Per-timing pruning is a future
    optimization (deferred — see D-PLAN-02 trade-off).

    :param term: The compiled λ-term (typically the output of
        ``compile_fsm`` / ``compile_fsm_cached``).
    :param handlers: List of ``FSMHandler`` instances. ``None`` or empty
        list short-circuits — splicer returns ``term`` unchanged. Typed as
        ``list[Any] | None`` because this module does not import the
        ``FSMHandler`` Protocol from ``fsm_llm.handlers`` (L1 closure
        rule); callers always pass a list of FSMHandler instances at the
        public API boundary.
    :param handler_runner_var: Env-binding name for the handler runner
        host-callable. Defaults to :data:`HANDLER_RUNNER_VAR_NAME`. Tests
        may override for isolation.
    :return: A new ``Term`` with handler splices, or ``term`` unchanged if
        no handlers were registered.
    """
    if not handlers:
        return term

    # Apply splices in a fixed order. Program-level splices wrap last
    # (outermost) so they fire first/last during evaluation. Inner splices
    # wrap first so they nest inside the program-level wrappers.
    spliced = term
    spliced = _splice_context_update(spliced, handler_runner_var)
    spliced = _splice_pre_transition(spliced, handler_runner_var)
    spliced = _splice_post_transition(spliced, handler_runner_var)
    spliced = _splice_pre_processing(spliced, handler_runner_var)
    spliced = _splice_post_processing(spliced, handler_runner_var)
    spliced = _splice_start_conversation(spliced, handler_runner_var)
    spliced = _splice_end_conversation(spliced, handler_runner_var)
    spliced = _splice_error(spliced, handler_runner_var)
    return spliced


# --------------------------------------------------------------
# Splice helpers
# --------------------------------------------------------------
#
# Each ``_splice_<timing>`` returns a Term. The shared building block
# below, :func:`_handler_invocation`, builds a single
# ``Combinator(HOST_CALL, ...)`` node bound in a Let around a body.


def _handler_invocation(timing_value: str, *, runner_var: str) -> Combinator:
    """Build a ``host_call(runner, timing, state, target, ctx, keys)`` node.

    The runner expects positional args in the same order as
    ``HandlerSystem.execute_handlers`` kwargs:
    ``(timing_str, current_state, target_state, context, updated_keys)``.

    The timing is encoded as a Var named ``_handler_timing_<value>``
    which the dialog-side caller binds to the literal string ``<value>``
    before each call. That keeps the AST free of Python literal embedding
    and reuses the existing env-binding machinery.
    """
    timing_var_name = _timing_var_name(timing_value)
    return host_call(
        runner_var,
        var(timing_var_name),
        var(CURRENT_STATE_VAR),
        var(TARGET_STATE_VAR),
        var(CONTEXT_DATA_VAR),
        var(UPDATED_KEYS_VAR),
    )


def _timing_var_name(timing_value: str) -> str:
    """Canonical Var name for a timing string literal in env.

    The dialog-side caller pre-binds ``_handler_timing_<value>`` →
    ``<value>`` for every ``HandlerTiming``. The splicer references
    these Vars rather than embedding string literals into the AST.
    """
    return f"_handler_timing_{timing_value}"


def required_env_bindings() -> dict[str, str]:
    """Return the static (timing-string) env bindings required by spliced terms.

    The dialog-side caller (Program/API/MessagePipeline) must merge this
    dict into the env passed to ``Executor.run``. The remaining bindings
    (CURRENT_STATE_VAR, TARGET_STATE_VAR, CONTEXT_DATA_VAR,
    UPDATED_KEYS_VAR, HANDLER_RUNNER_VAR_NAME) are runtime-dependent and
    are bound per-turn / per-call by the caller.

    :return: dict mapping ``_handler_timing_<value>`` → ``<value>`` for
        each ``HandlerTiming`` value.
    """
    return {_timing_var_name(t): t for t in _ALL_TIMING_VALUES}


# --------------------------------------------------------------
# Outer (program-level) splices — START/END_CONV, ERROR
# --------------------------------------------------------------


def _splice_start_conversation(term: Term, runner_var: str) -> Term:
    """Identity splice — START_CONVERSATION fires on the conversation
    lifecycle boundary, not inside the per-turn compiled term.

    The pre-R5 dispatch site lives in ``dialog/fsm.py:start_conversation``
    (lines ~267 — the `_execute_handlers(START_CONVERSATION, ...)` call).
    That call site is not part of the compiled λ-term — it runs once when
    a new conversation is created, before any turn is processed. Splicing
    it into the per-turn term would fire it on every turn, which is wrong.

    Step 4 of plan_43d56276 keeps the host-side ``_execute_handlers`` call
    at the lifecycle boundary; it routes through ``make_handler_runner``
    so the underlying execution path is unified, but the call site is
    host-side, not term-spliced. This splice function is therefore a
    structural placeholder: identity transform, exists so the 8-timing
    enumeration in :func:`required_env_bindings` and the orchestration in
    :func:`compose` is uniform.
    """
    return term


def _splice_end_conversation(term: Term, runner_var: str) -> Term:
    """Identity splice — END_CONVERSATION is host-side (see
    :func:`_splice_start_conversation` for rationale). Lifecycle boundary
    in ``dialog/fsm.py:end_conversation`` (lines ~292, ~305, ~567)."""
    return term


def _splice_error(term: Term, runner_var: str) -> Term:
    """Identity splice — ERROR handlers are host-trapped.

    The executor has no try/except machinery (would breach the kernel's
    purity invariants). The host (``dialog/fsm.py`` line ~417 catches at
    the orchestration boundary) catches the exception, then *separately*
    invokes the runner with timing=ERROR before re-raising. This splice
    is a structural placeholder so ``required_env_bindings()`` covers all
    8 timings; the actual ERROR dispatch lives in the host's exception
    handler (see D-PLAN-02 trade-off — host-side error boundary preserved).
    """
    return term


# --------------------------------------------------------------
# Turn-level splices — PRE/POST_PROCESSING
# --------------------------------------------------------------


def _splice_pre_processing(term: Term, runner_var: str) -> Term:
    """Wrap the inner per-turn body of ``term`` so PRE_PROCESSING fires first.

    The compiled FSM term shape (per ``dialog/compile_fsm.py:compile_fsm``)
    is::

        Abs(USER_MSG, Abs(STATE_ID, Abs(CONV_ID, Abs(INSTANCE, Case(...)))))

    The PRE/POST_PROCESSING splice points are inside the innermost ``Abs``
    body, around the ``Case``-on-state. We walk the four-deep Abs chain
    and rewrap the innermost body. If ``term`` is not an Abs (e.g. tests
    pass a bare term), we splice at the top — the caller is responsible
    for ensuring the term shape is appropriate.
    """
    return _wrap_innermost_abs_body(
        term,
        lambda inner: _wrap_pre(inner, _TIMING_PRE_PROCESSING, runner_var),
    )


def _splice_post_processing(term: Term, runner_var: str) -> Term:
    """Wrap the inner per-turn body of ``term`` so POST_PROCESSING fires last."""
    return _wrap_innermost_abs_body(
        term,
        lambda inner: _wrap_post(inner, _TIMING_POST_PROCESSING, runner_var),
    )


def _wrap_innermost_abs_body(term: Term, transform: Callable[[Term], Term]) -> Term:
    """Recursively walk an ``Abs`` chain and apply ``transform`` to its body.

    For ``Abs(p1, Abs(p2, ..., Abs(pN, body)))`` returns
    ``Abs(p1, Abs(p2, ..., Abs(pN, transform(body))))``. For non-``Abs``
    terms returns ``transform(term)``.

    Pure structural rewrite — does not interpret the body's shape.
    """
    if isinstance(term, Abs):
        return Abs(
            param=term.param,
            body=_wrap_innermost_abs_body(term.body, transform),
        )
    return transform(term)


def _wrap_pre(body: Term, timing_value: str, runner_var: str) -> Term:
    """Emit ``let _h_N = host_call(...) in body``."""
    h = _handler_invocation(timing_value, runner_var=runner_var)
    return let_(_fresh_discard_name(), h, body)


def _wrap_post(body: Term, timing_value: str, runner_var: str) -> Term:
    """Emit ``let _r = body in let _h_N = host_call(...) in _r``."""
    result_name = _fresh_discard_name() + "_result"
    h = _handler_invocation(timing_value, runner_var=runner_var)
    return let_(
        result_name,
        body,
        let_(_fresh_discard_name(), h, var(result_name)),
    )


# --------------------------------------------------------------
# Identity splices — PRE/POST_TRANSITION + CONTEXT_UPDATE (D-STEP-04-RESOLUTION)
# --------------------------------------------------------------
#
# These three timings are dispatched **host-side** in the R5-narrow scope
# (per Option gamma in plan_43d56276 D-STEP-04-RESOLUTION). The structural
# splicer approach (per-Case-branch wrap for PRE/POST_TRANSITION; per-Let
# wrap for CONTEXT_UPDATE) does not match the existing call-site cardinality
# and conditional gating semantics:
#
#   * PRE_TRANSITION / POST_TRANSITION — the host fires these only when an
#     actual transition is applied (and POST_TRANSITION has rollback-on-
#     failure semantics that require a kernel exception node we do not
#     emit). A per-Case-branch wrap would over-fire on every turn.
#   * CONTEXT_UPDATE — the host call is guarded by ``if extracted_data:``
#     and passes a per-call ``updated_keys`` set. A per-Let wrap cannot
#     thread per-Let key sets without structural changes.
#
# These splice functions therefore remain identity transforms — the
# call sites in ``dialog/turn.py`` keep calling
# ``MessagePipeline.execute_handlers(...)`` directly. The
# unified execution path is preserved at the
# ``HandlerSystem.execute_handlers`` boundary.
#
# Refining the splicer to honour these cardinality + gating semantics is
# deferred to a follow-up plan with dedicated test coverage. The
# enumeration is kept here so :func:`compose`'s loop is uniform and
# :func:`required_env_bindings` covers all 8 timings without special-
# casing.


def _splice_pre_transition(term: Term, runner_var: str) -> Term:
    """Identity splice — see module docstring above."""
    return term


def _splice_post_transition(term: Term, runner_var: str) -> Term:
    """Identity splice — see module docstring above."""
    return term


def _splice_context_update(term: Term, runner_var: str) -> Term:
    """Identity splice — see module docstring above."""
    return term
