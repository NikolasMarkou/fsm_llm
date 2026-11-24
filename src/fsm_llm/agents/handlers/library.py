"""The handler library: named FACTORIES a pattern file binds (plan a555dcfe step 7, D-013).

``agents.<name>`` -> ``factory(ctx, binding=None) -> list[Handler]``, registered at import in the
``HANDLERS`` namespace of ``fsm_llm.engine.executables.namespaces``. This module holds the react
family: ``agents.tool_executor``, ``agents.think_turn_limiter``, ``agents.iteration_limiter``,
``agents.think_fresh_keys``, ``agents.feedback_consumed``, ``agents.approval_fresh_keys`` and
``agents.approval_gate``. Each is its own function (a ``NameRegistry`` refuses one function under two
names). The handler bodies are the existing ones (``executor``, ``limiters``, ``freshness``,
``tools.make_hitl_checker``); a factory only builds the handler around them, with the name, timing,
state filter and priority the live registration of ``base/registration.py`` and ``base/hitl_gate.py``
uses, so a pattern file that binds them registers the same handlers in the same order.

Interface contract:

- ``ctx``: a ``HandlerContext`` (``context.py``), built once per run. ``binding``: the file's
  ``HandlerBinding`` or ``None``, read by attribute only (``states``, ``keys``, ``priority``,
  ``params``). A binding may override the ``states`` and the ``priority`` of a handler, never its
  timing; ``keys`` names the keys a fresh-keys handler clears or an ``iteration_limiter`` forces.
- Returns a list of built handlers (empty when the handler does not apply to this run: the two
  approval handlers of a run with no HITL). A factory is called once per run and builds new handler
  objects every call.
- Failure: ``AgentError`` for a missing ``tools`` registry (the executor and the limiter that shares
  it), a key that is internal-prefixed or secret-looking (D2), or a bad ``limit``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from fsm_llm.agents.constants import (
    REACT_THINK_FRESH_KEYS,
    AgentStates,
    ContextKeys,
    HandlerNames,
    HandlerPriorities,
)
from fsm_llm.agents.tools import make_hitl_checker
from fsm_llm.constants import has_internal_prefix, is_forbidden_context_entry
from fsm_llm.core.definitions import AgentError
from fsm_llm.engine.executables.namespaces import HANDLERS
from fsm_llm.handlers import HandlerTiming, create_handler

from .context import HandlerContext
from .executor import AgentHandlers
from .freshness import make_fresh_keys_handler
from .limiters import ThinkTurnLimiter, make_iteration_limiter

if TYPE_CHECKING:
    from fsm_llm.core.definitions.executable_pattern import HandlerBinding
    from fsm_llm.handlers import BaseHandler

__all__ = ["LIBRARY", "register_library"]

#: The ``scope`` keys of a run: the ``AgentHandlers`` shared by the executor and the limiter, and the
#: plain ``ThinkTurnLimiter`` of a limiter bound ``alone``.
_SHARED = "agent_handlers"
_ALONE = "think_limiter"


# DECISION plan-2026-11-24T043150-a555dcfe/D-013
# The registry stores FACTORIES. The ``AgentHandlers`` of a run (the tool executor and the think-turn
# limiter share ONE instance, as ``react.py`` shares it today) lives in ``ctx.scope``, a fresh dict per
# run: do NOT cache a built handler or an ``AgentHandlers`` on this module, on the registry or on an
# agent (per-run state on a shared object is the D-014 race of plan 089d0ec7). The executor is built
# under ``ctx.approval_predicate`` (the one predicate of the gate, the state and the refusal): do NOT
# build it without. A key a file names (``binding.keys``) goes through ``has_internal_prefix`` and
# ``is_forbidden_context_entry``: a file must never clear or force ``_approval_granted`` or a budget key.


def _states(binding: HandlerBinding | None, default: Sequence[str]) -> tuple[str, ...]:
    """The states of a handler: the binding's when it lists any, else the factory's own."""
    listed = tuple(getattr(binding, "states", None) or ())
    return listed or tuple(default)


def _priority(binding: HandlerBinding | None, default: int) -> int:
    """The priority of a handler: the binding's when it sets one, else the factory's own."""
    chosen = getattr(binding, "priority", None)
    return default if chosen is None else int(chosen)


def _params(binding: HandlerBinding | None) -> dict[str, Any]:
    return dict(getattr(binding, "params", None) or {})


def _keys(binding: HandlerBinding | None, default: Sequence[str]) -> tuple[str, ...]:
    """The keys of a handler: the binding's (each checked, D2) or the factory's own."""
    listed = tuple(getattr(binding, "keys", None) or ())
    for key in listed:
        # a neutral value: the name decides (patterns), a value-shape layer needs a real secret
        if has_internal_prefix(key) or is_forbidden_context_entry(key, "x"):
            raise AgentError(
                f"handler binding names key {key!r}: internal-prefixed and secret-looking "
                "keys are never named by a pattern file"
            )
    return listed or tuple(default)


def _agent_handlers(ctx: HandlerContext) -> AgentHandlers:
    """The run's one ``AgentHandlers`` (built on first use, under the run's one predicate)."""
    found = ctx.scope.get(_SHARED)
    if found is None:
        if ctx.tools is None:
            raise AgentError(
                "this handler needs the run's tools and the context has none"
            )
        found = AgentHandlers(ctx.tools, requires_approval=ctx.approval_predicate)
        ctx.scope[_SHARED] = found
    result: AgentHandlers = found
    return result


def tool_executor(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.tool_executor``: runs the selected tool on entry to ``act`` (AgentHandlers.execute_tool)."""
    handlers = _agent_handlers(ctx)
    return [
        create_handler(HandlerNames.TOOL_EXECUTOR)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_entry(*_states(binding, (AgentStates.ACT,)))
        .do(handlers.execute_tool)
    ]


def think_turn_limiter(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.think_turn_limiter``: the ReAct-family limiter, counting think turns on the instance.

    It shares the run's ``AgentHandlers`` with ``agents.tool_executor`` (whichever is bound first
    builds it). Param ``alone`` (true) gives a plain ``ThinkTurnLimiter`` for a pattern with no
    executor, built without a registry (D-042 of plan 944e2692: no ``gated=`` check there).
    """
    if _params(binding).get("alone") is True:
        limiter = ctx.scope.get(_ALONE)
        if limiter is None:
            limiter = ctx.scope[_ALONE] = ThinkTurnLimiter()
    else:
        limiter = _agent_handlers(ctx)
    return [
        create_handler(HandlerNames.ITERATION_LIMITER)
        .with_priority(_priority(binding, HandlerPriorities.ITERATION_LIMITER))
        .at(HandlerTiming.PRE_TRANSITION)
        .do(limiter.check_iteration_limit)
    ]


def iteration_limiter(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.iteration_limiter``: the limiter that counts in context (the non-ReAct patterns).

    Params: ``limit`` (int; default ``ctx.options["max_iterations"]``, else the config's
    ``max_iterations``), ``limit_key`` (a context key that overrides the limit at call time),
    ``early`` (default true: trigger at ``limit - 1``). ``binding.keys`` are the keys forced True when
    the limit is hit.
    """
    params = _params(binding)
    limit = params.get("limit", ctx.options.get("max_iterations"))
    if limit is None and ctx.config is not None:
        limit = ctx.config.max_iterations
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise AgentError(
            f"agents.iteration_limiter needs a positive integer limit, got {limit!r}"
        )
    limit_key = params.get("limit_key")
    forced = dict.fromkeys(_keys(binding, ()), True)
    check = make_iteration_limiter(
        limit,
        forced,
        context_max_key=limit_key if isinstance(limit_key, str) else None,
        early=params.get("early") is not False,
    )
    return [
        create_handler(HandlerNames.ITERATION_LIMITER)
        .with_priority(_priority(binding, HandlerPriorities.ITERATION_LIMITER))
        .at(HandlerTiming.PRE_TRANSITION)
        .do(check)
    ]


def think_fresh_keys(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.think_fresh_keys``: clears the loop keys on entry to ``think`` so it extracts them again."""
    # The 06a5ec0a D-051 rule of base/registration.py: keep_limiter_forced is False here. Do NOT keep
    # a limiter-forced should_terminate on think entry (the model is then never asked on its last turn).
    clear = make_fresh_keys_handler(
        _keys(binding, REACT_THINK_FRESH_KEYS), keep_limiter_forced=False
    )
    return [
        create_handler(HandlerNames.THINK_FRESH_KEYS)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_entry(*_states(binding, (AgentStates.THINK,)))
        .do(clear)
    ]


def feedback_consumed(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.feedback_consumed``: clears ``agent_feedback`` on EXIT from ``think`` (once read)."""
    # The 06a5ec0a D-029 rule of base/registration.py: on exit, never on entry (act writes the
    # feedback and the act -> think edge enters think before think extracts).
    clear = make_fresh_keys_handler(_keys(binding, (ContextKeys.AGENT_FEEDBACK,)))
    return [
        create_handler(HandlerNames.FEEDBACK_CONSUMED)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_exit(*_states(binding, (AgentStates.THINK,)))
        .do(clear)
    ]


def approval_fresh_keys(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.approval_fresh_keys``: clears ``should_terminate`` on entry to ``await_approval``.

    Only for a HITL-active run (``ctx.hitl_active``); else no handler.
    """
    if not ctx.hitl_active:
        return []
    clear = make_fresh_keys_handler(_keys(binding, (ContextKeys.SHOULD_TERMINATE,)))
    return [
        create_handler(HandlerNames.APPROVAL_FRESH_KEYS)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_entry(*_states(binding, (AgentStates.AWAIT_APPROVAL,)))
        .do(clear)
    ]


def approval_gate(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.approval_gate``: the HITL gate, set when a tool name is written, under the one predicate.

    Only for a run with a ``HumanInTheLoop`` and its predicate; else no handler. The ``await_approval``
    state, this gate and the executor's refusal share ``ctx.approval_predicate``. A binding changes
    nothing here: the trigger key (``tool_name``) and the priority are fixed, so a file can neither
    re-point the gate at another key nor move it behind the handlers it must run before.
    """
    if ctx.hitl is None or ctx.approval_predicate is None:
        return []
    check = make_hitl_checker(ctx.hitl, policy=ctx.approval_predicate)
    return [
        create_handler(HandlerNames.HITL_GATE)
        .with_priority(HandlerPriorities.HITL_GATE)
        .at(HandlerTiming.CONTEXT_UPDATE)
        .when_keys_updated(ContextKeys.TOOL_NAME)
        .do(check)
    ]


#: ``(registered name, factory)`` in a stable order; each function once.
LIBRARY: tuple[tuple[str, Callable[..., list[BaseHandler]]], ...] = (
    ("agents.tool_executor", tool_executor),
    ("agents.think_turn_limiter", think_turn_limiter),
    ("agents.iteration_limiter", iteration_limiter),
    ("agents.think_fresh_keys", think_fresh_keys),
    ("agents.feedback_consumed", feedback_consumed),
    ("agents.approval_fresh_keys", approval_fresh_keys),
    ("agents.approval_gate", approval_gate),
)


def register_library() -> None:
    """Register the library in ``HANDLERS`` (idempotent: a name is replaced by its own function)."""
    for name, factory in LIBRARY:
        HANDLERS.register(name, factory, override=True)


register_library()
