"""The react-family factories of the handler library (split from ``library.py``, plan a555dcfe).

``agents.tool_executor``, ``agents.think_turn_limiter``, ``agents.iteration_limiter``,
``agents.think_fresh_keys``, ``agents.feedback_consumed``, ``agents.approval_fresh_keys`` and
``agents.approval_gate``, the Reflexion three (bodies in ``reflection.py``), the prompt-chain gate
(``chain.py``) and the native function-calling group (``native_fc.py``). The interface contract of a
factory is in ``library.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from fsm_llm.agents.constants import (
    REACT_THINK_FRESH_KEYS,
    AgentStates,
    ContextKeys,
    Defaults,
    HandlerNames,
    HandlerPriorities,
    PromptChainStates,
    ReflexionStates,
)
from fsm_llm.agents.tools import make_hitl_checker
from fsm_llm.core.definitions import AgentError
from fsm_llm.handlers import HandlerTiming, create_handler

from .chain import make_gate_checker
from .context import HandlerContext
from .executor import AgentHandlers
from .freshness import make_fresh_keys_handler
from .library_support import _keys, _params, _priority, _refuse_flagged, _states
from .limiters import ThinkTurnLimiter, make_iteration_limiter
from .native_fc import NativeFCHandlers
from .reflection import (
    EVALUATION_KEYS,
    make_evaluation_handler,
    make_reflection_handler,
)

if TYPE_CHECKING:
    from fsm_llm.core.definitions.executable_pattern import HandlerBinding
    from fsm_llm.handlers import BaseHandler

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
    """``agents.tool_executor``: runs the selected tool on entry to ``act`` (AgentHandlers.execute_tool).

    Param ``optional`` (true): no handler when the run has no registry (Plan-and-Execute, whose tools
    are optional); without it a missing registry is an ``AgentError``.
    """
    if _params(binding).get("optional") is True and ctx.tools is None:
        return []
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


def iteration_check(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The closure ``agents.iteration_limiter`` registers (shared with ``OrchestratorAgent``'s test seam).

    Params: ``limit`` (int; default ``ctx.options["max_iterations"]``, else the config's
    ``max_iterations``), ``limit_key`` (a context key that overrides the limit at call time),
    ``early`` (default true: trigger at ``limit - 1``). ``binding.keys`` are the keys forced True when
    the limit is hit. ``AgentError`` for a limit that is not a positive integer.
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
    return make_iteration_limiter(
        limit,
        dict.fromkeys(_keys(binding, ()), True),
        context_max_key=limit_key if isinstance(limit_key, str) else None,
        early=params.get("early") is not False,
    )


def iteration_limiter(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.iteration_limiter``: the limiter that counts in context (the non-ReAct patterns); see ``iteration_check``."""
    return [
        create_handler(HandlerNames.ITERATION_LIMITER)
        .with_priority(_priority(binding, HandlerPriorities.ITERATION_LIMITER))
        .at(HandlerTiming.PRE_TRANSITION)
        .do(iteration_check(ctx, binding))
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


def reflexion_evaluator(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.reflexion_evaluator``: the verdict on entry to ``evaluate``.

    With the callable ``evaluation_fn`` (``ctx.callables``) the function writes the verdict;
    without it the stale verdict is cleared so ``evaluate`` extracts a new one.
    """
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-031
    # The verdict is produced on evaluate ENTRY: evaluation_fn writes it
    # there, otherwise the stale one is cleared so evaluate extracts a new
    # one. Do NOT run evaluation_fn at CONTEXT_UPDATE (core skips it when
    # extraction returns nothing, REACT-02) or at PRE_TRANSITION on
    # evaluate (the edge is already chosen by then, so a null extraction
    # takes the 900 fallback to reflect whatever the function said).
    # See decisions.md D-031.
    evaluation_fn = ctx.callables.get("evaluation_fn")
    do = (
        make_fresh_keys_handler(_keys(binding, EVALUATION_KEYS))
        if evaluation_fn is None
        else make_evaluation_handler(evaluation_fn)
    )
    return [
        create_handler(HandlerNames.REFLEXION_EVALUATOR)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_entry(*_states(binding, (ReflexionStates.EVALUATE,)))
        .do(do)
    ]


def reflexion_fresh_keys(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.reflexion_fresh_keys``: clears ``reflection`` and ``lessons`` on entry to ``reflect``."""
    # REACT-01: reflect produces reflection/lessons, so they are cleared on reflect entry and
    # recorded on reflect EXIT (agents.reflexion_reflector), after the reflect extraction.
    clear = make_fresh_keys_handler(
        _keys(binding, (ContextKeys.REFLECTION, ContextKeys.LESSONS))
    )
    return [
        create_handler(HandlerNames.REFLEXION_FRESH_KEYS)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_entry(*_states(binding, (ReflexionStates.REFLECT,)))
        .do(clear)
    ]


def reflexion_reflector(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.reflexion_reflector``: records the episode on exit from ``reflect``.

    The cap is the option ``max_reflections`` of the run (``ctx.options``), at least 1.
    """
    limit = ctx.options.get("max_reflections", Defaults.MAX_REFLECTIONS)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise AgentError(
            f"agents.reflexion_reflector needs max_reflections >= 1, got {limit!r}"
        )
    return [
        create_handler(HandlerNames.REFLEXION_REFLECTOR)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_exit(*_states(binding, (ReflexionStates.REFLECT,)))
        .do(make_reflection_handler(limit))
    ]


def chain_gate_checker(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.chain_gate_checker``: one entry handler per state after a chain step (D-019).

    Reads the options ``steps`` (the generator's own list) and the callables ``validation_fns``
    (one function or ``None`` per step, aligned with ``steps``): the state after step ``i`` records
    its result and runs its gate. A gated step with no function is refused: its gate edge would
    never be taken. A function for a step that is not gated is ignored (the topology decides).
    A binding may move the priority, never the states: they follow the chain.
    """
    steps: Sequence[Mapping[str, Any]] = ctx.options.get("steps") or ()
    functions: Sequence[Callable[[dict[str, Any]], bool] | None] = cast(
        "Sequence[Callable[[dict[str, Any]], bool] | None]",
        ctx.callables.get("validation_fns") or (),
    )
    built: list[BaseHandler] = []
    for index, previous in enumerate(steps, start=1):
        last = index == len(steps)
        validate = None
        if previous["gated"]:
            validate = functions[index - 1] if index <= len(functions) else None
            if validate is None:
                raise AgentError(
                    f"chain step {previous['name']!r} is gated and has no validation function"
                )
        state = (
            PromptChainStates.OUTPUT
            if last
            else f"{PromptChainStates.STEP_PREFIX}{index}"
        )
        suffix = "final" if last else index
        built.append(
            create_handler(f"{HandlerNames.CHAIN_GATE_CHECKER}_{suffix}")
            .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
            .on_state_entry(state)
            .do(make_gate_checker(index, previous["name"], validate, last))
        )
    return built


def native_fc(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.native_fc``: the GROUP of eight handlers of a native function-calling run (D-021).

    One name for a fixed set that shares one ``NativeFCHandlers`` (the run's tools, the turn cap, the
    forced tool, the parser): a binding changes nothing in it (states, keys and priority are fixed by
    the pattern). Reads ``ctx.tools`` and ``ctx.config`` (``max_iterations``, ``force_final_tool``,
    ``output_schema``) and the internal callable ``parse_structured`` the agent supplies; with no
    output schema no repair is ever flagged. ``AgentError`` for a context with no tools or config.
    """
    if ctx.tools is None or ctx.config is None:
        raise AgentError("agents.native_fc needs the run's tools and config")
    _refuse_flagged(ctx, "agents.native_fc")
    config = ctx.config
    return NativeFCHandlers(
        ctx.tools,
        max_tool_turns=config.max_iterations,
        force_final_tool=config.force_final_tool,
        parse_structured=(
            ctx.callables.get("parse_structured")
            if config.output_schema is not None
            else None
        ),
    ).handlers()


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
