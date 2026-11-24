"""The handler library: named FACTORIES a pattern file binds (plan a555dcfe step 7, D-013).

``agents.<name>`` -> ``factory(ctx, binding=None) -> list[Handler]``, registered at import in the
``HANDLERS`` namespace of ``fsm_llm.engine.executables.namespaces``. This module holds the react
family: ``agents.tool_executor``, ``agents.think_turn_limiter``, ``agents.iteration_limiter``,
``agents.think_fresh_keys``, ``agents.feedback_consumed``, ``agents.approval_fresh_keys`` and
``agents.approval_gate``, and the Reflexion three (``agents.reflexion_evaluator``,
``agents.reflexion_fresh_keys``, ``agents.reflexion_reflector``; bodies in ``reflection.py``) and the
prompt-chain gate (``agents.chain_gate_checker``; body in ``chain.py``), the native function-calling
GROUP ``agents.native_fc`` (eight handlers; bodies in ``native_fc.py``) and the orchestrator pair
(``agents.orchestrator_delegator``, body in ``orchestrator.py``, and ``agents.orchestrator_decision_reset``),
the REWOO executor (``agents.rewoo_executor``, body in ``rewoo.py``), the Plan-and-Execute four
(``agents.plan_step_executor``, ``plan_step_fresh_keys``, ``plan_step_checker`` and
``plan_replan_counter``; bodies in ``plan_execute.py``), plus two HOOKS,
``agents.raise_run_ending_error`` and ``agents.rewoo_cut_off_answer`` (below). Each is its own function (a ``NameRegistry`` refuses one function under two
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
- A HOOK is a library entry in the same namespace (D-004) bound under ``run.hooks``: ``factory(ctx,
  binding) -> Callable[[], None]``, called once per run, whose result the agent calls when the loop
  ends (``at: after_run``, in the declared order, also when the run raised). It reads the same
  per-run ``ctx`` (the holder a handler fills lives in ``ctx.scope``); a hook that rewrites the answer
  replaces ``ctx.scope[RUN_RESULT]``.
- Failure: ``AgentError`` for a missing ``tools`` registry (the executor and the limiter that shares
  it), a key that is internal-prefixed or secret-looking (D2), or a bad ``limit``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from fsm_llm.agents.constants import (
    REACT_THINK_FRESH_KEYS,
    AgentStates,
    ContextKeys,
    Defaults,
    ErrorMessages,
    HandlerNames,
    HandlerPriorities,
    OrchestratorStates,
    PlanExecuteStates,
    PromptChainStates,
    ReflexionStates,
    REWOOStates,
)
from fsm_llm.agents.tools import make_hitl_checker
from fsm_llm.constants import has_internal_prefix, is_forbidden_context_entry
from fsm_llm.core.definitions import AgentError
from fsm_llm.engine.executables.namespaces import GENERATORS, HANDLERS
from fsm_llm.handlers import HandlerTiming, create_handler
from fsm_llm.tools import require_int_limit

from .chain import make_gate_checker
from .context import RUN_RESULT, HandlerContext
from .executor import AgentHandlers
from .freshness import make_fresh_keys_handler
from .limiters import ThinkTurnLimiter, make_iteration_limiter
from .native_fc import NativeFCHandlers
from .orchestrator import delegate_to_workers
from .plan_execute import (
    make_replan_handler,
    make_result_checker,
    make_step_tracker,
)
from .reflection import (
    EVALUATION_KEYS,
    make_evaluation_handler,
    make_reflection_handler,
)
from .rewoo import execute_plans
from .run_guards import RunEndingErrorHolder

if TYPE_CHECKING:
    from fsm_llm.core.definitions.executable_pattern import HandlerBinding
    from fsm_llm.handlers import BaseHandler

__all__ = ["HOOKS", "LIBRARY", "iteration_check", "register_library"]

#: The ``scope`` keys of a run: the ``AgentHandlers`` shared by the executor and the limiter, and the
#: plain ``ThinkTurnLimiter`` of a limiter bound ``alone``.
_SHARED = "agent_handlers"
_ALONE = "think_limiter"
_HOLDER = "run_ending_error_holder"


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


def _run_holder(ctx: HandlerContext) -> RunEndingErrorHolder:
    """The run's one error holder (in ``ctx.scope``, so the delegator and the hook share it)."""
    holder = ctx.scope.get(_HOLDER)
    if holder is None:
        holder = ctx.scope[_HOLDER] = RunEndingErrorHolder()
    result: RunEndingErrorHolder = holder
    return result


def orchestrator_delegator(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.orchestrator_delegator``: runs the callable ``worker_factory`` per subtask on entry to ``delegate``.

    Reads the option ``max_workers`` (an int of at least 1) and the optional callable ``worker_factory``
    (none: the model fills each subtask). A worker's run-ending error goes into the run's holder, which
    ``agents.raise_run_ending_error`` re-raises after the run.
    """
    max_workers = ctx.options.get("max_workers", Defaults.MAX_WORKERS)
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise AgentError(
            f"agents.orchestrator_delegator needs max_workers >= 1, got {max_workers!r}"
        )
    delegate = ctx.callables.get("worker_factory")
    holder = _run_holder(ctx)
    return [
        create_handler(HandlerNames.ORCHESTRATOR_DELEGATOR)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_entry(*_states(binding, (OrchestratorStates.DELEGATE,)))
        .do(lambda context: delegate_to_workers(context, delegate, max_workers, holder))
    ]


def orchestrator_decision_reset(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.orchestrator_decision_reset``: clears ``all_collected`` on entry to ``orchestrate``."""
    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-009
    # Core extracts all_collected only while it is unset, so round 1's
    # False would stick and collect could never extract a later True.
    # Clear it on entry to orchestrate (the redo state), NOT on entry to
    # collect: that would erase the limiter's forced True (PRE_TRANSITION
    # runs before entry; plan-2026-09-24T045559-3e4eb3e5/D-013).
    return [
        create_handler(HandlerNames.ORCHESTRATOR_DECISION_RESET)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_entry(*_states(binding, (OrchestratorStates.ORCHESTRATE,)))
        .do(lambda _context: {ContextKeys.ALL_COLLECTED: None})
    ]


def raise_run_ending_error(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> Callable[[], None]:
    """``agents.raise_run_ending_error`` (a HOOK, ``at: after_run``): re-raise what a worker left in the holder."""
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-042: re-raise a worker's
    # AgentTimeoutError/BudgetExhaustedError HERE, after the run. Do NOT
    # re-raise inside the delegator (core's handler error mode swallows it)
    # and do NOT keep the holder on self (shared across runs, D-014).
    return _run_holder(ctx).raise_if_set


def rewoo_executor(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.rewoo_executor``: runs the plan with the run's tools on entry to ``execute_plans``."""
    tools = ctx.tools
    if tools is None:
        raise AgentError(
            "agents.rewoo_executor needs the run's tools and the context has none"
        )
    return [
        create_handler(HandlerNames.REWOO_EXECUTOR)
        .with_priority(_priority(binding, HandlerPriorities.TOOL_EXECUTOR))
        .on_state_entry(*_states(binding, (REWOOStates.EXECUTE_PLANS,)))
        .do(lambda context: execute_plans(context, tools))
    ]


def rewoo_cut_off_answer(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> Callable[[], None]:
    """``agents.rewoo_cut_off_answer`` (a HOOK, ``at: after_run``): a plan cut off at ``max_tokens`` ships its cause."""

    # DECISION plan-2026-11-06T093029-0068b055/D-025: a plan cut off at
    # max_tokens ships the cause as the answer, not the `solve` state's
    # filler about evidence that does not exist. The `solve` call has
    # already been made by then (the FSM is linear; adding a branch for the
    # rare cut is not worth a second plan path). Do NOT run the cut plan's
    # prefix and do NOT retry at a higher cap here.
    def rewrite() -> None:
        result = ctx.scope.get(RUN_RESULT)
        if (
            result is not None
            and result.final_context.get(ContextKeys.PLAN_CUT_OFF) is True
        ):
            ctx.scope[RUN_RESULT] = result.model_copy(
                update={
                    "answer": ErrorMessages.REWOO_PLAN_CUT_OFF,
                    "structured_output": None,
                }
            )

    return rewrite


def _on_entry(
    name: str,
    binding: HandlerBinding | None,
    state: str,
    priority: int,
    do: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[BaseHandler]:
    """One entry handler on *state* (the binding's ``states`` and ``priority`` win)."""
    return [
        create_handler(name)
        .with_priority(_priority(binding, priority))
        .on_state_entry(*_states(binding, (state,)))
        .do(do)
    ]


def plan_step_executor(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.plan_step_executor``: bounds the plan and describes the current step on entry to ``execute_step``."""
    return _on_entry(
        HandlerNames.PLAN_STEP_EXECUTOR,
        binding,
        PlanExecuteStates.EXECUTE_STEP,
        HandlerPriorities.TOOL_EXECUTOR,
        make_step_tracker(),
    )


def plan_step_fresh_keys(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.plan_step_fresh_keys``: clears ``step_result`` on entry to ``execute_step`` (core extracts a key only while unset)."""
    return _on_entry(
        HandlerNames.PLAN_STEP_FRESH_KEYS,
        binding,
        PlanExecuteStates.EXECUTE_STEP,
        HandlerPriorities.TOOL_EXECUTOR,
        make_fresh_keys_handler(_keys(binding, (ContextKeys.STEP_RESULT,))),
    )


def plan_step_checker(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.plan_step_checker``: records the step and decides ``step_failed`` on entry to ``check_result``.

    The cap is the option ``max_replans`` (at least 0: no replans). It runs one priority after the
    tool executor, which writes the status it reads.
    """
    # DECISION plan-2026-09-29T103145-06a5ec0a/D-032
    # step_failed is decided on check_result ENTRY, from the status of the
    # tool that just ran (the executor above, D-024 order), and routes the
    # check_result turn that follows. Do NOT move it to PRE_TRANSITION on
    # check_result (core picks the edge before PRE_TRANSITION runs, D-031)
    # and do NOT let the model extract step_failed (a False seed is never
    # re-extracted, and a guessed verdict is not a tool outcome). The
    # max_replans check lives here too, so a failure past the cap routes to
    # synthesize instead of entering replan. See decisions.md D-032.
    limit = require_int_limit(
        "max_replans",
        ctx.options.get("max_replans", Defaults.MAX_REPLANS),
        minimum=0,
    )
    return _on_entry(
        HandlerNames.PLAN_STEP_CHECKER,
        binding,
        PlanExecuteStates.CHECK_RESULT,
        HandlerPriorities.TOOL_EXECUTOR + 1,
        make_result_checker(limit),
    )


def plan_replan_counter(
    ctx: HandlerContext, binding: HandlerBinding | None = None
) -> list[BaseHandler]:
    """``agents.plan_replan_counter``: counts the replan and reopens ``plan_steps`` on entry to ``replan``."""
    return _on_entry(
        HandlerNames.PLAN_REPLANNER,
        binding,
        PlanExecuteStates.REPLAN,
        HandlerPriorities.TOOL_EXECUTOR,
        make_replan_handler(),
    )


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
    ("agents.reflexion_evaluator", reflexion_evaluator),
    ("agents.reflexion_fresh_keys", reflexion_fresh_keys),
    ("agents.reflexion_reflector", reflexion_reflector),
    ("agents.chain_gate_checker", chain_gate_checker),
    ("agents.native_fc", native_fc),
    ("agents.orchestrator_delegator", orchestrator_delegator),
    ("agents.orchestrator_decision_reset", orchestrator_decision_reset),
    ("agents.rewoo_executor", rewoo_executor),
    ("agents.plan_step_executor", plan_step_executor),
    ("agents.plan_step_fresh_keys", plan_step_fresh_keys),
    ("agents.plan_step_checker", plan_step_checker),
    ("agents.plan_replan_counter", plan_replan_counter),
    ("agents.approval_gate", approval_gate),
)

#: The hooks of the same namespace (``run.hooks``): ``factory(ctx, binding) -> Callable[[], None]``.
HOOKS: tuple[tuple[str, Callable[..., Callable[[], None]]], ...] = (
    ("agents.raise_run_ending_error", raise_run_ending_error),
    ("agents.rewoo_cut_off_answer", rewoo_cut_off_answer),
)


def register_library() -> None:
    """Register the library in ``HANDLERS`` and the generator in ``GENERATORS`` (idempotent: a name
    is replaced by its own function)."""
    # imported here: the generator module imports `agents.prompts` and the sibling builders
    from fsm_llm.agents.fsm_definitions.prompt_chain import prompt_chain_fsm

    for name, factory in (*LIBRARY, *HOOKS):
        HANDLERS.register(name, factory, override=True)
    GENERATORS.register("agents.prompt_chain", prompt_chain_fsm, override=True)


register_library()
