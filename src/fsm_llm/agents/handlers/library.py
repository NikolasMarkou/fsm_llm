"""The handler library: named FACTORIES a pattern file binds (plan a555dcfe step 7, D-013).

``agents.<name>`` -> ``factory(ctx, binding=None) -> list[Handler]``, registered at import in the
``HANDLERS`` namespace of ``fsm_llm.engine.executables.namespaces``. This module holds the two tables
(``LIBRARY``, ``HOOKS``) and ``register_library``; the factories live in four modules by family:
``library_react`` (the react family, ``agents.approval_gate``, the Reflexion three, the prompt-chain gate and
the native function-calling group ``agents.native_fc``), ``library_workers`` (the orchestrator pair, the REWOO
executor and the two HOOKS ``agents.raise_run_ending_error`` and ``agents.rewoo_cut_off_answer``),
``library_stepwise`` (the Plan-and-Execute four, the Maker-Checker five, the Debate three and the Evaluator-Optimizer three) and ``library_support`` (the
binding helpers they share). Each is its own function (a ``NameRegistry`` refuses one function under two
names). The handler bodies are the existing ones (``executor``, ``limiters``, ``freshness``,
``tools.make_hitl_checker``, and one module per pattern); a factory only builds the handler around them,
with the name, timing, state filter and priority the live registration of ``base/registration.py`` and
``base/hitl_gate.py`` uses, so a pattern file that binds them registers the same handlers in the same order.

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

from collections.abc import Callable
from typing import TYPE_CHECKING

from fsm_llm.engine.executables.namespaces import GENERATORS, HANDLERS

from .library_react import (
    approval_fresh_keys,
    approval_gate,
    chain_gate_checker,
    feedback_consumed,
    iteration_limiter,
    native_fc,
    reflexion_evaluator,
    reflexion_fresh_keys,
    reflexion_reflector,
    think_fresh_keys,
    think_turn_limiter,
    tool_executor,
)
from .library_stepwise import (
    debate_consensus_reset,
    debate_judge,
    debate_limiter,
    evalopt_evaluator,
    force_pass,
    maker_checker_checker,
    plan_replan_counter,
    plan_step_checker,
    plan_step_executor,
    plan_step_fresh_keys,
    refine_entry,
    refine_exit,
    revise_entry,
    revise_exit,
    revision_count,
)
from .library_workers import (
    orchestrator_decision_reset,
    orchestrator_delegator,
    raise_run_ending_error,
    rewoo_cut_off_answer,
    rewoo_executor,
)

if TYPE_CHECKING:
    from fsm_llm.handlers import BaseHandler

__all__ = ["HOOKS", "LIBRARY", "register_library"]


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
    ("agents.revision_count", revision_count),
    ("agents.maker_checker_checker", maker_checker_checker),
    ("agents.revise_entry", revise_entry),
    ("agents.revise_exit", revise_exit),
    ("agents.force_pass", force_pass),
    ("agents.debate_judge", debate_judge),
    ("agents.debate_consensus_reset", debate_consensus_reset),
    ("agents.debate_limiter", debate_limiter),
    ("agents.evalopt_evaluator", evalopt_evaluator),
    ("agents.refine_entry", refine_entry),
    ("agents.refine_exit", refine_exit),
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
