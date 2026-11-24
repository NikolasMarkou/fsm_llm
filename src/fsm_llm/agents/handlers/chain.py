"""The gate of one prompt-chain step (plan a555dcfe step 11; body moved from ``prompt_chain.py``).

``make_gate_checker`` builds the entry handler of the state after a step: it records the step's
result and runs that step's validation function, if it has one. It is wrapped by the library
factory ``agents.chain_gate_checker`` (not in ``handlers.__all__``).

Interface contract: ``make_gate_checker(step_index, previous_name, validate, last) -> body``:
``step_index`` is the 1-based index the state after the step records (``len(chain)`` on output),
``previous_name`` names the step before the state (logs only), ``validate`` is its validation
function or ``None``, ``last`` is true on the output state (the last step's result stays as the
success key). The body maps a context to the delta to write and raises only what ``validate``
raises.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fsm_llm.agents.constants import ContextKeys, StopReason
from fsm_llm.logging import logger

from .freshness import make_fresh_keys_handler


def make_gate_checker(
    step_index: int,
    previous_name: str,
    validate: Callable[[dict[str, Any]], bool] | None,
    last: bool,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The entry handler body of the state after step ``step_index - 1`` (see the module text)."""
    # The next step extracts its own result (skip-if-set); on output the last step's result stays
    # as the success key.
    refresh = None if last else make_fresh_keys_handler([ContextKeys.CHAIN_STEP_RESULT])

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-038
    # The gate runs on ENTRY to the next state, after core recorded the
    # step's chain_step_result; a failure writes forced_stop_reason
    # gate_failed and KEEPS chain_step_result, so the next state's turn
    # skips its extraction (skip-if-set) and its gate edge routes to
    # output. Do NOT move the gate to PRE_TRANSITION of the step (core
    # has already chosen the edge, D-031) or to CONTEXT_UPDATE (never
    # fires on a null extraction), and do NOT route on gate_passed:
    # caller context can seed it, forced_stop_reason it cannot.
    def check_gate(context: dict[str, Any]) -> dict[str, Any]:
        if context.get(ContextKeys.FORCED_STOP_REASON) == StopReason.GATE_FAILED:
            return {}  # an earlier gate stopped the chain
        updates: dict[str, Any] = {ContextKeys.CHAIN_STEP_INDEX: step_index}
        step_result = context.get(ContextKeys.CHAIN_STEP_RESULT)
        if step_result is not None:
            chain_results = list(context.get(ContextKeys.CHAIN_RESULTS, []))
            chain_results.append(step_result)
            updates[ContextKeys.CHAIN_RESULTS] = chain_results

        if validate is not None:
            passed = validate(context)
            updates[ContextKeys.GATE_PASSED] = passed
            if not passed:
                logger.warning(
                    f"Gate failed at step '{previous_name}', terminating chain"
                )
                updates[ContextKeys.FORCED_STOP_REASON] = StopReason.GATE_FAILED.value
                return updates

        if refresh is not None:
            updates.update(refresh(context))
        return updates

    return check_gate
