"""The handler registration helpers of ``BaseAgent`` (``RegistrationMixin``).

The iteration limiter, the tool executor, the ReAct-family think bookkeeping and the lifecycle handlers every
run registers. The method bodies are the old ones (plan d454fd62 step 87).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from fsm_llm import API
from fsm_llm.agents.constants import (
    AgentStates,
    ContextKeys,
    HandlerNames,
    HandlerPriorities,
)
from fsm_llm.agents.handlers import make_fresh_keys_handler
from fsm_llm.agents.summarization import make_observation_summarizer
from fsm_llm.context import ContextCompactor
from fsm_llm.handlers import HandlerTiming
from fsm_llm.logging import logger

from .surface import _AgentSurface


class RegistrationMixin(_AgentSurface):
    """The handler registration helpers of ``BaseAgent``."""

    def _register_iteration_limiter(
        self,
        api: API,
        handler_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register the standard iteration-limiter handler."""
        api.register_handler(
            api.create_handler(HandlerNames.ITERATION_LIMITER)
            .with_priority(HandlerPriorities.ITERATION_LIMITER)
            .at(HandlerTiming.PRE_TRANSITION)
            .do(handler_fn)
        )

    def _register_tool_executor(
        self,
        api: API,
        state: str,
        handler_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register the standard tool-executor handler on a state entry."""
        api.register_handler(
            api.create_handler(HandlerNames.TOOL_EXECUTOR)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(state)
            .do(handler_fn)
        )

    def _register_think_loop_handlers(
        self, api: API, fresh_keys: Sequence[str]
    ) -> None:
        """Register the ReAct-family think bookkeeping (react, reflexion,
        reasoning_react, parallel_react).

        ``fresh_keys`` are the loop values ``think`` produces; they are cleared
        on think entry so think extracts them again (skip-if-set, D-008 of plan
        06a5ec0a). A ``should_terminate`` the limiter forced is cleared too
        (the think conclude edge reads ``max_iterations_reached`` itself), so
        the last think turn gives its own verdict; one a forcing handler
        recorded a reason for is kept (D-051). ``agent_feedback`` is cleared
        on think exit, once the think turn has read it. Under
        :attr:`_hitl_active`, ``should_terminate`` is also cleared on
        ``await_approval`` entry (a forced True is kept).
        """
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-051: keep_limiter_forced
        # is False only here. Do NOT keep a limiter-forced should_terminate on
        # think entry: the model is then never asked on its last turn and a
        # real conclusion there reads as a forced max_iterations stop.
        api.register_handler(
            api.create_handler(HandlerNames.THINK_FRESH_KEYS)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_entry(AgentStates.THINK)
            .do(make_fresh_keys_handler(fresh_keys, keep_limiter_forced=False))
        )
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-029
        # agent_feedback is consumed by think, so it is cleared on think EXIT.
        # Do NOT clear it on think entry with the fresh keys: act writes it and
        # the act -> think transition enters think before think extracts, so an
        # entry clear erases every warning and denial unread (LOOP-06). Do NOT
        # make it a compactor transient key either. See decisions.md D-029.
        api.register_handler(
            api.create_handler(HandlerNames.FEEDBACK_CONSUMED)
            .with_priority(HandlerPriorities.TOOL_EXECUTOR)
            .on_state_exit(AgentStates.THINK)
            .do(make_fresh_keys_handler([ContextKeys.AGENT_FEEDBACK]))
        )
        if self._hitl_active:
            # D-029: an early should_terminate from the gated call's think turn
            # must not beat await_approval -> act; a forced True still routes
            # await_approval -> conclude.
            api.register_handler(
                api.create_handler(HandlerNames.APPROVAL_FRESH_KEYS)
                .with_priority(HandlerPriorities.TOOL_EXECUTOR)
                .on_state_entry(AgentStates.AWAIT_APPROVAL)
                .do(make_fresh_keys_handler([ContextKeys.SHOULD_TERMINATE]))
            )

    def _register_lifecycle_handlers(self, api: API, agent_type: str) -> None:
        """Register the ERROR, context-compactor and summarizer handlers.

        Shared by ``_standard_run``, its stream form and ADaPT, so every run
        path registers the exact same lifecycle handlers.
        """

        def _error_handler(ctx: dict[str, Any]) -> dict[str, Any]:
            logger.warning(
                f"Agent error in {agent_type}: state={ctx.get('_current_state', '?')}"
            )
            return {}

        api.register_handler(
            api.create_handler(HandlerNames.ERROR)
            .with_priority(HandlerPriorities.ERROR)
            .at(HandlerTiming.ERROR)
            .do(_error_handler)
        )

        compactor = ContextCompactor(
            transient_keys={
                ContextKeys.TOOL_RESULT,
                ContextKeys.TOOL_STATUS,
                ContextKeys.TOOL_ERROR,
            },
        )
        api.register_handler(
            api.create_handler(HandlerNames.CONTEXT_COMPACTOR)
            .with_priority(HandlerPriorities.END_CONVERSATION)
            .at(HandlerTiming.PRE_PROCESSING)
            .do(compactor.compact)
        )

        # Opt-in observation summarization (config.auto_summarize_after).
        # No-op for agents that don't accumulate observations.
        if self.config.auto_summarize_after:
            summarizer = make_observation_summarizer(self.config.auto_summarize_after)
            api.register_handler(
                api.create_handler(HandlerNames.OBSERVATION_SUMMARIZER)
                .with_priority(HandlerPriorities.END_CONVERSATION)
                .at(HandlerTiming.PRE_PROCESSING)
                .do(summarizer)
            )
