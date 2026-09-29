"""
Swarm Pattern — Emergent Agent Coordination.

Agents hand off to each other dynamically by returning the next agent ID,
a handoff message, and optional context. The swarm runner loops until an
agent returns no next_agent or the max handoff limit is reached.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from fsm_llm.logging import logger
from fsm_llm.memory import WorkingMemory

from .base import BaseAgent, strip_caller_context
from .constants import StopReason
from .definitions import AgentConfig, AgentResult, AgentTrace
from .exceptions import AgentTimeoutError, BudgetExhaustedError

# DECISION plan-2026-09-29T103145-06a5ec0a/D-045: per-hop routing outputs.
# The swarm reads them from one agent's result and never forwards them: an
# agent echoes its initial context into its result, so a forwarded
# ``next_agent`` would re-request the same handoff every hop. For the same
# reason a ``handoff_message`` equal to the one the agent was handed is an
# echo, not a new message. Do NOT forward these keys or trust an echoed message.
_ROUTING_KEYS = frozenset({"next_agent", "handoff_context"})


class SwarmAgent(BaseAgent):
    """Emergent coordination pattern where agents hand off to each other.

    Each agent in the swarm runs to completion, and its result is inspected
    for a ``next_agent`` key in ``final_context``.  If present, the named
    agent is run next on the original task, with the handoff message
    (``handoff_message``, default: the previous answer) and the accumulated
    context in its ``initial_context``. ``max_handoffs`` is the number of
    handoffs allowed; a request past it stops the run (``success=False``,
    ``max_iterations``). A handoff to a name not in the swarm stops it with
    ``success=False``, ``no_result`` (the requested work never ran).

    Nothing in the shipped patterns writes ``next_agent``: a member agent
    hands off only when its own code, a handler or a tool writes it into
    its context (swarm transfer tools are deferred, D-018 of plan 06a5ec0a).
    ``handoff_context`` must be a mapping; anything else is ignored with a
    WARNING.

    Example::

        from fsm_llm.agents.swarm import SwarmAgent

        swarm = SwarmAgent(
            agents={"triage": triage_agent, "billing": billing_agent, "support": support_agent},
            entry_agent="triage",
            max_handoffs=5,
        )
        result = swarm.run("I need help with my bill")

    ``memory`` receives two metadata writes per hop (``current_agent``,
    ``handoff_count``). Each ``set`` is atomic, but the pair is not: with
    concurrent ``run()`` calls on one instance the last writer wins and the
    two keys may come from different runs. Nothing in ``run()`` reads them
    back; they exist for callers observing a shared ``WorkingMemory``.
    """

    def __init__(
        self,
        agents: dict[str, BaseAgent],
        entry_agent: str,
        max_handoffs: int = 10,
        memory: WorkingMemory | None = None,
        config: AgentConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        super().__init__(config=config, **api_kwargs)
        if not agents:
            raise ValueError("SwarmAgent requires at least one agent")
        if entry_agent not in agents:
            raise ValueError(
                f"Entry agent '{entry_agent}' not found in agents. "
                f"Available: {sorted(agents.keys())}"
            )
        self._agents = agents
        self._entry_agent = entry_agent
        self._max_handoffs = max_handoffs
        self._memory = memory or WorkingMemory()

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the swarm starting from the entry agent.

        The swarm loops: run agent -> check for handoff -> run next agent.
        Terminates when an agent returns no next_agent or max handoffs reached.
        """
        start_time = time.monotonic()
        context = _without_routing_keys(
            strip_caller_context(initial_context, source="SwarmAgent initial_context")
        )
        context["task"] = task

        current_agent_name = self._entry_agent
        handoff_count = 0
        all_traces: list[dict[str, Any]] = []
        all_tool_calls: list[Any] = []
        last_result: AgentResult | None = None
        # Set when the handoff cap cut off a requested handoff (forced stop).
        capped = False
        unrouted = False
        handoff_chain: list[str] = [current_agent_name]

        logger.info(
            f"Swarm started with entry agent '{current_agent_name}', "
            f"max_handoffs={self._max_handoffs}"
        )

        while True:
            if time.monotonic() - start_time > self.config.timeout_seconds:
                raise AgentTimeoutError(self.config.timeout_seconds)

            agent = self._agents.get(current_agent_name)
            if agent is None:
                logger.error(f"Agent '{current_agent_name}' not found in swarm")
                break

            # Every agent works on the original task; the handoff message
            # travels in its context (PAT-08: it used to replace the task).
            agent_context = {
                **context,
                "_swarm_agent_name": current_agent_name,
                "_swarm_handoff_count": handoff_count,
                "_swarm_history": list(handoff_chain),
            }

            # Store swarm metadata in working memory
            self._memory.set("metadata", "current_agent", current_agent_name)
            self._memory.set("metadata", "handoff_count", handoff_count)

            try:
                result = agent.run(task, initial_context=agent_context)
                last_result = result
            except (BudgetExhaustedError, AgentTimeoutError):
                raise
            except Exception as e:
                logger.error(f"Agent '{current_agent_name}' failed: {e}")
                return AgentResult(
                    answer=f"Swarm failed at agent '{current_agent_name}': {e}",
                    success=False,
                    stop_reason=StopReason.NO_RESULT,
                    trace=AgentTrace(total_iterations=handoff_count),
                    final_context={
                        **context,
                        "_swarm_handoff_chain": handoff_chain,
                        "_swarm_error": str(e),
                    },
                )

            # Record trace and accumulate tool calls from all agents
            all_traces.append(
                {
                    "agent": current_agent_name,
                    "success": result.success,
                    "answer_preview": result.answer[:200] if result.answer else "",
                }
            )
            all_tool_calls.extend(result.trace.tool_calls)

            # Check for handoff
            next_agent = result.final_context.get("next_agent")
            handoff_message = result.final_context.get("handoff_message")
            if handoff_message is None or handoff_message == agent_context.get(
                "handoff_message"
            ):
                # Absent, or the message this agent was handed echoed back
                # through its final_context: the agent's own answer travels.
                handoff_message = result.answer
            handoff_context = result.final_context.get("handoff_context") or {}

            if not next_agent:
                logger.info(
                    f"Swarm completed at agent '{current_agent_name}' "
                    f"after {handoff_count} handoffs"
                )
                break

            # max_handoffs is the number of handoffs allowed: the cap refuses
            # the request past it (PAT-08: it used to allow max_handoffs - 1).
            if handoff_count >= self._max_handoffs:
                logger.warning(
                    f"Swarm reached max handoffs ({self._max_handoffs}), stopping"
                )
                capped = True
                break

            if next_agent not in self._agents:
                logger.error(
                    f"Handoff target '{next_agent}' not found in swarm. "
                    f"Available: {sorted(self._agents.keys())}"
                )
                # D-051 of plan 06a5ec0a: the agent asked for work it could
                # not do itself; dropping that request is not a success.
                unrouted = True
                break

            handoff_count += 1
            # Update context for next agent
            # handoff_context is model-written: it may not seed the next
            # agent's run outputs or approval grant (D-002 of plan 06a5ec0a).
            if isinstance(handoff_context, Mapping):
                context.update(
                    _without_routing_keys(
                        strip_caller_context(
                            handoff_context, source="SwarmAgent handoff_context"
                        )
                    )
                )
            else:
                logger.warning(
                    f"SwarmAgent: agent '{current_agent_name}' returned a "
                    f"non-mapping handoff_context "
                    f"({type(handoff_context).__name__}); ignored"
                )
            context["handoff_message"] = handoff_message
            context["previous_agent"] = current_agent_name
            context["previous_answer"] = result.answer

            current_agent_name = next_agent
            handoff_chain.append(current_agent_name)
            logger.info(
                f"Handoff #{handoff_count}: "
                f"'{handoff_chain[-2]}' → '{current_agent_name}'"
            )

        elapsed = time.monotonic() - start_time

        # Build final result from last agent
        if last_result is None:
            return AgentResult(
                answer="Swarm produced no results",
                success=False,
                stop_reason=StopReason.NO_RESULT,
                trace=AgentTrace(total_iterations=0),
                final_context=context,
            )

        # Merge traces from all agents in the handoff chain
        combined_trace = AgentTrace(
            tool_calls=all_tool_calls,
            total_iterations=handoff_count + 1,
        )

        final_context = {
            **last_result.final_context,
            "_swarm_handoff_chain": handoff_chain,
            "_swarm_handoff_count": handoff_count,
            "_swarm_traces": all_traces,
            "_swarm_elapsed_seconds": elapsed,
        }

        # A requested handoff the cap refused is a forced stop: the last
        # answer ships with success=False (D-011). A handoff to an unknown
        # agent is no_result: the requested work never ran (D-051).
        stop_reason: str | None
        if capped:
            success, stop_reason = False, StopReason.MAX_ITERATIONS
        elif unrouted:
            success, stop_reason = False, StopReason.NO_RESULT
        else:
            success, stop_reason = last_result.success, last_result.stop_reason
        return AgentResult(
            answer=last_result.answer,
            success=success,
            stop_reason=stop_reason,
            trace=combined_trace,
            final_context=final_context,
            structured_output=last_result.structured_output,
        )

    @property
    def agents(self) -> dict[str, BaseAgent]:
        """Return the agent registry."""
        return dict(self._agents)

    @property
    def entry_agent(self) -> str:
        """Return the entry agent name."""
        return self._entry_agent

    def add_agent(self, name: str, agent: BaseAgent) -> SwarmAgent:
        """Add an agent to the swarm. Returns self for chaining."""
        self._agents[name] = agent
        return self

    def _register_handlers(self, api: Any) -> None:
        """No handler registration needed — swarm delegates to sub-agents."""
        pass


def _without_routing_keys(context: dict[str, Any]) -> dict[str, Any]:
    """Return *context* without the per-hop ``_ROUTING_KEYS``."""
    return {k: v for k, v in context.items() if k not in _ROUTING_KEYS}
