from __future__ import annotations

"""
Swarm Pattern — Emergent Agent Coordination.

Agents hand off to each other dynamically by returning the next agent ID,
a handoff message, and optional context. The swarm runner loops until an
agent returns no next_agent or the max handoff limit is reached.
"""

import time
from typing import Any

from fsm_llm.logging import logger
from fsm_llm.memory import WorkingMemory

from .base import BaseAgent
from .definitions import AgentConfig, AgentResult, AgentTrace


class SwarmAgent(BaseAgent):
    """Emergent coordination pattern where agents hand off to each other.

    Each agent in the swarm runs to completion, and its result is inspected
    for a ``next_agent`` key in ``final_context``.  If present, the named
    agent is run next with the handoff message and accumulated context.

    Example::

        from fsm_llm_agents.swarm import SwarmAgent

        swarm = SwarmAgent(
            agents={"triage": triage_agent, "billing": billing_agent, "support": support_agent},
            entry_agent="triage",
            max_handoffs=5,
        )
        result = swarm.run("I need help with my bill")
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
        context = dict(initial_context or {})
        context["task"] = task

        current_agent_name = self._entry_agent
        handoff_count = 0
        all_traces: list[dict[str, Any]] = []
        last_result: AgentResult | None = None
        handoff_chain: list[str] = [current_agent_name]

        logger.info(
            f"Swarm started with entry agent '{current_agent_name}', "
            f"max_handoffs={self._max_handoffs}"
        )

        while True:
            agent = self._agents.get(current_agent_name)
            if agent is None:
                logger.error(f"Agent '{current_agent_name}' not found in swarm")
                break

            # Run the current agent
            current_task = context.get("handoff_message", task)
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
                result = agent.run(current_task, initial_context=agent_context)
                last_result = result
            except Exception as e:
                logger.error(f"Agent '{current_agent_name}' failed: {e}")
                return AgentResult(
                    answer=f"Swarm failed at agent '{current_agent_name}': {e}",
                    success=False,
                    trace=AgentTrace(total_iterations=handoff_count),
                    final_context={
                        **context,
                        "_swarm_handoff_chain": handoff_chain,
                        "_swarm_error": str(e),
                    },
                )

            # Record trace
            all_traces.append(
                {
                    "agent": current_agent_name,
                    "success": result.success,
                    "answer_preview": result.answer[:200] if result.answer else "",
                }
            )

            # Check for handoff
            next_agent = result.final_context.get("next_agent")
            handoff_message = result.final_context.get("handoff_message", result.answer)
            handoff_context = result.final_context.get("handoff_context", {})

            if not next_agent:
                logger.info(
                    f"Swarm completed at agent '{current_agent_name}' "
                    f"after {handoff_count} handoffs"
                )
                break

            handoff_count += 1
            if handoff_count > self._max_handoffs:
                logger.warning(
                    f"Swarm reached max handoffs ({self._max_handoffs}), stopping"
                )
                break

            if next_agent not in self._agents:
                logger.error(
                    f"Handoff target '{next_agent}' not found in swarm. "
                    f"Available: {sorted(self._agents.keys())}"
                )
                break

            # Update context for next agent
            context.update(handoff_context)
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
                trace=AgentTrace(total_iterations=0),
                final_context=context,
            )

        # Merge traces
        combined_trace = AgentTrace(
            tool_calls=last_result.trace.tool_calls,
            total_iterations=handoff_count + 1,
        )

        final_context = {
            **last_result.final_context,
            "_swarm_handoff_chain": handoff_chain,
            "_swarm_handoff_count": handoff_count,
            "_swarm_traces": all_traces,
            "_swarm_elapsed_seconds": elapsed,
        }

        return AgentResult(
            answer=last_result.answer,
            success=last_result.success,
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
