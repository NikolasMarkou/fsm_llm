"""
Graph-Based Agent Orchestration.

Wires agents as nodes in a directed graph with conditional edges.
Each node runs an agent, and its AgentResult.final_context becomes
the edge state that condition functions evaluate against.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from fsm_llm.logging import logger

from .base import BaseAgent, strip_caller_context
from .constants import StopReason
from .definitions import AgentConfig, AgentResult, AgentTrace
from .exceptions import AgentTimeoutError, BudgetExhaustedError


class AgentGraphBuilder:
    """Builder for constructing an AgentGraph.

    Example::

        graph = (
            AgentGraphBuilder()
            .add_node("classifier", classifier_agent)
            .add_node("billing", billing_agent)
            .add_node("support", support_agent)
            .add_edge("classifier", "billing", condition=lambda ctx: ctx.get("intent") == "billing")
            .add_edge("classifier", "support", condition=lambda ctx: ctx.get("intent") == "support")
            .set_entry("classifier")
            .build()
        )
        result = graph.run("I need help with my invoice")
    """

    def __init__(self) -> None:
        self._nodes: dict[str, BaseAgent] = {}
        self._edges: list[tuple[str, str, Callable[[dict], bool] | None]] = []
        self._entry: str | None = None

    def add_node(self, name: str, agent: BaseAgent) -> AgentGraphBuilder:
        """Add an agent as a named node in the graph."""
        self._nodes[name] = agent
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Callable[[dict[str, Any]], bool] | None = None,
    ) -> AgentGraphBuilder:
        """Add a directed edge from source to target.

        Args:
            source: Source node name.
            target: Target node name.
            condition: Optional function that receives the source agent's
                final_context and returns True if this edge should be taken.
                If None, the edge is unconditional (always taken).
        """
        self._edges.append((source, target, condition))
        return self

    def set_entry(self, name: str) -> AgentGraphBuilder:
        """Set the entry node for the graph."""
        self._entry = name
        return self

    def build(self) -> AgentGraph:
        """Build and validate the AgentGraph.

        Raises:
            ValueError: No entry, an unknown entry or edge endpoint, or a cycle
                (raised by ``AgentGraph``).
        """
        if self._entry is None:
            raise ValueError("Entry node must be set with set_entry()")
        if self._entry not in self._nodes:
            raise ValueError(
                f"Entry node '{self._entry}' not found in nodes. "
                f"Available: {sorted(self._nodes.keys())}"
            )

        # Validate edges reference existing nodes
        for source, target, _ in self._edges:
            if source not in self._nodes:
                raise ValueError(f"Edge source '{source}' not in nodes")
            if target not in self._nodes:
                raise ValueError(f"Edge target '{target}' not in nodes")

        # Build adjacency list
        adjacency: dict[str, list[tuple[str, Callable | None]]] = defaultdict(list)
        for source, target, condition in self._edges:
            adjacency[source].append((target, condition))

        return AgentGraph(
            nodes=dict(self._nodes),
            adjacency=dict(adjacency),
            entry=self._entry,
        )


class AgentGraph:
    """A directed acyclic graph of agents with conditional edges.

    Created via AgentGraphBuilder. Executes agents in topological order,
    evaluating edge conditions to determine the execution path.

    Raises:
        ValueError: The edges form a cycle.
    """

    def __init__(
        self,
        nodes: dict[str, BaseAgent],
        adjacency: dict[str, list[tuple[str, Callable | None]]],
        entry: str,
    ) -> None:
        self._nodes = nodes
        self._adjacency = adjacency
        self._entry = entry
        self._order = _topological_order(nodes, adjacency)

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Execute the graph starting from the entry node.

        Nodes run in topological order. A node other than the entry runs
        once every predecessor has been decided, and only if at least one
        predecessor that ran has a satisfied edge to it (``condition`` is
        None or returns True against that predecessor's ``final_context``).
        Its ``initial_context`` merges those predecessors' contexts in
        topological order (a later one wins a shared key), each minus
        ``RUN_OUTPUT_KEYS``, so a node starts its own run fresh. A failed
        node takes no outgoing edge. The answer is the last executed node's,
        which the order makes a sink of the executed subgraph.
        """
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-044: Kahn order, each node
        # once after all its predecessors, contexts merged. Do NOT go back to a
        # BFS with first-arrival contexts: a convergence node ran before a
        # longer branch reached it, with one branch's context, and the answer
        # came from whichever node the queue popped last (PAT-07).
        start_time = time.monotonic()
        context = strip_caller_context(
            initial_context, source="AgentGraph initial_context"
        )
        context["task"] = task

        results: dict[str, AgentResult] = {}
        execution_order: list[str] = []
        # Context each executed node hands to its successors.
        outgoing: dict[str, dict[str, Any]] = {}
        # target -> predecessors that ran and took their edge to it.
        activated_by: dict[str, list[str]] = defaultdict(list)

        for node_name in self._order:
            if node_name == self._entry:
                node_context = context
            elif activated_by.get(node_name):
                node_context = {}
                for source in activated_by[node_name]:
                    node_context.update(outgoing[source])
                node_context["task"] = task
            else:
                continue

            agent = self._nodes[node_name]
            logger.info(f"AgentGraph executing node '{node_name}'")

            try:
                result = agent.run(task, initial_context=node_context)
                results[node_name] = result
                execution_order.append(node_name)
            except (BudgetExhaustedError, AgentTimeoutError):
                # Hard limits must propagate, not be downgraded to a failed node.
                # Mirrors SwarmAgent (swarm.py) so the whole graph aborts when a
                # node exhausts its budget or times out (AI3-001).
                raise
            except Exception as e:
                logger.error(f"AgentGraph node '{node_name}' failed: {e}")
                results[node_name] = AgentResult(
                    answer=f"Node '{node_name}' failed: {e}",
                    success=False,
                    stop_reason=StopReason.NO_RESULT,
                    trace=AgentTrace(),
                    final_context=node_context,
                )
                execution_order.append(node_name)
                continue

            # The source node's run outputs (final_answer, should_terminate,
            # observation_count, ...) must not seed a successor's run (D-002
            # of plan 06a5ec0a); dropping them is expected here, so log at DEBUG.
            outgoing[node_name] = strip_caller_context(
                {**node_context, **result.final_context},
                source=f"AgentGraph node '{node_name}' output",
                warn=False,
            )

            # Evaluate outgoing edges
            for target, condition in self._adjacency.get(node_name, []):
                try:
                    take_edge = condition is None or condition(result.final_context)
                except Exception as e:
                    logger.warning(
                        f"AgentGraph edge condition '{node_name}'->'{target}' "
                        f"raised {type(e).__name__}: {e}; skipping edge"
                    )
                    continue
                if take_edge and node_name not in activated_by[target]:
                    activated_by[target].append(node_name)

        elapsed = time.monotonic() - start_time

        # Combine results — last executed node's answer is the final answer
        if not results:
            return AgentResult(
                answer="No agents executed",
                success=False,
                stop_reason=StopReason.NO_RESULT,
                trace=AgentTrace(),
                final_context=context,
            )

        last_node = execution_order[-1]
        last_result = results[last_node]

        # Merge all tool calls
        all_tool_calls = []
        for name in execution_order:
            all_tool_calls.extend(results[name].trace.tool_calls)

        combined_trace = AgentTrace(
            tool_calls=all_tool_calls,
            total_iterations=sum(r.trace.total_iterations for r in results.values()),
        )

        final_context = {
            **last_result.final_context,
            "_graph_execution_order": execution_order,
            "_graph_node_results": {
                name: {
                    "answer": r.answer,
                    "success": r.success,
                }
                for name, r in results.items()
            },
            "_graph_elapsed_seconds": elapsed,
        }

        # Success needs every executed node; the reason is the first failed
        # node's in execution order, else the answering node's.
        failed = [results[n] for n in execution_order if not results[n].success]
        return AgentResult(
            answer=last_result.answer,
            success=not failed,
            stop_reason=(
                failed[0].stop_reason or StopReason.NO_RESULT
                if failed
                else last_result.stop_reason
            ),
            trace=combined_trace,
            final_context=final_context,
            structured_output=last_result.structured_output,
        )

    @property
    def nodes(self) -> list[str]:
        """Return all node names."""
        return list(self._nodes.keys())

    @property
    def entry(self) -> str:
        """Return the entry node name."""
        return self._entry

    def get_edges(self, node: str) -> list[str]:
        """Return target names for edges from the given node."""
        return [target for target, _ in self._adjacency.get(node, [])]

    def get_terminal_nodes(self) -> list[str]:
        """Return nodes with no outgoing edges."""
        return [name for name in self._nodes if not self._adjacency.get(name)]


def _topological_order(
    nodes: dict[str, Any],
    adjacency: dict[str, list[tuple[str, Callable | None]]],
) -> list[str]:
    """Kahn's topological order of *nodes*, ties broken by insertion order.

    Iterative, so a long chain cannot hit the recursion limit (AG-001).

    Raises:
        ValueError: The edges form a cycle (some node never reaches in-degree 0).
    """
    in_degree = {name: 0 for name in nodes}
    for source in nodes:
        for target, _ in adjacency.get(source, []):
            in_degree[target] += 1
    ready = deque(name for name, degree in in_degree.items() if degree == 0)
    order: list[str] = []
    while ready:
        name = ready.popleft()
        order.append(name)
        for target, _ in adjacency.get(name, []):
            in_degree[target] -= 1
            if in_degree[target] == 0:
                ready.append(target)
    if len(order) != len(nodes):
        cyclic = sorted(name for name in nodes if name not in order)
        raise ValueError(
            f"Agent graph contains cycles (nodes {cyclic}). "
            "Use SwarmAgent for cyclic coordination patterns."
        )
    return order
