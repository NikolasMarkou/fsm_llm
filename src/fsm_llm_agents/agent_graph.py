from __future__ import annotations

"""
Graph-Based Agent Orchestration.

Wires agents as nodes in a directed graph with conditional edges.
Each node runs an agent, and its AgentResult.final_context becomes
the edge state that condition functions evaluate against.
"""

import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from fsm_llm.logging import logger

from .base import BaseAgent
from .definitions import AgentConfig, AgentResult, AgentTrace


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
        """Build and validate the AgentGraph."""
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

        # Check for cycles
        if self._has_cycles(adjacency):
            raise ValueError(
                "Agent graph contains cycles. "
                "Use SwarmAgent for cyclic coordination patterns."
            )

        return AgentGraph(
            nodes=dict(self._nodes),
            adjacency=dict(adjacency),
            entry=self._entry,
        )

    def _has_cycles(
        self,
        adjacency: dict[str, list[tuple[str, Callable | None]]],
    ) -> bool:
        """Detect cycles using DFS with a recursion stack."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {name: WHITE for name in self._nodes}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for target, _ in adjacency.get(node, []):
                if color[target] == GRAY:
                    return True
                if color[target] == WHITE and dfs(target):
                    return True
            color[node] = BLACK
            return False

        for node in self._nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False


class AgentGraph:
    """A directed acyclic graph of agents with conditional edges.

    Created via AgentGraphBuilder. Executes agents in topological order,
    evaluating edge conditions to determine the execution path.
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

    def run(
        self,
        task: str,
        initial_context: dict[str, Any] | None = None,
        config: AgentConfig | None = None,
    ) -> AgentResult:
        """Execute the graph starting from the entry node.

        Follows edges whose conditions evaluate to True against the
        source node's final_context. When multiple edges match, all
        targets are queued for execution.

        Note:
            Diamond convergence: When multiple paths converge on a
            single node, that node executes once using the context from
            whichever path reaches it first (BFS order). Contexts from
            later paths are not merged. If you need all upstream
            contexts merged before a convergence node, restructure as
            sequential edges or use an explicit merge node.
        """
        start_time = time.monotonic()
        context = dict(initial_context or {})
        context["task"] = task

        results: dict[str, AgentResult] = {}
        execution_order: list[str] = []

        # BFS-style execution from entry
        queue: deque[tuple[str, dict[str, Any]]] = deque()
        queue.append((self._entry, context))
        visited: set[str] = set()

        while queue:
            node_name, node_context = queue.popleft()
            if node_name in visited:
                continue
            visited.add(node_name)

            agent = self._nodes[node_name]
            logger.info(f"AgentGraph executing node '{node_name}'")

            try:
                result = agent.run(task, initial_context=node_context)
                results[node_name] = result
                execution_order.append(node_name)
            except Exception as e:
                logger.error(f"AgentGraph node '{node_name}' failed: {e}")
                results[node_name] = AgentResult(
                    answer=f"Node '{node_name}' failed: {e}",
                    success=False,
                    trace=AgentTrace(),
                    final_context=node_context,
                )
                execution_order.append(node_name)
                continue

            # Evaluate outgoing edges
            edges = self._adjacency.get(node_name, [])
            for target, condition in edges:
                if target in visited:
                    continue
                if condition is None or condition(result.final_context):
                    # Pass the source's final_context merged with original context
                    next_context = {**node_context, **result.final_context}
                    queue.append((target, next_context))

        elapsed = time.monotonic() - start_time

        # Combine results — last executed node's answer is the final answer
        if not results:
            return AgentResult(
                answer="No agents executed",
                success=False,
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
                    "answer": r.answer[:200],
                    "success": r.success,
                }
                for name, r in results.items()
            },
            "_graph_elapsed_seconds": elapsed,
        }

        return AgentResult(
            answer=last_result.answer,
            success=all(r.success for r in results.values()),
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
