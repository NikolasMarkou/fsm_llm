"""Graph analysis of ``FSMValidator`` (``GraphMixin``): complexity, cycles, traps, paths.

Layer ``core`` (plan d454fd62, step 75): split out of the old ``fsm_llm.validator``;
the method bodies are unchanged.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from fsm_llm.constants import TRANSITION_CLASSIFICATION_FALLBACK_INTENT
from fsm_llm.core.definitions import MAX_CLASSIFICATION_INTENTS, reachable_from

from .flow import strongly_connected_components

if TYPE_CHECKING:
    from .result import FSMValidationResult

# Caps of the cycle search (a dense graph has factorially many simple cycles):
# the most cycles reported, and the most steps of work (one step is one
# transition looked at, plus the states each start can still return through).
MAX_REPORTED_CYCLES = 100
MAX_CYCLE_SEARCH_STEPS = 200_000


class GraphMixin:
    """The graph analysis of ``FSMValidator`` (stages 4 to 6 of ``validate``): complexity, cycles, paths."""

    if TYPE_CHECKING:
        # Provided by ``FSMValidator`` (declared for mypy only).
        fsm_data: dict[str, Any]
        states: Any
        initial_state: Any
        result: FSMValidationResult
        _reachable_states_cache: set[str] | None

    def _analyze_state_complexity(self):
        """
        Analyze state complexity (number of transitions, conditions, etc.).

        Identifies states with many outgoing transitions that might be overly complex.
        """
        complex_states = []

        for state_id, state in self.states.items():
            transitions = state.get("transitions", [])

            # Consider states with more than 3 transitions as potentially complex
            # This threshold could be configurable in a future version
            if len(transitions) > 3:
                complex_states.append((state_id, len(transitions)))

        if complex_states:
            states_str = ", ".join(
                [f"'{s}' ({t} transitions)" for s, t in complex_states]
            )
            self.result.add_info(f"Complex states with many transitions: {states_str}")

        # A tie the classifier schema cannot hold stays in the state at run
        # time (D-006 of plan-2026-11-05T134105-5ed71cf2): the schema takes
        # MAX_CLASSIFICATION_INTENTS intents, one of them the fallback.
        max_targets = MAX_CLASSIFICATION_INTENTS - 1
        for state_id, state in self.states.items():
            by_priority: dict[Any, set[Any]] = {}
            for t in state.get("transitions", []):
                by_priority.setdefault(t.get("priority", 100), set()).add(
                    t.get("target_state")
                )
            for priority, targets in by_priority.items():
                if (
                    len(targets) > max_targets
                    or TRANSITION_CLASSIFICATION_FALLBACK_INTENT in targets
                ):
                    self.result.add_warning(
                        f"State '{state_id}': {len(targets)} transitions at "
                        f"priority {priority} go to distinct targets; if they "
                        f"tie the classifier cannot hold them (at most {max_targets}, none "
                        f"named '{TRANSITION_CLASSIFICATION_FALLBACK_INTENT}') "
                        f"and the conversation stays in the state"
                    )

    def _detect_cycles(self):
        """
        Report cycles (INFO) and trap regions (WARNING).

        A trap is a strongly connected component of reachable states that
        loops (two or more states, or a self-loop) and from which no terminal
        state can be reached: a conversation that enters it can never end.
        One WARNING per trap component; cycles outside traps are INFO.
        """
        cycles, capped = self._search_cycles()
        traps = self._find_trap_components()
        trapped = {state_id for component in traps for state_id in component}

        if not cycles and not traps and not capped:
            self.result.add_info("No cycles detected in the FSM")
            return

        for cycle in cycles:
            if not set(cycle) <= trapped:
                self.result.add_info(f"Cycle detected: {' → '.join(cycle)}")
        if capped:
            self.result.add_info(
                f"Cycle search stopped at its cap ({MAX_REPORTED_CYCLES} cycles or "
                f"{MAX_CYCLE_SEARCH_STEPS} steps): more cycles may exist than are listed"
            )
        for component in traps:
            states_str = ", ".join(f"'{state_id}'" for state_id in component)
            self.result.add_warning(
                f"Trap cycle with no path to a terminal state: {states_str} "
                "(a conversation that enters it can never end)"
            )

    def _find_trap_components(self) -> list[list[str]]:
        """Return every looping SCC of reachable states that cannot reach a
        terminal state, members and components in state-definition order.

        # DECISION plan-2026-09-21T203800-8a03483a/D-017 (B7)
        # Do NOT go back to checking each simple cycle for an edge leaving
        # it: two interlocking cycles (a<->b, b<->c) each "escape" into the
        # other, so neither was flagged although {a, b, c} has no way out.
        # The trap set is "reachable minus can-reach-a-terminal" (reverse
        # BFS), grouped by SCC. Tarjan is iterative on purpose: do NOT make
        # it recursive (long chained FSMs hit the Python recursion limit, the
        # same reason `_find_cycles` is iterative). See decisions.md D-017.
        """
        order = {state_id: i for i, state_id in enumerate(self.states)}
        adjacency: dict[str, list[str]] = {
            state_id: [
                target
                for transition in state.get("transitions", [])
                if (target := transition.get("target_state")) in order
            ]
            for state_id, state in self.states.items()
        }

        reverse: dict[str, list[str]] = {state_id: [] for state_id in adjacency}
        for state_id, targets in adjacency.items():
            for target in targets:
                reverse[target].append(state_id)
        can_finish = reachable_from(reverse, self._get_terminal_states())

        reachable = self._get_reachable_states()
        trapped = [s for s in self.states if s in reachable and s not in can_finish]
        trapped_set = set(trapped)
        successors = {s: [t for t in adjacency[s] if t in trapped_set] for s in trapped}

        components = strongly_connected_components(successors, order)
        looping = [c for c in components if len(c) > 1 or c[0] in successors[c[0]]]
        return sorted(looping, key=lambda c: order[c[0]])

    def _analyze_paths(self):
        """
        Analyze possible paths through the FSM to terminal states.

        Identifies the shortest path from initial state to each terminal state,
        helping understand the flow structure of the FSM.
        """
        terminal_states = self._get_terminal_states()

        if not terminal_states:
            return  # This would be caught by the terminal state validation

        # Find the shortest path to each terminal state using BFS
        for terminal in terminal_states:
            shortest_path = self._find_shortest_path(self.initial_state, terminal)
            # No path: caught by the terminal-state reachability check.
            if shortest_path:
                path_str = " → ".join(shortest_path)
                self.result.add_info(
                    f"Shortest path to terminal state '{terminal}': {path_str}"
                )

    def _get_reachable_states(self) -> set[str]:
        """
        Get all states reachable from the initial state (cached).

        Uses a breadth-first search approach to find all states that can be
        reached by following transitions from the initial state.

        Returns:
            Set of state IDs that are reachable from the initial state
        """
        if self._reachable_states_cache is not None:
            return self._reachable_states_cache
        adjacency = {
            state_id: [
                target
                for transition in state.get("transitions", [])
                if (target := transition.get("target_state", ""))
            ]
            for state_id, state in self.states.items()
        }
        self._reachable_states_cache = reachable_from(adjacency, [self.initial_state])
        return self._reachable_states_cache

    def _get_terminal_states(self) -> set[str]:
        """
        Get all terminal states (states with no outgoing transitions).

        Terminal states are essential for FSMs to have proper end points.

        Returns:
            Set of state IDs that have no outgoing transitions
        """
        return {
            state_id
            for state_id, state in self.states.items()
            if not state.get("transitions", [])
        }

    def _find_cycles(self) -> list[list[str]]:
        """
        Find every simple cycle reachable from the initial state, up to the caps.

        A cycle is a closed path that repeats no state; a self-loop is a cycle
        of one state. Each cycle is listed once, starting at its member the
        depth-first walk from the initial state reaches first. The search is
        capped: at most ``MAX_REPORTED_CYCLES`` cycles and ``MAX_CYCLE_SEARCH_STEPS``
        steps of work (see ``_search_cycles`` for the capped flag).

        Returns:
            List of cycles, where each cycle is a list of state IDs
        """
        return self._search_cycles()[0]

    def _search_cycles(self) -> tuple[list[list[str]], bool]:
        """``(cycles, capped)``: the simple cycles reachable from the initial
        state and whether a cap stopped the search before it was complete.

        Contract: ``cycles`` holds at most ``MAX_REPORTED_CYCLES`` entries, in
        order of their start state (the DFS discovery order from the initial
        state), then in the order of the transitions. ``capped`` is True only
        when more cycles may exist (a further cycle was found past the report
        cap, or the work cap of ``MAX_CYCLE_SEARCH_STEPS`` was spent). Never
        raises on a state with a missing or unknown target.
        """
        # DECISION plan_2026-05-29_73c30922/D-002 [STALE] (RWM-002)
        # Iterative everywhere (an explicit stack, never a recursive dfs()) so an
        # FSM of ~1000+ chained states does not hit the Python recursion limit.
        # Start-ordered search: each cycle is found once, from its member of
        # smallest order, over the states of larger order only. Do NOT skip a
        # target because it was visited before (the old `finished` set): that
        # skip dropped real cycles (a second path into a finished state).
        # Do NOT remove the two caps: a dense graph has factorially many cycles.
        adjacency: dict[str, list[str]] = {}
        for state_id, state in self.states.items():
            targets: list[str] = []
            for transition in state.get("transitions", []):
                target = transition.get("target_state", "")
                if target in self.states and target not in targets:
                    targets.append(target)
            adjacency[state_id] = targets

        # Discovery order of a depth-first walk from the initial state.
        order: dict[str, int] = {}
        pending = [self.initial_state]
        while pending:
            node = pending.pop()
            if node in order or node not in adjacency:
                continue
            order[node] = len(order)
            pending.extend(reversed(adjacency[node]))
        reverse: dict[str, list[str]] = {node: [] for node in order}
        for node in order:
            for target in adjacency[node]:
                reverse[target].append(node)

        cycles: list[list[str]] = []
        steps = 0
        for start in order:
            floor = order[start]
            # A cycle needs a way back into `start` from a state of larger order.
            if not any(order[p] >= floor for p in reverse[start]):
                continue
            # Only states of larger order that can get back to `start` matter.
            allowed = {start}
            queue = [start]
            while queue:
                for pred in reverse[queue.pop()]:
                    steps += 1
                    if order[pred] >= floor and pred not in allowed:
                        allowed.add(pred)
                        queue.append(pred)
            if steps > MAX_CYCLE_SEARCH_STEPS:
                return cycles, True
            path = [start]
            on_path = {start}
            frames = [iter(adjacency[start])]
            while frames:
                for target in frames[-1]:
                    steps += 1
                    if steps > MAX_CYCLE_SEARCH_STEPS:
                        return cycles, True
                    if target == start:
                        if len(cycles) == MAX_REPORTED_CYCLES:
                            return cycles, True
                        cycles.append(list(path))
                    elif target in allowed and target not in on_path:
                        path.append(target)
                        on_path.add(target)
                        frames.append(iter(adjacency[target]))
                        break
                else:
                    frames.pop()
                    on_path.discard(path.pop())
        return cycles, False

    def _find_shortest_path(self, start: str, end: str) -> list[str] | None:
        """
        Find the shortest path from start to end using BFS.

        This method is used to find the most direct route through the FSM
        from initial state to terminal states.

        Args:
            start: Starting state ID
            end: Target state ID

        Returns:
            List of state IDs representing the shortest path, or None if no path exists
        """
        if start == end:
            return [start]  # Already at the target

        # Use breadth-first search to find shortest path
        queue = deque([(start, [start])])  # (current_state, path_so_far)
        visited = {start}

        while queue:
            node, path = queue.popleft()
            state = self.states.get(node, {})
            transitions = state.get("transitions", [])

            for transition in transitions:
                target = transition.get("target_state", "")
                if not target:
                    continue

                # Check if we've reached the destination
                if target == end:
                    return [*path, target]  # Return the complete path

                # Otherwise continue the search if we haven't visited this state yet
                if target not in visited:
                    visited.add(target)
                    queue.append((target, [*path, target]))

        return None  # No path found
