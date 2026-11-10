"""Static analysis over FSM graphs: the shared strongly connected components.

A pure, stdlib-only core module (it imports no subpackage and nothing from the
pipeline). The validator's trap detection and the data-flow and liveness
analyses of runtime FSM synthesis (plan 2cefeb61) share one graph primitive
here, so a graph rule lives once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def strongly_connected_components(
    successors: Mapping[str, Sequence[str]], order: Mapping[str, int]
) -> list[list[str]]:
    """Return every strongly connected component of the graph `successors`.

    Contract:
        successors: node -> the nodes it points at. Its keys are the node set;
            a target that is not a key raises `KeyError`. Not mutated.
        order: node -> position, defined for every node. Members of a
            component are sorted by it, so the result never depends on dict
            iteration or discovery order.
    Returns: a list of components (each a list of nodes), every node in
        exactly one, singletons and self-loops included (the caller decides
        which components "loop": more than one member, or a node listed in
        its own successors). Components come in reverse topological order of
        the condensation: a component appears after every component it can
        reach.
    Raises: `KeyError` for a target that is not a node.
    """
    # Iterative on purpose, never recursive: a long chained FSM hits the Python
    # recursion limit (the rule is the D-017 anchor in
    # `FSMValidator._find_trap_components`; a 5,000-node chain is pinned in
    # tests/test_fsm_llm/test_flow_analysis.py).
    # Work frames are (node, index of the next successor to visit).
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []
    for root in successors:
        if root in index:
            continue
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        work = [(root, 0)]
        while work:
            node, i = work[-1]
            if i < len(successors[node]):
                work[-1] = (node, i + 1)
                nxt = successors[node][i]
                if nxt not in successors:
                    raise KeyError(nxt)
                if nxt not in index:
                    index[nxt] = low[nxt] = len(index)
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component, key=order.__getitem__))
    return components
