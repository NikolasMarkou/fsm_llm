"""Graph data (``FSMGraph``) and the Mermaid and DOT exporters.

Layer ``core`` (plan d454fd62, step 76): split out of the old ``fsm_llm.visualizer``.
``to_mermaid``, ``to_dot`` and the monitor's visualization all read ``build_fsm_graph``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from fsm_llm.core.definitions import FSMDefinition, Transition

from .shapes import (
    _DEFAULT_NAME,
    _is_terminal,
    _one_line,
    _require_definition_shapes,
    _require_initial_state,
    _require_valid_transition_targets,
    _transitions_of,
)


class FSMGraphNode(BaseModel):
    """One state of an FSM graph. ``is_terminal``: the state has no transitions."""

    model_config = ConfigDict(frozen=True)

    id: str
    description: str = ""
    purpose: str = ""
    is_initial: bool = False
    is_terminal: bool = False


class FSMGraphEdge(BaseModel):
    """One transition of an FSM graph, from state ``source`` to state ``target``."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    description: str = ""
    priority: int


class FSMGraph(BaseModel):
    """The graph of one FSM definition: what ``to_mermaid`` and ``to_dot`` render.

    ``nodes`` and ``edges`` keep the definition's order (states, then each
    state's transitions). Exactly one node has ``is_initial``. Frozen.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    initial_state: str
    nodes: tuple[FSMGraphNode, ...]
    edges: tuple[FSMGraphEdge, ...]


def build_fsm_graph(fsm: FSMDefinition | dict[str, Any]) -> FSMGraph:
    """Return the graph (states and transitions) of an FSM definition.

    Args:
        fsm: an ``FSMDefinition``, or its dict form (the JSON contract). A dict
            is not validated as an ``FSMDefinition``; a transition without
            ``priority`` gets the ``Transition`` model's default.

    Returns:
        An ``FSMGraph`` with full, untruncated descriptions.

    Raises:
        ValueError: ``states``, a state or a transition is not a mapping,
            ``initial_state`` is not a key of ``states``, a transition names a
            ``target_state`` that is not one, or a field has the wrong type
            (pydantic's ``ValidationError`` is a ``ValueError``).
    """
    # DECISION plan-2026-09-30T062855-07ad3f8c/D-018
    # This is the ONE definition of an FSM's graph: the Mermaid and DOT
    # exporters read it and so does the monitor's visualize endpoint. Do NOT
    # build nodes and edges from a definition anywhere else, and do NOT derive
    # them from `build_graph_representation`: its `(target, description,
    # required_keys)` tuples carry no priority and its shape is pinned by the
    # ASCII renderers and their tests. Layout and label length are the
    # consumer's business, so nothing is truncated here.
    data = fsm.model_dump() if isinstance(fsm, FSMDefinition) else fsm
    states: dict[str, Any] = data.get("states", {})
    initial_state = data.get("initial_state", "")
    _require_definition_shapes(states)
    _require_initial_state(states, initial_state)
    _require_valid_transition_targets(states)

    default_priority = Transition.model_fields["priority"].default
    nodes = []
    edges = []
    for state_id, state in states.items():
        transitions = _transitions_of(state)
        nodes.append(
            FSMGraphNode(
                id=state_id,
                description=state.get("description") or "",
                purpose=state.get("purpose") or "",
                is_initial=state_id == initial_state,
                is_terminal=_is_terminal(state),
            )
        )
        for transition in transitions:
            priority = transition.get("priority")
            edges.append(
                FSMGraphEdge(
                    source=state_id,
                    target=transition["target_state"],
                    description=transition.get("description") or "",
                    priority=default_priority if priority is None else priority,
                )
            )
    return FSMGraph(
        name=_one_line(data.get("name"), _DEFAULT_NAME),
        initial_state=initial_state,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def _edge_label(edge: FSMGraphEdge) -> str:
    """``P<priority> <description>`` on one line (whitespace runs collapsed)."""
    return f"P{edge.priority} {_one_line(edge.description, '')}".rstrip()


def _unused_id(wanted: str, taken: set[str]) -> str:
    """``wanted``, with ``_`` appended until it is not in ``taken``; adds it."""
    while wanted in taken:
        wanted += "_"
    taken.add(wanted)
    return wanted


_MERMAID_PLAIN_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


# Words the state-diagram grammar reads as syntax when used as a bare state id.
_MERMAID_KEYWORDS = frozenset(
    {"as", "class", "classdef", "direction", "end", "note", "state", "style"}
)


# Mermaid entity codes (`#name;` / `#number;`). One translate pass, so the `#`
# and `;` of an inserted code are never escaped again.
_MERMAID_ENTITIES = str.maketrans(
    {
        "#": "#35;",
        ";": "#59;",
        ":": "#58;",
        '"': "#quot;",
        "<": "#lt;",
        ">": "#gt;",
    }
)


def _mermaid_text(value: str) -> str:
    """``value`` as one line of Mermaid label text, significant characters escaped."""
    return _one_line(value, "").translate(_MERMAID_ENTITIES)


def to_mermaid(graph: FSMGraph) -> str:
    """Render an ``FSMGraph`` as a Mermaid ``stateDiagram-v2`` document.

    ``[*]`` points to the initial state and every terminal state points to
    ``[*]``. Edges are labelled ``P<priority> <description>``. A state id that
    is not a plain identifier (or is a Mermaid keyword) is declared under a
    generated alias with the real id as its display label. Pure function; the
    result has no trailing newline.
    """
    taken = {node.id for node in graph.nodes}
    alias: dict[str, str] = {}
    lines = ["stateDiagram-v2"]
    for index, node in enumerate(graph.nodes):
        if (
            _MERMAID_PLAIN_ID.match(node.id)
            and node.id.lower() not in _MERMAID_KEYWORDS
        ):
            alias[node.id] = node.id
            continue
        alias[node.id] = _unused_id(f"state_{index}", taken)
        lines.append(f'    state "{_mermaid_text(node.id)}" as {alias[node.id]}')
    lines.append(f"    [*] --> {alias[graph.initial_state]}")
    lines.extend(
        f"    {alias[edge.source]} --> {alias[edge.target]} : "
        f"{_mermaid_text(_edge_label(edge))}"
        for edge in graph.edges
    )
    lines.extend(
        f"    {alias[node.id]} --> [*]" for node in graph.nodes if node.is_terminal
    )
    return "\n".join(lines)


def _dot_quote(value: str) -> str:
    """``value`` as a DOT double-quoted string (usable as an id or a label)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def to_dot(graph: FSMGraph) -> str:
    """Render an ``FSMGraph`` as a Graphviz DOT ``digraph``.

    Every id is a quoted string. A ``point`` node points to the initial state,
    terminal states are ``doublecircle``, edges are labelled
    ``P<priority> <description>``. Pure function; needs no Graphviz install and
    the result has no trailing newline.
    """
    entry = _dot_quote(_unused_id("__start__", {node.id for node in graph.nodes}))
    lines = [
        f"digraph {_dot_quote(graph.name)} {{",
        "    rankdir=LR;",
        f"    {entry} [shape=point];",
    ]
    lines.extend(
        f"    {_dot_quote(node.id)}"
        f"{' [shape=doublecircle]' if node.is_terminal else ''};"
        for node in graph.nodes
    )
    lines.append(f"    {entry} -> {_dot_quote(graph.initial_state)};")
    lines.extend(
        f"    {_dot_quote(edge.source)} -> {_dot_quote(edge.target)} "
        f"[label={_dot_quote(_edge_label(edge))}];"
        for edge in graph.edges
    )
    lines.append("}")
    return "\n".join(lines)


_GRAPH_EXPORTERS: dict[str, Callable[[FSMGraph], str]] = {
    "mermaid": to_mermaid,
    "dot": to_dot,
}


# Output formats of `visualize_fsm_from_file` and `fsm-llm-visualize --format`.
OUTPUT_FORMATS: tuple[str, ...] = ("ascii", *_GRAPH_EXPORTERS)
