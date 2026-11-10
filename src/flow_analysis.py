"""Static analysis over FSM graphs: shared components and data flow.

A pure core module (it imports no subpackage and nothing from the pipeline,
only `definitions`). The validator's trap detection and the data-flow and
liveness analyses of runtime FSM synthesis (plan 2cefeb61) share one graph
primitive here, so a graph rule lives once. The analyses take a typed
`FSMDefinition` and return plain findings; they never log and never raise on a
definition the loader accepted. `FSMValidator.validate()` calls none of them
(it must never be stricter than the loader).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .definitions import (
    FSMDefinition,
    State,
    Transition,
    logic_has_unresolvable_reads,
    logic_referenced_keys,
    reachable_from,
)


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


@dataclass(frozen=True)
class FlowFinding:
    """One defect an analysis found: plain data, no logging.

    `analysis` names the analysis ("data-flow"), `code` the defect, `state` the
    state id it sits at, `message` a sentence quoting state ids and key names
    only. All of it may be model-authored text: the caller defangs it before it
    reaches a prompt.
    """

    analysis: str
    code: str
    state: str
    message: str


def _minted_keys(state: State) -> set[str]:
    """The keys the pipeline mints a plain extraction for in `state`.

    A deliberate mirror of `MessagePipeline._build_field_configs_from_state`
    (kept by a parity test over every example state, plan 2cefeb61 D-017 item
    7; importing the pipeline here would couple a pure analysis to its largest
    module): nothing for a completion or synthesize state; else
    `required_context_keys` plus every `requires_context_keys` of the state's
    own edges, minus keys a `classification_extractions` entry owns, plus the
    explicit `field_extractions` names (which win over that filter).
    """
    if state.completion is not None or state.synthesize is not None:
        return set()
    keys = set(state.required_context_keys or [])
    for transition in state.transitions or []:
        for condition in transition.conditions or []:
            keys.update(condition.requires_context_keys or [])
    owned = {c.field_name for c in state.classification_extractions or []}
    return (keys - owned) | {f.field_name for f in state.field_extractions or []}


def _result_key(state: State) -> str | None:
    config = state.completion or state.synthesize
    return config.result_key if config is not None else None


def _handler_only(definition: FSMDefinition) -> frozenset[str]:
    """Keys no extraction may write: the pipeline's `_handler_only_keys`."""
    keys = {k for s in definition.states.values() if (k := _result_key(s))}
    return frozenset(definition.handler_only_keys) | keys


def _produced(state: State, handler_only: frozenset[str]) -> set[str]:
    """Keys `state` can write on arrival or while the conversation sits in it:
    its minted and classification keys minus the handler-only set, plus its own
    completion or synthesize result key (only that state writes it)."""
    keys = _minted_keys(state) | {
        c.field_name for c in state.classification_extractions or []
    }
    keys -= handler_only
    own = _result_key(state)
    return keys | {own} if own else keys


def _proven(transition: Transition) -> set[str]:
    """Keys a taken edge proves present: every condition must pass, and a
    `requires_context_keys` entry passes only when the key is not missing."""
    return {
        k for c in transition.conditions or [] for k in c.requires_context_keys or []
    }


def _forward_meet(
    definition: FSMDefinition,
    entry: frozenset[str],
    edge_out: Callable[[State, Transition, frozenset[str]], frozenset[str]],
) -> dict[str, frozenset[str]]:
    """Forward must-analysis: for each reachable state, the keys present on
    arrival along EVERY path (intersection meet, greatest fixpoint, cycles
    allowed). `entry` holds at the initial state; `edge_out(state, edge, in_)`
    is what crossing `edge` out of `state` leaves. Unreachable states are
    ignored (they would only poison the intersection). The transfer functions
    are monotone gen-only, so the worklist result equals the meet over all
    paths. Targets that are not states are skipped."""
    states = definition.states
    initial = definition.initial_state
    reach = (
        reachable_from(
            {
                sid: [t.target_state for t in s.transitions or []]
                for sid, s in states.items()
            },
            [initial],
        )
        & states.keys()
    )
    # None is the lattice top ("no path seen yet"); only the initial state starts finite.
    arrival: dict[str, frozenset[str] | None] = {
        sid: None for sid in states if sid in reach
    }
    arrival[initial] = entry
    work = [initial]
    while work:
        sid = work.pop()
        here = arrival[sid]
        if here is None:
            continue
        for transition in states[sid].transitions or []:
            target = transition.target_state
            if target not in arrival:
                continue
            out = edge_out(states[sid], transition, here)
            known = arrival[target]
            new = out if known is None else known & out
            if new != known:
                arrival[target] = new
                work.append(target)
    return {sid: keys for sid, keys in arrival.items() if keys is not None}


def declared_available(
    definition: FSMDefinition, inputs: Iterable[str] = ()
) -> dict[str, frozenset[str]]:
    """For each reachable state, the keys declared on EVERY path by the time
    its outgoing gates read the context (its own production included).

    Contract:
        definition: a loaded `FSMDefinition`. inputs: key names the driver
            provides at the initial state.
    Returns: state id -> frozenset of key names, in definition order; states
        unreachable from the initial state are absent. Never raises.
    """
    handler_only = _handler_only(definition)
    arrival = _forward_meet(
        definition,
        frozenset(inputs),
        lambda state, _t, here: here | _produced(state, handler_only),
    )
    return {
        sid: keys | _produced(definition.states[sid], handler_only)
        for sid, keys in arrival.items()
    }


def analyze_data_flow(
    definition: FSMDefinition,
    *,
    inputs: Iterable[str] = (),
    return_keys: Iterable[str] = (),
    require_guaranteed_return_keys: bool = True,
) -> list[FlowFinding]:
    """G3: the data-flow defects of `definition`, in definition order.

    Codes (analysis "data-flow"): `undeclared-read` (a gate reads a key that is
    not declared on every path to it, or is handler-only and unproduced; one
    finding per state and key), `unresolvable-read` (a gate reads a key by a
    computed, empty or non-string name), `bulk-extraction` (extraction
    instructions with no declared extraction: the model could write any key),
    `return-key-undeclared` (a return key no reachable state produces and no
    input provides, per terminal), `return-key-not-guaranteed` (only when
    `require_guaranteed_return_keys`: some path reaches a terminal without an
    edge `requires_context_keys` proving the key present; `inputs` count as
    proven at the initial state). Value tests in `logic` are treated as
    satisfiable, so the analysis is coarse and may reject what a human would
    accept (the repair loop absorbs that). Keys the runtime seeds or a handler
    writes must be listed in `inputs`.

    Contract: `definition` a loaded `FSMDefinition`; findings are plain data
    (messages quote state ids and key names only). Never raises.
    """
    states = definition.states
    given = frozenset(inputs)
    handler_only = _handler_only(definition)
    available = declared_available(definition, given)
    findings: list[FlowFinding] = []

    def add(code: str, sid: str, message: str) -> None:
        findings.append(FlowFinding("data-flow", code, sid, message))

    for sid, keys in available.items():
        state = states[sid]
        classified = bool(state.classification_extractions)
        if (
            state.completion is None
            and state.extraction_instructions
            and not classified
            and not _minted_keys(state) - handler_only
        ):
            add(
                "bulk-extraction",
                sid,
                f"state '{sid}' has extraction_instructions but no declared extraction",
            )
        reads: set[str] = set()
        unresolvable = False
        for transition in state.transitions or []:
            for condition in transition.conditions or []:
                reads.update(condition.requires_context_keys or [])
                try:
                    reads |= logic_referenced_keys(condition.logic)
                    unresolvable |= logic_has_unresolvable_reads(condition.logic)
                except ValueError:
                    unresolvable = True
        if unresolvable:
            add(
                "unresolvable-read",
                sid,
                f"a gate of state '{sid}' reads a key by a computed or empty name",
            )
        for key in sorted(reads - keys):
            add(
                "undeclared-read",
                sid,
                f"a gate of state '{sid}' reads '{key}', which is not declared on every path to it",
            )

    wanted = tuple(dict.fromkeys(return_keys))
    terminals = [sid for sid in available if not states[sid].transitions]
    if wanted and terminals:
        ever = given.union(*(_produced(states[sid], handler_only) for sid in available))
        proven = _forward_meet(definition, given, lambda _s, t, here: here | _proven(t))
        for sid in terminals:
            for key in wanted:
                if key not in ever:
                    add(
                        "return-key-undeclared",
                        sid,
                        f"return key '{key}' is produced by no state; terminal '{sid}' cannot return it",
                    )
                elif require_guaranteed_return_keys and key not in proven[sid]:
                    add(
                        "return-key-not-guaranteed",
                        sid,
                        f"return key '{key}' is not proven present on every path to terminal '{sid}'",
                    )
    return findings
