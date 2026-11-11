"""
G6a, the dead-edge dry run: edges the real evaluator can never take.

G3 and G4 count every value test as satisfiable, so a compile can pass them
and still hold an edge no context ever opens (a bool field gated on ``== "yes"``).
This gate searches a small, type-aware pool of values per key through
``TransitionEvaluator`` itself and reports what no assignment fires. It is
model-free and costs milliseconds. Nothing here raises on a loaded definition,
logs or does I/O.

# DECISION plan-2026-11-10T194139-2cefeb61/D-042
# A finding must never be a false reject, so the gate is coarse on purpose. An
# edge is judged only inside a small fragment (plain key names, ``var`` against
# a scalar, ordering only against a number, ``in`` only over a list literal);
# anything else is unknown, silent, and counts as firing. A pool is narrowed to
# a type only for a key no other state mentions. Priority, ties and target
# collapse come from ``evaluate_transitions``. Do NOT widen the fragment to
# arithmetic or string ordering, do NOT narrow ``str``, do NOT copy the
# evaluator's priority rule, and do NOT add a finding the evaluator cannot
# reproduce.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from typing import Any

from fsm_llm.definitions import (
    FSMContext,
    FSMDefinition,
    State,
    Transition,
    TransitionEvaluationError,
    TransitionEvaluationResult,
    handler_only_names,
    reachable_from,
)
from fsm_llm.flow_analysis import FlowFinding
from fsm_llm.transition_evaluator import TransitionEvaluator

from .constants import MAX_ASSIGNMENTS

_ABSENT: Any = object()  # a key left out of the context
_ORDER = frozenset({">", ">=", "<", "<="})
_EQUAL = frozenset({"==", "===", "!=", "!=="})
_LOGICAL = frozenset({"!", "!!", "and", "or", "if"})
_KINDS = {"bool": "bool", "int": "number", "float": "number"}

# DECISION plan-2026-11-10T194139-2cefeb61/D-059
# The pool of an untyped key holds every kind of value a model can extract:
# the filler, each literal (also upper-cased), numbers and their string forms,
# ``""``, ``"true"``, ``"false"``. A key no logic tests gets absent or the filler
# only (presence is all that matters), which keeps three-key states under the
# cap. Do NOT trim these kinds to shrink the search (a missing kind is a false
# reject), and do NOT test the gate against a pool it shares with the oracle.
#: What an untyped key gets first: a value that equals no literal of any gate.
FILLER = "sample"
#: Strings a model can extract that no other pool value spells (D-059): the
#: empty string is "missing" but not null, and the evaluator reads "true" and
#: "false" apart from the booleans.
POOL_STRINGS = ("", "true", "false")


def _plain(name: Any) -> bool:
    return isinstance(name, str) and bool(name) and "." not in name


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _scalar(value: Any, lits: list[Any]) -> bool:
    ok = value is None or isinstance(value, str | int | float)
    if ok:
        lits.append(value)
    return ok


def _operand(node: Any, lits: list[Any], reads: set[str]) -> bool:
    if isinstance(node, dict):
        return list(node) == ["var"] and _judge(node, lits, reads)
    return _scalar(node, lits)


def _judge(node: Any, lits: list[Any], reads: set[str]) -> bool:
    """Whether ``node`` lies in the fragment; fills ``lits`` (scalar literals)
    and ``reads`` (plain key names). Anything else is False: unknown."""
    if not isinstance(node, dict):
        return _scalar(node, lits)
    if len(node) != 1:
        return False
    op, args = next(iter(node.items()))
    args = args if isinstance(args, list) else [args]
    if op == "var":
        ok = 1 <= len(args) <= 2 and _plain(args[0])
        ok = ok and (len(args) == 1 or _scalar(args[1], lits))
        if ok:
            reads.add(args[0])
        return ok
    if op == "missing":
        names = [n for a in args for n in (a if isinstance(a, list) else [a])]
        ok = all(_plain(n) for n in names)
        if ok:
            reads.update(names)
        return ok
    if op in _LOGICAL:
        return all(_judge(a, lits, reads) for a in args)
    if len(args) != 2:
        return False
    if op in ("in", "contains"):
        needle, haystack = args if op == "in" else args[::-1]
        return (
            isinstance(haystack, list)
            and all(_scalar(h, lits) for h in haystack)
            and _operand(needle, lits, reads)
        )
    if op not in _EQUAL | _ORDER or not all(_operand(a, lits, reads) for a in args):
        return False
    if op in _ORDER:  # complete only against one number literal
        literal = [a for a in args if not isinstance(a, dict)]
        return len(literal) == 1 and _number(literal[0])
    return True


def _scan(edge: Transition) -> tuple[list[Any], set[str], set[str]] | None:
    """(literals, key names read, key names the logic tests) of ``edge``; None
    when a gate is unknown. A key only ``requires_context_keys`` names is read
    for its presence alone."""
    lits: list[Any] = []
    tested: set[str] = set()
    required: set[str] = set()
    for condition in edge.conditions or []:
        names = condition.requires_context_keys or []
        if not all(_plain(n) for n in names) or not _judge(
            condition.logic, lits, tested
        ):
            return None
        required.update(names)
    return lits, tested | required, tested


def _numbers(lits: list[Any]) -> list[float]:
    """The numeric partition: each numeric literal (a numeric string as its
    number), 0 and 1, the midpoint of every adjacent pair and one step beyond
    both ends. It decides every ``var <op> number`` the fragment allows."""
    found = {0.0, 1.0}
    for value in lits:
        if _number(value) or isinstance(value, str):
            try:
                number = float(value)
            except (ValueError, OverflowError):
                continue
            if math.isfinite(number):
                found.add(number)
    points = sorted(found)
    mids = [a / 2 + b / 2 for a, b in itertools.pairwise(points)]
    return sorted([*points, *mids, points[0] - 1, points[-1] + 1])


def _strings(lits: list[Any]) -> list[str]:
    """``"0"``, ``"1"`` and each numeric literal as a model writes it (``3``
    for 3.0, else the repr); a non-finite literal has none."""
    found = ["0", "1"]
    for value in lits:
        if _number(value) and math.isfinite(value):
            found.append(str(int(value)) if value == int(value) else repr(float(value)))
    return found


def _pool(narrow: tuple[str, tuple[str, ...]] | None, lits: list[Any]) -> list[Any]:
    """The values tried for one key: absent first, then by kind. A key that is
    not narrowed gets the filler, every literal (a string also upper-cased:
    ``==`` ignores case), the numeric partition, 0, 1 and each numeric literal
    also as a string (a numeric string orders and equals like its number), then
    ``True``, ``False``, 0, 1 and ``POOL_STRINGS``. The order picks the witness:
    a natural value before a coerced one."""
    kind, intents = narrow or ("any", ())
    exact = [v for v in lits if v is not None]
    points = _numbers(lits)
    upper = [v.upper() for v in exact if isinstance(v, str)]
    values: dict[str, list[Any]] = {
        "bool": [True, False],
        "intent": list(intents),
        "number": [v for v in exact if _number(v)] + points,
        "any": [
            *[FILLER, *exact, *upper, *points, *_strings(exact)],
            *[True, False, 0, 1, *POOL_STRINGS],
        ],
    }
    seen: set[Any] = set()
    pool = []
    for value in [_ABSENT, *values[kind]]:
        if (type(value), value) not in seen:
            seen.add((type(value), value))
            pool.append(value)
    return pool


def _narrowing(
    state: State, sid: str, mentions: dict[str, set[str]], outside: frozenset[str]
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Keys whose value the state's own extraction types: bool, number or
    intent name. Only a key no other state mentions, no input and no handler
    can supply (another producer could write another type)."""
    found = {
        c.field_name: ("intent", tuple(i.name for i in c.intents))
        for c in state.classification_extractions or []
    }
    for f in state.field_extractions or []:
        found[f.field_name] = (_KINDS.get(f.field_type, "any"), ())
    return {
        k: v
        for k, v in found.items()
        if v[0] != "any" and mentions.get(k, set()) <= {sid} and k not in outside
    }


def _mentions(definition: FSMDefinition) -> dict[str, set[str]]:
    """key -> the states that declare, require or own it (a gate that only
    reads it in logic is not a producer)."""
    mentions: dict[str, set[str]] = {}
    for sid, state in definition.states.items():
        config = state.completion or state.synthesize
        names = [
            *(state.required_context_keys or []),
            *(f.field_name for f in state.field_extractions or []),
            *(c.field_name for c in state.classification_extractions or []),
            *(
                key
                for t in state.transitions or []
                for c in t.conditions or []
                for key in c.requires_context_keys or []
            ),
            *([config.result_key] if config else []),
        ]
        for name in names:
            mentions.setdefault(name, set()).add(sid)
    return mentions


def _first_fires(
    state: State,
    edges: list[Transition],
    wanted: list[int],
    keys: list[str],
    pools: list[list[Any]],
    evaluator: TransitionEvaluator,
    limit: int | None = None,
) -> dict[int, dict[str, Any]] | None:
    """Edge index -> the context (present keys only) of the FIRST assignment of
    ``pools`` (at most ``limit`` of them, in product order) under which the
    evaluator, run on ``state``, takes that edge's target and the edge passes,
    for the ``wanted`` edges that have one; ``None`` on an evaluator error."""
    found: dict[int, dict[str, Any]] = {}
    for values in itertools.islice(itertools.product(*pools), limit):
        data = {k: v for k, v in zip(keys, values, strict=True) if v is not _ABSENT}
        try:
            result = evaluator.evaluate_transitions(state, FSMContext(data=data))
        except TransitionEvaluationError:
            return None
        if result.result_type is TransitionEvaluationResult.DETERMINISTIC:
            wins = {result.deterministic_transition}
        else:
            wins = {o.target_state for o in result.available_options or ()}
        for i in wanted:
            if (
                i not in found
                and edges[i].target_state in wins
                and evaluator.transition_passes(edges[i], data)
            ):
                found[i] = data
        if len(found) == len(wanted):
            break
    return found


def _search(
    state: State,
    narrow: dict[str, tuple[str, tuple[str, ...]]],
    evaluator: TransitionEvaluator,
) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    """``(dead, witness)`` for the edges of ``state``. ``dead``: edge index ->
    the keys it reads with their kinds, for each edge no assignment lets the
    evaluator take. ``witness``: edge index -> the context (present keys only)
    of the FIRST assignment under which the evaluator takes that edge's target.
    An unknown edge is in neither. Only judged edges go to the search (an
    unknown competitor left out can only make an edge easier to fire); a
    completion or synthesize state yields nothing. A state whose pool product is
    over the cap has no ``dead`` (nothing is proven) and a ``witness`` only for
    the edges a bounded search over their own keys opens (D-071)."""
    edges = state.transitions or []
    scans = {i: _scan(t) for i, t in enumerate(edges)}
    judged = {i: s for i, s in scans.items() if s is not None}
    if not judged or state.completion is not None or state.synthesize is not None:
        return {}, {}
    lits = [v for found, _, _ in judged.values() for v in found]
    keys = sorted({k for _, reads, _ in judged.values() for k in reads})
    tested = {k for _, _, names in judged.values() for k in names}
    # a key no logic tests matters by presence alone: absent or any value
    pools = [
        _pool(narrow.get(k), lits) if k in tested else [_ABSENT, FILLER] for k in keys
    ]
    if math.prod(len(p) for p in pools) > MAX_ASSIGNMENTS:
        # DECISION plan-2026-11-10T194139-2cefeb61/D-071
        # Over the cap nothing is PROVEN dead (the verdict stays silent), but each
        # judged edge still gets a witness: searched over only the keys it reads,
        # the first MAX_ASSIGNMENTS assignments of its own product, and judged by
        # the whole state, unknown edges included. Do NOT return no witnesses here
        # (G6b then says the fixed samples and an honest three-key state fails),
        # do NOT report a dead edge from a prefix, and do NOT judge against the
        # probe that drops unknown competitors.
        found: dict[int, dict[str, Any]] = {}
        for i, (own, reads, names) in judged.items():
            own_keys = sorted(reads)
            own_pools = [
                _pool(narrow.get(k), own) if k in names else [_ABSENT, FILLER]
                for k in own_keys
            ]
            found |= (
                _first_fires(
                    state, edges, [i], own_keys, own_pools, evaluator, MAX_ASSIGNMENTS
                )
                or {}
            )
        return {}, found
    probe = state.model_copy(update={"transitions": [edges[i] for i in judged]})
    witness = _first_fires(probe, edges, list(judged), keys, pools, evaluator)
    if witness is None:
        return {}, {}
    kinds = {k: narrow.get(k, ("any",))[0] for k in keys}
    return {
        i: ", ".join(f"{k} ({kinds[k]})" for k in sorted(judged[i][1])) or "any key"
        for i in judged
        if i not in witness
    }, witness


def _states(
    definition: FSMDefinition, inputs: Iterable[str]
) -> dict[str, tuple[dict[int, str], dict[int, dict[str, Any]]]]:
    """The search of every state reachable over the edges, in definition order:
    state id -> ``_search`` result. The one pass ``analyze_dryrun`` and
    ``edge_witnesses`` both read."""
    states = definition.states
    graph = {
        sid: [t.target_state for t in s.transitions or []] for sid, s in states.items()
    }
    reach = reachable_from(graph, [definition.initial_state]) & states.keys()
    outside = frozenset(inputs) | handler_only_names(definition)
    mentions = _mentions(definition)
    evaluator = TransitionEvaluator()
    return {
        sid: _search(
            states[sid], _narrowing(states[sid], sid, mentions, outside), evaluator
        )
        for sid in states
        if sid in reach
    }


def edge_witnesses(
    definition: FSMDefinition, *, inputs: Iterable[str] = ()
) -> dict[str, dict[int, dict[str, Any] | None]]:
    """For each state reachable over the edges: edge index -> the context under
    which the evaluator takes that edge's target.

    Interface contract (caller: ``simulation.happy_path``; the same search as
    ``analyze_dryrun``, run once more): ``definition`` a loaded
    ``FSMDefinition``, ``inputs`` as there. A dead edge is absent; an edge
    outside the fragment (unknown, it counts as firing) maps to ``None``; every
    other edge maps to the present keys of the first assignment that opens it
    (pool values, ``FILLER`` among them, are plain Python scalars). Never
    raises.
    """
    return {
        sid: {
            i: found.get(i)
            for i in range(len(definition.states[sid].transitions or []))
            if i not in dead
        }
        for sid, (dead, found) in _states(definition, inputs).items()
    }


def analyze_dryrun(
    definition: FSMDefinition, *, inputs: Iterable[str] = ()
) -> list[FlowFinding]:
    """G6a: the dead edges of ``definition``, in definition order.

    Interface contract (callers: ``run_gates`` on the raw compile, so
    ``verify_stored`` re-runs it): ``definition`` a loaded ``FSMDefinition``,
    ``inputs`` the key names the driver provides (never narrowed). Returns
    plain ``FlowFinding`` rows (analysis ``"dry-run"``, messages quote state ids
    and key names only): ``edge-unfireable`` (judged edge, no assignment of its
    keys makes the evaluator choose its target) and ``terminal-unreachable`` (a
    terminal reachable over the edges but not over the edges that can fire).
    Logic outside the fragment is unknown and silent. Never raises.
    """
    states = definition.states
    initial = definition.initial_state
    findings: list[FlowFinding] = []
    live: dict[str, list[str]] = {}
    for sid, (dead, _) in _states(definition, inputs).items():
        edges = states[sid].transitions or []
        live[sid] = [t.target_state for i, t in enumerate(edges) if i not in dead]
        for i, keys in dead.items():
            findings.append(
                FlowFinding(
                    "dry-run",
                    "edge-unfireable",
                    sid,
                    f"edge '{sid}' -> '{edges[i].target_state}' can never fire: "
                    f"no value of {keys} lets the evaluator take it",
                )
            )
    firing = reachable_from(live, [initial])
    findings += [
        FlowFinding(
            "dry-run",
            "terminal-unreachable",
            sid,
            f"terminal '{sid}' cannot be reached over edges that can fire",
        )
        for sid in live
        if not states[sid].transitions and sid not in firing
    ]
    return findings
