"""
Tests for ``fsm_llm.synthesis.dryrun`` (G6a, the dead-edge gate).

A finding must be correct under the REAL evaluator: the fixture table pins
what is reported and, as hard, what stays silent; a seeded differential checks
the search against a brute-force oracle over a wider value grid that calls
``evaluate_logic`` directly (it shares the evaluator, not the search).
"""

import glob
import itertools
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.definitions import FSMContext, FSMDefinition
from fsm_llm.expressions import evaluate_logic, is_missing
from fsm_llm.flow_analysis import FlowFinding, analyze_liveness
from fsm_llm.synthesis import dryrun
from fsm_llm.synthesis.dryrun import analyze_dryrun
from fsm_llm.transition_evaluator import TransitionEvaluator
from fsm_llm.validator import FSMValidator

from .conftest import good_fsm

_REPO = Path(__file__).resolve().parents[2]


def _var(name: str) -> dict[str, Any]:
    return {"var": name}


def _edge(target: str, logic: Any = None, *, requires=(), priority: int = 1):
    condition: dict[str, Any] = {"description": "gate"}
    if logic is not None:
        condition["logic"] = logic
    if requires:
        condition["requires_context_keys"] = list(requires)
    edge: dict[str, Any] = {
        "target_state": target,
        "description": f"to {target}",
        "priority": priority,
    }
    if logic is not None or requires:
        edge["conditions"] = [condition]
    return edge


def _state(sid: str, edges=(), **extra: Any) -> tuple[str, dict[str, Any]]:
    state = {"id": sid, "description": sid, "purpose": sid}
    state.update(extra)
    if edges:
        state["transitions"] = list(edges)
    return sid, state


def _field(name: str, kind: str = "str") -> dict[str, Any]:
    return {"field_name": name, "field_type": kind, "extraction_instructions": "get it"}


def _classify(name: str, *intents: str) -> dict[str, Any]:
    return {
        "field_name": name,
        "intents": [{"name": i, "description": i} for i in intents],
        "fallback_intent": intents[-1],
    }


def _fsm(*states: tuple[str, dict[str, Any]]) -> FSMDefinition:
    return FSMDefinition(
        name="T",
        description="t",
        initial_state=states[0][0],
        states=dict(states),
    )


def _found(definition: FSMDefinition, **kw: Any) -> set[tuple[str, str]]:
    return {(f.code, f.state) for f in analyze_dryrun(definition, **kw)}


def _two(logic: Any, *, fields=(), classify=(), requires=(), **extra: Any):
    """ask -> done through one edge gated by ``logic``."""
    if fields:
        extra["field_extractions"] = list(fields)
    if classify:
        extra["classification_extractions"] = list(classify)
    return _fsm(
        _state("ask", [_edge("done", logic, requires=requires)], **extra),
        _state("done"),
    )


DEAD = {("edge-unfireable", "ask"), ("terminal-unreachable", "done")}
BOOL = [_field("ok", "bool")]
NUM = [_field("n", "float")]
X = _var("x")


class TestReported:
    @pytest.mark.parametrize(
        ("name", "logic", "fields", "classify"),
        [
            ("bool-gated-on-a-string", {"==": [_var("ok"), "yes"]}, BOOL, ()),
            ("bool-is-not-a-number-string", {"===": [_var("ok"), "true"]}, BOOL, ()),
            (
                "contradiction",
                {"and": [{">": [X, 5]}, {"<": [X, 3]}]},
                (),
                (),
            ),
            (
                "or-of-two-contradictions",
                {"and": [{"==": [X, "a"]}, {"==": [X, "b"]}]},
                (),
                (),
            ),
            ("number-never-equals-text", {"==": [_var("n"), "abc"]}, NUM, ()),
            (
                "unknown-intent",
                {"==": [_var("intent"), "purchase"]},
                (),
                [_classify("intent", "buy", "browse")],
            ),
            ("bool-in-a-text-list", {"in": [_var("ok"), ["a", "b"]]}, BOOL, ()),
            (
                "not-a-bool-and-true",
                {"and": [{"!": _var("ok")}, {"==": [_var("ok"), True]}]},
                BOOL,
                (),
            ),
            ("a-key-both-present-and-missing", {"missing": ["x"]}, (), ()),
        ],
    )
    def test_a_dead_edge_and_its_cut_off_terminal(self, name, logic, fields, classify):
        requires = ("x",) if name == "a-key-both-present-and-missing" else ()
        definition = _two(logic, fields=fields, classify=classify, requires=requires)
        assert _found(definition) == DEAD, name

    def test_a_chain_with_no_firing_edge_to_the_terminal(self):
        definition = _fsm(
            _state("a", [_edge("b", requires=())]),
            _state("b", [_edge("c", {"and": [{">": [X, 5]}, {"<": [X, 3]}]})]),
            _state("c"),
        )
        assert _found(definition) == {
            ("edge-unfireable", "b"),
            ("terminal-unreachable", "c"),
        }

    def test_an_edge_always_beaten_by_a_lower_priority_number(self):
        definition = _fsm(
            _state(
                "a",
                [
                    _edge("b", {">": [X, 0]}, priority=1),
                    _edge("c", {">": [X, 5]}, priority=2),
                ],
            ),
            _state("b"),
            _state("c"),
        )
        found = analyze_dryrun(definition)
        assert [(f.code, f.state) for f in found] == [
            ("edge-unfireable", "a"),
            ("terminal-unreachable", "c"),
        ]
        assert "'a' -> 'c'" in found[0].message

    def test_findings_are_plain_flow_findings_that_name_states_and_keys(self):
        definition = _two({"==": [_var("ok"), "yes"]}, fields=BOOL)
        found = analyze_dryrun(definition)
        assert all(isinstance(f, FlowFinding) for f in found)
        assert {f.analysis for f in found} == {"dry-run"}
        assert "'ask' -> 'done'" in found[0].message and "ok" in found[0].message

    def test_the_definition_is_never_mutated(self):
        definition = _two({"==": [_var("ok"), "yes"]}, fields=BOOL)
        before = definition.model_dump()
        analyze_dryrun(definition)
        assert definition.model_dump() == before


class TestSilent:
    @pytest.mark.parametrize(
        ("name", "logic", "fields", "classify"),
        [
            ("bool-gated-on-a-bool", {"==": [_var("ok"), True]}, BOOL, ()),
            ("bool-loose-one", {"==": [_var("ok"), 1]}, BOOL, ()),
            ("interval", {"and": [{">": [X, 5]}, {"<": [X, 6]}]}, (), ()),
            ("number-above", {">": [_var("n"), 3]}, NUM, ()),
            ("number-vs-numeric-text", {"==": [_var("n"), "5"]}, NUM, ()),
            ("number-vs-true", {"==": [_var("n"), True]}, NUM, ()),
            (
                "known-intent",
                {"==": [_var("intent"), "buy"]},
                (),
                [_classify("intent", "buy", "browse")],
            ),
            ("text-in-a-list", {"in": [X, ["a", "b"]]}, (), ()),
            ("contains", {"contains": [["a", "b"], X]}, (), ()),
            ("not-null", {"!=": [X, None]}, (), ()),
            ("absent-key", {"missing": ["x"]}, (), ()),
            ("truthy", {"!!": X}, (), ()),
            ("if", {"if": [{"==": [X, 1]}, True, False]}, (), ()),
            ("var-default", {"==": [{"var": ["x", 5]}, 5]}, (), ()),
        ],
    )
    def test_a_fireable_edge_is_not_reported(self, name, logic, fields, classify):
        definition = _two(logic, fields=fields, classify=classify)
        assert _found(definition) == set(), name

    @pytest.mark.parametrize(
        "logic",
        [
            {"==": [{"+": [_var("ok"), 1]}, 7]},  # arithmetic
            {"==": [{"cat": [_var("ok"), "a"]}, "never"]},
            {">": [_var("ok"), "b"]},  # string ordering
            {"<": [_var("ok"), _var("ok")]},  # two vars
            {"==": [_var("ok.x"), 1]},  # dotted read
            {"==": [{"var": {"cat": ["o", "k"]}}, "yes"]},  # computed name
            {"in": [_var("ok"), "abc"]},  # a text haystack
            {"in": ["a", _var("ok")]},  # a var haystack
            {"missing_some": [1, ["ok"]]},
            {"has_context": "ok"},
            {"==": [{"if": [True, _var("ok"), 3]}, "yes"]},
        ],
    )
    def test_logic_outside_the_fragment_is_unknown_not_rejected(self, logic):
        definition = _two(logic, fields=BOOL)
        assert _found(definition) == set()

    def test_a_pool_product_over_the_cap_is_unknown(self):
        keys = [f"k{i}" for i in range(4)]
        dead = {
            "and": [{">": [X, 5]}, {"<": [X, 3]}, *[{"==": [_var(k), 1]} for k in keys]]
        }
        small = {"and": [{">": [X, 5]}, {"<": [X, 3]}, {"==": [_var("k0"), 1]}]}
        assert _found(_two(dead)) == set()
        assert _found(_two(small)) == DEAD

    def test_the_cap_is_2048_assignments(self):
        assert dryrun.MAX_ASSIGNMENTS == 2048

    def test_a_completion_state_is_never_judged(self):
        tool = {
            "type": "function",
            "function": {
                "name": "t",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        dead = {"and": [{">": [X, 5]}, {"<": [X, 3]}]}
        definition = _two(dead, completion={"tools": [tool]})
        assert _found(definition) == set()

    def test_an_edge_with_no_logic_is_never_reported(self):
        definition = _two(None, requires=("x",))
        assert _found(definition) == set()

    def test_equal_priorities_to_two_targets_are_both_fireable(self):
        definition = _fsm(
            _state(
                "a",
                [
                    _edge("b", {">": [X, 0]}, priority=1),
                    _edge("c", {">": [X, 5]}, priority=1),
                ],
            ),
            _state("b"),
            _state("c"),
        )
        assert _found(definition) == set()

    def test_the_clean_compile_has_no_finding(self):
        assert _found(FSMDefinition(**good_fsm())) == set()


class TestNarrowing:
    def _typed(self, *, inputs=(), elsewhere=False, kind="bool"):
        mid = (
            _state("mid", [_edge("done")], required_context_keys=["ok"])
            if elsewhere
            else None
        )
        first = _state(
            "ask",
            [_edge("mid" if mid else "done", {"==": [_var("ok"), "yes"]})],
            field_extractions=[_field("ok", kind)],
        )
        states = [first, *([mid] if mid else []), _state("done")]
        return _found(_fsm(*states), inputs=inputs)

    def test_a_bool_key_only_this_state_declares_is_narrowed(self):
        assert self._typed() == DEAD

    def test_an_input_key_is_not_narrowed(self):
        assert self._typed(inputs=("ok",)) == set()

    def test_a_key_another_state_mentions_is_not_narrowed(self):
        assert self._typed(elsewhere=True) == set()

    def test_a_str_key_is_not_narrowed(self):
        assert self._typed(kind="str") == set()

    def test_a_handler_only_key_is_not_narrowed(self):
        definition = FSMDefinition(
            name="T",
            description="t",
            initial_state="ask",
            handler_only_keys=["ok"],
            states=dict(
                [
                    _state(
                        "ask",
                        [_edge("done", {"==": [_var("ok"), "yes"]}, requires=("ok",))],
                        field_extractions=[_field("ok", "bool")],
                    ),
                    _state("done"),
                ]
            ),
        )
        assert _found(definition) == set()


class TestEvaluatorIsTheOracle:
    def test_the_search_goes_through_the_real_evaluator(self, monkeypatch):
        calls = Counter()
        for name in ("evaluate_transitions", "transition_passes"):
            real = getattr(TransitionEvaluator, name)

            def spy(self, *a, _real=real, _name=name, **kw):
                calls[_name] += 1
                return _real(self, *a, **kw)

            monkeypatch.setattr(TransitionEvaluator, name, spy)
        definition = _two({"==": [_var("ok"), True]}, fields=BOOL)
        analyze_dryrun(definition)
        assert calls["evaluate_transitions"] > 0 and calls["transition_passes"] > 0

    def test_swapping_the_priorities_changes_the_answer(self):
        """Priority is the evaluator's rule: swap the two numbers and the dead
        edge becomes fireable (G6a copies no rule)."""
        definition = _fsm(
            _state(
                "a",
                [
                    _edge("b", {">": [X, 0]}, priority=1),
                    _edge("c", {">": [X, 5]}, priority=2),
                ],
            ),
            _state("b"),
            _state("c"),
        )
        assert ("edge-unfireable", "a") in _found(definition)
        flipped = definition.model_copy(deep=True)
        for edge in flipped.states["a"].transitions:
            edge.priority = 3 - edge.priority
        assert _found(flipped) == set()


# ---------------------------------------------------------------- differential

_KEYS = ("a", "b")
_LITS = (1, 2, 3, "a", "b", "buy", True)
_GRID = [-1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
_ABSENT = object()
_WIDE = [_ABSENT, True, False, "x", "a", "b", "buy", "browse", "1", "2", "3", *_GRID]
_WIDE += ["", "0", "true", "false", "A", "BUY"]  # the oracle's own list, not the gate's


def _leaf(rng: random.Random) -> Any:
    v = _var(rng.choice(_KEYS))
    kind = rng.choice(["eq", "ne", "hard", "ord", "in", "not", "bang", "missing"])
    if kind in ("eq", "ne", "hard"):
        op = {"eq": "==", "ne": "!=", "hard": "==="}[kind]
        return {op: [v, rng.choice(_LITS)]}
    if kind == "ord":
        op, n = rng.choice([">", ">=", "<", "<="]), rng.choice((1, 2, 3))
        return {op: [v, n]} if rng.random() < 0.7 else {op: [n, v]}
    if kind == "in":
        return {"in": [v, rng.sample(_LITS, 2)]}
    if kind == "missing":
        return {"missing": [v["var"]]}
    return {"!" if kind == "not" else "!!": v}


def _logic(rng: random.Random, depth: int = 2) -> Any:
    if depth == 0 or rng.random() < 0.5:
        return _leaf(rng)
    if rng.random() < 0.2:
        return {"if": [_leaf(rng), _leaf(rng), _leaf(rng)]}
    return {rng.choice(["and", "or"]): [_logic(rng, depth - 1), _logic(rng, depth - 1)]}


def _random_case(rng: random.Random):
    """(FSM definition, per state edges as (target, priority, logic, requires),
    the narrowed domain of each (state, key))."""
    n = rng.randint(2, 4)
    ids = [f"s{i}" for i in range(n)]
    typed = {k: rng.choice([None, "bool", "float", "class"]) for k in _KEYS}
    home = {k: rng.choice(ids[:-1]) for k in _KEYS}
    states, plan = [], {}
    for i, sid in enumerate(ids[:-1]):
        targets = [ids[i + 1], *[t for t in ids if rng.random() < 0.3]]
        edges = []
        for target in targets:
            logic = _logic(rng) if rng.random() < 0.85 else None
            keys = [k for k in _KEYS if typed[k] is None or home[k] == sid]
            requires = [rng.choice(keys)] if keys and rng.random() < 0.25 else []
            edges.append((target, rng.randint(1, 3), logic, requires))
        plan[sid] = edges
        extra: dict[str, Any] = {}
        mine = [k for k in _KEYS if typed[k] and home[k] == sid]
        fields = [_field(k, typed[k]) for k in mine if typed[k] != "class"]
        if fields:
            extra["field_extractions"] = fields
        classes = [_classify(k, "buy", "browse") for k in mine if typed[k] == "class"]
        if classes:
            extra["classification_extractions"] = classes
        states.append(
            _state(
                sid,
                [_edge(t, lg, requires=rq, priority=p) for t, p, lg, rq in edges],
                **extra,
            )
        )
    states.append(_state(ids[-1]))
    domains = {}
    for sid in ids[:-1]:
        for k in _KEYS:
            if typed[k] and home[k] == sid:
                domains[sid, k] = typed[k]
    return _fsm(*states), plan, domains


def _wide_pool(kind: str | None) -> list[Any]:
    if kind == "bool":
        return [_ABSENT, True, False]
    if kind == "float":
        return [_ABSENT, *_GRID, 1, 2, 3]
    if kind == "class":
        return [_ABSENT, "buy", "browse"]
    return _WIDE


def _oracle(definition, plan, domains) -> set[tuple[str, str]]:
    """The expected (code, state) set by brute force over the wide grid."""
    live: dict[str, list[str]] = {}
    found: set[tuple[str, str]] = set()
    for sid, edges in plan.items():
        keys = sorted(
            {
                k
                for _t, _p, logic, rq in edges
                for k in (*rq, *(["a", "b"] if logic is not None else []))
                if k in _KEYS
            }
        )
        pools = [_wide_pool(domains.get((sid, k))) for k in keys]
        fired: set[int] = set()
        for values in itertools.product(*pools):
            data = {k: v for k, v in zip(keys, values, strict=True) if v is not _ABSENT}
            passing = [
                (i, e)
                for i, e in enumerate(edges)
                if not any(is_missing(data, k) for k in e[3])
                and (e[2] is None or bool(evaluate_logic(e[2], data)))
            ]
            if passing:
                low = min(e[1] for _i, e in passing)
                wins = {e[0] for _i, e in passing if e[1] == low}
                fired |= {i for i, e in passing if e[0] in wins}
        live[sid] = [e[0] for i, e in enumerate(edges) if i in fired]
        found |= {("edge-unfireable", sid) for i in range(len(edges)) if i not in fired}
    allg = {sid: [e[0] for e in edges] for sid, edges in plan.items()}
    from fsm_llm.definitions import reachable_from

    start = definition.initial_state
    structural, firing = reachable_from(allg, [start]), reachable_from(live, [start])
    found |= {
        ("terminal-unreachable", sid)
        for sid, state in definition.states.items()
        if not state.transitions and sid in structural and sid not in firing
    }
    return found


class TestDifferential:
    def test_300_random_fsms_agree_with_brute_force(self):
        rng = random.Random(20261111)
        reported = silent = 0
        for _ in range(300):
            definition, plan, domains = _random_case(rng)
            got = _found(definition)
            want = _oracle(definition, plan, domains)
            assert got == want, (
                definition.model_dump(exclude_defaults=True),
                got,
                want,
            )
            reported += bool(got)
            silent += not got
        assert reported >= 30 and silent >= 30  # the generator exercises both


# The fuzz oracle owns its pool: it never imports the gate's ``POOL_STRINGS``
# (it would agree with itself). A test below pins that constant by value.
_FUZZ_LITS = ["yes", "No", "3", "3.0", "01", 3, 2.5, 0, -1, True, False, None, "x", ""]
_FUZZ_LITS += ["true", "1", 10]
_FUZZ_WIDE = [
    *[None, True, False, 0, 1, -1, 2, 3, 4, 10, 11, 2.5, 3.0, 0.5, -0.5],
    *["3", "3.0", " 3", "03", "yes", "YES", "no", "No", "true", "TRUE", "false"],
    *["x", "X", "", "1", "0", "01", "10", "2.5", "sample", "abc", 1e9, -1e9],
    *[[1], ["yes"], {"a": 1}],
]
_FUZZ_OPS = ["==", "===", "!=", "!==", ">", ">=", "<", "<=", "in", "!", "!!", "and"]
_FUZZ_OPS += ["or", "missing", "if"]
_FUZZ_ABSENT = object()


def _fuzz_logic(rng: random.Random, keys: list[str], depth: int = 0) -> Any:
    op = rng.choice(_FUZZ_OPS if depth < 2 else _FUZZ_OPS[:9])
    var = {"var": rng.choice(keys)}
    if op in ("==", "===", "!=", "!=="):
        args = [var, rng.choice([rng.choice(_FUZZ_LITS), {"var": rng.choice(keys)}])]
        rng.shuffle(args)
        return {op: args}
    if op in (">", ">=", "<", "<="):
        args = [var, rng.choice([3, 2.5, 0, 10, -1])]
        rng.shuffle(args)
        return {op: args}
    if op == "in":
        return {"in": [var, rng.sample(_FUZZ_LITS, 3)]}
    if op in ("!", "!!"):
        return {op: [rng.choice([var, _fuzz_logic(rng, keys, depth + 1)])]}
    if op == "missing":
        return {"missing": [rng.choice(keys)]}
    if op == "if":
        branches = [_fuzz_logic(rng, keys, depth + 1) for _ in range(2)]
        return {"if": [*branches, rng.choice([True, False])]}
    return {op: [_fuzz_logic(rng, keys, depth + 1) for _ in range(2)]}


def _fuzz_case(seed: int) -> tuple[FSMDefinition, list[str]]:
    rng = random.Random(seed)
    keys = rng.sample(["a", "b", "c"], rng.randint(1, 2))
    edges = []
    for _ in range(rng.randint(1, 3)):
        cond: dict[str, Any] = {"description": "c", "logic": _fuzz_logic(rng, keys)}
        if rng.random() < 0.4:
            cond["requires_context_keys"] = [rng.choice(keys)]
        edges.append(
            {
                "target_state": rng.choice(["t1", "t2"]),
                "description": "e",
                "priority": rng.choice([1, 2, 3]),
                "conditions": [cond],
            }
        )
    return (
        _fsm(_state("s0", edges), _state("t1"), _state("t2")),
        keys,
    )


def _fuzz_false_rejects(seed: int) -> list[Any]:
    """Edges G6a calls dead although some wide-pool context makes the REAL
    evaluator take their target (only edges alone on their target are judged:
    a shared target message cannot name one edge)."""
    try:
        definition, keys = _fuzz_case(seed)
    except Exception:
        return []
    dead = {
        f.message for f in analyze_dryrun(definition) if f.code == "edge-unfireable"
    }
    state, evaluator, bad = definition.states["s0"], TransitionEvaluator(), []
    for edge in state.transitions or []:
        alone = [t for t in state.transitions if t.target_state == edge.target_state]
        if len(alone) != 1 or not any(f"-> '{edge.target_state}'" in m for m in dead):
            continue
        for values in itertools.product(*[[_FUZZ_ABSENT, *_FUZZ_WIDE]] * len(keys)):
            data = {
                k: v for k, v in zip(keys, values, strict=True) if v is not _FUZZ_ABSENT
            }
            try:
                result = evaluator.evaluate_transitions(state, FSMContext(data=data))
            except Exception:
                continue
            wins = (
                {result.deterministic_transition}
                if result.deterministic_transition
                else {o.target_state for o in result.available_options or ()}
            )
            if edge.target_state in wins and evaluator.transition_passes(edge, data):
                bad.append((seed, edge.conditions[0].logic, data))
                break
    return bad


class TestAWiderPoolNeverFalselyRejects:
    def test_the_pool_strings_are_pinned_by_value(self):
        assert dryrun.POOL_STRINGS == ("", "true", "false")

    def test_a_string_literal_is_tried_in_upper_case_too(self):
        # `==` ignores case: the low-priority edge wins where `===` fails
        low = {"==": [_var("a"), "yes"]}
        strict = {"===": [_var("a"), "yes"]}
        definition = _fsm(
            _state(
                "s0",
                [_edge("t1", strict, priority=1), _edge("t2", low, priority=2)],
            ),
            _state("t1"),
            _state("t2"),
        )
        assert ("edge-unfireable", "s0") not in _found(definition)

    def test_the_empty_string_opens_a_not_null_edge(self):
        not_null = {"!": [{"==": [_var("c"), None]}]}
        same = {"===": [_var("a"), _var("a")]}
        definition = _fsm(
            _state(
                "s0",
                [
                    _edge("t1", not_null, priority=3),
                    _edge("t2", same, requires=("c",), priority=1),
                ],
            ),
            _state("t1"),
            _state("t2"),
        )
        assert ("edge-unfireable", "s0") not in _found(definition)

    def test_a_numeric_string_opens_a_not_true_and_positive_edge(self):
        logic = {"and": [{"!": [{"==": [_var("b"), True]}]}, {">": [_var("b"), 0]}]}
        definition = _fsm(
            _state(
                "s0",
                [
                    _edge("t1", logic, requires=("b",), priority=1),
                    _edge("t2", {"!!": [_var("b")]}, priority=2),
                ],
            ),
            _state("t1"),
            _state("t2"),
        )
        assert ("edge-unfireable", "s0") not in _found(definition)

    def test_the_found_seeds_and_a_fixed_block_have_no_false_reject(self):
        # seeds a 12,000-FSM scan found (the gate's pool lacked "", numeric and
        # bool strings and the upper-case form of a string literal), then 250 more
        found = [1941, 2280, 2291, 4519, 5809, 5892, 6362, 10368, 11338]
        seeds = [*found, *range(0, 250)]
        assert [bad for seed in seeds for bad in _fuzz_false_rejects(seed)] == []


class TestEdgeWitnesses:
    def test_a_witness_is_the_context_under_which_the_edge_wins(self):
        gate = {"in": [_var("k"), ["billing", "sales"]]}
        definition = _fsm(
            _state("ask", [_edge("done", gate, requires=("k",))]), _state("done")
        )
        found = dryrun.edge_witnesses(definition)
        assert found == {"ask": {0: {"k": "billing"}}, "done": {}}

    def test_the_witness_is_the_first_assignment_that_opens_the_edge(self):
        one = {"in": [_var("k"), ["billing", "sales"]]}
        two = {"==": [_var("k"), "other"]}
        definition = _fsm(
            _state(
                "ask",
                [
                    _edge("t1", one, requires=("k",)),
                    _edge("t2", two, requires=("k",), priority=2),
                ],
            ),
            _state("t1"),
            _state("t2"),
        )
        found = dryrun.edge_witnesses(definition)["ask"]
        assert found == {0: {"k": "billing"}, 1: {"k": "other"}}

    def test_a_dead_edge_is_absent_and_an_unknown_edge_is_none(self):
        dead = {"==": [_var("ok"), "yes"]}  # a bool key can never equal "yes"
        unknown = {"==": [{"var": "a.b"}, 1]}  # a dotted read is outside the fragment
        definition = _fsm(
            _state(
                "ask",
                [
                    _edge("done", dead, requires=("ok",)),
                    _edge("done", unknown, priority=2),
                ],
                field_extractions=BOOL,
            ),
            _state("done"),
        )
        assert dryrun.edge_witnesses(definition)["ask"] == {1: None}

    def test_the_inputs_are_never_narrowed_so_they_get_a_witness(self):
        gate = {"==": [_var("tier"), "gold"]}
        definition = _fsm(
            _state("ask", [_edge("done", gate, requires=("tier",))]), _state("done")
        )
        found = dryrun.edge_witnesses(definition, inputs=("tier",))
        assert found["ask"] == {0: {"tier": "gold"}}


class TestSpeed:
    def test_twelve_states_of_six_edges_in_under_two_seconds(self):
        ids = [f"s{i}" for i in range(12)]
        states = []
        for i, sid in enumerate(ids[:-1]):
            edges = [
                _edge(
                    ids[i + 1] if j == 0 else ids[min(i + j, 11)],
                    {
                        "and": [
                            {">": [_var("p"), j]},
                            {"<": [_var("q"), 5 - j]},
                            {"!=": [_var("p"), _var("q")]},
                        ]
                    },
                    priority=j + 1,
                )
                for j in range(6)
            ]
            states.append(_state(sid, edges))
        states.append(_state(ids[-1]))
        definition = _fsm(*states)
        start = time.perf_counter()
        analyze_dryrun(definition)
        assert time.perf_counter() - start < 2.0


# ------------------------------------------------------------- examples probe


def _clean_examples():
    for path in sorted(glob.glob(str(_REPO / "examples/**/*.json"), recursive=True)):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            definition = FSMDefinition(**data)
        except Exception:
            continue
        if FSMValidator(data).validate().is_valid:
            yield str(Path(path).relative_to(_REPO)), definition


# The read-only probe over the 14 loadable examples (manual read, step 1 of
# iteration 2). Every finding is a true dead edge under the evaluator: the first
# two are the shadowed stay edges G4 already reports (so ``repair_text`` hides
# them), the other two are G6a-specific: a priority-0 edge that needs only the
# same key present always beats the priority-1 edge, so that edge never fires.
_PROBE = {
    "examples/basic/form_filling/fsm.json": [("edge-unfireable", "confirm")],
    "examples/basic/simple_greeting/fsm.json": [("edge-unfireable", "conversation")],
    "examples/intermediate/book_recommendation/fsm.json": [
        ("edge-unfireable", "check_engagement"),
        ("terminal-unreachable", "end"),
    ],
    "examples/intermediate/product_recommendation/fsm.json": [
        ("edge-unfireable", "device_type_selection"),
        ("edge-unfireable", "smartphone_budget"),
        ("edge-unfireable", "laptop_budget"),
    ],
}


class TestExamplesProbe:
    def test_g6a_over_every_loadable_example_reports_only_true_dead_edges(self):
        rows = {
            name: [(f.code, f.state) for f in analyze_dryrun(definition)]
            for name, definition in _clean_examples()
        }
        assert len(rows) == 14
        assert {n: r for n, r in rows.items() if r} == _PROBE

    @pytest.mark.parametrize(
        ("name", "state", "key"),
        [
            ("book_recommendation", "check_engagement", "engagement_level"),
            ("product_recommendation", "device_type_selection", "device_type"),
            ("product_recommendation", "smartphone_budget", "budget"),
        ],
    )
    def test_the_evaluator_agrees_the_low_priority_edge_never_wins(
        self, name, state, key
    ):
        definition = dict(_clean_examples())[f"examples/intermediate/{name}/fsm.json"]
        winners = set()
        for value in ("low", "high", "x", 1, True, "0"):
            result = TransitionEvaluator().evaluate_transitions(
                definition.states[state], FSMContext(data={key: value})
            )
            winners.add(result.deterministic_transition)
        first = min(definition.states[state].transitions, key=lambda t: t.priority)
        assert winners == {first.target_state}

    def test_the_two_shadowed_stay_edges_are_the_ones_g4_already_reports(self):
        for name in ("basic/form_filling", "basic/simple_greeting"):
            definition = dict(_clean_examples())[f"examples/{name}/fsm.json"]
            shadowed = {
                f.state
                for f in analyze_liveness(definition, max_turns=200)
                if f.code == "shadowed-edge"
            }
            assert {f.state for f in analyze_dryrun(definition)} == shadowed
