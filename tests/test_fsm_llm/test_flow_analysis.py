"""Tests for `fsm_llm.flow_analysis` and `definitions.logic_has_unresolvable_reads`.

Plan 2cefeb61 step 2. The first section is a refactor proof: a golden table of
`FSMValidator._find_trap_components` written BEFORE the Tarjan function moved
out of the validator, green on the parent commit and still green after.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from fsm_llm import definitions, flow_analysis
from fsm_llm.definitions import FSMDefinition, State
from fsm_llm.pipeline import MessagePipeline
from fsm_llm.validator import FSMValidator

_REPO = Path(__file__).resolve().parents[2]


def _graph(edges: dict[str, list[str]]) -> dict:
    """An FSM dict from `{state: [targets]}`; the first key is the initial
    state, a state with no targets is terminal, dict order is definition order."""
    return {
        "name": "G",
        "initial_state": next(iter(edges)),
        "states": {
            state: {
                "id": state,
                "transitions": [{"target_state": target} for target in targets],
            }
            for state, targets in edges.items()
        },
    }


# 14 hand graphs, expected traps computed by hand (members and components in
# state-definition order). Cases: self-loop, plain loop, interlocking cycles,
# two traps, chain, loop with an exit, dangling target, unreachable trap, a
# diamond into a loop, a ring with a terminal exit, a tail into a loop, a
# bridge between two loops, members declared out of order, duplicate edges.
_GOLDEN_GRAPHS = {
    "self_loop": ({"a": ["a"]}, [["a"]]),
    "two_state_loop": ({"a": ["b"], "b": ["a"]}, [["a", "b"]]),
    "interlocking_cycles": (
        {"a": ["b"], "b": ["a", "c"], "c": ["b"]},
        [["a", "b", "c"]],
    ),
    "two_traps_ordered_by_first_member": (
        {
            "s": ["end", "t2", "t1"],
            "t2": ["t3"],
            "t3": ["t2"],
            "t1": ["t1"],
            "end": [],
        },
        [["t2", "t3"], ["t1"]],
    ),
    "chain_to_terminal": ({"a": ["b"], "b": ["c"], "c": ["end"], "end": []}, []),
    "loop_with_exit": ({"a": ["b"], "b": ["a", "end"], "end": []}, []),
    "dangling_exit_does_not_count": ({"a": ["b"], "b": ["a", "ghost"]}, [["a", "b"]]),
    "unreachable_trap_is_ignored": ({"a": ["end"], "end": [], "x": ["x"]}, []),
    "diamond_into_loop": (
        {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": ["e"], "e": ["d"]},
        [["d", "e"]],
    ),
    "ring_with_terminal_exit": (
        {"a": ["b"], "b": ["c"], "c": ["a", "end"], "end": []},
        [],
    ),
    "tail_into_loop": (
        {"a": ["b"], "b": ["c"], "c": ["d"], "d": ["c"]},
        [["c", "d"]],
    ),
    "bridge_between_two_loops": (
        {"a": ["b"], "b": ["a", "c"], "c": ["d"], "d": ["c"]},
        [["a", "b"], ["c", "d"]],
    ),
    "members_declared_out_of_order": (
        {"a": ["b"], "c": ["b"], "b": ["c"]},
        [["c", "b"]],
    ),
    "duplicate_edges": ({"a": ["b", "b"], "b": ["a", "a"]}, [["a", "b"]]),
}


class TestTrapComponentsGolden:
    """The validator's trap components, pinned across the Tarjan extraction."""

    def test_there_are_fourteen_hand_graphs(self):
        assert len(_GOLDEN_GRAPHS) == 14

    @pytest.mark.parametrize("name", sorted(_GOLDEN_GRAPHS))
    def test_hand_graph(self, name):
        edges, expected = _GOLDEN_GRAPHS[name]
        assert FSMValidator(_graph(edges))._find_trap_components() == expected

    def test_a_graph_without_states_has_no_trap(self):
        assert FSMValidator({"states": {}})._find_trap_components() == []

    @pytest.mark.parametrize(
        "path",
        sorted(glob.glob(str(_REPO / "examples/**/*.json"), recursive=True)),
        ids=lambda p: str(Path(p).relative_to(_REPO)),
    )
    def test_every_example_has_no_trap(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert FSMValidator(data)._find_trap_components() == []

    def test_the_example_glob_finds_the_fixtures(self):
        assert len(glob.glob(str(_REPO / "examples/**/*.json"), recursive=True)) >= 14

    def test_a_5000_state_chain_into_a_loop_does_not_recurse(self):
        n = 5000
        edges = {f"s{i}": [f"s{i + 1}"] for i in range(n - 1)}
        edges[f"s{n - 1}"] = [f"s{n - 1}"]
        traps = FSMValidator(_graph(edges))._find_trap_components()
        assert traps == [[f"s{n - 1}"]]

    def test_a_5000_state_ring_is_one_component(self):
        n = 5000
        edges = {f"s{i}": [f"s{(i + 1) % n}"] for i in range(n)}
        traps = FSMValidator(_graph(edges))._find_trap_components()
        assert traps == [[f"s{i}" for i in range(n)]]


# --------------------------------------------------------------
# strongly_connected_components
# --------------------------------------------------------------


def _scc(edges: dict[str, list[str]]) -> list[list[str]]:
    order = {node: i for i, node in enumerate(edges)}
    return flow_analysis.strongly_connected_components(edges, order)


def _brute_force_components(edges: dict[str, list[str]]) -> set[frozenset[str]]:
    """Components by mutual reachability: the definition, no Tarjan."""

    def reach(start: str) -> set[str]:
        seen, todo = {start}, [start]
        while todo:
            for nxt in edges[todo.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        return seen

    reachable = {node: reach(node) for node in edges}
    return {
        frozenset(m for m in edges if m in reachable[n] and n in reachable[m])
        for n in edges
    }


class TestStronglyConnectedComponents:
    def test_empty_graph(self):
        assert _scc({}) == []

    def test_every_node_is_in_exactly_one_component(self):
        edges = {"a": ["b"], "b": ["a", "c"], "c": [], "d": ["d"]}
        components = _scc(edges)
        assert sorted(m for c in components for m in c) == ["a", "b", "c", "d"]

    def test_singletons_and_loops_are_both_returned(self):
        assert sorted(_scc({"a": ["b"], "b": ["b"], "c": []})) == [
            ["a"],
            ["b"],
            ["c"],
        ]

    def test_members_follow_the_order_mapping_not_discovery(self):
        edges = {"a": ["b"], "c": ["b"], "b": ["c"]}
        assert [["c", "b"]] == [c for c in _scc(edges) if len(c) > 1]
        order = {"a": 0, "b": 1, "c": 2}
        found = flow_analysis.strongly_connected_components(edges, order)
        assert [c for c in found if len(c) > 1] == [["b", "c"]]

    def test_a_component_comes_after_every_component_it_reaches(self):
        edges = {"top": ["mid"], "mid": ["low", "mid2"], "mid2": ["mid"], "low": []}
        components = _scc(edges)
        position = {m: i for i, c in enumerate(components) for m in c}
        assert position["low"] < position["mid"] < position["top"]

    def test_interlocking_cycles_are_one_component(self):
        edges = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
        assert _scc(edges) == [["a", "b", "c"]]

    def test_a_target_that_is_not_a_node_raises(self):
        with pytest.raises(KeyError):
            _scc({"a": ["ghost"]})

    def test_the_input_is_not_mutated(self):
        edges = {"a": ["b"], "b": ["a"]}
        _scc(edges)
        assert edges == {"a": ["b"], "b": ["a"]}

    def test_a_5000_node_chain_does_not_recurse(self):
        n = 5000
        edges = {f"s{i}": [f"s{i + 1}"] for i in range(n - 1)}
        edges[f"s{n - 1}"] = []
        assert len(_scc(edges)) == n

    def test_a_5000_node_ring_is_one_component(self):
        n = 5000
        edges = {f"s{i}": [f"s{(i + 1) % n}"] for i in range(n)}
        assert _scc(edges) == [[f"s{i}" for i in range(n)]]

    def test_matches_brute_force_on_random_graphs(self):
        rng = random.Random(20261110)
        for _ in range(300):
            size = rng.randint(1, 8)
            nodes = [f"n{i}" for i in range(size)]
            edges = {n: [t for t in nodes if rng.random() < 0.25] for n in nodes}
            components = _scc(edges)
            assert {frozenset(c) for c in components} == _brute_force_components(edges)
            assert len(components) == len({frozenset(c) for c in components})
            for c in components:
                assert c == sorted(c, key=nodes.index)


# --------------------------------------------------------------
# logic_has_unresolvable_reads
# --------------------------------------------------------------

_UNRESOLVABLE = {
    "computed_var_name": {"var": {"cat": ["a", "b"]}},
    "empty_var_name": {"var": ""},
    "none_var_name": {"var": None},
    "numeric_var_name": {"var": 5},
    "var_without_arguments": {"var": []},
    "leading_dot_path": {"var": ".a"},
    "leading_dot_in_list_form": {"var": [".a", 1]},
    "computed_name_under_and": {
        "and": [{"==": [{"var": "a"}, 1]}, {"var": {"cat": ["x"]}}]
    },
    "empty_name_under_if": {"if": [{"var": "a"}, {"var": ""}, 1]},
    "empty_name_deep": {"or": [{"and": [{"!": {"var": ""}}]}]},
    "empty_missing_name": {"missing": ["a", ""]},
    "computed_missing_name": {"missing": [{"cat": ["a"]}]},
    "non_string_missing_some_name": {"missing_some": [1, ["a", 5]]},
    "computed_has_context_name": {"has_context": {"cat": ["a"]}},
    "empty_has_context_name": {"has_context": ""},
    "non_string_has_context_name": {"has_context": 3},
}

_RESOLVABLE = {
    "none": None,
    "empty_dict": {},
    "literal_comparison": {"==": [1, 1]},
    "plain_var": {"var": "a"},
    "dotted_var": {"var": "profile.email"},
    "var_with_default": {"var": ["a", 1]},
    "var_with_dict_default_is_raw": {"var": ["a", {"var": ""}]},
    "var_compared": {"==": [{"var": "a"}, {"var": "b"}]},
    "data_list_is_not_walked": {"in": [{"var": "x"}, [{"var": ""}]]},
    "missing_names": {"missing": ["a", "b"]},
    "missing_nothing": {"missing": []},
    "missing_some_names": {"missing_some": [1, ["a", "b"]]},
    "missing_some_wrong_arity_reads_nothing": {"missing_some": [1]},
    "has_context_name": {"has_context": "a"},
    "has_context_two_argument_form": {"has_context": [{"var": "ctx"}, "k"]},
    "nested_resolvable": {"and": [{"var": "a"}, {"or": [{"var": "b.c"}]}]},
}


class TestLogicHasUnresolvableReads:
    @pytest.mark.parametrize("name", sorted(_UNRESOLVABLE))
    def test_unresolvable_shape_is_flagged(self, name):
        assert definitions.logic_has_unresolvable_reads(_UNRESOLVABLE[name]) is True

    @pytest.mark.parametrize("name", sorted(_RESOLVABLE))
    def test_resolvable_shape_is_not_flagged(self, name):
        assert definitions.logic_has_unresolvable_reads(_RESOLVABLE[name]) is False

    def test_malformed_logic_raises_like_its_sibling(self):
        with pytest.raises(ValueError):
            definitions.logic_has_unresolvable_reads({"a": 1, "b": 2})

    def test_it_agrees_with_logic_referenced_keys_on_resolvable_names(self):
        logic = {"and": [{"var": "a.b"}, {"missing": ["c"]}]}
        assert definitions.logic_has_unresolvable_reads(logic) is False
        assert definitions.logic_referenced_keys(logic) == {"a", "c"}


# --------------------------------------------------------------
# G3 data flow (plan 2cefeb61 step 3)
# --------------------------------------------------------------


def _gate(key: str) -> dict:
    return {"==": [{"var": key}, 1]}


def _edge(target: str, *keys: str, logic: dict | None = None, priority: int = 100):
    """A transition dict; `keys` go to `requires_context_keys`."""
    condition: dict = {"description": "gate"}
    if keys:
        condition["requires_context_keys"] = list(keys)
    if logic is not None:
        condition["logic"] = logic
    edge: dict = {"target_state": target, "description": "e", "priority": priority}
    if len(condition) > 1:
        edge["conditions"] = [condition]
    return edge


def _state(sid: str, **spec) -> dict:
    return {"id": sid, "description": "d", "purpose": "p", **spec}


def _def(states: dict, handler_only=()) -> FSMDefinition:
    """An FSMDefinition from `{id: spec}`; the first key is the initial state."""
    return FSMDefinition(
        name="G",
        description="d",
        initial_state=next(iter(states)),
        handler_only_keys=list(handler_only),
        states={sid: _state(sid, **spec) for sid, spec in states.items()},
    )


def _field(name: str) -> dict:
    return {"field_name": name, "field_type": "str", "extraction_instructions": "x"}


_INTENTS = {
    "field_name": "intent",
    "intents": [{"name": "x", "description": "d"}, {"name": "y", "description": "d"}],
    "fallback_intent": "y",
}
_FINAL = {"==": [{"var": "completion_result.kind"}, "final"]}
_NOT_GUARANTEED = "return-key-not-guaranteed"
_UNDECLARED = "return-key-undeclared"

# name -> (states, expected [(code, state)], options). Options: handler_only,
# inputs, return_keys, guaranteed (the require_guaranteed_return_keys flag).
_FLOW_CASES = {
    "clean_linear": (
        {
            "a": {"required_context_keys": ["x"], "transitions": [_edge("b", "x")]},
            "b": {},
        },
        [],
        {"return_keys": ("x",)},
    ),
    "undeclared_logic_read": (
        {"a": {"transitions": [_edge("b", logic=_gate("ghost"))]}, "b": {}},
        [("undeclared-read", "a")],
        {},
    ),
    "handler_only_unproduced": (
        {"a": {"transitions": [_edge("b", "flag")]}, "b": {}},
        [("undeclared-read", "a")],
        {"handler_only": ("flag",)},
    ),
    "completion_result_is_produced": (
        {
            "a": {
                "completion": {"response_format_key": "_rf"},
                "transitions": [_edge("b", logic=_FINAL)],
            },
            "b": {"transitions": [_edge("c", logic=_gate("completion_result"))]},
            "c": {},
        },
        [],
        {},
    ),
    "unresolvable_read": (
        {
            "a": {"transitions": [_edge("b", logic={"var": {"cat": ["a", "b"]}})]},
            "b": {},
        },
        [("unresolvable-read", "a")],
        {},
    ),
    "bulk_extraction": (
        {
            "a": {"extraction_instructions": "collect", "transitions": [_edge("b")]},
            "b": {},
        },
        [("bulk-extraction", "a")],
        {},
    ),
    "bulk_with_a_declared_field": (
        {
            "a": {
                "extraction_instructions": "collect",
                "field_extractions": [_field("x")],
                "transitions": [_edge("b")],
            },
            "b": {},
        },
        [],
        {},
    ),
    "bulk_when_the_required_key_is_handler_only": (
        {
            "a": {
                "extraction_instructions": "collect",
                "required_context_keys": ["x"],
                "transitions": [_edge("b")],
            },
            "b": {},
        },
        [("bulk-extraction", "a")],
        {"handler_only": ("x",)},
    ),
    "input_provided_read": (
        {"a": {"transitions": [_edge("b", logic=_gate("uid"))]}, "b": {}},
        [],
        {"inputs": ("uid",)},
    ),
    "input_missing_read": (
        {"a": {"transitions": [_edge("b", logic=_gate("uid"))]}, "b": {}},
        [("undeclared-read", "a")],
        {},
    ),
    "diamond_one_branch_produces": (
        {
            "a": {"transitions": [_edge("b"), _edge("c")]},
            "b": {"required_context_keys": ["k"], "transitions": [_edge("d")]},
            "c": {"transitions": [_edge("d")]},
            "d": {"transitions": [_edge("e", logic=_gate("k"))]},
            "e": {},
        },
        [("undeclared-read", "d")],
        {},
    ),
    "diamond_both_branches_produce": (
        {
            "a": {"transitions": [_edge("b"), _edge("c")]},
            "b": {"required_context_keys": ["k"], "transitions": [_edge("d")]},
            "c": {"required_context_keys": ["k"], "transitions": [_edge("d")]},
            "d": {"transitions": [_edge("e", logic=_gate("k"))]},
            "e": {},
        },
        [],
        {},
    ),
    "cycle_reads_a_key_made_before_it": (
        {
            "a": {"required_context_keys": ["k"], "transitions": [_edge("b")]},
            "b": {"transitions": [_edge("a"), _edge("c", logic=_gate("k"))]},
            "c": {},
        },
        [],
        {},
    ),
    "cycle_key_made_only_inside_the_loop": (
        {
            "a": {"transitions": [_edge("b", logic=_gate("k"))]},
            "b": {
                "required_context_keys": ["k"],
                "transitions": [_edge("a"), _edge("c")],
            },
            "c": {},
        },
        [("undeclared-read", "a")],
        {},
    ),
    "classification_owned_gate_key": (
        {
            "a": {
                "classification_extractions": [_INTENTS],
                "transitions": [_edge("b", "intent")],
            },
            "b": {},
        },
        [],
        {"return_keys": ("intent",)},
    ),
    "return_key_produced_not_gated": (
        {"a": {"required_context_keys": ["k"], "transitions": [_edge("b")]}, "b": {}},
        [(_NOT_GUARANTEED, "b")],
        {"return_keys": ("k",)},
    ),
    "return_key_not_gated_but_flag_off": (
        {"a": {"required_context_keys": ["k"], "transitions": [_edge("b")]}, "b": {}},
        [],
        {"return_keys": ("k",), "guaranteed": False},
    ),
    "return_key_never_produced": (
        {"a": {"transitions": [_edge("b")]}, "b": {}},
        [(_UNDECLARED, "b")],
        {"return_keys": ("zzz",)},
    ),
    "return_key_never_produced_flag_off": (
        {"a": {"transitions": [_edge("b")]}, "b": {}},
        [(_UNDECLARED, "b")],
        {"return_keys": ("zzz",), "guaranteed": False},
    ),
    "return_key_gated_by_the_incoming_edge": (
        {"a": {"transitions": [_edge("b", "k")]}, "b": {}},
        [],
        {"return_keys": ("k",)},
    ),
    "return_key_from_inputs": (
        {"a": {"transitions": [_edge("b")]}, "b": {}},
        [],
        {"inputs": ("k",), "return_keys": ("k",)},
    ),
    "return_key_gated_on_one_branch_only": (
        {
            "a": {"transitions": [_edge("b", "k"), _edge("c", priority=200)]},
            "b": {},
            "c": {},
        },
        [(_NOT_GUARANTEED, "c")],
        {"return_keys": ("k",)},
    ),
    "return_key_proven_by_an_earlier_edge": (
        {
            "a": {"transitions": [_edge("b", "k")]},
            "b": {"transitions": [_edge("c")]},
            "c": {},
        },
        [],
        {"return_keys": ("k",)},
    ),
}


def _run_case(states, options):
    definition = _def(states, options.get("handler_only", ()))
    return flow_analysis.analyze_data_flow(
        definition,
        inputs=options.get("inputs", ()),
        return_keys=options.get("return_keys", ()),
        require_guaranteed_return_keys=options.get("guaranteed", True),
    )


class TestAnalyzeDataFlowTable:
    @pytest.mark.parametrize("name", sorted(_FLOW_CASES))
    def test_exact_codes(self, name):
        states, expected, options = _FLOW_CASES[name]
        findings = _run_case(states, options)
        assert sorted((f.code, f.state) for f in findings) == sorted(expected)

    def test_every_code_has_a_case(self):
        codes = {
            code for _, expected, _ in _FLOW_CASES.values() for code, _ in expected
        }
        assert codes == {
            "undeclared-read",
            "unresolvable-read",
            "bulk-extraction",
            _NOT_GUARANTEED,
            _UNDECLARED,
        }

    def test_a_finding_is_frozen_plain_data_naming_state_and_key(self):
        states, _, options = _FLOW_CASES["undeclared_logic_read"]
        (finding,) = _run_case(states, options)
        assert finding.analysis == "data-flow"
        assert "'a'" in finding.message and "'ghost'" in finding.message
        with pytest.raises(dataclasses.FrozenInstanceError):
            finding.code = "other"  # type: ignore[misc]

    def test_an_unreachable_state_poisons_neither_the_meet_nor_the_findings(self):
        definition = _def(
            {
                "a": {"required_context_keys": ["k"], "transitions": [_edge("b")]},
                "b": {"transitions": [_edge("c", logic=_gate("k"))]},
                "c": {},
            }
        )
        stray = State(**_state("z", transitions=[_edge("b", logic=_gate("ghost"))]))
        definition = definition.model_copy(
            update={"states": {**definition.states, "z": stray}}
        )
        assert flow_analysis.analyze_data_flow(definition) == []
        assert "z" not in flow_analysis.declared_available(definition, ())

    def test_declared_available_lists_what_every_path_made_by_the_gate(self):
        states, _, _ = _FLOW_CASES["diamond_one_branch_produces"]
        got = flow_analysis.declared_available(_def(states), ("i",))
        assert got["a"] == {"i"}
        assert got["b"] == {"i", "k"}
        assert got["c"] == {"i"}
        assert got["d"] == {"i"}


# --------------------------------------------------------------
# Seeded differential: worklist vs. brute-force path enumeration
# --------------------------------------------------------------

_KEYS = ("k0", "k1", "k2")


def _random_edge(rng: random.Random, target: str) -> dict:
    keys = [k for k in _KEYS if rng.random() < 0.3]
    logic = _gate(rng.choice(_KEYS)) if rng.random() < 0.25 else None
    return _edge(target, *keys, logic=logic)


def _random_fsm(rng: random.Random):
    """(states spec, handler_only, inputs, return_keys); every state is
    reachable and the last has no edge, so the definition always loads."""
    n = rng.randint(2, 6)
    ids = [f"s{i}" for i in range(n)]
    targets: dict[str, list[str]] = {sid: [] for sid in ids}
    for i in range(1, n):
        targets[ids[rng.randrange(i)]].append(ids[i])
    for _ in range(rng.randint(0, 3)):
        targets[rng.choice(ids[:-1])].append(rng.choice(ids))
    states: dict = {}
    for sid in ids:
        spec: dict = {"transitions": [_random_edge(rng, t) for t in targets[sid]]}
        required = [k for k in _KEYS if rng.random() < 0.3]
        fields = [_field(k) for k in _KEYS if rng.random() < 0.2]
        if required:
            spec["required_context_keys"] = required
        if fields:
            spec["field_extractions"] = fields
        states[sid] = spec

    def pick(p: float) -> list[str]:
        return [k for k in _KEYS if rng.random() < p]

    return states, pick(0.2), pick(0.25), pick(0.4)


def _brute_force(states, handler_only, inputs, return_keys):
    """The abstract rules by enumeration: a key is available at a state when
    EVERY simple path from the initial state makes it (a walk with a cycle only
    adds keys, so simple paths carry the intersection)."""

    def condition(t):
        return (t.get("conditions") or [{}])[0]

    def reads(spec):
        out = set()
        for t in spec["transitions"]:
            out |= set(condition(t).get("requires_context_keys", ()))
            if "logic" in condition(t):
                out.add(condition(t)["logic"]["=="][0]["var"])
        return out

    def produced(spec):
        keys = set(spec.get("required_context_keys", ()))
        keys |= {f["field_name"] for f in spec.get("field_extractions", ())}
        for t in spec["transitions"]:
            keys |= set(condition(t).get("requires_context_keys", ()))
        return keys - set(handler_only)

    declared: dict[str, set] = {}
    guaranteed: dict[str, set] = {}

    def walk(sid, visited, made, proven):
        out = made | produced(states[sid])
        declared[sid] = declared[sid] & out if sid in declared else set(out)
        guaranteed[sid] = guaranteed[sid] & proven if sid in guaranteed else set(proven)
        for t in states[sid]["transitions"]:
            if t["target_state"] not in visited | {sid}:
                walk(
                    t["target_state"],
                    visited | {sid},
                    out,
                    proven | set(condition(t).get("requires_context_keys", ())),
                )

    walk(next(iter(states)), set(), set(inputs), set(inputs))
    expected: Counter = Counter()
    for sid in declared:
        for _ in reads(states[sid]) - declared[sid]:
            expected[("undeclared-read", sid)] += 1
    ever = set(inputs).union(*(produced(states[sid]) for sid in declared))
    for sid in declared:
        if states[sid]["transitions"]:
            continue
        for key in dict.fromkeys(return_keys):
            if key not in ever:
                expected[(_UNDECLARED, sid)] += 1
            elif key not in guaranteed[sid]:
                expected[(_NOT_GUARANTEED, sid)] += 1
    return declared, expected


class TestDifferentialAgainstPathEnumeration:
    def test_300_random_fsms_agree_with_brute_force(self):
        rng = random.Random(20261110)
        seen: Counter = Counter()
        for _ in range(300):
            states, handler_only, inputs, return_keys = _random_fsm(rng)
            definition = _def(states, handler_only)
            declared, expected = _brute_force(states, handler_only, inputs, return_keys)
            got = flow_analysis.declared_available(definition, inputs)
            assert {k: set(v) for k, v in got.items()} == declared
            findings = flow_analysis.analyze_data_flow(
                definition, inputs=inputs, return_keys=return_keys
            )
            assert Counter((f.code, f.state) for f in findings) == expected
            seen.update(code for code, _ in expected)
        # not vacuous: every data-flow code the generator can reach was met
        assert {"undeclared-read", _UNDECLARED, _NOT_GUARANTEED} <= set(seen)


# --------------------------------------------------------------
# The produced-keys mirror, pinned to the pipeline (D-017 item 7)
# --------------------------------------------------------------

_HAND_STATES = {
    "required_and_edge_keys": {
        "required_context_keys": ["a"],
        "transitions": [_edge("t", "b"), _edge("t", "a", "c")],
    },
    "classification_owned_key_is_not_minted": {
        "required_context_keys": ["intent", "z"],
        "classification_extractions": [_INTENTS],
        "transitions": [_edge("t", "intent")],
    },
    "explicit_field_wins_over_the_owned_filter": {
        "classification_extractions": [_INTENTS],
        "field_extractions": [_field("intent")],
        "transitions": [_edge("t", "intent")],
    },
    "explicit_field_overrides_a_required_key": {
        "required_context_keys": ["a"],
        "field_extractions": [_field("a"), _field("b")],
    },
    "no_edges_no_keys": {},
    "completion_state_mints_nothing": {
        "completion": {"response_format_key": "_rf"},
        "transitions": [_edge("t", "x")],
    },
    "synthesize_state_mints_nothing": {
        "synthesize": {"goal": "g", "return_keys": ["r"], "on_failure": "t"},
        "transitions": [_edge("t", "x")],
    },
    "unconditional_and_logic_edges": {
        "transitions": [_edge("t"), _edge("t", logic=_gate("v"))],
    },
}


def _pipeline_keys(state: State) -> set[str]:
    return {
        c.field_name for c in MessagePipeline._build_field_configs_from_state(state)
    }


class TestProducedKeysMirrorsThePipeline:
    @pytest.mark.parametrize("name", sorted(_HAND_STATES))
    def test_hand_state(self, name):
        state = State(**_state("s", **_HAND_STATES[name]))
        assert flow_analysis._minted_keys(state) == _pipeline_keys(state)

    def test_every_example_state(self):
        checked, drifted = 0, []
        for path in sorted(
            glob.glob(str(_REPO / "examples/**/*.json"), recursive=True)
        ):
            try:
                definition = FSMDefinition(**json.loads(Path(path).read_text("utf-8")))
            except Exception:
                continue
            for sid, state in definition.states.items():
                checked += 1
                if flow_analysis._minted_keys(state) != _pipeline_keys(state):
                    drifted.append(f"{path}::{sid}")
        assert checked > 50
        assert drifted == []


# --------------------------------------------------------------
# G4: analyze_liveness
# --------------------------------------------------------------


def _chain(n: int, tail: list | None = None) -> dict:
    """s0 -> s1 -> ... -> s{n-1}; the last state is terminal or has `tail`."""
    states = {f"s{i}": {"transitions": [_edge(f"s{i + 1}")]} for i in range(n - 1)}
    states[f"s{n - 1}"] = {"transitions": tail or []}
    return states


_GHOST = _gate("ghost")  # read by no state: never declared
_TRAP, _NO_EXIT, _SHADOW = "trap", "cycle-no-exit", "shadowed-edge"
_TIE, _BUDGET = "permanent-tie", "path-exceeds-budget"

# name -> (states, expected [(code, state)], options). Options: inputs,
# max_turns. A gated edge's `requires_context_keys` key is minted at its own
# state, so it is always fireable there; a `logic` read of "ghost" never is.
_LIVE_CASES = {
    "loop_with_a_gated_exit": (
        {
            "a": {"transitions": [_edge("b")]},
            "b": {"transitions": [_edge("a"), _edge("end", "k", priority=10)]},
            "end": {},
        },
        [],
        {},
    ),
    "self_loop_with_an_exit": (
        {
            "s": {"transitions": [_edge("s"), _edge("end", "k", priority=10)]},
            "end": {},
        },
        [],
        {},
    ),
    "self_loop_without_an_exit": (
        {
            "a": {"transitions": [_edge("t", "k"), _edge("end", "k", priority=200)]},
            "t": {"transitions": [_edge("t")]},
            "end": {},
        },
        [(_TRAP, "t")],
        {},
    ),
    "interlocking_cycles_are_one_trap": (
        {
            "a": {"transitions": [_edge("b", "k"), _edge("end", "k", priority=200)]},
            "b": {"transitions": [_edge("c")]},
            "c": {"transitions": [_edge("b"), _edge("d", "k")]},
            "d": {"transitions": [_edge("c")]},
            "end": {},
        },
        [(_TRAP, "b")],
        {},
    ),
    "exit_gated_on_a_key_no_path_produces": (
        {
            "a": {"transitions": [_edge("b")]},
            "b": {"transitions": [_edge("c")]},
            "c": {"transitions": [_edge("b"), _edge("end", logic=_GHOST)]},
            "end": {},
        },
        [(_NO_EXIT, "b")],
        {},
    ),
    "the_same_exit_with_the_key_as_an_input": (
        {
            "a": {"transitions": [_edge("b")]},
            "b": {"transitions": [_edge("c")]},
            "c": {"transitions": [_edge("b"), _edge("end", logic=_GHOST)]},
            "end": {},
        },
        [],
        {"inputs": ("ghost",)},
    ),
    "exit_shadowed_by_an_unconditional_loop": (
        {
            "a": {"transitions": [_edge("b")]},
            "b": {
                "transitions": [_edge("b", priority=1), _edge("end", "k", priority=5)]
            },
            "end": {},
        },
        [(_SHADOW, "b"), (_NO_EXIT, "b")],
        {},
    ),
    "shadowed_edge_alone": (
        {
            "a": {"transitions": [_edge("x", priority=1), _edge("end", priority=2)]},
            "x": {"transitions": [_edge("end")]},
            "end": {},
        },
        [(_SHADOW, "a")],
        {},
    ),
    "equal_priority_edges_to_one_target_are_not_a_tie": (
        {"a": {"transitions": [_edge("end"), _edge("end")]}, "end": {}},
        [],
        {},
    ),
    "unconditional_edges_to_two_targets_tie_forever": (
        {
            "a": {"transitions": [_edge("x"), _edge("end")]},
            "x": {"transitions": [_edge("end")]},
            "end": {},
        },
        [(_TIE, "a")],
        {},
    ),
    "a_lower_ranked_gated_edge_means_no_permanent_tie": (
        {
            "a": {
                "transitions": [_edge("x", "k", priority=1), _edge("x"), _edge("end")]
            },
            "x": {"transitions": [_edge("end")]},
            "end": {},
        },
        [],
        {},
    ),
    "chain_one_over_the_budget": (_chain(6), [(_BUDGET, "s0")], {"max_turns": 4}),
    "chain_exactly_at_the_budget": (_chain(6), [], {"max_turns": 5}),
    "a_loop_weighs_its_state_count": (
        {
            "a": {"transitions": [_edge("b")]},
            "b": {"transitions": [_edge("c"), _edge("end", "k", priority=10)]},
            "c": {"transitions": [_edge("b")]},
            "end": {},
        },
        [(_BUDGET, "a")],
        {"max_turns": 2},
    ),
}


def _liveness(states, options):
    return flow_analysis.analyze_liveness(
        _def(states),
        inputs=options.get("inputs", ()),
        max_turns=options.get("max_turns", 40),
    )


class TestAnalyzeLivenessTable:
    @pytest.mark.parametrize("name", sorted(_LIVE_CASES))
    def test_exact_codes(self, name):
        states, expected, options = _LIVE_CASES[name]
        findings = _liveness(states, options)
        assert sorted((f.code, f.state) for f in findings) == sorted(expected)

    def test_every_code_has_a_case(self):
        codes = {c for _, expected, _ in _LIVE_CASES.values() for c, _ in expected}
        assert codes == {_TRAP, _NO_EXIT, _SHADOW, _TIE, _BUDGET}

    def test_a_finding_is_plain_data_that_quotes_state_ids(self):
        (finding,) = _liveness(*_LIVE_CASES["interlocking_cycles_are_one_trap"][::2])
        assert finding.analysis == "liveness"
        assert all(f"'{sid}'" in finding.message for sid in "bcd")
        with pytest.raises(dataclasses.FrozenInstanceError):
            finding.code = "x"  # type: ignore[misc]

    def test_an_unreachable_loop_is_ignored(self):
        definition = _def({"a": {"transitions": [_edge("end")]}, "end": {}})
        stray = State(**_state("dead", transitions=[_edge("dead")]))
        definition = definition.model_copy(
            update={"states": {**definition.states, "dead": stray}}
        )
        assert flow_analysis.analyze_liveness(definition) == []

    def test_a_5000_state_chain_is_judged_without_recursion(self):
        findings = flow_analysis.analyze_liveness(_def(_chain(5000)))
        assert [(f.code, f.state) for f in findings] == [(_BUDGET, "s0")]

    def test_a_5000_state_chain_into_a_loop_is_a_trap(self):
        loop = {
            "s0": {"transitions": [_edge("s1", "k"), _edge("end", "k", priority=2)]}
        }
        states = {**loop, **{k: v for k, v in _chain(5000).items() if k != "s0"}}
        states["s4999"] = {"transitions": [_edge("s4999")]}
        states["end"] = {}
        findings = flow_analysis.analyze_liveness(_def(states), max_turns=10**6)
        assert [(f.code, f.state) for f in findings] == [(_TRAP, "s4999")]


def _random_live_fsm(rng: random.Random):
    """(states spec, inputs, max_turns); every state is reachable, the last has
    no edge (so the definition loads), other leaves may be terminal too."""
    n = rng.randint(2, 7)
    ids = [f"s{i}" for i in range(n)]
    targets: dict[str, list[str]] = {sid: [] for sid in ids}
    for i in range(1, n):
        targets[ids[rng.randrange(i)]].append(ids[i])
    for _ in range(rng.randint(0, 4)):
        targets[rng.choice(ids[:-1])].append(rng.choice(ids))
    states: dict = {}
    for sid in ids:
        edges = []
        for target in targets[sid]:
            kind = rng.random()
            gate = (
                {"keys": (rng.choice(_KEYS),)}
                if kind < 0.2
                else {"logic": _gate(rng.choice(_KEYS))}
                if kind < 0.5
                else {}
            )
            edges.append(
                _edge(
                    target,
                    *gate.get("keys", ()),
                    logic=gate.get("logic"),
                    priority=rng.choice([1, 2, 100]),
                )
            )
        spec: dict = {"transitions": edges}
        required = [k for k in _KEYS if rng.random() < 0.3]
        if required:
            spec["required_context_keys"] = required
        states[sid] = spec
    return states, [k for k in _KEYS if rng.random() < 0.25], rng.randint(1, 5)


def _liveness_oracle(states, available, max_turns):
    """The G4 rules by transitive closure over plain dicts (no SCC code, no
    shared helper): the expected Counter of (code, state)."""
    order = list(states)
    initial = order[0]

    def edges(sid):
        for t in states[sid]["transitions"]:
            cond = (t.get("conditions") or [{}])[0]
            reads = set(cond.get("requires_context_keys", ()))
            if "logic" in cond:
                reads.add(cond["logic"]["=="][0]["var"])
            yield t["target_state"], t["priority"], not t.get("conditions"), reads

    def closure(adj):  # nodes reachable in one or more steps
        out = {}
        for s in adj:
            seen, todo = set(), list(adj[s])
            while todo:
                x = todo.pop()
                if x not in seen:
                    seen.add(x)
                    todo.extend(adj[x])
            out[s] = seen
        return out

    full = {s: [t for t, *_ in edges(s)] for s in order}
    reach = {initial} | closure(full)[initial]
    live: dict[str, list[str]] = {}
    expected: Counter = Counter()
    for s in reach:
        floor = min((p for _, p, u, _ in edges(s) if u), default=None)
        live[s] = []
        for target, prio, _, reads in edges(s):
            if floor is not None and prio > floor:
                expected[(_SHADOW, s)] += 1
            elif reads <= available[s]:
                live[s].append(target)
        lowest = min((p for _, p, _, _ in edges(s)), default=None)
        if len({t for t, p, u, _ in edges(s) if u and p == lowest}) > 1:
            expected[(_TIE, s)] += 1
    g_reach, e_reach = closure({s: full[s] for s in reach}), closure(live)
    finishing = {
        s
        for s in reach
        if not states[s]["transitions"]
        or any(not states[t]["transitions"] for t in g_reach[s])
    }

    def klass(s, closed):
        return frozenset({s} | {x for x in closed[s] if s in closed[x]})

    traps = {klass(s, g_reach) for s in reach if s in g_reach[s] and s not in finishing}
    expected.update((_TRAP, min(c, key=order.index)) for c in traps)
    classes = {klass(s, e_reach) for s in reach}
    for comp in classes:
        s = next(iter(comp))
        exits = {x for m in comp for x in live[m]} - comp
        if s in e_reach[s] and s not in {m for c in traps for m in c} and not exits:
            expected[(_NO_EXIT, min(comp, key=order.index))] += 1

    def longest(comp):
        onward = {klass(x, e_reach) for m in comp for x in live[m]} - {comp}
        return len(comp) + max((longest(c) for c in onward), default=0)

    if longest(klass(initial, e_reach)) - 1 > max_turns:
        expected[(_BUDGET, initial)] += 1
    return expected


class TestLivenessDifferential:
    def test_300_random_fsms_agree_with_closure_reachability(self):
        rng = random.Random(20261111)
        seen: Counter = Counter()
        for _ in range(300):
            states, inputs, max_turns = _random_live_fsm(rng)
            definition = _def(states)
            available = flow_analysis.declared_available(definition, inputs)
            expected = _liveness_oracle(states, available, max_turns)
            findings = flow_analysis.analyze_liveness(
                definition, inputs=inputs, max_turns=max_turns
            )
            assert Counter((f.code, f.state) for f in findings) == expected
            seen.update(code for code, _ in expected)
        # not vacuous: every liveness code the generator can reach was met
        assert {_TRAP, _NO_EXIT, _SHADOW, _TIE, _BUDGET} <= set(seen)


# --------------------------------------------------------------
# Examples probe (read-only): G3 and G4 over every loadable example
# --------------------------------------------------------------


def _clean_examples():
    for path in sorted(glob.glob(str(_REPO / "examples/**/*.json"), recursive=True)):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            definition = FSMDefinition(**data)
        except Exception:
            continue
        if FSMValidator(data).validate().is_valid:
            yield str(Path(path).relative_to(_REPO)), definition


class TestExamplesProbe:
    def test_both_analyses_run_over_every_clean_example(self, capsys):
        rows = []
        for name, definition in _clean_examples():
            findings = flow_analysis.analyze_data_flow(definition) + (
                flow_analysis.analyze_liveness(definition, max_turns=200)
            )
            codes = Counter(f"{f.analysis}:{f.code}" for f in findings)
            rows.append((name, len(definition.states), dict(codes)))
        flagged = [r for r in rows if r[2]]
        with capsys.disabled():
            for name, states, codes in rows:
                print(f"PROBE {name} states={states} findings={codes}")
            print(f"PROBE flagged {len(flagged)} of {len(rows)}")
        assert len(rows) >= 10
