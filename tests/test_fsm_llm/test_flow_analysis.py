"""Tests for `fsm_llm.flow_analysis` and `definitions.logic_has_unresolvable_reads`.

Plan 2cefeb61 step 2. The first section is a refactor proof: a golden table of
`FSMValidator._find_trap_components` written BEFORE the Tarjan function moved
out of the validator, green on the parent commit and still green after.
"""

from __future__ import annotations

import glob
import json
import random
from pathlib import Path

import pytest

from fsm_llm import definitions, flow_analysis
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
