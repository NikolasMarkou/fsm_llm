"""Tests for `fsm_llm.flow_analysis` and `definitions.logic_has_unresolvable_reads`.

Plan 2cefeb61 step 2. The first section is a refactor proof: a golden table of
`FSMValidator._find_trap_components` written BEFORE the Tarjan function moved
out of the validator, green on the parent commit and still green after.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

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
