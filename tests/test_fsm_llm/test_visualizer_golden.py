"""Golden pins for the ASCII, Mermaid and DOT renderers
(plan-2026-11-05T185903-5980fbb7 REVIEW-C4, C7).

Step 8 moved the three ASCII styles onto shared helpers (``_state_icons``,
``_state_type_label``, ``_box_style_name``, compact/minimal prefixes). The
values below are sha256 of the whole output, captured from the code as it was
before any later edit; a change to a glyph, an icon threshold or a label in any
style fails here. Update a hash only for a deliberate output change, and say so
in the CHANGELOG.

``find_longest_path`` is also pinned: exact on an acyclic graph, never above
the true longest simple path on a cyclic one (brute force, 20,000 random
graphs).
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import pytest

from fsm_llm.visualizer import (
    ASCII_STYLES,
    build_fsm_graph,
    find_longest_path,
    to_dot,
    to_mermaid,
    visualize_fsm_ascii,
)


def _state(
    sid: str, targets: tuple[str, ...] = (), keys: tuple[str, ...] = ()
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "id": sid,
        "description": f"State {sid}",
        "purpose": f"Purpose of {sid}",
        "transitions": [
            {
                "target_state": t,
                "description": f"{sid} to {t}",
                "priority": 100 + i,
            }
            for i, t in enumerate(targets)
        ],
    }
    if keys:
        state["required_context_keys"] = list(keys)
    return state


def _fsm(name: str, initial: str, *states: dict[str, Any], persona: str | None = None):
    data: dict[str, Any] = {
        "name": name,
        "description": f"{name} description",
        "initial_state": initial,
        "states": {s["id"]: s for s in states},
    }
    if persona:
        data["persona"] = persona
    return data


# `merge` has exactly two incoming edges (the `> 1` icon threshold), `fork`
# branches, `ask` needs user input, `done`/`failed` are terminal.
_MERGE = _fsm(
    "Merge Bot",
    "start",
    _state("start", ("fork",)),
    _state("fork", ("left", "right")),
    _state("left", ("merge",), keys=("name",)),
    _state("right", ("merge",)),
    _state("merge", ("done", "failed")),
    _state("done"),
    _state("failed"),
    persona="A careful assistant",
)
# A loop (ask <-> check), a self loop, and one terminal.
_LOOP = _fsm(
    "Loop Bot",
    "ask",
    _state("ask", ("check",), keys=("answer",)),
    _state("check", ("ask", "retry", "end")),
    _state("retry", ("retry", "ask")),
    _state("end"),
)
_SOLO = _fsm("Solo", "only", _state("only"))
_LINEAR = _fsm(
    "Linear Bot",
    "a",
    _state("a", ("b",)),
    _state("b", ("c",)),
    _state("c"),
)

_FSMS = {"merge": _MERGE, "loop": _LOOP, "solo": _SOLO, "linear": _LINEAR}

_GOLDEN: dict[str, str] = {
    "linear/compact": "f90142092557c492684b7f2b17a43e433d0bc02940ebb451795001aecfa067a7",
    "linear/dot": "4504b1dcc64f796df898442a21888b749abc167dd2fa3fec21d3b80c4a340e97",
    "linear/full": "9de3e02a4ad7dd1157ac8b8c677142c27dd52357e17ed8215167eac954a62480",
    "linear/mermaid": "4a40325d9f7e0ff1bbbeee8abf33d70828261ff77c7886bfa69561f494e7913b",
    "linear/minimal": "4a4569b1bf501ff1772ade02a2f9bb5970b3a1dc8f25b64ec2ac53461c885896",
    "loop/compact": "24776a0d39cb42b9e3f48577df25e27b8932fd59595a9445517dd5eb5b26e059",
    "loop/dot": "958d3339f562ccf6a233b3e01656c43e77f001035f886e3dfa562c33ad80664e",
    "loop/full": "6a1636c369e14aebf315b43f882831177c128f4f2389c02817b184aedb66f583",
    "loop/mermaid": "5b751bf771afedcc418b3906925377c59a4379dbe5b6f2844ab034ffaea3eb75",
    "loop/minimal": "812eaad116c806e82ae9e96522f5be72a5e802090849a0c4714a7974f4b44fdf",
    "merge/compact": "5b714cd97199801cb382c4682b51090e86bba90b92e743dbe0b63546afe253d3",
    "merge/dot": "a0d2736323556aa1f850310dd704312ca9f4997afd5cf21ee61fa0a0df2dccbf",
    "merge/full": "67f47be3f72051191f572983c88045717b63186c23b6b8206a1f6a1c94c0242d",
    "merge/mermaid": "e1b74f7ebeef85bdbf43358dc7124d5e22c7da2195ac91deee2f14e2e69e0db2",
    "merge/minimal": "fd88feef1df83aa9d74b8024173f92362a960c85152cb2329c2c2ba815893d6f",
    "solo/compact": "cdf6346c2d0c7bec7312ad877ac3890ab58f14ce54b82252ad03d095cbf87e9d",
    "solo/dot": "37183c0a0544c9bbdfd59b7bce6ea8299b7e331ae0156b5069413dc26ed8a21f",
    "solo/full": "716437fbd33358ded9997643746748017f3765b9b03232f7e8192723cc9ab04d",
    "solo/mermaid": "598efa464c47bbe25c17decbd12c92bff7f2d083c6f138be2f31a383a91f6bc3",
    "solo/minimal": "17478707073b5099332ec5764edd70f37706bdccba480cc594bac0414d8280f3",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _renders() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, fsm in _FSMS.items():
        for style in ASCII_STYLES:
            out[f"{key}/{style}"] = visualize_fsm_ascii(fsm, style)
        graph = build_fsm_graph(fsm)
        out[f"{key}/mermaid"] = to_mermaid(graph)
        out[f"{key}/dot"] = to_dot(graph)
    return out


class TestRenderersAreGolden:
    @pytest.mark.parametrize("case", sorted(_GOLDEN))
    def test_output_hash_is_unchanged(self, case):
        assert _sha(_renders()[case]) == _GOLDEN[case]

    def test_every_case_is_pinned(self):
        assert sorted(_renders()) == sorted(_GOLDEN)

    def test_the_merge_icon_and_terminal_glyph_are_in_the_output(self):
        """Readable anchors next to the hashes: the two things the dedup
        helpers decide."""
        from fsm_llm.visualizer import ICONS

        full = visualize_fsm_ascii(_MERGE, "full")
        assert ICONS["merge"] in full
        assert "(TERMINAL)" in full
        assert "(INITIAL)" in full
        assert "┗━━━> " in visualize_fsm_ascii(_MERGE, "compact")


# --------------------------------------------------------------
# find_longest_path
# --------------------------------------------------------------


def _random_graph(rng: random.Random) -> dict[str, list]:
    n = rng.randint(1, 6)
    names = [f"s{i}" for i in range(n)]
    graph: dict[str, list] = {}
    for name in names:
        targets = rng.sample(names, rng.randint(0, min(3, n)))
        graph[name] = [(t, "", []) for t in targets]
    return graph


def _brute_force(graph: dict[str, list], start: str) -> int:
    """States on the longest SIMPLE path from ``start`` (exponential)."""
    best = 0

    def walk(state: str, seen: frozenset[str]) -> None:
        nonlocal best
        best = max(best, len(seen))
        for target, _d, _k in graph.get(state, []):
            if target not in seen:
                walk(target, seen | {target})

    walk(start, frozenset({start}))
    return best


def _is_acyclic(graph: dict[str, list], start: str) -> bool:
    colour: dict[str, int] = {}

    def visit(state: str) -> bool:
        colour[state] = 1
        for target, _d, _k in graph.get(state, []):
            if colour.get(target) == 1:
                return False
            if target not in colour and not visit(target):
                return False
        colour[state] = 2
        return True

    return visit(start)


class TestFindLongestPath:
    def test_differential_against_brute_force(self):
        """Acyclic: exact. Cyclic: a lower bound (the memo is valid for DAG
        nodes only, so a cyclic answer can depend on transition order)."""
        rng = random.Random(20260506)
        exact = lower_bound = 0
        for _ in range(20_000):
            graph = _random_graph(rng)
            got = find_longest_path(graph, "s0")
            want = _brute_force(graph, "s0")
            if _is_acyclic(graph, "s0"):
                assert got == want, graph
                exact += 1
            else:
                assert 1 <= got <= want, graph
                lower_bound += 1
        assert exact > 1000 and lower_bound > 1000  # both branches exercised

    def test_a_cyclic_answer_may_depend_on_transition_order(self):
        """The documented limit: never above the truth, not always equal to it."""
        a = {
            "s": [("a", "", []), ("b", "", [])],
            "a": [("b", "", [])],
            "b": [("a", "", []), ("t", "", [])],
            "t": [],
        }
        b = {**a, "s": [("b", "", []), ("a", "", [])]}
        assert _brute_force(a, "s") == _brute_force(b, "s") == 4
        assert find_longest_path(a, "s") <= 4
        assert find_longest_path(b, "s") <= 4
