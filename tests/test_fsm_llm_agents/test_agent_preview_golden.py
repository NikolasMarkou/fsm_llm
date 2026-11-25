"""The parent goldens of plan 659c07c3 (X3b): what the monitor will draw and launch by executable id.

``tests/fixtures/agent_preview_golden.json`` was written ONCE from a ``git archive`` of the parent
commit ab2b4499 by ``agent_preview_cells.py`` (never regenerated to silence a failure). These rows

- load it: the id set is the 18 ``create_agent`` ids and the 3 shipped presets, every graph has one
  initial node, swarm has no graph, native_fc previews 5 states against 3 in a default run;
- hold the tree to it (G1 the drawing route and the real agents' FSMs, G2 ``create_agent(<id>)``
  builds what the old ``_AGENT_CLASSES`` table built, G3 the ``/api/fsm/visualize`` bytes);
- prove the comparer can FAIL: each planted mutant of the actual document must print a difference.

Doubles only: a ``ScriptedLLM`` and stub tools; the suite refuses every network connect.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from tests.test_fsm_llm_agents import agent_preview_cells as cells

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = ROOT / "tests" / "fixtures" / "agent_preview_golden.json"

#: the 18 ``create_agent`` ids and the 3 shipped presets: a literal, so a new id is a visible diff
CREATE_AGENT_IDS = [
    "adapt",
    "auto_memory",
    "debate",
    "evaluator_optimizer",
    "maker_checker",
    "meta_builder",
    "native_fc",
    "orchestrator",
    "parallel_react",
    "plan_execute",
    "prompt_chain",
    "react",
    "reasoning_react",
    "reflexion",
    "rewoo",
    "self_consistency",
    "swarm",
    "verified_react",
]
PRESET_IDS = ["code-review", "data-extraction", "summarize"]
FILE_IDS = {
    "auto_memory",
    "debate",
    "evaluator_optimizer",
    "maker_checker",
    "native_fc",
    "orchestrator",
    "plan_execute",
    "prompt_chain",
    "react",
    "reflexion",
    "rewoo",
    "verified_react",
    *PRESET_IDS,
}
BUILDER_IDS = {
    "adapt",
    "meta_builder",
    "parallel_react",
    "reasoning_react",
    "self_consistency",
}


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def document(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """G1, G2 and G3 of the tree (one run: the cells build real agents and a TestClient)."""
    import os

    saved = os.environ.pop("FSM_LLM_MONITOR_API_KEY", None)
    try:
        return cells.compute()
    finally:
        if saved is not None:
            os.environ["FSM_LLM_MONITOR_API_KEY"] = saved


class TestTheGoldenFile:
    def test_the_id_set_is_the_18_ids_and_the_3_presets(self, golden):
        assert sorted(golden["g1"]) == sorted([*CREATE_AGENT_IDS, *PRESET_IDS])
        assert len(golden["g1"]) == 21

    def test_every_graph_has_exactly_one_initial_node(self, golden):
        for pattern, cell in golden["g1"].items():
            if cell["graph"] is None:
                assert pattern == "swarm", pattern
                continue
            initial = [n for n in cell["graph"]["nodes"] if n[1]]
            assert len(initial) == 1, pattern
            ids = [n[0] for n in cell["graph"]["nodes"]]
            assert len(ids) == len(set(ids)), pattern
            assert all(e[0] in ids and e[1] in ids for e in cell["graph"]["edges"])

    def test_the_route_of_each_id(self, golden):
        routes = {p: c["route"] for p, c in golden["g1"].items()}
        assert {p for p, r in routes.items() if r == "file"} == FILE_IDS
        assert {p for p, r in routes.items() if r == "builder"} == BUILDER_IDS
        assert [p for p, r in routes.items() if r == "none"] == ["swarm"]

    def test_a_file_fills_its_variables_with_placeholders(self, golden):
        for pattern in FILE_IDS - {"evaluator_optimizer", "prompt_chain"}:
            assert golden["g1"][pattern]["placeholders"], pattern
        assert golden["meta"]["placeholder"] == "<{}>"

    def test_native_fc_draws_five_states_and_a_default_run_has_three(self, golden):
        cell = golden["g1"]["native_fc"]
        assert len(cell["graph"]["nodes"]) == 5
        assert cell["real"]["states"] == 3

    def test_the_preview_equals_the_real_fsm_but_for_native_fc(self, golden):
        compared = 0
        for pattern, cell in golden["g1"].items():
            if cell["real"] is None:
                continue
            compared += 1
            same = cell["graph"] == cell["real"]["graph"]
            assert same is (pattern != "native_fc"), pattern
            assert len(cell["graph"]["nodes"]) == cell["real"]["states"] or (
                pattern == "native_fc"
            )
        assert (
            compared == 16
        )  # 12 file ids and 4 builder ids; swarm, meta_builder, presets have none

    def test_the_seven_old_types_are_recorded(self, golden):
        assert sorted(golden["g2"]) == sorted(cells.OLD_TYPES)
        for name, cell in golden["g2"].items():
            assert cell["id"] == cells.OLD_TYPES[name]
            assert cell["class"].endswith(f".{name}")
            assert len(cell["fsm_digest"]) == 64

    def test_g3_is_one_good_response_with_the_label_cut(self, golden):
        g3 = golden["g3"]
        assert g3["status"] == 200
        body = json.loads(g3["body"])
        assert set(body) == {"fsm", "nodes", "edges"}
        assert [e["label"] for e in body["edges"]] == [
            "No answer yet",
            "The user gave a complete and u",  # cut at VIZ_EDGE_LABEL_CHARS (30)
        ]

    def test_it_was_written_from_the_parent_and_holds_no_anchor(self, golden):
        assert "ab2b4499" in golden["meta"]["written_from"]
        assert "DECISION" not in GOLDEN_PATH.read_text(encoding="utf-8")


class TestTheTreeEqualsTheGolden:
    def test_g1_the_drawing_route_and_the_real_fsms(self, golden, document):
        assert cells.compare(golden["g1"], document["g1"], "/g1") == []

    def test_g2_create_agent_by_id_builds_what_the_old_table_built(
        self, golden, document
    ):
        assert cells.compare(golden["g2"], document["g2"]["new"], "/g2") == []

    def test_g2_the_old_way_is_still_the_same_agent(self, golden, document):
        assert cells.compare(golden["g2"], document["g2"]["old"], "/g2") == []

    def test_g3_the_one_drawing_path_did_not_move(self, golden, document):
        assert cells.compare(golden["g3"], document["g3"], "/g3") == []


class TestTheComparerCanFail:
    """The differential is void unless it fails under each planted mutant (printed in progress.md)."""

    @pytest.mark.parametrize("name", list(cells.MUTANTS))
    def test_each_planted_mutant_is_found(self, name, golden, document):
        mutated = cells.MUTANTS[name](document)
        found = cells.compare(_judged(golden), cells.tree_view(mutated))
        assert found, name

    def test_the_unmutated_document_has_no_difference(self, golden, document):
        assert cells.compare(_judged(golden), cells.tree_view(document)) == []

    def test_a_missing_key_a_new_key_and_a_changed_type_are_differences(self):
        assert cells.compare({"a": 1}, {}) == ["/a: missing in actual"]
        assert cells.compare({}, {"a": 1}) == ["/a: not in golden"]
        assert cells.compare({"a": True}, {"a": 1})
        assert cells.compare([1, 2], [1])

    def test_a_golden_is_refused_from_a_work_tree(self, tmp_path):
        if not (ROOT / ".git").exists():
            pytest.skip("not a git work tree (an archive copy)")
        assert cells.main(["--write", str(tmp_path / "x.json")]) == 1
        assert not (tmp_path / "x.json").exists()


def _judged(golden: dict[str, Any]) -> dict[str, Any]:
    """The part of the golden a tree is judged against (the meta block is a note)."""
    return {key: golden[key] for key in ("g1", "g2", "g3")}


def test_the_mutants_do_not_touch_the_document_they_are_given(document):
    before = copy.deepcopy(document)
    for name in ("edge dropped from one preview", "two ids swapped"):
        cells.MUTANTS[name](document)
    assert document == before
