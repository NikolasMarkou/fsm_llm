"""RunContext adoption by every runner (plan 7efd5cb4, roadmap row 1.14): the parity class.

Invariant 8 of the plan: with ``context=`` unset every runner is byte for byte what it was at the
plan base 51366ed3. ``tests/run_context_rig.collect_sections`` runs the parity matrix (PG-agent,
PG-reasoning, PG-workflow, PG-adapters: request bytes as sha256 of the normalised JSON, the stable
fields of every event, the results, the errors, the step numbers every ``before_step`` hook
received) and ``tests/fixtures/run_context_parity_golden.json`` holds what the same function
returned on a ``git archive`` of the base. The golden is the PARENT's, written by an out-of-repo
script that imports the archive copy: never regenerate it from the tree to make a row pass (a
difference is a runner that now sends, logs or counts something else; a deliberate change names the
decision and the section it moves).

The comparer is shown able to FAIL: ``TestTheComparerCanFail`` plants a changed request byte, an
extra hook call, an extra event and a dropped result field in a copy of the golden, and every one
is reported. Later steps of the plan add the acceptance class to this file.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from tests import run_context_rig as rig

REPO = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO / "tests" / "fixtures" / "run_context_parity_golden.json"
GOLDEN: dict[str, dict[str, Any]] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
SECTIONS = ("PG-adapters", "PG-agent", "PG-reasoning", "PG-workflow")


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")


def first_difference(got: Any, want: Any, path: str = "") -> str | None:
    """The path of the first place ``got`` differs from ``want``, or ``None`` when equal."""
    if isinstance(got, dict) and isinstance(want, dict):
        for key in sorted(set(got) | set(want)):
            if key not in got:
                return f"{path}/{key}: missing from the run"
            if key not in want:
                return f"{path}/{key}: not in the golden"
            found = first_difference(got[key], want[key], f"{path}/{key}")
            if found:
                return found
        return None
    if isinstance(got, list) and isinstance(want, list):
        for index, (a, b) in enumerate(zip(got, want, strict=False)):
            found = first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
        if len(got) != len(want):
            return f"{path}: {len(got)} items against {len(want)}"
        return None
    return None if got == want else f"{path}: {got!r} against {want!r}"


@pytest.fixture(scope="module")
def sections(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """The whole matrix, run once for the module (about 25 s: every case is a scripted solve)."""
    work = tmp_path_factory.mktemp("parity")
    return json.loads(json.dumps(rig.collect_sections(work)))


class TestTheGoldenIsNotVacuous:
    def test_every_section_is_there_and_holds_cases(self) -> None:
        assert tuple(sorted(GOLDEN)) == SECTIONS
        assert all(len(GOLDEN[name]) >= 5 for name in SECTIONS)

    def test_every_case_holds_a_result_or_an_error(self) -> None:
        for name in SECTIONS:
            for case, record in GOLDEN[name].items():
                assert ("result" in record) != ("error" in record), (name, case)

    def test_the_agent_section_reached_its_stop_paths(self) -> None:
        agent = GOLDEN["PG-agent"]
        assert agent["react_ok"]["result"]["stop_reason"] == "answered"
        assert agent["react_flagged_no_hitl"]["error"]["type"] == "AgentError"
        for case in ("react_max_iterations", "react_timeout", "react_tool_error"):
            assert "result" in agent[case] or "error" in agent[case], case
        assert agent["react_cut_reply"]["requests"]
        assert agent["react_stream"]["result"]["chunks"]
        assert agent["react_ok"]["runs"][0]["hook"] is True

    def test_the_reasoning_section_reached_the_solve_and_both_budget_errors(
        self,
    ) -> None:
        reasoning = GOLDEN["PG-reasoning"]
        assert reasoning["deductive_ok"]["result"]["visited"][-1] == "final_answer"
        assert reasoning["calculator_ok"]["result"]["solution"] == "36"
        assert "within 7 steps" in reasoning["never_ending_7"]["error"]["message"]
        assert reasoning["never_ending_7"]["error"]["details"] == [
            "conversation_id",
            "partial_context",
            "responses_so_far",
        ]
        assert "10 steps" in reasoning["classifier_budget"]["error"]["message"]

    def test_the_solve_loop_has_no_hook_and_its_stream_has_one(self) -> None:
        reasoning = GOLDEN["PG-reasoning"]
        assert [r["hook"] for r in reasoning["deductive_ok"]["runs"]] == [False, False]
        assert reasoning["stream_ok"]["runs"][0]["hook"] is True
        assert reasoning["stream_ok"]["runs"][0]["steps"]

    def test_the_workflow_and_adapter_sections_reached_their_outcomes(self) -> None:
        workflow = GOLDEN["PG-workflow"]
        assert workflow["linear"]["result"]["status"] == "completed"
        assert "Step budget exceeded" in workflow["step_budget"]["result"]["error"]
        adapters = GOLDEN["PG-adapters"]
        assert adapters["fsm_ok"]["result"]["status"] == "ok"
        assert adapters["fsm_budget"]["result"]["status"] == "aborted"
        assert adapters["fsm_cancelled"]["result"]["reason"] == "cancelled"
        assert adapters["solve_budget"]["result"]["reason"] == "budget_spent"
        assert adapters["workflow_budget"]["result"]["reason"] == "budget_spent"
        assert adapters["pattern_react_max_steps"]["error"]["type"] == "ValueError"


class TestTheComparerCanFail:
    """A comparer that cannot fail proves nothing: four planted mutants, each reported."""

    def test_identical_data_is_equal(self) -> None:
        assert first_difference(GOLDEN, copy.deepcopy(GOLDEN)) is None

    def test_a_changed_request_byte(self) -> None:
        mutant = copy.deepcopy(GOLDEN)
        row = mutant["PG-agent"]["react_ok"]["requests"][0]
        row["sha256"] = row["sha256"][:-1] + ("0" if row["sha256"][-1] != "0" else "1")
        found = first_difference(mutant, GOLDEN)
        assert found and "requests[0]/sha256" in found

    def test_an_extra_hook_call(self) -> None:
        mutant = copy.deepcopy(GOLDEN)
        mutant["PG-agent"]["react_ok"]["runs"][0]["steps"].append(99)
        found = first_difference(mutant, GOLDEN)
        assert found and "runs[0]/steps" in found

    def test_an_extra_event(self) -> None:
        mutant = copy.deepcopy(GOLDEN)
        mutant["PG-reasoning"]["deductive_ok"]["events"].append({"type": "extra"})
        found = first_difference(mutant, GOLDEN)
        assert found and "events" in found

    def test_a_dropped_result_field(self) -> None:
        mutant = copy.deepcopy(GOLDEN)
        del mutant["PG-adapters"]["fsm_ok"]["result"]["answer"]
        found = first_difference(mutant, GOLDEN)
        assert found and "result/answer" in found


class TestParityWithTheBase:
    """Invariant 8: with ``context=`` unset the tree is the base, case by case."""

    def test_the_run_has_the_golden_sections_and_cases(
        self, sections: dict[str, dict[str, Any]]
    ) -> None:
        assert {name: sorted(cases) for name, cases in sections.items()} == {
            name: sorted(cases) for name, cases in GOLDEN.items()
        }

    @pytest.mark.parametrize(
        ("section", "case"),
        [(name, case) for name in SECTIONS for case in sorted(GOLDEN[name])],
    )
    def test_the_case_equals_the_golden(
        self, sections: dict[str, dict[str, Any]], section: str, case: str
    ) -> None:
        found = first_difference(sections[section][case], GOLDEN[section][case])
        assert found is None, f"{section}/{case}{found}"
