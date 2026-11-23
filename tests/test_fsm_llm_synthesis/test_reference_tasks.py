"""The reference tasks of the retired synthesis bench and their user-turn loop, offline.

``reference_tasks.drive_fsm`` is the loop the step 5 rig generalises; these rows keep the
ported path honest: every hand-written child runs on the shared host with a scripted cheap
model, the strong model is never called (an unscripted ``ScriptedLLM`` raises on any call),
and the grader reads the values the run returned.
"""

from pathlib import Path

import pytest

from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM
from tests.test_fsm_llm_synthesis.reference_tasks import (
    HAND,
    TASKS,
    Task,
    drive_fsm,
    grade,
    hand_fsm,
    holds,
    host_definition,
)


def _typed(value: str) -> int | str:
    return int(value) if value.isdigit() else value


class TestTheHandArmRunsEveryTaskOffline:
    @pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
    def test_the_child_returns_the_gold_values_with_no_strong_call(
        self, task: Task, tmp_path: Path
    ):
        cheap = ScriptedRunnerLLM({k: _typed(v) for k, v in task.gold.items()}, {})
        strong = ScriptedLLM([])
        row = drive_fsm(
            HAND, task, tmp_path, {"compile": strong, "sim": strong, "run": cheap}
        )
        assert row["status"] == "ok", row
        assert row["per_key"] == dict.fromkeys(task.gold, True)
        assert row["source"] == "library"
        assert row["child_states"]

    def test_a_branch_task_that_asked_the_other_branch_is_graded_wrong(self):
        task = next(t for t in TASKS if t.id == "b1")
        asked = grade(task, task.gold, ["Which city is the destination?"], 0)
        assert asked["correct"] is False and asked["checks"]["other_branch"] is False
        clean = grade(task, task.gold, ["What is the company name?"], 0)
        assert clean["correct"] is True


class TestTheGrader:
    def test_a_gold_inside_a_list_of_labels_fails(self):
        banned = frozenset({"billing", "technical", "sales", "other"})
        assert holds("billing", "Billing", banned)
        assert not holds("billing", "billing, technical", banned)

    def test_the_host_names_the_task_goal_and_its_return_keys(self):
        for task in TASKS:
            work = host_definition(task)["states"]["work"]["synthesize"]
            assert work["goal"] == task.goal
            assert work["return_keys"] == list(task.gold)
            assert set(task.gold) <= {
                f["field_name"]
                for s in hand_fsm(task)["states"].values()
                for f in s.get("field_extractions", [])
            }
