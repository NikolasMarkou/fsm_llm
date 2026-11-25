"""G1 of plan 7d9c0955: the serialize and load bytes of the workflow definitions must not move.

``tests/fixtures/workflow_serialize_golden.json`` was written by ``golden_corpus.snapshot()`` run
against a ``git archive`` of the parent commit ``a8ad5d31``, never against the tree. The rows
compare the tree to it: the bytes of ``serialize()`` and ``serialize(callables=)`` (key order
included), the round trip through ``load_workflow``, and the key sets of the two tables of
named-object fields. Invariant 8 of the plan: ``load(serialize(w)) == w`` stays true for every
serializable definition, and the format of a document that is already written does not change.
The comparer was proven able to FAIL under four planted mutants (a field dropped, a type tag
renamed, keys sorted, a table entry removed): the record is in the plan's progress file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_fsm_llm_workflows import golden_corpus

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / ("workflow_serialize_golden.json")
)


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict[str, Any]:
    return golden_corpus.snapshot()


def test_the_corpus_is_not_vacuous(golden: dict[str, Any]) -> None:
    assert golden["base"] == "a8ad5d31"
    assert set(golden["cases"]) == set(golden_corpus.CORPUS)
    assert len(golden["cases"]) >= 5
    for name, case in golden["cases"].items():
        assert len(case["serialize"]) > 100, name
        assert len(case["serialize_named"]) > 100, name
        assert case["round_trip_equal"] is True, name
        assert case["round_trip_bytes_equal"] is True, name
        assert case["round_trip_error"] is None, name
    # every built-in serializable step class appears in the corpus
    tags = {
        step["type"]
        for case in golden["cases"].values()
        for step in json.loads(case["serialize_named"])["steps"].values()
    }
    assert {
        "AutoStep",
        "ConditionStep",
        "APIStep",
        "SwitchStep",
        "WaitEventStep",
        "TimerStep",
        "ParallelStep",
        "RetryStep",
        "ConversationStep",
        "AgentStep",
        "LLMStep",
        "ExecutableStep",
        "GoldenPingStep",
    } <= tags
    assert golden["tables"]["named_object_fields"]
    assert golden["tables"]["step_callable_exclude_fields"]


@pytest.mark.parametrize("name", sorted(golden_corpus.CORPUS))
def test_serialize_bytes_equal_the_parent(
    name: str, golden: dict[str, Any], current: dict[str, Any]
) -> None:
    assert current["cases"][name]["serialize"] == golden["cases"][name]["serialize"]


@pytest.mark.parametrize("name", sorted(golden_corpus.CORPUS))
def test_named_serialize_bytes_equal_the_parent(
    name: str, golden: dict[str, Any], current: dict[str, Any]
) -> None:
    assert (
        current["cases"][name]["serialize_named"]
        == golden["cases"][name]["serialize_named"]
    )


@pytest.mark.parametrize("name", sorted(golden_corpus.CORPUS))
def test_the_round_trip_still_holds(name: str, current: dict[str, Any]) -> None:
    assert current["cases"][name]["round_trip_error"] is None
    assert current["cases"][name]["round_trip_equal"] is True
    assert current["cases"][name]["round_trip_bytes_equal"] is True


def test_the_two_tables_keep_their_names(
    golden: dict[str, Any], current: dict[str, Any]
) -> None:
    assert current["tables"] == golden["tables"]
    exclude = set(golden["tables"]["step_callable_exclude_fields"])
    named = set(golden["tables"]["named_object_fields"])
    assert exclude < named
    assert named - exclude == {"agent", "registry", "callables"}
