"""G2 of plan 7d9c0955: the two workflow presets of the dashboard must not move.

``tests/fixtures/workflow_preset_golden.json`` was written by ``workflow_preset_corpus`` run
against a ``git archive`` of the parent commit ``a8ad5d31``, never against the tree. Per preset and
per value of ``amount`` (absent, 0, 999, 1000, ``"1500"``, ``None``, ``"abc"``, ``True``) the engine
run (status, error, public context, history, every lifecycle hook) and the monitor surface (the
preset list, the refusal texts, the bytes of ``GET /api/workflow/presets``, the launch route's error
bodies, one launched run per preset: info, status, instances, events) are compared to it.

Named deltas (decided in the plan, measured at BASE and kept in the fixture): when the presets
become documents their last step is ``AutoStep(next_step_id="")`` where ``_TerminalStep`` returned
no next step. That moves two values and nothing else: the ``next_step_id`` of the ``finish``
``step_completed`` hook (``None`` to ``""``) and the message of the ``finish`` history entry
(``"Reached terminal step 'finish'"`` to ``"Auto-transitioned to "``), the second visible on the
wire in ``/status`` and ``/instances``. ``EXPECTED`` names the ``document_*`` halves; the rows
that pin the deltas stay.

The comparer was proven able to FAIL under four planted mutants (a branch inverted, a context key
renamed, a step added, the threshold changed).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_fsm_llm_monitor import workflow_preset_corpus as corpus

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / ("workflow_preset_golden.json")
)
#: Which halves of the fixture the tree must equal: the presets are documents since step 7.
EXPECTED = {"engine": "document_runs", "monitor": "document_monitor"}


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current_runs() -> dict[str, Any]:
    return json.loads(json.dumps(corpus.engine_runs()))


@pytest.fixture(scope="module")
def current_monitor() -> dict[str, Any]:
    return json.loads(json.dumps(corpus.monitor_surface()))


def test_the_corpus_is_not_vacuous(golden: dict[str, Any]) -> None:
    assert golden["base"] == "a8ad5d31"
    assert sorted(golden["engine_runs"]) == ["demo_branching", "demo_linear"]
    for half in ("engine_runs", "document_runs"):
        for preset, runs in golden[half].items():
            assert sorted(runs) == sorted(label for label, _ in corpus.AMOUNTS)
            statuses = {r["status"] for r in runs.values()}
            # the branching preset fails on "abc"; the linear one ignores amount
            assert statuses == (
                {"completed", "failed"} if preset == "demo_branching" else {"completed"}
            )
            assert all(r["hooks"] and r["history"] for r in runs.values())
    for half in ("monitor", "document_monitor"):
        surface = golden[half]
        assert [p["id"] for p in surface["presets"]] == [
            "demo_branching",
            "demo_linear",
        ]
        assert len(surface["refused"]) == len(corpus.REFUSED_LAUNCHES)
        assert len(surface["launches"]) == 2 * len(corpus.LAUNCH_CONTEXTS)
        for launch in surface["launches"].values():
            assert launch["status"]["status"] == "completed"
            assert launch["events"][0][0] == "workflow_started"
            assert launch["events"][-1][0] == "workflow_completed"
    assert golden["named_deltas"]


def test_the_named_deltas_are_exactly_the_two_documented_ones(
    golden: dict[str, Any],
) -> None:
    engine = {
        (d[1], d[2], d[3]) for d in golden["named_deltas"] if d[0] == "engine_runs"
    }
    assert engine == {
        (
            "history.*.*",
            json.dumps("Reached terminal step 'finish'"),
            '"Auto-transitioned to "',
        ),
        ("hooks.*.*.next_step_id", "null", '""'),
    }
    for kind, path, base, document in golden["named_deltas"]:
        if kind == "monitor":
            assert path.endswith(".history.*.message"), path
            assert (base, document) == (
                json.dumps("Reached terminal step 'finish'"),
                '"Auto-transitioned to "',
            )


def test_the_document_form_differs_from_base_only_at_the_deltas(
    golden: dict[str, Any],
) -> None:
    """Every other value of the document-built presets equals the BASE presets."""
    for preset, runs in golden["engine_runs"].items():
        for label, base in runs.items():
            document = golden["document_runs"][preset][label]
            assert base["status"] == document["status"]
            assert base["error"] == document["error"]
            assert base["context"] == document["context"]
            assert [h[:2] for h in base["history"]] == [
                h[:2] for h in document["history"]
            ]
            assert [h[:1] for h in base["hooks"]] == [h[:1] for h in document["hooks"]]


@pytest.mark.parametrize("preset", ["demo_branching", "demo_linear"])
def test_engine_runs_equal_the_expected_half(
    preset: str, golden: dict[str, Any], current_runs: dict[str, Any]
) -> None:
    for label, _ in corpus.AMOUNTS:
        assert (
            current_runs[preset][label] == golden[EXPECTED["engine"]][preset][label]
        ), (preset, label)


def test_the_monitor_lists_and_refuses_as_expected(
    golden: dict[str, Any], current_monitor: dict[str, Any]
) -> None:
    expected = golden[EXPECTED["monitor"]]
    assert current_monitor["presets"] == expected["presets"]
    assert current_monitor["refused"] == expected["refused"]


def test_the_http_bytes_equal_the_expected_half(
    golden: dict[str, Any], current_monitor: dict[str, Any]
) -> None:
    expected = golden[EXPECTED["monitor"]]
    assert current_monitor["http_presets"] == expected["http_presets"]
    assert current_monitor["http_refused"] == expected["http_refused"]
    assert current_monitor["http_launch_linear"] == expected["http_launch_linear"]


def test_the_launched_runs_equal_the_expected_half(
    golden: dict[str, Any], current_monitor: dict[str, Any]
) -> None:
    expected = golden[EXPECTED["monitor"]]["launches"]
    assert sorted(current_monitor["launches"]) == sorted(expected)
    for key, launch in current_monitor["launches"].items():
        assert launch == expected[key], key


def with_the_two_deltas(base_run: dict[str, Any]) -> dict[str, Any]:
    """A BASE engine run with exactly the two named deltas applied, and nothing else."""
    run = copy.deepcopy(base_run)
    for entry in run["history"]:
        if entry[2] == "Reached terminal step 'finish'":
            entry[2] = "Auto-transitioned to "
    for name, data in run["hooks"]:
        if name == "step_completed" and data["step_id"] == "finish":
            assert data["next_step_id"] is None
            data["next_step_id"] = ""
    return run


@pytest.mark.parametrize("preset", ["demo_branching", "demo_linear"])
def test_the_tree_is_base_plus_exactly_the_two_deltas(
    preset: str, golden: dict[str, Any], current_runs: dict[str, Any]
) -> None:
    """The BASE half plus the two named deltas is the tree: a third difference fails here."""
    for label, _ in corpus.AMOUNTS:
        base = golden["engine_runs"][preset][label]
        moved = with_the_two_deltas(base)
        if base["status"] == "completed":
            assert moved != base, (preset, label)  # the deltas are not vacuous
        assert current_runs[preset][label] == moved, (preset, label)


def test_the_http_presets_bytes_are_the_base_bytes(
    golden: dict[str, Any], current_monitor: dict[str, Any]
) -> None:
    """The presets route answers the bytes of a8ad5d31, not only the document half."""
    assert current_monitor["http_presets"] == golden["monitor"]["http_presets"]
    assert current_monitor["presets"] == golden["monitor"]["presets"]
    assert current_monitor["refused"] == golden["monitor"]["refused"]
    assert current_monitor["http_refused"] == golden["monitor"]["http_refused"]
