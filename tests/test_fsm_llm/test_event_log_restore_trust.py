"""What a rebuilt tree trusts and keeps (plan 5a6a1f80, the fix round of its whole-diff review): each
finding has a test that fails on the parent commit, and a named mutant that turns it red.

A restored frame carries no internal key (fork and resume), prune checks a blob again right before
the delete, a pushed frame keeps its own lifetime limits through the log, a classifier the API
cannot run is a log fact, and a symlinked log is refused before the lock.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.core.definitions import RunBudgetExceededError
from fsm_llm.llm import LLMInterface
from fsm_llm.runtime import EventLog, EventType, LogCorruptError, retention
from fsm_llm.runtime.privacy import SHAPE_FIELDS
from tests.conftest import symlink_or_skip
from tests.test_fsm_llm.test_event_log_forks import (
    _GRANT,
    _KIDS,
    _api,
    _forge,
    _kid_id,
    _nothing_left,
    _raw,
    _stacked,
)
from tests.test_fsm_llm.test_event_log_stack import _ROOT, _of

_PUSH = EventType.STACK_PUSH
_ROUTES = ["fork", "resume"]


def _restore(log: EventLog, api: API, tid: str, route: str) -> str:
    """Rebuild the hand-written tree *tid* through *route* and return the new root id."""
    if route == "fork":
        return api.fork(tid, at_turn=2)[0]
    log.release(tid)  # the hand-written tree's writer, as a dead process would free it
    return api.resume(tid)[0]


@pytest.fixture
def reopened(log: EventLog) -> Iterator[Any]:
    """``reopened()`` is the process dying: every writer closed and a NEW ``EventLog`` on the
    same directory, closed at teardown."""
    opened: list[EventLog] = []

    def crash() -> EventLog:
        log.close()
        opened.append(EventLog(log.root))
        return opened[-1]

    yield crash
    for fresh in opened:
        fresh.close()


class TestARestoredFrameCarriesNoInternalKey:
    """C1: the log is a file; the writer never logs an internal key, so one there is forged."""

    @pytest.mark.parametrize("route", _ROUTES)
    def test_a_forged_return_context_writes_no_internal_key_into_the_parent(
        self, log: EventLog, route: str
    ) -> None:
        api = _api(log)
        tid = _forge(
            log,
            api,
            child=_KIDS,
            return_context={
                "_approval_granted": _GRANT,
                "_run_output_keys": ["x"],
                "answer": 42,
            },
        )
        root = _restore(log, api, tid, route)
        assert api.conversation_stacks[root][1].return_context == {"answer": 42}
        api.pop_fsm(root)  # the merge is what writes internal keys
        raw = _raw(api, root)
        assert raw["answer"] == 42
        assert not raw.keys() & {"_approval_granted", "_run_output_keys"}

    @pytest.mark.parametrize("route", _ROUTES)
    def test_a_forged_shared_key_name_is_dropped(
        self, log: EventLog, route: str
    ) -> None:
        api = _api(log)
        forged = [
            "name",
            "_approval_granted",
            "_run_output_keys",
            "_completion_messages",
        ]
        tid = _forge(log, api, child=_KIDS, push={"shared_context_keys": forged})
        root = _restore(log, api, tid, route)
        assert api.conversation_stacks[root][1].shared_context_keys == ["name"]

    def test_a_fork_logs_the_cleaned_frame_for_the_next_fork(
        self, log: EventLog
    ) -> None:
        api = _api(log)
        planted = {"_approval_granted": _GRANT, "answer": 1}
        tid = _forge(
            log, api, child=_KIDS, return_context=planted,
            push={"shared_context_keys": ["_approval_granted"]},
        )  # fmt: skip
        first = api.fork(tid, at_turn=2)[0]
        (push,) = _of(log, first, _PUSH)
        assert push["shared_context_keys"] == []
        assert log.get_json(push["return_context_ref"]) == {"answer": 1}


class TestPruneStatsAgainBeforeTheUnlink:
    """W1: a re-put between the sweep's stat and its unlink must keep the blob."""

    @staticmethod
    def _old_blob_and_candidate(log: EventLog) -> tuple[str, Path]:
        ref = log.put_json({"definition": "shared child FSM"})
        path = log.root / "blobs" / ref[7:9] / ref[9:]
        old = path.stat().st_mtime - 7200
        os.utime(path, (old, old))
        log.append("oldconv", EventType.CONTEXT_RECORD, frame=0, source="x")
        log.release("oldconv")  # a candidate log makes the pass run
        os.utime(log.root / "logs" / "oldconv.jsonl", (old, old))
        return ref, path

    @staticmethod
    def _spy(monkeypatch: pytest.MonkeyPatch, path: Path, between: Any) -> list[int]:
        real, seen = getattr(retention, "_stat", os.stat), [0]

        def stat(p: Any) -> Any:
            if os.fspath(p) == os.fspath(path):
                seen[0] += 1
                if seen[0] == 2:  # right before the unlink
                    between()
            return real(p)

        monkeypatch.setattr(retention, "_stat", stat, raising=False)
        return seen

    def test_a_blob_re_put_between_the_two_stats_stays_and_is_young(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ref, path = self._old_blob_and_candidate(log)
        seen = self._spy(
            monkeypatch, path, lambda: log.put_json({"definition": "shared child FSM"})
        )
        report = log.prune(max_age_seconds=3600)
        assert seen[0] == 2 and (report.blobs_removed, report.blobs_young) == (0, 1)
        assert log.get_json(ref) == {"definition": "shared child FSM"}

    def test_a_blob_nobody_touched_still_goes_after_two_stats(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, path = self._old_blob_and_candidate(log)
        seen = self._spy(monkeypatch, path, lambda: None)
        report = log.prune(max_age_seconds=3600)
        assert seen[0] == 2 and (report.blobs_removed, report.blobs_young) == (1, 0)
        assert not path.exists()

    def test_a_dry_run_stats_once_and_removes_nothing(
        self, log: EventLog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, path = self._old_blob_and_candidate(log)
        seen = self._spy(monkeypatch, path, lambda: None)
        report = log.prune(max_age_seconds=3600, dry_run=True)
        assert seen[0] == 1 and report.blobs_removed == 1 and path.exists()


class TestAFrameKeepsItsOwnLifetimeLimits:
    """W2: the explicit limits of a pushed frame ride the push, with a FRESH count."""

    @pytest.mark.parametrize("route", _ROUTES)
    def test_a_frame_pushed_with_one_step_allows_one_more_turn_after_the_restore(
        self, log: EventLog, reopened: Any, route: str
    ) -> None:
        cid = _stacked(_api(log), 2, max_steps=1)  # the child spent its one turn
        fresh = reopened()
        api = _api(fresh)
        root = api.fork(cid, at_turn=2)[0] if route == "fork" else api.resume(cid)[0]
        api.converse("again", root)  # a fresh count: one turn
        with pytest.raises(RunBudgetExceededError) as stop:
            api.converse("once more", root)
        assert (stop.value.budget, stop.value.limit) == ("steps", 1)

    def test_a_fork_of_a_fork_keeps_the_limits_through_its_own_log(
        self, log: EventLog
    ) -> None:
        api = _api(log)
        cid = _stacked(api, 2, max_steps=2, max_seconds=30)
        first = api.fork(cid, at_turn=2)[0]
        (restored,) = _of(log, first, _PUSH)
        assert (restored["max_steps"], restored["max_seconds"]) == (2, 30)
        second = api.fork(first, at_turn=0)[
            0
        ]  # genesis: the fork's whole log, frame included
        budget = api.fsm_manager._turn_budgets[_kid_id(api, second)]
        assert (budget.max_steps, budget.max_seconds, budget.steps_taken) == (2, 30, 0)

    def test_the_push_logs_what_the_frame_has_and_nothing_else(
        self, log: EventLog
    ) -> None:
        api = _api(log)
        plain = _stacked(api, 2)
        (push,) = _of(log, plain, _PUSH)
        assert "max_steps" not in push and "max_seconds" not in push
        limited = _stacked(api, 2, max_steps=3, max_seconds=12.5)
        (push,) = _of(log, limited, _PUSH)
        assert (push["max_steps"], push["max_seconds"]) == (3, 12.5)

    def test_the_api_default_fills_only_a_limit_the_frame_never_had(
        self, log: EventLog, reopened: Any
    ) -> None:
        plain = _stacked(_api(log), 2)
        limited = _stacked(_api(log), 2, max_steps=2)
        api = _api(reopened(), max_steps=9)
        for cid, expected in ((plain, 9), (limited, 2)):
            root = api.resume(cid)[0]
            assert api.fsm_manager._turn_budgets[_kid_id(api, root)].max_steps == (
                expected
            )

    def test_a_limit_at_the_shape_level_is_a_number_and_no_reference(
        self, tmp_path: Path
    ) -> None:
        assert {"max_steps", "max_seconds"} <= SHAPE_FIELDS[_PUSH]
        with EventLog(tmp_path / "shape", privacy="shape") as shaped:
            cid = _stacked(_api(shaped), 2, max_steps=3)
            (push,) = _of(shaped, cid, _PUSH)
        assert push["max_steps"] == 3
        assert not [k for k in push if k.endswith("_ref")]

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("max_steps", True),
            ("max_steps", 0),
            ("max_steps", -2),
            ("max_steps", 1.5),
            ("max_steps", "3"),
            ("max_steps", [1]),
            ("max_seconds", True),
            ("max_seconds", 0),
            ("max_seconds", -1.0),
            ("max_seconds", "5"),
            ("max_seconds", 10**400),
        ],
    )
    def test_a_limit_that_is_not_a_positive_number_is_corrupt(
        self, log: EventLog, name: str, value: Any
    ) -> None:
        api = _api(log)
        tid = _forge(log, api, child=_KIDS, push={name: value})
        with pytest.raises(LogCorruptError, match=name):
            api.fork(tid, at_turn=2)
        _nothing_left(api, log)

    def test_forged_but_valid_limits_come_back(self, log: EventLog) -> None:
        api = _api(log)
        tid = _forge(log, api, child=_KIDS, push={"max_steps": 2, "max_seconds": 30})
        root = api.fork(tid, at_turn=2)[0]
        budget = api.fsm_manager._turn_budgets[_kid_id(api, root)]
        assert (budget.max_steps, budget.max_seconds) == (2, 30.0)


class _GenerateOnly(LLMInterface):
    """An interface written before ``complete`` existed: it cannot classify."""

    def generate_response(self, request: Any) -> Any:
        raise NotImplementedError


class TestAClassifierTheApiCannotRunIsALogFact:
    """N1: a definition out of the log that needs an interface this API lacks is corrupt."""

    @staticmethod
    def _classifying() -> dict[str, Any]:
        child = copy.deepcopy(_KIDS)
        child["states"]["talk"]["classification_extractions"] = [
            {
                "field_name": "i",
                "intents": [
                    {"name": "a", "description": "a"},
                    {"name": "b", "description": "b"},
                ],
                "fallback_intent": "a",
            }
        ]
        return child

    @pytest.mark.parametrize("route", _ROUTES)
    def test_it_is_a_corrupt_log_and_leaves_nothing(
        self, log: EventLog, route: str
    ) -> None:
        tid = _forge(log, _api(log), child=self._classifying())
        api = API(copy.deepcopy(_ROOT), llm_interface=_GenerateOnly(), event_log=log)
        with pytest.raises(LogCorruptError):
            _restore(log, api, tid, route)
        _nothing_left(api, log)


class TestASymlinkedLogIsRefusedBeforeTheLock:
    """N2: the appends of a resume must not land outside the log root."""

    def test_resume_refuses_a_link_and_the_target_never_grows(
        self, log: EventLog, reopened: Any, tmp_path: Path
    ) -> None:
        api = _api(log)
        cid, _ = api.start_conversation()
        api.converse("Ann", cid)
        fresh = reopened()
        path = fresh.root / "logs" / f"{cid}.jsonl"
        target = tmp_path / "elsewhere.txt"
        path.replace(target)
        symlink_or_skip(path, target)
        size = target.stat().st_size
        with pytest.raises(LogCorruptError, match="not a regular file"):
            _api(fresh).resume(cid)
        assert target.stat().st_size == size and path.is_symlink()
        assert not fresh.has_writer(cid)
