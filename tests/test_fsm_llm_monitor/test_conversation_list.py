"""The conversation list is one pass over the managed APIs (plan 9283c1fd step 3, audit claim O2).

``get_all_conversation_snapshots`` asked every API for its ids while holding ``_lock`` and then
probed EVERY managed API in turn for each id (``M x N(N+1)/2`` reads for ``N`` FSMs of ``M``
conversations). The rows pin the cost shape (each id is read once, each running API is listed once,
no API call under ``_lock``) and the list itself: ``tests/fixtures/conversation_list_golden.json``
was written from the BASE ``c8c96d5b`` by ``conversation_list_corpus``, so the list and its order
(attached API, active ids in dict order, ids of ``conversation_ids`` not yet seen, the ended cache)
are the old method's.

The comparer is list equality, so a reordered list fails it; that was proven by planted mutants (the
ended cache read before the owner probe, the two id loops swapped).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tests.test_fsm_llm_monitor import conversation_list_corpus as corpus

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "conversation_list_golden.json"
)
RUNNING_FSMS = (
    corpus.FSM_COUNT - 1
)  # fsm04 is a completed instance: never listed as active


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def fixture():
    built = corpus.build()
    yield built
    built.manager.close()


class TestTheListEqualsTheOldMethod:
    @pytest.mark.parametrize("include_ended", [True, False])
    def test_list_and_order_equal_the_golden(self, fixture, golden, include_ended):
        key = f"include_ended_{str(include_ended).lower()}"
        got = corpus.rows(
            fixture.manager.get_all_conversation_snapshots(include_ended=include_ended)
        )
        assert got == golden[key]

    def test_the_golden_holds_every_shape(self, golden):
        every = {r[0]: r for r in golden["include_ended_true"]}
        assert every["att-0"][1] == ""  # the attached API has no instance id
        assert every["fsm02-c1"][2:] == ["s0", False]  # live beats its cached copy
        assert every["fsm13-extra"][1] == "fsm13"  # live, not in conversation_ids
        assert every["fsm04-c0"][3] is False  # a completed instance's live conversation
        assert every["fsm00-c3"][2:] == ["end", True]  # ended, kept in conversation_ids
        assert every["old-0"][1] == "gone-fsm" and every["old-1"][1] == ""
        ended = [r[0] for r in golden["include_ended_false"] if r[3]]
        assert "old-0" not in ended and "old-1" not in ended
        assert len(golden["include_ended_true"]) == len(
            {r[0] for r in golden["include_ended_true"]}
        )

    def test_the_attached_conversation_has_an_empty_instance_id(self, fixture):
        snaps = {
            s.conversation_id: s
            for s in fixture.manager.get_all_conversation_snapshots()
        }
        assert [snaps[c].instance_id for c in corpus.ATTACHED_IDS] == ["", ""]
        assert snaps["fsm07-c0"].instance_id == "fsm07"


class TestTheListIsOnePass:
    def test_every_id_is_read_once(self, fixture):
        fixture.manager.get_all_conversation_snapshots()
        reads = fixture.probe.reads()
        assert Counter(reads) == Counter(dict.fromkeys(fixture.reachable, 1))
        assert len(reads) == len(fixture.reachable)

    def test_every_running_api_is_listed_once(self, fixture):
        fixture.manager.get_all_conversation_snapshots()
        listed = fixture.probe.lists()
        assert sorted(listed) == sorted(
            ["attached"] + [f"fsm{i:02d}" for i in range(corpus.FSM_COUNT) if i != 4]
        )
        assert len(listed) == RUNNING_FSMS + 1

    def test_no_api_is_called_while_the_lock_is_held(self, fixture):
        fixture.manager.get_all_conversation_snapshots()
        assert fixture.probe.calls  # the rows below read something
        assert fixture.probe.calls_under_lock() == []

    def test_the_lock_is_taken_a_linear_number_of_times(self, fixture):
        fixture.manager.get_all_conversation_snapshots()
        assert fixture.probe.lock.acquisitions <= len(fixture.reachable) + 4

    def test_get_active_conversations_calls_no_api_under_the_lock(self, fixture):
        ids = fixture.manager.get_active_conversations()
        assert ids[:2] == corpus.ATTACHED_IDS
        assert fixture.probe.calls_under_lock() == []
