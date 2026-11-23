"""A call run in the event log: fork, resume and the crash sweep (plan baa37f40, step 13 = K2.5).

The design needs no change of the log, the fold or ``stack/push`` (D-002, D-023): a call frame is an
ordinary ``stack/push`` above a parent whose logged state is the call state with the ``running``
record, so a restore re-derives the request from the parent's own state and ``settle`` (rule (b))
pushes the child again when the push was lost, and pops it (rule (a)) when its end was. The order
the writer must keep is: the parent turn (its ``turn/end`` holds the ``running`` record), THEN
``stack/push``; the reverse would leave a child with no marker.

Every row runs with the log on and compares with what the LIVE ``API`` showed (``_Watched``), never
with a second fold:

1. a fork at every turn of one call run, driven to the end, with the source log byte for byte as it
   was;
2. a death after the ``running`` record and before ``stack/push`` (a failing ``record_push``), and a
   death inside the child's turn: ``resume`` and the run finish with ONE push and ONE pop;
3. a death at EVERY event offset of one call run, and a torn tail of lengths of every line: the log
   is the acknowledged prefix, ``resume`` shows the golden view of that prefix, and the run goes on
   to the same result with one push and one pop whatever the offset;
4. a plain ``restore_session`` in the middle of a call (the snapshot restores the root frame only)
   restarts the call through rule (b); a stale final ``call_result`` pushes nothing.

Offline: a scripted model, a one-class resolver, no sleep, no thread.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fsm_llm import API, FileSessionStore, create_handler
from fsm_llm.core.definitions import ConversationBusyError, FSMError
from fsm_llm.handlers import clear_keys_on_entry
from fsm_llm.runtime import EventLog
from fsm_llm.runtime.stores import fold_stack, jsonl
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.event_log_support import _Dead
from tests.test_fsm_llm.test_call_settle import (
    _TARGET,
    _child_definition,
    _final,
    _flow,
    _Resolver,
)
from tests.test_fsm_llm.test_call_state import _parent
from tests.test_fsm_llm.test_event_log_crash import _Crash
from tests.test_fsm_llm.test_event_log_stack import (
    _FOLD_TYPES,
    _folded_view,
    _view,
    _Watched,
)

_CARRY = {kind.value for kind in _FOLD_TYPES}
_View = tuple[int, list[tuple[str, dict[str, Any]]]]
_STEPS = (
    2  # the child is c1 -> c2 -> c3: c2 speaks, c3 is terminal and writes `solution`
)


def _api(log: EventLog, **kwargs: Any) -> API:
    """The parent API of every row: a fresh one is the 'other process' of a resume."""
    resolver = _Resolver({_TARGET: _child_definition(_STEPS, None, "c-speaks")})
    answer = (
        create_handler("answer")
        .on_state_entry(f"c{_STEPS + 1}")
        .do(lambda ctx: {"solution": f"solved {ctx.get('topic')}"})
    )
    return API.from_definition(
        _parent(None, response_instructions=""),
        llm_interface=kwargs.pop("llm", None) or ScriptedLLM(default_response="r"),
        targets=resolver,
        handlers=[answer],
        event_log=log,
        **kwargs,
    )


def _play(api: API) -> str:
    """Start, move into the call state, then run: the request, the push, the child, the pop."""
    cid, _ = api.start_conversation({"topic": "why"})
    api.advance(cid)
    api.run_until_terminal(cid, max_steps=10)
    return cid


def _parent_state(api: API, cid: str) -> str:
    return api.fsm_manager.get_conversation_state(
        api.conversation_stacks[cid][0].conversation_id
    )


def _finish(api: API, cid: str, *, run: bool = False) -> None:
    """Drive a restored conversation to the end: one ``advance`` per round (the entry runs
    ``settle``), or with *run* ONE ``run_until_terminal`` and nothing else (the loop most hosts
    call, which must settle an ended call child itself)."""
    if run:
        api.run_until_terminal(cid, max_steps=10)
        return
    for _ in range(8):
        if api.get_stack_depth(cid) == 1 and _parent_state(api, cid) == "done":
            return
        api.advance(cid)
    pytest.fail("the call run did not reach its terminal state")


def _count(log: EventLog, cid: str, kind: str) -> int:
    return sum(e["type"] == kind for e in log.read(cid))


def _assert_finished(
    api: API, log: EventLog, cid: str, where: str, *, pushes: int = 1, pops: int = 1
) -> None:
    """The one result of every offset: the success edge fired, the declared return merged once,
    the final record as a run with no crash writes it, nothing left behind, ONE push, ONE pop
    in the log of *cid* (a fork after the pop has none of either in its own log)."""
    frame = api.conversation_stacks[cid][0].conversation_id
    assert api.get_stack_depth(cid) == 1 and _parent_state(api, cid) == "done", where
    assert api.get_data(cid)["answer"] == "solved why", where
    assert "solution" not in api.get_data(cid), where
    raw = api.fsm_manager.copy_raw_context_data(frame)
    assert raw["call_result"] == _final("ok", None), where
    assert list(api.fsm_manager.instances) == [frame], where  # no orphan child
    assert _count(log, cid, "stack/push") == pushes, where
    assert _count(log, cid, "stack/pop") == pops, where
    folded = fold_stack(log.read(cid), log.get_json)
    assert folded is not None and folded.depth == 1, where


@dataclass(frozen=True)
class _Golden:
    """The uncrashed run: its directory and root id, its events, and the live view at each
    state-carrying event keyed by ``seq``."""

    root: Path
    cid: str
    events: list[dict[str, Any]]
    views: dict[int, _View]

    def of(self, events: list[dict[str, Any]]) -> _View | None:
        last = None
        for event in events:
            if event["type"] in _CARRY:
                last = event["seq"]
        return None if last is None else self.views[last]

    def at(self, cut: int) -> _View | None:
        return self.of(self.events[:cut])

    def seqs(self, kind: str) -> list[int]:
        return [e["seq"] for e in self.events if e["type"] == kind]


def _run_golden(root: Path) -> _Golden:
    with _Watched(root) as log:
        api = _api(log)
        log.api = api  # the hook reads the live view after each state-carrying append
        cid = _play(api)
        log.api = None
        _assert_finished(api, log, cid, "golden")
        events = list(log.read(cid))
        views = dict(log.seen)
    return _Golden(root, cid, events, views)


@pytest.fixture(scope="module")
def golden(tmp_path_factory: pytest.TempPathFactory) -> _Golden:
    return _run_golden(tmp_path_factory.mktemp("call_golden"))


class TestTheScenario:
    def test_the_run_holds_what_the_sweep_claims_to_cover(
        self, golden: _Golden
    ) -> None:
        """Non-vacuity: the order the design needs, and every offset kind the sweep visits."""
        (push,) = golden.seqs("stack/push")
        (pop,) = golden.seqs("stack/pop")
        by_seq = {e["seq"]: e for e in golden.events}
        # the parent turn that wrote the request is the one right before the push, same frame
        assert by_seq[push - 1]["type"] == "turn/end" and by_seq[push - 1]["frame"] == 0
        assert by_seq[push]["depth"] == 2 and by_seq[pop]["depth"] == 1
        assert golden.views[push - 1][1][0][1]["call_result"]["status"] == "running"
        assert golden.views[push][0] == 2 and golden.views[pop][0] == 1
        assert golden.views[pop][1][0][1]["call_result"]["status"] == "ok"
        # the child's model call and the parent's last turn are inside the swept range
        child = [e["type"] for e in golden.events if e["frame"] == 1]
        assert "llm/request" in child and child.count("turn/end") == _STEPS
        assert (
            golden.events[-1]["type"] == "turn/end" and golden.events[-1]["frame"] == 0
        )
        assert {view[0] for view in golden.views.values()} == {1, 2}


def _die_after(root: Path, survive: int) -> str | None:
    """Run the scenario until the (survive+1)-th write dies; return the root id of the log."""
    crash = _Crash(survive)
    with EventLog(root) as log, mock.patch.object(jsonl, "_write", crash):
        api = _api(log)
        with pytest.raises(_Dead):
            _play(api)
    assert crash.dead
    with EventLog(
        root
    ) as reopened:  # the dead writer's lock is free once its fd is closed
        ids = reopened.conversation_ids()
    return ids[0] if ids else None


def _recover(
    root: Path,
    golden: _Golden,
    cid: str,
    cut: int,
    *,
    finish: bool = True,
    run: bool = False,
) -> None:
    """Resume the log in *root* (its first *cut* events, maybe a torn tail after them) in a FRESH
    API: the view equals what the live API showed at the last state-carrying event of the prefix,
    no model is asked to resume, and the run then goes on to the one result."""
    where = f"after {cut} events"
    with EventLog(root) as log:
        events = list(log.read(cid))
        assert [e["seq"] for e in events] == list(range(1, cut + 1)), where
        want = golden.at(cut)
        folded = fold_stack(iter(events), log.get_json)
        assert (None if folded is None else _folded_view(folded)) == want, where
        llm = ScriptedLLM(default_response="r")
        api = _api(log, llm=llm)
        if want is None:
            with pytest.raises(FSMError):
                api.resume(cid)
            return
        resumed, state = api.resume(cid)
        assert resumed == cid and _view(api, cid) == want, where
        assert state.metadata.get("unknown_tool_calls", []) == [], where
        assert llm.call_history == [], where  # resuming asks no model
        if finish:
            _finish(api, cid, run=run)
            _assert_finished(api, log, cid, where)
        api.end_conversation(cid)


class TestEveryDeathOffset:
    def test_each_offset_resumes_to_the_golden_view_and_one_run_finishes_with_one_push_and_one_pop(
        self, golden: _Golden, tmp_path: Path
    ) -> None:
        covered: set[int] = set()
        for survive in range(len(golden.events)):
            root = tmp_path / f"k{survive}"
            cid = _die_after(root, survive)
            if cid is None:
                assert survive == 0  # nothing landed: nothing to resume
                covered.add(survive)
                continue
            with EventLog(root) as log:
                types = [(e["type"], e["frame"]) for e in log.read(cid)]
            # the log is exactly the acknowledged prefix of the golden run, nothing more
            assert types == [(e["type"], e["frame"]) for e in golden.events[:survive]]
            _recover(root, golden, cid, survive, run=True)
            covered.add(survive)
        full = tmp_path / "full"
        shutil.copytree(golden.root, full)
        _recover(full, golden, golden.cid, len(golden.events), run=True)
        covered.add(len(golden.events))
        assert covered == set(
            range(len(golden.events) + 1)
        )  # EVERY k, including 0 and all


class TestEveryTornTail:
    def test_a_torn_line_is_cut_and_resume_equals_the_clean_prefix(
        self, golden: _Golden, tmp_path: Path
    ) -> None:
        """Every proper prefix of every line of the stack and of the parent's request turn (a
        stride over the long ones, the first and last byte always), half of every other line. The
        tail is cut by the writer and what stands is the acknowledged prefix."""
        raw = (golden.root / "logs" / f"{golden.cid}.jsonl").read_bytes()
        lines = raw.splitlines(keepends=True)
        assert len(lines) == len(golden.events) and b"".join(lines) == raw
        scratch = tmp_path / "scratch"
        shutil.copytree(golden.root, scratch)
        path = scratch / "logs" / f"{golden.cid}.jsonl"
        (push,) = golden.seqs("stack/push")
        (pop,) = golden.seqs("stack/pop")
        tried: dict[int, int] = {}
        for seq, line in enumerate(lines, start=1):
            acked = b"".join(lines[: seq - 1])
            if seq in (push - 1, push, pop):
                stride = max(1, len(line) // 40)
                lengths = sorted({*range(0, len(line), stride), len(line) - 1})
            else:
                lengths = [len(line) // 2]
            for length in lengths:
                path.write_bytes(acked + line[:length])
                # only the last prefix of a line is driven to the end (the rest end the same)
                _recover(
                    scratch, golden, golden.cid, seq - 1, finish=length == lengths[-1]
                )
                data = path.read_bytes()  # the writer cut the tail, acked bytes stand
                assert data.startswith(acked) and (data == b"" or data.endswith(b"\n"))
                tried[seq] = tried.get(seq, 0) + 1
        assert set(tried) == set(range(1, len(lines) + 1))
        assert min(tried[push - 1], tried[push], tried[pop]) > 20  # non-vacuity


class TestAKillBeforeTheLogHasThePush:
    def test_resume_asks_for_the_push_once_and_the_child_finishes_and_pops_once(
        self, tmp_path: Path
    ) -> None:
        """The process dies in ``record_push``: the child exists only in memory and the log ends
        with the parent turn that wrote the ``running`` record. The fresh API finds the request in
        the parent's own state, pushes the child ONCE (rule (b), a new child id), and the log has
        one push and one pop. Kills: a settle that cannot re-request from a restored parent (the
        call would hang), or one that pushes twice."""
        root = tmp_path / "log"
        with EventLog(root) as log:
            api = _api(log)
            with (
                mock.patch("fsm_llm.engine.api.stack.record_push", side_effect=_Dead),
                pytest.raises(_Dead),
            ):
                _play(api)
            doomed = api.conversation_stacks
            (cid,) = log.conversation_ids()
            dead_child = next(iter(doomed.values()))[-1].conversation_id
            last = list(log.read(cid))[-1]
        assert last["type"] == "turn/end" and last["frame"] == 0
        with EventLog(root) as log:
            api = _api(log)
            api.resume(cid)
            frame = api.conversation_stacks[cid][0].conversation_id
            raw = api.fsm_manager.copy_raw_context_data(frame)
            assert (
                api.get_stack_depth(cid) == 1
                and raw["call_result"]["status"] == "running"
            )
            assert _count(log, cid, "stack/push") == 0
            api.advance(cid)  # the entry settles: the push
            assert _count(log, cid, "stack/push") == 1 and api.get_stack_depth(cid) == 2
            assert api.conversation_stacks[cid][1].conversation_id != dead_child
            _finish(api, cid)
            _assert_finished(api, log, cid, "kill before the push")
            # a second resume of the finished conversation asks for nothing more
            api.end_conversation(cid)
        with EventLog(root) as log:
            again = _api(log)
            again.resume(cid)
            assert (
                again.get_stack_depth(cid) == 1 and _parent_state(again, cid) == "done"
            )
            assert (
                _count(log, cid, "stack/push") == 1
                and _count(log, cid, "stack/pop") == 1
            )


class TestAKillInsideTheChild:
    def test_resume_runs_the_child_again_from_its_last_commit_and_pops_it_once(
        self, golden: _Golden, tmp_path: Path
    ) -> None:
        """The process dies while the child's first turn waits for the model (the request is
        logged, the response is not). The open turn contributes its pre-turn state, so the child
        restarts at ``c1`` on the SAME child id; the push is not repeated; the pop appears once and
        the fold has depth 1."""
        (request,) = [
            e["seq"]
            for e in golden.events
            if e["type"] == "llm/request" and e["frame"] == 1
        ]
        root = tmp_path / "log"
        cid = _die_after(root, request)  # the request landed, its response did not
        assert cid is not None
        with EventLog(root) as log:
            before = list(log.read(cid))
            assert before[-1]["type"] == "llm/request" and before[-1]["frame"] == 1
            (pushed,) = [e for e in before if e["type"] == "stack/push"]
            api = _api(log)
            api.resume(cid)
            assert api.get_stack_depth(cid) == 2
            assert (
                api.conversation_stacks[cid][1].conversation_id
                == pushed["child_conversation_id"]
            )
            assert (
                api.fsm_manager.get_conversation_state(pushed["child_conversation_id"])
                == "c1"
            )
            _finish(api, cid)
            _assert_finished(api, log, cid, "kill inside the child")
            after = list(log.read(cid))
            assert [e["type"] for e in after if e["type"] == "stack/pop"] == [
                "stack/pop"
            ]
            assert fold_stack(iter(after), log.get_json).depth == 1  # type: ignore[union-attr]


class TestAForkOfACallRun:
    def test_a_fork_at_the_childs_first_turn_has_depth_two_and_the_child_finishes_in_it(
        self, golden: _Golden, tmp_path: Path
    ) -> None:
        """Turn 3 of the golden run is the child's first turn. The fork has depth 2, the parent
        in the call state with the ``running`` record, the child at ``c2``; one ``advance`` runs
        the child's last step and pops it IN THE FORK; the source tree is byte for byte as it
        was. Kills: a fork that loses the child frame or re-pushes a second child."""
        root = tmp_path / "log"
        shutil.copytree(golden.root, root)
        ends = golden.seqs("turn/end")
        first_child_turn = next(
            n
            for n, seq in enumerate(ends, start=1)
            if next(e for e in golden.events if e["seq"] == seq)["frame"] == 1
        )
        with EventLog(root) as log:
            before = (root / "logs" / f"{golden.cid}.jsonl").read_bytes()
            api = _api(log)
            new_id, state = api.fork(golden.cid, at_turn=first_child_turn)
            assert new_id != golden.cid and state.stack_depth == 2
            assert _view(api, new_id) == golden.at(ends[first_child_turn - 1])
            assert api.conversation_stacks[new_id][1].conversation_id != golden.events[
                golden.seqs("stack/push")[0] - 1
            ].get("child_conversation_id")
            assert api.get_current_state(new_id) == "c2"
            api.advance(new_id)  # the child's last step; settle pops it
            assert api.get_stack_depth(new_id) == 1
            _finish(api, new_id)
            _assert_finished(api, log, new_id, "fork at the child's first turn")
            # the fork's own log is self-sufficient: one `restored` push, one pop
            folded = fold_stack(log.read(new_id), log.get_json)
            assert folded is not None and folded.depth == 1
            assert api.get_data(new_id)["answer"] == "solved why"
            # the source: its log is the bytes it had, and it is still the run it was
            assert (root / "logs" / f"{golden.cid}.jsonl").read_bytes() == before
            assert before == (golden.root / "logs" / f"{golden.cid}.jsonl").read_bytes()
            assert list(log.read(golden.cid)) == golden.events

    def test_a_fork_at_every_turn_goes_on_to_the_one_result(
        self, golden: _Golden, tmp_path: Path
    ) -> None:
        """The fork before any turn (0), with the request and no child (1, the arrival turn:
        its `turn/end` holds the request and rule (b) pushes in the fork), with the child alive
        (2), with the child ended and unpopped (3) and after the success edge (4): every fork
        shows what the live source showed and finishes with one push and one pop in ITS log."""
        root = tmp_path / "log"
        shutil.copytree(golden.root, root)
        ends = golden.seqs("turn/end")
        depths = []
        with EventLog(root) as log:
            api = _api(log)
            for turn in range(len(ends) + 1):
                new_id, state = api.fork(golden.cid, at_turn=turn)
                cut = ends[turn - 1] if turn else golden.seqs("turn/start")[0] - 1
                want = golden.at(cut)
                assert _view(api, new_id) == want and state.stack_depth == want[0], turn  # type: ignore[index]
                depths.append(state.stack_depth)
                _finish(api, new_id)
                after_the_pop = turn == len(
                    ends
                )  # nothing left to push or pop in this log
                one = 0 if after_the_pop else 1
                _assert_finished(
                    api, log, new_id, f"fork at turn {turn}", pushes=one, pops=one
                )
                api.end_conversation(new_id)
        assert depths == [1, 1, 2, 2, 1]  # non-vacuity: depth 2 seen at turns 2 and 3


class TestAPlainRestoreMidCall:
    def test_a_snapshot_taken_while_the_child_runs_restarts_the_call_through_rule_b(
        self, tmp_path: Path
    ) -> None:
        """``save_session`` writes the ROOT frame only: the parent in the call state with the
        ``running`` record, stack depth 2 in the file. ``restore_session`` rebuilds the root, so
        the child is gone and the request is still in the parent's own state: the next entry
        pushes a NEW child (rule (b)) and the restored conversation finishes with one push and
        one pop in its own log. The source API's child is not shared. Kills: a restore that
        trusts ``stack_depth`` of the file, or a settle that needs the child frame to exist."""
        store = FileSessionStore(tmp_path / "sessions")
        with EventLog(tmp_path / "source") as source_log:
            source = _api(source_log, session_store=store)
            cid, _ = source.start_conversation({"topic": "why"})
            source.advance(cid)  # start -> work
            source.advance(cid)  # the request; settle pushes the child
            assert source.get_stack_depth(cid) == 2
            source.save_session(cid)
            saved = store.load(cid)
            assert saved is not None and saved.stack_depth == 2
            assert saved.current_state == "work"
            assert saved.context_data["call_result"]["status"] == "running"
            old_child = source.conversation_stacks[cid][1].conversation_id
            with EventLog(tmp_path / "restored") as log:
                api = _api(log, session_store=store)
                restored = api.restore_session(cid)
                assert restored is not None
                new_id, _ = restored
                assert api.get_stack_depth(new_id) == 1  # the root only
                frame = api.conversation_stacks[new_id][0].conversation_id
                raw = api.fsm_manager.copy_raw_context_data(frame)
                assert raw["call_result"]["status"] == "running"
                api.advance(new_id)  # settle at the start of the entry: the push
                assert api.get_stack_depth(new_id) == 2
                assert api.conversation_stacks[new_id][1].conversation_id != old_child
                _finish(api, new_id)
                _assert_finished(api, log, new_id, "restore mid call")
            # the source is untouched by all of it, and finishes on its own
            assert source.conversation_stacks[cid][1].conversation_id == old_child
            _finish(source, cid)
            _assert_finished(source, source_log, cid, "source after the restore")


def _looping_api(log: EventLog, *, clear: bool) -> API:
    """``done`` goes back to ``work``, so the call state is visited again; with *clear* a handler
    clears the result key on entry (the documented way to ask again)."""
    definition = _parent(None, response_instructions="")
    definition["states"]["done"]["transitions"] = [
        {"target_state": "work", "description": "again"}
    ]
    answer = (
        create_handler("answer")
        .on_state_entry(f"c{_STEPS + 1}")
        .do(lambda ctx: {"solution": f"solved {ctx.get('topic')}"})
    )
    handlers = [answer]
    if clear:
        handlers.append(clear_keys_on_entry(["call_result"], state="work"))
    return API.from_definition(
        definition,
        llm_interface=ScriptedLLM(default_response="r"),
        targets=_Resolver({_TARGET: _child_definition(_STEPS, None, "c-speaks")}),
        handlers=handlers,
        event_log=log,
    )


def _raw_record(api: API, cid: str) -> Any:
    frame = api.conversation_stacks[cid][0].conversation_id
    return api.fsm_manager.copy_raw_context_data(frame).get("call_result")


class TestAStaleResult:
    def test_a_final_result_left_in_the_context_pushes_nothing_on_a_second_visit(
        self, tmp_path: Path
    ) -> None:
        """Skip-if-set: a second visit to the call state finds the old FINAL record, asks for
        nothing and the success edge fires again (the documented limit: authors clear the key on
        entry). The log keeps ONE push and ONE pop, live and after a resume. Kills: a settle that
        treats any record as a request, or an executor that overwrites a final one."""
        root = tmp_path / "log"
        with EventLog(root) as log:
            api = _looping_api(log, clear=False)
            cid, _ = api.start_conversation({"topic": "why"})
            states = [api.advance(cid).state_after for _ in range(10)]
            assert states.count("done") >= 3 and "work" in states[6:]  # visited again
            assert _raw_record(api, cid) == _final("ok", None)
            assert (
                _count(log, cid, "stack/push") == 1
                and _count(log, cid, "stack/pop") == 1
            )
        with EventLog(root) as log:
            again = _looping_api(log, clear=False)
            again.resume(cid)
            assert _raw_record(again, cid) == _final("ok", None)
            for _ in range(4):
                again.advance(cid)
            assert again.get_stack_depth(cid) == 1
            assert (
                _count(log, cid, "stack/push") == 1
                and _count(log, cid, "stack/pop") == 1
            )

    def test_a_cleared_key_asks_again_and_a_death_before_the_second_push_is_asked_once(
        self, tmp_path: Path
    ) -> None:
        """With the key cleared on entry the second visit is a second call: push, pop, depth-indexed
        in the same log. The process dies in ``record_push`` of the SECOND push; the resumed run
        pushes the second child once. Two pushes, two pops, fold depth 1."""
        from fsm_llm.engine.api import events as api_events

        root = tmp_path / "log"
        real = api_events.record_push
        pushes: list[int] = []

        def second_dies(*args: Any, **kwargs: Any) -> None:
            pushes.append(1)
            if len(pushes) == 2:
                raise _Dead
            real(*args, **kwargs)

        with EventLog(root) as log:
            api = _looping_api(log, clear=True)
            cid, _ = api.start_conversation({"topic": "why"})
            with (
                mock.patch(
                    "fsm_llm.engine.api.stack.record_push", side_effect=second_dies
                ),
                pytest.raises(_Dead),
            ):
                for _ in range(12):
                    api.advance(cid)
            assert (
                _count(log, cid, "stack/push") == 1
                and _count(log, cid, "stack/pop") == 1
            )
        with EventLog(root) as log:
            api = _looping_api(log, clear=True)
            api.resume(cid)
            assert api.get_stack_depth(cid) == 1
            assert _raw_record(api, cid)["status"] == "running"  # the second request
            for _ in range(8):
                api.advance(cid)
                if _count(log, cid, "stack/pop") == 2 and api.get_stack_depth(cid) == 1:
                    break
            assert (
                _count(log, cid, "stack/push") == 2
                and _count(log, cid, "stack/pop") == 2
            )
            assert _raw_record(api, cid) == _final("ok", None)
            folded = fold_stack(log.read(cid), log.get_json)
            assert folded is not None and folded.depth == 1
            assert list(api.fsm_manager.instances) == [
                api.conversation_stacks[cid][0].conversation_id
            ]


class TestARunLoopSettlesAnEndedCallChild:
    """An ended call child that is still on the stack is the engine's to pop (D-024): a run loop
    that stops at once on it, as it does for a frame a driver pops, leaves a call that never
    finishes while the conversation reads as ended."""

    def test_the_resumed_run_loop_pops_a_child_that_ended_before_its_pop_was_logged(
        self, golden: _Golden, tmp_path: Path
    ) -> None:
        """The offset after the child's last ``turn/end`` and before ``stack/pop``. The fold has
        depth 2 and the child at its terminal state. ONE ``run_until_terminal`` pops it with one
        merge, runs the parent's success edge and ends on ``done``. Kills: a loop that returns at
        once because the top of the stack has ended (zero steps, depth 2, no pop)."""
        (pop,) = golden.seqs("stack/pop")
        root = tmp_path / "log"
        cid = _die_after(root, pop - 1)
        assert cid is not None
        with EventLog(root) as log:
            api = _api(log)
            api.resume(cid)
            assert api.get_stack_depth(cid) == 2 and api.has_conversation_ended(cid)
            results = api.run_until_terminal(cid, max_steps=10)
            assert [r.state_after for r in results] == ["done"]
            _assert_finished(api, log, cid, "the loop popped the ended child")

    async def test_the_live_run_loop_pops_a_child_whose_end_was_refused_after_the_merge(
        self,
    ) -> None:
        """The merge landed and ending the child was refused: the entry returned its step (the
        refusal is a warning, D-026) and the record is final with the child still on the stack and ended. Another ``arun_until_terminal`` pops
        it (no second merge) and finishes the run."""
        flow = await _flow(steps=1)
        await flow.api.aadvance(flow.cid)  # the request, the push
        child = flow.child_frame()
        manager = flow.api.fsm_manager
        real_end = manager.end_conversation
        refusals: list[str] = []

        def refuse_once(conversation_id: str, *args: Any, **kwargs: Any) -> None:
            if conversation_id == child and not refusals:
                refusals.append(conversation_id)
                raise ConversationBusyError("refused", conversation_id=conversation_id)
            return real_end(conversation_id, *args, **kwargs)

        manager.end_conversation = refuse_once  # type: ignore[method-assign]
        # the child's last step; its pop is refused and the committed step is not lost (D-026)
        step = await flow.api.aadvance(flow.cid)
        assert step.state_after == "c2"
        assert flow.depth() == 2 and flow.record() == _final("ok", None)
        results = await flow.api.arun_until_terminal(flow.cid, max_steps=10)
        assert [r.state_after for r in results] == ["done"]
        assert flow.depth() == 1 and flow.record() == _final("ok", None)
        assert flow.api.get_data(flow.cid)["answer"] == "solved why"
        assert flow.api._call_marks.ending == set()
