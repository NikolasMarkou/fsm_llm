"""The seams of fork and resume (plan 5a6a1f80 step 3, roadmap row 1.4, part 1): the id-preserving
``FSMManager.start_conversation(conversation_id=)`` (D-008) and the ONE restore body
(``SessionsMixin._restore_conversation``) that ``restore_session`` shares with ``fork`` and ``resume``.

Part 2 (step 5) is ``API.resume`` itself, at the end of this file: the same id continues the same log
with the writer lock taken before any read, the stack comes back with its own ids, an in-flight tool is
reported once and never run, a killed turn stays open, and every failure after the lock frees it.
Part 1: the id is kept, a live or malformed id is refused with
nothing registered, the default path still mints one uuid, the lock sequence is the same with and
without an id, and the helper tears the half-built conversation down at every stage. Counts and
sequences, never timing.
"""

from __future__ import annotations

import copy
import threading
import uuid
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm import API, FileSessionStore
from fsm_llm.core.definitions import FSMDefinition, FSMError
from fsm_llm.engine.api import forks
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.handlers import HandlerTiming
from fsm_llm.runtime import EventLog, EventType, LogCorruptError, LogLockedError
from fsm_llm.runtime.stores import fold_stack, jsonl
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm import event_log_child as child
from tests.test_fsm_llm.event_log_support import _Dead
from tests.test_fsm_llm.test_completion_state import (
    _MESSAGES,
    _structured_fsm,
    _tool_fsm,
)
from tests.test_fsm_llm.test_event_log_forks import (
    _DROP,
    _GRANT,
    _KIDS,
    _chain,
    _forge,
    _kid_id,
    _nothing_left,
    _raw,
    _shown,
    _stacked,
)
from tests.test_fsm_llm.test_event_log_forks import (
    _api as _logged_api,
)
from tests.test_fsm_llm.test_event_log_stack import _blob_path, _folded_view, _of, _view
from tests.test_fsm_llm.test_fsm_lock_order import (
    _FSM,
    _RecLock,
    _short,
    order_problems,
)


def _llm() -> ScriptedLLM:
    return ScriptedLLM(responses={"chat": "ok", "done": "bye"})


def _manager() -> FSMManager:
    return FSMManager(
        fsm_loader=lambda fsm_id: FSMDefinition(**_FSM), llm_interface=_llm()
    )


def _api(tmp_path=None, **kw) -> API:
    store = FileSessionStore(tmp_path) if tmp_path is not None else None
    return API(
        fsm_definition=FSMDefinition(**_FSM),
        llm_interface=_llm(),
        session_store=store,
        **kw,
    )


class TestManagerKeepsAnId:
    @pytest.mark.parametrize("suppress", [True, False])
    def test_the_given_id_is_kept(self, suppress: bool):
        manager = _manager()
        conversation_id, _ = manager.start_conversation(
            "fsm", suppress_start=suppress, conversation_id="kept-1"
        )
        assert conversation_id == "kept-1"
        assert set(manager.instances) == {"kept-1"} == set(manager._conversation_locks)
        assert manager.copy_raw_context_data("kept-1")["_conversation_id"] == "kept-1"

    def test_a_live_id_is_refused_and_the_first_conversation_is_untouched(self):
        manager = _manager()
        manager.start_conversation("fsm", suppress_start=True, conversation_id="a-1")
        instance, lock = manager.instances["a-1"], manager._conversation_locks["a-1"]
        with pytest.raises(FSMError, match="already live"):
            manager.start_conversation(
                "fsm", suppress_start=True, conversation_id="a-1"
            )
        assert manager.instances["a-1"] is instance
        assert manager._conversation_locks["a-1"] is lock
        manager.end_conversation("a-1")  # an ended id is free again
        manager.start_conversation("fsm", suppress_start=True, conversation_id="a-1")

    @pytest.mark.parametrize(
        "bad", ["", "A-1", "../x", "a/b", "x" * 65, "con", "nul.txt", "a b"]
    )
    def test_a_malformed_id_is_refused_with_nothing_registered(self, bad: str):
        manager = _manager()
        with pytest.raises(FSMError):
            manager.start_conversation("fsm", suppress_start=True, conversation_id=bad)
        assert not manager.instances and not manager._conversation_locks

    def test_the_default_path_mints_one_uuid_and_checks_no_id(self):
        manager = _manager()
        fixed = uuid.UUID(int=7)
        with (
            patch("fsm_llm.engine.fsm.manager.uuid.uuid4", return_value=fixed) as mint,
            patch("fsm_llm.engine.fsm.manager.check_conversation_id") as check,
        ):
            conversation_id, _ = manager.start_conversation("fsm", suppress_start=True)
        assert conversation_id == str(fixed)
        assert mint.call_count == 1 and check.call_count == 0

    def test_two_threads_with_one_id_cannot_both_start(self):
        manager = _manager()
        barrier = threading.Barrier(8)
        outcomes: list[object] = []

        def start() -> None:
            barrier.wait()
            try:
                manager.start_conversation(
                    "fsm", suppress_start=True, conversation_id="race-1"
                )
                outcomes.append("ok")
            except FSMError as e:
                outcomes.append(e)

        threads = [threading.Thread(target=start) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert outcomes.count("ok") == 1 and len(outcomes) == 8
        assert list(manager.instances) == ["race-1"]


class TestApiPassThrough:
    def test_a_private_id_is_kept_and_keys_the_stack(self):
        api = _api()
        conversation_id, _ = api.start_conversation(
            _suppress_start=True, _conversation_id="kept-2"
        )
        assert conversation_id == "kept-2"
        assert list(api.conversation_stacks) == ["kept-2"]
        assert api.get_current_state("kept-2") == "chat"

    def test_a_refused_id_leaves_no_conversation_behind(self):
        api = _api()
        with pytest.raises(FSMError):
            api.start_conversation(_suppress_start=True, _conversation_id="BAD id")
        assert api.list_active_conversations() == []
        assert not api.conversation_stacks and not api.fsm_manager.instances

    def test_the_default_call_to_the_manager_names_no_id(self):
        api = _api()
        with patch.object(
            api.fsm_manager,
            "start_conversation",
            wraps=api.fsm_manager.start_conversation,
        ) as spy:
            api.start_conversation()
        assert spy.call_args.kwargs == {
            "initial_context": None,
            "suppress_start": False,
        }


class TestLockSequenceIsUnchanged:
    @pytest.mark.parametrize("suppress", [True, False])
    def test_an_explicit_id_takes_the_same_locks_as_a_minted_one(self, suppress: bool):
        sequences = []
        for given in (None, "kept-3"):
            manager = _manager()
            events: list[tuple[str, str, int]] = []
            manager._lock = _RecLock("lock", manager._lock, events)  # type: ignore[assignment]
            manager.start_conversation(
                "fsm", suppress_start=suppress, conversation_id=given
            )
            assert order_problems(events) == []
            sequences.append(_short(events))
        assert sequences[0] == sequences[1] and sequences[0]
        if suppress:
            assert sequences[0] == ["lock+", "lock-"]


def _saved(api: API) -> tuple[str, object]:
    conversation_id, _ = api.start_conversation()
    api.converse("hello", conversation_id)
    api.save_session(conversation_id)
    return conversation_id, api.load_session(conversation_id)


class TestOneRestoreBody:
    def test_restore_session_goes_through_the_helper_once(self, tmp_path):
        api = _api(tmp_path)
        source, _ = _saved(api)
        with patch.object(API, "_restore_conversation", autospec=True) as helper:
            helper.return_value = "stub-id"
            new_id, state = api.restore_session(source)
        assert helper.call_count == 1 and new_id == "stub-id"
        assert helper.call_args.kwargs == {}  # no id kept for a snapshot restore
        assert state.conversation_id == source

    def test_the_helper_keeps_an_id_and_seeds_the_state(self, tmp_path):
        api = _api(tmp_path)
        source, state = _saved(api)
        seen: list[str] = []
        kept = api._restore_conversation(state, seen.append, conversation_id="kept-4")
        assert kept == "kept-4" and seen == ["kept-4"]
        assert api.get_current_state("kept-4") == state.current_state
        assert api.get_conversation_history("kept-4") == state.conversation_history
        assert api.get_data("kept-4") == api.get_data(source)

    @pytest.mark.parametrize("stage", ["seed", "set_state", "record"])
    @pytest.mark.parametrize("kept", [None, "kept-5"])
    def test_a_failure_at_every_stage_tears_the_conversation_down(
        self, tmp_path, stage: str, kept: str | None
    ):
        api = _api(tmp_path)
        source, state = _saved(api)
        ended: list[int] = []
        api.register_handler(
            api.create_handler("end-watch")
            .at(HandlerTiming.END_CONVERSATION)
            .do(lambda ctx: ended.append(1) or {})
        )
        boom = RuntimeError(f"{stage} boom")

        def record(conversation_id: str) -> None:
            if stage == "record":
                raise boom

        names = {
            "seed": "seed_restored_conversation",
            "set_state": "set_conversation_state",
        }
        injected = (
            patch.object(api.fsm_manager, names[stage], side_effect=boom)
            if stage in names
            else nullcontext()
        )
        with injected, pytest.raises(RuntimeError, match=f"{stage} boom"):
            api._restore_conversation(state, record, conversation_id=kept)
        assert api.list_active_conversations() == [source]
        assert set(api.fsm_manager.instances) == {source}
        assert list(api.conversation_stacks) == [source]
        assert len(ended) == 1  # END handlers fire on the teardown, on purpose
        if kept:  # the kept id is free again
            api.start_conversation(_suppress_start=True, _conversation_id=kept)

    def test_a_failing_teardown_never_masks_the_original_error(self, tmp_path):
        api = _api(tmp_path)
        _, state = _saved(api)

        def refuse_end(conversation_id: str) -> None:
            raise FSMError("teardown refused")

        with patch.object(api, "end_conversation", refuse_end):
            with pytest.raises(RuntimeError, match="record boom"):
                api._restore_conversation(
                    state,
                    lambda cid: (_ for _ in ()).throw(RuntimeError("record boom")),
                )

    def test_a_refused_start_raises_before_any_teardown(self, tmp_path):
        api = _api(tmp_path)
        source, state = _saved(api)
        with pytest.raises(FSMError):
            api._restore_conversation(state, lambda cid: None, conversation_id="BAD id")
        assert api.list_active_conversations() == [source]


# ==============================================================
# Part 2: API.resume (plan 5a6a1f80 step 5)
# ==============================================================

_PUSH = EventType.STACK_PUSH


@pytest.fixture
def after_crash(log: EventLog) -> Iterator[Any]:
    """``after_crash()`` is the process dying: every writer closed (the locks are free, the files
    stay) and a NEW ``EventLog`` on the same directory, which the fixture closes."""
    opened: list[EventLog] = []

    def crash() -> EventLog:
        log.close()
        opened.append(EventLog(log.root))
        return opened[-1]

    yield crash
    for fresh in opened:
        fresh.close()


def _lock_is_free(root: Any, cid: str) -> bool:
    """True when a brand new ``EventLog`` takes the writer of *cid* at once (and gives it back)."""
    with EventLog(root) as fresh:
        try:
            fresh.open_existing(cid)
        except LogLockedError:
            return False
        return True


def _tool_api(log: EventLog, llm: ScriptedLLM | None = None) -> API:
    llm = llm or child.ChildLLM(child.scripted_replies())
    return API.from_definition(child.tool_fsm(), llm_interface=llm, event_log=log)


class _DieAfterTurnStart:
    """``jsonl._write`` stand-in: the ``turn/start`` line lands, the next write dies."""

    def __init__(self) -> None:
        self.seen = self.dead = False
        self._real = jsonl._write

    def __call__(self, fd: int, data: Any) -> int:
        if self.dead:
            return len(data)
        if self.seen:
            self.dead = True
            raise _Dead
        self.seen = b'"turn/start"' in bytes(data)
        return self._real(fd, data)


class TestResumeContinuesTheSameLog:
    def test_the_same_id_the_next_seq_and_one_resume_record(
        self, log: EventLog, after_crash: Any
    ) -> None:
        api = _logged_api(log, _chain())
        cid, _ = api.start_conversation()
        for text in ("Ann", "b"):
            api.converse(text, cid)
        shown, path = _shown(api, cid), log.root / "logs" / f"{cid}.jsonl"
        raw, count = path.read_bytes(), len(list(log.read(cid)))
        again = after_crash()
        api2 = _logged_api(again, _chain())
        new_id, state = api2.resume(cid)
        assert new_id == cid == state.conversation_id
        assert _shown(api2, cid) == shown and again.conversation_ids() == [cid]
        assert path.read_bytes().startswith(raw)  # the same file, appended to
        (record,) = [e for e in again.read(cid) if e["seq"] > count]
        assert (record["seq"], record["type"], record["frame"]) == (
            count + 1,
            "context/record",
            0,
        )
        assert record["source"] == "resume" and record["resumed_from"] == {"seq": count}
        api2.converse("c", cid)
        seqs = [e["seq"] for e in again.read(cid)]  # a restart or a gap would raise
        assert seqs == list(range(1, len(seqs) + 1))
        assert api2.get_current_state(cid) == "w3"

    def test_the_whole_stack_returns_with_its_own_ids_and_pops_as_the_source_would(
        self, log: EventLog, after_crash: Any
    ) -> None:
        api = _logged_api(log)
        cid = _stacked(
            api,
            depth=3,
            return_context={"answer": 42},
            shared_context_keys=["name"],
            preserve_history=True,
        )
        ids = [f.conversation_id for f in api.conversation_stacks[cid]]
        options = [
            (f.fsm_id, f.shared_context_keys, f.preserve_history, f.return_context)
            for f in api.conversation_stacks[cid]
        ]
        observed = _view(api, cid)
        again = after_crash()
        api2 = _logged_api(again)
        _, state = api2.resume(cid)
        stack = api2.conversation_stacks[cid]
        assert state.stack_depth == 3 and [f.conversation_id for f in stack] == ids
        assert [
            (f.fsm_id, f.shared_context_keys, f.preserve_history, f.return_context)
            for f in stack
        ] == options
        assert _view(api2, cid) == observed
        api2.converse("more", cid)
        api2.pop_fsm(cid, context_to_return={"r": 1})
        assert api2.get_stack_depth(cid) == 2
        folded = fold_stack(again.read(cid), again.get_json)
        assert folded is not None and _folded_view(folded) == _view(api2, cid)

    def test_a_torn_tail_is_cut_by_the_writer_and_the_seq_continues(
        self, log: EventLog, after_crash: Any
    ) -> None:
        api = _logged_api(log, _chain())
        cid, _ = api.start_conversation()
        api.converse("Ann", cid)
        shown, path = _shown(api, cid), log.root / "logs" / f"{cid}.jsonl"
        raw, count = path.read_bytes(), len(list(log.read(cid)))
        again = after_crash()
        path.write_bytes(raw + b'{"v":1,"seq":99,"ty')  # the dying write, half a line
        api2 = _logged_api(again, _chain())
        api2.resume(cid)
        assert _shown(api2, cid) == shown
        assert (
            path.read_bytes().startswith(raw) and b'"seq":99' not in path.read_bytes()
        )
        assert [e["seq"] for e in again.read(cid)][-1] == count + 1

    def test_a_fork_resumes_on_its_own_log(
        self, log: EventLog, after_crash: Any
    ) -> None:
        api = _logged_api(log)
        cid = _stacked(api)
        forked, _ = api.fork(cid, at_turn=2)
        observed = _view(api, forked)
        again = after_crash()
        api2 = _logged_api(again)
        api2.resume(forked)
        assert _view(api2, forked) == observed
        assert _kid_id(api2, forked) == _kid_id(api, forked)

    def test_a_terminal_conversation_resumes_as_ended(
        self, log: EventLog, after_crash: Any
    ) -> None:
        api = _logged_api(log, _chain())
        cid, _ = api.start_conversation()
        for text in ("Ann", "b", "c", "d", "e"):
            api.converse(text, cid)
        assert api.has_conversation_ended(cid)
        api2 = _logged_api(after_crash(), _chain())
        api2.resume(cid)
        assert api2.get_current_state(cid) == "w5" and api2.has_conversation_ended(cid)

    def test_an_ended_conversation_resumes_and_its_cache_entry_goes(
        self, log: EventLog
    ) -> None:
        api = _logged_api(log, _chain())
        cid, _ = api.start_conversation()
        for text in ("Ann", "b"):
            api.converse(text, cid)
        api.end_conversation(cid)
        assert cid in api._ended_conversations and not log.has_writer(cid)
        api.resume(cid)
        assert cid not in api._ended_conversations
        assert api.get_current_state(cid) == "w2" and not api.has_conversation_ended(
            cid
        )
        api.converse("c", cid)
        assert api.get_current_state(cid) == "w3"


class TestTheLockDecidesOwnership:
    def test_the_lock_is_taken_before_anything_is_read(
        self, log: EventLog, after_crash: Any
    ) -> None:
        api = _logged_api(log)
        cid = _stacked(api)
        again = after_crash()
        api2, probed = _logged_api(again), []
        real_fold = forks.fold_stack

        def mid_fold(events: Any, load_blob: Any) -> Any:
            with EventLog(
                again.root
            ) as other:  # a second resumer, mid fold of the first
                with pytest.raises(LogLockedError):
                    other.open_existing(cid)
                with pytest.raises(LogLockedError):
                    _logged_api(other).resume(cid)
            probed.append(1)
            return real_fold(events, load_blob)

        order: list[str] = []
        real_open, real_read = EventLog.open_existing, EventLog.read

        def spy_open(self: EventLog, *a: Any, **k: Any) -> int:
            order.append("lock")
            return real_open(self, *a, **k)

        def spy_read(self: EventLog, *a: Any, **k: Any) -> Any:
            order.append("read")
            return real_read(self, *a, **k)

        with (
            patch.object(forks, "fold_stack", mid_fold),
            patch.object(EventLog, "open_existing", spy_open),
            patch.object(EventLog, "read", spy_read),
        ):
            api2.resume(cid)
        assert probed == [1] and order[0] == "lock" and "read" in order

    def test_a_refused_resumer_never_frees_the_writer_of_the_one_that_won(
        self, log: EventLog
    ) -> None:
        api = _logged_api(log, _chain())
        cid, _ = api.start_conversation()
        api.converse("Ann", cid)
        api.end_conversation(cid)
        winner, loser = _logged_api(log, _chain()), _logged_api(log, _chain())
        winner.resume(cid)
        with pytest.raises(LogLockedError):  # two APIs, one EventLog, one writer
            loser.resume(cid)
        with pytest.raises(FSMError, match="live"):
            winner.resume(cid)
        assert loser.list_active_conversations() == [] and log.has_writer(cid)
        winner.converse("b", cid)  # the winner still appends
        winner.end_conversation(cid)
        assert not log.has_writer(cid)
        loser.resume(cid)
        assert loser.get_current_state(cid) == "w2"

    @pytest.mark.parametrize("shared", [True, False], ids=["one-log", "log-each"])
    def test_exactly_one_of_many_concurrent_resumers_wins(
        self, log: EventLog, after_crash: Any, shared: bool
    ) -> None:
        api = _logged_api(log, _chain())
        cid, _ = api.start_conversation()
        api.converse("Ann", cid)
        first = after_crash()
        logs = [first] + [EventLog(first.root) for _ in range(5)]
        apis = [_logged_api(first if shared else one, _chain()) for one in logs]
        barrier, outcomes = threading.Barrier(len(apis)), []

        def go(one: API) -> None:
            barrier.wait(30)
            try:
                one.resume(cid)
                outcomes.append("won")
            except LogLockedError:
                outcomes.append("locked")

        threads = [threading.Thread(target=go, args=(one,)) for one in apis]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        for extra in logs[1:]:
            extra.close()
        assert sorted(outcomes) == ["locked"] * 5 + ["won"]
        assert sum(bool(one.list_active_conversations()) for one in apis) == 1

    def test_resume_takes_the_manager_locks_in_the_canonical_order(
        self, log: EventLog, after_crash: Any
    ) -> None:
        api = _logged_api(log)
        cid = _stacked(api)
        api2 = _logged_api(after_crash())
        manager, events = api2.fsm_manager, []
        manager._lock = _RecLock("lock", manager._lock, events)  # type: ignore[assignment]
        manager._fsm_cache_lock = _RecLock("cache", manager._fsm_cache_lock, events)  # type: ignore[assignment]

        class _Wrapped(dict):  # type: ignore[type-arg]
            def __setitem__(self, key: Any, value: Any) -> None:
                super().__setitem__(key, _RecLock("conv", value, events))

        manager._conversation_locks = _Wrapped()  # type: ignore[assignment]
        api2.resume(cid)
        assert order_problems(events) == []
        assert {name for name, _, _ in events} >= {"lock", "conv"}


class TestRefusals:
    def test_a_log_off_api_refuses(self) -> None:
        api = API(copy.deepcopy(_KIDS), llm_interface=ScriptedLLM([]))
        with pytest.raises(FSMError, match="needs an event log"):
            api.resume("any-id")

    def test_an_unknown_id_creates_no_file_no_lock_and_no_directory(
        self, tmp_path: Any
    ) -> None:
        root = tmp_path / "events"
        with EventLog(root) as log:
            api = _logged_api(log)
            with pytest.raises(FSMError, match="no event log"):
                api.resume("nope")
            with pytest.raises(FileNotFoundError):
                log.open_existing("nope")
            assert not root.exists() and not log._writers
            other, _ = api.start_conversation()  # now the log directory exists
            before = sorted(p.name for p in root.rglob("*"))
            for bad in ("nope", "../x", "UPPER"):
                with pytest.raises(FSMError):
                    api.resume(bad)
            assert sorted(p.name for p in root.rglob("*")) == before
            assert api.list_active_conversations() == [other]
            assert set(log._writers) <= {other}

    def test_an_empty_log_has_no_state_and_the_lock_is_given_back(
        self, log: EventLog
    ) -> None:
        log.append("ghost", EventType.TURN_START, frame=0)
        log.release("ghost")
        api = _logged_api(log)
        with pytest.raises(FSMError, match="no recorded state"):
            api.resume("ghost")
        assert not log.has_writer("ghost") and api.list_active_conversations() == []

    def test_open_existing_unit(self, log: EventLog) -> None:
        log.append("one", EventType.CONTEXT_RECORD, source="x")
        assert log.has_writer("one")
        with pytest.raises(LogLockedError, match="already has a live writer"):
            log.open_existing("one")
        log.release("one")
        assert not log.has_writer("one") and log.open_existing("one") == 1
        log.close()
        with pytest.raises(Exception, match="closed"):
            log.open_existing("one")

    def test_a_damaged_log_is_corrupt_and_not_locked(self, log: EventLog) -> None:
        log.append("bad", EventType.CONTEXT_RECORD, source="x")
        log.release("bad")
        (log.root / "logs" / "bad.jsonl").write_bytes(b'{"not":"an event"}\n')
        for _ in range(
            2
        ):  # the second try would be LogLockedError if the lock had leaked
            with pytest.raises(LogCorruptError):
                log.open_existing("bad")
        assert not log.has_writer("bad")


class TestAnInFlightToolIsReportedOnce:
    @staticmethod
    def _crashed_in_a_tool(log: EventLog) -> str:
        api = _tool_api(log)
        cid = child.start(api)
        child.drive_exchange(api, log, cid)
        with pytest.raises(_Dead):
            with log.tool_call(cid, name="charge", call_id="c-1", arguments={"n": 5}):
                raise _Dead
        return cid

    def test_unknown_once_resolved_in_the_log_and_never_run(
        self, log: EventLog, after_crash: Any
    ) -> None:
        cid = self._crashed_in_a_tool(log)
        again = after_crash()
        llm = ScriptedLLM([])
        api2 = _tool_api(again, llm)
        calls = [e for e in again.read(cid) if e["type"] == "tool/call"]
        assert [c["call_id"] for c in again.unknown_tool_calls(cid)] == ["c-1"]
        _, state = api2.resume(cid)
        (unknown,) = state.metadata["unknown_tool_calls"]
        assert (unknown["call_id"], unknown["status"]) == ("c-1", "unknown")
        events = list(again.read(cid))
        assert [
            e for e in events if e["type"] == "tool/call"
        ] == calls  # nothing re-run
        assert llm.call_history == [] and again.unknown_tool_calls(cid) == []
        (result,) = [
            e for e in events if e["type"] == "tool/result" and e["call_id"] == "c-1"
        ]
        assert (result["status"], result["frame"], result["call_frame"]) == (
            "unknown",
            0,
            0,
        )
        api2.end_conversation(cid)
        _, second = api2.resume(cid)
        assert "unknown_tool_calls" not in second.metadata
        results = [e for e in again.read(cid) if e["type"] == "tool/result"]
        assert sum(e["call_id"] == "c-1" for e in results) == 1

    def test_a_resume_that_fails_leaves_the_call_to_be_reported_again(
        self, log: EventLog, after_crash: Any
    ) -> None:
        cid = self._crashed_in_a_tool(log)
        again = after_crash()
        api2 = _tool_api(again, ScriptedLLM([]))
        with (
            patch.object(forks, "record_resumed", side_effect=FSMError("injected")),
            pytest.raises(FSMError, match="injected"),
        ):
            api2.resume(cid)
        assert [c["call_id"] for c in again.unknown_tool_calls(cid)] == ["c-1"]
        _, state = api2.resume(cid)
        assert [c["call_id"] for c in state.metadata["unknown_tool_calls"]] == ["c-1"]

    def test_a_tool_of_a_lower_frame_is_resolved_without_popping_the_frames_above(
        self, log: EventLog, after_crash: Any
    ) -> None:
        api = _logged_api(log)
        cid, _ = api.start_conversation()
        api.converse("Ann", cid)
        # a runner tool is in flight at frame 0 while it pushes and drives a child
        log.append(
            cid,
            EventType.TOOL_CALL,
            frame=0,
            name="runner",
            call_id="r-1",
            side_effecting=True,
            arguments={},
        )
        api.push_fsm(cid, copy.deepcopy(_KIDS))
        api.converse("hi", cid)
        again = after_crash()
        api2 = _logged_api(again)
        _, state = api2.resume(cid)
        assert [c["call_id"] for c in state.metadata["unknown_tool_calls"]] == ["r-1"]
        assert api2.get_stack_depth(cid) == 2
        (result,) = [e for e in again.read(cid) if e["type"] == "tool/result"]
        assert (result["frame"], result["call_frame"]) == (1, 0)
        folded = fold_stack(again.read(cid), again.get_json)
        assert folded is not None and folded.depth == 2
        assert again.unknown_tool_calls(cid) == []


class TestAKilledTurnStaysOpen:
    def test_a_turn_killed_at_frame_one_resumes_at_the_pre_turn_state(
        self, log: EventLog, after_crash: Any
    ) -> None:
        api = _logged_api(log)
        cid = _stacked(api)
        observed = _view(api, cid)
        dying = _DieAfterTurnStart()
        with patch.object(jsonl, "_write", dying), pytest.raises(_Dead):
            api.converse("boom", cid)
        again = after_crash()
        events = list(again.read(cid))
        assert (events[-1]["type"], events[-1]["frame"]) == ("turn/start", 1)
        ends = sum(e["type"] == "turn/end" for e in events)
        api2 = _logged_api(again)
        api2.resume(cid)
        assert _view(api2, cid) == observed  # the state before the killed turn
        written = list(again.read(cid))[len(events) :]
        assert [(e["type"], e["frame"], e["source"]) for e in written] == [
            ("context/record", 1, "resume")
        ]  # at the TOP frame, so the fold does not read it as a pop
        assert sum(e["type"] == "turn/end" for e in again.read(cid)) == ends
        folded = fold_stack(again.read(cid), again.get_json)
        assert folded is not None and folded.depth == 2
        api2.converse("again", cid)  # the next turn supersedes the open one
        folded = fold_stack(again.read(cid), again.get_json)
        assert folded is not None and _folded_view(folded) == _view(api2, cid)
        assert api2.get_stack_depth(cid) == 2


class TestAFailureAfterTheLockFreesIt:
    @staticmethod
    def _nth(real: Any, n: int, error: Exception) -> Any:
        calls: list[int] = []

        def flaky(*args: Any, **kw: Any) -> Any:
            calls.append(1)
            if len(calls) == n:
                raise error
            return real(*args, **kw)

        return flaky

    @pytest.mark.parametrize(
        ("target", "n"),
        [
            ("fsm_llm.engine.api.forks.fold_stack", 1),
            ("fsm_llm.engine.api.forks._prepare", 1),
            ("fsm_llm.engine.fsm.manager.FSMManager.seed_restored_conversation", 1),
            ("fsm_llm.engine.fsm.manager.FSMManager.set_conversation_state", 1),
            ("fsm_llm.engine.fsm.manager.FSMManager.set_conversation_state", 3),
            ("fsm_llm.engine.fsm.manager.FSMManager.start_conversation", 3),
            ("fsm_llm.engine.api.forks.record_resumed", 1),
        ],
        ids=[
            "fold",
            "prepare",
            "seed",
            "root",
            "frame-two",
            "start-frame-two",
            "record",
        ],
    )
    def test_the_writer_is_released_and_everything_built_is_ended(
        self, log: EventLog, after_crash: Any, target: str, n: int
    ) -> None:
        api = _logged_api(log)
        cid = _stacked(api, depth=3)
        again = after_crash()
        before = list(again.read(cid))
        api2 = _logged_api(again)
        owner, _, name = target.rpartition(".")
        real = getattr(FSMManager if owner.endswith("FSMManager") else forks, name)
        error = (
            LogCorruptError("injected")
            if name == "fold_stack"
            else FSMError("injected")
        )
        with (
            patch(target, self._nth(real, n, error)),
            pytest.raises(type(error), match="injected"),
        ):
            api2.resume(cid)
        _nothing_left(api2, again)
        assert not again.has_writer(cid) and _lock_is_free(again.root, cid)
        assert list(again.read(cid)) == before  # a failed resume wrote nothing
        _, state = api2.resume(cid)  # and the same resume works once the fault is gone
        assert state.stack_depth == 3

    def test_a_definition_the_log_cannot_give_back_frees_the_lock(
        self, log: EventLog, after_crash: Any
    ) -> None:
        api = _logged_api(log)
        cid = _stacked(api)
        (push,) = _of(log, cid, _PUSH)
        path = _blob_path(log, push["definition_ref"])
        again = after_crash()
        path.write_bytes(path.read_bytes().replace(b"Kid hello", b"KID HELLO"))
        api2 = _logged_api(again)
        with pytest.raises(LogCorruptError):
            api2.resume(cid)
        _nothing_left(api2, again)
        assert not again.has_writer(cid) and _lock_is_free(again.root, cid)

    @pytest.mark.parametrize("owned", [False, True], ids=["shared-log", "owned-log"])
    def test_closing_the_api_after_a_resume_frees_the_lock(
        self, log: EventLog, after_crash: Any, owned: bool
    ) -> None:
        api = _logged_api(log)
        cid = _stacked(api)
        again = after_crash()
        if owned:
            again.close()
            api2 = _logged_api(log.root)  # type: ignore[arg-type]
        else:
            api2 = _logged_api(again)
        api2.resume(cid)
        assert not _lock_is_free(log.root, cid)  # held while the conversation lives
        api2.close()
        assert _lock_is_free(log.root, cid)


class TestAFailedRestoreLeavesTheEndedCacheAsItWas:
    """A2 (engine audit 2.3): the teardown of a half-built restore ends the conversation, and
    ending caches it. A failed ``restore_session``, ``fork`` or ``resume`` must not leave an
    "ended" entry that answers for an id that was never restored, and must not overwrite an
    entry the id legitimately had."""

    @staticmethod
    def _fail(target: str) -> Any:
        return patch(target, side_effect=FSMError("injected"))

    def test_a_failed_resume_does_not_make_the_id_look_ended(
        self, log: EventLog
    ) -> None:
        first = _logged_api(log)
        cid = _stacked(first)
        first.end_conversation(
            cid
        )  # frees the writer; the id is NOT cached in `second`
        second = _logged_api(log)
        with self._fail("fsm_llm.engine.api.forks.record_resumed"):
            with pytest.raises(FSMError, match="injected"):
                second.resume(cid)
        assert not second.has_conversation_ended(cid)
        with pytest.raises(ValueError, match="Unknown conversation ID"):
            second.get_data(cid)
        _nothing_left(second, log, writers=frozenset({cid}))

    def test_a_failed_resume_keeps_the_entry_the_id_already_had(
        self, log: EventLog
    ) -> None:
        api = _logged_api(log)
        cid = _stacked(api)
        data = api.get_data(cid)
        api.end_conversation(cid)
        prior = api._ended_conversations[cid]
        with self._fail("fsm_llm.engine.api.forks.record_resumed"):
            with pytest.raises(FSMError, match="injected"):
                api.resume(cid)
        assert api._ended_conversations[cid] is prior
        assert api.has_conversation_ended(cid) and api.get_data(cid) == data

    @pytest.mark.parametrize("route", ["fork", "restore_session"])
    def test_a_failed_fork_or_restore_caches_no_unreturned_id(
        self, log: EventLog, tmp_path: Any, route: str
    ) -> None:
        api = _logged_api(log, session_store=FileSessionStore(tmp_path))
        cid = _stacked(api)
        api.save_session(cid)
        before = dict(api._ended_conversations)
        if route == "fork":
            target, attempt = (
                "fsm_llm.engine.api.forks.record_forked",
                lambda: api.fork(cid, at_turn=1),
            )
        else:
            target, attempt = (
                "fsm_llm.engine.api.sessions.record_restored",
                lambda: api.restore_session(cid),
            )
        with self._fail(target), pytest.raises(FSMError, match="injected"):
            attempt()
        assert api._ended_conversations == before


class TestForgedRoutesThroughResume:
    """The routes of ``API.fork`` (step 4, D-009), through ``resume``: ONE trust rule."""

    @staticmethod
    def _resume(log: EventLog, api: API, tid: str) -> str:
        log.release(
            tid
        )  # the hand-written tree's writer, as a dead process would free it
        resumed, _ = api.resume(tid)
        return resumed

    @pytest.mark.parametrize("route", ["blob", "delta", "context_data"])
    def test_a_grant_in_the_root_blob_or_a_delta_is_not_seeded(
        self, log: EventLog, route: str
    ) -> None:
        planted = {"_approval_granted": _GRANT, "_run_output_keys": ["x"]}
        api = _logged_api(log)
        tid = _forge(
            log,
            api,
            child=_KIDS,
            root_internal=planted if route == "blob" else None,
            delta_internal=planted if route == "delta" else None,
            root_context=planted if route == "context_data" else None,
        )
        self._resume(log, api, tid)
        assert _raw(api, tid)["name"] == "Ann"
        assert not _raw(api, tid).keys() & set(planted)

    def test_a_forged_messages_key_or_response_format_key_seeds_no_grant(
        self, log: EventLog
    ) -> None:
        forged = {"_approval_granted": _GRANT}
        as_structured = _structured_fsm()
        as_structured["states"]["call_model"]["completion"]["response_format_key"] = (
            "_approval_granted"
        )
        for child_def in (_tool_fsm(messages_key="_approval_granted"), as_structured):
            api = _logged_api(log)
            tid = _forge(
                log,
                api,
                child=child_def,
                child_state="call_model",
                child_internal=forged,
            )
            self._resume(log, api, tid)
            assert api.get_current_state(tid) == "call_model"
            assert "_approval_granted" not in _raw(api, "kid-1")
            api.end_conversation(tid)
            log.release(tid)
            for suffix in (".jsonl", ".lock"):
                (log.root / "logs" / f"{tid}{suffix}").unlink(missing_ok=True)

    def test_only_the_default_transcript_survives_in_a_child(
        self, log: EventLog
    ) -> None:
        api = _logged_api(log)
        transcript = [{"role": "user", "content": "task"}]
        tid = _forge(
            log,
            api,
            child=_tool_fsm(),
            child_state="call_model",
            child_internal={
                _MESSAGES: transcript,
                "_approval_granted": _GRANT,
                "_run_output_keys": ["x"],
                "_counter": 7,
            },
            child_context={"_approval_granted": _GRANT, "keep": 1},
        )
        self._resume(log, api, tid)
        raw = _raw(api, "kid-1")
        assert raw[_MESSAGES] == transcript and raw["keep"] == 1
        assert not raw.keys() & {"_approval_granted", "_run_output_keys", "_counter"}

    def test_a_placeholder_in_the_return_context_is_dropped_with_a_warning(
        self, log: EventLog
    ) -> None:
        api = _logged_api(log)
        tid = _forge(
            log,
            api,
            child=_KIDS,
            return_context={"api_token": "<redacted>", "answer": 42},
        )
        with patch("fsm_llm.engine.api.forks.logger") as warned:
            self._resume(log, api, tid)
        assert api.conversation_stacks[tid][1].return_context == {"answer": 42}
        assert warned.warning.call_count == 1

    @pytest.mark.parametrize(
        "drift", ["unrecorded", "no_ref", "deleted", "wrong_fsm_id", "not_loadable"]
    )
    def test_each_drift_is_corrupt_leaves_nothing_and_frees_the_lock(
        self, log: EventLog, drift: str
    ) -> None:
        push: dict[str, Any] = {
            "unrecorded": {"definition_ref": _DROP, "definition_unrecorded": True},
            "no_ref": {"definition_ref": _DROP},
            "wrong_fsm_id": {"fsm_id": "fsm_Kids_00000000"},
            "not_loadable": {"definition_ref": log.put_json({"name": "x"})},
        }.get(drift, {})
        api = _logged_api(log)
        tid = _forge(log, api, child=_KIDS, push=push)
        if drift == "deleted":
            ref = _of(log, tid, _PUSH)[0]["definition_ref"]
            _blob_path(log, ref).unlink()
        log.release(tid)
        with pytest.raises(LogCorruptError):
            api.resume(tid)
        _nothing_left(api, log)
        assert not log.has_writer(tid) and _lock_is_free(log.root, tid)
