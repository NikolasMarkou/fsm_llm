"""Row 1.18 tail: open writer handles of an ``EventLog`` (plan 77e5fef9, steps 1, 3a and 3b).

One ``JsonlWriter`` (the log descriptor and its lock descriptor) is opened on the first append of
a conversation tree. Before step 3a it stayed in ``EventLog._writers`` until ``release`` or
``close`` and nothing bounded the table; now ``EventLog(max_open_writers=N)`` closes the least
recently used idle writer (default 64, ``None`` is the unbounded table of before, ``0`` closes a
writer the moment its append ends). This file is the one new test file of the plan (every other
row goes into an existing suite).

Step 1 baselines, kept as the explicit ``max_open_writers=None`` rows (non-vacuity of the rest):

- ``TestDescriptorBaseline``: 1,000 idle conversations (a start and one turn each, one shared
  ``EventLog``, doubles only) hold 2,000 descriptors, 2.00 each, sync and async; ending 500 frees
  1,000; closing the log frees the rest. Counted through ``/proc/self/fd`` (Linux), so the count
  rows skip elsewhere; the writer-table rows are cross-platform.
- ``TestExhaustionBaseline``: a child process under ``RLIMIT_NOFILE`` 256 fails with
  ``LogWriteError`` (cause ``EMFILE``) near conversation 126 on the unbounded log.

Step 3a rows (the bound, the pins, the lease set; plan D-004):

- ``TestBoundedDescriptors``: with ``max_open_writers=16`` the same 1,000 conversations hold at
  most ``2 * 16`` descriptors above the baseline, every log is complete and paired, a fresh log
  resumes the first one; the exhaustion child passes 1,000 where the unbounded one failed.
- ``TestTheBoundIsValidated``, ``TestEviction`` (LRU order, pins, poison, the lock fd closed,
  bound 0), ``TestTheLeaseSet`` (``open_existing``, ``has_writer``, ``prune``, ``release``,
  ``close``), ``TestReopen`` (same file, ``seq`` continues, fail closed on a link, a vanished
  file or a held lock), ``TestRaces`` (pin against evict on threads, recorded; an eviction from a
  lane thread; queued lane jobs of an evicted id).

Step 3b rows (the tail-read re-open, D-005): ``TestReopenFailsClosed`` (a log appended to, cut
or replaced while idle is refused and never forked, the model call is not made, a big log is
re-opened by reading the window only) and ``TestAKilledChurningChild`` (a kill in the write of
a re-opened writer, then a lock holder that continues). The equivalence of the tail read and
the full scan (G2) is in ``test_event_log_jsonl.py`` (``TestTailReopen``).

Interface contract of the shared helper ``open_log(root, bound)``: *bound* is None for the
unbounded log or a non-negative int for ``max_open_writers``; returns an ``EventLog`` on *root*.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.runtime import EventLog
from fsm_llm.runtime.events import LogCorruptError, LogLockedError, LogWriteError
from fsm_llm.runtime.stores import jsonl
from fsm_llm.runtime.stores.jsonl import JsonlWriter, exclusive_lock, read_events
from tests.conftest import current_timing_factor, posix_only, symlink_or_skip
from tests.test_fsm_llm.event_log_support import REPO, child_env, reopen_after_kill
from tests.test_fsm_llm.test_event_log_census import _chat, _llm
from tests.test_fsm_llm.test_event_log_jsonl import can_lock_in_child

IDLE_CONVERSATIONS = 1000
DESCRIPTORS_PER_IDLE_CONVERSATION = 2  # the log and its lock file
CHILD_NOFILE_LIMIT = 256
CHILD_CONVERSATIONS = 1000
#: the child fails when the table holds about (limit - baseline descriptors) / 2 writers
EXHAUSTION_WINDOW = (100, 128)
#: a hard cap on a hang, scaled by the timing factor; no assertion reads a duration
CHILD_TIMEOUT_SECONDS = 300


def open_log(root: Path, bound: int | None) -> EventLog:
    """An ``EventLog`` on *root* with *bound* open writers (None: unbounded)."""
    return EventLog(root, max_open_writers=bound)


def open_descriptors() -> int:
    """The descriptors this process holds now (Linux ``/proc``)."""
    return len(os.listdir("/proc/self/fd"))


def _needs_proc() -> None:
    if not Path("/proc/self/fd").is_dir():
        pytest.skip("counts descriptors through /proc/self/fd (Linux)")


def _converse_idle(api: API, count: int, *, use_async: bool) -> list[str]:
    """Start *count* conversations on *api*, one turn each, and leave them idle."""
    ids: list[str] = []
    if not use_async:
        for _ in range(count):
            cid, _greeting = api.start_conversation()
            api.converse("my name is Ada", cid)
            ids.append(cid)
        return ids

    async def run() -> None:
        for _ in range(count):
            cid, _greeting = await api.astart_conversation()
            await api.aconverse("my name is Ada", cid)
            ids.append(cid)

    asyncio.run(run())
    return ids


class TestDescriptorBaseline:
    """The idle cost of one conversation on the unbounded log: two descriptors, 2.00 each."""

    @posix_only("counts descriptors through /proc/self/fd")
    @pytest.mark.timeout(
        900
    )  # 1,000 logged conversations: about 45 s, more under coverage
    @pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
    def test_idle_conversations_hold_two_descriptors_each(
        self, tmp_path: Path, use_async: bool
    ) -> None:
        _needs_proc()
        before = open_descriptors()
        log = open_log(tmp_path, None)
        api = API.from_definition(
            _chat(silent=False), llm_interface=_llm(), event_log=log
        )
        try:
            ids = _converse_idle(api, IDLE_CONVERSATIONS, use_async=use_async)
            idle = open_descriptors() - before
            assert idle == DESCRIPTORS_PER_IDLE_CONVERSATION * IDLE_CONVERSATIONS, idle
            assert len(log._writers) == IDLE_CONVERSATIONS
            assert all(log.has_writer(cid) for cid in ids)
            # the files behind the descriptors: one log and one lock per conversation
            names = os.listdir(tmp_path / "logs")
            assert sum(n.endswith(".jsonl") for n in names) == IDLE_CONVERSATIONS
            assert sum(n.endswith(".lock") for n in names) == IDLE_CONVERSATIONS
            # ending 500 frees exactly their two descriptors each: nothing else held one
            half = IDLE_CONVERSATIONS // 2
            for cid in ids[:half]:
                api.end_conversation(cid)
            assert open_descriptors() - before == (
                DESCRIPTORS_PER_IDLE_CONVERSATION * (IDLE_CONVERSATIONS - half)
            )
            assert len(log._writers) == IDLE_CONVERSATIONS - half
        finally:
            api.close()
            log.close()
        assert open_descriptors() == before

    @pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
    def test_every_idle_conversation_keeps_its_own_writer(
        self, tmp_path: Path, use_async: bool
    ) -> None:
        """The cross-platform form: the table holds one writer per idle conversation, the
        writer of an ended one goes, and the log leaves none behind when it closes."""
        count = 40
        log = open_log(tmp_path, None)
        api = API.from_definition(
            _chat(silent=False), llm_interface=_llm(), event_log=log
        )
        try:
            ids = _converse_idle(api, count, use_async=use_async)
            assert len(log._writers) == count
            assert all(log.has_writer(cid) for cid in ids)
            api.end_conversation(ids[0])
            assert not log.has_writer(ids[0])
            assert len(log._writers) == count - 1
        finally:
            api.close()
            log.close()
        assert len(log._writers) == 0


#: The exhaustion child: lowers ``RLIMIT_NOFILE`` after its imports, then starts conversations
#: (one turn each) on one ``EventLog`` until one fails or *count* is reached, and prints one JSON
#: line. argv: root, limit, count, bound ("none" or an int). Run from the repository root so
#: ``tests`` imports; ``resource`` exists only where the row runs (POSIX).
_EXHAUSTION_CHILD = """
import errno, json, resource, sys
from pathlib import Path
from fsm_llm import API
from tests.test_fsm_llm.test_event_log_bounded import open_descriptors, open_log
from tests.test_fsm_llm.test_event_log_census import _chat, _llm

root, limit, count, bound = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limit))
log = open_log(root, None if bound == "none" else int(bound))
api = API.from_definition(_chat(silent=False), llm_interface=_llm(), event_log=log)
result = {"failed_at": None, "error": None, "errno": None}
for number in range(1, count + 1):
    try:
        cid, _ = api.start_conversation()
        api.converse("my name is Ada", cid)
    except Exception as exc:
        result["failed_at"], result["error"] = number, type(exc).__name__
        link = exc
        while link is not None and not isinstance(link, OSError):
            link = link.__cause__ or link.__context__
        result["errno"] = None if link is None else link.errno
        break
result["descriptors"] = open_descriptors()
result["writers"] = len(log._writers)
print(json.dumps(result))
"""


def run_exhaustion_child(root: Path, *, bound: int | None) -> dict[str, Any]:
    """Run the exhaustion child; return its JSON line (``failed_at`` None when all passed)."""
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            _EXHAUSTION_CHILD,
            str(root),
            str(CHILD_NOFILE_LIMIT),
            str(CHILD_CONVERSATIONS),
            "none" if bound is None else str(bound),
        ],
        cwd=REPO,
        env=child_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=CHILD_TIMEOUT_SECONDS * current_timing_factor(),
    )
    assert done.returncode == 0, done.stderr[-2000:]
    lines = done.stdout.strip().splitlines()
    assert lines, done.stderr[-2000:]
    result: dict[str, Any] = json.loads(lines[-1])
    return result


class TestExhaustionBaseline:
    """The unbounded log fails by descriptor exhaustion near limit / 2 conversations."""

    @posix_only("resource.setrlimit and /proc/self/fd")
    def test_the_unbounded_log_fails_near_conversation_126_under_256_descriptors(
        self, tmp_path: Path
    ) -> None:
        _needs_proc()
        seen = run_exhaustion_child(tmp_path, bound=None)
        # the right reason: the log refused the write because open() hit EMFILE, with the
        # table holding the writers of the conversations that came before
        assert seen["error"] == LogWriteError.__name__, seen
        assert seen["errno"] == errno.EMFILE, seen
        low, high = EXHAUSTION_WINDOW
        assert low <= seen["failed_at"] <= high, seen
        assert seen["writers"] == seen["failed_at"] - 1, seen
        # two descriptors per writer plus the process's own fill the limit
        assert CHILD_NOFILE_LIMIT - 8 <= seen["descriptors"] <= CHILD_NOFILE_LIMIT, seen


BOUND = 16


def _paired(log: EventLog, cid: str) -> bool:
    """Every ``llm/request`` has its one outcome and every ``turn/start`` its ``turn/end``."""
    kinds = [e["type"] for e in log.read(cid)]
    return (
        kinds.count("llm/request")
        == kinds.count("llm/response") + kinds.count("provider/error")
        and kinds.count("turn/start") == kinds.count("turn/end")
        and kinds.count("turn/end") >= 1
    )


class TestBoundedDescriptors:
    """The done-when of row 1.18: 1,000 idle conversations under a bound of 16."""

    @posix_only("counts descriptors through /proc/self/fd")
    @pytest.mark.timeout(900)
    @pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
    def test_idle_conversations_hold_at_most_two_descriptors_per_open_writer(
        self, tmp_path: Path, use_async: bool
    ) -> None:
        """Kills: eviction disabled (M14), a bound ignored, a lock descriptor left open on
        eviction (the count is two per table entry, not more)."""
        _needs_proc()
        before = open_descriptors()
        log = open_log(tmp_path, BOUND)
        api = API.from_definition(
            _chat(silent=False), llm_interface=_llm(), event_log=log
        )
        try:
            ids = _converse_idle(api, IDLE_CONVERSATIONS, use_async=use_async)
            idle = open_descriptors() - before
            assert log.open_writer_count() <= BOUND
            assert idle == DESCRIPTORS_PER_IDLE_CONVERSATION * log.open_writer_count()
            assert idle <= DESCRIPTORS_PER_IDLE_CONVERSATION * BOUND
            assert len(log._writers) == log.open_writer_count()
            # every conversation is still live and its log is whole, paired and gapless
            assert all(log.has_writer(cid) for cid in ids)
            assert all(_paired(log, cid) for cid in ids)
            seqs = [e["seq"] for e in log.read(ids[0])]
            assert seqs == list(range(1, len(seqs) + 1))
            # ending an evicted conversation frees nothing more and forgets it
            api.end_conversation(ids[0])
            assert not log.has_writer(ids[0])
        finally:
            api.close()
            log.close()
        assert open_descriptors() == before
        # a fresh log folds the first conversation back (a full-scan open of an evicted id)
        with open_log(tmp_path, BOUND) as again:
            api2 = API.from_definition(
                _chat(silent=False), llm_interface=_llm(), event_log=again
            )
            resumed, state = api2.resume(ids[1])
            assert resumed == ids[1] == state.conversation_id
            assert api2.get_current_state(ids[1]) == "bye"
            api2.close()

    @pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
    def test_the_table_never_holds_more_writers_than_the_bound(
        self, tmp_path: Path, use_async: bool
    ) -> None:
        """The cross-platform form of the row above (no /proc)."""
        count = 40
        log = open_log(tmp_path, 4)
        api = API.from_definition(
            _chat(silent=False), llm_interface=_llm(), event_log=log
        )
        try:
            ids = _converse_idle(api, count, use_async=use_async)
            assert log.open_writer_count() == 4
            assert all(log.has_writer(cid) for cid in ids)
            assert all(_paired(log, cid) for cid in ids)
        finally:
            api.close()
            log.close()
        assert log.open_writer_count() == 0

    @posix_only("resource.setrlimit and /proc/self/fd")
    def test_a_bounded_log_passes_where_the_unbounded_one_failed_under_256_descriptors(
        self, tmp_path: Path
    ) -> None:
        """Non-vacuity of the done-when: the same child, the same limit, now with a bound."""
        _needs_proc()
        seen = run_exhaustion_child(tmp_path, bound=BOUND)
        assert seen["failed_at"] is None and seen["error"] is None, seen
        assert seen["writers"] <= BOUND, seen
        assert seen["descriptors"] <= CHILD_NOFILE_LIMIT // 2, seen


class Closes(list):  # type: ignore[type-arg]
    """Every ``JsonlWriter.close`` of ``log`` as ``(id, pins of the id, guard held)``."""

    log: EventLog | None = None


@pytest.fixture
def closes(monkeypatch: pytest.MonkeyPatch) -> Closes:
    """Records each first close of a writer at the call, read from the log that owns it (set
    ``closes.log``): what the pin/evict rows assert, never a clock."""
    seen = Closes()
    real = JsonlWriter.close

    def recording(self: JsonlWriter) -> None:
        if seen.log is not None and self._fd >= 0:
            log = seen.log
            seen.append(
                (
                    self.conversation_id,
                    log._pins.get(self.conversation_id, 0),
                    log._guard.locked(),
                )
            )
        real(self)

    monkeypatch.setattr(JsonlWriter, "close", recording)
    return seen


def _put(log: EventLog, cid: str, *, checkpoint: bool = False) -> int:
    return log.append(cid, "turn/end" if checkpoint else "turn/start")


def _ids_on_two_lanes(log: EventLog) -> tuple[str, str]:
    """Two ids whose trees run on different lanes."""
    first = "lane-a"
    for n in range(100):
        other = f"lane-b{n}"
        if log._lanes.lane_of(other) != log._lanes.lane_of(first):
            return first, other
    raise AssertionError("no second lane found")


class TestTheBoundIsValidated:
    @pytest.mark.parametrize("bound", [None, 0, 1, 64])
    def test_none_zero_and_whole_numbers_are_accepted(
        self, tmp_path: Path, bound: int | None
    ) -> None:
        with open_log(tmp_path, bound) as log:
            assert log.open_writer_count() == 0

    @pytest.mark.parametrize("bound", [True, False, -1, 1.5, "8", 2**70 * 1.0, [1]])
    def test_a_bool_a_negative_a_float_and_a_string_are_refused_by_name(
        self, tmp_path: Path, bound: Any
    ) -> None:
        with pytest.raises(ValueError, match="max_open_writers"):
            EventLog(tmp_path, max_open_writers=bound)

    def test_the_default_is_sixty_four_and_it_is_the_default(
        self, tmp_path: Path
    ) -> None:
        """Kills: None mapped to 64 (the unbounded row), the default left unbounded."""
        from fsm_llm.runtime.constants import DEFAULT_MAX_OPEN_WRITERS

        assert DEFAULT_MAX_OPEN_WRITERS == 64
        with EventLog(tmp_path / "default") as log:
            for n in range(70):
                _put(log, f"c{n}")
            assert log.open_writer_count() == 64
            assert all(log.has_writer(f"c{n}") for n in range(70))
        with open_log(tmp_path / "none", None) as log:
            for n in range(70):
                _put(log, f"c{n}")
            assert log.open_writer_count() == 70

    def test_zero_is_not_the_unbounded_log(self, tmp_path: Path) -> None:
        """Kills: 0 read as falsy and mapped to unbounded."""
        with open_log(tmp_path, 0) as log:
            for n in range(5):
                _put(log, f"c{n}")
            assert log.open_writer_count() == 0


class TestEviction:
    def test_the_least_recently_used_idle_writer_is_the_victim_and_a_use_refreshes_it(
        self, tmp_path: Path, closes: Closes
    ) -> None:
        """Kills: no ``move_to_end`` (the hot writer is the victim, M10)."""
        with open_log(tmp_path, 2) as log:
            closes.log = log
            for cid in ("a", "b", "a", "c"):
                _put(log, cid)
            assert list(log._writers) == ["a", "c"]
            assert [c[0] for c in closes] == ["b"]
            assert all(log.has_writer(cid) for cid in "abc")

    def test_an_eviction_closes_both_descriptors_under_the_guard_of_an_unpinned_id(
        self, tmp_path: Path, closes: Closes
    ) -> None:
        """Kills: the lock fd left open (a held flock after eviction), close outside
        ``_guard`` (M5), a pinned writer evicted (M4)."""
        with open_log(tmp_path, 1) as log:
            closes.log = log
            _put(log, "a")
            victim = log._writers["a"]
            _put(log, "b")
            assert victim._fd < 0 and victim._lock_fd < 0
            with exclusive_lock(log.root / "logs", "a") as free:
                assert free, "the evicted id still holds its writer lock"
            assert closes == [("a", 0, True)]  # unpinned, `_guard` held at the close

    def test_a_writer_in_use_is_never_evicted_and_the_idle_ones_go_around_it(
        self, tmp_path: Path, closes: Closes
    ) -> None:
        """Kills: eviction without the pin check (M4): the pinned ``a`` is the oldest, so the
        first eviction would take it."""
        with open_log(tmp_path, 1) as log:
            closes.log = log
            with log._lease("a") as held:
                _put(log, "b")
                _put(log, "c")
                assert held._fd >= 0 and list(log._writers) == ["a"]
                assert log._pins == {"a": 1}
            assert list(log._writers) == ["a"] and log._pins == {}
            assert [c[0] for c in closes] == [
                "b",
                "c",
            ]  # each idle one went at its unpin
            assert all(c[1] == 0 and c[2] for c in closes)
            _put(log, "a")  # the survivor is still the live, open writer of a
            assert held._fd >= 0

    def test_a_poisoned_writer_is_never_evicted_and_heals_only_by_release(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kills: a poisoned writer evicted (M6): the next append would re-open and succeed."""
        with open_log(tmp_path, 1) as log:
            with monkeypatch.context() as broken:
                broken.setattr(
                    jsonl,
                    "_fsync",
                    lambda fd: (_ for _ in ()).throw(OSError(5, "gone")),
                )
                with pytest.raises(LogWriteError):
                    _put(log, "a", checkpoint=True)
            bad = log._writers["a"]
            assert bad.poisoned
            _put(log, "b")
            _put(log, "c")
            assert (
                log._writers["a"] is bad and bad._fd >= 0
            )  # still there, still poisoned
            with pytest.raises(LogWriteError, match="poisoned"):
                _put(log, "a")
            log.release("a")
            assert _put(log, "a") >= 1  # the explicit release healed it

    def test_bound_zero_closes_a_writer_the_moment_its_append_ends(
        self, tmp_path: Path, closes: Closes
    ) -> None:
        """Kills: bound 0 keeping the writer (M11)."""
        with open_log(tmp_path, 0) as log:
            closes.log = log
            seqs = []
            for _ in range(3):
                seqs.append(_put(log, "a", checkpoint=True))
                assert log.open_writer_count() == 0 and log.has_writer("a")
            assert seqs == [
                1,
                2,
                3,
            ]  # every append re-opened the same log and continued
            assert len(closes) == 3 and all(c[1] == 0 and c[2] for c in closes)

    def test_the_guard_is_never_taken_while_a_writer_lock_is_held(
        self, tmp_path: Path
    ) -> None:
        """The lock order ``_guard -> writer locks``, recorded at every acquisition (one
        thread, so a writer lock that is locked is held by this thread)."""
        taken: list[bool] = []

        class Recording:
            def __init__(self, inner: Any, log: EventLog) -> None:
                self.inner, self.log = inner, log

            def __enter__(self) -> None:
                held = [
                    w._sync_guard.locked() or w._guard.locked()
                    for w in list(self.log._writers.values())
                ]
                taken.append(any(held))
                self.inner.acquire()

            def __exit__(self, *exc: object) -> None:
                self.inner.release()

            def locked(self) -> bool:
                return bool(self.inner.locked())

        with open_log(tmp_path, 1) as log:
            log._guard = Recording(log._guard, log)  # type: ignore[assignment]
            for cid in ("a", "b", "c", "a", "b"):
                _put(log, cid, checkpoint=True)
        assert len(taken) >= 10 and not any(taken)


class TestTheLeaseSet:
    """After an eviction the flock is gone; the lease set is the in-process owner rule."""

    def test_an_evicted_id_is_still_live_and_open_existing_refuses_it(
        self, tmp_path: Path
    ) -> None:
        """Kills: ``open_existing`` reading the table not the lease set (M7), ``has_writer``
        reading only the table (M12), a lease dropped on eviction, a ``release`` that leaves
        an evicted id live (M9: resume refused forever)."""
        with open_log(tmp_path, 1) as log:
            _put(log, "a")
            _put(log, "b")
            assert "a" not in log._writers  # evicted
            assert log.has_writer("a")
            with pytest.raises(LogLockedError, match="live writer"):
                log.open_existing("a")
            assert log.open_writer_count() == 1  # the refusal opened nothing
            log.release("a")
            assert not log.has_writer("a")
            assert log.open_existing("a") == 1  # the writer lock and the seq of the log
            assert log.has_writer("a")

    def test_open_existing_makes_room_first_and_its_writer_is_evictable_like_any(
        self, tmp_path: Path
    ) -> None:
        with open_log(tmp_path / "other", None) as writer:
            _put(writer, "x")
        with open_log(tmp_path / "other", 1) as log:
            _put(log, "y")
            assert log.open_existing("x") == 1
            assert log.open_writer_count() == 1 and log.has_writer("y")
            _put(log, "z")
            assert log.open_writer_count() == 1  # the opened one went as any idle one
            assert log.has_writer("x") and log.has_writer("z")

    @pytest.mark.parametrize("criterion", ["max_age_seconds", "max_logs"])
    def test_prune_keeps_the_log_of_a_live_evicted_id_and_deletes_a_released_one(
        self, tmp_path: Path, criterion: str
    ) -> None:
        """Kills: ``prune`` not skipping live ids (M8): the evicted id has a free lock and the
        oldest mtime, so it would be deleted mid-life."""
        with open_log(tmp_path, 1) as log:
            _put(log, "live")
            _put(log, "gone")
            log.release("gone")
            _put(log, "other")  # evicts "live": its lock is free
            assert "live" not in log._writers
            report = log.prune(**{criterion: 0})
            files = {p.name for p in (tmp_path / "logs").glob("*.jsonl")}
            assert "live.jsonl" in files and "gone.jsonl" not in files
            assert report.logs_removed >= 1 and report.logs_busy >= 1
            assert _put(log, "live") == 2  # the conversation went on, history intact

    def test_close_forgets_the_live_ids_and_refuses_a_later_append(
        self, tmp_path: Path
    ) -> None:
        log = open_log(tmp_path, 1)
        _put(log, "a")
        _put(log, "b")
        log.close()
        assert not log.has_writer("a") and not log.has_writer("b")
        assert log.open_writer_count() == 0
        with pytest.raises(LogWriteError, match="closed"):
            _put(log, "a")

    def test_a_dropped_log_frees_its_evicted_conversations_nothing(
        self, tmp_path: Path
    ) -> None:
        """An evicted id holds no descriptor, so ``close`` has nothing to close for it and the
        files stay (the log is never deleted by a close)."""
        with open_log(tmp_path, 1) as log:
            for cid in ("a", "b", "c"):
                _put(log, cid)
        assert {p.stem for p in (tmp_path / "logs").glob("*.jsonl")} == {"a", "b", "c"}


class TestReopen:
    """A re-open is the same open as a first one: it appends to the same file and fails closed."""

    def test_a_reopen_appends_to_the_same_file_and_continues_seq(
        self, tmp_path: Path
    ) -> None:
        with open_log(tmp_path, 1) as log:
            assert [_put(log, "a") for _ in range(3)] == [1, 2, 3]
            _put(log, "b")
            assert "a" not in log._writers
            assert _put(log, "a") == 4
            before = (tmp_path / "logs" / "a.jsonl").read_bytes()
            assert before.count(b"\n") == 4
            assert [e["seq"] for e in log.read("a")] == [1, 2, 3, 4]
            assert sorted(p.name for p in (tmp_path / "logs").iterdir()) == [
                "a.jsonl",
                "a.lock",
                "b.jsonl",
                "b.lock",
            ]

    def test_a_reopen_cuts_a_torn_tail_and_continues(self, tmp_path: Path) -> None:
        with open_log(tmp_path, 1) as log:
            _put(log, "a")
            _put(log, "b")
            with (tmp_path / "logs" / "a.jsonl").open("ab") as torn:
                torn.write(
                    b'{"v":1,"seq":2,"ty'
                )  # a writer died mid line while a was idle
            assert _put(log, "a") == 2
            assert [e["seq"] for e in log.read("a")] == [1, 2]

    def test_a_link_swapped_in_after_an_eviction_is_refused_and_its_target_untouched(
        self, tmp_path: Path
    ) -> None:
        """Kills: a re-open that skips the ``lstat`` regular-file check (it would be the
        ``O_NOFOLLOW`` error of another type) or follows the link."""
        target = tmp_path / "elsewhere.jsonl"
        target.write_bytes(b"")
        with open_log(tmp_path / "root", 1) as log:
            _put(log, "a")
            _put(log, "b")
            path = tmp_path / "root" / "logs" / "a.jsonl"
            path.unlink()
            symlink_or_skip(path, target)
            with pytest.raises(LogCorruptError, match="not a regular file"):
                _put(log, "a")
            assert target.read_bytes() == b""
            assert log.has_writer("a") and "a" not in log._writers

    def test_a_lock_file_swapped_for_a_link_is_refused_by_o_nofollow(
        self, tmp_path: Path
    ) -> None:
        """Kills: ``O_NOFOLLOW`` dropped on the re-open (M15): the lock would be taken on the
        link's target and the append would go on."""
        target = tmp_path / "elsewhere.lock"
        target.write_bytes(b"")
        with open_log(tmp_path / "root", 1) as log:
            _put(log, "a")
            _put(log, "b")
            lock = tmp_path / "root" / "logs" / "a.lock"
            lock.unlink()
            symlink_or_skip(lock, target)
            with pytest.raises(LogWriteError, match="cannot open"):
                _put(log, "a")
            assert (tmp_path / "root" / "logs" / "a.jsonl").read_bytes().count(
                b"\n"
            ) == 1

    def test_a_vanished_log_is_refused_and_never_recreated(
        self, tmp_path: Path
    ) -> None:
        """Kills: a re-open that creates (M20): the log would restart at seq 1."""
        with open_log(tmp_path, 1) as log:
            _put(log, "a")
            _put(log, "b")
            (tmp_path / "logs" / "a.jsonl").unlink()
            with pytest.raises(LogCorruptError, match="vanished"):
                _put(log, "a")
            assert not (tmp_path / "logs" / "a.jsonl").exists()
            assert "a" not in log._writers  # nothing half open stays in the table

    def test_a_lock_held_by_another_holder_refuses_the_append_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        """Fail closed: the checkpoint raises (so the model call is not made), never waits, and
        the conversation goes on once the holder is gone."""
        with open_log(tmp_path, 1) as log:
            _put(log, "a")
            _put(log, "b")
            with exclusive_lock(tmp_path / "logs", "a") as held:
                assert held
                with pytest.raises(LogLockedError):
                    _put(log, "a", checkpoint=True)
                assert len(list(log.read("a"))) == 1
                assert "a" not in log._writers and log.open_writer_count() <= 1
            assert _put(log, "a", checkpoint=True) == 2

    @pytest.mark.timeout(120)
    def test_a_lock_held_by_another_process_refuses_the_append(
        self, tmp_path: Path
    ) -> None:
        holder = (
            "import sys; from pathlib import Path\n"
            "from fsm_llm.runtime.stores.jsonl import exclusive_lock\n"
            "with exclusive_lock(Path(sys.argv[1]), sys.argv[2]) as held:\n"
            "    print('held' if held else 'free', flush=True); sys.stdin.readline()\n"
        )
        with open_log(tmp_path, 1) as log:
            _put(log, "a")
            _put(log, "b")
            with subprocess.Popen(
                [sys.executable, "-c", holder, str(tmp_path / "logs"), "a"],
                cwd=REPO,
                env=child_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            ) as proc:
                assert proc.stdout is not None and proc.stdin is not None
                try:
                    assert proc.stdout.readline().strip() == "held"
                    with pytest.raises(LogLockedError):
                        _put(log, "a", checkpoint=True)
                    assert len(list(log.read("a"))) == 1
                finally:
                    proc.stdin.write("\n")
                    proc.stdin.close()
                    proc.wait(timeout=60 * current_timing_factor())
            assert _put(log, "a") == 2


def _point_of(log: EventLog, cid: str) -> tuple[int, int, int]:
    """The resume point the log recorded when it evicted *cid* (``(last seq, dev, inode)``)."""
    point = log._live[cid]
    assert point is not None and cid not in log._writers
    return point


class TestReopenFailsClosed:
    """A re-open trusts the middle of an idle log and checks its tail (D-005): a file another
    writer changed, cut or replaced while the lock was free is refused, never forked."""

    def _evicted(self, tmp_path: Path) -> tuple[EventLog, Path]:
        log = open_log(tmp_path, 1)
        _put(log, "a")
        _put(log, "a")
        _put(log, "b")  # evicts "a"
        assert "a" not in log._writers and log.has_writer("a")
        return log, tmp_path / "logs" / "a.jsonl"

    def test_the_resume_point_is_recorded_at_the_eviction(self, tmp_path: Path) -> None:
        log, path = self._evicted(tmp_path)
        with log:
            seq, dev, ino = _point_of(log, "a")
            assert seq == 2
            assert (dev, ino) == (path.stat().st_dev, path.stat().st_ino)
            assert log._live["b"] is None  # an open writer has none

    def test_a_log_another_writer_appended_to_is_refused_not_forked(
        self, tmp_path: Path
    ) -> None:
        """Kills: continuing from the recorded ``seq`` without reading the file, an expected
        ``seq`` check dropped (M21): the next append would be a duplicate ``seq`` 3."""
        log, path = self._evicted(tmp_path)
        with log:
            other = JsonlWriter(path)  # the lock is free while "a" idles
            other.append(
                {"ts": "t", "conversation_id": "a", "frame": 0, "type": "turn/start"}
            )
            other.close()
            before = path.read_bytes()
            with pytest.raises(LogCorruptError, match="changed while idle"):
                _put(log, "a", checkpoint=True)
            assert path.read_bytes() == before  # we wrote and cut nothing
            assert log.has_writer("a") and "a" not in log._writers
            with pytest.raises(LogCorruptError):  # and it stays refused
                _put(log, "a")
            log.release("a")  # the way out: a first open, a full scan
            assert _put(log, "a") == 4

    def test_a_log_cut_back_while_idle_is_refused(self, tmp_path: Path) -> None:
        log, path = self._evicted(tmp_path)
        with log:
            data = path.read_bytes()
            cut = data[: data.index(b"\n") + 1]
            path.write_bytes(cut)
            with pytest.raises(LogCorruptError, match="changed while idle"):
                _put(log, "a")
            assert path.read_bytes() == cut

    def test_a_replaced_file_is_refused_though_its_last_seq_is_the_same(
        self, tmp_path: Path
    ) -> None:
        """Kills: a swapped file accepted (M16): a byte copy of the log renamed over it has the
        recorded last ``seq``, and only its inode says it is not the file that was closed."""
        log, path = self._evicted(tmp_path)
        with log:
            if _point_of(log, "a")[2] == 0:
                pytest.skip("this file system gives no inode: nothing to compare")
            copy = path.with_name("copy.tmp")
            copy.write_bytes(path.read_bytes())
            os.replace(copy, path)
            with pytest.raises(LogCorruptError, match="replaced while idle"):
                _put(log, "a")
            assert path.read_bytes().count(b"\n") == 2
            assert not (tmp_path / "logs" / "a.jsonl.tmp").exists()

    def test_the_model_call_is_not_made_when_the_reopen_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The checkpoint raises before the call it protects: a changed log of an idle
        conversation refuses its next turn and the double is asked nothing."""
        llm = _llm()
        with open_log(tmp_path, 1) as log:
            api = API.from_definition(
                _chat(silent=False), llm_interface=llm, event_log=log
            )
            first, _ = api.start_conversation()
            second, _ = api.start_conversation()  # evicts the writer of `first`
            assert first not in log._writers
            path = tmp_path / "logs" / f"{first}.jsonl"
            other = JsonlWriter(path)
            other.append(
                {"ts": "t", "conversation_id": first, "frame": 0, "type": "turn/start"}
            )
            other.close()
            asked = len(llm.requests)
            calls = len(llm.call_history)
            with pytest.raises(LogCorruptError):
                api.converse("my name is Ada", first)
            assert (len(llm.requests), len(llm.call_history)) == (asked, calls)
            api.converse("my name is Ada", second)  # the other conversation is fine
            api.close()

    @pytest.mark.timeout(300)
    def test_a_reopen_of_a_big_log_reads_the_window_and_not_the_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Counts, never a clock. Kills: the tail read replaced by a full scan (the equal
        result at the cost of the log)."""
        writer = JsonlWriter(tmp_path / "logs" / "a.jsonl")
        for _ in range(100):
            writer.append(
                {
                    "ts": "t",
                    "conversation_id": "a",
                    "frame": 0,
                    "type": "turn/start",
                    "pad": "x" * 30_000,
                }
            )
        writer.close()
        size = (tmp_path / "logs" / "a.jsonl").stat().st_size
        assert size > 2_900_000
        with open_log(tmp_path, 1) as log:
            assert _put(log, "a") == 101  # the first open scans the log, once
            _put(log, "b")  # evicts "a"
            me = threading.get_ident()
            read: list[int] = []

            def counting(real: Any) -> Any:
                def spy(*args: Any) -> bytes:
                    data = real(*args)
                    if threading.get_ident() == me:
                        read.append(len(data))
                    return data

                return spy

            monkeypatch.setattr(os, "read", counting(os.read))
            if hasattr(os, "pread"):
                monkeypatch.setattr(os, "pread", counting(os.pread))
            assert _put(log, "a") == 102
            monkeypatch.undo()
            assert 0 < sum(read) <= 2 * jsonl.MAX_EVENT_BYTES + 1, sum(read)


KILL_CHURN_CHILD = """
import os, sys
from fsm_llm.runtime import EventLog
from fsm_llm.runtime.stores import jsonl

writes = 0
real = os.write

def torn(fd, data):
    global writes
    writes += 1
    if writes == 21:  # the 11th line of "a", through a re-opened writer: half of it, then stop
        real(fd, bytes(data)[: len(data) // 2])
        print("torn", flush=True)
        sys.stdin.readline()
    return real(fd, data)

jsonl._write = torn
log = EventLog(sys.argv[1], max_open_writers=1)
for _ in range(15):
    log.append("a", "turn/start")
    log.append("b", "turn/start")
"""


class TestAKilledChurningChild:
    @pytest.mark.timeout(300)
    def test_a_kill_in_the_write_of_a_reopened_writer_leaves_a_log_that_continues(
        self, tmp_path: Path
    ) -> None:
        """Bound 1, two conversations taking turns, so every write after the first two goes
        through a tail-read re-open; the child dies in the middle of the 11th line of ``a``
        (``Popen.kill()``). The next lock holder cuts the torn half, continues ``seq`` without a
        duplicate or a gap, and ``b`` is untouched."""
        if not can_lock_in_child():
            pytest.skip("this platform cannot start a child that locks a file")
        root = tmp_path / "log"
        with subprocess.Popen(
            [sys.executable, "-c", KILL_CHURN_CHILD, str(root)],
            cwd=REPO,
            env=child_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        ) as proc:
            assert proc.stdout is not None
            try:
                assert proc.stdout.readline().strip() == "torn"
                proc.kill()  # SIGKILL on POSIX, TerminateProcess on Windows
                proc.wait(timeout=60 * current_timing_factor())
            finally:
                if proc.stdin is not None:
                    proc.stdin.close()
        path_a = root / "logs" / "a.jsonl"
        committed = path_a.read_bytes().rfind(b"\n") + 1
        assert path_a.stat().st_size > committed  # the torn half is on disk
        with reopen_after_kill(lambda: EventLog(root, max_open_writers=1)) as log:
            assert [e["seq"] for e in log.read("a")] == list(range(1, 11))
            assert [e["seq"] for e in log.read("b")] == list(range(1, 11))
            assert log.open_existing("a") == 10
            assert _put(log, "a") == 11
            assert _put(log, "b") == 11
            assert [e["seq"] for e in log.read("a")] == list(range(1, 12))
        assert path_a.read_bytes().endswith(b"\n")


WORKERS = 6
ROUNDS = 25
RACE_IDS = 5
RACE_REPEATS = 4
JOIN_CAP_SECONDS = 120


class TestRaces:
    @pytest.mark.timeout(600)
    @pytest.mark.parametrize("repeat", range(RACE_REPEATS))
    def test_pin_and_evict_race_closes_only_unpinned_writers_under_the_guard(
        self, tmp_path: Path, closes: Closes, repeat: int
    ) -> None:
        """Threads append (checkpoints, so each pin spans an fsync) over more ids than the
        bound. Recorded at every close: the id was unpinned and ``_guard`` was held. No append
        meets a closed writer (no spurious ``LogWriteError``) and no lock (no ``LogLockedError``).
        Kills: eviction without the pin check (M4), close outside ``_guard`` (M5)."""
        errors: list[BaseException] = []
        with open_log(tmp_path, 2) as log:
            closes.log = log
            gate = threading.Barrier(WORKERS, timeout=JOIN_CAP_SECONDS)

            def work(index: int) -> None:
                try:
                    gate.wait()
                    for step in range(ROUNDS):
                        _put(log, f"c{(index + step) % RACE_IDS}", checkpoint=True)
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=work, args=(i,)) for i in range(WORKERS)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(JOIN_CAP_SECONDS * current_timing_factor())
            assert not any(t.is_alive() for t in threads)
            assert errors == []
            assert len(closes) >= RACE_IDS  # the race happened: evictions were recorded
            assert all(pins == 0 and guard for _, pins, guard in closes), closes
            total = sum(len(list(log.read(f"c{n}"))) for n in range(RACE_IDS))
            assert total == WORKERS * ROUNDS
            for n in range(RACE_IDS):
                seqs = [e["seq"] for e in log.read(f"c{n}")]
                assert seqs == list(range(1, len(seqs) + 1))

    async def test_a_lane_thread_evicts_a_writer_of_another_lane_without_waiting(
        self, tmp_path: Path
    ) -> None:
        """The eviction runs on the lane thread of the id that needed room, closes the other
        lane's idle writer at once (even with that lane busy) and waits for no lane job.
        Kills: an eviction queued behind the victim's lane (it would still be open here)."""
        with open_log(tmp_path, 1) as log:
            first, second = _ids_on_two_lanes(log)
            await log.lane_call(first, _put, log, first)
            victim = log._writers[first]
            parked, gate = threading.Event(), threading.Event()

            def park() -> None:
                parked.set()
                gate.wait(JOIN_CAP_SECONDS)

            job = log._lanes.after(first, park)  # the victim's lane is busy
            assert job is not None
            while not parked.is_set():
                await asyncio.sleep(0)
            try:
                await log.lane_call(second, _put, log, second)
                assert victim._fd < 0 and victim._lock_fd < 0
                assert log.open_writer_count() == 1
            finally:
                gate.set()
            await asyncio.wrap_future(job)

    async def test_jobs_queued_for_an_evicted_id_re_open_and_close_writes_them(
        self, tmp_path: Path
    ) -> None:
        """A job queued behind a busy lane meets its writer evicted, re-opens it and loses no
        event; ``close`` then waits for it. Kills: the closed-log refusal applied to a live
        evicted id."""
        log = open_log(tmp_path, 1)
        first = "lane-a"
        await log.lane_call(first, _put, log, first)
        parked, gate = threading.Event(), threading.Event()

        def park() -> None:
            parked.set()
            gate.wait(JOIN_CAP_SECONDS)

        busy = log._lanes.after(first, park)
        queued = log._lanes.after(first, lambda: _put(log, first, checkpoint=True))
        assert busy is not None and queued is not None
        while not parked.is_set():
            await asyncio.sleep(0)
        _put(log, "other")  # evicts the writer of `first`, no job holds it
        assert "lane-a" not in log._writers and log.has_writer(first)
        gate.set()
        await asyncio.to_thread(log.close)  # the lane drains, the queued job re-opens
        assert queued.exception() is None
        assert [e["seq"] for e in read_events(tmp_path / "logs" / "lane-a.jsonl")] == [
            1,
            2,
        ]
        assert not log.has_writer(first) and log.open_writer_count() == 0


class TestForkAndResumeOfAnEvictedConversation:
    """An evicted conversation is still a conversation of this log: a fork reads it with no
    lock, a resume of it is refused while it is live and works once it is ended."""

    def _api(self, log: EventLog) -> API:
        return API.from_definition(
            _chat(silent=False), llm_interface=_llm(), event_log=log
        )

    def test_a_fork_of_an_evicted_source_works_and_leaves_the_source_alone(
        self, tmp_path: Path
    ) -> None:
        with open_log(tmp_path, 1) as log:
            api = self._api(log)
            first, _ = api.start_conversation()
            api.converse("my name is Ada", first)
            second, _ = api.start_conversation()  # evicts the writer of `first`
            assert first not in log._writers and log.has_writer(first)
            before = (tmp_path / "logs" / f"{first}.jsonl").read_bytes()
            forked, _ = api.fork(first, at_turn=1)
            assert forked not in (first, second)
            assert (tmp_path / "logs" / f"{first}.jsonl").read_bytes() == before
            assert log.open_writer_count() <= 1
            api.close()

    def test_a_resume_of_a_live_evicted_id_is_refused_and_works_after_it_ends(
        self, tmp_path: Path
    ) -> None:
        """Kills: ``open_existing`` and ``API.resume`` reading the table, not the lease set
        (M7), and a ``release`` that keeps the lease (M9)."""
        with open_log(tmp_path, 1) as log:
            api = self._api(log)
            first, _ = api.start_conversation()
            api.converse("my name is Ada", first)
            api.start_conversation()  # evicts the writer of `first`
            other = self._api(log)  # a second API on the same log
            with pytest.raises(LogLockedError, match="live writer"):
                other.resume(first)
            assert other.list_active_conversations() == []
            api.end_conversation(first)  # releases the lease
            resumed, state = other.resume(first)
            assert resumed == first == state.conversation_id
            api.close()
            other.close()
