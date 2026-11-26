"""Row 1.18 tail: open writer handles of an ``EventLog`` (plan 77e5fef9, step 1: the baselines).

Today one ``JsonlWriter`` (the log descriptor and its lock descriptor) is opened on the first
append of a conversation tree and stays in ``EventLog._writers`` until ``release`` or ``close``:
nothing bounds the table. This file pins that as numbers a bound must change, and it is the one
new test file of the plan (every other row goes into an existing suite); step 3 adds the bounded
rows to it, and the baselines below stay valid beside them (the unbounded form is
``max_open_writers=None`` from step 3a, the only form before).

Baselines (this commit, no source change, so every row passes on the parent tree as well):

- ``TestDescriptorBaseline``: 1,000 idle conversations (a start and one turn each, one shared
  ``EventLog``, doubles only) hold 2,000 descriptors, 2.00 each, sync and async; ending 500 frees
  1,000; closing the log frees the rest. Counted through ``/proc/self/fd`` (Linux), so the count
  rows skip elsewhere; the writer-table rows are cross-platform.
- ``TestExhaustionBaseline``: a child process under ``RLIMIT_NOFILE`` 256 fails with
  ``LogWriteError`` (cause ``EMFILE``) near conversation 126 on the unbounded log. That is the
  non-vacuity of the bounded row of step 3a: the same child, given a bound, must pass 1,000.

Interface contract of the shared helper ``open_log(root, bound)``: *bound* is None for the
unbounded log (today's behaviour) or a non-negative int for ``max_open_writers``; returns an
``EventLog`` on *root*. Until step 3a adds the keyword only None is accepted (an int raises
``TypeError``, which is what the bounded rows of step 3a turn from red to green); step 3a
reduces the body to ``EventLog(root, max_open_writers=bound)``.
"""

from __future__ import annotations

import asyncio
import errno
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from fsm_llm import API
from fsm_llm.runtime import EventLog
from fsm_llm.runtime.events import LogWriteError
from tests.conftest import current_timing_factor, posix_only
from tests.test_fsm_llm.event_log_support import REPO, child_env
from tests.test_fsm_llm.test_event_log_census import _chat, _llm

IDLE_CONVERSATIONS = 1000
DESCRIPTORS_PER_IDLE_CONVERSATION = 2  # the log and its lock file
CHILD_NOFILE_LIMIT = 256
CHILD_CONVERSATIONS = 1000
#: the child fails when the table holds about (limit - baseline descriptors) / 2 writers
EXHAUSTION_WINDOW = (100, 128)
#: a hard cap on a hang, scaled by the timing factor; no assertion reads a duration
CHILD_TIMEOUT_SECONDS = 300


def open_log(root: Path, bound: int | None) -> EventLog:
    """An ``EventLog`` on *root* with *bound* open writers (None: unbounded, today's form)."""
    if "max_open_writers" in inspect.signature(EventLog).parameters:
        return EventLog(root, max_open_writers=bound)
    if bound is not None:
        raise TypeError("EventLog takes no max_open_writers before step 3a")
    return EventLog(root)


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
