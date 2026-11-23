"""The run event vocabulary (plan ace7a8c8 step 22.3, roadmap E2.1, D-059): vocabulary only.

``run/start``, ``run/end`` and ``executable/resolve`` are declared event types with no producer
yet (phase S writes them). Adding members is additive: no ``LOG_FORMAT_VERSION`` bump, an
unknown type is still refused on write, and a log that holds the new types reads and folds like
one that does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fsm_llm import API
from fsm_llm.monitor.eventlog import timeline
from fsm_llm.runtime import EventLog
from fsm_llm.runtime.constants import LOG_FORMAT_VERSION
from fsm_llm.runtime.events import (
    DECLARED_TYPES,
    PRODUCED_TYPES,
    EventPayloadError,
    EventType,
)
from fsm_llm.runtime.stores import fold_session, fold_stack
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm import kind_corpus as kc

RUN_TYPES = [EventType.EXECUTABLE_RESOLVE, EventType.RUN_START, EventType.RUN_END]


def test_the_three_types_are_declared_and_not_produced() -> None:
    assert [t.value for t in RUN_TYPES] == [
        "executable/resolve",
        "run/start",
        "run/end",
    ]
    for member in RUN_TYPES:
        assert member in DECLARED_TYPES and member not in PRODUCED_TYPES
        assert EventType.from_string(member.value) is member
    assert LOG_FORMAT_VERSION == 1  # additive: the format version did not move


def test_an_unknown_type_is_still_refused_on_write(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    with pytest.raises((ValueError, EventPayloadError)):
        log.append("c1", "run/finish")


@pytest.mark.parametrize("member", RUN_TYPES)
def test_each_type_is_written_and_read_back_and_folds_ignore_it(
    member: EventType, tmp_path: Path
) -> None:
    fsm = kc._fsm(
        "Probe",
        "a",
        kc._state("a", response_instructions="A.", transitions=[kc._go("b")]),
        kc._state("b", response_instructions="B."),
    )
    log = EventLog(tmp_path)
    api = API.from_definition(
        fsm, llm_interface=ScriptedLLM(default_response="r"), event_log=log
    )
    cid, _ = api.start_conversation()
    api.run_until_terminal(cid, max_steps=5)
    before = list(log.read(cid))
    folded = fold_session(before, log.get_json)
    stack = fold_stack(before, log.get_json)
    assert folded is not None and stack is not None

    seq = log.append(
        cid,
        member,
        frame=0,
        template_identity="t" * 64,
        instance_identity="i" * 64,
        source="run",
    )
    after = list(log.read(cid))
    assert after[-1]["seq"] == seq and after[-1]["type"] == member.value
    assert after[-1]["template_identity"] == "t" * 64
    assert after[:-1] == before  # the new event changed nothing before it
    # the folds read a log with the event as they read it without
    # only the position of the last event moves (``last_seq``); the conversation is the same
    again = fold_session(after, log.get_json)
    assert again is not None and again.last_seq == seq
    assert again.model_dump(exclude={"last_seq"}) == folded.model_dump(
        exclude={"last_seq"}
    )
    stack_again = fold_stack(after, log.get_json)
    assert stack_again is not None and stack_again.root.last_seq == seq
    assert stack_again.model_dump(exclude={"root": {"last_seq"}}) == stack.model_dump(
        exclude={"root": {"last_seq"}}
    )
    # the monitor browse lists it by type
    shown = timeline(log, cid, limit=500)
    assert shown["events"][-1]["type"] == member.value
    assert shown["skipped"] == 0 and shown["corrupt_after_seq"] is None
    api.close()
