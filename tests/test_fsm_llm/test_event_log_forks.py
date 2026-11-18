"""``API.fork`` (plan 5a6a1f80 step 4, roadmap row 1.4): a new conversation tree from the state of an
old one at the end of its Nth turn, with the whole stack rebuilt from the log.

Four layers: the happy paths on real conversations (``ScriptedLLM``, compared with what the LIVE
source showed, never with a second fold), the forged routes (a log is a file: a hand-written tree
whose blobs lie), the refusals, and the teardown at every stage of the rebuild. The D-009 trust
rule has one RED test per route. Counts, bytes and sequences, never timing.
"""

from __future__ import annotations

import copy
import threading
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm import API
from fsm_llm.core.definitions import FSMError
from fsm_llm.engine.api import forks
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.runtime import EventLog, EventType, LogCorruptError
from fsm_llm.runtime.stores import fold_stack
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.event_log_support import tree_files
from tests.test_fsm_llm.support import _state
from tests.test_fsm_llm.test_completion_state import (
    _MESSAGES,
    _structured_fsm,
    _tool_fsm,
)
from tests.test_fsm_llm.test_event_log_stack import _ROOT, _blob_path, _of, _view

_GRANT = {"tool": "add", "arguments": {"a": 2, "b": 3}}
_START, _END, _REC, _PUSH = (
    EventType.TURN_START,
    EventType.TURN_END,
    EventType.CONTEXT_RECORD,
    EventType.STACK_PUSH,
)
_DROP: Any = object()  # a push field a test removes

_KIDS: dict[str, Any] = {
    "name": "Kids",
    "description": "A child that survives a turn",
    "initial_state": "hello",
    "states": {
        "hello": _state(
            "hello",
            response_instructions="Kid hello",
            transitions=[{"target_state": "talk", "description": "go"}],
        ),
        "talk": _state(
            "talk",
            response_instructions="Kid talk",
            transitions=[
                {
                    "target_state": "end",
                    "description": "never",
                    "conditions": [{"description": "n", "logic": {"==": [1, 2]}}],
                }
            ],
        ),
        "end": _state("end"),
    },
}


def _chain() -> dict[str, Any]:
    """w0 -> w1 -> ... -> w5 (terminal): every turn changes the state."""
    states = {
        f"w{i}": _state(
            f"w{i}",
            response_instructions=f"Say {i}",
            required_context_keys=["name"] if i == 0 else [],
            transitions=[{"target_state": f"w{i + 1}", "description": "go"}],
        )
        for i in range(5)
    }
    states["w5"] = _state("w5", response_instructions="Done")
    return {
        "name": "Chain",
        "description": "A chain",
        "initial_state": "w0",
        "states": states,
    }


def _api(log: EventLog, definition: Any = None, **kw: Any) -> API:
    llm = ScriptedLLM([], extraction_data={"name": "Ann"}, default_response="ok")
    return API(
        copy.deepcopy(definition or _ROOT), llm_interface=llm, event_log=log, **kw
    )


def _stacked(api: API, depth: int = 2, **push_options: Any) -> str:
    """Start, turn 1 at the root, ``depth - 1`` pushes, turn 2 at the top frame."""
    cid, _ = api.start_conversation()
    api.converse("Ann", cid)
    for _ in range(depth - 1):
        api.push_fsm(
            cid, copy.deepcopy(_KIDS), context_to_pass={"topic": "x"}, **push_options
        )
    api.converse("hi", cid)
    return cid


def _shown(api: API, cid: str) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    return (
        api.get_current_state(cid),
        api.get_data(cid),
        api.get_conversation_history(cid),
    )


def _raw(api: API, cid: str) -> dict[str, Any]:
    return api.fsm_manager.copy_raw_context_data(cid) or {}


def _kid_id(api: API, root: str, index: int = 1) -> str:
    return api.conversation_stacks[root][index].conversation_id


class TestAForkAtEveryTurn:
    def test_each_turn_of_a_scripted_conversation_forks_to_what_the_source_showed(
        self, log: EventLog
    ) -> None:
        api = _api(log, _chain())
        cid, _ = api.start_conversation()
        seen = [_shown(api, cid)]
        for text in ("Ann", "b", "c", "d"):
            api.converse(text, cid)
            seen.append(_shown(api, cid))
        assert [s[0] for s in seen] == ["w0", "w1", "w2", "w3", "w4"]
        for turn, shown in enumerate(seen):
            new_id, state = api.fork(cid, at_turn=turn)
            assert new_id != cid and _shown(api, new_id) == shown, f"turn {turn}"
            assert (state.current_state, state.stack_depth) == (shown[0], 1)
            assert not _raw(api, new_id).keys() & {"_approval_granted"}

    def test_two_forks_of_one_source_are_independent(self, log: EventLog) -> None:
        api = _api(log, _chain())
        cid, _ = api.start_conversation()
        api.converse("Ann", cid)
        (one, _), (two, _) = api.fork(cid, at_turn=1), api.fork(cid, at_turn=1)
        before = _shown(api, two)
        api.converse("more", one)
        assert (api.get_current_state(one), _shown(api, two)) == ("w2", before)
        assert api.get_current_state(cid) == "w1" and len({cid, one, two}) == 3

    def test_a_fork_continues_as_its_own_conversation(self, log: EventLog) -> None:
        api = _api(log, _chain())
        cid, _ = api.start_conversation()
        api.converse("Ann", cid)
        api.converse("b", cid)
        new_id, _ = api.fork(cid, at_turn=1)
        api.converse("c", new_id)
        assert api.get_current_state(new_id) == "w2"
        again, _ = api.fork(new_id, at_turn=1)  # a fork of a fork
        assert api.get_current_state(again) == "w2"
        assert api.get_current_state(cid) == "w2"

    def test_another_history_window_trims_the_fork_not_the_source(
        self, log: EventLog
    ) -> None:
        api = _api(log, _chain())
        cid, _ = api.start_conversation()
        for text in ("Ann", "b", "c"):
            api.converse(text, cid)
        small = _api(log, _chain(), max_history_size=1)
        new_id, _ = small.fork(cid, at_turn=3)
        kept = small.get_conversation_history(new_id)
        assert kept == [{"user": "c"}, {"system": "ok"}]  # one exchange
        assert len(api.get_conversation_history(cid)) > len(kept)
        assert small.get_current_state(new_id) == "w3"


class TestAStackedFork:
    def test_the_whole_stack_comes_back_and_pops_as_the_source_would(
        self, log: EventLog
    ) -> None:
        api = _api(log)
        cid = _stacked(
            api,
            return_context={"answer": 42, "password": "hunter2"},
            shared_context_keys=["name"],
            preserve_history=True,
        )
        observed = _view(api, cid)
        new_id, state = api.fork(cid, at_turn=2)
        assert observed[0] == 2 == state.stack_depth
        assert _view(api, new_id) == observed
        old, new = api.conversation_stacks[cid][1], api.conversation_stacks[new_id][1]
        assert new.conversation_id != old.conversation_id and new.fsm_id == old.fsm_id
        assert (new.shared_context_keys, new.preserve_history) == (["name"], True)
        assert new.return_context == {"answer": 42}  # the placeholder is not merged
        assert "hunter2" not in str(_raw(api, new_id)) + str(new.return_context)
        old.return_context.pop("password")  # what the source merges, minus the secret
        for target in (cid, new_id):
            api.pop_fsm(target, context_to_return={"result": 1})
        assert api.get_data(new_id) == api.get_data(cid)
        assert api.get_data(cid)["answer"] == 42 == api.get_data(new_id)["answer"]
        summary = "_sub_conversation_summary"
        assert _raw(api, new_id)[summary] == _raw(api, cid)[summary]

    def test_a_turn_at_a_restored_frame_and_a_fork_of_the_fork(
        self, log: EventLog
    ) -> None:
        api = _api(log)
        cid = _stacked(api)
        first, _ = api.fork(cid, at_turn=2)
        api.converse("again", first)  # turn 1 of the fork, at frame 1
        mid = _view(api, first)
        from_turn, _ = api.fork(first, at_turn=1)
        from_genesis, _ = api.fork(first, at_turn=0)
        assert _view(api, from_turn) == mid
        assert _view(api, from_genesis)[0] == 2
        assert _view(api, from_genesis) == _view(api, cid)  # the state it was forked at
        for tail in (first, from_turn, from_genesis):
            folded = fold_stack(log.read(tail), log.get_json)
            assert folded is not None and folded.depth == 2

    def test_a_pushed_child_that_was_popped_is_not_in_a_fork_after_the_pop(
        self, log: EventLog
    ) -> None:
        api = _api(log)
        cid = _stacked(api)
        api.pop_fsm(cid, context_to_return={"r": 1})
        api.converse("back", cid)  # turn 3, at the root
        before_pop, _ = api.fork(cid, at_turn=2)
        after_pop, _ = api.fork(cid, at_turn=3)
        assert (api.get_stack_depth(before_pop), api.get_stack_depth(after_pop)) == (
            2,
            1,
        )
        assert "r" not in api.get_data(before_pop) and api.get_data(after_pop)["r"] == 1

    def test_the_forks_own_log_stands_alone(self, log: EventLog) -> None:
        api = _api(log)
        cid = _stacked(api, shared_context_keys=["name"], preserve_history=True)
        cut = [e for e in log.read(cid) if e["type"] == "turn/end"][1]["seq"]
        source_log = (log.root / "logs" / f"{cid}.jsonl").read_bytes()
        files = {p for p in tree_files(log.root) if "blobs" in p.parts}
        new_id, _ = api.fork(cid, at_turn=2)
        events = list(log.read(new_id))
        assert [(e["type"], e["frame"]) for e in events] == [
            ("context/record", 0),
            ("stack/push", 0),
        ]
        record, push = events
        assert record["source"] == "fork" and record["state"] == "chat"
        assert record["forked_from"] == {
            "conversation_id": cid,
            "seq": cut,
            "turn": 2,
        }
        assert push["restored"] is True and push["depth"] == 2
        assert push["child_conversation_id"] == _kid_id(api, new_id)
        assert (push["shared_context_keys"], push["preserve_history"]) == (
            ["name"],
            True,
        )
        assert log.get_json(push["state_ref"])["current_state"] == "talk"
        assert (log.root / "logs" / f"{cid}.jsonl").read_bytes() == source_log
        fresh = {p for p in tree_files(log.root) if "blobs" in p.parts} - files
        assert not fresh  # the same state is the same blob: nothing is copied

    def test_a_fork_takes_no_conversation_lock_and_no_writer_of_the_source(
        self, log: EventLog
    ) -> None:
        api = _api(log)
        cid = _stacked(api)
        held, release = threading.Event(), threading.Event()

        def hold() -> None:
            with api.fsm_manager._conversation_locks[_kid_id(api, cid)]:
                held.set()
                release.wait(30)

        holder = threading.Thread(target=hold)
        holder.start()
        assert held.wait(30)
        out: list[Any] = []
        forker = threading.Thread(target=lambda: out.append(api.fork(cid, at_turn=2)))
        forker.start()
        forker.join(30)
        blocked = forker.is_alive()
        release.set()
        holder.join(30)
        forker.join(30)
        assert not blocked and len(out) == 1
        with EventLog(log.root) as other:  # a reader's own log object opens no writer
            _api(other).fork(cid, at_turn=2)
            assert cid not in other._writers

    def test_the_source_may_keep_appending_between_the_lookup_and_the_fold(
        self, log: EventLog
    ) -> None:
        api = _api(log)
        cid = _stacked(api)
        observed, real = _view(api, cid), fold_stack

        def racing(events: Any, load_blob: Any) -> Any:
            api.update_context(cid, {"late": 1})
            api.converse("more", cid)  # the source moves on before the fold reads
            return real(events, load_blob)

        with patch("fsm_llm.engine.api.forks.fold_stack", racing):
            new_id, _ = api.fork(cid, at_turn=2)
        assert _view(api, new_id) == observed != _view(api, cid)


class TestRefusals:
    def test_a_log_off_api_refuses(self) -> None:
        api = API(copy.deepcopy(_ROOT), llm_interface=ScriptedLLM([]))
        with pytest.raises(FSMError, match="needs an event log"):
            api.fork("any-id", at_turn=0)

    def test_an_unknown_id_creates_no_file_and_no_conversation(
        self, log: EventLog
    ) -> None:
        api = _api(log)
        before = sorted(p.name for p in log.root.rglob("*"))
        for bad in ("nope", "../x"):
            with pytest.raises(FSMError):
                api.fork(bad, at_turn=0)
        assert sorted(p.name for p in log.root.rglob("*")) == before
        assert api.list_active_conversations() == [] and not log._writers

    @pytest.mark.parametrize(
        "at_turn", [True, False, -1, 1.0, "1", None, 99, 3], ids=repr
    )
    def test_an_invalid_or_unfinished_turn_is_a_value_error(
        self, log: EventLog, at_turn: Any
    ) -> None:
        api = _api(log)
        cid = _stacked(api)  # two finished turns
        ref = log.put_json({"fsm_id": "x", "current_state": "chat"})
        log.append(cid, _START, frame=1, state_ref=ref)  # a third turn that never ended
        with pytest.raises(ValueError):
            api.fork(cid, at_turn=at_turn)
        assert api.list_active_conversations() == [cid]
        api.fork(cid, at_turn=2)  # the finished turns still fork

    def test_a_log_with_no_state_at_the_cut_is_refused(self, log: EventLog) -> None:
        log.append("ghost", _START, frame=0)  # genesis is empty and has no state
        with pytest.raises(FSMError, match="no recorded state"):
            _api(log).fork("ghost", at_turn=0)


# --------------------------------------------------------------
# A log is a file: forged trees
# --------------------------------------------------------------


def _sblob(fsm_id: str, state: str, internal: Any = None, **context: Any) -> Any:
    return {
        "fsm_id": fsm_id,
        "current_state": state,
        "context_data": context,
        "conversation_history": [],
        "conversation_summary": None,
        "working_memory": None,
        "hidden_buffers": [],
        "provenance": {},
        "internal": internal or {},
    }


def _forge(
    log: EventLog,
    api: API,
    *,
    child: Any = None,
    child_state: str = "hello",
    root_internal: Any = None,
    root_context: Any = None,
    delta_internal: Any = None,
    child_internal: Any = None,
    child_context: Any = None,
    return_context: Any = None,
    push: Any = None,
) -> str:
    """Write the tree ``forged``: turn 1 at the root, then (with *child*) a push and turn 2."""
    put = log.put_json
    root = put(
        _sblob(api.fsm_id, "chat", root_internal, name="Ann", **(root_context or {}))
    )
    log.append("forged", _REC, frame=0, source="start", state="chat", state_ref=root)
    log.append("forged", _START, frame=0, state_ref=root, has_user_message=True)
    log.append("forged", _END, frame=0, outcome="committed", state_ref=root)
    if delta_internal is not None:
        delta = put({"context": {}, "internal": delta_internal})
        log.append("forged", _REC, frame=0, source="u", keys=[], delta_ref=delta)
    if child is not None:
        definition, fsm_id = API.process_fsm_definition(child)
        state = put(
            _sblob(fsm_id, child_state, child_internal, **(child_context or {}))
        )
        fields: dict[str, Any] = {
            "depth": 2,
            "child_conversation_id": "kid-1",
            "fsm_id": fsm_id,
            "definition_ref": put(definition.model_dump(mode="json")),
            "return_context_ref": put(return_context or {}),
            "shared_context_keys": [],
            "preserve_history": False,
            "state_ref": state,
        }
        fields.update(push or {})
        log.append(
            "forged",
            _PUSH,
            frame=0,
            **{k: v for k, v in fields.items() if v is not _DROP},
        )
        log.append("forged", _START, frame=1, state_ref=state, has_user_message=True)
        log.append("forged", _END, frame=1, outcome="committed", state_ref=state)
    return "forged"


def _nothing_left(
    api: API, log: EventLog, writers: frozenset[str] = frozenset()
) -> None:
    """*api* holds no conversation, no temporary definition and no pending push, and the log
    has no writer but the *writers* of the source (and the hand-written ``forged`` tree)."""
    assert not api.list_active_conversations() and not api.conversation_stacks
    assert not api.fsm_manager.instances and not api.fsm_manager._conversation_locks
    assert not api._temp_fsm_definitions and not api._pending_push_ids
    assert set(log._writers) <= writers | {"forged"}


class TestForgedRoutes:
    """One RED test per route that could carry a forged driver grant into a rebuilt frame."""

    @pytest.mark.parametrize("route", ["blob", "delta", "context_data"])
    def test_a_grant_in_the_root_blob_or_a_delta_is_not_seeded(
        self, log: EventLog, route: str
    ) -> None:
        planted = {"_approval_granted": _GRANT, "_run_output_keys": ["x"]}
        api = _api(log)
        tid = _forge(
            log,
            api,
            child=_KIDS,  # turn 2 comes after the delta, so the delta is inside the cut
            root_internal=planted if route == "blob" else None,
            delta_internal=planted if route == "delta" else None,
            root_context=planted if route == "context_data" else None,
        )
        new_id, _ = api.fork(tid, at_turn=2)
        assert _raw(api, new_id)["name"] == "Ann"
        assert not _raw(api, new_id).keys() & set(planted)

    def test_a_forged_messages_key_seeds_no_grant(self, log: EventLog) -> None:
        api = _api(log)
        child = _tool_fsm(messages_key="_approval_granted")
        tid = _forge(
            log,
            api,
            child=child,
            child_state="call_model",
            child_internal={"_approval_granted": _GRANT},
        )
        new_id, _ = api.fork(tid, at_turn=2)
        assert api.get_current_state(new_id) == "call_model"
        assert "_approval_granted" not in _raw(api, _kid_id(api, new_id))

    def test_a_forged_response_format_key_seeds_no_grant(self, log: EventLog) -> None:
        api = _api(log)
        child = _structured_fsm()
        child["states"]["call_model"]["completion"]["response_format_key"] = (
            "_approval_granted"
        )
        tid = _forge(
            log,
            api,
            child=child,
            child_state="call_model",
            child_internal={"_approval_granted": _GRANT},
        )
        new_id, _ = api.fork(tid, at_turn=2)
        assert "_approval_granted" not in _raw(api, _kid_id(api, new_id))

    def test_only_the_default_transcript_survives_in_a_child(
        self, log: EventLog
    ) -> None:
        api = _api(log)
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
        new_id, _ = api.fork(tid, at_turn=2)
        raw = _raw(api, _kid_id(api, new_id))
        assert raw[_MESSAGES] == transcript and raw["keep"] == 1
        assert not raw.keys() & {"_approval_granted", "_run_output_keys", "_counter"}

    def test_a_non_default_transcript_name_of_a_child_is_not_restored(
        self, log: EventLog
    ) -> None:
        api = _api(log)
        tid = _forge(
            log,
            api,
            child=_tool_fsm(messages_key="_my_transcript"),
            child_state="call_model",
            child_internal={"_my_transcript": [{"role": "user", "content": "t"}]},
        )
        new_id, _ = api.fork(tid, at_turn=2)
        assert "_my_transcript" not in _raw(api, _kid_id(api, new_id))

    def test_a_placeholder_in_the_return_context_is_dropped_with_a_warning(
        self, log: EventLog
    ) -> None:
        api = _api(log)
        tid = _forge(
            log,
            api,
            child=_KIDS,
            return_context={"api_token": "<redacted>", "answer": 42},
        )
        with patch("fsm_llm.engine.api.forks.logger") as warned:
            new_id, _ = api.fork(tid, at_turn=2)
        assert api.conversation_stacks[new_id][1].return_context == {"answer": 42}
        assert warned.warning.call_count == 1


class TestADefinitionTheLogCannotGiveBack:
    def _blob(self, log: EventLog, tid: str) -> Any:
        return _blob_path(log, _of(log, tid, _PUSH)[0]["definition_ref"])

    @pytest.mark.parametrize(
        "drift",
        [
            "unrecorded",
            "no_ref",
            "deleted",
            "tampered",
            "wrong_fsm_id",
            "not_loadable",
        ],
    )
    def test_each_drift_is_corrupt_and_leaves_nothing_registered(
        self, log: EventLog, drift: str
    ) -> None:
        push: dict[str, Any] = {
            "unrecorded": {"definition_ref": _DROP, "definition_unrecorded": True},
            "no_ref": {"definition_ref": _DROP},
            "wrong_fsm_id": {"fsm_id": "fsm_Kids_00000000"},
            "not_loadable": {"definition_ref": log.put_json({"name": "x"})},
        }.get(drift, {})
        api = _api(log)
        tid = _forge(log, api, child=_KIDS, push=push)
        if drift == "deleted":
            self._blob(log, tid).unlink()
        elif drift == "tampered":
            path = self._blob(log, tid)
            path.write_bytes(path.read_bytes().replace(b"Kid hello", b"KID HELLO"))
        with pytest.raises(LogCorruptError):
            api.fork(tid, at_turn=2)
        _nothing_left(api, log)


class TestATeardownAtEveryStage:
    @staticmethod
    def _nth(real: Any, n: int, log_calls: list[str]) -> Any:
        def flaky(*args: Any, **kw: Any) -> Any:
            log_calls.append("call")
            if len(log_calls) == n:
                raise FSMError("injected")
            return real(*args, **kw)

        return flaky

    @pytest.mark.parametrize(
        ("target", "n"),
        [
            (("fsm_llm.engine.fsm.manager.FSMManager.set_conversation_state"), 2),
            (("fsm_llm.engine.fsm.manager.FSMManager.set_conversation_state"), 3),
            (("fsm_llm.engine.fsm.manager.FSMManager.start_conversation"), 3),
            (("fsm_llm.engine.api.forks.record_restored_frame"), 1),
            (("fsm_llm.engine.api.forks.record_restored_frame"), 2),
            (("fsm_llm.engine.api.forks.record_forked"), 1),
        ],
    )
    def test_a_failure_after_k_frames_tears_down_all_k(
        self, log: EventLog, target: str, n: int
    ) -> None:
        source = _api(log)
        cid = _stacked(source, depth=3)  # root and two children of one definition
        api = _api(log)  # no conversation of its own: nothing may stay behind
        owner, _, name = target.rpartition(".")
        real = getattr(FSMManager if owner.endswith("FSMManager") else forks, name)
        calls: list[str] = []
        with (
            patch(target, self._nth(real, n, calls)),
            pytest.raises(FSMError, match="injected"),
        ):
            api.fork(cid, at_turn=2)
        assert len(calls) == n
        _nothing_left(api, log, frozenset({cid}))
        assert api.fork(cid, at_turn=2)  # the same fork works once the fault is gone
        assert source.get_stack_depth(cid) == 3

    def test_the_trust_rule_is_one_helper(self) -> None:
        child = API.process_fsm_definition(_tool_fsm(messages_key="_approval_granted"))[
            0
        ]
        carried = {"_approval_granted": _GRANT, _MESSAGES: ["m"]}
        assert forks._frame_internal(child, carried) == {}  # declares the other name
        kept = API.process_fsm_definition(_tool_fsm())[0]
        assert forks._frame_internal(kept, carried) == {_MESSAGES: ["m"]}
        assert forks._frame_internal(kept, {"_approval_granted": _GRANT}) == {}
