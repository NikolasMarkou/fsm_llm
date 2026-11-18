"""The seams of fork and resume (plan 5a6a1f80 step 3, roadmap row 1.4, part 1): the id-preserving
``FSMManager.start_conversation(conversation_id=)`` (D-008) and the ONE restore body
(``SessionsMixin._restore_conversation``) that ``restore_session`` shares with ``fork`` and ``resume``.

Part 2 (step 5) adds ``resume`` itself. Here: the id is kept, a live or malformed id is refused with
nothing registered, the default path still mints one uuid, the lock sequence is the same with and
without an id, and the helper tears the half-built conversation down at every stage. Counts and
sequences, never timing.
"""

from __future__ import annotations

import threading
import uuid
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from fsm_llm import API, FileSessionStore
from fsm_llm.core.definitions import FSMDefinition, FSMError
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.handlers import HandlerTiming
from fsm_llm.testing import ScriptedLLM
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
