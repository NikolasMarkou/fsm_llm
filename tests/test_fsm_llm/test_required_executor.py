"""A state kind that requires an executor, and the list form of ``executors=`` (plan baa37f40, step 3).

``KindSpec.requires_executor`` is data (default ``False``); no built-in kind sets it in this plan
step, so these rows flag the real ``synthesize`` spec for the length of one test (as the marker kind
of ``test_executors.py`` does for its own). A flagged kind with no executor, in the per-instance map
or the module table, is a named ``MissingExecutorError`` at the three sites a definition enters an
``API`` (construction, ``push_fsm``, the frame of ``fork``) and at the first Pass 1 of a bare
``FSMManager``. ``executors=`` also takes a list of executors, each carrying its ``kind``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock

import pytest

import fsm_llm
from fsm_llm import API
from fsm_llm.core import state_kinds as sk
from fsm_llm.core.definitions import FSMDefinition, FSMError, MissingExecutorError
from fsm_llm.core.names import NameCollisionError
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.engine.kinds import registry as ex
from fsm_llm.engine.turn.message_pipeline import MessagePipeline
from fsm_llm.runtime import EventLog
from tests.test_fsm_llm.support import executor_table_copy as copy_of_table
from tests.test_fsm_llm.test_event_log_forks import _api as _log_api
from tests.test_fsm_llm.test_executor_map import (
    _build_api,
    _build_manager,
    _build_pipeline,
    _llm,
)
from tests.test_fsm_llm.test_synthesize_field import _parent

_SYNTHESIZE = "synthesize"


class _Synth(ex.ConversationExecutor):
    """A stand-in executor of the synthesize kind: the conversation body, a kind of its own."""

    kind = _SYNTHESIZE

    def __init__(self) -> None:
        self.extract_calls = 0

    def extract(self, *args: Any, **kwargs: Any) -> Any:
        self.extract_calls += 1
        return super().extract(*args, **kwargs)


@pytest.fixture
def flagged(monkeypatch):
    """The synthesize kind flagged ``requires_executor`` for one test."""
    spec = replace(sk.SYNTHESIZE, requires_executor=True)
    monkeypatch.setattr(
        sk, "KINDS", tuple(spec if k.name == _SYNTHESIZE else k for k in sk.KINDS)
    )
    return spec


def _definition() -> FSMDefinition:
    return FSMDefinition(**_parent())


def _bare_api(**kw: Any) -> API:
    return API.from_definition(_definition(), llm_interface=_llm(), **kw)


def _plain_api() -> API:
    plain = {
        "name": "Plain",
        "description": "d",
        "initial_state": "a",
        "states": {
            "a": {
                "id": "a",
                "description": "x",
                "purpose": "p",
                "response_instructions": "hi",
                "transitions": [{"target_state": "b", "description": "go"}],
            },
            "b": {"id": "b", "description": "y", "purpose": "q"},
        },
    }
    return API.from_definition(FSMDefinition(**plain), llm_interface=_llm())


class TestTheFlag:
    def test_no_built_in_kind_requires_an_executor(self):
        assert [s.name for s in sk.KINDS if s.requires_executor] == []

    def test_the_default_is_false_and_the_field_is_a_bool(self):
        assert sk.KindSpec(name="x").requires_executor is False
        assert (
            replace(sk.CONVERSATION, requires_executor=True).requires_executor is True
        )

    def test_a_spec_stays_frozen(self):
        with pytest.raises(FrozenInstanceError):
            sk.CONVERSATION.requires_executor = True  # type: ignore[misc]

    def test_an_unflagged_synthesize_state_still_runs_as_conversation(self):
        """Nothing else moves: a bare API on a synthesize state builds and stays (D-002 of 2cefeb61)."""
        api = _bare_api()
        conversation_id, _ = api.start_conversation()
        api.converse("go", conversation_id)
        assert api.get_current_state(conversation_id) == "solve"


class TestTheNamedError:
    def test_it_is_an_fsm_error_and_a_value_error(self):
        assert issubclass(MissingExecutorError, FSMError)
        assert issubclass(MissingExecutorError, ValueError)

    def test_it_is_importable_from_the_package_root(self):
        assert fsm_llm.MissingExecutorError is MissingExecutorError
        assert "MissingExecutorError" in fsm_llm.__all__

    def test_it_carries_the_ids_it_names_keyword_only(self):
        error = MissingExecutorError("m", state_id="s", kind="k", details={"a": 1})
        assert (error.state_id, error.kind, error.details) == ("s", "k", {"a": 1})
        assert str(error) == "m"
        with pytest.raises(TypeError):
            MissingExecutorError("m", "s")  # type: ignore[misc]
        bare = MissingExecutorError("m")
        assert (bare.state_id, bare.kind, bare.details) == (None, None, {})

    def test_the_message_names_the_state_the_kind_and_the_remedy(self, flagged):
        with pytest.raises(MissingExecutorError) as raised:
            _bare_api()
        text = str(raised.value)
        assert "'solve'" in text and "'synthesize'" in text
        assert "executors=" in text and "register_executor" in text
        assert (raised.value.state_id, raised.value.kind) == ("solve", _SYNTHESIZE)


class TestExecutorFor:
    def test_a_flagged_kind_with_no_executor_raises_instead_of_falling_back(
        self, flagged
    ):
        state = _definition().states["solve"]
        with pytest.raises(MissingExecutorError):
            ex.executor_for(state)
        # the conversation entry of the map does not stand in for a flagged kind
        with pytest.raises(MissingExecutorError):
            ex.executor_for(state, {"conversation": ex.ConversationExecutor()})

    def test_the_per_instance_map_satisfies_it(self, flagged):
        mine = _Synth()
        state = _definition().states["solve"]
        assert ex.executor_for(state, {_SYNTHESIZE: mine}) is mine

    def test_the_module_table_satisfies_it(self, flagged, monkeypatch):
        monkeypatch.setattr(ex, "_EXECUTORS", copy_of_table())
        mine = _Synth()
        ex.register_executor(_SYNTHESIZE, mine)
        assert ex.executor_for(_definition().states["solve"]) is mine

    def test_an_unflagged_kind_keeps_the_conversation_fallback(self):
        state = _definition().states["solve"]
        assert ex.executor_for(state) is ex.executor_for(_definition().states["start"])


class TestTheSitesThatRefuse:
    def test_a_bare_api_refuses_at_construction(self, flagged):
        with pytest.raises(MissingExecutorError):
            _bare_api()

    def test_a_bare_api_refuses_with_an_unrelated_map(self, flagged):
        with pytest.raises(MissingExecutorError):
            _bare_api(executors={"conversation": ex.ConversationExecutor()})

    def test_an_api_with_the_executor_builds_and_runs(self, flagged):
        mine = _Synth()
        api = _bare_api(executors={_SYNTHESIZE: mine})
        conversation_id, _ = api.start_conversation()
        api.converse("go", conversation_id)
        api.converse("more", conversation_id)
        assert api.get_current_state(conversation_id) == "solve"
        assert mine.extract_calls >= 1

    def test_push_fsm_refuses_and_leaves_nothing_behind(self, flagged):
        api = _plain_api()
        conversation_id, _ = api.start_conversation()
        with pytest.raises(MissingExecutorError):
            api.push_fsm(conversation_id, _parent())
        assert len(api.conversation_stacks[conversation_id]) == 1
        assert api._temp_fsm_definitions == {}
        assert api._pending_push_ids == set()
        assert api.get_current_state(conversation_id) == "a"

    def test_push_fsm_accepts_it_when_the_api_holds_the_executor(self, flagged):
        api = API.from_definition(
            FSMDefinition(**_parent()),
            llm_interface=_llm(),
            executors={_SYNTHESIZE: _Synth()},
        )
        conversation_id, _ = api.start_conversation()
        api.push_fsm(conversation_id, _parent())
        assert len(api.conversation_stacks[conversation_id]) == 2

    def test_fork_refuses_a_frame_whose_kind_this_api_lacks(
        self, flagged, log: EventLog
    ):
        source = _log_api(log, executors=[_Synth()])
        conversation_id, _ = source.start_conversation()
        source.converse("one", conversation_id)
        source.push_fsm(conversation_id, _parent())
        source.converse("two", conversation_id)
        assert len(source.conversation_stacks[conversation_id]) == 2
        # the same log read by an API with no executor for the child's kind
        bare = _log_api(log)
        with pytest.raises(MissingExecutorError) as raised:
            bare.fork(conversation_id, at_turn=2)
        assert raised.value.kind == _SYNTHESIZE
        # the control: the same fork on an API that holds it
        armed = _log_api(log, executors=[_Synth()])
        new_id, state = armed.fork(conversation_id, at_turn=2)
        assert state.stack_depth == 2 and new_id != conversation_id

    def test_a_bare_manager_fails_at_the_first_pass_one(self, flagged):
        definition = _parent()
        manager = FSMManager(
            fsm_loader=lambda fsm_id: FSMDefinition(**definition),
            llm_interface=_llm(),
        )
        conversation_id, _ = manager.start_conversation("any")
        # the first turn moves `start` to `solve`: the executor of the arrival state is asked
        with pytest.raises(MissingExecutorError):
            manager.process_message(conversation_id, "go")

    def test_a_manager_with_the_executor_runs_the_turn(self, flagged):
        definition = _parent()
        mine = _Synth()
        manager = FSMManager(
            fsm_loader=lambda fsm_id: FSMDefinition(**definition),
            llm_interface=_llm(),
            executors=[mine],
        )
        conversation_id, _ = manager.start_conversation("any")
        manager.process_message(conversation_id, "go")
        manager.process_message(conversation_id, "again")
        assert mine.extract_calls == 1


class TestTheListForm:
    @pytest.mark.parametrize("build", [_build_api, _build_manager, _build_pipeline])
    def test_a_list_or_tuple_of_executors_is_accepted_by_all_three(self, build):
        build([ex.ConversationExecutor()])
        build((ex.CompletionExecutor(),))
        build([])

    def test_the_list_form_equals_the_mapping_form(self):
        one, two = ex.ConversationExecutor(), ex.CompletionExecutor()
        from_list = ex.check_executors([one, two])
        from_map = ex.check_executors({"conversation": one, "completion": two})
        assert dict(from_list) == dict(from_map)
        assert isinstance(from_list, MappingProxyType)

    def test_the_built_in_executors_name_their_kind(self):
        assert ex.ConversationExecutor.kind == "conversation"
        assert ex.CompletionExecutor.kind == "completion"
        assert ex.StateExecutor.kind is None

    def test_a_list_run_equals_a_mapping_run(self, flagged):
        results = []
        for form in (lambda e: {_SYNTHESIZE: e}, lambda e: [e]):
            mine = _Synth()
            api = _bare_api(executors=form(mine))
            conversation_id, _ = api.start_conversation()
            api.converse("go", conversation_id)
            api.converse("more", conversation_id)
            results.append(
                (
                    api.get_current_state(conversation_id),
                    api.get_data(conversation_id),
                    mine.extract_calls,
                )
            )
        assert results[0] == results[1]

    @pytest.mark.parametrize("build", [_build_api, _build_manager, _build_pipeline])
    def test_two_executors_of_one_kind_collide(self, build):
        with pytest.raises(NameCollisionError):
            build([ex.ConversationExecutor(), ex.ConversationExecutor()])

    @pytest.mark.parametrize("build", [_build_api, _build_manager, _build_pipeline])
    def test_an_executor_with_no_kind_is_refused_with_its_index(self, build):
        with pytest.raises(TypeError, match=r"executors\[1\].*kind"):
            build([ex.ConversationExecutor(), ex.StateExecutor()])

    @pytest.mark.parametrize("bad", ["", "  ", 3, None])
    def test_a_kind_that_is_not_a_name_is_refused(self, bad):
        executor = ex.StateExecutor()
        executor.kind = bad  # type: ignore[assignment]
        with pytest.raises(TypeError, match=r"executors\[0\].*kind"):
            ex.check_executors([executor])

    @pytest.mark.parametrize("build", [_build_api, _build_manager, _build_pipeline])
    def test_an_unknown_kind_in_a_list_is_a_value_error(self, build):
        executor = ex.StateExecutor()
        executor.kind = "completon"
        with pytest.raises(ValueError, match="unknown state kind 'completon'"):
            build([executor])

    @pytest.mark.parametrize("build", [_build_api, _build_manager, _build_pipeline])
    def test_an_entry_that_is_not_an_executor_is_refused(self, build):
        with pytest.raises(TypeError, match=r"executors\[0\] must be a StateExecutor"):
            build([("conversation", ex.ConversationExecutor())])
        with pytest.raises(TypeError, match=r"executors\[0\] must be a StateExecutor"):
            build([ex.ConversationExecutor])  # the class, not an instance

    @pytest.mark.parametrize("bad", ["conversation", b"x", 5, {1, 2}, object()])
    def test_a_value_that_is_neither_a_mapping_nor_a_list_is_refused(self, bad):
        with pytest.raises(TypeError, match="mapping of kind name to StateExecutor"):
            ex.check_executors(bad)

    def test_a_refused_list_leaves_the_module_table_alone(self):
        before = dict(ex._EXECUTORS.items())
        with pytest.raises(NameCollisionError):
            ex.check_executors([ex.ConversationExecutor(), ex.ConversationExecutor()])
        assert dict(ex._EXECUTORS.items()) == before

    def test_the_pipeline_holds_the_checked_copy_of_a_list(self):
        mine = _Synth()
        pipeline = MessagePipeline(
            _llm(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            executors=[mine],
        )
        assert dict(pipeline._executors) == {_SYNTHESIZE: mine}
