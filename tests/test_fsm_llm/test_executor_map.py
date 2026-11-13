"""The per-API executor map and a kind added at runtime (plan d454fd62 step 54).

Row 0.14: a state kind is a ``KindSpec`` (static registry) plus an executor
(per instance). ``API(executors={kind_name: StateExecutor})``, passed through
``FSMManager`` to ``MessagePipeline``, adds or replaces the executor of a kind
for that one ``API`` and never touches the module table of ``engine/kinds/registry.py``.
The tests below build a test-only kind, ``reply_marker``, with no edit in core:
a ``State`` subclass carries its config field, a ``KindSpec`` is appended to the
registry for the length of one test, and a small executor writes a context key
without any model call. The keyword is keyword-only and internal (not exported
from ``fsm_llm``) until the kind extension point is public (roadmap C10).

On the parent commit the keyword does not exist, so every test that passes
``executors=`` fails with ``TypeError``.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from fsm_llm import API
from fsm_llm.core import state_kinds as sk
from fsm_llm.core.definitions import (
    DataExtractionResponse,
    FSMDefinition,
    FSMInstance,
    State,
)
from fsm_llm.engine.fsm.manager import FSMManager
from fsm_llm.engine.kinds import registry as ex
from fsm_llm.engine.turn.message_pipeline import MessagePipeline
from fsm_llm.testing import MockLLM2Interface

_KIND = "reply_marker"


class _MarkerConfig(BaseModel):
    key: str = "marker"
    value: str = "set"


class _MarkerState(State):
    """A State that carries the test kind's config: no field added to core."""

    reply_marker: _MarkerConfig | None = None


_MARKER_SPEC = replace(sk.CONVERSATION, name=_KIND, config_field="reply_marker")


class _MarkerExecutor(ex.StateExecutor):
    """Writes the configured key; makes no model call; records its turns."""

    def __init__(self, value: str | None = None) -> None:
        self.value = value
        self.extract_calls = 0

    def extract(
        self,
        pipeline: Any,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        self.extract_calls += 1
        config = state.reply_marker  # type: ignore[attr-defined]
        response = DataExtractionResponse(
            extracted_data={config.key: self.value or config.value}, confidence=1.0
        )
        instance.last_extraction_response = response
        return response


class _CountingConversation(ex.ConversationExecutor):
    """The conversation executor, counting the turns it runs."""

    def __init__(self) -> None:
        self.extract_calls = 0

    def extract(self, *args: Any, **kwargs: Any) -> DataExtractionResponse:
        self.extract_calls += 1
        return super().extract(*args, **kwargs)


def _definition() -> FSMDefinition:
    marker = _MarkerState(
        id="marker",
        description="A state of the test kind",
        purpose="Hold the marker",
        response_instructions="Confirm.",
        reply_marker=_MarkerConfig(),
        transitions=[
            {
                "target_state": "done",
                "description": "The marker was written",
                "conditions": [
                    {
                        "description": "marker is set",
                        "requires_context_keys": ["marker"],
                        "logic": {"!=": [{"var": "marker"}, None]},
                    }
                ],
            }
        ],
    )
    done = State(
        id="done",
        description="End",
        purpose="End",
        response_instructions="Say goodbye.",
    )
    return FSMDefinition(
        name="MarkerFSM",
        description="A test-only kind end to end",
        initial_state="marker",
        persona="p",
        states={"marker": marker, "done": done},
    )


def _plain_definition() -> FSMDefinition:
    return FSMDefinition(
        name="Plain",
        description="Two plain states",
        initial_state="only",
        persona="p",
        states={
            "only": State(
                id="only",
                description="d",
                purpose="p",
                response_instructions="Reply.",
                transitions=[{"target_state": "end", "description": "always"}],
            ),
            "end": State(id="end", description="e", purpose="e"),
        },
    )


def _llm() -> MockLLM2Interface:
    return MockLLM2Interface(response_text="ok")


def _run(api: API) -> tuple[str, dict[str, Any]]:
    conversation_id, _ = api.start_conversation()
    api.converse("hello", conversation_id)
    return api.get_current_state(conversation_id), api.get_data(conversation_id)


@pytest.fixture
def marker_kind(monkeypatch):
    """The test kind's spec in the registry for one test; the module executor
    table must come out of the test as it went in."""
    before = dict(ex._EXECUTORS)
    monkeypatch.setattr(sk, "KINDS", (*sk.KINDS, _MARKER_SPEC))
    yield
    assert ex._EXECUTORS == before


class TestARuntimeKindEndToEnd:
    def test_a_registered_kind_runs_a_full_conversation_with_no_core_edit(
        self, marker_kind
    ):
        executor = _MarkerExecutor()
        llm = _llm()
        api = API.from_definition(
            _definition(), llm_interface=llm, executors={_KIND: executor}
        )
        state, data = _run(api)
        assert state == "done"
        assert data["marker"] == "set"
        assert executor.extract_calls == 1
        # the kind's executor replaced Pass 1: no field extraction call was made
        assert [name for name, _ in llm.call_history if name == "extract_field"] == []

    def test_the_state_reports_the_runtime_kind(self, marker_kind):
        assert _definition().states["marker"].kind == _KIND
        assert _definition().states["done"].kind == "conversation"

    def test_a_kind_with_no_executor_runs_as_conversation(self, marker_kind):
        api = API.from_definition(_definition(), llm_interface=_llm())
        state, data = _run(api)
        assert "marker" not in data
        assert state == "marker"

    def test_the_module_table_is_unchanged_by_a_per_api_map(self, marker_kind):
        before = dict(ex._EXECUTORS)
        API.from_definition(
            _definition(), llm_interface=_llm(), executors={_KIND: _MarkerExecutor()}
        )
        assert ex._EXECUTORS == before
        assert _KIND not in ex._EXECUTORS


class TestIsolation:
    def test_two_apis_with_different_maps_do_not_affect_each_other(self, marker_kind):
        first, second = _MarkerExecutor("one"), _MarkerExecutor("two")
        api_one = API.from_definition(
            _definition(), llm_interface=_llm(), executors={_KIND: first}
        )
        api_two = API.from_definition(
            _definition(), llm_interface=_llm(), executors={_KIND: second}
        )
        api_plain = API.from_definition(_definition(), llm_interface=_llm())
        assert _run(api_one)[1]["marker"] == "one"
        assert _run(api_two)[1]["marker"] == "two"
        assert "marker" not in _run(api_plain)[1]
        assert (first.extract_calls, second.extract_calls) == (1, 1)

    def test_a_per_api_executor_wins_over_the_module_table(self, marker_kind):
        ex.register_executor(_KIND, _MarkerExecutor("module"))
        try:
            from_module = API.from_definition(_definition(), llm_interface=_llm())
            own = API.from_definition(
                _definition(),
                llm_interface=_llm(),
                executors={_KIND: _MarkerExecutor("own")},
            )
            assert _run(from_module)[1]["marker"] == "module"
            assert _run(own)[1]["marker"] == "own"
        finally:
            del ex._EXECUTORS[_KIND]

    def test_a_builtin_kind_can_be_replaced_for_one_api_only(self):
        counting = _CountingConversation()
        replaced = API.from_definition(
            _plain_definition(),
            llm_interface=_llm(),
            executors={"conversation": counting},
        )
        untouched = API.from_definition(_plain_definition(), llm_interface=_llm())
        _run(replaced)
        _run(untouched)
        assert counting.extract_calls == 1
        assert ex._EXECUTORS["conversation"] is not counting

    def test_the_map_is_a_read_only_copy(self, marker_kind):
        given: dict[str, ex.StateExecutor] = {_KIND: _MarkerExecutor("one")}
        api = API.from_definition(_definition(), llm_interface=_llm(), executors=given)
        given[_KIND] = _MarkerExecutor("changed")
        given["conversation"] = _CountingConversation()
        held = api.fsm_manager._pipeline._executors
        assert isinstance(held, MappingProxyType)
        assert set(held) == {_KIND}
        assert _run(api)[1]["marker"] == "one"
        with pytest.raises(TypeError):
            held["conversation"] = _CountingConversation()  # type: ignore[index]


def _build_api(executors: Any) -> None:
    API.from_definition(_plain_definition(), llm_interface=_llm(), executors=executors)


def _build_manager(executors: Any) -> None:
    FSMManager(llm_interface=_llm(), executors=executors)


def _build_pipeline(executors: Any) -> None:
    MessagePipeline(
        _llm(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        executors=executors,
    )


_BUILDERS = [_build_api, _build_manager, _build_pipeline]


class TestRefusals:
    @pytest.mark.parametrize("build", _BUILDERS)
    def test_a_value_that_is_not_a_mapping_raises_type_error(self, build):
        with pytest.raises(TypeError, match="mapping of kind name to StateExecutor"):
            build([("conversation", ex.ConversationExecutor())])

    @pytest.mark.parametrize("build", _BUILDERS)
    @pytest.mark.parametrize(
        "bad", [ex.ConversationExecutor, object(), None.__class__, lambda: None]
    )
    def test_a_value_that_is_not_an_executor_instance_raises_type_error(
        self, build, bad
    ):
        with pytest.raises(TypeError, match=r"executors\['conversation'\]"):
            build({"conversation": bad})

    @pytest.mark.parametrize("build", _BUILDERS)
    def test_a_key_that_is_not_a_name_raises_type_error(self, build):
        with pytest.raises(TypeError, match="kind names"):
            build({1: ex.ConversationExecutor()})

    @pytest.mark.parametrize("build", _BUILDERS)
    def test_an_unknown_kind_name_raises_value_error(self, build):
        with pytest.raises(ValueError, match="unknown state kind 'completon'"):
            build({"completon": ex.ConversationExecutor()})

    @pytest.mark.parametrize("build", _BUILDERS)
    def test_none_and_an_empty_map_are_accepted(self, build):
        build(None)
        build({})

    def test_the_keyword_is_keyword_only(self):
        with pytest.raises(TypeError):
            FSMManager(  # type: ignore[misc]
                lambda fsm_id: None,
                _llm(),
                None,
                None,
                None,
                None,
                5,
                1000,
                None,
                "continue",
                64,
                {},
            )

    def test_a_refused_map_leaves_the_module_table_alone(self):
        before = dict(ex._EXECUTORS)
        with pytest.raises(ValueError):
            _build_api({"nope": ex.ConversationExecutor()})
        assert ex._EXECUTORS == before


class _RecordingConversation(ex.ConversationExecutor):
    """The conversation executor, recording the id of each state it extracts for."""

    def __init__(self) -> None:
        self.states: list[str] = []

    def extract(self, pipeline, instance, user_message, conversation_id, state):
        self.states.append(state.id)
        return super().extract(pipeline, instance, user_message, conversation_id, state)


def _synthesize_state() -> State:
    return State(
        id="solve",
        description="d",
        purpose="p",
        synthesize={"goal": "g", "return_keys": ["k"], "on_failure": "t"},
        transitions=[{"target_state": "t", "description": "go"}],
    )


class TestPassiveKindsRunAsTheInstancesConversation:
    """Review gate 1, concern 11: a kind with no executor of its own (``synthesize``,
    a runtime kind nobody registered) runs as ``conversation``, so an
    ``executors={"conversation": X}`` override reaches it too. Resolution order:
    the per-instance map for the kind, the module table for the kind, the
    per-instance ``conversation`` entry, the module ``conversation`` entry."""

    def test_a_synthesize_state_follows_the_per_api_conversation_entry(self):
        mine = ex.ConversationExecutor()
        assert ex.executor_for(_synthesize_state(), {"conversation": mine}) is mine

    def test_an_unregistered_runtime_kind_follows_it_too(self, marker_kind):
        marker = _definition().states["marker"]
        assert marker.kind == _KIND
        mine = ex.ConversationExecutor()
        assert ex.executor_for(marker, {"conversation": mine}) is mine

    def test_without_a_map_both_fall_back_to_the_module_conversation_entry(
        self, marker_kind
    ):
        module_conversation = ex._EXECUTORS["conversation"]
        assert ex.executor_for(_synthesize_state()) is module_conversation
        assert ex.executor_for(_definition().states["marker"]) is module_conversation

    def test_a_map_without_conversation_does_not_hide_the_module_entry(self):
        other = ex.CompletionExecutor()
        assert (
            ex.executor_for(_synthesize_state(), {"completion": other})
            is (ex._EXECUTORS["conversation"])
        )

    def test_the_per_api_entry_of_the_kind_itself_still_wins(self):
        own, conversation = ex.ConversationExecutor(), ex.ConversationExecutor()
        mapping = {"synthesize": own, "conversation": conversation}
        assert ex.executor_for(_synthesize_state(), mapping) is own

    def test_a_module_executor_of_the_kind_beats_the_per_api_conversation_entry(
        self, monkeypatch
    ):
        monkeypatch.setattr(ex, "_EXECUTORS", dict(ex._EXECUTORS))
        registered = ex.ConversationExecutor()
        ex.register_executor("synthesize", registered)
        mine = ex.ConversationExecutor()
        assert (
            ex.executor_for(_synthesize_state(), {"conversation": mine}) is registered
        )

    def test_a_per_api_conversation_override_reaches_a_synthesize_state_in_a_turn(self):
        from tests.test_fsm_llm.kind_corpus import _synth_fsm

        recorder = _RecordingConversation()
        api = API.from_definition(
            _synth_fsm(), llm_interface=_llm(), executors={"conversation": recorder}
        )
        conversation_id, _ = api.start_conversation()
        api.converse("hello", conversation_id)  # welcome -> solve
        assert api.get_current_state(conversation_id) == "solve"
        api.converse("anything", conversation_id)  # a bare converse on solve
        assert recorder.states == ["welcome", "solve"]

    def test_a_per_api_conversation_override_reaches_an_unregistered_kind_in_a_turn(
        self, marker_kind
    ):
        recorder = _RecordingConversation()
        api = API.from_definition(
            _definition(), llm_interface=_llm(), executors={"conversation": recorder}
        )
        _run(api)
        assert recorder.states[0] == "marker"
