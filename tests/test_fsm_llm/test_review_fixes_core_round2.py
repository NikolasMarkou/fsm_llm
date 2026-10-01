"""Review round 2, core fixes (plan 944e2692, step 18.5, D-037 to D-039).

Each class pins one accepted finding of findings/review-iter-1-pass8.md
(warnings 1 to 6, notes 8 and 10). Every test here fails on the parent
commit 7ccd722 except where its docstring says it kills a mutant instead
(the meter's clear inside the lock passes there; it exists so that moving
the clear out of the lock fails).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import fsm_llm.llm as llm_module
from fsm_llm import API, FileSessionStore, tool_exchange
from fsm_llm.classification import Classifier
from fsm_llm.constants import CONTEXT_KEY_RESPONSE_TEMPERATURE
from fsm_llm.definitions import (
    ClassificationResponseError,
    CompletionRequest,
    CompletionResponse,
    FSMError,
    LLMResponseError,
    ModelToolCall,
)
from fsm_llm.handlers import create_handler
from fsm_llm.llm import LiteLLMInterface
from tests.conftest import MockLLM2Interface
from tests.test_fsm_llm.test_completion_state import (
    _MESSAGES,
    _api,
    _calls,
    _final,
    _ScriptedLLM,
    _start,
    _to_call_model,
    _tool_fsm,
    _tool_runner,
)
from tests.test_fsm_llm.test_llm_complete import _call, _Provider, _reply
from tests.test_fsm_llm.test_review_fixes_core import (
    _INTENTS,
    _chain,
    _classifying_fsm,
    _Custom,
    _tie_fsm,
)

_USER = [{"role": "user", "content": "What is 2 + 3?"}]


class TestToolTurnNeedsAUserMessage:
    """Pass 8 warning 1: only a fully empty transcript was refused, so a tool
    exchange rebuilt after a restore (no task) was sent and its answer to
    nothing became the result."""

    def test_a_transcript_holding_only_a_tool_exchange_is_refused(self):
        llm = _ScriptedLLM(_final("answer to nothing"))
        api = _api(_tool_fsm(), llm)
        exchange = tool_exchange(None, _calls(("add", {"a": 2, "b": 3})).calls, ["5"])
        conv_id, _ = api.start_conversation(initial_context={_MESSAGES: exchange})
        _to_call_model(api, conv_id)
        with pytest.raises(LLMResponseError, match="no user message"):
            api.advance(conv_id)
        assert llm.requests == []

    def test_a_restore_after_a_crashed_tool_turn_fails_loudly(self, tmp_path):
        """The review's reproduction: a critical ``run_tools`` handler crashes
        once, the session is saved with the result but not the transcript,
        and after the restore the runner rebuilds a task-less exchange."""
        fsm = _tool_fsm()
        llm = _ScriptedLLM(_calls(("add", {"a": 2, "b": 3})), _final("never sent"))
        store = FileSessionStore(str(tmp_path))
        api = API.from_definition(fsm, llm_interface=llm, session_store=store)
        crashed: list[bool] = []

        def crash_once(context: dict[str, Any]) -> dict[str, Any]:
            if not crashed:
                crashed.append(True)
                raise RuntimeError("tool host down")
            return {}

        api.register_handler(
            create_handler("crash")
            .on_state_entry("run_tools")
            .critical()
            .with_priority(1)
            .do(crash_once)
        )
        conv_id = _start(api)
        api.advance(conv_id)  # intake -> call_model
        with pytest.raises(FSMError):
            api.advance(conv_id)  # the call is made, run_tools entry crashes
        assert len(llm.requests) == 1
        api.save_session(conv_id)
        (saved,) = list(tmp_path.iterdir())

        restored = API.from_definition(fsm, llm_interface=llm, session_store=store)
        restored.register_handler(_tool_runner())
        restored_id, _ = restored.restore_session(saved.stem)
        with pytest.raises(LLMResponseError, match="no user message"):
            for _ in range(4):
                restored.advance(restored_id)
        assert len(llm.requests) == 1  # nothing was sent after the restore

    def test_a_transcript_with_the_task_still_runs(self):
        llm = _ScriptedLLM(_calls(("add", {"a": 2, "b": 3})), _final("5"))
        api = _api(_tool_fsm(), llm)
        conv_id = _start(api)
        api.run_until_terminal(conv_id, max_steps=6)
        assert api.get_current_state(conv_id) == "done"
        assert [m["role"] for m in llm.requests[1].messages] == [
            "system",
            "user",
            "assistant",
            "tool",
        ]


def _raising(error: BaseException) -> Any:
    def bug(request: Any) -> Any:
        raise error

    return bug


_BUGS = [
    RuntimeError("bug in a custom complete"),
    ValueError("bad payload"),
    KeyError("missing"),
    TypeError("bad type"),
    OSError("disk"),
]


class TestClassifierSoftFailsOnlyOnItsOwnErrors:
    """Pass 8 warning 2: the pipeline's soft-fail tuple still held ValueError,
    TypeError, KeyError, RuntimeError and OSError, so those bugs in a custom
    interface became a silent stay or a skipped field."""

    @pytest.mark.parametrize("fsm", [_tie_fsm, _classifying_fsm])
    @pytest.mark.parametrize("error", _BUGS, ids=lambda e: type(e).__name__)
    def test_a_bug_in_a_custom_complete_fails_the_turn(self, fsm, error):
        api = API.from_definition(fsm(), llm_interface=_Custom(_raising(error)))
        conv_id, _ = api.start_conversation()
        start_state = api.get_current_state(conv_id)
        with pytest.raises((FSMError, type(error))) as caught:
            api.converse("I want to buy it", conv_id)
        assert type(error) in _chain(caught.value)
        assert api.get_current_state(conv_id) == start_state

    @pytest.mark.parametrize("fsm", [_tie_fsm, _classifying_fsm])
    def test_an_unreadable_reply_still_fails_soft(self, fsm):
        def unreadable(request: Any) -> CompletionResponse:
            return CompletionResponse(kind="final", text="not json at all")

        api = API.from_definition(fsm(), llm_interface=_Custom(unreadable))
        conv_id, _ = api.start_conversation()
        start_state = api.get_current_state(conv_id)
        api.converse("I want to buy it", conv_id)
        assert api.get_current_state(conv_id) == start_state

    def test_the_unreadable_reply_is_a_classification_error(self):
        def unreadable(request: Any) -> CompletionResponse:
            return CompletionResponse(kind="final", text="not json at all")

        with pytest.raises(ClassificationResponseError):
            Classifier(_INTENTS, llm=_Custom(unreadable)).classify("buy")


class _Garbled:
    """A provider reply whose message raises from ``content``."""

    @property
    def content(self) -> Any:
        raise RuntimeError("garbled message object")


class TestCompleteIsTotal:
    """Pass 8 warning 3: a garbled reply raised a raw TypeError out of
    ``complete`` (and so out of the classifier) instead of LLMResponseError."""

    @pytest.mark.parametrize("tool_calls", [5, "call_1", {"id": "call_1"}, 2.5])
    def test_a_tool_calls_value_that_is_not_a_list_is_a_response_error(
        self, tool_calls
    ):
        reply = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="x", tool_calls=tool_calls)
                )
            ]
        )
        with _Provider(reply), pytest.raises(LLMResponseError, match="tool_calls"):
            LiteLLMInterface("gpt-4o").complete(CompletionRequest(messages=_USER))

    def test_an_unforeseen_reply_error_is_a_response_error(self):
        reply = SimpleNamespace(choices=[SimpleNamespace(message=_Garbled())])
        with _Provider(reply), pytest.raises(LLMResponseError, match="garbled"):
            LiteLLMInterface("gpt-4o").complete(CompletionRequest(messages=_USER))

    def test_the_classifier_turns_a_garbled_reply_into_a_classification_error(self):
        reply = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="x", tool_calls=5))
            ]
        )
        classifier = Classifier(_INTENTS, llm=LiteLLMInterface("gpt-4o"))
        with _Provider(reply), pytest.raises(FSMError) as caught:
            classifier.classify("I want to buy it")
        assert type(caught.value).__name__ == "ClassificationError"


class TestAnIdLessToolCallIsMalformed:
    """Pass 8 note 10: a call without an id was returned as a runnable call
    (id ``""``), so its tool ran before the next turn refused the transcript."""

    @pytest.mark.parametrize("call_id", [None, "", 7])
    def test_the_turn_is_malformed_with_no_calls(self, call_id):
        calls = [
            _call("add", '{"a": 2, "b": 3}', call_id="call_ok"),
            {**_call("add", '{"a": 1, "b": 1}'), "id": call_id},
        ]
        provider_reply = _reply(None, tool_calls=calls)
        # litellm mints an id for a None one; read the shape a non-litellm
        # reply object would carry instead.
        provider_reply.choices[0].message.tool_calls[1].id = call_id
        with _Provider(provider_reply):
            reply = LiteLLMInterface("gpt-4o").complete(
                CompletionRequest(messages=_USER, tools=[_ADD_TOOL])
            )
        assert reply.kind == "malformed"
        assert reply.calls == ()

    def test_a_model_tool_call_needs_an_id(self):
        with pytest.raises(ValidationError):
            ModelToolCall(id="", name="add", arguments={})


_ADD_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
    },
}


class TestApiRefusesEverySettingBesideAnInterface:
    """Pass 8 warning 4: temperature, max_tokens and model beside an injected
    interface were accepted and silently ignored."""

    @pytest.mark.parametrize(
        ("setting", "value"),
        [("temperature", 0.9), ("max_tokens", 7), ("model", "gpt-4o"), ("seed", 7)],
    )
    def test_each_setting_is_refused(self, setting, value):
        injected = LiteLLMInterface("ollama_chat/qwen3.5:4b", temperature=0.5)
        with pytest.raises(ValueError, match=f"takes no LLM settings.*{setting}"):
            API.from_definition(_tool_fsm(), llm_interface=injected, **{setting: value})

    def test_a_model_on_an_interface_without_one_is_refused(self):
        with pytest.raises(ValueError, match="model"):
            API.from_definition(
                _tool_fsm(), llm_interface=MockLLM2Interface(), model="gpt-4o"
            )

    def test_the_interface_own_model_is_accepted(self):
        injected = LiteLLMInterface("ollama_chat/qwen3.5:4b")
        api = API.from_definition(
            _tool_fsm(), llm_interface=injected, model="ollama_chat/qwen3.5:4b"
        )
        assert api.llm_interface is injected


class TestResponseTemperature:
    """D-038: a conversation's Pass-2 temperature travels on the request
    (``CONTEXT_KEY_RESPONSE_TEMPERATURE``), so it applies through an injected
    interface; the parent had no such setting (a per-sample temperature
    beside an injected interface was dropped)."""

    @staticmethod
    def _speaking_fsm() -> dict[str, Any]:
        state = {
            "description": "Talk",
            "purpose": "Talk",
            "response_instructions": "Say something",
        }
        return {
            "name": "Talk",
            "description": "A greeting, then a reply",
            "initial_state": "hello",
            "states": {
                "hello": {
                    "id": "hello",
                    **state,
                    "transitions": [{"target_state": "bye", "description": "On"}],
                },
                "bye": {"id": "bye", **state},
            },
        }

    def test_the_greeting_and_the_reply_carry_it(self):
        injected = LiteLLMInterface("gpt-4o", temperature=0.5)
        api = API.from_definition(self._speaking_fsm(), llm_interface=injected)
        with _Provider(_reply('{"message": "hi"}')) as provider:
            conv_id, _ = api.start_conversation({CONTEXT_KEY_RESPONSE_TEMPERATURE: 1.3})
            api.converse("hello", conv_id)
        assert api.get_current_state(conv_id) == "bye"
        assert len(provider.calls) == 2  # the greeting and the reply
        assert [call["temperature"] for call in provider.calls] == [1.3, 1.3]

    def test_the_stream_carries_it(self):
        request_temperatures: list[Any] = []
        original = LiteLLMInterface._build_call_params

        def spy(self: Any, messages: Any, call_type: str, **kw: Any) -> Any:
            if kw.get("stream"):
                request_temperatures.append(kw.get("temperature"))
            return original(self, messages, call_type, **kw)

        fsm = self._speaking_fsm()
        api = API.from_definition(fsm, llm_interface=LiteLLMInterface("gpt-4o"))
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))]
        )
        with (
            _Provider(_reply('{"message": "hi"}')) as provider,
            pytest.MonkeyPatch.context() as mp,
        ):
            mp.setattr(LiteLLMInterface, "_build_call_params", spy)
            conv_id, _ = api.start_conversation({CONTEXT_KEY_RESPONSE_TEMPERATURE: 0.2})
            provider.replies = [iter([chunk])]
            "".join(api.converse_stream("hello", conv_id))
        assert request_temperatures == [0.2]

    def test_absent_keeps_the_interface_temperature(self):
        api = API.from_definition(
            self._speaking_fsm(),
            llm_interface=LiteLLMInterface("gpt-4o", temperature=0.4),
        )
        with _Provider(_reply('{"message": "hi"}')) as provider:
            api.start_conversation()
        assert provider.calls[0]["temperature"] == 0.4

    @pytest.mark.parametrize("bad", [True, "hot", 3.0, -0.1])
    def test_a_bad_value_fails_the_turn(self, bad):
        api = API.from_definition(
            self._speaking_fsm(), llm_interface=LiteLLMInterface("gpt-4o")
        )
        with _Provider(_reply('{"message": "hi"}')) as provider:
            with pytest.raises((FSMError, ValidationError)):
                api.start_conversation({CONTEXT_KEY_RESPONSE_TEMPERATURE: bad})
        assert provider.calls == []


class TestMeterResetClearsInsideTheLock:
    """Pass 8 warning 6: moving ``self._counts = {}`` out of the locked block
    of ``snapshot`` (mutant M5) survived every test. Here a bump is run
    right after the snapshot releases its lock: with the clear inside the
    lock the bump lands in the fresh counts; with the clear outside, it lands
    in the harvested dict and is then thrown away. Passes on the parent; it
    exists to kill M5 deterministically (no timing)."""

    def test_a_count_made_as_the_reset_releases_its_lock_is_kept(self):
        meter = llm_module._UsageMeter()
        meter._bump("complete", calls=3)
        real = meter._lock
        armed = [True]

        class _BumpOnRelease:
            def __enter__(self) -> None:
                real.acquire()

            def __exit__(self, *exc: object) -> None:
                real.release()
                if armed[0]:
                    armed[0] = False
                    meter._bump("complete", calls=1)

        meter._lock = _BumpOnRelease()
        harvested = meter.snapshot(reset=True)
        remaining = meter.snapshot()
        assert harvested.calls == 3
        assert harvested.calls + remaining.calls == 4


class TestSecondsExemptStatesAreStatesOfTheDefinition:
    """Pass 8 note 8: an unknown (typo) state id was accepted and never
    waived anything."""

    def test_an_unknown_state_is_refused_at_call_time(self):
        llm = _ScriptedLLM(_final("5"))
        api = _api(_tool_fsm(), llm)
        conv_id = _start(api)
        with pytest.raises(ValueError, match=r"\['run_tool'\]"):
            api.run_until_terminal(
                conv_id, max_steps=5, seconds_exempt_states=("run_tool",)
            )
        with pytest.raises(ValueError, match=r"\['run_tool'\]"):
            api.run_until_terminal_stream(
                conv_id, max_steps=5, seconds_exempt_states=("run_tool",)
            )
        assert llm.requests == []
        assert api.get_current_state(conv_id) == "intake"

    def test_known_states_run(self):
        llm = _ScriptedLLM(_final("5"))
        api = _api(_tool_fsm(), llm)
        conv_id = _start(api)
        api.run_until_terminal(
            conv_id, max_steps=5, seconds_exempt_states=("run_tools", "done")
        )
        assert api.get_current_state(conv_id) == "done"

    def test_an_ended_conversation_runs_nothing_and_is_not_checked(self):
        llm = _ScriptedLLM(_final("5"))
        api = _api(_tool_fsm(), llm)
        conv_id = _start(api)
        api.run_until_terminal(conv_id, max_steps=5)
        assert (
            api.run_until_terminal(
                conv_id, max_steps=5, seconds_exempt_states=("nowhere",)
            )
            == ()
        )
