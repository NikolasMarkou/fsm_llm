"""``API.advance``: one turn of the current state with no user message.

A step runs the same turn body as ``converse``. The three differences are
pinned here: no user exchange in the history, no ``[state]`` marker and no
Pass-2 LLM call for a silent state, and prompts without a user message.
"""

from __future__ import annotations

import copy
import gc
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import fsm_llm
from fsm_llm import API, AdvanceResult, FSMError, HandlerTiming, create_handler
from fsm_llm.definitions import (
    BulkExtractionRequest,
    ClassificationResult,
    DataExtractionResponse,
    FieldExtractionRequest,
    FieldExtractionResponse,
    LLMResponseError,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    TransitionEvaluationResult,
)
from fsm_llm.llm import LLMInterface
from fsm_llm.memory import WorkingMemory
from fsm_llm.session import FileSessionStore

_REPLY = "Here is the plan for your trip."


class _ScriptedLLM(LLMInterface):
    """Offline LLM: fixed field values, a fixed bulk payload, a fixed reply.

    Records every request so a test can assert which calls were made.
    """

    def __init__(
        self,
        fields: dict[str, Any] | None = None,
        bulk: dict[str, Any] | None = None,
        *,
        fail_response: bool = False,
    ) -> None:
        self.fields = dict(fields or {})
        self.bulk = dict(bulk or {})
        self.fail_response = fail_response
        self.model = "gpt-4"
        self.field_requests: list[FieldExtractionRequest] = []
        self.bulk_requests: list[BulkExtractionRequest] = []
        self.response_requests: list[ResponseGenerationRequest] = []

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        self.response_requests.append(request)
        if self.fail_response:
            raise LLMResponseError("provider down")
        return ResponseGenerationResponse(message=_REPLY)

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        self.field_requests.append(request)
        value = self.fields.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            is_valid=value is not None,
        )

    def extract_bulk_data(
        self, request: BulkExtractionRequest
    ) -> DataExtractionResponse:
        self.bulk_requests.append(request)
        return DataExtractionResponse(extracted_data=dict(self.bulk))

    def reset(self) -> None:
        self.field_requests.clear()
        self.bulk_requests.clear()
        self.response_requests.clear()


def _state(
    state_id: str,
    *,
    speaks: bool = False,
    transitions: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": state_id,
        "description": f"State {state_id}",
        "purpose": f"Purpose of {state_id}",
        "response_instructions": "Tell the user the plan." if speaks else "",
        "transitions": transitions or [],
        **extra,
    }


def _when_set(target: str, key: str, priority: int = 100) -> dict[str, Any]:
    return {
        "target_state": target,
        "description": f"Go to {target} once {key} is known",
        "priority": priority,
        "conditions": [
            {
                "description": f"{key} is set",
                "requires_context_keys": [key],
                "logic": {"!!": [{"var": key}]},
            }
        ],
    }


def _always(target: str, priority: int = 100) -> dict[str, Any]:
    return {
        "target_state": target,
        "description": f"Go to {target}",
        "priority": priority,
    }


def _trip_fsm(*, plan_speaks: bool = False, **fsm_extra: Any) -> dict[str, Any]:
    """collect (silent, needs ``city``) -> plan -> done (terminal, speaking)."""
    return {
        "name": "trip",
        "description": "Plan a trip",
        "initial_state": "collect",
        "states": {
            "collect": _state(
                "collect",
                required_context_keys=["city"],
                transitions=[_when_set("plan", "city")],
            ),
            "plan": _state("plan", speaks=plan_speaks, transitions=[_always("done")]),
            "done": _state("done", speaks=True),
        },
        **fsm_extra,
    }


def _ambiguous_fsm() -> dict[str, Any]:
    """route (silent, two tied transitions) -> billing | support (terminal)."""
    return {
        "name": "route",
        "description": "Route a request",
        "initial_state": "route",
        "states": {
            "route": _state(
                "route", transitions=[_always("billing"), _always("support")]
            ),
            "billing": _state("billing", speaks=True),
            "support": _state("support", speaks=True),
        },
    }


def _start(
    fsm: dict[str, Any],
    llm: _ScriptedLLM,
    initial_context: dict[str, Any] | None = None,
    **api_kwargs: Any,
) -> tuple[API, str]:
    api = API.from_definition(fsm, llm_interface=llm, **api_kwargs)
    conv_id, _ = api.start_conversation(initial_context)
    llm.reset()
    return api, conv_id


def _instance(api: API, conv_id: str) -> Any:
    return api.fsm_manager.instances[conv_id]


def _record_all_timings(api: API) -> list[str]:
    """Register one recording handler per timing; return the shared event list."""
    events: list[str] = []
    for timing in HandlerTiming:

        def _record(context: dict[str, Any], _name: str = timing.name) -> dict:
            events.append(_name)
            return {}

        api.register_handler(
            create_handler(f"record_{timing.name}").at(timing).do(_record)
        )
    return events


def _classifier_choosing(intent: str) -> MagicMock:
    classifier = MagicMock()
    classifier.classify.return_value = ClassificationResult(
        reasoning="mock", intent=intent, confidence=0.9
    )
    return classifier


# ---------------------------------------------------------------------------


class TestAdvanceResult:
    def test_exported_in_static_all(self):
        assert "AdvanceResult" in fsm_llm.__all__
        assert fsm_llm.AdvanceResult is AdvanceResult

    def test_is_frozen(self):
        result = AdvanceResult(
            state_before="a",
            state_after="b",
            transition_outcome=TransitionEvaluationResult.DETERMINISTIC,
            ended=False,
        )
        assert result.response is None
        with pytest.raises(ValidationError):
            result.state_after = "c"

    def test_fields_of_a_deterministic_step(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        result = api.advance(conv_id)
        assert result == AdvanceResult(
            state_before="collect",
            state_after="plan",
            transition_outcome=TransitionEvaluationResult.DETERMINISTIC,
            response=None,
            ended=False,
        )

    def test_step_into_a_terminal_state_reports_ended(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        api.advance(conv_id)
        result = api.advance(conv_id)
        assert (result.state_before, result.state_after) == ("plan", "done")
        assert result.ended is True
        assert result.response == _REPLY
        assert api.has_conversation_ended(conv_id)

    def test_blocked_step_stays(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM())
        result = api.advance(conv_id)
        assert result.transition_outcome is TransitionEvaluationResult.BLOCKED
        assert (result.state_before, result.state_after) == ("collect", "collect")
        assert result.ended is False


class TestAdvanceHandlerParity:
    """The handler-timing sequence of a step equals that of a converse turn."""

    def _events(self, fsm: dict[str, Any], fields: dict[str, Any], *, step: bool):
        llm = _ScriptedLLM(fields)
        api = API.from_definition(fsm, llm_interface=llm)
        events = _record_all_timings(api)
        conv_id, _ = api.start_conversation()
        assert events == ["START_CONVERSATION"]
        events.clear()
        if step:
            api.advance(conv_id)
        else:
            api.converse("go on", conv_id)
        return events, api.get_current_state(conv_id)

    def test_deterministic(self):
        step, step_state = self._events(_trip_fsm(), {"city": "Paris"}, step=True)
        turn, turn_state = self._events(_trip_fsm(), {"city": "Paris"}, step=False)
        assert step == turn
        assert step_state == turn_state == "plan"
        assert step == [
            "PRE_PROCESSING",
            "CONTEXT_UPDATE",
            "PRE_TRANSITION",
            "POST_TRANSITION",
            "POST_PROCESSING",
        ]

    def test_blocked_never_fires_pre_transition(self):
        fsm = _trip_fsm()
        fsm["states"]["collect"]["required_context_keys"] = ["city", "days"]
        fsm["states"]["collect"]["transitions"] = [_when_set("plan", "days")]
        step, step_state = self._events(fsm, {"city": "Paris"}, step=True)
        turn, turn_state = self._events(fsm, {"city": "Paris"}, step=False)
        assert step == turn
        assert step_state == turn_state == "collect"
        assert step == ["PRE_PROCESSING", "CONTEXT_UPDATE", "POST_PROCESSING"]
        assert "PRE_TRANSITION" not in step

    def test_ambiguous(self):
        with patch("fsm_llm.pipeline.Classifier") as classifier_cls:
            classifier_cls.return_value = _classifier_choosing("billing")
            step, step_state = self._events(_ambiguous_fsm(), {}, step=True)
            turn, turn_state = self._events(_ambiguous_fsm(), {}, step=False)
        assert step == turn
        assert step_state == turn_state == "billing"
        assert step == [
            "PRE_PROCESSING",
            "PRE_TRANSITION",
            "POST_TRANSITION",
            "POST_PROCESSING",
        ]

    def test_ambiguous_outcome_and_message_free_classifier_call(self):
        api, conv_id = _start(_ambiguous_fsm(), _ScriptedLLM())
        with patch("fsm_llm.pipeline.Classifier") as classifier_cls:
            classifier = _classifier_choosing("support")
            classifier_cls.return_value = classifier
            result = api.advance(conv_id)
        assert result.transition_outcome is TransitionEvaluationResult.AMBIGUOUS
        assert result.state_after == "support"
        assert classifier.classify.call_args.args == (None,)


class TestAdvanceHistory:
    def test_silent_state_adds_nothing(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        before = api.get_conversation_history(conv_id)

        result = api.advance(conv_id)

        assert result.state_after == "plan"
        assert result.response is None
        assert api.get_conversation_history(conv_id) == before
        assert llm.response_requests == []

    def test_no_user_exchange_and_no_marker_over_a_whole_run(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        before = api.get_conversation_history(conv_id)

        api.advance(conv_id)
        api.advance(conv_id)

        added = api.get_conversation_history(conv_id)[len(before) :]
        assert added == [{"system": _REPLY}]
        assert all("user" not in entry for entry in added)

    def test_speaking_state_reply_is_the_system_message(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(plan_speaks=True), llm)

        result = api.advance(conv_id)

        assert result.response == _REPLY
        assert api.get_conversation_history(conv_id)[-1] == {"system": _REPLY}
        (request,) = llm.response_requests
        assert request.user_message == ""
        assert "<user_message>" not in request.system_prompt


class TestAdvanceExtraction:
    def test_field_prompt_has_no_user_message(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        api.advance(conv_id)
        (request,) = llm.field_requests
        assert request.field_name == "city"
        assert request.user_message == ""
        assert "User message:" not in request.system_prompt

    def test_skip_if_set(self):
        llm = _ScriptedLLM({"city": "Rome"})
        api, conv_id = _start(_trip_fsm(), llm, {"city": "Paris"})
        result = api.advance(conv_id)
        assert llm.field_requests == []
        assert result.state_after == "plan"
        assert api.get_data(conv_id)["city"] == "Paris"

    def test_handler_only_keys_are_never_extracted(self):
        fsm = _trip_fsm(handler_only_keys=["is_admin"])
        fsm["states"]["collect"]["required_context_keys"] = ["city", "is_admin"]
        llm = _ScriptedLLM({"city": "Paris", "is_admin": True})
        api, conv_id = _start(fsm, llm)
        api.advance(conv_id)
        assert [r.field_name for r in llm.field_requests] == ["city"]
        assert "is_admin" not in api.get_data(conv_id)

    def test_provenance_recorded(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        api.advance(conv_id)
        metadata = api.fsm_manager.get_complete_conversation(conv_id)["metadata"]
        assert "city" in metadata["_pipeline_extracted"]

    def test_bulk_prompt_is_built_from_context(self):
        fsm = _trip_fsm()
        collect = fsm["states"]["collect"]
        collect["required_context_keys"] = []
        collect["extraction_instructions"] = "Extract the destination city."
        llm = _ScriptedLLM(bulk={"city": "Reykjavik"})
        api, conv_id = _start(
            fsm, llm, {"note": "wants to see a volcano", "api_key": "sk-secret-123456"}
        )

        result = api.advance(conv_id)

        (request,) = llm.bulk_requests
        assert request.user_message == ""
        assert "There is no user message" in request.system_prompt
        assert "User message:" not in request.system_prompt
        assert "wants to see a volcano" in request.system_prompt
        assert "sk-secret-123456" not in request.system_prompt
        assert api.get_data(conv_id)["city"] == "Reykjavik"
        assert result.state_after == "plan"


class TestAdvanceBulkNeverOverwrites:
    """Decision D-023 end to end: with no message the additive bulk pass fills
    only unset keys, never overwrites a set value, reports no correction."""

    def _fsm(self) -> dict[str, Any]:
        fsm = _trip_fsm()
        collect = fsm["states"]["collect"]
        collect["extraction_instructions"] = "Extract the city and the budget."
        collect["transitions"] = [_when_set("plan", "approved")]
        return fsm

    def _after_first_step(self) -> tuple[API, str, _ScriptedLLM]:
        llm = _ScriptedLLM({"city": "Paris"}, bulk={"budget": "low"})
        api, conv_id = _start(self._fsm(), llm)
        api.advance(conv_id)
        data = api.get_data(conv_id)
        assert (data["city"], data["budget"]) == ("Paris", "low")
        # The model now re-reads the context and proposes other values.
        llm.bulk = {"city": "Rome", "budget": "high", "season": "spring"}
        llm.reset()
        return api, conv_id, llm

    def test_step_fills_unset_keys_only(self):
        api, conv_id, llm = self._after_first_step()

        result = api.advance(conv_id)

        assert len(llm.bulk_requests) == 1
        data = api.get_data(conv_id)
        assert data["city"] == "Paris"
        assert data["budget"] == "low"
        assert data["season"] == "spring"
        extraction = _instance(api, conv_id).last_extraction_response
        assert extraction.extracted_data == {"season": "spring"}
        assert extraction.rejected_corrections == {}
        assert result.transition_outcome is TransitionEvaluationResult.BLOCKED

    def test_a_user_message_still_corrects_the_same_key(self):
        # Same fixture through converse: the pipeline-extracted key IS
        # overwritten, so the step test above exercises the no-message rule.
        api, conv_id, _ = self._after_first_step()
        api.converse("Rome actually", conv_id)
        assert api.get_data(conv_id)["city"] == "Rome"


class TestAdvanceAtomicity:
    def _snapshot(self, api: API, conv_id: str) -> dict[str, Any]:
        instance = _instance(api, conv_id)
        return {
            "state": instance.current_state,
            "data": dict(instance.context.data),
            "metadata": dict(instance.context.metadata),
            "memory": instance.context.working_memory.to_dict(),
            "history": list(instance.context.conversation.exchanges),
        }

    def _prepared(self, llm: _ScriptedLLM) -> tuple[API, str]:
        api, conv_id = _start(_trip_fsm(plan_speaks=True), llm)
        instance = _instance(api, conv_id)
        instance.context.working_memory = WorkingMemory()
        instance.context.working_memory.set("core", "seen", "before")

        def _touch_everything(context: dict[str, Any]) -> dict[str, Any]:
            instance.context.working_memory.set("core", "seen", "during")
            instance.context.metadata["touched"] = True
            return {"entered_plan": True}

        api.register_handler(
            create_handler("touch")
            .at(HandlerTiming.POST_TRANSITION)
            .do(_touch_everything)
        )
        return api, conv_id

    def test_pass_2_failure_restores_the_whole_step(self):
        llm = _ScriptedLLM({"city": "Paris"}, fail_response=True)
        api, conv_id = self._prepared(llm)
        before = self._snapshot(api, conv_id)

        with pytest.raises(LLMResponseError):
            api.advance(conv_id)

        assert len(llm.response_requests) == 1
        assert self._snapshot(api, conv_id) == before
        assert before["state"] == "collect"

        llm.fail_response = False
        assert api.advance(conv_id).state_after == "plan"

    def test_handler_failure_restores_the_whole_step(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = self._prepared(llm)

        def _boom(context: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("post-processing exploded")

        api.register_handler(
            create_handler("boom")
            .at(HandlerTiming.POST_PROCESSING)
            .critical()
            .do(_boom)
        )
        before = self._snapshot(api, conv_id)

        with pytest.raises(FSMError, match="post-processing exploded"):
            api.advance(conv_id)

        assert self._snapshot(api, conv_id) == before
        assert llm.response_requests == []

    def test_failed_step_fires_error_handlers(self):
        llm = _ScriptedLLM({"city": "Paris"}, fail_response=True)
        api, conv_id = self._prepared(llm)
        seen: list[str] = []
        api.register_handler(
            create_handler("on_error")
            .at(HandlerTiming.ERROR)
            .do(lambda context: seen.append(context["_error"]) or {})
        )
        with pytest.raises(LLMResponseError):
            api.advance(conv_id)
        assert seen == ["provider down"]

    def test_failed_step_pops_no_unrelated_history_entry(self):
        llm = _ScriptedLLM({"city": "Paris"}, fail_response=True)
        api, conv_id = self._prepared(llm)
        # A bare user entry left by an earlier turn is the last exchange.
        _instance(api, conv_id).context.conversation.add_user_message("earlier")
        before = self._snapshot(api, conv_id)
        assert before["history"][-1] == {"user": "earlier"}

        with pytest.raises(LLMResponseError):
            api.advance(conv_id)

        assert self._snapshot(api, conv_id)["history"] == before["history"]

    def test_unexpected_error_is_wrapped_as_fsm_error(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        with patch.object(
            api.fsm_manager._pipeline, "advance", side_effect=RuntimeError("bug")
        ):
            with pytest.raises(FSMError, match="Failed to advance conversation: bug"):
                api.advance(conv_id)
        assert api.advance(conv_id).state_after == "plan"


class TestAdvanceGuards:
    def test_unknown_conversation_raises_value_error(self):
        api, _ = _start(_trip_fsm(), _ScriptedLLM())
        with pytest.raises(ValueError, match="Unknown conversation ID"):
            api.advance("nope")

    def test_terminal_state_raises(self):
        api, conv_id = _start(_trip_fsm(), _ScriptedLLM({"city": "Paris"}))
        api.advance(conv_id)
        api.advance(conv_id)
        history = api.get_conversation_history(conv_id)
        with pytest.raises(FSMError, match="Conversation has ended"):
            api.advance(conv_id)
        assert api.get_conversation_history(conv_id) == history

    def test_reentrant_call_from_a_handler_raises(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        caught: list[BaseException] = []

        def _reenter(context: dict[str, Any]) -> dict[str, Any]:
            try:
                api.advance(conv_id)
            except FSMError as e:
                caught.append(e)
            return {}

        api.register_handler(
            create_handler("reenter").at(HandlerTiming.PRE_PROCESSING).do(_reenter)
        )

        result = api.advance(conv_id)

        assert len(caught) == 1
        assert "already being processed" in str(caught[0])
        assert result.state_after == "plan"
        assert len(llm.field_requests) == 1

    def test_concurrent_call_raises(self):
        inside = threading.Event()
        release = threading.Event()

        class _BlockingLLM(_ScriptedLLM):
            def extract_field(
                self, request: FieldExtractionRequest
            ) -> FieldExtractionResponse:
                inside.set()
                assert release.wait(timeout=10)
                return super().extract_field(request)

        api, conv_id = _start(_trip_fsm(), _BlockingLLM({"city": "Paris"}))
        results: list[AdvanceResult] = []
        worker = threading.Thread(target=lambda: results.append(api.advance(conv_id)))
        worker.start()
        try:
            assert inside.wait(timeout=10)
            with pytest.raises(FSMError, match="already being processed"):
                api.advance(conv_id)
            with pytest.raises(FSMError, match="already being processed"):
                api.converse("hello", conv_id)
        finally:
            release.set()
            worker.join(timeout=10)
        assert [r.state_after for r in results] == ["plan"]


class TestAdvanceStack:
    def test_step_runs_the_pushed_sub_fsm(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        api.advance(conv_id)
        assert api.get_current_state(conv_id) == "plan"

        api.push_fsm(conv_id, _ambiguous_fsm())
        with patch("fsm_llm.pipeline.Classifier") as classifier_cls:
            classifier_cls.return_value = _classifier_choosing("billing")
            result = api.advance(conv_id)

        assert (result.state_before, result.state_after) == ("route", "billing")
        assert result.ended is True

        api.pop_fsm(conv_id)
        assert api.get_current_state(conv_id) == "plan"
        result = api.advance(conv_id)
        assert (result.state_before, result.state_after) == ("plan", "done")


class TestAdvanceSession:
    def test_auto_save_after_a_step(self, tmp_path):
        store = FileSessionStore(tmp_path)
        api, conv_id = _start(
            _trip_fsm(), _ScriptedLLM({"city": "Paris"}), session_store=store
        )
        assert store.load(conv_id) is None

        api.advance(conv_id)

        saved = store.load(conv_id)
        assert saved is not None
        assert saved.current_state == "plan"
        assert saved.context_data["city"] == "Paris"

    def test_failed_step_does_not_save(self, tmp_path):
        store = FileSessionStore(tmp_path)
        api, conv_id = _start(
            _trip_fsm(plan_speaks=True),
            _ScriptedLLM({"city": "Paris"}, fail_response=True),
            session_store=store,
        )
        with pytest.raises(LLMResponseError):
            api.advance(conv_id)
        assert store.load(conv_id) is None


class TestConverseOnASilentState:
    """Plan 07ad3f8c step 16 (D-029, D-037): a ``converse`` turn that ends on
    a silent state keeps the user exchange, makes no Pass-2 call, returns the
    empty string and records no ``[state]`` marker."""

    def test_converse_returns_no_text_makes_no_call_and_keeps_the_user_turn(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        before = api.get_conversation_history(conv_id)

        reply = api.converse("I want to go to Paris", conv_id)

        assert api.get_current_state(conv_id) == "plan"
        assert reply == ""
        added = api.get_conversation_history(conv_id)[len(before) :]
        assert added == [{"user": "I want to go to Paris"}]
        assert llm.response_requests == []

    def test_converse_and_advance_interleave(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        api.converse("Paris please", conv_id)
        result = api.advance(conv_id)
        assert (result.state_before, result.state_after) == ("plan", "done")
        assert api.get_conversation_history(conv_id)[-2:] == [
            {"user": "Paris please"},
            {"system": _REPLY},
        ]

    def test_failed_converse_still_pops_its_user_message(self):
        llm = _ScriptedLLM({"city": "Paris"}, fail_response=True)
        api, conv_id = _start(_trip_fsm(plan_speaks=True), llm)
        before = api.get_conversation_history(conv_id)
        with pytest.raises(LLMResponseError):
            api.converse("Paris", conv_id)
        assert api.get_conversation_history(conv_id) == before


class TestStepResponsePrompt:
    """Plan 07ad3f8c step 13 (D-035): a step's Pass-2 prompt has the
    no-message wording; a ``converse`` turn and the greeting keep theirs."""

    _ACK = "Acknowledge this transition naturally."

    def test_advance_reply_prompt_asks_for_no_acknowledgement(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(plan_speaks=True), llm)

        result = api.advance(conv_id)

        assert (result.state_before, result.state_after) == ("collect", "plan")
        (request,) = llm.response_requests
        assert "<current_state>plan</current_state>" in request.system_prompt
        assert "No user message was sent on this step." in request.system_prompt
        assert "cknowledge" not in request.system_prompt
        assert "<transition_info>" not in request.system_prompt
        # The request model still carries an empty string (D-015).
        assert request.user_message == ""

    def test_converse_reply_prompt_keeps_the_conversational_wording(self):
        llm = _ScriptedLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(plan_speaks=True), llm)

        api.converse("I want to go to Paris", conv_id)

        (request,) = llm.response_requests
        assert "<current_state>plan</current_state>" in request.system_prompt
        assert self._ACK in request.system_prompt
        assert "No user message was sent" not in request.system_prompt

    def test_greeting_prompt_keeps_the_conversational_wording(self):
        llm = _ScriptedLLM()
        fsm = _trip_fsm()
        fsm["states"]["collect"]["response_instructions"] = "Ask for the city."
        api = API.from_definition(fsm, llm_interface=llm)

        api.start_conversation()

        (request,) = llm.response_requests
        assert "conversational AI system" in request.system_prompt
        assert "No user message was sent" not in request.system_prompt


# ---------------------------------------------------------------------------
# advance_stream
# ---------------------------------------------------------------------------

_CHUNKS = ["Here is ", "the plan ", "for your trip."]


class _StreamingLLM(_ScriptedLLM):
    """``_ScriptedLLM`` whose Pass-2 stream yields ``_CHUNKS`` one by one.

    ``fail_after`` raises ``LLMResponseError`` after that many chunks.
    """

    def __init__(
        self,
        fields: dict[str, Any] | None = None,
        *,
        fail_after: int | None = None,
    ) -> None:
        super().__init__(fields)
        self.fail_after = fail_after
        self.stream_requests: list[ResponseGenerationRequest] = []

    def generate_response_stream(self, request: ResponseGenerationRequest):
        self.stream_requests.append(request)
        for index, chunk in enumerate(_CHUNKS):
            if self.fail_after is not None and index == self.fail_after:
                raise LLMResponseError("stream broke")
            yield chunk


def _full_snapshot(api: API, conv_id: str) -> dict[str, Any]:
    instance = _instance(api, conv_id)
    return {
        "state": instance.current_state,
        "data": copy.deepcopy(instance.context.data),
        "metadata": copy.deepcopy(instance.context.metadata),
        "memory": instance.context.working_memory.to_dict(),
        "history": list(instance.context.conversation.exchanges),
    }


def _prepared_stream(llm: _StreamingLLM, **api_kwargs: Any) -> tuple[API, str]:
    """Trip FSM with a speaking ``plan`` state, working memory, and a
    POST_TRANSITION handler that writes data, metadata and working memory."""
    api, conv_id = _start(_trip_fsm(plan_speaks=True), llm, **api_kwargs)
    instance = _instance(api, conv_id)
    instance.context.working_memory = WorkingMemory()
    instance.context.working_memory.set("core", "seen", "before")

    def _touch_everything(context: dict[str, Any]) -> dict[str, Any]:
        instance.context.working_memory.set("core", "seen", "during")
        instance.context.metadata["touched"] = True
        return {"entered_plan": True}

    api.register_handler(
        create_handler("touch").at(HandlerTiming.POST_TRANSITION).do(_touch_everything)
    )
    return api, conv_id


class TestAdvanceStream:
    # -- lazy lock ---------------------------------------------------------

    def test_no_lock_and_no_work_before_the_first_next(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)
        manager = api.fsm_manager

        stream = api.advance_stream(conv_id)

        assert conv_id not in manager._active_turns
        assert llm.field_requests == []
        assert api.get_current_state(conv_id) == "collect"
        # Another thread can take the conversation lock: nobody holds it.
        acquired: list[bool] = []
        lock = manager._conversation_locks[conv_id]
        worker = threading.Thread(
            target=lambda: (
                acquired.append(lock.acquire(blocking=False))
                or (acquired[-1] and lock.release())
            )
        )
        worker.start()
        worker.join(timeout=10)
        assert acquired == [True]
        # A never-iterated stream does not block a later step.
        assert api.advance(conv_id).state_after == "plan"
        stream.close()

    def test_manager_stream_takes_no_lock_before_the_first_next(self):
        # `API.advance_stream` defers the manager call to its own closure, so
        # the manager's laziness is pinned on the manager entry itself.
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)
        manager = api.fsm_manager

        stream = manager.advance_stream(conv_id)

        assert conv_id not in manager._active_turns
        assert llm.field_requests == []
        assert api.advance(conv_id).state_after == "plan"
        del stream
        gc.collect()
        assert api.advance(conv_id).state_after == "done"

    def test_open_stream_refuses_every_other_turn(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)
        stream = api.advance_stream(conv_id)

        assert next(stream) == _CHUNKS[0]

        assert conv_id in api.fsm_manager._active_turns
        with pytest.raises(FSMError, match="already being processed"):
            api.advance(conv_id)
        with pytest.raises(FSMError, match="already being processed"):
            api.converse("hello", conv_id)
        with pytest.raises(FSMError, match="already being processed"):
            next(api.advance_stream(conv_id))
        with pytest.raises(FSMError, match="already being processed"):
            next(api.converse_stream("hello", conv_id))

        assert list(stream) == _CHUNKS[1:]
        assert conv_id not in api.fsm_manager._active_turns
        assert api.advance(conv_id).state_after == "done"

    def test_concurrent_thread_is_refused_during_an_open_stream(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)
        stream = api.advance_stream(conv_id)
        next(stream)
        errors: list[BaseException] = []

        def _other_thread() -> None:
            try:
                api.advance(conv_id)
            except FSMError as e:
                errors.append(e)

        worker = threading.Thread(target=_other_thread)
        worker.start()
        worker.join(timeout=10)
        assert len(errors) == 1
        assert "already being processed" in str(errors[0])
        stream.close()

    # -- abandonment -------------------------------------------------------

    def test_closed_stream_releases_the_lock_and_keeps_the_partial_reply(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)
        before = api.get_conversation_history(conv_id)
        stream = api.advance_stream(conv_id)
        assert next(stream) == _CHUNKS[0]

        stream.close()

        assert conv_id not in api.fsm_manager._active_turns
        assert api.get_current_state(conv_id) == "plan"
        assert api.get_conversation_history(conv_id) == [
            *before,
            {"system": _CHUNKS[0]},
        ]
        result = api.advance(conv_id)
        assert (result.state_before, result.state_after) == ("plan", "done")

    def test_garbage_collected_stream_releases_the_lock(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)
        stream = api.advance_stream(conv_id)
        next(stream)
        assert conv_id in api.fsm_manager._active_turns

        del stream
        gc.collect()

        assert conv_id not in api.fsm_manager._active_turns
        assert api.advance(conv_id).state_after == "done"

    # -- atomicity ---------------------------------------------------------

    def test_mid_stream_failure_restores_the_whole_step(self):
        llm = _StreamingLLM({"city": "Paris"}, fail_after=1)
        api, conv_id = _prepared_stream(llm)
        before = _full_snapshot(api, conv_id)
        assert before["state"] == "collect"
        seen: list[str] = []
        stream = api.advance_stream(conv_id)

        with pytest.raises(LLMResponseError, match="stream broke"):
            for chunk in stream:
                seen.append(chunk)

        assert seen == _CHUNKS[:1]
        assert _full_snapshot(api, conv_id) == before
        assert conv_id not in api.fsm_manager._active_turns

        llm.fail_after = None
        assert list(api.advance_stream(conv_id)) == _CHUNKS
        assert api.get_current_state(conv_id) == "plan"
        after = _full_snapshot(api, conv_id)
        assert after["data"]["entered_plan"] is True
        assert after["metadata"]["touched"] is True
        assert after["memory"] != before["memory"]

    def test_post_processing_failure_restores_the_whole_step(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)

        def _boom(context: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("post-processing exploded")

        api.register_handler(
            create_handler("boom")
            .at(HandlerTiming.POST_PROCESSING)
            .critical()
            .do(_boom)
        )
        before = _full_snapshot(api, conv_id)

        with pytest.raises(FSMError, match="post-processing exploded"):
            list(api.advance_stream(conv_id))

        assert _full_snapshot(api, conv_id) == before
        assert llm.stream_requests == []

    def test_failed_stream_pops_no_unrelated_history_entry(self):
        llm = _StreamingLLM({"city": "Paris"}, fail_after=0)
        api, conv_id = _prepared_stream(llm)
        _instance(api, conv_id).context.conversation.add_user_message("earlier")
        before = _full_snapshot(api, conv_id)
        assert before["history"][-1] == {"user": "earlier"}

        with pytest.raises(LLMResponseError):
            list(api.advance_stream(conv_id))

        assert _full_snapshot(api, conv_id)["history"] == before["history"]

    def test_closed_stream_pops_no_unrelated_history_entry(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        _instance(api, conv_id).context.conversation.add_user_message("earlier")
        calls: list[str] = []
        original = api.fsm_manager._pipeline.advance_stream

        def _yield_then_run(instance: Any, conversation_id: str):
            calls.append(conversation_id)
            yield "early"
            yield from original(instance, conversation_id)

        with patch.object(api.fsm_manager._pipeline, "advance_stream", _yield_then_run):
            stream = api.advance_stream(conv_id)
            assert next(stream) == "early"
            stream.close()

        assert len(calls) == 1
        assert api.get_conversation_history(conv_id)[-1] == {"user": "earlier"}

    def test_failed_stream_fires_error_handlers(self):
        llm = _StreamingLLM({"city": "Paris"}, fail_after=1)
        api, conv_id = _prepared_stream(llm)
        errors: list[str] = []
        api.register_handler(
            create_handler("on_error")
            .at(HandlerTiming.ERROR)
            .do(lambda context: errors.append(context["_error"]) or {})
        )

        with pytest.raises(LLMResponseError):
            list(api.advance_stream(conv_id))

        assert errors == ["stream broke"]

    def test_closed_stream_fires_no_error_handler(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)
        errors: list[str] = []
        api.register_handler(
            create_handler("on_error")
            .at(HandlerTiming.ERROR)
            .do(lambda context: errors.append(context["_error"]) or {})
        )
        stream = api.advance_stream(conv_id)
        next(stream)
        stream.close()
        assert errors == []

    def test_unexpected_error_is_wrapped_as_fsm_error(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        with patch.object(
            api.fsm_manager._pipeline,
            "advance_stream",
            side_effect=RuntimeError("bug"),
        ):
            with pytest.raises(FSMError, match="Failed to advance conversation: bug"):
                list(api.advance_stream(conv_id))
        assert list(api.advance_stream(conv_id)) == []
        assert api.get_current_state(conv_id) == "plan"

    # -- silent state ------------------------------------------------------

    def test_silent_state_yields_nothing_and_calls_no_llm(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        before = api.get_conversation_history(conv_id)

        chunks = list(api.advance_stream(conv_id))

        assert chunks == []
        assert api.get_current_state(conv_id) == "plan"
        assert api.has_conversation_ended(conv_id) is False
        assert llm.stream_requests == []
        assert llm.response_requests == []
        assert api.get_conversation_history(conv_id) == before
        assert len(llm.field_requests) == 1

    def test_converse_stream_on_a_silent_state_yields_nothing(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        before = api.get_conversation_history(conv_id)

        chunks = list(api.converse_stream("Paris please", conv_id))

        assert api.get_current_state(conv_id) == "plan"
        assert chunks == []
        assert api.get_conversation_history(conv_id)[len(before) :] == [
            {"user": "Paris please"},
        ]
        assert llm.stream_requests == []
        assert llm.response_requests == []

    # -- handler parity ----------------------------------------------------

    def _events(self, *, step: bool) -> tuple[list[str], str, list[str]]:
        llm = _StreamingLLM({"city": "Paris"})
        api = API.from_definition(_trip_fsm(plan_speaks=True), llm_interface=llm)
        events = _record_all_timings(api)
        conv_id, _ = api.start_conversation()
        events.clear()
        if step:
            chunks = list(api.advance_stream(conv_id))
        else:
            chunks = list(api.converse_stream("go on", conv_id))
        return events, api.get_current_state(conv_id), chunks

    def test_handler_sequence_equals_converse_stream(self):
        step, step_state, step_chunks = self._events(step=True)
        turn, turn_state, turn_chunks = self._events(step=False)
        assert step == turn
        assert step_state == turn_state == "plan"
        assert step_chunks == turn_chunks == _CHUNKS
        assert step == [
            "PRE_PROCESSING",
            "CONTEXT_UPDATE",
            "PRE_TRANSITION",
            "POST_TRANSITION",
            "POST_PROCESSING",
        ]

    def test_handler_sequence_equals_advance(self):
        llm = _StreamingLLM({"city": "Paris"})
        api = API.from_definition(_trip_fsm(plan_speaks=True), llm_interface=llm)
        events = _record_all_timings(api)
        conv_id, _ = api.start_conversation()
        events.clear()
        api.advance(conv_id)
        assert events == self._events(step=True)[0]

    # -- history and prompts -----------------------------------------------

    def test_speaking_step_stores_the_full_reply_as_one_system_message(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)
        before = api.get_conversation_history(conv_id)

        chunks = list(api.advance_stream(conv_id))

        assert chunks == _CHUNKS
        history = api.get_conversation_history(conv_id)
        assert history == [*before, {"system": "".join(_CHUNKS)}]
        assert not any("user" in exchange for exchange in history)

    def test_stream_request_carries_no_user_message(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)

        list(api.advance_stream(conv_id))

        (request,) = llm.stream_requests
        assert request.user_message == ""
        assert "Continue" not in request.system_prompt
        (field_request,) = llm.field_requests
        assert "User message" not in field_request.system_prompt

    def test_whole_run_by_streamed_steps(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        before = api.get_conversation_history(conv_id)

        assert list(api.advance_stream(conv_id)) == []
        assert api.has_conversation_ended(conv_id) is False
        assert list(api.advance_stream(conv_id)) == _CHUNKS
        assert api.has_conversation_ended(conv_id) is True

        assert api.get_conversation_history(conv_id) == [
            *before,
            {"system": "".join(_CHUNKS)},
        ]

    # -- guards ------------------------------------------------------------

    def test_unknown_conversation_raises_at_call_time(self):
        api, _ = _start(_trip_fsm(), _StreamingLLM())
        with pytest.raises(ValueError, match="Unknown conversation ID"):
            api.advance_stream("nope")

    def test_manager_unknown_conversation_raises_at_call_time(self):
        api, _ = _start(_trip_fsm(), _StreamingLLM())
        with pytest.raises(FSMError, match="not found"):
            api.fsm_manager.advance_stream("nope")

    def test_terminal_state_raises_at_first_next(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        list(api.advance_stream(conv_id))
        list(api.advance_stream(conv_id))
        assert api.has_conversation_ended(conv_id) is True
        history = api.get_conversation_history(conv_id)

        stream = api.advance_stream(conv_id)
        with pytest.raises(FSMError, match="Conversation has ended"):
            next(stream)

        assert api.get_conversation_history(conv_id) == history
        assert conv_id not in api.fsm_manager._active_turns

    def test_pipeline_entry_refuses_a_terminal_state(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        api.advance(conv_id)
        api.advance(conv_id)
        llm.reset()
        pipeline = api.fsm_manager._pipeline
        with pytest.raises(FSMError, match="Cannot advance from terminal state"):
            list(pipeline.advance_stream(_instance(api, conv_id), conv_id))
        assert llm.field_requests == []

    def test_reentrant_stream_from_a_handler_raises(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)
        caught: list[BaseException] = []

        def _reenter(context: dict[str, Any]) -> dict[str, Any]:
            try:
                list(api.advance_stream(conv_id))
            except FSMError as e:
                caught.append(e)
            return {}

        api.register_handler(
            create_handler("reenter").at(HandlerTiming.PRE_PROCESSING).do(_reenter)
        )

        assert list(api.advance_stream(conv_id)) == _CHUNKS

        assert len(caught) == 1
        assert "already being processed" in str(caught[0])
        assert len(llm.field_requests) == 1
        assert len(llm.stream_requests) == 1

    # -- stack and session -------------------------------------------------

    def test_stream_runs_the_pushed_sub_fsm(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        api.advance(conv_id)
        api.push_fsm(conv_id, _ambiguous_fsm())

        with patch("fsm_llm.pipeline.Classifier") as classifier_cls:
            classifier_cls.return_value = _classifier_choosing("support")
            chunks = list(api.advance_stream(conv_id))

        assert chunks == _CHUNKS
        sub_id = api.get_sub_conversation_id(conv_id)
        assert api.fsm_manager.get_conversation_state(sub_id) == "support"
        api.pop_fsm(conv_id)
        assert api.get_current_state(conv_id) == "plan"

    def test_top_of_stack_is_resolved_at_call_time(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _start(_trip_fsm(), llm)
        stream = api.advance_stream(conv_id)
        api.push_fsm(conv_id, _ambiguous_fsm())

        assert list(stream) == []

        assert api.fsm_manager.get_conversation_state(conv_id) == "plan"
        sub_id = api.get_sub_conversation_id(conv_id)
        assert api.fsm_manager.get_conversation_state(sub_id) == "route"

    def test_auto_save_after_a_streamed_step(self, tmp_path):
        store = FileSessionStore(tmp_path)
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm, session_store=store)
        stream = api.advance_stream(conv_id)
        assert store.load(conv_id) is None

        assert list(stream) == _CHUNKS

        saved = store.load(conv_id)
        assert saved is not None
        assert saved.current_state == "plan"
        assert saved.conversation_history[-1] == {"system": "".join(_CHUNKS)}

    def test_streamed_step_prompt_asks_for_no_acknowledgement(self):
        # Step 13 (D-035): the stream site passes the same "no message" value.
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)

        assert list(api.advance_stream(conv_id)) == _CHUNKS

        (request,) = llm.stream_requests
        assert "No user message was sent on this step." in request.system_prompt
        assert "Respond with plain text only: the output itself." in (
            request.system_prompt
        )
        assert "cknowledge" not in request.system_prompt

    def test_streamed_converse_prompt_keeps_the_conversational_wording(self):
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm)

        assert list(api.converse_stream("Paris please", conv_id)) == _CHUNKS

        (request,) = llm.stream_requests
        assert "Acknowledge this transition naturally." in request.system_prompt
        assert "No user message was sent" not in request.system_prompt

    def test_auto_save_when_the_stream_is_abandoned(self, tmp_path):
        store = FileSessionStore(tmp_path)
        llm = _StreamingLLM({"city": "Paris"})
        api, conv_id = _prepared_stream(llm, session_store=store)
        stream = api.advance_stream(conv_id)
        next(stream)
        assert store.load(conv_id) is None

        stream.close()

        saved = store.load(conv_id)
        assert saved is not None
        assert saved.current_state == "plan"
        assert saved.conversation_history[-1] == {"system": _CHUNKS[0]}

    def test_auto_save_after_a_failed_stream_holds_the_rolled_back_state(
        self, tmp_path
    ):
        store = FileSessionStore(tmp_path)
        llm = _StreamingLLM({"city": "Paris"}, fail_after=1)
        api, conv_id = _prepared_stream(llm, session_store=store)

        with pytest.raises(LLMResponseError):
            list(api.advance_stream(conv_id))

        saved = store.load(conv_id)
        assert saved is not None
        assert saved.current_state == "collect"
        assert "city" not in saved.context_data
