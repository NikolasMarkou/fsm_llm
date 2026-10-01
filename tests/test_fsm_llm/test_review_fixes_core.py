"""Review round 1, core fixes (plan 944e2692, step 18.1, D-029, D-030).

Each class pins one accepted finding of the step-18 review
(findings/review-iter-1*.md). Every test here fails on the parent commit
86b271a, except the two meter-lock tests and the nested-transcript test,
which pass there and exist to kill a mutant the earlier tests let live
(the lock removed, the transcript deep copy removed).
"""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from litellm.types.utils import EmbeddingResponse

import fsm_llm.llm as llm_module
from fsm_llm import API, FileSessionStore
from fsm_llm.classification import Classifier, HierarchicalClassifier
from fsm_llm.constants import RESERVED_LLM_CALL_KWARGS
from fsm_llm.definitions import (
    ClassificationSchema,
    CompletionRequest,
    CompletionResponse,
    FieldExtractionResponse,
    FSMDefinition,
    FSMError,
    HierarchicalSchema,
    IntentDefinition,
    LLMResponseError,
    ResponseGenerationResponse,
)
from fsm_llm.llm import (
    LiteLLMEmbedder,
    LiteLLMInterface,
    LLMInterface,
    check_tool_transcript,
)
from fsm_llm.validator import FSMValidator
from tests.conftest import MockLLM2Interface
from tests.test_fsm_llm.test_completion_state import (
    _MESSAGES,
    _RESULT,
    _api,
    _calls,
    _context,
    _final,
    _ScriptedLLM,
    _start,
    _state,
    _to_call_model,
    _tool_fsm,
    _tool_runner,
)
from tests.test_fsm_llm.test_llm_complete import _ADD_TOOL, _call, _Provider, _reply

_REPO = Path(__file__).resolve().parents[2]
_USER = [{"role": "user", "content": "What is 2 + 3?"}]


def _sent(llm: LiteLLMInterface, request: CompletionRequest) -> dict[str, Any]:
    """The provider kwargs of one ``complete`` call (supported params incl.
    ``response_format``)."""
    with _Provider(_reply("ok")) as provider:
        llm.complete(request)
    (params,) = provider.calls
    return params


# --------------------------------------------------------------
# Area 1: the request builder and the LLM layer
# --------------------------------------------------------------


class TestToolRulesHoldWhateverTheCallType:
    """Review area 1, concern 1: the builder sent a ``response_format``
    beside ``tools`` and forced Ollama temperature 0 on a tool turn when the
    free-string ``call_type`` named an extraction or classification."""

    @pytest.mark.parametrize(
        "call_type", ["data_extraction", "field_extraction", "classification"]
    )
    @pytest.mark.parametrize("model", ["ollama_chat/qwen3.5:4b", "gpt-4o"])
    def test_a_tool_turn_gets_no_response_format_and_keeps_its_temperature(
        self, model: str, call_type: str
    ):
        llm = LiteLLMInterface(model, temperature=0.7)
        params = _sent(
            llm,
            CompletionRequest(messages=_USER, tools=[_ADD_TOOL], call_type=call_type),
        )
        assert params["tools"] == [_ADD_TOOL]
        assert "response_format" not in params
        assert params["temperature"] == 0.7
        # Nothing structured leaks into the user turn either (Ollama echo).
        assert "schema" not in params["messages"][-1]["content"].lower()

    def test_the_builder_refuses_tools_beside_a_response_format(self):
        llm = LiteLLMInterface("gpt-4o")
        with (
            _Provider(_reply("ok")),
            pytest.raises(ValueError, match="never both"),
        ):
            llm._build_call_params(
                _USER,
                "completion",
                tools=[_ADD_TOOL],
                response_format={"type": "json_object"},
            )

    def test_a_structured_turn_on_ollama_still_runs_at_temperature_zero(self):
        llm = LiteLLMInterface("ollama_chat/qwen3.5:4b", temperature=0.7)
        params = _sent(
            llm,
            CompletionRequest(messages=_USER, response_format={"type": "json_object"}),
        )
        assert params["temperature"] == 0


class TestCompleteBuildsInsideItsErrorBoundary:
    """Review area 1, concern 8: a builder error escaped ``complete`` raw."""

    def test_a_failing_supported_params_lookup_is_an_llm_response_error(self):
        llm = LiteLLMInterface("gpt-4o")
        with (
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                side_effect=ConnectionError("registry down"),
            ),
            patch("fsm_llm.llm.completion") as provider,
            pytest.raises(LLMResponseError, match="registry down") as caught,
        ):
            llm.complete(CompletionRequest(messages=_USER))
        assert isinstance(caught.value.__cause__, ConnectionError)
        provider.assert_not_called()


class TestReservedCallKwargs:
    """Review area 1, concern 11: the legacy tool kwargs and ``n`` set on a
    constructor reached every extraction, classifier and Pass-2 call."""

    @pytest.mark.parametrize("name", ["functions", "function_call", "n"])
    def test_constructor_kwarg_never_reaches_a_request(self, name: str):
        assert name in RESERVED_LLM_CALL_KWARGS
        llm = LiteLLMInterface("gpt-4o", **{name: [{"name": "f"}]})
        params = _sent(llm, CompletionRequest(messages=_USER))
        assert name not in params


class TestEmbedder:
    """Review area 1, concerns 4 and 12."""

    def test_a_generator_of_texts_is_embedded(self):
        def provider(**params: Any) -> EmbeddingResponse:
            return EmbeddingResponse(
                data=[
                    {"object": "embedding", "index": i, "embedding": [1.0, float(i)]}
                    for i, _ in enumerate(params["input"])
                ]
            )

        embedder = LiteLLMEmbedder("ollama/qwen3-embedding:0.6b")
        with patch("fsm_llm.llm.embedding", side_effect=provider) as sent:
            vectors = embedder.embed(t for t in ["hello", "world"])
        assert vectors == [[1.0, 0.0], [1.0, 1.0]]
        assert sent.call_args.kwargs["input"] == ["hello", "world"]

    def test_a_generator_holding_a_non_string_is_refused(self):
        embedder = LiteLLMEmbedder("ollama/qwen3-embedding:0.6b")
        with pytest.raises(TypeError):
            embedder.embed(t for t in ["hello", 3])  # type: ignore[misc]

    @pytest.mark.parametrize(
        "vectors, message",
        [
            ([[], []], "empty"),
            ([[1.0, 2.0], [3.0]], "dimensions"),
            ([[float("nan")], [1.0]], "non-finite"),
            ([[1.0], [float("inf")]], "non-finite"),
        ],
    )
    def test_an_unusable_vector_is_an_llm_response_error(
        self, vectors: list[list[float]], message: str
    ):
        reply = EmbeddingResponse(
            data=[
                {"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(vectors)
            ]
        )
        embedder = LiteLLMEmbedder("ollama/qwen3-embedding:0.6b")
        with (
            patch("fsm_llm.llm.embedding", return_value=reply),
            pytest.raises(LLMResponseError, match=message),
        ):
            embedder.embed(["a", "b"])


class TestUsageMeterLocks:
    """Review area 1, concern 2: removing any meter lock left every test
    green. These force the interleaving the locks exclude, so each fails
    deterministically with its lock removed (no timing luck): the paused
    thread waits on an event the other thread can only set when it is not
    excluded. With the lock in place the pause times out and the result is
    exact; the wait bounds the cost of the passing run."""

    PAUSE = 0.5

    def test_two_first_calls_build_one_meter(self, monkeypatch):
        llm = LiteLLMInterface("gpt-4o")
        built: list[Any] = []
        first_inside = threading.Event()
        second_built = threading.Event()
        original = llm_module._UsageMeter.__init__

        def init(meter: Any) -> None:
            original(meter)
            built.append(meter)
            if len(built) == 1:
                first_inside.set()
                # Unlocked, the second caller also sees no meter and builds
                # one now; locked, it waits on the creation lock instead.
                second_built.wait(self.PAUSE)
            else:
                second_built.set()

        monkeypatch.setattr(llm_module._UsageMeter, "__init__", init)
        meters: list[Any] = []
        first = threading.Thread(
            target=lambda: meters.append(llm_module._meter_of(llm))
        )
        second = threading.Thread(
            target=lambda: meters.append(llm_module._meter_of(llm))
        )
        first.start()
        assert first_inside.wait(5)
        second.start()
        first.join(5)
        second.join(5)

        assert len(built) == 1
        assert len(meters) == 2 and meters[0] is meters[1] is llm._usage_meter

    def test_a_reset_during_a_count_loses_nothing(self):
        meter = llm_module._UsageMeter()
        inside = threading.Event()
        release = threading.Event()
        pause = self.PAUSE

        class _PausingCounts(dict):
            """Counts that pause the bumping thread between its read of the
            counts dict and its add (the window the lock closes)."""

            paused = False

            def __getitem__(self, key: str) -> int:
                if not _PausingCounts.paused:
                    _PausingCounts.paused = True
                    inside.set()
                    release.wait(pause)
                return super().__getitem__(key)

        meter._counts["complete"] = _PausingCounts(
            dict.fromkeys(llm_module._COUNTER_FIELDS, 0)
        )
        harvested: list[int] = []
        bump = threading.Thread(target=lambda: meter._bump("complete", calls=1))
        reset = threading.Thread(
            target=lambda: harvested.append(meter.snapshot(reset=True).calls)
        )
        bump.start()
        assert inside.wait(5)
        reset.start()
        # Unlocked, the reset finishes inside the window; locked, it waits.
        reset.join(pause / 2)
        release.set()
        bump.join(5)
        reset.join(5)

        assert harvested[0] + meter.snapshot().calls == 1


# --------------------------------------------------------------
# Area 2: the completion state
# --------------------------------------------------------------


class TestPushedChildOwnsItsCompletionResult:
    """Review area 2, concern 1: a pushed child inherited the parent's
    completion result, skipped its own model call (skip-if-set) and
    answered with the parent's reply."""

    def _parent(self) -> dict[str, Any]:
        parent = _tool_fsm()
        parent["states"]["call_model"]["transitions"][1]["target_state"] = "followup"
        parent["states"]["followup"] = _state(
            "followup",
            transitions=[{"target_state": "done", "description": "End"}],
        )
        return parent

    def test_the_child_makes_its_own_call(self):
        child = _tool_fsm()
        child["name"] = "Child"
        llm = _ScriptedLLM(_final("parent answer"), _final("child answer"))
        api = _api(self._parent(), llm)
        conv_id = _start(api, "parent task")
        api.advance(conv_id)
        api.advance(conv_id)
        assert api.get_data(conv_id)[_RESULT]["text"] == "parent answer"

        api.push_fsm(
            conv_id,
            child,
            context_to_pass={_MESSAGES: [{"role": "user", "content": "child task"}]},
        )
        api.advance(conv_id)
        api.advance(conv_id)

        assert len(llm.requests) == 2
        assert llm.requests[1].messages[-1] == {
            "role": "user",
            "content": "child task",
        }
        assert api.get_data(conv_id)[_RESULT]["text"] == "child answer"

    def test_an_explicitly_passed_result_is_kept(self):
        child = _tool_fsm()
        child["name"] = "Child"
        llm = _ScriptedLLM(_final("parent answer"))
        api = _api(self._parent(), llm)
        conv_id = _start(api, "parent task")
        api.advance(conv_id)
        api.advance(conv_id)
        handed = {"kind": "final", "text": "handed over", "calls": []}
        api.push_fsm(conv_id, child, context_to_pass={_RESULT: handed})
        assert api.get_data(conv_id)[_RESULT] == handed


class TestToolTurnWithoutATaskIsRefused:
    """Review area 2, concern 2: a session restored in the middle of a tool
    loop (the transcript is an internal key, not saved) sent only the
    instructions and took the model's answer to nothing as the result."""

    def test_a_restore_mid_tool_loop_fails_loudly(self, tmp_path: Path):
        fsm = _tool_fsm()
        llm = _ScriptedLLM(_calls(("add", {"a": 2, "b": 3})), _final("never sent"))
        store = FileSessionStore(str(tmp_path))
        api = API.from_definition(fsm, llm_interface=llm, session_store=store)
        api.register_handler(_tool_runner())
        conv_id = _start(api)
        for _ in range(3):  # intake -> call_model -> run_tools -> call_model
            api.advance(conv_id)
        assert api.get_current_state(conv_id) == "call_model"
        api.save_session(conv_id)
        (saved,) = list(tmp_path.iterdir())

        restored = API.from_definition(fsm, llm_interface=llm, session_store=store)
        restored.register_handler(_tool_runner())
        restored_id, _ = restored.restore_session(saved.stem)
        with pytest.raises(LLMResponseError, match="empty"):
            restored.advance(restored_id)
        assert len(llm.requests) == 1  # only the call made before the save
        assert restored.get_current_state(restored_id) == "call_model"

    def test_an_absent_transcript_on_a_tool_turn_is_refused(self):
        llm = _ScriptedLLM(_final("never sent"))
        api = _api(_tool_fsm(), llm)
        conv_id, _ = api.start_conversation()
        _to_call_model(api, conv_id)
        with pytest.raises(LLMResponseError, match="empty"):
            api.advance(conv_id)
        assert llm.requests == []


class TestTranscriptPairingIsExact:
    """Review area 2, concern 4: ``check_tool_transcript`` accepted
    duplicate, missing and coerced ids, non-text results, calls without a
    function and user turns with no content."""

    @staticmethod
    def _round(
        calls: list[dict[str, Any]], results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": calls},
            *results,
        ]

    @staticmethod
    def _result(call_id: Any, content: Any = "1") -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    @pytest.mark.parametrize(
        "case",
        [
            "duplicate ids",
            "missing ids",
            "empty ids",
            "int id answered by its str",
            "None id answered by 'None'",
            "dict tool content",
            "None tool content",
            "call without a function",
            "arguments as a dict",
            "call without a name",
            "user content None",
        ],
    )
    def test_refused(self, case: str):
        tc = _call
        transcripts = {
            "duplicate ids": self._round(
                [tc("add", "{}", "a"), tc("add", "{}", "a")],
                [self._result("a"), self._result("a", "2")],
            ),
            "missing ids": self._round(
                [{"type": "function", "function": {"name": "add", "arguments": "{}"}}],
                [{"role": "tool", "content": "1"}],
            ),
            "empty ids": self._round([tc("add", "{}", "")], [self._result("")]),
            "int id answered by its str": self._round(
                [{**tc("add", "{}"), "id": 1}], [self._result("1")]
            ),
            "None id answered by 'None'": self._round(
                [{**tc("add", "{}"), "id": None}], [self._result("None")]
            ),
            "dict tool content": self._round(
                [tc("add", "{}", "a")], [self._result("a", {"x": 1})]
            ),
            "None tool content": self._round(
                [tc("add", "{}", "a")], [self._result("a", None)]
            ),
            "call without a function": self._round(
                [{"id": "a", "type": "function"}], [self._result("a")]
            ),
            "arguments as a dict": self._round(
                [tc("add", {"a": 1}, "a")], [self._result("a")]
            ),
            "call without a name": self._round(
                [tc("", "{}", "a")], [self._result("a")]
            ),
            "user content None": [{"role": "user", "content": None}],
        }
        with pytest.raises(LLMResponseError):
            check_tool_transcript(transcripts[case])

    def test_reordered_results_with_distinct_ids_still_pass(self):
        check_tool_transcript(
            self._round(
                [_call("add", "{}", "a"), _call("add", "{}", "b")],
                [self._result("b"), self._result("a", "2")],
            )
        )


class TestTranscriptIsDeepCopied:
    """Review area 2, concern 7: removing the transcript deep copy left the
    suite green (the old test mutated only a top-level ``content``, which
    pydantic copies anyway). An interface that mutates a nested tool call
    must not rewrite the context transcript."""

    def test_a_nested_mutation_of_the_request_leaves_the_context(self):
        llm = _ScriptedLLM(_calls(("add", {"a": 2, "b": 3})), _final("5"))
        api = _api(_tool_fsm(), llm)
        conv_id = _start(api)
        for _ in range(4):  # ... -> call_model (second model turn) -> done
            api.advance(conv_id)
        before = copy.deepcopy(_context(api, conv_id)[_MESSAGES])
        sent = llm.requests[1].messages
        assert sent[2]["tool_calls"][0]["function"]["name"] == "add"
        sent[2]["tool_calls"][0]["function"]["arguments"] = "mutated"
        assert _context(api, conv_id)[_MESSAGES] == before


class TestValidatorWarnsOnCompletionStates:
    """Review area 2, concerns 5 and 8."""

    def test_an_uncovered_result_kind_is_a_warning(self):
        fsm = _tool_fsm()
        call_model = fsm["states"]["call_model"]
        call_model["transitions"] = call_model["transitions"][:2]
        del fsm["states"]["failed"]
        result = FSMValidator(fsm).validate()
        assert result.is_valid
        assert any("'call_model'" in w and "'malformed'" in w for w in result.warnings)

    def test_full_coverage_or_a_fallback_edge_gives_no_warning(self):
        covered = FSMValidator(_tool_fsm()).validate()
        fallback = _tool_fsm()
        transitions = fallback["states"]["call_model"]["transitions"]
        transitions[2] = {"target_state": "failed", "description": "Otherwise"}
        transitions[1]["conditions"] = []
        for result in (covered, FSMValidator(fallback).validate()):
            assert not any("result kind" in w for w in result.warnings)

    @pytest.mark.parametrize(
        "channel, entry",
        [
            (
                "field_extractions",
                [
                    {
                        "field_name": _RESULT,
                        "field_type": "str",
                        "extraction_instructions": "the result",
                    }
                ],
            ),
            ("required_context_keys", [_RESULT]),
            (
                "classification_extractions",
                [
                    {
                        "field_name": _RESULT,
                        "intents": [
                            {"name": "a", "description": "a"},
                            {"name": "b", "description": "b"},
                        ],
                        "fallback_intent": "b",
                    }
                ],
            ),
        ],
    )
    def test_an_extraction_of_a_completion_result_key_is_a_warning(
        self, channel: str, entry: list[Any]
    ):
        fsm = _tool_fsm()
        fsm["states"]["intake"][channel] = entry
        result = FSMValidator(fsm).validate()
        assert any(
            f"'intake' {channel} names '{_RESULT}'" in w for w in result.warnings
        )

    def test_an_extraction_of_a_handler_only_key_is_a_warning(self):
        fsm = _tool_fsm()
        fsm["handler_only_keys"] = ["verdict"]
        fsm["states"]["intake"]["required_context_keys"] = ["verdict"]
        result = FSMValidator(fsm).validate()
        assert any(
            "'intake' required_context_keys names 'verdict'" in w
            for w in result.warnings
        )


# The ids of the 14 example FSMs, computed with this rule over the models
# of e1f63a9 (the iteration-1 close, before `completion` existed): the
# same ids as at HEAD, so the fields this plan added changed none.
_E1F63A9_IDS: dict[str, str] = {
    "examples/advanced/yoga_instructions/fsm.json": (
        "fsm_Adaptive Yoga Instruction_bb1015b5"
    ),
    "examples/basic/form_filling/fsm.json": "fsm_Form Filling FSM_fbe5e0cc",
    "examples/basic/multi_turn_extraction/fsm.json": (
        "fsm_MultiTurnExtraction_d84a9e85"
    ),
    "examples/basic/simple_greeting/fsm.json": "fsm_Simple Greeting FSM_51af6495",
    "examples/basic/story_time/fsm.json": (
        "fsm_Three Little Pigs Interactive Story_aae22383"
    ),
    "examples/classification/classified_transitions/fsm.json": (
        "fsm_Classified Transitions Demo_81032c47"
    ),
    "examples/classification/classified_transitions/fsm_manual.json": (
        "fsm_Classified Transitions Manual Mode_20183191"
    ),
    "examples/classification/smart_helpdesk/fsm_account.json": (
        "fsm_account_management_flow_2e543fe9"
    ),
    "examples/classification/smart_helpdesk/fsm_troubleshooting.json": (
        "fsm_troubleshooting_flow_400616f4"
    ),
    "examples/intermediate/adaptive_quiz/fsm.json": "fsm_adaptive_quiz_576b9b07",
    "examples/intermediate/book_recommendation/fsm.json": (
        "fsm_Book Recommendation System_1bd2d04b"
    ),
    "examples/intermediate/product_recommendation/fsm.json": (
        "fsm_Product Recommendation System_5f2f95d5"
    ),
    "examples/reasoning/math_tutor/fsm.json": "fsm_math_tutor_51465ff2",
    "examples/workflows/order_processing/fsm_order_form.json": (
        "fsm_order_form_04ab22cd"
    ),
}


class TestFsmIdIgnoresDefaults:
    """Review area 2, concern 3 (D-030): the id hashed the full dump, so
    every optional field added to the models (``completion``) re-ided every
    FSM. It now hashes ``model_dump(exclude_defaults=True)``."""

    @pytest.mark.parametrize("path", sorted(_E1F63A9_IDS))
    def test_example_ids_did_not_move_with_the_new_fields(self, path: str):
        raw = json.loads((_REPO / path).read_text())
        _, fsm_id = API.process_fsm_definition(raw)
        assert fsm_id == _E1F63A9_IDS[path]

    def test_an_explicit_default_and_an_omitted_field_hash_the_same(self):
        omitted = _tool_fsm()
        explicit = copy.deepcopy(omitted)
        explicit["version"] = "4.1"
        explicit["states"]["intake"]["completion"] = None
        explicit["states"]["intake"]["extraction_retries"] = 1
        explicit["states"]["call_model"]["completion"]["result_key"] = _RESULT
        assert (
            API.process_fsm_definition(explicit)[1]
            == API.process_fsm_definition(omitted)[1]
        )

    def test_different_content_still_gets_a_different_id(self):
        changed = _tool_fsm()
        changed["states"]["intake"]["extraction_retries"] = 2
        assert (
            API.process_fsm_definition(changed)[1]
            != API.process_fsm_definition(_tool_fsm())[1]
        )

    def test_construction_paths_agree(self, tmp_path: Path):
        raw = _tool_fsm()
        path = tmp_path / "fsm.json"
        path.write_text(json.dumps(raw))
        ids = {
            API.process_fsm_definition(raw)[1],
            API.process_fsm_definition(FSMDefinition(**raw))[1],
            API.process_fsm_definition(str(path))[1],
        }
        assert len(ids) == 1


# --------------------------------------------------------------
# Area 3: the classifier on the conversation interface
# --------------------------------------------------------------


_INTENTS = ClassificationSchema(
    intents=[
        IntentDefinition(name="buy", description="wants to buy"),
        IntentDefinition(name="browse", description="just looking"),
    ],
    fallback_intent="browse",
)


class _Custom(LLMInterface):
    """A user's own interface: Pass 1/2 answer, ``complete`` runs ``bug``."""

    model = "custom/m"

    def __init__(self, bug: Any) -> None:
        self.bug = bug

    def generate_response(self, request: Any) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="ok", message_type="response", reasoning=""
        )

    def extract_field(self, request: Any) -> FieldExtractionResponse:
        return FieldExtractionResponse(
            field_name=request.field_name, value=None, confidence=0.0, is_valid=False
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        return self.bug(request)


def _no_such_attribute(request: Any) -> Any:
    return request.no_such_attribute


def _outage(request: Any) -> Any:
    raise LLMResponseError("Completion call failed: provider down")


def _tie_fsm() -> dict[str, Any]:
    return {
        "name": "Tie",
        "description": "Two transitions at one priority: the classifier decides",
        "initial_state": "route",
        "states": {
            "route": _state(
                "route",
                response_instructions="Respond",
                transitions=[
                    {"target_state": "a", "description": "A"},
                    {"target_state": "b", "description": "B"},
                ],
            ),
            "a": _state("a"),
            "b": _state("b"),
        },
    }


def _classifying_fsm() -> dict[str, Any]:
    return {
        "name": "Classify",
        "description": "One classification field",
        "initial_state": "triage",
        "states": {
            "triage": _state(
                "triage",
                response_instructions="Respond",
                classification_extractions=[
                    {
                        "field_name": "intent",
                        "intents": [
                            {"name": "buy", "description": "wants to buy"},
                            {"name": "browse", "description": "just looking"},
                        ],
                        "fallback_intent": "browse",
                    }
                ],
                transitions=[
                    {
                        "target_state": "done",
                        "description": "Classified",
                        "conditions": [
                            {
                                "description": "intent known",
                                "requires_context_keys": ["intent"],
                                "logic": {"==": [{"var": "intent"}, "buy"]},
                            }
                        ],
                    }
                ],
            ),
            "done": _state("done"),
        },
    }


def _chain(error: BaseException) -> list[type[BaseException]]:
    kinds: list[type[BaseException]] = []
    current: BaseException | None = error
    while current is not None:
        kinds.append(type(current))
        current = current.__cause__ or current.__context__
    return kinds


class TestClassifierErrorContract:
    """Review area 3, concern 1: a bug inside a custom interface's
    ``complete`` became a silent "stay" or a skipped field, while the same
    bug in ``generate_response`` failed the turn (0d9c218e/D-001)."""

    @pytest.mark.parametrize("fsm", [_tie_fsm, _classifying_fsm])
    def test_a_programming_error_in_complete_fails_the_turn(self, fsm: Any):
        api = API.from_definition(fsm(), llm_interface=_Custom(_no_such_attribute))
        conv_id, _ = api.start_conversation()
        with pytest.raises((AttributeError, FSMError)) as caught:
            api.converse("I want to buy it", conv_id)
        assert AttributeError in _chain(caught.value)

    def test_a_provider_error_is_a_classification_error(self):
        classifier = Classifier(_INTENTS, llm=_Custom(_outage))
        with pytest.raises(FSMError, match="provider down") as caught:
            classifier.classify("I want to buy it")
        assert type(caught.value).__name__ == "ClassificationError"

    def test_a_provider_error_at_a_tie_still_stays(self):
        api = API.from_definition(_tie_fsm(), llm_interface=_Custom(_outage))
        conv_id, _ = api.start_conversation()
        api.converse("I want to buy it", conv_id)
        assert api.get_current_state(conv_id) == "route"


class TestClassifierNeedsAnInterfaceWithComplete:
    """Review area 3, concerns 2 and 3: an interface without ``complete``
    degraded every classification per turn; a model beside an interface
    without one was accepted and mislabelled."""

    @pytest.mark.parametrize("llm", ["garbage", MockLLM2Interface()])
    def test_classifier_refuses(self, llm: Any):
        with pytest.raises(ValueError, match="implements complete"):
            Classifier(_INTENTS, llm=llm)

    def test_api_refuses_a_classifying_definition(self):
        with pytest.raises(ValueError, match="does not implement complete"):
            API.from_definition(_classifying_fsm(), llm_interface=MockLLM2Interface())

    def test_api_refuses_a_classifying_child_on_push(self):
        api = API.from_definition(_tie_fsm(), llm_interface=MockLLM2Interface())
        conv_id, _ = api.start_conversation()
        with pytest.raises(ValueError, match="does not implement complete"):
            api.push_fsm(conv_id, _classifying_fsm())
        assert api.get_stack_depth(conv_id) == 1

    def test_api_accepts_it_without_classification(self):
        API.from_definition(_tool_fsm(), llm_interface=MockLLM2Interface())

    def test_api_accepts_it_when_every_entry_names_its_own_model(self):
        fsm = _classifying_fsm()
        fsm["states"]["triage"]["classification_extractions"][0]["model"] = "gpt-4o"
        API.from_definition(fsm, llm_interface=MockLLM2Interface())

    def test_a_model_beside_an_interface_without_one_is_refused(self):
        class _NoModel(_Custom):
            model = None  # type: ignore[assignment]

        with pytest.raises(ValueError, match="different model"):
            Classifier(_INTENTS, model="openai/gpt-4o", llm=_NoModel(_outage))
        assert Classifier(_INTENTS, llm=_NoModel(_outage)).model is None


class TestHierarchicalClassifierTakesAnInterface:
    """Review area 3, concern 5: ``HierarchicalClassifier`` had no ``llm=``
    and built one private interface per stage."""

    def _schema(self) -> HierarchicalSchema:
        return HierarchicalSchema(
            domain_schema=ClassificationSchema(
                intents=[
                    IntentDefinition(name="shop", description="shopping"),
                    IntentDefinition(name="other", description="anything else"),
                ],
                fallback_intent="other",
            ),
            intent_schemas={"shop": _INTENTS},
        )

    def test_every_stage_sends_through_the_one_interface(self):
        llm = _Custom(_outage)
        hierarchical = HierarchicalClassifier(self._schema(), llm=llm)
        stages = [
            hierarchical._domain_classifier,
            *hierarchical._intent_classifiers.values(),
        ]
        assert [stage._llm for stage in stages] == [llm, llm]
        assert {stage.model for stage in stages} == {"custom/m"}

    def test_settings_beside_the_interface_are_refused(self):
        with pytest.raises(ValueError, match="no api_key"):
            HierarchicalClassifier(self._schema(), llm=_Custom(_outage), api_key="k")


class TestApiRefusesSettingsBesideAnInjectedInterface:
    """Review area 7, concern 1: ``API(llm_interface=..., seed=7)`` dropped
    the seed silently (native_fc's ``agent.seed`` still reported it)."""

    @pytest.mark.parametrize(
        "settings", [{"seed": 7}, {"api_key": "k"}, {"timeout": 5, "caching": True}]
    )
    def test_refused(self, settings: dict[str, Any]):
        with pytest.raises(ValueError, match="takes no LLM settings"):
            API.from_definition(
                _tool_fsm(), llm_interface=MockLLM2Interface(), **settings
            )

    def test_native_fc_seed_with_an_injected_interface_is_refused(self):
        from fsm_llm.agents import AgentConfig, ToolRegistry, create_agent

        def echo(x: str) -> str:
            """Echo."""
            return x

        registry = ToolRegistry()
        registry.register_function(echo, name="echo")
        agent = create_agent(
            "native_fc",
            registry,
            config=AgentConfig(model="gpt-4o-mini"),
            seed=7,
            llm_interface=LiteLLMInterface(model="gpt-4o-mini"),
        )
        # The provider binding is patched so a regression never sends.
        with (
            _Provider(_reply("done")) as provider,
            pytest.raises(ValueError, match=r"\['seed'\]"),
        ):
            agent.run("hi")
        assert provider.calls == []
