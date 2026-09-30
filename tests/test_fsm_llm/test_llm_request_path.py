"""One LLM request path: the classifier sends through ``fsm_llm.llm`` and no
provider request carries an empty user turn (plan 07ad3f8c, step 3)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import fsm_llm.classification as classification_module
from fsm_llm import constants
from fsm_llm.api import API
from fsm_llm.classification import Classifier
from fsm_llm.definitions import (
    ClassificationError,
    ClassificationResponseError,
    ClassificationSchema,
    IntentDefinition,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LiteLLMInterface
from fsm_llm.prompts import ClassificationPromptConfig

OLLAMA_MODEL = "ollama_chat/qwen3.5:4b"


def _response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.reasoning_content = None
    return response


def _schema() -> ClassificationSchema:
    return ClassificationSchema(
        intents=[
            IntentDefinition(name="buy", description="User wants to purchase"),
            IntentDefinition(name="browse", description="User is just looking"),
        ],
        fallback_intent="browse",
    )


def _shop_fsm() -> dict:
    """A speaking initial state with a classified intent gating ``shop``."""
    return {
        "name": "Shop",
        "description": "d",
        "initial_state": "triage",
        "persona": "A shop assistant",
        "states": {
            "triage": {
                "id": "triage",
                "description": "Find out what the user wants",
                "purpose": "Learn the intent",
                "response_instructions": "Greet the user and ask what they need.",
                "classification_extractions": [
                    {
                        "field_name": "intent",
                        "intents": [
                            {"name": "buy", "description": "User wants to purchase"},
                            {"name": "browse", "description": "User is just looking"},
                        ],
                        "fallback_intent": "browse",
                        "confidence_threshold": 0.7,
                    }
                ],
                "transitions": [
                    {
                        "target_state": "shop",
                        "description": "The user wants to buy",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "intent is buy",
                                "requires_context_keys": ["intent"],
                                "logic": {"==": [{"var": "intent"}, "buy"]},
                            }
                        ],
                    }
                ],
            },
            "shop": {
                "id": "shop",
                "description": "Shopping",
                "purpose": "Close the sale",
                "response_instructions": "Confirm the order.",
            },
        },
    }


class _Provider:
    """Scripted ``fsm_llm.llm.completion``: records every request's kwargs."""

    def __init__(self, supported: list[str] | None = None):
        self.calls: list[dict[str, Any]] = []
        self._supported = ["response_format"] if supported is None else supported

    def completion(self, **kwargs: Any) -> MagicMock:
        self.calls.append(kwargs)
        name = (kwargs.get("response_format") or {}).get("json_schema", {}).get("name")
        system = kwargs["messages"][0]["content"]
        if name == "intent_classification" or "intent" in system[:200].lower():
            return _response(
                json.dumps({"reasoning": "r", "intent": "buy", "confidence": 0.95})
            )
        return _response(json.dumps({"message": "Hello there", "reasoning": ""}))

    def __enter__(self) -> _Provider:
        self._patches = [
            patch("fsm_llm.llm.completion", side_effect=self.completion),
            patch(
                "fsm_llm.llm.get_supported_openai_params",
                return_value=self._supported,
            ),
            # The parent commit's classifier binding: reaching it is a failure.
            patch(
                "fsm_llm.classification.completion",
                side_effect=AssertionError("classifier bypassed the LLM layer"),
                create=True,
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for p in reversed(self._patches):
            p.stop()

    def user_turns(self) -> list[str]:
        return [
            m["content"]
            for call in self.calls
            for m in call["messages"]
            if m["role"] == "user"
        ]

    def classifier_calls(self) -> list[dict[str, Any]]:
        return [
            c
            for c in self.calls
            if (c.get("response_format") or {}).get("json_schema", {}).get("name")
            == "intent_classification"
        ]


class TestClassifierUsesTheLLMLayer:
    def test_classify_issues_one_call_through_the_llm_binding(self):
        with _Provider() as provider:
            result = Classifier(_schema(), model="gpt-4o").classify("I want a phone")
        assert result.intent == "buy"
        assert len(provider.calls) == 1
        call = provider.calls[0]
        assert call["model"] == "gpt-4o"
        assert call["response_format"]["json_schema"]["name"] == (
            "intent_classification"
        )
        assert call["messages"][1] == {"role": "user", "content": "I want a phone"}

    def test_classification_module_has_no_provider_binding(self):
        assert not hasattr(classification_module, "get_supported_openai_params")
        assert "completion" not in vars(classification_module)

    def test_pipeline_built_classifier_goes_through_the_llm_binding(self):
        with _Provider() as provider:
            api = API.from_definition(_shop_fsm(), model="gpt-4o", api_key="sk-x")
            cid, _ = api.start_conversation()
            api.converse("I would like to buy a phone", cid)
            assert api.get_current_state(cid) == "shop"
        calls = provider.classifier_calls()
        assert len(calls) == 1
        assert calls[0]["api_key"] == "sk-x"

    def test_prompt_config_sets_temperature_and_max_tokens(self):
        config = ClassificationPromptConfig(temperature=0.3, max_tokens=77)
        with _Provider() as provider:
            Classifier(_schema(), model="gpt-4o", config=config).classify("hi")
        assert provider.calls[0]["temperature"] == 0.3
        assert provider.calls[0]["max_tokens"] == 77

    def test_reserved_kwargs_are_ignored_not_a_type_error(self):
        with _Provider() as provider:
            Classifier(
                _schema(), model="gpt-4o", temperature=1.5, max_tokens=5, stream=True
            ).classify("hi")
        call = provider.calls[0]
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == 512
        assert "stream" not in call

    @pytest.mark.parametrize(("given", "sent"), [({}, 120.0), ({"timeout": 7}, 7)])
    def test_timeout_default_and_override(self, given, sent):
        with _Provider() as provider:
            Classifier(_schema(), model="gpt-4o", **given).classify("hi")
        assert provider.calls[0]["timeout"] == sent

    def test_unsupported_model_gets_no_response_format(self):
        with _Provider(supported=[]) as provider:
            result = Classifier(_schema(), model="gpt-4o").classify("I want it")
        assert result.intent == "buy"
        assert "response_format" not in provider.calls[0]

    def test_ollama_preparation_applies_to_the_classifier_call(self):
        with _Provider() as provider:
            Classifier(
                _schema(),
                model=OLLAMA_MODEL,
                config=ClassificationPromptConfig(temperature=0.9),
            ).classify("I want a phone")
        call = provider.calls[0]
        assert call["temperature"] == 0
        assert call["reasoning_effort"] == "none"
        user = call["messages"][1]["content"]
        assert user.startswith("/nothink\nI want a phone")
        assert "Respond in JSON matching this schema:" in user

    def test_non_ollama_call_gets_no_ollama_preparation(self):
        with _Provider() as provider:
            Classifier(_schema(), model="gpt-4o").classify("I want a phone")
        call = provider.calls[0]
        assert "reasoning_effort" not in call
        assert call["messages"][1]["content"] == "I want a phone"

    def test_provider_failure_is_a_classification_error(self):
        with (
            patch("fsm_llm.llm.completion", side_effect=RuntimeError("down")),
            patch("fsm_llm.llm.get_supported_openai_params", return_value=[]),
            pytest.raises(ClassificationError, match="Classification LLM call failed"),
        ):
            Classifier(_schema(), model="gpt-4o").classify("hi")

    def test_malformed_shape_is_a_classification_response_error(self):
        response = MagicMock()
        response.choices = [object()]  # a choice without .message
        with (
            patch("fsm_llm.llm.completion", return_value=response),
            patch("fsm_llm.llm.get_supported_openai_params", return_value=[]),
            pytest.raises(ClassificationResponseError, match="Malformed"),
        ):
            Classifier(_schema(), model="gpt-4o").classify("hi")


class TestNeverAnEmptyUserTurn:
    def test_greeting_and_classifier_requests_hold_no_empty_user_content(self):
        with _Provider() as provider:
            api = API.from_definition(_shop_fsm(), model="gpt-4o", api_key="sk-x")
            cid, greeting = api.start_conversation()
            api.converse("", cid)
        assert greeting == "Hello there"
        assert len(provider.classifier_calls()) == 1
        turns = provider.user_turns()
        assert len(turns) >= 3  # greeting, classifier, reply
        assert all(turn.strip() for turn in turns)
        assert set(turns) == {constants.NEUTRAL_USER_TURN}

    def test_classifier_with_an_empty_message_sends_the_neutral_turn(self):
        with _Provider() as provider:
            Classifier(_schema(), model="gpt-4o").classify("")
        assert provider.user_turns() == [constants.NEUTRAL_USER_TURN]

    def test_ollama_greeting_prepares_the_neutral_turn(self):
        with _Provider() as provider:
            api = API.from_definition(_shop_fsm(), model=OLLAMA_MODEL)
            api.start_conversation()
        assert provider.user_turns() == [f"/nothink\n{constants.NEUTRAL_USER_TURN}"]

    def test_stream_request_holds_no_empty_user_content(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "Hi"
        chunk.choices[0].delta.reasoning_content = None
        llm = LiteLLMInterface(model="gpt-4o")
        request = ResponseGenerationRequest(system_prompt="Greet.", user_message="  ")
        with (
            patch("fsm_llm.llm.completion", return_value=iter([chunk])) as completion,
            patch("fsm_llm.llm.get_supported_openai_params", return_value=[]),
        ):
            assert list(llm.generate_response_stream(request)) == ["Hi"]
        sent = completion.call_args.kwargs["messages"]
        assert sent[1] == {"role": "user", "content": constants.NEUTRAL_USER_TURN}

    def test_neutral_turn_never_enters_history_or_a_prompt(self):
        with _Provider() as provider:
            api = API.from_definition(_shop_fsm(), model="gpt-4o", api_key="sk-x")
            cid, _ = api.start_conversation()
            api.converse("", cid)
            history = api.get_conversation_history(cid)
            data = api.get_data(cid)
        neutral = constants.NEUTRAL_USER_TURN
        assert neutral not in json.dumps(history)
        assert neutral not in json.dumps(data, default=str)
        for call in provider.calls:
            assert neutral not in call["messages"][0]["content"]

    def test_real_user_message_is_sent_unchanged(self):
        with _Provider() as provider:
            api = API.from_definition(_shop_fsm(), model="gpt-4o", api_key="sk-x")
            cid, _ = api.start_conversation()
            api.converse("I would like to buy a phone", cid)
        assert provider.user_turns()[1:] == ["I would like to buy a phone"] * 2

    def test_fill_does_not_mutate_the_callers_messages(self):
        from fsm_llm.llm import _fill_empty_user_turns

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
            {"role": "user", "content": "hi"},
        ]
        filled = _fill_empty_user_turns(messages)
        assert messages[1]["content"] == ""
        assert filled[0] is messages[0]  # an empty system turn is not a user turn
        assert filled[1] == {"role": "user", "content": constants.NEUTRAL_USER_TURN}
        assert filled[2] is messages[2]
