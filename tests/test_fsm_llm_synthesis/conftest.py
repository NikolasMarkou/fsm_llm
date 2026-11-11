"""Shared fixtures of the synthesis suite: no network, a valid compile and a
scripted strong model."""

import re
from typing import Any

import pytest

import fsm_llm.llm as llm_module
from fsm_llm.definitions import (
    CompletionRequest,
    CompletionResponse,
    DataExtractionResponse,
    LLMUsage,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from tests.conftest import MockLLM2Interface, block_network


@pytest.fixture(autouse=True)
def _offline_network(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Refuse IPv4/IPv6 connects (``tests.conftest.block_network``)."""
    block_network(monkeypatch, request.node)


@pytest.fixture
def foreign_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every ``LiteLLMInterface`` built and every provider call made.

    Each one is a foreign interface: the tests inject their own doubles, so a
    real interface appears only when a compile steers the pipeline to a model
    of its own. The constructor and ``fsm_llm.llm.completion`` raise as well,
    but a pipeline may swallow that, so the test asserts the list is empty.
    """
    seen: list[str] = []

    def built(self: Any, model: str = "?", *args: Any, **kwargs: Any) -> None:
        seen.append(f"interface:{model}")
        raise AssertionError(f"a foreign interface was built for {model!r}")

    def sent(**kwargs: Any) -> None:
        seen.append(f"completion:{kwargs.get('model')}")
        raise AssertionError("a provider call was made")

    monkeypatch.setattr(llm_module.LiteLLMInterface, "__init__", built)
    monkeypatch.setattr(llm_module, "completion", sent)
    return seen


class ScriptedStrongLLM(LLMInterface):
    """A strong model that answers ``complete`` from a queue.

    Each item is a reply text, a ``CompletionResponse`` or an exception
    instance the call raises. Every request is kept in ``requests``. An
    exhausted queue raises ``IndexError`` (an unexpected call).
    """

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, CompletionResponse):
            return reply
        return CompletionResponse(kind="final", text=reply)

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        raise NotImplementedError("the strong model only completes")


_STATE_TAG = re.compile(r"<current_state>([^<]+)</current_state>")


class ScriptedRunnerLLM(MockLLM2Interface):
    """The cheap model under a pushed procedure.

    ``extraction_data`` answers ``extract_field`` (a name it lacks is not
    extracted, so a child that needs it stays); ``responses`` maps a state id,
    read from the reply prompt's ``<current_state>`` tag, to the reply text.
    ``shown(kind, state=None)`` is the text of every recorded request of that
    kind (system prompt, user message, extraction context), optionally only
    the replies written for one state. ``usage()`` is the meter a simulator
    reads: one call per recorded request.
    """

    def __init__(
        self,
        extraction_data: dict[str, Any] | None = None,
        responses: dict[str, str] | None = None,
    ) -> None:
        super().__init__(extraction_data=extraction_data)
        self.responses = responses or {}

    def usage(self) -> LLMUsage:
        return LLMUsage(calls=len(self.call_history))

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        reply = super().generate_response(request)
        text = self.responses.get(self._state_of(request))
        return reply if text is None else reply.model_copy(update={"message": text})

    @staticmethod
    def _state_of(request: Any) -> str:
        found = _STATE_TAG.search(request.system_prompt)
        return found.group(1) if found else ""

    def count(self, kind: str, state: str | None = None) -> int:
        """How many requests of ``kind`` were made (for ``state``, if given)."""
        return sum(
            1
            for name, r in self.call_history
            if name == kind and (state is None or self._state_of(r) == state)
        )

    def shown(self, kind: str, state: str | None = None) -> str:
        return "\n".join(
            f"{r.system_prompt}\n{r.user_message}\n{getattr(r, 'context', None)}"
            for name, r in self.call_history
            if name == kind and (state is None or self._state_of(r) == state)
        )


def _typed(value: str | None, kind: str) -> Any:
    """``value``, the text a turn said, as a ``kind`` (``None`` when it is not one)."""
    if value is None:
        return None
    if kind == "int":
        return int(value) if re.fullmatch(r"-?\d+", value) else None
    if kind == "float":
        try:
            return float(value)
        except ValueError:
            return None
    if kind == "bool":
        return {"yes": True, "true": True, "no": False, "false": False}.get(value)
    return value


class ByTurn(ScriptedRunnerLLM):
    """The cheap model, honest: a key is extracted only if the turn states it,
    read as ``<key> is <value>`` up to a ``;`` and typed by the field's type."""

    def extract_field(self, request: Any) -> Any:
        found = re.search(
            rf"\b{re.escape(request.field_name)} is ([^;]+)", request.user_message or ""
        )
        value = _typed(found[1].strip() if found else None, request.field_type)
        self.extraction_data = {} if value is None else {request.field_name: value}
        return super().extract_field(request)

    def extract_bulk_data(self, request: Any) -> DataExtractionResponse:
        self.call_history.append(("extract_bulk_data", request))
        said = re.findall(r"(\w+) is ([^;]+)", request.user_message or "")
        return DataExtractionResponse(extracted_data={k: v.strip() for k, v in said})


def good_fsm() -> dict[str, Any]:
    """A fresh compile (new dict per call) that passes G1 to G5 with
    ``return_keys=("answer",)`` under the default policy."""
    return {
        "name": "Collect",
        "description": "Collect one answer",
        "initial_state": "ask",
        "states": {
            "ask": {
                "id": "ask",
                "description": "ask for the answer",
                "purpose": "get the answer",
                "response_instructions": "Ask for the answer",
                "field_extractions": [
                    {
                        "field_name": "answer",
                        "field_type": "str",
                        "extraction_instructions": "The user's answer",
                    }
                ],
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "answer given",
                        "priority": 1,
                        "conditions": [
                            {
                                "description": "has answer",
                                "requires_context_keys": ["answer"],
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "finished",
                "purpose": "wrap up",
                "response_instructions": "Thank the user",
            },
        },
    }


def stuck_fsm() -> dict[str, Any]:
    """ask -> mid -> done, where mid's edge needs a value the extraction never
    gives: stuck at ``mid`` with the return key present (so only ``ended``
    tells it from a pass)."""
    fsm = good_fsm()
    fsm["states"]["ask"]["transitions"][0]["target_state"] = "mid"
    never = {"==": [{"var": "answer"}, "never"]}
    fsm["states"]["mid"] = {
        "id": "mid",
        "description": "confirm",
        "purpose": "confirm",
        "response_instructions": "Confirm the answer",
        "transitions": [
            {
                "target_state": "done",
                "description": "confirmed",
                "priority": 1,
                "conditions": [
                    {
                        "description": "confirmed",
                        "requires_context_keys": ["answer"],
                        "logic": never,
                    }
                ],
            }
        ],
    }
    return fsm


@pytest.fixture
def valid_fsm() -> dict[str, Any]:
    """A fresh compile that passes G1 to G5 with ``return_keys=("answer",)``."""
    return good_fsm()


def host_fsm(
    *,
    name: str,
    description: str,
    persona: str,
    states: tuple[str, str, str, str],
    goal: str,
    inputs: list[str],
    return_keys: list[str],
    greet_text: str,
    work_description: str,
    ok_text: str,
    fail_text: str,
    edge_prefix: str = "",
    synthesize: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A host FSM: greet -> work (``synthesize``) -> ok on ``status == ok``,
    fail on ``status in [failed, aborted]``.

    ``states`` is ``(greet, work, ok, fail)`` (state ids); ``edge_prefix`` leads
    each edge description; ``synthesize`` overrides or adds keys of the state's
    ``synthesize`` config (``on_failure`` defaults to the ``fail`` state). Every
    call builds a fresh dict.
    """
    greet, work, ok, fail = states
    status = {"var": "synthesis_result.status"}

    def edge(target: str, priority: int, logic: dict[str, Any]) -> dict[str, Any]:
        label = f"{edge_prefix}{target}"
        condition = {
            "description": label,
            "requires_context_keys": ["synthesis_result"],
            "logic": logic,
        }
        return {
            "target_state": target,
            "description": label,
            "priority": priority,
            "conditions": [condition],
        }

    def end(state: str, text: str) -> dict[str, Any]:
        return {
            "id": state,
            "description": state,
            "purpose": state,
            "response_instructions": text,
        }

    return {
        "name": name,
        "description": description,
        "persona": persona,
        "initial_state": greet,
        "states": {
            greet: {
                "id": greet,
                "description": "greet",
                "purpose": "greet",
                "response_instructions": greet_text,
                "transitions": [{"target_state": work, "description": "go"}],
            },
            work: {
                "id": work,
                "description": work_description,
                "purpose": work_description,
                "response_instructions": "",
                "synthesize": {
                    "goal": goal,
                    "inputs": inputs,
                    "return_keys": return_keys,
                    "on_failure": fail,
                    **(synthesize or {}),
                },
                "transitions": [
                    edge(ok, 1, {"==": [status, "ok"]}),
                    edge(fail, 2, {"in": [status, ["failed", "aborted"]]}),
                ],
            },
            ok: end(ok, ok_text),
            fail: end(fail, fail_text),
        },
    }
