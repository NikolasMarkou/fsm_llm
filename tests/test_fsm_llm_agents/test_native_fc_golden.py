"""Golden provider requests of ``NativeFunctionCallingReactAgent``.

The native tool-calling path is measured-good (agents-react B0 37/38 at 2.39
calls, harness L4 B1 40/40), and its request bytes decide that: the system
message placement alone moved a 4B worker from 0/5 to 4/5 writes (D-021 of plan
bf7ffe24). Plan 944e2692 rebuilds this agent as an FSM run by core (D-009); this
module pins every model-visible key of every request it sends, captured at
e1f63a9, so the rebuild is proven to send the same requests.

The recording stub replaces BOTH ``litellm.completion`` (today's call site) and
``fsm_llm.llm.completion`` (core's one binding, where the rebuilt agent sends),
so the test survives the move unchanged. Only the transport keys in
``_TRANSPORT_KEYS`` are left out of the comparison: they never reach the model.

Do NOT regenerate the fixture to make a failure go away: a difference in any
other key is STOP IF 1 of plan 944e2692 (the rebuilt path sends different
bytes).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from litellm import ModelResponse
from pydantic import BaseModel

from fsm_llm.agents import (
    AgentConfig,
    NativeFunctionCallingReactAgent,
    ToolRegistry,
)
from fsm_llm.agents.definitions import ToolDefinition

_FIXTURE = Path(__file__).parent / "fixtures" / "native_fc_golden_requests.json"

#: Keys that configure the transport, not the model's input (D-009).
_TRANSPORT_KEYS = frozenset({"timeout", "max_retries", "api_key", "api_base"})

#: Both completion bindings the stub replaces (today's and core's).
_BINDINGS = ("litellm.completion", "fsm_llm.llm.completion")

_MODELS = {"ollama": "ollama_chat/qwen3.5:4b", "openai": "gpt-4o-mini"}


class _WeatherReport(BaseModel):
    """The ``output_schema`` of the repair-turn scenario."""

    city: str
    conditions: str


def _weather(city: str) -> str:
    return f"sunny, 21 C in {city}"


def _write_note(text: str) -> str:
    return f"saved {len(text)} chars"


def _registry() -> ToolRegistry:
    """Two tools with explicit schemas, so tool-schema inference cannot drift."""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="weather",
            description="Get the current weather for a city.",
            parameter_schema={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City"}},
                "required": ["city"],
            },
            execute_fn=_weather,
        )
    )
    registry.register(
        ToolDefinition(
            name="write_note",
            description="Save a note with the final findings.",
            parameter_schema={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Note"}},
                "required": ["text"],
            },
            execute_fn=_write_note,
        )
    )
    return registry


def _final(text: str) -> ModelResponse:
    """A provider reply with text and no tool calls."""
    return ModelResponse(
        model="stub",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _calls(*calls: tuple[str, str, dict[str, Any]]) -> ModelResponse:
    """A provider reply carrying ``(id, name, arguments)`` tool calls."""
    return ModelResponse(
        model="stub",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        for call_id, name, args in calls
                    ],
                },
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _raw_calls(*calls: tuple[str, str, str]) -> ModelResponse:
    """A provider reply carrying ``(id, name, raw arguments text)`` tool calls."""
    return ModelResponse(
        model="stub",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": raw},
                        }
                        for call_id, name, raw in calls
                    ],
                },
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


_PARIS = ("call_1", "weather", {"city": "Paris"})
_ROME = ("call_2", "weather", {"city": "Rome"})
_NOTE = ("call_9", "write_note", {"text": "Paris: sunny, 21 C"})
_REPORT = '{"city": "Paris", "conditions": "sunny"}'


def _scenario_specs() -> dict[str, dict[str, Any]]:
    """Base scenarios: task, scripted replies and agent settings."""
    return {
        "direct_answer": {
            "task": "What is the capital of France?",
            "script": lambda: [_final("Paris is the capital of France.")],
        },
        "one_tool_call": {
            "task": "What is the weather in Paris?",
            "script": lambda: [_calls(_PARIS), _final("It is sunny in Paris.")],
        },
        "two_calls_one_turn": {
            "task": "Compare the weather in Paris and Rome.",
            "script": lambda: [
                _calls(_PARIS, ("call_2", "weather", {"city": "Rome"})),
                _final("Both cities are sunny."),
            ],
        },
        "forced_final_tool": {
            "task": "Check the weather in Paris and record it.",
            "config": {"force_final_tool": "write_note"},
            "script": lambda: [
                _calls(_PARIS),
                _final("Paris is sunny."),
                _calls(("call_9", "write_note", {"text": "Paris: sunny, 21 C"})),
            ],
        },
        "repair_turn": {
            "task": "Report the weather in Paris.",
            "config": {"output_schema": _WeatherReport},
            "script": lambda: [
                _calls(_PARIS),
                _final("Paris is sunny."),
                _final('{"city": "Paris", "conditions": "sunny"}'),
            ],
        },
        "seed_7": {
            "task": "What is the weather in Paris?",
            "seed": 7,
            "script": lambda: [_calls(_PARIS), _final("It is sunny in Paris.")],
        },
        "system_policy": {
            "task": "What is the weather in Paris?",
            "system_policy": "Answer in one sentence. Never guess a temperature.",
            "script": lambda: [_calls(_PARIS), _final("It is sunny in Paris.")],
        },
        # Added in step 18.2 (review round 1, D-029), captured at e1f63a9 like
        # the scenarios above: the harness EXPLORE configuration (forced
        # tool plus output schema), the iteration limit (alone and with both
        # post-loop turns), a malformed turn and a provider error.
        "forced_tool_output_schema": {
            "task": "Check the weather in Paris, record it and report it.",
            "config": {
                "force_final_tool": "write_note",
                "output_schema": _WeatherReport,
            },
            "script": lambda: [
                _calls(_PARIS),
                _final("Paris is sunny."),
                _calls(_NOTE),
                _final(_REPORT),
            ],
        },
        "iteration_limit": {
            "task": "Compare the weather in Paris and Rome.",
            "config": {"max_iterations": 2},
            "script": lambda: [_calls(_PARIS), _calls(_ROME)],
        },
        "iteration_limit_forced_repair": {
            "task": "Check the weather in Paris, record it and report it.",
            "config": {
                "max_iterations": 1,
                "force_final_tool": "write_note",
                "output_schema": _WeatherReport,
            },
            "script": lambda: [_calls(_PARIS), _calls(_NOTE), _final(_REPORT)],
        },
        "malformed_turn": {
            "task": "What is the weather in Paris?",
            "script": lambda: [
                _calls(_PARIS),
                _raw_calls(("call_2", "weather", '{"city": ')),
            ],
        },
        "provider_error": {
            "task": "What is the weather in Paris?",
            "raises": True,
            "script": lambda: [_calls(_PARIS), RuntimeError("provider is down")],
        },
    }


def _scenario_names() -> list[str]:
    return [f"{base}/{label}" for base in _scenario_specs() for label in _MODELS]


class _Recorder:
    """A completion stub: answers from a script, records each request.

    A scripted exception is raised instead of answered (a provider error).
    """

    def __init__(self, script: list[ModelResponse | Exception]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> ModelResponse:
        payload = {k: v for k, v in kwargs.items() if k not in _TRANSPORT_KEYS}
        # JSON round trip at call time: the caller may mutate its message
        # list after the call, and the provider sees JSON anyway.
        self.requests.append(json.loads(json.dumps(payload)))
        if not self.script:
            raise AssertionError("native_fc sent more requests than scripted")
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _shown(value: Any) -> Any:
    return value.model_dump() if isinstance(value, BaseModel) else value


def _run_scenario(name: str, mp: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run scenario *name* (``base/model``) and return its requests and outcome."""
    base, label = name.split("/")
    spec = _scenario_specs()[base]
    script: Callable[[], list[ModelResponse | Exception]] = spec["script"]
    recorder = _Recorder(script())
    for binding in _BINDINGS:
        mp.setattr(binding, recorder)
    settings: dict[str, Any] = {"max_iterations": 5, **spec.get("config", {})}
    config = AgentConfig(
        model=_MODELS[label], temperature=0.5, max_tokens=1000, **settings
    )
    agent = NativeFunctionCallingReactAgent(
        _registry(),
        config=config,
        system_policy=spec.get("system_policy"),
        seed=spec.get("seed"),
    )
    if spec.get("raises"):
        # Only the requests and the error type are pinned: the error text
        # names the code path (D-027 moved the wrap to `_standard_run`).
        with pytest.raises(Exception) as excinfo:
            agent.run(spec["task"])
        assert recorder.script == [], "a scripted reply was never requested"
        return {
            "requests": recorder.requests,
            "outcome": {"raised": type(excinfo.value).__name__},
        }
    result = agent.run(spec["task"])
    assert recorder.script == [], "a scripted reply was never requested"
    return {
        "requests": recorder.requests,
        "outcome": {
            "answer": result.answer,
            "success": result.success,
            "stop_reason": result.stop_reason,
            "tool_calls": [
                [call.tool_name, call.parameters] for call in result.trace.tool_calls
            ],
            "structured_output": _shown(result.structured_output),
        },
    }


def _golden() -> dict[str, Any]:
    with _FIXTURE.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


class TestGoldenFixture:
    def test_fixture_covers_exactly_the_scenarios(self) -> None:
        assert sorted(_golden()["scenarios"]) == sorted(_scenario_names())

    def test_fixture_holds_no_transport_key(self) -> None:
        for scenario in _golden()["scenarios"].values():
            for request in scenario["requests"]:
                assert not _TRANSPORT_KEYS & set(request)

    def test_fixture_pins_the_request_rules(self) -> None:
        """Spot checks that the capture holds the rules it exists to pin."""
        scenarios = _golden()["scenarios"]
        for name, scenario in scenarios.items():
            for request in scenario["requests"]:
                # tools XOR response_format (D-002 of bf7ffe24).
                assert ("tools" in request) != ("response_format" in request), name
                # seed absent unless set (D-008 of 879d04a0).
                assert ("seed" in request) == name.startswith("seed_7/"), name
                # Ollama preparation only on Ollama (D-003 of bf7ffe24).
                ollama = name.endswith("/ollama")
                assert ("reasoning_effort" in request) == ollama, name
        system = scenarios["system_policy/openai"]["requests"][0]["messages"][0]
        assert system["role"] == "system"
        assert system["content"].endswith(
            "\n\nAnswer in one sentence. Never guess a temperature."
        )


class TestGoldenRequests:
    @pytest.mark.parametrize("name", _scenario_names())
    def test_requests_equal_the_golden_capture(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = _golden()["scenarios"][name]
        captured = _run_scenario(name, monkeypatch)
        assert len(captured["requests"]) == len(expected["requests"])
        for index, (got, want) in enumerate(
            zip(captured["requests"], expected["requests"], strict=True)
        ):
            assert got == want, f"{name} request {index} differs"
        assert captured["outcome"] == expected["outcome"]
