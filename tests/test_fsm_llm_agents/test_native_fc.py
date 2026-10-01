"""Tests for NativeFunctionCallingReactAgent (provider-native tool calling).

The agent is an FSM run by core (plan 944e2692 step 16, D-009, D-028): every
model turn is a core completion state. The loop tests inject a scripted
``LLMInterface`` through ``llm_interface=`` (the seam that replaced
``complete_fn``); the argument-shape tests stub the one provider binding
``fsm_llm.llm.completion`` under the agent's own ``LiteLLMInterface``. The
request rules (tools XOR response_format, Ollama preparation, seed, reply
normalisation) are core's and tested in ``tests/test_fsm_llm/test_llm_complete.py``;
the request bytes are pinned by ``test_native_fc_golden.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from fsm_llm.agents import (
    AgentConfig,
    NativeFunctionCallingReactAgent,
    ToolRegistry,
    tool,
)
from fsm_llm.agents.base import BaseAgent, _output_response_format
from fsm_llm.agents.constants import ContextKeys, NativeFCContextKeys, StopReason
from fsm_llm.agents.definitions import AgentTrace, ToolCall
from fsm_llm.agents.exceptions import AgentError, BudgetExhaustedError
from fsm_llm.agents.native_fc import _SYSTEM_PROMPT
from fsm_llm.definitions import (
    CompletionResponse,
    LLMResponseError,
    ModelToolCall,
    RunBudgetExceededError,
)
from tests.test_fsm_llm.test_completion_state import _ScriptedLLM


@tool
def weather(city: str) -> str:
    """Get weather for a city."""
    return f"sunny in {city}"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(weather._tool_definition)
    return reg


def _calls(*calls: tuple[str, dict[str, Any]], text: str | None = None) -> Any:
    """A ``calls`` reply: ``(name, arguments)`` pairs, ids ``c1``, ``c2``..."""
    return CompletionResponse(
        kind="calls",
        text=text,
        calls=tuple(
            ModelToolCall(id=f"c{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls, start=1)
        ),
    )


def _final(text: str | None) -> CompletionResponse:
    return CompletionResponse(kind="final", text=text)


_MALFORMED = CompletionResponse(kind="malformed")


def _agent(
    llm: Any, *, registry: ToolRegistry | None = None, **config: Any
) -> NativeFunctionCallingReactAgent:
    return NativeFunctionCallingReactAgent(
        tools=registry or _registry(),
        config=AgentConfig(model="mock/model", **config),
        llm_interface=llm,
    )


# ---------------------------------------------------------------------------
# Provider-binding stub (argument shapes a scripted interface cannot carry)
# ---------------------------------------------------------------------------


def _provider_reply(content: Any = None, calls: list[Any] | None = None) -> Any:
    """A provider reply read by attribute, like litellm's: any argument value."""
    tool_calls = [
        SimpleNamespace(
            id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
        )
        for call_id, name, arguments in calls or []
    ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _stub_provider(
    monkeypatch: pytest.MonkeyPatch, *replies: Any
) -> list[dict[str, Any]]:
    """Patch core's one ``completion`` binding; return the recorded requests."""
    sent: list[dict[str, Any]] = []
    queue = list(replies)

    def fake(**kwargs: Any) -> Any:
        sent.append(kwargs)
        return queue.pop(0)

    monkeypatch.setattr("fsm_llm.llm.completion", fake)
    monkeypatch.setattr(
        "fsm_llm.llm.get_supported_openai_params",
        lambda **_: ["response_format", "tools", "tool_choice"],
    )
    return sent


# ---------------------------------------------------------------------------
# Construction and API (kept)
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_registry_rejected(self):
        with pytest.raises(AgentError, match="empty tool registry"):
            NativeFunctionCallingReactAgent(tools=ToolRegistry())

    def test_seed_defaults_to_none(self):
        agent = NativeFunctionCallingReactAgent(
            tools=_registry(), config=AgentConfig(model="mock/model")
        )
        assert agent.seed is None


class TestSystemPolicy:
    """Standing instructions go in the SYSTEM message, not the user turn.

    Measured on `ollama_chat/qwen3.5:4b` (see decisions.md D-021 of plan
    bf7ffe24): the harness's role prompt delivered entirely in the user turn
    produced ZERO writes in 5/5 dispatches; the same text with its standing
    half moved here produced 4/5. These tests pin the seam: the no-policy path
    is the base prompt and the policy, when set, lands in the system message
    and nowhere else.
    """

    @staticmethod
    def _first_request(agent, llm, task="q"):
        agent.run(task)
        return llm.requests[0].messages

    def _agent(self, llm, **kwargs):
        return NativeFunctionCallingReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model"),
            llm_interface=llm,
            **kwargs,
        )

    def test_no_policy_leaves_the_system_message_exactly_as_it_was(self):
        llm = _ScriptedLLM(_final("done"))
        messages = self._first_request(self._agent(llm), llm)

        assert messages[0] == {"role": "system", "content": _SYSTEM_PROMPT}
        assert messages[1] == {"role": "user", "content": "q"}

    def test_a_constructor_policy_is_appended_to_the_base_prompt(self):
        """Appended, never substituted: the base prompt is what offers tools."""
        llm = _ScriptedLLM(_final("done"))
        agent = self._agent(llm, system_policy="RULES:\n- write the file")

        messages = self._first_request(agent, llm)

        assert messages[0]["content"] == (
            f"{_SYSTEM_PROMPT}\n\nRULES:\n- write the file"
        )
        assert messages[1] == {"role": "user", "content": "q"}

    def test_the_policy_is_read_at_run_time_not_at_construction(self):
        """`roles.py` sets this on an agent it received from a factory."""
        llm = _ScriptedLLM(_final("done"))
        agent = self._agent(llm)
        agent.system_policy = "EXIT GATE: stop when done"

        messages = self._first_request(agent, llm)

        assert messages[0]["content"].endswith("EXIT GATE: stop when done")

    def test_a_blank_policy_is_treated_as_no_policy(self):
        """An empty string must not append a trailing separator to the prompt."""
        agent = self._agent(_ScriptedLLM(), system_policy="")

        assert agent._system_message() == _SYSTEM_PROMPT

    def test_the_policy_does_not_leak_into_the_user_turn(self):
        """The whole point is that the standing half is NOT in the task."""
        llm = _ScriptedLLM(_final("done"))
        agent = self._agent(llm, system_policy="RULES:\n- write the file")

        messages = self._first_request(agent, llm, task="GOAL: ship it")

        assert messages[1]["content"] == "GOAL: ship it"
        assert "RULES:" not in messages[1]["content"]


class _Answer(BaseModel):
    findings_count: int
    needs_explore: bool


_VALID_PAYLOAD = '{"findings_count": 3, "needs_explore": false}'


class TestOutputResponseFormatHelper:
    """``base._output_response_format`` is the single envelope builder shared by
    ``_init_context`` (which the native repair state reads) and Pass 2.
    """

    def test_none_schema_returns_none(self):
        assert _output_response_format(None) is None

    def test_non_pydantic_object_returns_none(self):
        assert _output_response_format(object()) is None

    def test_envelope_matches_the_pre_extraction_shape(self):
        assert _output_response_format(_Answer) == {
            "type": "json_schema",
            "json_schema": {
                "name": "_Answer",
                "schema": _Answer.model_json_schema(),
            },
        }

    def test_init_context_still_stores_the_same_envelope(self):
        agent = NativeFunctionCallingReactAgent(
            tools=_registry(),
            config=AgentConfig(model="mock/model", output_schema=_Answer),
        )
        context = agent._init_context("task")
        assert context["_output_response_format"] == _output_response_format(_Answer)

    def test_init_context_omits_the_key_without_a_schema(self):
        agent = NativeFunctionCallingReactAgent(
            tools=_registry(), config=AgentConfig(model="mock/model")
        )
        assert "_output_response_format" not in agent._init_context("task")


# ---------------------------------------------------------------------------
# The model loop (ported from the private loop to the FSM run)
# ---------------------------------------------------------------------------


class TestLoop:
    def test_direct_answer_no_tools(self):
        llm = _ScriptedLLM(_final("42"))
        result = _agent(llm).run("what is the answer?")
        assert result.answer == "42"
        assert result.success is True
        assert result.stop_reason == StopReason.ANSWERED
        assert result.tools_used == []
        assert len(llm.requests) == 1

    def test_single_tool_then_answer(self):
        llm = _ScriptedLLM(
            _calls(("weather", {"city": "Paris"})), _final("It's sunny in Paris.")
        )
        result = _agent(llm).run("weather in Paris?")
        assert result.answer == "It's sunny in Paris."
        assert result.tools_used == ["weather"]
        assert result.success
        tool_message = llm.requests[1].messages[-1]
        assert tool_message == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "sunny in Paris",
        }

    def test_multiple_tool_calls_in_one_turn(self):
        llm = _ScriptedLLM(
            _calls(("weather", {"city": "Paris"}), ("weather", {"city": "Rome"})),
            _final("Done."),
        )
        result = _agent(llm).run("compare weather")
        assert [c.parameters for c in result.trace.tool_calls] == [
            {"city": "Paris"},
            {"city": "Rome"},
        ]
        assert [m["role"] for m in llm.requests[1].messages[-2:]] == ["tool", "tool"]

    def test_max_iterations_exhausted(self):
        llm = _ScriptedLLM(*[_calls(("weather", {"city": "X"})) for _ in range(3)])
        result = _agent(llm, max_iterations=3).run("loop")
        # No final answer; the tool calls still land in the trace. Whether that
        # counts as success is D-005's question, asserted in TestSuccessSignal.
        assert result.answer == ""
        assert len(result.trace.tool_calls) == 3
        assert len(llm.requests) == 3
        assert result.iterations_used == 3

    def test_uses_get_json_schemas(self):
        llm = _ScriptedLLM(_final("ok"))
        registry = _registry()
        _agent(llm, registry=registry).run("q")
        request = llm.requests[0]
        assert list(request.tools or []) == registry.get_json_schemas()
        assert request.tools[0]["function"]["name"] == "weather"
        assert request.tool_choice is None  # core sends "auto"


class TestSuccessSignal:
    """DECISION plan-2026-07-21T191807-bf7ffe24/D-005 (carried by D-027 of plan
    944e2692): ``success`` must distinguish a working run from a doomed one.
    The old ``bool(answer) or bool(trace_calls)`` reported True on three live
    runs that wrote nothing and answered nothing.
    """

    def test_tool_calls_without_a_final_answer_are_not_success(self):
        llm = _ScriptedLLM(_calls(("weather", {"city": "Oslo"})), _final(""))
        result = _agent(llm).run("q")
        assert result.tools_used == ["weather"]
        assert result.answer == ""
        assert result.success is False
        assert result.stop_reason == StopReason.NO_RESULT

    def test_exhausted_max_iterations_is_not_success(self):
        llm = _ScriptedLLM(*[_calls(("weather", {"city": "X"})) for _ in range(3)])
        result = _agent(llm, max_iterations=3).run("loop")
        assert len(result.trace.tool_calls) == 3
        assert result.success is False
        assert result.stop_reason == StopReason.MAX_ITERATIONS

    def test_final_answer_after_tool_use_is_success(self):
        llm = _ScriptedLLM(
            _calls(("weather", {"city": "Oslo"})), _final("It is sunny.")
        )
        result = _agent(llm).run("q")
        assert result.answer == "It is sunny."
        assert result.success is True


class TestTerminalConstrainedDecoding:
    """DECISION plan-2026-07-21T191807-bf7ffe24/D-002: after the loop, when a
    schema is configured and the free-text answer does not validate, make
    EXACTLY ONE extra completion carrying ``response_format`` and NO tools.
    """

    def test_does_not_fire_without_an_output_schema(self):
        llm = _ScriptedLLM(_final("just prose"))
        result = _agent(llm).run("q")
        assert len(llm.requests) == 1
        assert result.answer == "just prose"

    def test_does_not_fire_when_the_answer_already_parses(self):
        llm = _ScriptedLLM(_final(_VALID_PAYLOAD))
        result = _agent(llm, output_schema=_Answer).run("q")
        assert len(llm.requests) == 1
        assert result.structured_output.findings_count == 3

    def test_fires_once_when_the_answer_does_not_parse(self):
        llm = _ScriptedLLM(_final("I looked at three files."), _final(_VALID_PAYLOAD))
        result = _agent(llm, output_schema=_Answer).run("q")
        # EXACTLY one repair attempt: no retry loop.
        assert len(llm.requests) == 2
        assert result.structured_output.findings_count == 3
        assert result.answer == _VALID_PAYLOAD
        repair = llm.requests[1]
        assert repair.tools is None
        assert repair.response_format == _output_response_format(_Answer)
        # The repair turn appends its own user message so the Ollama schema
        # echo lands on the LAST message, not on the buried original task.
        assert repair.messages[-1]["role"] == "user"
        assert "single JSON object" in repair.messages[-1]["content"]

    def test_repair_reuses_the_tool_result_history(self):
        llm = _ScriptedLLM(
            _calls(("weather", {"city": "Oslo"})),
            _final("It was sunny."),
            _final(_VALID_PAYLOAD),
        )
        result = _agent(llm, output_schema=_Answer).run("q")
        assert len(llm.requests) == 3
        roles = [m["role"] for m in llm.requests[2].messages]
        assert "tool" in roles
        assert result.structured_output.needs_explore is False

    def test_unparseable_repair_leaves_the_original_answer_intact(self):
        llm = _ScriptedLLM(
            _final("a perfectly usable prose answer"), _final("still not JSON")
        )
        result = _agent(llm, output_schema=_Answer).run("q")
        assert len(llm.requests) == 2
        assert result.answer == "a perfectly usable prose answer"
        assert result.structured_output is None
        assert result.success is True

    def test_repair_can_rescue_an_empty_final_answer(self):
        """The measured live failure: Ollama returns empty content on the final
        turn. Success is decided AFTER the repair, so the rescue counts."""
        llm = _ScriptedLLM(_final(""), _final(_VALID_PAYLOAD))
        result = _agent(llm, output_schema=_Answer).run("q")
        assert result.answer == _VALID_PAYLOAD
        assert result.success is True

    def test_exhausted_loop_is_not_relabelled_success_by_a_repair(self):
        """D-005 stays intact: a loop that never concluded did not finish its
        work, so a payload extracted afterwards must not report success."""
        llm = _ScriptedLLM(
            _calls(("weather", {"city": "X"})),
            _calls(("weather", {"city": "X"})),
            _final(_VALID_PAYLOAD),
        )
        result = _agent(llm, output_schema=_Answer, max_iterations=2).run("loop")
        assert result.structured_output.findings_count == 3
        assert result.success is False
        assert result.stop_reason == StopReason.MAX_ITERATIONS


class TestForcedFinalTool:
    """DECISION plan-2026-07-23T073649-bb230f18/D-003 (carried by D-027 of plan
    944e2692): when ``force_final_tool`` is set and the loop never called that
    tool, the agent makes EXACTLY ONE post-loop model turn with ``tool_choice``
    pinned to that function, so the MODEL itself emits the write, run through
    ``self.tools.execute``. Default ``None`` adds no turn.
    """

    def test_forced_turn_fires_and_executes_after_toolless_conclusion(self):
        reg = _registry()
        executed: list[str] = []
        original_execute = reg.execute

        def spy(call, **kwargs):
            executed.append(call.tool_name)
            return original_execute(call, **kwargs)

        reg.execute = spy  # type: ignore[method-assign]

        llm = _ScriptedLLM(
            _final("here is my prose answer"),
            _calls(("weather", {"city": "Paris"})),
        )
        result = _agent(llm, registry=reg, force_final_tool="weather").run(
            "explore the tree"
        )

        # One loop turn + exactly one forced turn.
        assert len(llm.requests) == 2
        forced = llm.requests[1]
        assert forced.tool_choice == {
            "type": "function",
            "function": {"name": "weather"},
        }
        assert forced.response_format is None
        assert executed == ["weather"]
        assert [c.tool_name for c in result.trace.tool_calls] == ["weather"]
        assert result.answer == "here is my prose answer"

    def test_default_off_issues_no_forced_turn(self):
        llm = _ScriptedLLM(_final("prose answer"))
        result = _agent(llm).run("q")

        assert len(llm.requests) == 1
        assert result.trace.tool_calls == []

    def test_forced_turn_fires_exactly_once(self):
        llm = _ScriptedLLM(_final("prose"), _calls(("weather", {"city": "X"})))
        _agent(llm, force_final_tool="weather").run("q")

        # Exactly two completions; a third would find no scripted reply.
        assert len(llm.requests) == 2
        assert llm.replies == []

    def test_malformed_forced_turn_is_absorbed(self):
        llm = _ScriptedLLM(_final("prose"), _MALFORMED)
        result = _agent(llm, force_final_tool="weather").run("q")

        assert result.answer == "prose"
        assert [c.tool_name for c in result.trace.tool_calls] == []
        assert result.success is True

    def test_non_degradable_forced_turn_error_still_propagates(self):
        """A genuine outage on the forced turn is NOT swallowed."""
        outage = LLMResponseError("Completion call failed: 503")
        llm = _ScriptedLLM(_final("prose"), outage)

        with pytest.raises(AgentError) as excinfo:
            _agent(llm, force_final_tool="weather").run("q")

        assert excinfo.value.__cause__ is outage

    def test_forced_turn_skipped_when_tool_already_called(self):
        llm = _ScriptedLLM(_calls(("weather", {"city": "Paris"})), _final("done"))
        result = _agent(llm, force_final_tool="weather", max_iterations=3).run("q")

        assert len(llm.requests) == 2
        assert [c.tool_name for c in result.trace.tool_calls] == ["weather"]

    def test_forced_turn_skipped_when_tool_absent_from_registry(self):
        llm = _ScriptedLLM(_final("prose"))
        result = _agent(llm, force_final_tool="not_a_tool").run("q")

        assert len(llm.requests) == 1
        assert result.trace.tool_calls == []


class TestMalformedToolCallDegrades:
    """DECISION plan-2026-07-21T191807-bf7ffe24/D-016 (carried by D-027 of plan
    944e2692): a provider that garbles ONE tool-call turn (core returns
    ``kind="malformed"``) must not delete the run's trace and answer. A
    genuine outage (core raises) still ends the run.
    """

    def test_the_already_populated_trace_and_answer_survive(self):
        llm = _ScriptedLLM(
            _calls(("weather", {"city": "Oslo"}), text="checking the weather"),
            _MALFORMED,
        )
        result = _agent(llm, max_iterations=6).run("q")

        assert [c.tool_name for c in result.trace.tool_calls] == ["weather"]
        assert result.success is False
        assert result.stop_reason == StopReason.NO_RESULT
        assert len(llm.requests) == 2

    def test_the_repair_turn_still_runs_after_a_malformed_tool_turn(self):
        llm = _ScriptedLLM(_MALFORMED, _final(_VALID_PAYLOAD))
        result = _agent(llm, output_schema=_Answer).run("q")

        assert llm.requests[1].tools is None
        assert result.structured_output.findings_count == 3
        assert result.success is False  # the loop never concluded (D-005)

    def test_a_genuine_outage_still_ends_the_run(self):
        outage = LLMResponseError("Completion call failed: 503")
        llm = _ScriptedLLM(outage)

        with pytest.raises(AgentError) as excinfo:
            _agent(llm, output_schema=_Answer).run("q")

        assert excinfo.value.__cause__ is outage

    def test_a_malformed_repair_turn_keeps_the_original_answer(self):
        llm = _ScriptedLLM(_final("a usable prose answer"), _MALFORMED)
        result = _agent(llm, output_schema=_Answer).run("q")

        assert result.answer == "a usable prose answer"
        assert result.structured_output is None
        assert result.success is True


class TestMalformedArgumentsAreNotExecuted:
    """REACT-11 (D-025 of plan 06a5ec0a, carried by D-027 of plan 944e2692): a
    tool call whose arguments are not a JSON object makes the whole turn
    malformed in core: no call of it runs, the loop ends, the trace is kept.
    Driven through the agent's own ``LiteLLMInterface`` with the provider
    binding stubbed, so the arguments arrive exactly as a provider sends them.
    """

    @staticmethod
    def _counting_registry() -> tuple[ToolRegistry, list[dict]]:
        invocations: list[dict] = []

        def ping(note: str = "") -> str:
            """Record a call."""
            invocations.append({"note": note})
            return f"pong {note}"

        reg = ToolRegistry()
        reg.register_function(ping, name="ping", description="Record a call.")
        return reg, invocations

    @staticmethod
    def _agent(reg: ToolRegistry, **config: Any) -> NativeFunctionCallingReactAgent:
        return NativeFunctionCallingReactAgent(
            tools=reg, config=AgentConfig(model="mock/model", **config)
        )

    @pytest.mark.parametrize("bad", ["{not json", "[1, 2]", '"text"', "null", 7, ["a"]])
    def test_turn_with_bad_arguments_runs_nothing(self, bad, monkeypatch):
        reg, invocations = self._counting_registry()
        sent = _stub_provider(monkeypatch, _provider_reply(None, [("c1", "ping", bad)]))

        result = self._agent(reg, max_iterations=5).run("q")

        assert invocations == []
        assert result.trace.tool_calls == []
        # The loop ENDS (D-016), it does not retry the same turn.
        assert len(sent) == 1
        assert result.success is False

    def test_earlier_calls_survive_and_no_call_of_the_bad_turn_runs(self, monkeypatch):
        reg, invocations = self._counting_registry()
        _stub_provider(
            monkeypatch,
            _provider_reply(None, [("a", "ping", '{"note": "one"}')]),
            _provider_reply(
                None, [("b", "ping", '{"note": "two"}'), ("c", "ping", "{broken")]
            ),
        )
        result = self._agent(reg, max_iterations=5).run("q")

        assert invocations == [{"note": "one"}]
        assert [c.parameters for c in result.trace.tool_calls] == [{"note": "one"}]

    @pytest.mark.parametrize("empty", ["", "  ", None, {}])
    def test_blank_arguments_still_mean_no_arguments(self, empty, monkeypatch):
        reg, invocations = self._counting_registry()
        _stub_provider(
            monkeypatch,
            _provider_reply(None, [("c1", "ping", empty)]),
            _provider_reply("done"),
        )
        result = self._agent(reg).run("q")

        assert invocations == [{"note": ""}]
        assert result.success is True

    def test_forced_final_turn_with_bad_arguments_runs_nothing(self, monkeypatch):
        reg, invocations = self._counting_registry()
        _stub_provider(
            monkeypatch,
            _provider_reply("prose answer"),
            _provider_reply(None, [("f1", "ping", "{x")]),
        )
        result = self._agent(reg, force_final_tool="ping").run("q")

        assert invocations == []
        assert result.trace.tool_calls == []
        assert result.answer == "prose answer"


# ---------------------------------------------------------------------------
# On core (plan 944e2692 step 16; RED on the parent commit ced9eb5)
# ---------------------------------------------------------------------------


class TestRunsOnCore:
    def test_every_model_call_goes_to_the_injected_interface(self, monkeypatch):
        """Parent: the private loop called ``litellm.completion`` and ignored
        ``llm_interface``."""

        def refuse(**kwargs: Any) -> Any:
            raise AssertionError("a provider binding was called")

        monkeypatch.setattr("fsm_llm.llm.completion", refuse)
        monkeypatch.setattr("litellm.completion", refuse)
        llm = _ScriptedLLM(
            _calls(("weather", {"city": "Oslo"})),
            _final("Sunny."),
            _final(_VALID_PAYLOAD),
        )
        result = _agent(llm, output_schema=_Answer).run("q")

        assert len(llm.requests) == 3
        assert llm.other_calls == []
        assert result.success is True

    def test_requests_carry_the_interface_timeout(self, monkeypatch):
        """D-009: a native request now has core's per-call timeout (120 s by
        default). Parent: no ``timeout`` key (no bound on a hung provider)."""
        sent = _stub_provider(monkeypatch, _provider_reply("done"))

        NativeFunctionCallingReactAgent(
            tools=_registry(), config=AgentConfig(model="mock/model")
        ).run("q")

        assert sent[0]["timeout"] == 120.0

    def test_seed_reaches_the_provider_and_is_read_at_run_time(self, monkeypatch):
        sent = _stub_provider(monkeypatch, _provider_reply("a"), _provider_reply("b"))
        agent = NativeFunctionCallingReactAgent(
            tools=_registry(), config=AgentConfig(model="mock/model"), seed=7
        )
        agent.run("q")
        agent.seed = None
        agent.run("q")

        assert sent[0]["seed"] == 7
        assert "seed" not in sent[1]

    def test_initial_context_goes_through_init_context(self):
        """D-009: caller context is the run's context like every pattern's
        (parent: ignored, ``final_context == {"task": task}``); run outputs
        and the pattern's own keys are stripped, and the model never sees it."""
        llm = _ScriptedLLM(_final("real answer"))
        result = _agent(llm).run(
            "q",
            initial_context={
                "domain": "weather",
                ContextKeys.FINAL_ANSWER: "PWNED",
                NativeFCContextKeys.ANSWER: "PWNED",
                ContextKeys.FORCED_STOP_REASON: StopReason.MAX_ITERATIONS,
            },
        )

        assert result.final_context["domain"] == "weather"
        assert ContextKeys.FINAL_ANSWER not in result.final_context
        assert result.answer == "real answer"
        assert (result.success, result.stop_reason) == (True, StopReason.ANSWERED)
        assert "PWNED" not in json.dumps(llm.requests[0].messages)
        assert "weather" not in json.dumps(llm.requests[0].messages)

    @pytest.mark.parametrize("max_iterations", [1, 2])
    def test_the_step_ceiling_covers_every_turn(self, max_iterations):
        """An exhausted loop, the forced turn and the repair turn all fit."""
        llm = _ScriptedLLM(
            *[_calls(("weather", {"city": "X"})) for _ in range(max_iterations)],
            _calls(("write_note", {"text": "notes"})),
            _final(_VALID_PAYLOAD),
        )
        registry = _registry()
        registry.register_function(
            lambda text: f"saved {text}", name="write_note", description="Save."
        )

        result = _agent(
            llm,
            registry=registry,
            max_iterations=max_iterations,
            force_final_tool="write_note",
            output_schema=_Answer,
        ).run("q")

        assert llm.replies == []
        assert result.stop_reason == StopReason.MAX_ITERATIONS
        assert result.structured_output.findings_count == 3

    def test_step_ceiling_and_budget_error_text(self):
        agent = _agent(_ScriptedLLM(), max_iterations=3)

        assert agent._step_ceiling(3) == (
            10,
            "2 x max_iterations 3 + 4 post-loop steps",
        )
        error = agent._budget_error(RunBudgetExceededError("steps", 10, 10))
        assert isinstance(error, BudgetExhaustedError)
        assert "10 loop turns = 2 x max_iterations 3 + 4 post-loop steps" in str(error)


class TestSuccessSeam:
    """D-028: the one seam honours a handler-recorded ``no_result``."""

    def test_recorded_no_result_beats_a_traced_tool_call(self):
        """Parent: a run with a tool call read as ``(True, "answered")``."""
        trace = AgentTrace(
            tool_calls=[ToolCall(tool_name="weather", parameters={})],
            total_iterations=2,
        )
        outcome = BaseAgent._run_outcome(
            {ContextKeys.FORCED_STOP_REASON: StopReason.NO_RESULT},
            trace,
            [NativeFCContextKeys.ANSWER],
        )
        assert outcome == (False, StopReason.NO_RESULT)

    def test_without_a_recorded_reason_the_rule_is_unchanged(self):
        trace = AgentTrace(tool_calls=[], total_iterations=1)
        assert BaseAgent._run_outcome(
            {NativeFCContextKeys.ANSWER: "Paris"}, trace, [NativeFCContextKeys.ANSWER]
        ) == (True, StopReason.ANSWERED)
