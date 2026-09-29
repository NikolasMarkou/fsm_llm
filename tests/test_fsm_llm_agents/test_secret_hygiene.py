"""Secret-looking values stay out of memory listings, observations, traces, logs.

Plan: plan-2026-09-29T103145-06a5ec0a / D-016 (SEC-05, SEC-06).

Before the fix ``list_memories`` printed the hidden ``metadata`` buffer and any
``api_key`` value verbatim, and every tool executor wrote its raw tool input
into the observation text, ``AgentStep.action``, the trace ``tool_input`` and
an INFO log. The fix redacts a COPY: the tool still receives the real input
and ``ApprovalRequest.parameters`` stays raw (the approver sees the exact
call). Each test drives the real handler or run and asserts the secret string
appears nowhere in what the run hands back or logs.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.definitions import AgentConfig, ApprovalRequest, ToolCall
from fsm_llm.agents.handlers import AgentHandlers
from fsm_llm.agents.hitl import HumanInTheLoop
from fsm_llm.agents.memory_tools import create_memory_tools
from fsm_llm.agents.native_fc import NativeFunctionCallingReactAgent
from fsm_llm.agents.parallel_react import TOOL_CALLS_KEY, ParallelReactAgent
from fsm_llm.agents.rewoo import REWOOAgent
from fsm_llm.agents.tools import ToolRegistry, redact_secret_entries
from fsm_llm.logging import logger
from fsm_llm.memory import BUFFER_CORE, BUFFER_METADATA, WorkingMemory

_SECRET = "sk-SECRET-abc123def456ghi789"
_INPUT = {"q": "weather", "api_key": _SECRET}


def _dump(value: Any) -> str:
    return json.dumps(value, default=repr)


class _Capture:
    """Capture every library log line at DEBUG and above for one block."""

    def __enter__(self) -> list[str]:
        self.lines: list[str] = []
        logger.enable("fsm_llm")  # library logging is off by default
        self._sink = logger.add(lambda m: self.lines.append(str(m)), level="DEBUG")
        return self.lines

    def __exit__(self, *exc: object) -> None:
        logger.remove(self._sink)
        logger.disable("fsm_llm")


class _Tool:
    """A ``lookup(q, api_key)`` tool recording the real arguments it got."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.registry = ToolRegistry()

        def lookup(q: str, api_key: str) -> str:
            self.calls.append({"q": q, "api_key": api_key})
            return "sunny"

        self.registry.register_function(
            lookup,
            name="lookup",
            description="Look something up",
            parameter_schema={
                "properties": {
                    "q": {"type": "string"},
                    "api_key": {"type": "string"},
                },
                "required": ["q", "api_key"],
            },
        )

    def assert_got_real_input(self) -> None:
        assert self.calls == [{"q": "weather", "api_key": _SECRET}]


class TestRedactHelper:
    def test_redacts_nested_and_keeps_keys(self):
        value = {"q": "x", "creds": {"password": "hunter2"}, "items": [_INPUT]}
        out = redact_secret_entries(value)
        assert "hunter2" not in _dump(out) and _SECRET not in _dump(out)
        assert out["q"] == "x"
        assert set(out["creds"]) == {"password"}
        assert out["items"][0]["q"] == "weather"

    def test_never_mutates_the_input(self):
        value = {"api_key": _SECRET, "nested": {"password": "p"}}
        redact_secret_entries(value)
        assert value == {"api_key": _SECRET, "nested": {"password": "p"}}

    def test_non_mapping_passes_through(self):
        assert redact_secret_entries("plain text") == "plain text"
        assert redact_secret_entries(None) is None


class TestMemoryTools:
    """SEC-05: hidden buffers and secret values never reach the model."""

    @staticmethod
    def _tools(memory: WorkingMemory) -> dict[str, Any]:
        return {t.name: t.execute_fn for t in create_memory_tools(memory)}

    def _memory(self) -> WorkingMemory:
        memory = WorkingMemory()
        memory.set(BUFFER_METADATA, "session_note", _SECRET)
        memory.set(BUFFER_CORE, "api_key", _SECRET)
        memory.set(BUFFER_CORE, "user_name", "Alice")
        return memory

    def test_list_all_skips_hidden_and_redacts(self):
        out = self._tools(self._memory())["list_memories"]()
        assert _SECRET not in out
        assert BUFFER_METADATA not in out
        assert "user_name = Alice" in out
        assert "api_key" in out  # the key stays visible, only its value goes

    def test_list_hidden_buffer_is_refused(self):
        out = self._tools(self._memory())["list_memories"](buffer=BUFFER_METADATA)
        assert _SECRET not in out
        assert "does not exist" in out

    def test_list_one_buffer_redacts(self):
        out = self._tools(self._memory())["list_memories"](buffer=BUFFER_CORE)
        assert _SECRET not in out
        assert "user_name = Alice" in out

    def test_recall_redacts(self):
        memory = WorkingMemory()
        memory.set(BUFFER_CORE, "api_key", _SECRET)
        out = self._tools(memory)["recall"](query="api")
        assert "api_key" in out
        assert _SECRET not in out

    def test_forget_skips_hidden(self):
        memory = self._memory()
        out = self._tools(memory)["forget"](key="session_note")
        assert "not found" in out
        assert memory.get(BUFFER_METADATA, "session_note") == _SECRET

    def test_remember_rejects_hidden_buffer(self):
        memory = WorkingMemory()
        with pytest.raises(ValueError, match="hidden"):
            self._tools(memory)["remember"](key="k", value="v", buffer=BUFFER_METADATA)
        assert not memory.has_buffer(BUFFER_METADATA)

    def test_remember_via_registry_is_a_failed_call(self):
        registry = ToolRegistry()
        for tool_def in create_memory_tools(WorkingMemory()):
            registry.register(tool_def)
        result = registry.execute(
            ToolCall(
                tool_name="remember",
                parameters={"key": "k", "value": "v", "buffer": BUFFER_METADATA},
            )
        )
        assert result.success is False


class TestHandlersRedactToolInput:
    """SEC-06: AgentHandlers.execute_tool (ReAct, Reflexion, VerifiedReact...)."""

    def test_no_secret_in_delta_or_logs_tool_gets_real_input(self):
        tool = _Tool()
        handlers = AgentHandlers(tool.registry)
        context = {
            ContextKeys.TOOL_NAME: "lookup",
            ContextKeys.TOOL_INPUT: dict(_INPUT),
            ContextKeys.OBSERVATIONS: [],
            ContextKeys.AGENT_TRACE: [],
        }
        with _Capture() as lines:
            delta = handlers.execute_tool(context)
        tool.assert_got_real_input()
        assert delta[ContextKeys.TOOL_STATUS] == "success"
        assert _SECRET not in _dump(delta)
        assert _SECRET not in "\n".join(lines)
        trace_step = delta[ContextKeys.AGENT_TRACE][0]
        assert trace_step["tool_input"]["q"] == "weather"
        assert "api_key" in trace_step["tool_input"]
        assert "weather" in delta[ContextKeys.OBSERVATIONS][0]


class TestSiblingExecutors:
    """The same redaction on every other executor that records tool input."""

    def test_parallel_react_dispatch(self):
        tool = _Tool()
        agent = ParallelReactAgent(
            tools=tool.registry, config=AgentConfig(model="mock/model")
        )
        context = {
            TOOL_CALLS_KEY: [{"tool_name": "lookup", "tool_input": dict(_INPUT)}],
            ContextKeys.OBSERVATIONS: [],
            ContextKeys.AGENT_TRACE: [],
        }
        with _Capture() as lines:
            delta = agent._dispatch_parallel(context)
        tool.assert_got_real_input()
        assert _SECRET not in _dump(delta)
        assert _SECRET not in "\n".join(lines)

    def test_rewoo_execute_all_plans(self):
        tool = _Tool()
        agent = REWOOAgent(tools=tool.registry, config=AgentConfig(model="mock/model"))
        context = {
            ContextKeys.PLAN_BLUEPRINT: [
                {"plan_id": 1, "tool_name": "lookup", "tool_input": dict(_INPUT)}
            ],
        }
        with _Capture() as lines:
            delta = agent._execute_all_plans(context)
        tool.assert_got_real_input()
        assert _SECRET not in _dump(delta[ContextKeys.AGENT_TRACE])
        assert _SECRET not in "\n".join(lines)

    def test_native_fc_trace(self):
        tool = _Tool()
        turns = iter(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "name": "lookup", "arguments": dict(_INPUT)}
                    ],
                },
                {"content": "done", "tool_calls": []},
            ]
        )

        def complete_fn(model, messages, schemas):
            return next(turns)

        agent = NativeFunctionCallingReactAgent(
            tools=tool.registry,
            config=AgentConfig(model="mock/model"),
            complete_fn=complete_fn,
        )
        with _Capture() as lines:
            result = agent.run("q")
        tool.assert_got_real_input()
        assert [c.tool_name for c in result.trace.tool_calls] == ["lookup"]
        assert _SECRET not in result.trace.model_dump_json()
        assert _SECRET not in _dump(result.final_context)
        assert _SECRET not in "\n".join(lines)

    def test_reasoning_react_reason_tool(self):
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        registry = ToolRegistry()
        registry.register_function(lambda q: "x", name="noop", description="No-op")
        agent = ReasoningReactAgent(
            tools=registry, config=AgentConfig(model="mock/model")
        )
        executor = agent._make_reasoning_tool_executor(AgentHandlers(agent.tools))
        problems: list[str] = []

        def solve(problem: str) -> tuple[str, dict[str, Any]]:
            problems.append(problem)
            return "solved", {}

        context = {
            ContextKeys.TOOL_NAME: "reason",
            ContextKeys.TOOL_INPUT: {"question": "why", "api_key": _SECRET},
            ContextKeys.OBSERVATIONS: [],
            ContextKeys.AGENT_TRACE: [],
        }
        with (
            patch.object(agent._reasoning_engine, "solve_problem", side_effect=solve),
            _Capture() as lines,
        ):
            delta = executor(context)
        assert len(problems) == 1 and _SECRET in problems[0]  # engine gets it
        assert _SECRET not in _dump(delta)
        assert _SECRET not in "\n".join(lines)


class TestApprovalSummary:
    """SEC-06: the approval context summary drops secret-looking entries."""

    def test_summary_drops_forbidden_parameters_stay_raw(self):
        seen: list[ApprovalRequest] = []

        def approve(request: ApprovalRequest) -> bool:
            seen.append(request)
            return True

        hitl = HumanInTheLoop(approval_callback=approve)
        call = ToolCall(tool_name="send", parameters=dict(_INPUT))
        context = {
            "task": "send it",
            "password": "hunter2",
            "profile": {"name": "Bob", "api_key": _SECRET},
        }
        with _Capture() as lines:
            assert hitl.request_approval(call, context) is True
        summary = seen[0].context_summary
        assert "password" not in summary
        assert summary["task"] == "send it"
        assert summary["profile"]["name"] == "Bob"
        assert "hunter2" not in _dump(summary)
        assert _SECRET not in _dump(summary)
        # D-016: the approver must see the exact call.
        assert seen[0].parameters == _INPUT
        assert _SECRET not in "\n".join(lines)
