"""Agents foundation fixes (plan 5980fbb7 step 10): each class pins one finding.

F-AB-01 (loguru has no ``exc_info``), F-AB-02/F-AR-02/F-AI-08 (one observation
recorder), F-AB-03 (a failed ``reason`` call is an observation), F-AB-04 (timed
approval runs like a timed tool), F-AB-07 (``from_functions`` without
``__name__``), F-AB-06/F-AP-14 (one tool normaliser), F-AB-11/F-AI-05 (memory
tools built by ``@tool``).
"""

from __future__ import annotations

import contextvars
import functools
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from fsm_llm.agents.constants import ContextKeys, Defaults
from fsm_llm.agents.definitions import (
    ApprovalRequest,
    EvaluationResult,
    ToolCall,
    ToolDefinition,
)
from fsm_llm.agents.tools import ToolRegistry
from fsm_llm.logging import enable_library_logging, logger

SRC = Path(__file__).resolve().parents[2] / "src" / "fsm_llm"


@pytest.fixture
def logging_on():
    """Library logging as a host enables it, with a sink that formats records."""
    enable_library_logging()
    captured: list[str] = []
    sink = logger.add(lambda msg: captured.append(str(msg)), level="DEBUG")
    try:
        yield captured
    finally:
        logger.remove(sink)
        logger.disable("fsm_llm")


class TestEvaluationFnNeverRaises:
    """F-AB-01: the error text is data, not a format string."""

    def test_braces_in_the_error_text_with_logging_on(self, logging_on):
        from fsm_llm.agents.handlers import run_evaluation_fn

        def verifier(_ctx):
            raise RuntimeError('verifier said {"ok": false}')

        result = run_evaluation_fn(verifier, {})
        assert isinstance(result, EvaluationResult)
        assert result.passed is False
        assert '{"ok": false}' in result.feedback
        assert any('{"ok": false}' in line for line in logging_on)

    def test_no_loguru_call_in_src_passes_exc_info(self):
        offenders = [
            f"{path.relative_to(SRC)}:{n}"
            for path in SRC.rglob("*.py")
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if re.search(r"\bexc_info\s*=", line)
        ]
        assert offenders == []


def _context(n_observations=0, tool_input=None):
    return {
        ContextKeys.TASK: "Is 97 prime?",
        ContextKeys.TOOL_NAME: "reason",
        ContextKeys.TOOL_INPUT: tool_input if tool_input is not None else {},
        ContextKeys.REASONING: "needs a proof",
        ContextKeys.OBSERVATIONS: [f"old {i}" for i in range(n_observations)],
        ContextKeys.AGENT_TRACE: [{"old": i} for i in range(n_observations)],
    }


def _reason_executor():
    from fsm_llm.agents.handlers import AgentHandlers
    from fsm_llm.agents.reasoning_react import ReasoningReactAgent

    agent = ReasoningReactAgent(tools=ToolRegistry())
    executor = agent._make_reasoning_tool_executor(AgentHandlers(agent.tools))
    return agent, executor


class TestReasonPathRecordsLikeEveryExecutor:
    """F-AB-02 / F-AR-02 / F-AI-08: caps, prune and trace input on ``reason``."""

    def test_entry_is_capped_pruned_and_traced(self):
        agent, executor = _reason_executor()
        with patch.object(
            agent._reasoning_engine, "solve_problem", return_value=("x" * 10_000, {})
        ):
            delta = executor(_context(25, {"problem": "Is 91 prime?"}))
        observations = delta[ContextKeys.OBSERVATIONS]
        assert len(observations) == Defaults.MAX_OBSERVATIONS
        assert delta[ContextKeys.OBSERVATION_COUNT] == Defaults.MAX_OBSERVATIONS
        assert len(observations[-1]) <= Defaults.MAX_OBSERVATION_LENGTH
        assert observations[-1].startswith("[Step 26] Tool: reason | Input: ")
        step = delta[ContextKeys.AGENT_TRACE][-1]
        assert step["tool_input"] == {"problem": "Is 91 prime?"} or "91" in str(
            step["tool_input"]
        )
        assert step[ContextKeys.TOOL_STATUS] == "success"
        assert step["action"].startswith("reason(")

    def test_trace_rebuilds_the_reason_parameters(self):
        agent, executor = _reason_executor()
        with patch.object(
            agent._reasoning_engine, "solve_problem", return_value=("yes", {})
        ):
            delta = executor(_context(0, {"problem": "Is 91 prime?"}))
        calls = agent._build_trace(
            {ContextKeys.AGENT_TRACE: delta[ContextKeys.AGENT_TRACE]}, 1
        ).tool_calls
        assert calls and calls[0].parameters


class TestFailedReasonCallIsAnObservation:
    """F-AB-03: the model learns the call failed (and logging cannot raise)."""

    def test_failure_appends_a_failed_observation(self, logging_on):
        agent, executor = _reason_executor()
        error = RuntimeError('engine said {"budget": 170}')
        with patch.object(agent._reasoning_engine, "solve_problem", side_effect=error):
            delta = executor(_context(0, {"problem": "p"}))
        assert delta[ContextKeys.TOOL_STATUS] == "failed"
        assert str(delta[ContextKeys.TOOL_RESULT]).startswith("Reasoning failed")
        (entry,) = delta[ContextKeys.OBSERVATIONS]
        assert "[TOOL FAILED]" in entry and "Tool: reason" in entry
        assert delta[ContextKeys.OBSERVATION_COUNT] == 1
        assert delta[ContextKeys.AGENT_TRACE][-1][ContextKeys.TOOL_STATUS] == "failed"
        assert delta[ContextKeys.SHOULD_TERMINATE] is None


class TestTimedApprovalRunsLikeATimedTool:
    """F-AB-04: context variables, SystemExit and a bad timeout."""

    @staticmethod
    def _call():
        return ToolCall(tool_name="send", parameters={})

    def test_a_context_variable_reaches_the_callback(self):
        from fsm_llm.agents.hitl import HumanInTheLoop

        request_id = contextvars.ContextVar("request_id", default="unset")
        seen: list[str] = []

        def approve(_request: ApprovalRequest) -> bool:
            seen.append(request_id.get())
            return True

        request_id.set("request-42")
        hitl = HumanInTheLoop(approval_callback=approve, approval_timeout=5.0)
        assert hitl.request_approval(self._call(), {}) is True
        assert seen == ["request-42"]

    def test_system_exit_is_raised_not_a_false_timeout(self):
        from fsm_llm.agents.hitl import HumanInTheLoop

        def closed(_request: ApprovalRequest) -> bool:
            raise SystemExit(0)

        hitl = HumanInTheLoop(approval_callback=closed, approval_timeout=5.0)
        with pytest.raises(SystemExit):
            hitl.request_approval(self._call(), {})

    @pytest.mark.parametrize("timeout", [0, -1.0, float("nan")])
    def test_a_bad_timeout_is_refused_not_a_silent_denial(self, timeout):
        from fsm_llm.agents.exceptions import ToolExecutionError
        from fsm_llm.agents.hitl import HumanInTheLoop

        hitl = HumanInTheLoop(
            approval_callback=lambda _request: True, approval_timeout=timeout
        )
        with pytest.raises(ToolExecutionError):
            hitl.request_approval(self._call(), {})


class TestFromFunctionsWithoutAName:
    """F-AB-07: a partial has no ``__name__``; the skip warning must not crash."""

    def test_a_partial_is_skipped_with_a_warning(self, logging_on):
        from fsm_llm.agents.python_tools import PythonToolLoader

        assert PythonToolLoader.from_functions(functools.partial(print)) == []
        assert any("no _tool_definition" in line for line in logging_on)


class TestOneToolNormaliser:
    """F-AB-06 / F-AP-14: ``create_agent`` and the builder take the same tools."""

    @staticmethod
    def _td():
        return ToolDefinition(
            name="lookup",
            description="Look something up",
            parameter_schema={},
            execute_fn=lambda params: "found",
        )

    def test_create_agent_accepts_a_tool_definition(self):
        from fsm_llm.agents import create_agent

        agent = create_agent("react", [self._td()])
        assert "lookup" in agent.tools

    def test_builder_and_create_agent_register_the_same_tools(self):
        from fsm_llm.agents import ConfiguredAgentBuilder, create_agent, tool

        @tool
        def plain(query: str) -> str:
            """Plain."""
            return query

        def bare(params):
            """Bare."""
            return "x"

        tools = [self._td(), plain, bare]
        builder = ConfiguredAgentBuilder()
        for item in tools:
            builder.add_tool(item)
        built = builder.build()
        made = create_agent("react", tools)
        assert built.tools.tool_names == made.tools.tool_names

    def test_an_empty_tool_list_is_no_tools(self):
        from fsm_llm.agents import create_agent

        assert create_agent("debate", tools=[]) is not None


class TestMemoryToolsAreBuiltByTheToolDecorator:
    """F-AB-11 / F-AI-05: the same extras as every ``@tool`` tool."""

    @staticmethod
    def _sets():
        from fsm_llm import WorkingMemory
        from fsm_llm.agents.memory_tools import create_memory_tools
        from fsm_llm.agents.semantic_memory import (
            SemanticMemoryStore,
            create_semantic_memory_tools,
        )

        return {
            "working": create_memory_tools(WorkingMemory()),
            "semantic": create_semantic_memory_tools(SemanticMemoryStore()),
        }

    def test_every_memory_tool_has_an_args_model(self):
        for tools in self._sets().values():
            assert all(t.args_model is not None for t in tools)

    def test_reads_are_annotated_retry_safe(self):
        by_name = {t.name: t for kind in self._sets().values() for t in kind}
        reads = [
            t
            for t in (*self._sets()["working"], *self._sets()["semantic"])
            if t.name in ("recall", "list_memories")
        ]
        assert reads and all(t.annotations.retry_safe for t in reads)
        assert not by_name["remember"].annotations.retry_safe
        assert not by_name["forget"].annotations.retry_safe
