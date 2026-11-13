"""Tool timeouts end as an unknown outcome: bound, status mapping, parallel batches, PlanExecute, REWOO, native_fc.

A timed-out call may have run its side effect, so every executor reports ``unknown``, never ``failed``,
and PlanExecute never re-runs such a step (plan 944e2692 review fixes D-032, D-041 to D-044).
"""

from __future__ import annotations

import contextvars
import math
import threading
import time
from typing import Any

import pydantic
import pytest

from fsm_llm.agents import (
    AgentConfig,
    NativeFunctionCallingReactAgent,
    PlanExecuteAgent,
    REWOOAgent,
    ToolAnnotations,
)
from fsm_llm.agents.constants import ContextKeys, StopReason, ToolRunStatus
from fsm_llm.agents.handlers import AgentHandlers
from fsm_llm.agents.hitl import HumanInTheLoop
from fsm_llm.agents.parallel_react import ParallelReactAgent
from fsm_llm.agents.tool_registries import RetryingToolRegistry
from fsm_llm.testing import PromptGroundedLLM
from fsm_llm.tools import ToolCall, ToolDefinition, ToolRegistry, ToolResult
from tests.test_fsm_llm_agents import test_native_fc_golden as golden
from tests.test_fsm_llm_agents.test_toolspec import (
    _approved_context,
    _policy_gates_everything,
)


def _hanging_pay(release: threading.Event, ran: list[str]) -> Any:
    def pay(amount: str) -> str:
        release.wait(5.0)
        ran.append(amount)
        return "paid"

    return pay


def _hanging_registry(
    release: threading.Event, ran: list[str], *, gated: bool = True
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        _hanging_pay(release, ran),
        name="pay",
        description="Pay an amount.",
        requires_approval=gated,
        timeout_seconds=0.1,
    )
    return registry


@pytest.fixture
def release() -> Any:
    event = threading.Event()
    yield event
    event.set()


class TestToolResultStatus:
    @pytest.mark.parametrize(
        ("result", "status"),
        [
            (ToolResult(tool_name="t", success=True, result=1), "success"),
            (ToolResult(tool_name="t", success=False, error="e"), "failed"),
            (
                ToolResult(tool_name="t", success=False, error="e", timed_out=True),
                "unknown",
            ),
        ],
        ids=["success", "failed", "timed_out"],
    )
    def test_status_mapping(self, result, status):
        assert result.status == status

    def test_unknown_is_a_constant(self):
        assert ToolRunStatus.UNKNOWN == "unknown"


class TestTimedOutCallIsUnknown:
    def test_executor_context_and_trace_say_unknown(self, release):
        ran: list[str] = []
        handlers = AgentHandlers(
            _hanging_registry(release, ran), requires_approval=_policy_gates_everything
        )
        delta = handlers.execute_tool(_approved_context())
        assert delta[ContextKeys.TOOL_STATUS] == "unknown"
        assert delta[ContextKeys.AGENT_TRACE][-1][ContextKeys.TOOL_STATUS] == "unknown"

    def test_the_approvers_next_request_sees_unknown_not_failed(self, release):
        """Pass 9 hitl_timeout.py: after an approved ``pay`` times out, the
        approver's ``context_summary`` said ``tool_status='failed'``."""
        ran: list[str] = []
        handlers = AgentHandlers(
            _hanging_registry(release, ran), requires_approval=_policy_gates_everything
        )
        context = _approved_context()
        context.update(handlers.execute_tool(context))
        seen: list[Any] = []
        hitl = HumanInTheLoop(
            approval_callback=lambda request: seen.append(request) or True
        )
        call = ToolCall(tool_name="pay", parameters={"amount": "9"})
        hitl.request_approval(call, {**context, ContextKeys.APPROVAL_REQUIRED: True})
        assert seen[0].context_summary[ContextKeys.TOOL_STATUS] == "unknown"

    def test_success_and_failure_statuses_are_unchanged(self):
        registry = ToolRegistry()
        registry.register_function(lambda amount: "paid", name="pay", description="d")
        context = {**_approved_context(), ContextKeys.DRIVER_APPROVAL: None}
        delta = AgentHandlers(registry).execute_tool(context)
        assert delta[ContextKeys.TOOL_STATUS] == "success"

        def broken(amount: str) -> str:
            raise RuntimeError("down")

        failing = ToolRegistry()
        failing.register_function(broken, name="pay", description="d")
        delta = AgentHandlers(failing).execute_tool(context)
        assert delta[ContextKeys.TOOL_STATUS] == "failed"

    def test_parallel_batch_with_a_timeout_is_unknown(self, release):
        ran: list[str] = []
        registry = _hanging_registry(release, ran, gated=False)
        registry.register_function(lambda q: q, name="echo", description="Echo q")
        agent = ParallelReactAgent(registry, config=AgentConfig(model="m"))
        delta = agent._dispatch_parallel(
            {
                "tool_calls": [
                    {"tool_name": "echo", "tool_input": {"q": "hi"}},
                    {"tool_name": "pay", "tool_input": {"amount": "9"}},
                ],
                ContextKeys.AGENT_TRACE: [],
                ContextKeys.OBSERVATIONS: [],
            }
        )
        assert delta[ContextKeys.TOOL_STATUS] == "unknown"
        statuses = [s[ContextKeys.TOOL_STATUS] for s in delta[ContextKeys.AGENT_TRACE]]
        assert statuses == ["success", "unknown"]

    def test_parallel_batch_without_a_timeout_keeps_any_success(self):
        registry = ToolRegistry()
        registry.register_function(lambda q: q, name="echo", description="Echo q")
        agent = ParallelReactAgent(registry, config=AgentConfig(model="m"))
        delta = agent._dispatch_parallel(
            {
                "tool_calls": [
                    {"tool_name": "echo", "tool_input": {"q": "hi"}},
                    {"tool_name": "missing", "tool_input": {}},
                ],
                ContextKeys.AGENT_TRACE: [],
            }
        )
        assert delta[ContextKeys.TOOL_STATUS] == "success"

    def test_rewoo_evidence_status_says_unknown(self, release):
        ran: list[str] = []
        agent = REWOOAgent(
            _hanging_registry(release, ran, gated=False),
            config=AgentConfig(model="m"),
        )
        delta = agent._execute_all_plans(
            {
                ContextKeys.PLAN_BLUEPRINT: [
                    {"plan_id": 1, "tool_name": "pay", "tool_input": {"amount": "9"}}
                ]
            }
        )
        entry = delta[ContextKeys.EVIDENCE_STATUS][0]
        assert entry["success"] is False
        assert entry[ContextKeys.TOOL_STATUS] == "unknown"
        assert delta[ContextKeys.AGENT_TRACE][-1][ContextKeys.TOOL_STATUS] == "unknown"

    def test_native_trace_says_unknown(self, monkeypatch, release):
        ran: list[str] = []
        registry = golden._registry()
        registry.register_function(
            _hanging_pay(release, ran),
            name="pay",
            description="Pay an amount.",
            timeout_seconds=0.1,
        )
        recorder = golden._Recorder(
            [golden._calls(("call_1", "pay", {"amount": "9"})), golden._final("x")]
        )
        for binding in golden._BINDINGS:
            monkeypatch.setattr(binding, recorder)
        agent = NativeFunctionCallingReactAgent(
            registry, config=AgentConfig(model="gpt-4o-mini", max_iterations=5)
        )
        result = agent.run("t")
        step = result.final_context[ContextKeys.AGENT_TRACE][-1]
        assert step[ContextKeys.TOOL_STATUS] == "unknown"
        tool_turn = recorder.requests[1]["messages"][-1]
        assert tool_turn["content"].startswith("[TOOL OUTCOME UNKNOWN]")


class TestPlanExecuteNeverReRunsAnUnknownStep:
    """Pass 9 W4: PlanExecute read a timed-out step as ``step_failed``, so a
    step whose side effect may have happened was replanned and re-run."""

    def _checker(self) -> Any:
        registry = ToolRegistry()
        registry.register_function(lambda amount: "paid", name="pay", description="d")
        return PlanExecuteAgent(
            registry, config=AgentConfig(model="m"), max_replans=2
        )._make_result_checker()

    def _context(self, status: str, result: str) -> dict[str, Any]:
        return {
            ContextKeys.PLAN_STEPS: ["pay the invoice", "email the receipt"],
            ContextKeys.CURRENT_STEP_INDEX: 0,
            ContextKeys.STEP_RESULTS: [],
            ContextKeys.TOOL_STATUS: status,
            ContextKeys.TOOL_RESULT: result,
            "_replan_count": 0,
        }

    def test_unknown_step_goes_to_synthesis_not_replan(self):
        delta = self._checker()(
            self._context("unknown", "[TOOL OUTCOME UNKNOWN] Error: timed out")
        )
        assert delta[ContextKeys.STEP_FAILED] is False
        assert delta[ContextKeys.ALL_STEPS_COMPLETE] is True
        assert delta[ContextKeys.FORCED_STOP_REASON] == StopReason.NO_RESULT
        assert ContextKeys.CURRENT_STEP_INDEX not in delta
        entry = delta[ContextKeys.STEP_RESULTS][-1]
        assert entry["success"] is False
        assert "OUTCOME UNKNOWN" in entry["result"]

    def test_failed_step_still_replans(self):
        delta = self._checker()(self._context("failed", "[TOOL FAILED] Error: x"))
        assert delta[ContextKeys.STEP_FAILED] is True
        assert ContextKeys.ALL_STEPS_COMPLETE not in delta

    def test_successful_step_still_advances(self):
        delta = self._checker()(self._context("success", "paid"))
        assert delta[ContextKeys.STEP_FAILED] is False
        assert delta[ContextKeys.CURRENT_STEP_INDEX] == 1
        assert delta[ContextKeys.STEP_RESULTS][-1]["success"] is True

    def test_end_to_end_a_timed_out_step_runs_once(self, release):
        """Step 1 succeeds, step 2 times out, step 3 never runs. Review round
        3 (pass 13 W1, D-047): the cut-short plan reported ``success=True``
        on step 1's evidence at f85e07a; it is ``no_result`` now."""
        ran: list[str] = []
        registry = ToolRegistry()

        def pay(amount: str) -> str:
            ran.append(amount)
            if len(ran) >= 2:
                release.wait(5.0)
            return "paid " + amount

        registry.register_function(
            pay, name="pay", description="Pay an amount.", timeout_seconds=0.1
        )
        llm = PromptGroundedLLM(
            facts={
                "plan_steps": (
                    ["pay the invoice", "pay the second invoice", "email it"],
                    "invoice",
                ),
                "tool_name": ("pay", "invoice"),
                "tool_input": ({"amount": "9"}, "invoice"),
                "step_result": ("paying", "invoice"),
            },
            default_response="All three steps completed.",
        )
        agent = PlanExecuteAgent(
            registry,
            config=AgentConfig(model="m", max_iterations=10),
            max_replans=2,
            llm_interface=llm,
        )
        calls: list[str] = []
        original = registry.execute

        def counting(tool_call: ToolCall, *, gated: bool = False) -> ToolResult:
            calls.append(tool_call.tool_name)
            return original(tool_call, gated=gated)

        registry.execute = counting  # type: ignore[method-assign]
        result = agent.run("pay the invoice, pay the second invoice and email it")
        assert calls == ["pay", "pay"], "the unknown-outcome step was run again"
        assert result.final_context.get("_replan_count", 0) == 0
        assert result.final_context[ContextKeys.STEP_FAILED] is False
        assert [e["success"] for e in result.final_context["step_results"]] == [
            True,
            False,
        ]
        assert result.success is False
        assert result.stop_reason == StopReason.NO_RESULT


def _counting_tool(ran: list[int], seconds: float = 0.0) -> Any:
    def slow(x: int) -> str:
        time.sleep(seconds)
        ran.append(x)
        return "ok"

    return slow


class TestTimeoutBound:
    @pytest.mark.parametrize("bad", [math.inf, 1e12, math.nan])
    def test_definition_refuses_an_unwaitable_timeout(self, bad):
        with pytest.raises(pydantic.ValidationError):
            ToolDefinition(
                name="t", description="d", execute_fn=lambda: 1, timeout_seconds=bad
            )

    @pytest.mark.parametrize("bad", [math.inf, 1e12])
    def test_register_function_refuses_it(self, bad):
        with pytest.raises(ValueError):
            ToolRegistry().register_function(
                _counting_tool([]), name="slow", description="d", timeout_seconds=bad
            )

    def test_the_largest_waitable_timeout_is_accepted(self):
        definition = ToolDefinition(
            name="t",
            description="d",
            execute_fn=lambda: 1,
            timeout_seconds=threading.TIMEOUT_MAX,
        )
        assert definition.timeout_seconds == threading.TIMEOUT_MAX

    def test_a_timeout_set_after_registration_never_runs_the_tool(self):
        """Bypassing the field check is refused before the worker starts.

        On the parent the worker ran the tool, then the wait raised
        ``OverflowError``: a tool that ran was reported failed and a
        retry-safe one ran three times.
        """
        ran: list[int] = []
        registry = RetryingToolRegistry(max_retries=2)
        registry.register_function(
            _counting_tool(ran),
            name="slow",
            description="d",
            timeout_seconds=1.0,
            annotations=ToolAnnotations(idempotent=True),
        )
        registry.get("slow").timeout_seconds = 1e12
        result = registry.execute(ToolCall(tool_name="slow", parameters={"x": 1}))
        time.sleep(0.2)
        assert ran == []
        assert result.success is False
        assert "invalid timeout_seconds" in (result.error or "")
        assert result.timed_out is False


class TestTimedOutOutcomeIsUnknown:
    def _registry(self, release: threading.Event, ran: list[str]) -> ToolRegistry:
        def pay(amount: str) -> str:
            release.wait(5.0)
            ran.append(amount)
            return "paid"

        registry = ToolRegistry()
        registry.register_function(
            pay,
            name="pay",
            description="d",
            requires_approval=True,
            timeout_seconds=0.1,
        )
        return registry

    def test_result_is_marked_timed_out_and_says_so(self):
        release, ran = threading.Event(), []
        result = self._registry(release, ran).execute(
            ToolCall(tool_name="pay", parameters={"amount": "9"})
        )
        release.set()
        assert result.success is False
        assert result.timed_out is True
        assert "outcome is unknown" in (result.error or "")
        assert "may still happen" in (result.error or "")
        assert result.observation.startswith("[TOOL OUTCOME UNKNOWN] ")

    def test_hitl_path_tells_the_model_the_outcome_is_unknown(self):
        """An approved (gated) call that times out is not shown as failed."""
        release, ran = threading.Event(), []
        registry = self._registry(release, ran)
        handlers = AgentHandlers(registry, requires_approval=_policy_gates_everything)
        delta = handlers.execute_tool(_approved_context())
        release.set()
        observation = delta[ContextKeys.OBSERVATIONS][-1]
        assert "[TOOL OUTCOME UNKNOWN]" in observation
        assert "[TOOL FAILED]" not in observation
        assert "outcome is unknown" in observation

    def test_failure_and_success_observations_are_unchanged(self):
        failed = ToolResult(tool_name="t", success=False, error="boom")
        assert failed.observation == "[TOOL FAILED] Error: boom"
        ok = ToolResult(tool_name="t", success=True, result="fine")
        assert ok.observation == "fine"

    def test_worker_sees_the_callers_context_variables(self):
        """Review NOTE 5 (mutant M6): the worker runs in a copy of the
        caller's context variables."""
        var: contextvars.ContextVar[str] = contextvars.ContextVar("v", default="-")
        var.set("caller")

        def read() -> str:
            return var.get()

        registry = ToolRegistry()
        registry.register_function(
            read, name="read", description="d", timeout_seconds=5.0
        )
        assert registry.execute(ToolCall(tool_name="read")).result == "caller"
