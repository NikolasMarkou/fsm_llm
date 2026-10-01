"""Review round 2 fixes for agents tools and native function calling (plan
944e2692 step 18.6, D-037 item "18.6", review pass 9 warnings 1, 2, 4, 5;
decisions D-041 to D-044).

Each class pins one accepted finding and fails on the parent commit 1be35d7:
``$ref`` (Enum) parameters shown to prompt-mode models with their real type,
ParallelReact accepting an old-signature registry it never sends ``gated``,
a timed-out call reported as ``tool_status: unknown`` everywhere it is
visible (PlanExecute never re-runs it), and native_fc's ``run_tools`` state
exempt from the wall clock.
"""

from __future__ import annotations

import enum
import json
import threading
from typing import Any

import pytest

from fsm_llm.agents import (
    AgentConfig,
    NativeFunctionCallingReactAgent,
    PlanExecuteAgent,
    ReactAgent,
    REWOOAgent,
)
from fsm_llm.agents.constants import (
    ContextKeys,
    NativeFCStates,
    StopReason,
    ToolRunStatus,
)
from fsm_llm.agents.definitions import ToolCall, ToolResult
from fsm_llm.agents.exceptions import AgentError, AgentTimeoutError
from fsm_llm.agents.handlers import AgentHandlers
from fsm_llm.agents.hitl import HumanInTheLoop
from fsm_llm.agents.parallel_react import ParallelReactAgent
from fsm_llm.agents.prompts import build_think_extraction_instructions
from fsm_llm.agents.tools import ToolRegistry, schema_types
from tests.conftest import PromptGroundedLLM
from tests.test_fsm_llm_agents import test_native_fc_golden as golden
from tests.test_fsm_llm_agents.test_review_fixes_tools_native import (
    _OldRegistry,
    _timed_recorder,
    _with_tool,
)
from tests.test_fsm_llm_agents.test_toolspec import (
    _approved_context,
    _policy_gates_everything,
)

# ---------------------------------------------------------------------------
# W1: `$ref` parameters are described with the type they point to
# ---------------------------------------------------------------------------


class _Color(str, enum.Enum):
    RED = "red"
    BLUE = "blue"


class _Size(enum.IntEnum):
    S = 1
    L = 2


def _paint(color: _Color, size: _Size = _Size.S) -> str:
    """Paint the fence."""
    return f"{color.value} {int(size)}"


def _enum_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(_paint, name="paint")
    return registry


class TestRefParametersKeepTheirType:
    def test_description_shows_str_enum_as_string_and_int_enum_as_integer(self):
        text = _enum_registry().to_prompt_description()
        assert "color [REQUIRED] (string)" in text
        assert "size [optional] (integer)" in text
        assert "(object)" not in text

    def test_think_example_uses_the_referenced_type(self):
        text = build_think_extraction_instructions(_enum_registry())
        example = next(
            json.loads(line) for line in text.splitlines() if '"paint"' in line
        )
        assert example["tool_input"]["color"] == "<color>"
        assert example["tool_input"]["size"] == 0

    def test_prompt_mode_agent_is_shown_the_enum_types(self):
        """The text a ReAct agent's model actually receives."""
        llm = PromptGroundedLLM()
        agent = ReactAgent(
            _enum_registry(),
            config=AgentConfig(model="m", max_iterations=1),
            llm_interface=llm,
        )
        try:
            agent.run("paint the fence red")
        except AgentError:
            pass
        prompts = [request.system_prompt for request in llm.calls("extract_field")]
        assert prompts
        assert any("color [REQUIRED] (string)" in p for p in prompts)
        assert not any("color [REQUIRED] (object)" in p for p in prompts)

    def test_a_model_reference_is_still_an_object(self):
        root = {
            "properties": {"p": {"$ref": "#/$defs/M"}},
            "$defs": {"M": {"type": "object", "properties": {}}},
        }
        assert schema_types(root["properties"]["p"], root=root) == ["object"]

    @pytest.mark.parametrize(
        ("prop", "root"),
        [
            ({"$ref": "#/$defs/Missing"}, {"$defs": {}}),
            ({"$ref": "#/$defs/Color"}, None),
            ({"$ref": "https://example.com/schema.json"}, {"$defs": {}}),
        ],
        ids=["dangling", "no_root", "non_local"],
    )
    def test_an_unresolved_reference_reads_as_any_not_object(self, prop, root):
        assert schema_types(prop, root=root) == []

    def test_a_reference_cycle_terminates(self):
        root = {"$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}}}
        assert schema_types({"$ref": "#/$defs/A"}, root=root) == []

    def test_optional_enum_is_string_or_null(self):
        root = {
            "$defs": {"Color": {"enum": ["red"], "type": "string"}},
        }
        prop = {"anyOf": [{"$ref": "#/$defs/Color"}, {"type": "null"}]}
        assert schema_types(prop, root=root) == ["string", "null"]


# ---------------------------------------------------------------------------
# W2: agents that never pass `gated` accept an old-signature registry
# ---------------------------------------------------------------------------


class TestAgentsThatNeverPassGatedRunOldRegistries:
    """D-033 says REWOO, ParallelReact and native_fc call ``execute(call)``
    and so accept an override written against the old signature. On the
    parent ParallelReact refused it at ``run()`` start (it built an
    ``AgentHandlers`` only for the iteration limiter)."""

    def test_parallel_react_runs_the_old_registry_tool(self):
        ran: list[str] = []

        class _Recording(_OldRegistry):
            def execute(self, tool_call: ToolCall) -> ToolResult:  # type: ignore[override]
                ran.append(tool_call.tool_name)
                return super().execute(tool_call)

        llm = PromptGroundedLLM(
            facts={
                "tool_calls": (
                    [{"tool_name": "echo", "tool_input": {"q": "hi"}}],
                    "echo hi",
                ),
            }
        )
        agent = ParallelReactAgent(
            _with_tool(_Recording()),
            config=AgentConfig(model="m", max_iterations=2),
            llm_interface=llm,
        )
        agent.run("echo hi")
        assert llm.requests, "the run made no model call"
        assert "echo" in ran

    @pytest.mark.parametrize(
        "build",
        [
            lambda r: REWOOAgent(r, config=AgentConfig(model="m")),
            lambda r: NativeFunctionCallingReactAgent(r, config=AgentConfig(model="m")),
            lambda r: ParallelReactAgent(r, config=AgentConfig(model="m")),
        ],
        ids=["rewoo", "native_fc", "parallel_react"],
    )
    def test_construction_accepts_it(self, build):
        build(_with_tool(_OldRegistry()))


# ---------------------------------------------------------------------------
# W4: a timed-out call is `tool_status: unknown` everywhere it is visible
# ---------------------------------------------------------------------------


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
        timeout_s=0.1,
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
            timeout_s=0.1,
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
            pay, name="pay", description="Pay an amount.", timeout_s=0.1
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


# ---------------------------------------------------------------------------
# W5: native_fc's run_tools state is exempt from the wall clock (D-032)
# ---------------------------------------------------------------------------


class _RecordingRegistry(ToolRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.ran: list[str] = []

    def execute(self, tool_call: ToolCall, *, gated: bool = False) -> ToolResult:
        self.ran.append(tool_call.tool_name)
        return super().execute(tool_call, gated=gated)


def _recording_native(**config: Any) -> tuple[Any, _RecordingRegistry]:
    registry = _RecordingRegistry()
    for tool_def in golden._registry().list_tools():
        registry.register(tool_def)
    settings: dict[str, Any] = {"model": "gpt-4o-mini", **config}
    agent = NativeFunctionCallingReactAgent(registry, config=AgentConfig(**settings))
    return agent, registry


class TestRunToolsFinishesPastTheDeadline:
    """Each model call costs 1.2 s of a 1.0 s budget, so the deadline passes
    during the first calls turn. e1f63a9 ran that turn's tools; with
    ``RUN_TOOLS`` removed from ``SECONDS_EXEMPT`` the run raised before."""

    def test_run_tools_is_exempt(self):
        assert NativeFCStates.RUN_TOOLS in NativeFCStates.SECONDS_EXEMPT
        assert NativeFCStates.CALL_MODEL not in NativeFCStates.SECONDS_EXEMPT

    def test_exhausted_loop_still_runs_its_tools_and_the_forced_write(
        self, monkeypatch
    ):
        recorder = _timed_recorder(
            monkeypatch,
            [
                golden._calls(golden._PARIS),
                golden._calls(golden._NOTE),
                golden._final("x"),
            ],
            1.2,
        )
        agent, registry = _recording_native(
            max_iterations=1, timeout_seconds=1.0, force_final_tool="write_note"
        )
        result = agent.run("t")
        assert len(recorder.requests) == 2
        assert registry.ran == ["weather", "write_note"]
        assert [c.tool_name for c in result.trace.tool_calls] == [
            "weather",
            "write_note",
        ]

    def test_mid_loop_runs_the_turns_tools_then_times_out(self, monkeypatch):
        """e1f63a9 parity: the clock stops the run at the next model turn.
        (The tools run on ``run_tools`` entry, so this case alone does not
        tell the mutant apart; the exhausted case above does.)"""
        recorder = _timed_recorder(
            monkeypatch,
            [golden._calls(golden._PARIS), golden._final("x")],
            1.2,
        )
        agent, registry = _recording_native(max_iterations=5, timeout_seconds=1.0)
        with pytest.raises(AgentTimeoutError):
            agent.run("t")
        assert len(recorder.requests) == 1
        assert registry.ran == ["weather"]
