from __future__ import annotations

"""Tests for fsm_llm.agents.react module."""

import threading
from unittest.mock import MagicMock

import pytest

from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.definitions import AgentConfig
from fsm_llm.agents.handlers import AgentHandlers
from fsm_llm.agents.react import ReactAgent
from fsm_llm.core.definitions import (
    AgentError,
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm.testing import PromptGroundedLLM
from fsm_llm.tools import ToolRegistry


def _search(params):
    return f"Results for: {params.get('query', '')}"


def _calculate(params):
    return eval(params.get("expression", "0"))


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(_search, name="search", description="Search the web")
    registry.register_function(
        _calculate, name="calculate", description="Calculate expression"
    )
    return registry


class TestReactAgentCreation:
    """Tests for ReactAgent initialization."""

    def test_create_with_tools(self):
        registry = _make_registry()
        agent = ReactAgent(tools=registry)
        assert agent.tools is registry
        assert agent.config is not None
        assert agent.hitl is None

    def test_create_with_config(self):
        registry = _make_registry()
        config = AgentConfig(max_iterations=5, model="gpt-4o-mini")
        agent = ReactAgent(tools=registry, config=config)
        assert agent.config.max_iterations == 5
        assert agent.config.model == "gpt-4o-mini"

    def test_create_empty_registry_raises(self):
        registry = ToolRegistry()
        with pytest.raises(AgentError, match="empty tool registry"):
            ReactAgent(tools=registry)

    def test_create_with_hitl(self):
        from fsm_llm.policy import HumanInTheLoop

        registry = _make_registry()
        hitl = HumanInTheLoop(
            approval_policy=lambda call, ctx: True,
            approval_callback=lambda req: True,
        )
        agent = ReactAgent(tools=registry, hitl=hitl)
        assert agent.hitl is hitl


class TestReactAgentHitlGating:
    """Regression tests for F-01 (plan_2026-05-29_1d66f861 / D-001).

    The await_approval FSM state must be built whenever the runtime approval
    gate is active. Approval is policy-driven; a per-tool requires_approval
    attribute must NOT be required to activate gating, otherwise a policy-only
    HITL config (the documented usage) executes tools un-gated.
    """

    @staticmethod
    def _policy_only_hitl():
        from fsm_llm.policy import HumanInTheLoop

        # Tools from _make_registry() default to requires_approval=False.
        return HumanInTheLoop(
            approval_policy=lambda call, ctx: True,
            approval_callback=lambda req: True,
        )

    def test_hitl_active_true_for_policy_only(self):
        agent = ReactAgent(tools=_make_registry(), hitl=self._policy_only_hitl())
        assert agent._hitl_active is True

    def test_hitl_inactive_without_hitl(self):
        agent = ReactAgent(tools=_make_registry())
        assert agent._hitl_active is False

    def test_hitl_inactive_without_policy(self):
        from fsm_llm.policy import HumanInTheLoop

        hitl = HumanInTheLoop(approval_callback=lambda req: True)  # no policy
        agent = ReactAgent(tools=_make_registry(), hitl=hitl)
        assert agent._hitl_active is False

    def test_policy_only_builds_await_approval_state(self):
        # The FSM that run() would build for a policy-only HITL must contain the
        # await_approval gate state; otherwise the gate handler sets
        # approval_required=True with no state to intercept it and the tool
        # executes before _handle_hitl_approval can request approval.
        from fsm_llm.agents.fsm_definitions import build_react_fsm

        agent = ReactAgent(tools=_make_registry(), hitl=self._policy_only_hitl())
        fsm = build_react_fsm(agent.tools, include_approval_state=agent._hitl_active)
        assert "await_approval" in fsm["states"]


class _DeterministicMockLLM(LLMInterface):
    """Field-name-keyed mock LLM: every extract_field call's answer depends
    ONLY on the requested field_name, never on call order/index.

    This makes it safe to share a single instance across two CONCURRENT
    conversations (as F9's regression test below needs): whichever thread
    asks for a given field always gets the same deterministic answer,
    unlike a call-index-based mock (see SequenceMockLLM in
    support.py), which would itself race under concurrent use.
    """

    def __init__(self) -> None:
        self.model = "mock-model"

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        value = {
            "tool_name": "noop",
            "tool_input": {},
            "reasoning": "call the tool",
            "should_terminate": False,
            "final_answer": "done",
        }.get(request.field_name)
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=1.0 if value is not None else 0.0,
            reasoning="mock field extraction",
            is_valid=value is not None,
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="ok", message_type="response", reasoning="mock response"
        )


class TestReactAgentConcurrentRuns:
    """F9 regression: two overlapping run() calls on the SAME ReactAgent
    instance (the AgentServer /invoke concurrency scenario, remote.py) must
    not share an AgentHandlers instance, and must each execute their own
    tool call and reach their own correct answer.

    D-004's first fix (`self._handlers = AgentHandlers(...)` assigned in
    run(), read back later by `_register_handlers`) did NOT close this race:
    `_register_handlers` reads `self._handlers` well AFTER the assignment
    line (after `build_react_fsm()`, `_init_context()`, and entry into
    `base/runs.py::_standard_run`), so two threads could each reassign
    `self._handlers` before either's `_register_handlers` read it back — both
    calls would then bind to whichever assignment landed last. D-014 replaced
    this with a true call-local `handlers` variable threaded explicitly into
    `_register_handlers` via `_standard_run`'s `handlers=` parameter, never
    round-tripped through `self`. See decisions.md D-004/D-013/D-014.

    This test's barrier straddles the ACTUAL vulnerable window: the gap
    between `AgentHandlers` construction (the first line of `run()`) and its
    use inside `_register_handlers` (reached only after `build_react_fsm()` +
    `_init_context()` + entry into `_standard_run`). The prior version of this
    test synchronized threads inside `execute_tool`, strictly AFTER that
    window closes, so it could not detect the D-004 race (it passed 20/20
    alone and only failed ~35% of the time under load — a scheduler lottery,
    not a reliable reproduction). Patching `build_react_fsm` (called
    immediately after handlers construction, before `_register_handlers` runs)
    forces both threads to have already created/assigned their own handlers
    before either proceeds — the exact interleaving needed to force D-004's
    shared-instance collision, and the exact interleaving D-014's local
    variable is immune to.
    """

    def test_concurrent_run_uses_isolated_handler_instances(self, monkeypatch):
        registry = ToolRegistry()
        registry.register_function(
            lambda params: "noop-result", name="noop", description="No-op tool"
        )

        config = AgentConfig(max_iterations=3, model="mock/model")
        agent = ReactAgent(
            tools=registry, config=config, llm_interface=_DeterministicMockLLM()
        )

        # Warm up once, single-threaded, before installing the spy/barrier and
        # spawning the concurrent pair below. This is NOT part of the F9
        # regression being tested: pydantic's FSMDefinition model-build/
        # validation is itself not thread-safe on first use (an unrelated,
        # pre-existing library behavior), so the first-ever validation must
        # happen single-threaded or two genuinely simultaneous first-time
        # validations can raise spuriously. Every run() after this one hits
        # the already-built validator.
        agent.run("warm-up")

        seen_ids: list[int] = []
        seen_lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=10)

        import fsm_llm.agents.react as react_module

        original_build_react_fsm = react_module.build_react_fsm

        def spy_build_react_fsm(*args, **kwargs):
            # Both threads reach here only AFTER their own run()'s handlers
            # construction/assignment line has already executed — this is
            # the vulnerable window's far edge. Waiting on a 2-party barrier
            # here forces BOTH threads to have completed that line before
            # either is allowed to proceed into `_register_handlers`, which
            # is the exact interleaving that exposes D-004's shared-instance
            # collision (and that D-014's local variable cannot collide
            # under, since no thread ever overwrites another's local).
            barrier.wait()
            return original_build_react_fsm(*args, **kwargs)

        monkeypatch.setattr(react_module, "build_react_fsm", spy_build_react_fsm)

        original_execute_tool = AgentHandlers.execute_tool

        def spy_execute_tool(self, context):
            with seen_lock:
                seen_ids.append(id(self))
            return original_execute_tool(self, context)

        monkeypatch.setattr(AgentHandlers, "execute_tool", spy_execute_tool)

        results: dict[str, object] = {}

        def _run(key: str, task: str) -> None:
            results[key] = agent.run(task)

        t1 = threading.Thread(target=_run, args=("a", "Task A"))
        t2 = threading.Thread(target=_run, args=("b", "Task B"))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not t1.is_alive() and not t2.is_alive()
        # max_iterations=3 is 3 think turns and 2 tool turns per run (D-028 of
        # plan 06a5ec0a), each run on its own handler instance.
        assert len(seen_ids) == 4
        assert len(set(seen_ids)) == 2, (
            "both concurrent run() calls dispatched into the SAME "
            "AgentHandlers instance — the F9/D-014 fix regressed"
        )
        assert sorted(seen_ids.count(i) for i in set(seen_ids)) == [2, 2]
        # The mock never sets should_terminate, so each run ends on its own
        # iteration budget: a forced stop ships its answer with success=False
        # (D-011 of plan 06a5ec0a), identically for both isolated runs.
        for key in ("a", "b"):
            assert results[key].answer == "ok"
            assert results[key].success is False
            assert results[key].stop_reason == "max_iterations"
        assert len(results["a"].trace.tool_calls) == 2
        assert len(results["b"].trace.tool_calls) == 2


class _GatedToolMockLLM(_DeterministicMockLLM):
    """Selects the approval-gated ``danger`` tool on every think turn."""

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        value = {
            "tool_name": "danger",
            "tool_input": {"x": "1"},
            "reasoning": "call the gated tool",
            "should_terminate": False,
        }.get(request.field_name, "text")
        return FieldExtractionResponse(
            field_name=request.field_name,
            value=value,
            confidence=0.9,
            reasoning="mock field extraction",
            is_valid=True,
        )


class TestReactAgentApprovalIsSingleUse:
    """D-015: an approval covers the one tool call it was granted for.

    The policy gates every call, so each execution of ``danger`` must be
    preceded by its own approval request. Before D-015 ``approval_granted``
    was never reset: after one grant the callback was skipped and
    ``await_approval`` routed to ``act`` on the stale True.
    """

    @staticmethod
    def _run(decide):
        from fsm_llm.core.definitions import AgentError
        from fsm_llm.policy import HumanInTheLoop

        executions: list[dict] = []
        asks: list[bool] = []

        def danger(params):
            executions.append(params)
            return "done"

        def callback(request):
            approved = decide(len(asks))
            asks.append(approved)
            return approved

        registry = ToolRegistry()
        registry.register_function(
            danger,
            name="danger",
            description="Gated tool",
            parameter_schema={"properties": {"x": {"type": "string"}}},
        )
        hitl = HumanInTheLoop(
            approval_policy=lambda call, ctx: True, approval_callback=callback
        )
        agent = ReactAgent(
            tools=registry,
            config=AgentConfig(max_iterations=8, model="mock/model"),
            llm_interface=_GatedToolMockLLM(),
            hitl=hitl,
        )
        try:
            agent.run("do it")
        except AgentError:
            pass
        return asks, executions

    def test_approval_of_first_call_does_not_cover_later_calls(self):
        asks, executions = self._run(lambda n: n == 0)
        assert len(asks) >= 2, "a later gated call skipped the approval callback"
        assert len(executions) == 1

    def test_each_gated_execution_is_asked_for(self):
        asks, executions = self._run(lambda n: True)
        assert len(executions) >= 2
        # One ask per execution; the last grant may be cut off by the budget
        # before its act turn runs.
        assert len(asks) - len(executions) in (0, 1)

    def test_denial_then_new_call_asks_again(self):
        asks, executions = self._run(lambda n: n >= 1)
        assert asks[:2] == [False, True]
        assert len(executions) >= 1
        assert asks.count(True) - len(executions) in (0, 1)


class TestReactFamilySharedSkeleton:
    """F-AR-01, F-AR-09, F-AR-11: the ReAct family shares its run pieces."""

    @staticmethod
    def _registry() -> ToolRegistry:
        registry = ToolRegistry()
        registry.register_function(lambda p: "ok", name="t", description="a tool")
        return registry

    def test_run_and_run_stream_build_the_same_fsm_and_context(self):
        agent = ReactAgent(tools=self._registry(), llm_interface=PromptGroundedLLM())
        fsm_def, context, handlers = agent._react_inputs("a task", {"hint": "x"})
        again = agent._react_inputs("a task", {"hint": "x"})
        assert fsm_def == again[0]
        assert context == again[1]
        assert handlers is not again[2]  # call-local, never shared
        assert context[ContextKeys.OBSERVATIONS] == []
        assert context[ContextKeys.ITERATION_BUDGET] == agent.config.max_iterations

    @pytest.mark.parametrize(
        "module, cls",
        [
            ("react", "ReactAgent"),
            ("reflexion", "ReflexionAgent"),
            ("reasoning_react", "ReasoningReactAgent"),
            ("parallel_react", "ParallelReactAgent"),
        ],
    )
    def test_register_handlers_without_handlers_names_the_class(self, module, cls):
        import importlib

        agent_cls = getattr(importlib.import_module(f"fsm_llm.agents.{module}"), cls)
        agent = agent_cls(tools=self._registry(), llm_interface=PromptGroundedLLM())
        with pytest.raises(AgentError, match=f"{cls}._register_handlers called"):
            agent._register_handlers(MagicMock(), None)

    @pytest.mark.parametrize("with_fn", [False, True])
    def test_reflexion_registers_one_evaluator_handler(self, with_fn):
        from fsm_llm.agents import ReflexionAgent
        from fsm_llm.agents.constants import HandlerNames
        from fsm_llm.agents.definitions import EvaluationResult

        agent = ReflexionAgent(
            tools=self._registry(),
            llm_interface=PromptGroundedLLM(),
            evaluation_fn=(lambda ctx: EvaluationResult(passed=True))
            if with_fn
            else None,
        )
        api = MagicMock()
        # the registration reads the call-local HandlerContext (plan a555dcfe step 10)
        agent._register_handlers(api, agent._handler_context())
        names = [c.args[0].name for c in api.register_handler.call_args_list]
        assert names.count(HandlerNames.REFLEXION_EVALUATOR) == 1

    def test_reflexion_shares_the_react_loop_states(self):
        from fsm_llm.agents.constants import AgentStates, ReflexionStates

        assert ReflexionStates.THINK is AgentStates.THINK
        assert ReflexionStates.ACT is AgentStates.ACT
        assert ReflexionStates.CONCLUDE is AgentStates.CONCLUDE

    def test_rewoo_trace_is_the_base_trace_and_keeps_a_name_with_a_paren(self):
        from fsm_llm.agents.rewoo import REWOOAgent

        assert "_build_trace" not in vars(REWOOAgent)
        agent = REWOOAgent(tools=self._registry(), llm_interface=PromptGroundedLLM())
        trace = agent._build_trace(
            {
                ContextKeys.AGENT_TRACE: [
                    {
                        "action": "calc(x)(1)",
                        "tool_name": "calc(x)",
                        "tool_input": {"a": 1},
                        "thought": "why",
                    },
                    {"action": "plain(2)", "thought": "t"},
                ],
                ContextKeys.ITERATION_COUNT: 2,
            },
            2,
        )
        assert [c.tool_name for c in trace.tool_calls] == ["calc(x)", "plain"]
        assert trace.tool_calls[0].parameters == {"a": 1}
        assert trace.tool_calls[0].reasoning == "why"
