from __future__ import annotations

"""Tests for fsm_llm_agents.react module."""

import threading

import pytest

from fsm_llm.definitions import (
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm_agents.definitions import AgentConfig
from fsm_llm_agents.exceptions import AgentError
from fsm_llm_agents.handlers import AgentHandlers
from fsm_llm_agents.react import ReactAgent
from fsm_llm_agents.tools import ToolRegistry


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
        from fsm_llm_agents.hitl import HumanInTheLoop

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
        from fsm_llm_agents.hitl import HumanInTheLoop

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
        from fsm_llm_agents.hitl import HumanInTheLoop

        hitl = HumanInTheLoop(approval_callback=lambda req: True)  # no policy
        agent = ReactAgent(tools=_make_registry(), hitl=hitl)
        assert agent._hitl_active is False

    def test_policy_only_builds_await_approval_state(self):
        # The FSM that run() would build for a policy-only HITL must contain the
        # await_approval gate state; otherwise the gate handler sets
        # approval_required=True with no state to intercept it and the tool
        # executes before _handle_hitl_approval can request approval.
        from fsm_llm_agents.fsm_definitions import build_react_fsm

        agent = ReactAgent(tools=_make_registry(), hitl=self._policy_only_hitl())
        fsm = build_react_fsm(agent.tools, include_approval_state=agent._hitl_active)
        assert "await_approval" in fsm["states"]


class TestReactAgentIntegration:
    """Integration tests for ReactAgent.run() — require mocking LLM."""

    @pytest.mark.slow
    def test_run_requires_llm(self):
        """ReactAgent.run() needs a real or mock LLM — skip in unit tests."""
        pytest.skip("Requires LLM interface — run with real_llm marker")


class _DeterministicMockLLM(LLMInterface):
    """Field-name-keyed mock LLM: every extract_field call's answer depends
    ONLY on the requested field_name, never on call order/index.

    This makes it safe to share a single instance across two CONCURRENT
    conversations (as F9's regression test below needs): whichever thread
    asks for a given field always gets the same deterministic answer,
    unlike a call-index-based mock (see SequenceMockLLM in
    test_bug_fixes.py), which would itself race under concurrent use.
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
    `base.py::_standard_run`), so two threads could each reassign
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

        import fsm_llm_agents.react as react_module

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
        assert len(seen_ids) == 2
        assert seen_ids[0] != seen_ids[1], (
            "both concurrent run() calls dispatched into the SAME "
            "AgentHandlers instance — the F9/D-014 fix regressed"
        )
        assert results["a"].success is True
        assert results["b"].success is True
        assert len(results["a"].trace.tool_calls) == 1
        assert len(results["b"].trace.tool_calls) == 1
