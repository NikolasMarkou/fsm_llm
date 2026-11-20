from __future__ import annotations

"""Tests for BaseAgent abstract class."""

import time
from unittest.mock import patch

import pytest

from fsm_llm.agents.base import BaseAgent
from fsm_llm.agents.definitions import AgentConfig, AgentResult
from fsm_llm.agents.exceptions import AgentTimeoutError, BudgetExhaustedError
from fsm_llm.core.codes import Status
from fsm_llm.core.definitions import AgentError
from fsm_llm.tools import ToolRegistry


class ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing base class."""

    def __init__(self, config=None, **api_kwargs):
        super().__init__(config, **api_kwargs)

    def run(self, task, initial_context=None):
        return AgentResult(answer=f"Answer: {task}", status=Status.OK)

    def _register_handlers(self, api):
        pass


class TestBaseAgentCallable:
    """Tests for __call__ support."""

    def test_call_delegates_to_run(self):
        agent = ConcreteAgent()
        result = agent("What is 2+2?")
        assert result.answer == "Answer: What is 2+2?"
        assert result.success is True

    def test_call_with_kwargs(self):
        class KwargsAgent(BaseAgent):
            def run(self, task, initial_context=None):
                ctx = initial_context or {}
                return AgentResult(
                    answer=f"{task} [{ctx.get('mode', 'default')}]", status=Status.OK
                )

            def _register_handlers(self, api):
                pass

        agent = KwargsAgent()
        result = agent("hello", initial_context={"mode": "fast"})
        assert "[fast]" in result.answer

    def test_str_representation(self):
        agent = ConcreteAgent()
        assert "ConcreteAgent" in str(agent)
        assert "model=" in str(agent)


class TestBaseAgentBudgets:
    """Tests for budget enforcement."""

    def test_check_budgets_timeout(self):
        agent = ConcreteAgent(config=AgentConfig(timeout_seconds=0.001))
        time.sleep(0.01)
        with pytest.raises(AgentTimeoutError):
            agent._check_budgets(time.monotonic() - 1.0)

    def test_check_budgets_ok(self):
        agent = ConcreteAgent(config=AgentConfig(max_iterations=10))
        # Should not raise
        agent._check_budgets(time.monotonic())

    def test_run_budgets_are_the_step_ceiling_and_the_remaining_time(
        self, timing_factor
    ):
        agent = ConcreteAgent(config=AgentConfig(max_iterations=5, timeout_seconds=60))
        max_steps, max_seconds = agent._run_budgets(time.monotonic() - 20.0)
        assert max_steps == 15  # 5 * FSM_BUDGET_MULTIPLIER
        # 20 s were spent out of 60; the slack is the time between two clock
        # reads (1 s, scaled by the factor on a traced or loaded run)
        assert 40.0 - timing_factor < max_seconds <= 40.0
        assert agent._run_budgets(time.monotonic(), 2)[0] == 6

    def test_run_budgets_raise_timeout_when_the_clock_is_spent(self):
        agent = ConcreteAgent(config=AgentConfig(timeout_seconds=1.0))
        with pytest.raises(AgentTimeoutError):
            agent._run_budgets(time.monotonic() - 2.0)

    def test_budget_error_maps_the_core_budgets(self):
        from fsm_llm import RunBudgetExceededError

        agent = ConcreteAgent(config=AgentConfig(max_iterations=5, timeout_seconds=7))
        steps = agent._budget_error(RunBudgetExceededError("steps", 15, 15))
        assert isinstance(steps, BudgetExhaustedError)
        assert steps.limit == 15  # 5 * 3
        seconds = agent._budget_error(RunBudgetExceededError("seconds", 3.2, 4))
        assert isinstance(seconds, AgentTimeoutError)
        assert seconds.timeout_seconds == 7


class TestBaseAgentAnswerExtraction:
    """Tests for answer extraction."""

    def test_final_answer_key_is_not_an_answer_source(self):
        # D-046 (plan 07ad3f8c): no pattern writes final_answer.
        agent = ConcreteAgent()
        ctx = {"final_answer": "A planted answer."}
        assert agent._extract_answer(ctx, ["The reply."]) == "The reply."

    def test_extract_from_extra_keys(self):
        agent = ConcreteAgent()
        ctx = {"judge_verdict": "Consensus: yes, it is correct."}
        answer = agent._extract_answer(ctx, [], extra_keys=["judge_verdict"])
        assert "Consensus" in answer

    def test_extract_from_responses(self):
        agent = ConcreteAgent()
        ctx = {}
        answer = agent._extract_answer(ctx, ["short", "This is a longer response."])
        assert answer == "This is a longer response."

    def test_extract_default(self):
        agent = ConcreteAgent()
        assert "could not" in agent._extract_answer({}, [])

    def test_extra_key_wins_over_a_final_answer_key(self):
        agent = ConcreteAgent()
        ctx = {
            "final_answer": "A planted answer.",
            "judge_verdict": "Verdict text here.",
        }
        answer = agent._extract_answer(ctx, ["response"], extra_keys=["judge_verdict"])
        assert answer == "Verdict text here."


class TestBaseAgentTraceBuilding:
    """Tests for trace building."""

    def test_build_trace_empty(self):
        agent = ConcreteAgent()
        trace = agent._build_trace({}, 5)
        assert trace.total_iterations == 5
        assert trace.tool_calls == []

    def test_build_trace_with_steps(self):
        agent = ConcreteAgent()
        ctx = {
            "agent_trace": [
                {"action": "search(query)", "thought": "Need to search"},
                {"action": "none", "thought": "Done"},
                {"action": "calculate(expr)", "thought": "Math time"},
            ]
        }
        trace = agent._build_trace(ctx, 3)
        assert len(trace.tool_calls) == 2
        assert trace.tool_calls[0].tool_name == "search"
        assert trace.tool_calls[1].tool_name == "calculate"


class TestBaseAgentContextFiltering:
    """Tests for context filtering."""

    def test_filter_removes_private_keys(self):
        ctx = {"answer": "yes", "_internal": "secret", "_max": 10, "data": 42}
        filtered = BaseAgent._filter_context(ctx)
        assert "answer" in filtered
        assert "data" in filtered
        assert "_internal" not in filtered
        assert "_max" not in filtered

    def test_filter_removes_every_internal_prefix_case_insensitively(self):
        """`_filter_context` must strip the FULL internal-prefix class.

        Pins step 3 of plan-2026-07-20T040150-876e7164 (F-13). The previous
        `k.startswith("_")` check was case-sensitive and saw only the literal
        `_` prefix, so `system_password`, `internal_token` and `__dunder` all
        passed straight through into caller-visible output.
        """
        ctx = {
            # Must survive.
            "answer": "yes",
            "data": 42,
            # `_`-prefixed (already caught by the old check).
            "_internal": "secret",
            # The prefixes the old check MISSED entirely.
            "system_password": "hunter2",
            "internal_token": "tok",
            "__dunder": "no",
            # Case variants of the same prefixes.
            "System_Password": "hunter2",
            "SYSTEM_secret": "shh",
            "Internal_token": "tok",
            "INTERNAL_state": "x",
        }
        filtered = BaseAgent._filter_context(ctx)

        assert filtered == {"answer": "yes", "data": 42}

    def test_final_context_does_not_leak_system_prefixed_keys(self):
        """The filter's output is what reaches `AgentResult.final_context`.

        End-to-end assertion of the same defect at the surface a caller
        actually sees, not just at the static helper.
        """
        result = AgentResult(
            answer="done",
            status=Status.OK,
            final_context=BaseAgent._filter_context(
                {
                    "order_id": "A-1",
                    "system_password": "hunter2",
                    "Internal_token": "tok",
                }
            ),
        )

        assert result.final_context == {"order_id": "A-1"}
        assert "system_password" not in result.final_context
        assert "Internal_token" not in result.final_context


class TestBaseAgentCreateApi:
    """Tests for API factory helper."""

    @patch("fsm_llm.agents.base.runs.API.from_definition")
    def test_create_api(self, mock_from_def):
        agent = ConcreteAgent(config=AgentConfig(model="test-model"))
        fsm_def = {"name": "test", "initial_state": "start", "states": {}}
        agent._create_api(fsm_def)

        mock_from_def.assert_called_once_with(
            fsm_def,
            model="test-model",
            temperature=agent.config.temperature,
            max_tokens=agent.config.max_tokens,
        )


class TestBaseAgentConfig:
    """Tests for configuration handling."""

    def test_default_config(self):
        agent = ConcreteAgent()
        assert agent.config.max_iterations > 0
        assert agent.config.timeout_seconds > 0

    def test_custom_config(self):
        config = AgentConfig(model="custom-model", max_iterations=20)
        agent = ConcreteAgent(config=config)
        assert agent.config.model == "custom-model"
        assert agent.config.max_iterations == 20

    def test_api_kwargs_stored(self):
        agent = ConcreteAgent(api_key="test-key", extra="value")
        assert agent._api_kwargs == {"api_key": "test-key", "extra": "value"}


class TestCreateAgent:
    """Tests for create_agent() convenience function."""

    def test_create_react_agent(self):
        from fsm_llm.agents import create_agent
        from fsm_llm.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        agent = create_agent(tools=[search])
        from fsm_llm.agents import ReactAgent

        assert isinstance(agent, ReactAgent)

    def test_create_debate_agent(self):
        from fsm_llm.agents import DebateAgent, create_agent

        agent = create_agent(pattern="debate")
        assert isinstance(agent, DebateAgent)

    def test_create_with_registry(self):
        from fsm_llm.agents import ToolRegistry, create_agent

        registry = ToolRegistry()
        registry.register_function(lambda p: "ok", name="test", description="Test tool")
        agent = create_agent(tools=registry)
        assert len(agent.tools) == 1

    def test_unknown_pattern_raises(self):
        from fsm_llm.agents import create_agent

        with pytest.raises(ValueError, match="Unknown pattern"):
            create_agent(pattern="nonexistent")

    def test_create_with_config(self):
        from fsm_llm.agents import AgentConfig, create_agent

        config = AgentConfig(max_iterations=5)
        agent = create_agent(pattern="debate", config=config)
        assert agent.config.max_iterations == 5

    def test_create_parallel_react_agent(self):
        from fsm_llm.agents import ParallelReactAgent, create_agent
        from fsm_llm.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        agent = create_agent(pattern="parallel_react", tools=[search])
        assert isinstance(agent, ParallelReactAgent)

    def test_create_native_fc_agent(self):
        from fsm_llm.agents import NativeFunctionCallingReactAgent, create_agent
        from fsm_llm.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        agent = create_agent(pattern="native_fc", tools=[search])
        assert isinstance(agent, NativeFunctionCallingReactAgent)

    def test_create_verified_react_agent(self):
        from fsm_llm.agents import VerifiedReactAgent, create_agent
        from fsm_llm.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        agent = create_agent(pattern="verified_react", tools=[search])
        assert isinstance(agent, VerifiedReactAgent)

    def test_create_auto_memory_agent(self):
        from fsm_llm.agents import AutoMemoryReactAgent, create_agent
        from fsm_llm.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        agent = create_agent(pattern="auto_memory", tools=[search])
        assert isinstance(agent, AutoMemoryReactAgent)


class TestOneEmptyRegistryGuard:
    """F-AP-05: the six tool patterns refuse an empty registry the same way."""

    @pytest.mark.parametrize(
        "pattern",
        [
            "react",
            "reflexion",
            "rewoo",
            "parallel_react",
            "native_fc",
        ],
    )
    def test_create_agent_refuses_an_empty_registry(self, pattern):
        from fsm_llm.agents import create_agent

        with pytest.raises(AgentError, match="empty tool registry"):
            create_agent(pattern, ToolRegistry())


class TestBaseAgentPackageLayout:
    """``fsm_llm.agents.base`` is a package (plan d454fd62, step 87): ``BaseAgent`` is a shell
    over four mixins and the module helpers sit in ``helpers``. Pins the facade, the
    composition and that every declaration of ``_AgentSurface`` (mypy only) is backed."""

    PUBLIC = (
        "BaseAgent",
        "accepts_tools",
        "artifact_text",
        "caller_prompt_keys",
        "flagged_tool_names",
        "pattern_run_output_keys",
        "reject_misplaced_kwargs",
        "strip_caller_context",
    )

    def test_the_facade_exports_the_public_names_of_the_old_module(self):
        import fsm_llm.agents.base as package

        assert tuple(package.__all__) == self.PUBLIC
        for name in self.PUBLIC:
            assert getattr(package, name) is not None

    def test_the_shell_is_composed_of_the_mixins(self):
        from fsm_llm.agents.base.hitl_gate import HitlGateMixin
        from fsm_llm.agents.base.outcome import OutcomeMixin
        from fsm_llm.agents.base.registration import RegistrationMixin
        from fsm_llm.agents.base.runs import RunsMixin
        from fsm_llm.agents.base.surface import _AgentSurface

        mro = [k.__name__ for k in BaseAgent.__mro__]
        assert mro == [
            "BaseAgent",
            "HitlGateMixin",
            "OutcomeMixin",
            "RunsMixin",
            "RegistrationMixin",
            "_AgentSurface",
            "ABC",
            "object",
        ]
        for mixin in (HitlGateMixin, OutcomeMixin, RunsMixin, RegistrationMixin):
            assert issubclass(mixin, _AgentSurface)
            assert not issubclass(mixin, BaseAgent)

    def test_the_two_members_that_name_the_class_stay_in_the_shell(self):
        assert "_run_outcome" in vars(BaseAgent)
        assert BaseAgent.__module__ == "fsm_llm.agents.base.agent"

    def test_answer_source_keys_sits_beside_extract_answer(self):
        # D-027: a pattern that overrides `_extract_answer` overrides this too.
        assert (
            BaseAgent._answer_source_keys.__qualname__.split(".")[0]
            == (BaseAgent._extract_answer.__qualname__.split(".")[0])
        )

    def test_every_surface_declaration_is_defined_by_the_agent(self):
        """mypy cannot tell that a declaration is backed by nothing (the step-81 lesson):
        every name of ``_AgentSurface`` is defined by a class of the ``BaseAgent`` MRO or
        assigned as ``self.<name>`` in one of them."""
        import ast
        import inspect

        from fsm_llm.agents.base import surface

        tree = ast.parse(inspect.getsource(surface))
        surface_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_AgentSurface"
        )
        declared: set[str] = set()
        for node in surface_class.body:
            if not isinstance(node, ast.If):  # the TYPE_CHECKING block
                continue
            for member in node.body:
                if isinstance(member, ast.AnnAssign) and isinstance(
                    member.target, ast.Name
                ):
                    declared.add(member.target.id)
                elif isinstance(member, ast.FunctionDef):
                    declared.add(member.name)
        assert len(declared) >= 12  # an empty scan would pass vacuously

        defined: set[str] = set()
        for klass in BaseAgent.__mro__:
            if klass in (surface._AgentSurface, object):
                continue
            defined |= set(vars(klass))
            for node in ast.walk(ast.parse(inspect.getsource(klass))):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    defined.add(node.attr)
        assert declared - defined == set()


class TestHandlerRegistrationHelpers:
    """``_register_iteration_limiter`` and ``_register_tool_executor`` (``base/registration.py``).

    Step 87 of plan d454fd62 found no test of the priorities they register: a mutant that
    registered the tool executor at the limiter's priority survived the agents suites and a
    3,000-program differential (no built-in handler shares its timing and state, so only a
    caller's own handler could see the order). The constants of ``HandlerPriorities`` are the
    documented order, pinned here."""

    @staticmethod
    def _api():
        from fsm_llm import API
        from fsm_llm.testing import PromptGroundedLLM

        fsm = {
            "name": "t",
            "description": "d",
            "initial_state": "a",
            "persona": "p",
            "states": {
                "a": {
                    "id": "a",
                    "description": "a",
                    "purpose": "a",
                    "transitions": [
                        {"target_state": "b", "description": "go", "priority": 100}
                    ],
                },
                "b": {"id": "b", "description": "b", "purpose": "b"},
            },
        }
        return API.from_definition(fsm, llm_interface=PromptGroundedLLM())

    def test_the_limiter_runs_before_a_transition_at_its_own_priority(self):
        from fsm_llm.agents.constants import HandlerNames, HandlerPriorities
        from fsm_llm.handlers import HandlerTiming

        api = self._api()
        ConcreteAgent()._register_iteration_limiter(api, lambda ctx: {})

        (handler,) = api.handler_system.handlers
        assert handler.name == HandlerNames.ITERATION_LIMITER
        assert handler.priority == HandlerPriorities.ITERATION_LIMITER == 50
        assert api.handler_system.handlers_at(HandlerTiming.PRE_TRANSITION) == [handler]

    def test_the_tool_executor_runs_on_entry_after_the_limiter_and_approval(self):
        from fsm_llm.agents.constants import HandlerNames, HandlerPriorities
        from fsm_llm.handlers import HandlerTiming

        api = self._api()
        ConcreteAgent()._register_tool_executor(api, "b", lambda ctx: {})

        (handler,) = api.handler_system.handlers
        assert handler.name == HandlerNames.TOOL_EXECUTOR
        assert handler.priority == HandlerPriorities.TOOL_EXECUTOR == 100
        assert api.handler_system.handlers_at(HandlerTiming.POST_TRANSITION) == [
            handler
        ]
        assert HandlerPriorities.ITERATION_LIMITER < handler.priority
