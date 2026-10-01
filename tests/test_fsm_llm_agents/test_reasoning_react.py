from __future__ import annotations

"""Tests for fsm_llm.agents.reasoning_react module."""


from unittest.mock import patch

import pytest

from fsm_llm.agents.constants import ContextKeys, ReasoningIntegrationKeys
from fsm_llm.agents.definitions import AgentConfig
from fsm_llm.agents.exceptions import AgentError
from fsm_llm.agents.tools import ToolRegistry


def _dummy_tool(params):
    """Dummy tool for testing."""
    return "dummy result"


class TestReasoningIntegrationKeys:
    """Tests for ReasoningIntegrationKeys constants."""

    def test_keys_are_strings(self):
        assert isinstance(ReasoningIntegrationKeys.REASONING_RESULT, str)
        assert isinstance(ReasoningIntegrationKeys.REASONING_TYPE_USED, str)
        assert isinstance(ReasoningIntegrationKeys.REASONING_CONFIDENCE, str)
        assert isinstance(ReasoningIntegrationKeys.REASONING_TOOL_NAME, str)

    def test_no_collision_with_context_keys(self):
        """ReasoningIntegrationKeys must not collide with ContextKeys."""
        reasoning_values = {
            v
            for k, v in vars(ReasoningIntegrationKeys).items()
            if not k.startswith("_") and isinstance(v, str)
        }
        context_values = {
            v
            for k, v in vars(ContextKeys).items()
            if not k.startswith("_") and isinstance(v, str)
        }
        collision = reasoning_values & context_values
        assert not collision, f"Key collision: {collision}"

    def test_keys_have_namespace_prefix(self):
        """All reasoning integration keys should be namespaced."""
        assert ReasoningIntegrationKeys.REASONING_RESULT.startswith(
            "reasoning_integration_"
        )
        assert ReasoningIntegrationKeys.REASONING_TYPE_USED.startswith(
            "reasoning_integration_"
        )
        assert ReasoningIntegrationKeys.REASONING_CONFIDENCE.startswith(
            "reasoning_integration_"
        )

    def test_reason_tool_name(self):
        assert ReasoningIntegrationKeys.REASONING_TOOL_NAME == "reason"


class TestReasoningReactAgentImport:
    """Tests for optional import handling."""

    def test_import_succeeds_when_reasoning_available(self):
        """fsm_llm.reasoning ships in every install, so the import never fails."""
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        assert ReasoningReactAgent is not None

    def test_missing_reasoning_raises_agent_error(self):
        """ReasoningReactAgent should raise AgentError if reasoning is missing."""
        # Mock the _HAS_REASONING flag to simulate missing package
        import fsm_llm.agents.reasoning_react as rr_module

        original = rr_module._HAS_REASONING

        try:
            rr_module._HAS_REASONING = False

            registry = ToolRegistry()
            registry.register_function(_dummy_tool, name="dummy", description="Dummy")

            with pytest.raises(AgentError, match=r"requires fsm_llm\.reasoning"):
                rr_module.ReasoningReactAgent(tools=registry)
        finally:
            rr_module._HAS_REASONING = original

    def test_conditional_import_in_init(self):
        """ReasoningReactAgent is imported unconditionally (reasoning always ships)."""
        import fsm_llm.agents
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        assert "ReasoningReactAgent" in fsm_llm.agents.__all__
        assert fsm_llm.agents.ReasoningReactAgent is ReasoningReactAgent


class TestReasonReToolAutoRegistration:
    """Tests for auto-registration of the reason pseudo-tool."""

    def test_reason_tool_auto_registered(self):
        """ReasoningReactAgent should auto-register a 'reason' tool."""
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")

        assert "reason" not in registry

        agent = ReasoningReactAgent(tools=registry)

        assert "reason" in agent.tools
        assert len(agent.tools) == 2

    def test_existing_reason_tool_not_overwritten(self):
        """If 'reason' already exists in registry, don't overwrite it."""
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        def custom_fn(params):
            return "custom reason"

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")
        registry.register_function(
            custom_fn, name="reason", description="Custom reason"
        )

        agent = ReasoningReactAgent(tools=registry)

        # Should keep the original function
        assert agent.tools.get("reason").execute_fn is custom_fn


class TestReasoningReactAgentPlaceholder:
    """Tests for the reason placeholder function."""

    def test_placeholder_returns_string(self):
        """Placeholder should return a descriptive string."""
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        result = ReasoningReactAgent._reason_placeholder({})
        assert isinstance(result, str)
        assert "reasoning" in result.lower() or "Reasoning" in result


class _StubRunError(Exception):
    """Raised by the stubbed ``_standard_run`` so run() never reaches an LLM."""


class TestReasoningReactAgentHandlerReset:
    """Regression: each run() must get independent handler state — no shared
    `AgentHandlers` instance across calls (decisions.md D-012, the D-014
    race-fix pattern applied to this class; see also react.py's
    ``TestReactAgentHandlersRace``-style regression)."""

    def test_no_persistent_handlers_attribute(self):
        """AgentHandlers is call-local (built inside run()), never stored on
        self — the attribute must not exist post-construction or post-run."""
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")

        agent = ReasoningReactAgent(tools=registry)

        assert not hasattr(agent, "_handlers")

        # Stub the FSM loop so run() makes no LLM call: with a reachable
        # Ollama daemon an unpatched run() executes a real multi-minute
        # conversation (RB-05). run() builds its handlers before this point.
        with patch.object(agent, "_standard_run", side_effect=_StubRunError):
            with pytest.raises(_StubRunError):
                agent.run("test task")

        assert not hasattr(agent, "_handlers")

    def test_repeated_run_builds_fresh_handlers_each_call(self):
        """Two consecutive run() calls each get their OWN AgentHandlers
        instance starting at iteration 0 — proves the shared-mutable-slot
        race this fix closes cannot reoccur (a stale `_current_iteration`
        from a prior call can never leak into the next call's instance)."""
        from fsm_llm.agents import reasoning_react as rr_module

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")

        agent = rr_module.ReasoningReactAgent(tools=registry)

        created_instances = []
        real_agent_handlers = rr_module.AgentHandlers

        def tracking_ctor(registry_arg, **kwargs):
            instance = real_agent_handlers(registry_arg, **kwargs)
            created_instances.append(instance)
            return instance

        # Stub the FSM loop so run() makes no LLM call (RB-05): an unpatched
        # run() against a reachable Ollama advances the second instance's
        # _current_iteration before the assertion below.
        with (
            patch.object(rr_module, "AgentHandlers", side_effect=tracking_ctor),
            patch.object(agent, "_standard_run", side_effect=_StubRunError),
        ):
            for _ in range(2):
                with pytest.raises(_StubRunError):
                    agent.run("test task")

        assert len(created_instances) == 2, (
            "run() must build a new AgentHandlers instance every call"
        )
        assert created_instances[0] is not created_instances[1]
        # Simulate the first call's handlers having advanced state; the
        # second call's instance must be unaffected (proves no shared slot).
        created_instances[0]._current_iteration = 99
        assert created_instances[1]._current_iteration == 0


class TestReasoningReactAgentConfig:
    """Tests for ReasoningReactAgent configuration."""

    def test_default_config(self):
        """Should accept default config."""
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")

        agent = ReasoningReactAgent(tools=registry)

        assert agent.config.max_iterations == 10  # default

    def test_custom_config(self):
        """Should accept custom config."""
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")
        config = AgentConfig(max_iterations=5)

        agent = ReasoningReactAgent(tools=registry, config=config)

        assert agent.config.max_iterations == 5

    def test_empty_registry_gets_only_reason_tool(self):
        """An empty registry does not raise: 'reason' is auto-registered before
        BaseAgent's non-empty check, so the agent holds exactly that tool."""
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        agent = ReasoningReactAgent(tools=ToolRegistry())

        assert [t.name for t in agent.tools.list_tools()] == ["reason"]


def _reason_context(tool_input, task="Is 97 prime?"):
    return {
        ContextKeys.TASK: task,
        ContextKeys.TOOL_NAME: "reason",
        ContextKeys.TOOL_INPUT: tool_input,
        ContextKeys.OBSERVATIONS: [],
        ContextKeys.AGENT_TRACE: [],
    }


class TestReasonToolInputAndShadowing:
    """REACT-05: the reason tool gets the task, and never shadows a user tool."""

    @pytest.mark.parametrize("tool_input", [None, {}, "", {"problem": "  "}])
    def test_empty_input_reasons_about_the_task(self, tool_input):
        from fsm_llm.agents.handlers import AgentHandlers
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")
        agent = ReasoningReactAgent(tools=registry)
        executor = agent._make_reasoning_tool_executor(AgentHandlers(agent.tools))
        with patch.object(
            agent._reasoning_engine, "solve_problem", return_value=("yes", {})
        ) as solve:
            delta = executor(_reason_context(tool_input))
        solve.assert_called_once_with("Is 97 prime?")
        assert delta[ContextKeys.TOOL_STATUS] == "success"

    def test_named_problem_wins_over_the_task(self):
        from fsm_llm.agents.handlers import AgentHandlers
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        agent = ReasoningReactAgent(tools=ToolRegistry())
        executor = agent._make_reasoning_tool_executor(AgentHandlers(agent.tools))
        with patch.object(
            agent._reasoning_engine, "solve_problem", return_value=("s", {})
        ) as solve:
            executor(_reason_context({"problem": "Is 91 prime?"}))
        solve.assert_called_once_with("Is 91 prime?")

    def test_user_reason_tool_is_the_one_invoked(self):
        from fsm_llm.agents.handlers import AgentHandlers
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent

        calls: list[object] = []

        def user_reason(params):
            calls.append(params)
            return "user reasoning"

        registry = ToolRegistry()
        registry.register_function(
            user_reason, name="reason", description="The user's own reason tool"
        )
        agent = ReasoningReactAgent(tools=registry)
        assert agent.tools.get("reason").execute_fn is user_reason
        assert len(agent.tools) == 1  # no reasoning tool added
        executor = agent._make_reasoning_tool_executor(AgentHandlers(agent.tools))
        with patch.object(agent._reasoning_engine, "solve_problem") as solve:
            delta = executor(_reason_context({"problem": "p"}))
        solve.assert_not_called()
        assert len(calls) == 1
        assert delta[ContextKeys.TOOL_STATUS] == "success"
        assert "user reasoning" in str(delta[ContextKeys.TOOL_RESULT])

    def test_registry_subclass_behaviour_is_kept(self):
        """A CachingToolRegistry still caches through the agent's copy."""
        from fsm_llm.agents.handlers import AgentHandlers
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent
        from fsm_llm.agents.tool_registries import CachingToolRegistry

        runs: list[object] = []

        def lookup(params):
            runs.append(params)
            return "found"

        registry = CachingToolRegistry()
        registry.register_function(lookup, name="lookup", description="Lookup")
        agent = ReasoningReactAgent(tools=registry)
        assert "reason" not in registry  # caller registry not mutated
        handlers = AgentHandlers(agent.tools)
        for _ in range(2):
            ctx = {
                ContextKeys.TASK: "t",
                ContextKeys.TOOL_NAME: "lookup",
                ContextKeys.TOOL_INPUT: {"q": "x"},
                ContextKeys.OBSERVATIONS: [],
                ContextKeys.AGENT_TRACE: [],
            }
            assert handlers.execute_tool(ctx)[ContextKeys.TOOL_STATUS] == "success"
        assert len(runs) == 1
        assert registry.cache_hits == 1


class TestReasonToolEngineFailure:
    """An engine that raises (e.g. a spent solve budget, D-014 of plan
    944e2692) becomes a failed tool call, never a crash of the run."""

    def test_spent_budget_is_a_failed_tool_call(self):
        from fsm_llm import RunBudgetExceededError
        from fsm_llm.agents.handlers import AgentHandlers
        from fsm_llm.agents.reasoning_react import ReasoningReactAgent
        from fsm_llm.reasoning import ReasoningExecutionError

        error = ReasoningExecutionError(
            "Reasoning did not finish within 170 steps",
            details={
                "conversation_id": "c",
                "responses_so_far": 3,
                "partial_context": {"problem_type": "logic"},
            },
        )
        error.__cause__ = RunBudgetExceededError("steps", 170, 170)
        agent = ReasoningReactAgent(tools=ToolRegistry())
        executor = agent._make_reasoning_tool_executor(AgentHandlers(agent.tools))
        with patch.object(agent._reasoning_engine, "solve_problem", side_effect=error):
            delta = executor(_reason_context({"problem": "Is 91 prime?"}))

        assert delta[ContextKeys.TOOL_STATUS] == "failed"
        assert "did not finish" in delta[ContextKeys.TOOL_ERROR]
        assert str(delta[ContextKeys.TOOL_RESULT]).startswith("Reasoning failed")
        assert ReasoningIntegrationKeys.REASONING_RESULT not in delta
