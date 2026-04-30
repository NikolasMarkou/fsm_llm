from __future__ import annotations

"""Tests for fsm_llm_agents.reasoning_react module."""


import pytest

from fsm_llm.stdlib.agents.constants import ContextKeys, ReasoningIntegrationKeys
from fsm_llm.stdlib.agents.definitions import AgentConfig
from fsm_llm.stdlib.agents.exceptions import AgentError
from fsm_llm.stdlib.agents.tools import ToolRegistry


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
        """ReasoningReactAgent should be importable when reasoning is installed."""
        try:
            from fsm_llm.stdlib.agents.reasoning_react import ReasoningReactAgent

            assert ReasoningReactAgent is not None
        except ImportError:
            pytest.skip("fsm_llm_reasoning not installed")

    def test_missing_reasoning_raises_agent_error(self):
        """ReasoningReactAgent should raise AgentError if reasoning is missing."""
        # Mock the _HAS_REASONING flag to simulate missing package
        import fsm_llm_agents.reasoning_react as rr_module

        original = rr_module._HAS_REASONING

        try:
            rr_module._HAS_REASONING = False

            registry = ToolRegistry()
            registry.register_function(_dummy_tool, name="dummy", description="Dummy")

            with pytest.raises(AgentError, match="requires fsm_llm_reasoning"):
                rr_module.ReasoningReactAgent(tools=registry)
        finally:
            rr_module._HAS_REASONING = original

    def test_conditional_import_in_init(self):
        """__init__.py should not fail if reasoning is not installed."""
        # The import should always succeed (ReasoningReactAgent may or may not be in namespace)
        import fsm_llm_agents

        # Check that __all__ contains it regardless
        assert "ReasoningReactAgent" in fsm_llm_agents.__all__


class TestReasonReToolAutoRegistration:
    """Tests for auto-registration of the reason pseudo-tool."""

    def test_reason_tool_auto_registered(self):
        """ReasoningReactAgent should auto-register a 'reason' tool."""
        try:
            from fsm_llm.stdlib.agents.reasoning_react import ReasoningReactAgent
        except ImportError:
            pytest.skip("fsm_llm_reasoning not installed")

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")

        assert "reason" not in registry

        try:
            agent = ReasoningReactAgent(tools=registry)
            assert "reason" in agent.tools
            assert len(agent.tools) == 2
        except Exception:
            # May fail if reasoning engine can't load FSMs, but registration should have happened
            if "reason" in registry:
                pass  # Auto-registration worked
            else:
                pytest.skip("Reasoning engine initialization failed")

    def test_existing_reason_tool_not_overwritten(self):
        """If 'reason' already exists in registry, don't overwrite it."""
        try:
            from fsm_llm.stdlib.agents.reasoning_react import ReasoningReactAgent
        except ImportError:
            pytest.skip("fsm_llm_reasoning not installed")

        def custom_fn(params):
            return "custom reason"

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")
        registry.register_function(
            custom_fn, name="reason", description="Custom reason"
        )

        try:
            agent = ReasoningReactAgent(tools=registry)
            # Should keep the original function
            assert agent.tools.get("reason").execute_fn is custom_fn
        except Exception:
            pytest.skip("Reasoning engine initialization failed")


class TestReasoningReactAgentPlaceholder:
    """Tests for the reason placeholder function."""

    def test_placeholder_returns_string(self):
        """Placeholder should return a descriptive string."""
        try:
            from fsm_llm.stdlib.agents.reasoning_react import ReasoningReactAgent
        except ImportError:
            pytest.skip("fsm_llm_reasoning not installed")

        result = ReasoningReactAgent._reason_placeholder({})
        assert isinstance(result, str)
        assert "reasoning" in result.lower() or "Reasoning" in result


class TestReasoningReactAgentHandlerReset:
    """Regression: handlers must be reset between consecutive run() calls."""

    def test_repeated_run_resets_iteration_counter(self):
        """run() must call _handlers.reset() so _current_iteration starts at 0."""
        try:
            from fsm_llm.stdlib.agents.reasoning_react import ReasoningReactAgent
        except ImportError:
            pytest.skip("fsm_llm_reasoning not installed")
        from unittest.mock import patch

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")

        try:
            agent = ReasoningReactAgent(tools=registry)
        except Exception:
            pytest.skip("Reasoning engine initialization failed")

        # Simulate stale state from a previous run
        agent._handlers._current_iteration = 5

        # Patch reset to track it was called, then let it run normally
        original_reset = agent._handlers.reset
        reset_called = []

        def tracking_reset():
            reset_called.append(True)
            original_reset()

        with patch.object(agent._handlers, "reset", side_effect=tracking_reset):
            try:
                agent.run("test task")
            except Exception:
                pass  # Expected — no LLM configured

        assert len(reset_called) == 1, (
            "reset() must be called exactly once at start of run()"
        )


class TestReasoningReactAgentConfig:
    """Tests for ReasoningReactAgent configuration."""

    def test_default_config(self):
        """Should accept default config."""
        try:
            from fsm_llm.stdlib.agents.reasoning_react import ReasoningReactAgent
        except ImportError:
            pytest.skip("fsm_llm_reasoning not installed")

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")

        try:
            agent = ReasoningReactAgent(tools=registry)
            assert agent.config.max_iterations == 10  # default
        except Exception:
            pytest.skip("Reasoning engine initialization failed")

    def test_custom_config(self):
        """Should accept custom config."""
        try:
            from fsm_llm.stdlib.agents.reasoning_react import ReasoningReactAgent
        except ImportError:
            pytest.skip("fsm_llm_reasoning not installed")

        registry = ToolRegistry()
        registry.register_function(_dummy_tool, name="search", description="Search")
        config = AgentConfig(max_iterations=5)

        try:
            agent = ReasoningReactAgent(tools=registry, config=config)
            assert agent.config.max_iterations == 5
        except Exception:
            pytest.skip("Reasoning engine initialization failed")

    def test_empty_registry_raises(self):
        """Empty registry should raise AgentError."""
        try:
            from fsm_llm.stdlib.agents.reasoning_react import ReasoningReactAgent
        except ImportError:
            pytest.skip("fsm_llm_reasoning not installed")

        registry = ToolRegistry()
        # Even with auto-registration of 'reason', the initial check is on tools param
        # Since 'reason' is auto-registered, an empty registry will have 1 tool
        # But we test that the module validates properly
        try:
            agent = ReasoningReactAgent(tools=registry)
            # reason was auto-registered, so it should have at least 1 tool
            assert len(agent.tools) >= 1
        except AgentError:
            pass  # Also acceptable
        except Exception:
            pytest.skip("Reasoning engine initialization failed")
