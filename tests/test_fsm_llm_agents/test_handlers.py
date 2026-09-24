from __future__ import annotations

"""Tests for fsm_llm_agents.handlers module."""

from typing import ClassVar

from fsm_llm_agents.constants import ContextKeys, Defaults
from fsm_llm_agents.handlers import AgentHandlers, make_iteration_limiter
from fsm_llm_agents.tools import ToolRegistry


def _echo(params):
    """Echo the input."""
    return f"Echo: {params.get('input', '')}"


def _add(params):
    """Add a and b."""
    return params["a"] + params["b"]


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(_echo, name="echo", description="Echo input")
    registry.register_function(_add, name="add", description="Add numbers")
    return registry


class TestAgentHandlers:
    """Tests for AgentHandlers."""

    def test_execute_tool_success(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: {"input": "hello"},
            ContextKeys.REASONING: "Testing echo",
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)

        assert result[ContextKeys.TOOL_STATUS] == "success"
        assert "Echo: hello" in result[ContextKeys.TOOL_RESULT]
        assert len(result[ContextKeys.OBSERVATIONS]) == 1
        assert result[ContextKeys.OBSERVATION_COUNT] == 1

    def test_execute_tool_with_kwargs(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "add",
            ContextKeys.TOOL_INPUT: {"a": 3, "b": 7},
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)

        assert result[ContextKeys.TOOL_STATUS] == "success"
        assert "10" in result[ContextKeys.TOOL_RESULT]

    def test_execute_tool_none_selected(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "none",
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)
        assert result[ContextKeys.TOOL_STATUS] == "skipped"

    def test_execute_tool_no_tool_name(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {ContextKeys.TOOL_NAME: None, ContextKeys.OBSERVATIONS: []}
        result = handlers.execute_tool(context)
        assert result[ContextKeys.TOOL_STATUS] == "skipped"

    def test_execute_tool_nonexistent(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "nonexistent",
            ContextKeys.TOOL_INPUT: {},
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)
        # An unknown name is a no-tool turn: feedback only, no observation.
        assert result[ContextKeys.TOOL_STATUS] == "skipped"
        assert "nonexistent" in result[ContextKeys.TOOL_RESULT]
        assert ContextKeys.OBSERVATIONS not in result

    def test_execute_tool_string_input_normalized(self):
        """String tool_input should be normalized to dict."""
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: "hello world",
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)
        assert result[ContextKeys.TOOL_STATUS] == "success"

    def test_execute_tool_clears_selection(self):
        """After execution, tool selection keys should be cleared."""
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: {"input": "test"},
            ContextKeys.OBSERVATIONS: [],
        }
        result = handlers.execute_tool(context)
        assert result[ContextKeys.TOOL_NAME] is None
        assert result[ContextKeys.TOOL_INPUT] is None

    def test_execute_tool_accumulates_observations(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        existing = ["[Step 1] Previous observation"]
        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: {"input": "test"},
            ContextKeys.OBSERVATIONS: existing,
        }
        result = handlers.execute_tool(context)
        assert len(result[ContextKeys.OBSERVATIONS]) == 2

    def test_execute_tool_prunes_observations(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        # Create more than MAX_OBSERVATIONS existing observations
        existing = [f"[Step {i}] Obs {i}" for i in range(Defaults.MAX_OBSERVATIONS + 5)]
        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: {"input": "test"},
            ContextKeys.OBSERVATIONS: existing,
        }
        result = handlers.execute_tool(context)
        assert len(result[ContextKeys.OBSERVATIONS]) <= Defaults.MAX_OBSERVATIONS

    def test_execute_tool_builds_trace(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {
            ContextKeys.TOOL_NAME: "echo",
            ContextKeys.TOOL_INPUT: {"input": "test"},
            ContextKeys.REASONING: "Testing",
            ContextKeys.OBSERVATIONS: [],
            ContextKeys.AGENT_TRACE: [],
        }
        result = handlers.execute_tool(context)
        trace = result[ContextKeys.AGENT_TRACE]
        assert len(trace) == 1
        assert trace[0]["action"] == "echo({'input': 'test'})"

    def test_check_iteration_limit_below(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        context = {"_max_iterations": 10}
        result = handlers.check_iteration_limit(context)
        assert ContextKeys.ITERATION_COUNT in result
        assert ContextKeys.MAX_ITERATIONS_REACHED not in result

    def test_check_iteration_limit_reached(self):
        registry = _make_registry()
        handlers = AgentHandlers(registry)

        # Simulate reaching the limit
        context = {"_max_iterations": 2}
        handlers.check_iteration_limit(context)  # iteration 1
        result = handlers.check_iteration_limit(context)  # iteration 2

        assert result[ContextKeys.MAX_ITERATIONS_REACHED] is True
        assert result[ContextKeys.SHOULD_TERMINATE] is True


class TestMakeIterationLimiter:
    """Tests for the shared context-counting limiter factory (D-003)."""

    _FORCED: ClassVar[dict[str, bool]] = {
        ContextKeys.SHOULD_TERMINATE: True,
        ContextKeys.CHECKER_PASSED: True,
    }

    def test_count_increments_below_threshold(self):
        limiter = make_iteration_limiter(10, self._FORCED)
        assert limiter({}) == {ContextKeys.ITERATION_COUNT: 1}
        assert limiter({ContextKeys.ITERATION_COUNT: 3}) == {
            ContextKeys.ITERATION_COUNT: 4
        }

    def test_triggers_at_max_minus_one(self):
        limiter = make_iteration_limiter(5, self._FORCED)
        assert limiter({ContextKeys.ITERATION_COUNT: 2}) == {
            ContextKeys.ITERATION_COUNT: 3
        }
        assert limiter({ContextKeys.ITERATION_COUNT: 3}) == {
            ContextKeys.ITERATION_COUNT: 4,
            **self._FORCED,
        }

    def test_max_iterations_one_triggers_on_first_call(self):
        limiter = make_iteration_limiter(1, self._FORCED)
        assert limiter({})[ContextKeys.SHOULD_TERMINATE] is True

    def test_max_iterations_two_triggers_on_first_call(self):
        limiter = make_iteration_limiter(2, self._FORCED)
        result = limiter({})
        assert result[ContextKeys.ITERATION_COUNT] == 1
        assert result[ContextKeys.CHECKER_PASSED] is True

    def test_context_key_overrides_limit(self):
        limiter = make_iteration_limiter(100, self._FORCED, context_max_key="_max")
        assert ContextKeys.SHOULD_TERMINATE in limiter(
            {"_max": 3, ContextKeys.ITERATION_COUNT: 1}
        )

    def test_context_key_absent_falls_back_to_max_iterations(self):
        limiter = make_iteration_limiter(3, self._FORCED, context_max_key="_max")
        assert ContextKeys.SHOULD_TERMINATE not in limiter({})
        assert ContextKeys.SHOULD_TERMINATE in limiter({ContextKeys.ITERATION_COUNT: 1})

    def test_forced_keys_returned_verbatim_and_copied(self):
        forced = {ContextKeys.ALL_COLLECTED: True, ContextKeys.SHOULD_TERMINATE: True}
        limiter = make_iteration_limiter(1, forced)
        forced[ContextKeys.ALL_COLLECTED] = False
        result = limiter({})
        assert result == {
            ContextKeys.ITERATION_COUNT: 1,
            ContextKeys.ALL_COLLECTED: True,
            ContextKeys.SHOULD_TERMINATE: True,
        }


class TestEmptyToolInputRecovery:
    """CR-02: the task-string recovery only fills a string-compatible param."""

    _TASK = "What is 17 plus 25?"

    def _run(self, fn, schema):
        registry = ToolRegistry()
        registry.register_function(fn, name="spy", parameter_schema=schema)
        handlers = AgentHandlers(registry)
        context = {
            ContextKeys.TOOL_NAME: "spy",
            ContextKeys.TOOL_INPUT: None,
            ContextKeys.TASK: self._TASK,
            ContextKeys.OBSERVATIONS: [],
        }
        return handlers.execute_tool(context)

    def test_int_param_never_receives_task_string(self):
        received: list[object] = []

        def add_numbers(a: int, b: int = 0) -> int:
            received.append(a)
            return a + b

        schema = {"properties": {"a": {"type": "integer"}}, "required": ["a"]}
        result = self._run(add_numbers, schema)

        assert self._TASK not in received
        assert result[ContextKeys.TOOL_STATUS] == "failed"
        observation = result[ContextKeys.TOOL_RESULT]
        assert "can only concatenate" not in observation
        assert "'a'" in observation

    def _recovered_query(self, param_schema) -> list[object]:
        received: list[object] = []

        def search(query) -> str:
            received.append(query)
            return "ok"

        schema = {"properties": {"query": param_schema}, "required": ["query"]}
        result = self._run(search, schema)
        assert result[ContextKeys.TOOL_STATUS] == "success"
        return received

    def test_string_param_still_recovers(self):
        assert self._recovered_query({"type": "string"}) == [self._TASK]

    def test_untyped_param_still_recovers(self):
        assert self._recovered_query({}) == [self._TASK]

    def test_union_with_string_param_still_recovers(self):
        assert self._recovered_query({"type": ["string", "null"]}) == [self._TASK]

    def test_bool_property_schema_recovers_without_error(self):
        # JSON Schema allows `true` as a property schema; base recovered here.
        assert self._recovered_query(True) == [self._TASK]

    def test_string_property_schema_recovers_without_error(self):
        # Flat description shape nested under "properties"; base recovered here.
        assert self._recovered_query("search text") == [self._TASK]
