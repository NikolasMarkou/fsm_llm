from __future__ import annotations

"""Tests for fsm_llm.agents.definitions module."""

import pytest

from fsm_llm.agents.definitions import (
    AgentConfig,
    AgentResult,
    AgentStep,
    AgentTrace,
    ApprovalRequest,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_basic_creation(self):
        tool = ToolDefinition(name="search", description="Search the web")
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert tool.parameter_schema == {}
        assert tool.requires_approval is False
        assert tool.execute_fn is None

    def test_with_schema(self):
        schema = {"properties": {"query": {"type": "string"}}}
        tool = ToolDefinition(
            name="search",
            description="Search",
            parameter_schema=schema,
        )
        assert tool.parameter_schema == schema

    def test_with_execute_fn(self):
        fn = lambda params: "result"  # noqa: E731
        tool = ToolDefinition(
            name="search",
            description="Search",
            execute_fn=fn,
        )
        assert tool.execute_fn is fn

    def test_invalid_name_special_chars(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            ToolDefinition(name="bad-name!", description="Bad")

    def test_valid_name_underscores(self):
        tool = ToolDefinition(name="web_search", description="Search")
        assert tool.name == "web_search"

    def test_requires_approval(self):
        tool = ToolDefinition(
            name="delete", description="Delete", requires_approval=True
        )
        assert tool.requires_approval is True


class TestToolDefinitionNameCharset:
    """Tool names are a provider function-calling contract: ASCII, 1-64 chars.

    DECISION plan-2026-07-20T040150-876e7164/D-008 [STALE]. These pin BOTH directions —
    a fix that only tightened the charset would break `web-search`, and a fix that
    only capped the length would still accept Unicode.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "検索",  # CJK — accepted by the old str.isalnum() check (G-09)
            "ｆｕｌｌｗｉｄｔｈ１２３",  # fullwidth letters+digits (G-09)
            "café_lookup",  # Latin-1 accent (G-09)
            "search tool",  # embedded whitespace
            "search\n",  # trailing newline must not slip past the end anchor
            "bad-name!",  # punctuation outside the charset
            "",  # empty: rejected before this step and still rejected
        ],
    )
    def test_rejects_non_ascii_and_malformed_names(self, name):
        with pytest.raises(ValueError, match="alphanumeric"):
            ToolDefinition(name=name, description="d")

    @pytest.mark.parametrize(
        "name",
        [
            "web-search",  # hyphen: why core's ASCII_IDENTIFIER_PATTERN is wrong here
            "get_weather",
            "Tool123",
            "9lives",  # leading digit is legal for OpenAI function names
            "-lead",  # leading hyphen: deliberately allowed (spec-exact)
            "trail-",  # trailing hyphen: deliberately allowed (spec-exact)
            "___",  # underscores only: LOOSENED vs the old check, see D-008
            "--",  # hyphens only: LOOSENED vs the old check, see D-008
        ],
    )
    def test_accepts_ascii_names_the_providers_accept(self, name):
        assert ToolDefinition(name=name, description="d").name == name

    def test_length_boundary_is_exactly_64_both_sides(self):
        """An off-by-one in the cap is invisible to a one-sided test."""
        at_limit = "a" * 64
        assert ToolDefinition(name=at_limit, description="d").name == at_limit

        over_limit = "a" * 65
        with pytest.raises(ValueError, match="alphanumeric"):
            ToolDefinition(name=over_limit, description="d")

    def test_length_cap_counts_characters_not_bytes(self):
        """A 64-char cap must not be satisfiable by a short-but-multibyte name."""
        with pytest.raises(ValueError, match="alphanumeric"):
            ToolDefinition(name="é" * 10, description="d")


class TestToolCall:
    """Tests for ToolCall model."""

    def test_basic_creation(self):
        call = ToolCall(tool_name="search", parameters={"query": "test"})
        assert call.tool_name == "search"
        assert call.parameters == {"query": "test"}
        assert call.reasoning == ""

    def test_with_reasoning(self):
        call = ToolCall(
            tool_name="calc",
            parameters={"expr": "2+2"},
            reasoning="Need to compute",
        )
        assert call.reasoning == "Need to compute"


class TestToolResult:
    """Tests for ToolResult model."""

    def test_success_result(self):
        result = ToolResult(
            tool_name="search",
            success=True,
            result="Found: Paris has 2M people",
            execution_time_ms=150.0,
        )
        assert result.success is True
        assert "Paris" in result.summary

    def test_failure_result(self):
        result = ToolResult(
            tool_name="search",
            success=False,
            error="Network timeout",
        )
        assert result.success is False
        assert "Error: Network timeout" in result.summary

    def test_summary_truncation(self):
        long_result = "x" * 3000
        result = ToolResult(tool_name="t", success=True, result=long_result)
        assert len(result.summary) < 3000
        assert "truncated" in result.summary


class TestAgentStep:
    """Tests for AgentStep model."""

    def test_basic_creation(self):
        step = AgentStep(
            iteration=1,
            thought="Need to search",
            action="search({query: test})",
            observation="Found results",
        )
        assert step.iteration == 1
        assert step.timestamp is not None


class TestAgentTrace:
    """Tests for AgentTrace model."""

    def test_empty_trace(self):
        trace = AgentTrace()
        assert trace.total_iterations == 0
        assert trace.tools_used == []

    def test_tools_used(self):
        trace = AgentTrace(
            tool_calls=[
                ToolCall(tool_name="search", parameters={}),
                ToolCall(tool_name="calc", parameters={}),
                ToolCall(tool_name="search", parameters={"q": "2"}),
            ],
            total_iterations=3,
        )
        assert set(trace.tools_used) == {"search", "calc"}


class TestAgentConfig:
    """Tests for AgentConfig model."""

    def test_defaults(self):
        config = AgentConfig()
        assert config.max_iterations == 10
        assert config.timeout_seconds == 300.0
        assert config.temperature == 0.5

    def test_invalid_max_iterations(self):
        with pytest.raises(ValueError, match="at least 1"):
            AgentConfig(max_iterations=0)

    def test_custom_values(self):
        config = AgentConfig(model="gpt-4", max_iterations=20, timeout_seconds=60.0)
        assert config.model == "gpt-4"
        assert config.max_iterations == 20


_NUMERIC_FIELDS = (
    "max_iterations",
    "timeout_seconds",
    "temperature",
    "max_tokens",
    "max_history_size",
    "reflect_every_n",
    "auto_summarize_after",
)


class TestAgentConfigNumericFields:
    """F-AM-06b: no bool and no non-finite float is a configured number."""

    @pytest.mark.parametrize("field", _NUMERIC_FIELDS)
    @pytest.mark.parametrize("value", [True, False])
    def test_a_bool_is_refused_for_every_numeric_field(self, field, value):
        with pytest.raises(ValueError, match="not a bool"):
            AgentConfig(**{field: value})

    @pytest.mark.parametrize("field", ["timeout_seconds", "temperature"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_are_refused(self, field, value):
        with pytest.raises(ValueError):
            AgentConfig(**{field: value})

    @pytest.mark.parametrize("field", ["timeout_seconds", "temperature"])
    def test_a_non_finite_digit_string_is_refused(self, field):
        for text in ("nan", "inf", "-inf"):
            with pytest.raises(ValueError):
                AgentConfig(**{field: text})

    @pytest.mark.parametrize("field", ["timeout_seconds", "temperature"])
    @pytest.mark.parametrize(
        "big",
        [
            pytest.param(10**400, id="10e400"),
            pytest.param(-(10**400), id="neg-10e400"),
            pytest.param(10**5000, id="5000-digits"),
        ],
    )
    def test_an_int_too_large_for_a_float_is_a_clean_error(self, field, big):
        # The error path must not stringify the int in full (the 4300-digit
        # limit would turn it into a ValueError from the formatter).
        with pytest.raises(ValueError) as info:
            AgentConfig(**{field: big})
        assert len(str(info.value)) < 2000

    def test_the_timeout_message_names_the_rule(self):
        with pytest.raises(ValueError, match="finite positive"):
            AgentConfig(timeout_seconds=float("nan"))

    def test_valid_values_are_exactly_as_before(self):
        """Pin: ints for float fields, digit strings and integral floats coerce."""
        config = AgentConfig(
            max_iterations="7",
            timeout_seconds=30,
            temperature=1,
            max_tokens=5.0,
            max_history_size=3,
        )
        assert config.max_iterations == 7
        assert config.timeout_seconds == 30.0
        assert config.temperature == 1.0
        assert config.max_tokens == 5
        assert AgentConfig(temperature=0).temperature == 0.0
        assert AgentConfig(timeout_seconds=1e300).timeout_seconds == 1e300
        assert AgentConfig(max_history_size=None).max_history_size is None

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_model_is_refused(self, blank):
        with pytest.raises(ValueError, match="model must not be blank"):
            AgentConfig(model=blank)

    def test_meta_builder_config_inherits_the_rules(self):
        from fsm_llm.agents.definitions import MetaBuilderConfig

        with pytest.raises(ValueError):
            MetaBuilderConfig(timeout_seconds=float("nan"))
        with pytest.raises(ValueError):
            MetaBuilderConfig(temperature=True)
        assert MetaBuilderConfig().timeout_seconds > 0


class TestAgentResult:
    """Tests for AgentResult model."""

    def test_basic_result(self):
        result = AgentResult(answer="42", success=True)
        assert result.answer == "42"
        assert result.iterations_used == 0
        assert result.tools_used == []

    def test_with_trace(self):
        trace = AgentTrace(
            tool_calls=[ToolCall(tool_name="calc", parameters={})],
            total_iterations=3,
        )
        result = AgentResult(answer="42", success=True, trace=trace)
        assert result.iterations_used == 3
        assert result.tools_used == ["calc"]


class TestApprovalRequest:
    """Tests for ApprovalRequest model."""

    def test_basic_request(self):
        req = ApprovalRequest(
            tool_name="send_email",
            parameters={"to": "user@example.com"},
            reasoning="Need to notify",
        )
        assert req.tool_name == "send_email"
        assert req.parameters["to"] == "user@example.com"
