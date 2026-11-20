from __future__ import annotations

"""Tests for structured output support (output_schema on AgentConfig)."""

import json

from pydantic import BaseModel

from fsm_llm.agents import PromptChainAgent
from fsm_llm.agents.base import BaseAgent
from fsm_llm.agents.definitions import AgentConfig, AgentResult, ChainStep
from fsm_llm.core.codes import Status
from fsm_llm.testing import PromptGroundedLLM
from tests.test_fsm_llm_agents.support import _Out


class SimpleReport(BaseModel):
    title: str
    summary: str
    score: float


class DetailedReport(BaseModel):
    title: str
    findings: list[str]
    confidence: float


class ConcreteAgent(BaseAgent):
    """Minimal agent that returns a preset answer."""

    def __init__(self, answer="test", config=None, **api_kwargs):
        super().__init__(config, **api_kwargs)
        self._answer = answer

    def run(self, task, initial_context=None):
        answer = self._answer
        structured = self._try_parse_structured_output(answer)
        return AgentResult(
            answer=answer,
            status=Status.OK,
            structured_output=structured,
        )

    def _register_handlers(self, api):
        pass


class TestStructuredOutput:
    """Tests for output_schema on AgentConfig."""

    def test_no_schema_returns_none(self):
        agent = ConcreteAgent(answer="plain text")
        result = agent.run("test")
        assert result.structured_output is None

    def test_valid_json_parsed(self):
        data = {"title": "Test Report", "summary": "All good", "score": 0.95}
        config = AgentConfig(output_schema=SimpleReport)
        agent = ConcreteAgent(answer=json.dumps(data), config=config)
        result = agent.run("test")

        assert result.structured_output is not None
        assert isinstance(result.structured_output, SimpleReport)
        assert result.structured_output.title == "Test Report"
        assert result.structured_output.score == 0.95

    def test_json_in_code_block(self):
        data = {"title": "Report", "summary": "Summary text", "score": 0.8}
        answer = f"Here is the report:\n```json\n{json.dumps(data)}\n```"
        config = AgentConfig(output_schema=SimpleReport)
        agent = ConcreteAgent(answer=answer, config=config)
        result = agent.run("test")

        assert result.structured_output is not None
        assert result.structured_output.title == "Report"

    def test_invalid_json_returns_none(self):
        config = AgentConfig(output_schema=SimpleReport)
        agent = ConcreteAgent(answer="This is not JSON at all.", config=config)
        result = agent.run("test")

        assert result.structured_output is None
        assert result.answer == "This is not JSON at all."

    def test_wrong_schema_returns_none(self):
        """JSON is valid but doesn't match the schema."""
        data = {"wrong_key": "value"}
        config = AgentConfig(output_schema=SimpleReport)
        agent = ConcreteAgent(answer=json.dumps(data), config=config)
        result = agent.run("test")

        # Pydantic will fail validation — structured_output should be None
        assert result.structured_output is None
        assert result.success is True  # Agent still succeeds

    def test_list_field_parsed(self):
        data = {
            "title": "Findings",
            "findings": ["bug A", "bug B", "bug C"],
            "confidence": 0.75,
        }
        config = AgentConfig(output_schema=DetailedReport)
        agent = ConcreteAgent(answer=json.dumps(data), config=config)
        result = agent.run("test")

        assert result.structured_output is not None
        assert len(result.structured_output.findings) == 3
        assert result.structured_output.confidence == 0.75

    def test_str_uses_structured_output(self):
        data = {"title": "Report", "summary": "Good", "score": 1.0}
        config = AgentConfig(output_schema=SimpleReport)
        agent = ConcreteAgent(answer=json.dumps(data), config=config)
        result = agent.run("test")

        s = str(result)
        assert "Report" in s

    def test_str_uses_answer_when_no_schema(self):
        agent = ConcreteAgent(answer="plain answer")
        result = agent.run("test")
        assert str(result) == "plain answer"


class TestAgentConfigOutputSchema:
    """Tests for output_schema on AgentConfig."""

    def test_default_none(self):
        config = AgentConfig()
        assert config.output_schema is None

    def test_set_schema(self):
        config = AgentConfig(output_schema=SimpleReport)
        assert config.output_schema is SimpleReport

    def test_schema_excluded_from_serialization(self):
        config = AgentConfig(output_schema=SimpleReport)
        dumped = config.model_dump()
        assert "output_schema" not in dumped


class TestAgentResultStructuredOutput:
    """Tests for structured_output on AgentResult."""

    def test_default_none(self):
        result = AgentResult(answer="test", status=Status.OK)
        assert result.structured_output is None

    def test_set_structured_output(self):
        report = SimpleReport(title="T", summary="S", score=0.5)
        result = AgentResult(answer="test", status=Status.OK, structured_output=report)
        assert result.structured_output is report


class TestStructuredOutputDiagnostic:
    """DECISION plan_2026-05-30_26c9510a/D-001 [STALE] — a partial-context construction
    failure must emit a diagnostic instead of being silently swallowed."""

    def test_partial_context_logs_diagnostic_and_falls_through(self):
        from fsm_llm.logging import logger

        msgs: list[str] = []
        logger.enable("fsm_llm")  # library logging is off by default (D-004)
        sink_id = logger.add(lambda m: msgs.append(str(m)), level="DEBUG")
        try:
            config = AgentConfig(output_schema=SimpleReport)
            agent = ConcreteAgent(answer="prose, no json", config=config)
            # Only 'title' present → schema(**fields) raises (summary/score missing).
            result = agent._try_parse_structured_output(
                "prose, no json", {"title": "X"}
            )
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        # Falls through to JSON parse (prose → None), no crash.
        assert result is None
        # Diagnostic emitted naming the failed construction.
        assert any("context-key construction" in m for m in msgs)


class TestStructuredOutputPrefersTheModelsAnswer:
    """F-AP-01: a context key named like a schema field never replaces the answer."""

    def _agent(self) -> PromptChainAgent:
        return PromptChainAgent(
            chain=[
                ChainStep(
                    step_id="s",
                    name="s",
                    extraction_instructions="x",
                    response_instructions="y",
                )
            ],
            config=AgentConfig(output_schema=_Out),
            llm_interface=PromptGroundedLLM(),
        )

    def test_the_answer_json_outranks_a_caller_hint(self):
        answer = json.dumps({"title": "Model title", "answer": "x"})
        out = self._agent()._try_parse_structured_output(
            answer, {"title": "CALLER HINT"}
        )
        assert out == _Out(title="Model title", answer="x")

    def test_context_keys_still_fill_in_when_the_answer_is_prose(self):
        out = self._agent()._try_parse_structured_output(
            "prose, no json", {"title": "T", "answer": "A"}
        )
        assert out == _Out(title="T", answer="A")
