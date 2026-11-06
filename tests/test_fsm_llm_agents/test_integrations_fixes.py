"""Agents integrations review fixes (plan 5980fbb7 step 15).

Each class pins one finding (F-AI-nn) of ``findings/agents-integrations.md``;
every test fails on the parent commit a740e0b.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from fsm_llm.agents import (
    OKFMemoryBackend,
    PythonToolLoader,
    SemanticMemoryStore,
    ToolRegistry,
    create_okf_tools,
)
from fsm_llm.agents.definitions import AgentResult, AgentTrace, ToolCall
from fsm_llm.agents.mcp import _format_mcp_result
from fsm_llm.agents.semantic_tools import SemanticToolRegistry
from fsm_llm.agents.sop import SOPRegistry
from fsm_llm.agents.swarm import SwarmAgent
from fsm_llm.memory import WorkingMemory
from fsm_llm.okf import OKFBundle, OKFConcept


def _agent(
    answer: str = "done",
    *,
    iterations: int = 1,
    calls: int = 0,
    context: dict[str, Any] | None = None,
) -> Mock:
    agent = Mock()
    agent.run = Mock(
        return_value=AgentResult(
            answer=answer,
            success=True,
            trace=AgentTrace(
                total_iterations=iterations,
                tool_calls=[ToolCall(tool_name="t", parameters={})] * calls,
            ),
            final_context=context or {},
        )
    )
    return agent


class TestEmptyCallerContainerIsKept:
    """F-AI-01: ``x or Default()`` replaced an empty (falsy) caller-owned one."""

    def test_swarm_writes_to_an_empty_caller_memory(self):
        memory = WorkingMemory()
        swarm = SwarmAgent(agents={"a": _agent()}, entry_agent="a", memory=memory)
        swarm.run("task")
        assert swarm._memory is memory
        assert memory.get("metadata", "current_agent") == "a"

    def test_to_tool_registry_fills_an_empty_caller_registry(self):
        def echo(text: str) -> str:
            """Echo."""
            return text

        from fsm_llm.agents import tool

        tools = PythonToolLoader.from_functions(tool(echo))
        registry = ToolRegistry()
        returned = PythonToolLoader.to_tool_registry(tools, registry=registry)
        assert returned is registry
        assert len(registry) == 1


def _bundle() -> OKFBundle:
    return OKFBundle(
        [
            OKFConcept(
                id="docs/zephyr",
                type="Doc",
                title="Zephyr gateway",
                body="The Zephyr gateway accepts payloads up to 7341 KB.\n",
            ),
            OKFConcept(id="docs/banana", type="Doc", title="Banana", body="Fruit."),
        ]
    )


class TestOKFBackendStoreOutage:
    """F-AI-02: a store that cannot embed hid the lexical fallback."""

    def test_provider_down_after_reindex_falls_back_to_lexical(self):
        up = {"ok": True}

        def embed(text: str) -> list[float]:
            if not up["ok"]:
                raise RuntimeError("embedding provider down")
            return [1.0 if "zephyr" in text.lower() else 0.0, 0.1]

        store = SemanticMemoryStore(embed_fn=embed)
        backend = OKFMemoryBackend(_bundle(), store=store)
        backend.reindex()
        question = "what payload size does the zephyr gateway accept"
        assert backend.search(question)[0][2]["id"] == "docs/zephyr"
        up["ok"] = False
        hits = backend.search(question)
        assert [h[2]["id"] for h in hits][:1] == ["docs/zephyr"]


class TestSwarmIterationsAgreeWithGraph:
    """F-AI-07: Swarm summed agents run, AgentGraph sums member iterations."""

    def test_total_iterations_sums_member_iterations(self):
        a = _agent("one", iterations=5, context={"next_agent": "b"})
        b = _agent("two", iterations=5)
        result = SwarmAgent(agents={"a": a, "b": b}, entry_agent="a").run("task")
        assert result.trace.total_iterations == 10

    def test_failure_keeps_what_already_ran(self):
        a = _agent("one", iterations=4, calls=1, context={"next_agent": "b"})
        b = Mock()
        b.run = Mock(side_effect=RuntimeError("boom"))
        result = SwarmAgent(agents={"a": a, "b": b}, entry_agent="a").run("task")
        assert result.success is False
        assert result.trace.total_iterations == 4
        assert len(result.trace.tool_calls) == 1
        assert [t["agent"] for t in result.final_context["_swarm_traces"]] == ["a"]
        assert result.final_context["_swarm_handoff_count"] == 1


class TestOKFBoolOffset:
    """F-AI-10 (OKF side): a bool offset is 0, as in the skill tools."""

    @pytest.mark.parametrize("tool", ["okf_read", "okf_list"])
    def test_bool_offset_is_zero(self, tool: str):
        registry = create_okf_tools(_bundle())
        params = {"id": "docs/zephyr"} if tool == "okf_read" else {}

        def run(offset: Any) -> str:
            result = registry.execute(
                ToolCall(tool_name=tool, parameters={**params, "offset": offset})
            )
            assert result.success, result.error
            return str(result.result)

        assert run(True) == run(0)


class _FakeResult(BaseModel):
    content: list[Any] = []
    isError: bool = False
    structuredContent: dict[str, Any] | None = None


class TestFormatMcpResultWithoutContent:
    """F-AI-14: an empty ``content`` showed the model a pydantic repr."""

    def test_structured_content_is_json(self):
        text = _format_mcp_result(_FakeResult(structuredContent={"answer": 42}))
        assert json.loads(text) == {"answer": 42}

    def test_nothing_at_all_is_empty_text(self):
        assert _format_mcp_result(_FakeResult()) == ""

    def test_content_still_wins(self):
        result = SimpleNamespace(
            content=[SimpleNamespace(text="hi")], structuredContent={"a": 1}
        )
        assert _format_mcp_result(result) == "hi"


class TestSemanticToolRetrieveTopK:
    """F-AI-18: ``top_k=0`` meant the default and ``-1`` all but one."""

    @staticmethod
    def _registry() -> SemanticToolRegistry:
        registry = SemanticToolRegistry(
            embed_fn=lambda texts: [[float(len(t)), 1.0] for t in texts]
        )
        for i in range(12):
            registry.register_function(
                lambda: "x", name=f"tool_{i}", description="d" * (i + 1)
            )
        return registry

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_top_k_is_refused(self, bad: int):
        with pytest.raises(ValueError, match="top_k"):
            self._registry().retrieve("query", top_k=bad)

    def test_a_valid_top_k_still_limits(self):
        assert len(self._registry().retrieve("query", top_k=2)) == 2


class TestSopLoadErrors:
    """F-AI-19: a bad SOP file raised three unrelated types."""

    @pytest.mark.parametrize(
        ("name", "text"),
        [
            ("list.yaml", "- a\n- b\n"),
            ("empty.yaml", ""),
            ("noname.json", '{"description": "x"}'),
            ("bad.json", "{not json"),
        ],
    )
    def test_one_error_type_naming_the_file(self, tmp_path: Path, name: str, text: str):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=name):
            SOPRegistry().register_from_file(path)


class TestSemanticStoreLoaders:
    """F-AI-13: ``load`` and the constructor disagreed on a blank file."""

    def test_blank_file_loads_as_an_empty_store(self, tmp_path: Path):
        path = tmp_path / "mem.json"
        path.write_text("  \n", encoding="utf-8")
        embed = lambda text: [1.0]  # noqa: E731
        assert len(SemanticMemoryStore(persist_path=str(path), embed_fn=embed)) == 0
        assert len(SemanticMemoryStore.load(str(path), embed_fn=embed)) == 0

    def test_invalid_payload_names_the_file(self, tmp_path: Path):
        path = tmp_path / "mem.json"
        path.write_text('{"entries": [1]}', encoding="utf-8")
        with pytest.raises(ValueError, match=r"mem\.json"):
            SemanticMemoryStore.load(str(path), embed_fn=lambda text: [1.0])
