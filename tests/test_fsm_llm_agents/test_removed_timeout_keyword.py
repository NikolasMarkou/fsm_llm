"""The removed ``timeout_s`` spelling is refused, never silently dropped.

Pydantic ignores unknown keywords by default, so before ``ToolDefinition`` set
``extra="forbid"`` a tool built with ``timeout_s=0.01`` ran with no time limit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fsm_llm.agents import tool
from fsm_llm.agents.definitions import ToolDefinition
from fsm_llm.agents.python_tools import PythonToolDefinition, PythonToolLoader


def _run(**kwargs):
    return "ok"


class TestRemovedTimeoutKeyword:
    def test_tool_definition_refuses_timeout_s(self):
        with pytest.raises(ValueError, match="timeout_s"):
            ToolDefinition(name="x", description="d", execute_fn=_run, timeout_s=0.01)  # type: ignore[call-arg]

    def test_tool_definition_takes_timeout_seconds(self):
        definition = ToolDefinition(
            name="x", description="d", execute_fn=_run, timeout_seconds=0.01
        )
        assert definition.timeout_seconds == 0.01

    def test_tool_definition_refuses_any_unknown_keyword(self):
        with pytest.raises(ValueError, match="no_such_field"):
            ToolDefinition(name="x", description="d", no_such_field=1)  # type: ignore[call-arg]

    def test_tool_decorator_refuses_timeout_s(self):
        with pytest.raises(TypeError, match="timeout_s"):
            tool(timeout_s=1)  # type: ignore[call-overload]

    def test_python_tool_definition_refuses_timeout_s(self):
        with pytest.raises(TypeError, match="timeout_s"):
            PythonToolDefinition(name="x", description="d", execute=_run, timeout_s=1)  # type: ignore[call-arg]

    def test_a_tool_file_with_timeout_s_lands_in_the_load_errors(self, tmp_path: Path):
        (tmp_path / "old.py").write_text(
            "from fsm_llm.agents import tool\n"
            "@tool(timeout_s=1)\n"
            "def slow(query: str) -> str:\n"
            '    """Look up."""\n'
            "    return query\n"
        )

        report = PythonToolLoader.load_directory(tmp_path)

        assert report.tools == ()
        assert len(report.errors) == 1
        assert report.errors[0][0].endswith("old.py")
        assert "timeout_s" in report.errors[0][1]
