from __future__ import annotations

"""
Tool registry for agent tool management.
"""

import inspect
import time
from collections.abc import Callable
from typing import Any

from fsm_llm.logging import logger

from .constants import ErrorMessages
from .definitions import ToolCall, ToolDefinition, ToolResult
from .exceptions import ToolExecutionError, ToolNotFoundError


class ToolRegistry:
    """
    Registry for managing tools available to agents.

    Provides registration, lookup, prompt generation, and execution.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> ToolRegistry:
        """Register a tool definition. Returns self for chaining."""
        if tool.execute_fn is None:
            raise ValueError(f"Tool '{tool.name}' must have an execute_fn")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
        return self

    def register_function(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        parameter_schema: dict[str, Any] | None = None,
        requires_approval: bool = False,
    ) -> ToolRegistry:
        """Register a function as a tool. Returns self for chaining."""
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or f"Tool: {tool_name}"

        tool = ToolDefinition(
            name=tool_name,
            description=tool_desc.strip(),
            parameter_schema=parameter_schema or {},
            requires_approval=requires_approval,
            execute_fn=fn,
        )
        return self.register(tool)

    def get(self, name: str) -> ToolDefinition:
        """Get a tool by name."""
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    @property
    def tool_names(self) -> list[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        if tool_call.tool_name not in self._tools:
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                error=ErrorMessages.TOOL_NOT_FOUND.format(name=tool_call.tool_name),
            )

        tool = self._tools[tool_call.tool_name]
        start_time = time.monotonic()

        try:
            fn = tool.execute_fn
            if fn is None:
                raise ToolExecutionError(
                    "Tool has no execute function", tool_name=tool.name
                )

            # Warn on missing required parameters per schema
            schema = tool.parameter_schema or {}
            required_keys = schema.get("required", [])
            if required_keys and isinstance(tool_call.parameters, dict):
                missing = [k for k in required_keys if k not in tool_call.parameters]
                if missing:
                    logger.warning(
                        f"Tool '{tool.name}' missing required parameters: "
                        f"{missing}. Call may fail."
                    )

            # Call with params as kwargs if function accepts them
            sig = inspect.signature(fn)
            if len(sig.parameters) == 0:
                result = fn()
            elif len(sig.parameters) == 1:
                result = fn(tool_call.parameters)
            else:
                result = fn(**tool_call.parameters)

            elapsed_ms = (time.monotonic() - start_time) * 1000

            return ToolResult(
                tool_name=tool_call.tool_name,
                success=True,
                result=result,
                execution_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                ErrorMessages.TOOL_EXECUTION_FAILED.format(
                    name=tool_call.tool_name, error=str(e)
                )
            )
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )

    def to_prompt_description(self) -> str:
        """Generate a prompt-friendly description of all available tools."""
        if not self._tools:
            return "No tools available."

        lines = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            if tool.parameter_schema:
                params = tool.parameter_schema.get("properties", {})
                if params:
                    param_parts = []
                    for pname, pschema in params.items():
                        ptype = pschema.get("type", "any")
                        pdesc = pschema.get("description", "")
                        param_parts.append(f"{pname} ({ptype}): {pdesc}")
                    lines.append(f"  Parameters: {', '.join(param_parts)}")

        return "\n".join(lines)

    def to_classification_schema(self) -> dict[str, Any]:
        """
        Generate a ClassificationSchema-compatible dict for tool selection.

        Can be passed to fsm_llm_classification.ClassificationSchema().
        """
        intents = [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]
        # Add a fallback intent for when no tool matches
        if not any(t["name"] == "none" for t in intents):
            intents.append(
                {
                    "name": "none",
                    "description": "No tool is needed; answer directly or terminate",
                }
            )

        return {
            "intents": intents,
            "fallback_intent": "none",
            "confidence_threshold": 0.4,
        }


def tool(
    name: str | None = None,
    description: str | None = None,
    parameter_schema: dict[str, Any] | None = None,
    requires_approval: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to mark a function as an agent tool.

    Usage::

        @tool(description="Search the web for information")
        def web_search(query: str) -> str:
            return search_engine.search(query)

    The decorated function gains a `_tool_definition` attribute
    that can be used with ToolRegistry.register().
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or f"Tool: {tool_name}"

        fn._tool_definition = ToolDefinition(  # type: ignore[attr-defined]
            name=tool_name,
            description=tool_desc.strip(),
            parameter_schema=parameter_schema or {},
            requires_approval=requires_approval,
            execute_fn=fn,
        )
        return fn

    return decorator
