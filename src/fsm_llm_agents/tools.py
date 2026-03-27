from __future__ import annotations

"""
Tool registry for agent tool management.
"""

import inspect
import time
import typing
from collections.abc import Callable
from typing import Any, get_type_hints

from fsm_llm.logging import logger

from .constants import ContextKeys, ErrorMessages
from .definitions import ToolCall, ToolDefinition, ToolResult
from .exceptions import ToolExecutionError, ToolNotFoundError

# Python type → JSON Schema type mapping for @tool auto-inference
_PYTHON_TO_JSON_SCHEMA: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def normalize_tool_input(raw: Any) -> dict[str, Any]:
    """Normalize tool input to a dict.

    Handles string, dict, None, and other types by wrapping non-dict values
    in ``{"input": value}``.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return {"input": raw}
    return {"input": str(raw)}


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
        if tool.name == ContextKeys.NO_TOOL:
            raise ValueError(
                f"Tool name '{tool.name}' is reserved (ContextKeys.NO_TOOL)"
            )
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

    @staticmethod
    def _validate_tool_params(
        tool: ToolDefinition,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate parameters against a tool's schema.

        Returns the schema ``properties`` dict (may be empty).
        """
        schema = tool.parameter_schema or {}
        required_keys = schema.get("required", [])
        if required_keys and isinstance(parameters, dict):
            missing = [k for k in required_keys if k not in parameters]
            if missing:
                logger.warning(
                    f"Tool '{tool.name}' missing required parameters: "
                    f"{missing}. Call may fail."
                )

        schema_props: dict[str, Any] = schema.get("properties", {})
        if schema_props and isinstance(parameters, dict):
            unknown = [k for k in parameters if k not in schema_props]
            if unknown:
                logger.warning(
                    f"Tool '{tool.name}' received unknown parameters: "
                    f"{unknown}. They will be ignored."
                )
        return schema_props

    @staticmethod
    def _invoke_tool_fn(
        fn: Callable[..., Any],
        parameters: dict[str, Any],
        schema_props: dict[str, Any],
    ) -> Any:
        """Call a tool function using the appropriate calling convention."""
        sig = inspect.signature(fn)
        param_count = len(sig.parameters)
        if param_count == 0:
            return fn()
        if param_count == 1 and not schema_props:
            # Legacy pattern: single param with no schema -> pass dict
            return fn(parameters)
        # Multi-param or schema-aware: pass as **kwargs
        params = parameters
        if schema_props and isinstance(params, dict):
            params = {k: v for k, v in params.items() if k in schema_props}
        return fn(**params)

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

            schema_props = self._validate_tool_params(tool, tool_call.parameters)
            result = self._invoke_tool_fn(fn, tool_call.parameters, schema_props)
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

    def register_agent(
        self,
        agent: Any,
        name: str,
        description: str,
    ) -> ToolRegistry:
        """Register an agent as a tool, enabling supervisor/orchestrator patterns.

        The agent must have a ``run(task: str)`` method returning an object with
        an ``answer`` attribute (i.e. :class:`AgentResult`).

        Args:
            agent: An agent instance with a ``.run()`` method.
            name: Tool name for the registry.
            description: Description exposed to the LLM.

        Returns:
            Self for chaining.
        """
        if not hasattr(agent, "run") or not callable(agent.run):
            raise ValueError(f"Agent must have a callable run() method: {agent}")

        def _agent_tool(task: str) -> str:
            result = agent.run(task)
            return str(getattr(result, "answer", result))

        return self.register_function(
            _agent_tool,
            name=name,
            description=description,
            parameter_schema={
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or query to send to the agent",
                    }
                },
                "required": ["task"],
            },
        )

    def to_classification_schema(self) -> dict[str, Any]:
        """
        Generate a ClassificationSchema-compatible dict for tool selection.

        Can be passed to fsm_llm.ClassificationSchema().
        """
        intents = [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]
        # Add a fallback intent for when no tool matches
        if not any(t["name"] == ContextKeys.NO_TOOL for t in intents):
            intents.append(
                {
                    "name": ContextKeys.NO_TOOL,
                    "description": "No tool is needed; answer directly or terminate",
                }
            )

        return {
            "intents": intents,
            "fallback_intent": ContextKeys.NO_TOOL,
            "confidence_threshold": 0.4,
        }

    def register_skill(self, skill: Any) -> ToolRegistry:
        """Register a ``SkillDefinition`` as a tool.

        Convenience method that calls ``skill.to_tool_definition()`` and
        registers the result.  Returns *self* for chaining.
        """
        return self.register(skill.to_tool_definition())


def _infer_schema_from_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """Infer a JSON-style parameter schema from a function's type hints.

    Supports standard Python types (str, int, float, bool, list, dict) and
    ``typing.Annotated[T, "description"]`` for per-parameter descriptions.

    Returns an empty dict for legacy single-dict-param functions (``params: dict``)
    and for zero-parameter functions.
    """
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception as exc:
        logger.debug(f"Could not resolve type hints for {fn.__name__}: {exc}")
        return {}

    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]

    if not params:
        return {}

    # Legacy pattern: single dict parameter → skip inference
    if len(params) == 1:
        raw_hint = hints.get(params[0].name)
        if raw_hint is dict or raw_hint is None:
            return {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in params:
        hint = hints.get(param.name)

        # Handle Annotated[T, "description"]
        param_desc = ""
        if typing.get_origin(hint) is typing.Annotated:
            args = typing.get_args(hint)
            real_type = args[0] if args else str
            for meta in args[1:]:
                if isinstance(meta, str):
                    param_desc = meta
                    break
            hint = real_type

        # Default to "string" for missing or unknown types
        if hint is None:
            hint = str
        json_type = _PYTHON_TO_JSON_SCHEMA.get(hint, "string")
        prop: dict[str, Any] = {"type": json_type}
        if param_desc:
            prop["description"] = param_desc

        properties[param.name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param.name)

    schema: dict[str, Any] = {}
    if properties:
        schema["properties"] = properties
    if required:
        schema["required"] = required
    return schema


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameter_schema: dict[str, Any] | None = None,
    requires_approval: bool = False,
) -> Any:
    """Decorator to mark a function as an agent tool.

    Supports three usage forms::

        @tool
        def search(query: str) -> str:
            \"\"\"Search the web.\"\"\"
            ...

        @tool(description="Search the web")
        def search(query: str) -> str: ...

        @tool(parameter_schema={"properties": {"query": {"type": "string"}}})
        def search(params: dict) -> str: ...

    When called without explicit *parameter_schema*, the decorator infers a
    JSON schema from the function's type hints (str→string, int→integer, etc.).
    Use ``typing.Annotated[str, "description"]`` for per-parameter descriptions.

    The decorated function gains a ``_tool_definition`` attribute
    that can be used with ``ToolRegistry.register()``.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        raw_doc = fn.__doc__ or ""
        tool_desc = (
            description or raw_doc.strip().split("\n")[0] or f"Tool: {tool_name}"
        )

        if parameter_schema is not None:
            schema = parameter_schema
        else:
            schema = _infer_schema_from_hints(fn)

        fn._tool_definition = ToolDefinition(  # type: ignore[attr-defined]
            name=tool_name,
            description=tool_desc.strip(),
            parameter_schema=schema,
            requires_approval=requires_approval,
            execute_fn=fn,
        )
        return fn

    if fn is not None:
        # Called as bare @tool (no parentheses)
        return decorator(fn)
    # Called as @tool(...) with keyword arguments
    return decorator
