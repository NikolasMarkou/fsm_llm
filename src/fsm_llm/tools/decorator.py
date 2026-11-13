"""The ``@tool`` decorator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .hints import callable_name
from .models import ToolAnnotations, ToolDefinition
from .schema import _args_model_from_hints, infer_schema_from_hints


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameter_schema: dict[str, Any] | None = None,
    requires_approval: bool = False,
    annotations: ToolAnnotations | None = None,
    timeout_seconds: float | None = None,
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
    An inferred schema also gets an ``args_model`` (pydantic, from the
    signature) whose exact JSON schema ``get_json_schemas`` emits.
    ``annotations`` (:class:`ToolAnnotations`) and ``timeout_seconds`` go to the
    ``ToolDefinition``.

    The decorated function gains a ``_tool_definition`` attribute
    that can be used with ``ToolRegistry.register()``.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or callable_name(fn)
        raw_doc = fn.__doc__ or ""
        tool_desc = (
            description or raw_doc.strip().split("\n")[0] or f"Tool: {tool_name}"
        )

        args_model: type[BaseModel] | None = None
        if parameter_schema is not None:
            schema = parameter_schema
        else:
            schema = infer_schema_from_hints(fn)
            if schema:
                args_model = _args_model_from_hints(fn, tool_name)

        fn._tool_definition = ToolDefinition(  # type: ignore[attr-defined]
            name=tool_name,
            description=tool_desc.strip(),
            parameter_schema=schema,
            requires_approval=requires_approval,
            annotations=annotations or ToolAnnotations(),
            timeout_seconds=timeout_seconds,
            execute_fn=fn,
            args_model=args_model,
        )
        return fn

    if fn is not None:
        # Called as bare @tool (no parentheses)
        return decorator(fn)
    # Called as @tool(...) with keyword arguments
    return decorator
