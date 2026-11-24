"""Tool registry: registration, lookup, prompt generation and execution."""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

from fsm_llm.core.codes import Status, ToolErrorCode
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.tool_observer import ToolHandle, begin_tool, end_tool
from fsm_llm.logging import logger

from .arguments import _bind_by_name, _fallback_arguments, _unwrap_nested_tool_input
from .constants import ContextKeys, Defaults, ErrorMessages, ToolRunStatus
from .errors import ToolArgumentError, ToolExecutionError, ToolNotFoundError
from .hints import (
    _accepts_extra_keywords,
    _introspection_target,
    _is_dict_annotation,
    callable_name,
)
from .models import ToolAnnotations, ToolCall, ToolDefinition, ToolResult
from .schema import (
    _args_model_from_hints,
    infer_schema_from_hints,
    schema_types,
    tool_parameters_schema,
)
from .timeouts import ToolTimedOut, call_with_timeout


def _error_code(error: Exception) -> ToolErrorCode:
    """The code of a failed call, from the class of the error (never its text)."""
    if isinstance(error, ToolTimedOut):
        return ToolErrorCode.TIMEOUT
    if isinstance(error, ToolArgumentError):
        return ToolErrorCode.INVALID_ARGS
    return ToolErrorCode.EXECUTION_FAILED


def _logged_status(outcome: ToolResult) -> str:
    """The ``status`` of a ``tool/result`` event: ``ok``, ``failed`` or ``unknown``."""
    if outcome.success:
        return Status.OK.value
    return ToolRunStatus.UNKNOWN if outcome.timed_out else Status.FAILED.value


class ToolRegistry:
    """
    Registry for managing tools available to agents.

    Provides registration, lookup, prompt generation, and execution.
    """

    def __init__(self) -> None:
        # DECISION plan-2026-07-20T040150-876e7164/D-005 [STALE]: ONE NON-REENTRANT leaf
        # lock guards EVERY read, write and iteration of the tools. It is the lock of
        # the `NameRegistry` below (core/names.py, plan ace7a8c8/D-049: the old private
        # `_tools` dict and `_tools_lock` are gone). Registration racing prompt/schema
        # building was measured to raise `RuntimeError: dictionary changed size during
        # iteration` in 20/20 trials (`ParallelReactAgent` dispatches tools on a
        # ThreadPoolExecutor while the main thread rebuilds the prompt), so the lock is
        # NOT optional on the iteration sites. Do NOT "fix" a deadlock here by switching
        # to `RLock`: every acquisition is leaf-level and releases before calling any
        # other method, mirroring `WorkingMemory._lock`'s shell/`_locked()` split
        # (D-007 of plan-2026-07-19T191147-4b664252) and
        # `CachingToolRegistry._cache_lock`. Iterating methods snapshot via
        # `list_tools()` and iterate the snapshot outside the lock, so the lock is
        # never held across a tool `execute_fn`, an LLM call or a user callback.
        self._registry: NameRegistry[ToolDefinition] = NameRegistry("tools")

    def register(self, tool: ToolDefinition, *, replace: bool = False) -> ToolRegistry:
        """Register a tool definition. Returns self for chaining.

        A name that is already taken raises ``NameCollisionError`` and leaves the first
        tool in place; ``replace=True`` replaces it (the tool keeps its position). The
        guard matters: a tool built ungated must never silently take the place of one
        that ``requires_approval``.
        """
        if tool.execute_fn is None:
            raise ValueError(f"Tool '{tool.name}' must have an execute_fn")
        if tool.name == ContextKeys.NO_TOOL:
            raise ValueError(
                f"Tool name '{tool.name}' is reserved (ContextKeys.NO_TOOL)"
            )
        replaced = replace and tool.name in self._registry
        self._registry.register(tool.name, tool, override=replace)
        if replaced:
            logger.debug(f"Replaced tool: {tool.name}")
        else:
            logger.debug(f"Registered tool: {tool.name}")
        return self

    def register_function(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        parameter_schema: dict[str, Any] | None = None,
        requires_approval: bool = False,
        *,
        annotations: ToolAnnotations | None = None,
        timeout_seconds: float | None = None,
        replace: bool = False,
    ) -> ToolRegistry:
        """Register a function as a tool. Returns self for chaining.

        ``annotations`` and ``timeout_seconds`` go to the ``ToolDefinition`` (see
        there). An inferred schema also gets an ``args_model``; an explicit
        ``parameter_schema`` does not. ``replace`` is that of ``register``.
        """
        tool_name = name or callable_name(fn)
        tool_desc = description or fn.__doc__ or f"Tool: {tool_name}"

        args_model: type[BaseModel] | None = None
        if parameter_schema is None:
            # TOOL-03: infer like ``@tool``; both go through the one rule in
            # ``infer_schema_from_hints`` (D-009), so they cannot disagree.
            parameter_schema = infer_schema_from_hints(fn)
            if parameter_schema:
                args_model = _args_model_from_hints(fn, tool_name)

        tool = ToolDefinition(
            name=tool_name,
            description=tool_desc.strip(),
            parameter_schema=parameter_schema,
            requires_approval=requires_approval,
            annotations=annotations or ToolAnnotations(),
            timeout_seconds=timeout_seconds,
            execute_fn=fn,
            args_model=args_model,
        )
        return self.register(tool, replace=replace)

    def get(self, name: str) -> ToolDefinition:
        """Get a tool by name."""
        tool = self._registry.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools.

        Returns a point-in-time **snapshot**: callers (including this class's own
        ``to_prompt_description``/``get_json_schemas``/``to_classification_schema``)
        iterate the returned list, never the live dict, so a concurrent
        ``register()`` cannot raise ``RuntimeError: dictionary changed size during
        iteration``. Never holds the registry's lock on return.
        """
        return [tool for _, tool in self._registry.items()]

    @property
    def tool_names(self) -> list[str]:
        """Get all registered tool names, in registration order."""
        return [name for name, _ in self._registry.items()]

    def __len__(self) -> int:
        return len(self._registry.items())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

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
            if unknown and tool.execute_fn is not None:
                try:
                    takes_extras = _accepts_extra_keywords(
                        inspect.signature(tool.execute_fn)
                    )
                except (TypeError, ValueError):
                    takes_extras = False
                if takes_extras:
                    unknown = []  # they reach the function's ``**kw``
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
        """Call a tool function using the appropriate calling convention.

        Tries **kwargs expansion first, then falls back to positional
        argument mapping when the model provides misnamed or missing keys.
        Handles async tool functions (e.g. MCP tools) by running them
        synchronously via asyncio.
        """
        # Handle async tool functions (e.g. MCP tools)
        # DECISION plan_2026-05-29_d9092060/D-006 [STALE]
        # original_fn is assigned unconditionally so inspect.signature always sees
        # the real function, not the sync wrapper.  The coroutine is created
        # inside the worker lambda so it belongs to the worker thread's event loop
        # (A-ISSUE-001, A-ISSUE-002).
        original_fn = fn
        # A partial of a coroutine function, or an instance with an async
        # __call__, is async too (review item 3).
        is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            _introspection_target(fn)
        )
        if is_async:
            import asyncio

            def fn(*args: Any, **kwargs: Any) -> Any:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None:
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        # Create the coroutine inside the worker thread so it
                        # belongs to the new event loop, not the calling thread.
                        return pool.submit(
                            lambda: asyncio.run(original_fn(*args, **kwargs))
                        ).result()
                return asyncio.run(original_fn(*args, **kwargs))

        sig = inspect.signature(original_fn)
        param_count = len(sig.parameters)
        if param_count == 0:
            return fn()
        first_param = next(iter(sig.parameters.values()))  # param_count >= 1 here
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-024: a lone ``**kwargs``
        # is not the dict-style param. Do NOT pass it the dict positionally:
        # every zero-argument MCP tool and monitor stub tool failed that way.
        # DECISION plan-2026-11-06T093029-0068b055/D-018 (extends D-009): the
        # dict goes in BY POSITION, so only a parameter that can take one is
        # dict-style. A keyword-only ``payload: dict`` (plain, ``dict[...]``, or
        # ``= None``) is a named schema property: call it by keyword. Do NOT pass
        # it the dict positionally (TypeError on every call) and do NOT widen
        # the bare-dict rule beyond the single-parameter case.
        by_position = first_param.kind not in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        )
        if param_count == 1 and not schema_props and by_position:
            # Dict-style tool: single param with no schema -> pass dict
            return fn(parameters)
        # Detect the dict-style pattern: fn(params: dict) with schema
        ann = first_param.annotation
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-024
        # A dict[...] generic or a string annotation (__future__) is the
        # dict-style form only when the schema does not name the parameter; do
        # NOT widen the bare-`dict` rule, a schema-named dict param is called
        # by keyword.
        if (
            param_count == 1
            and by_position
            and (
                ann is dict
                or (_is_dict_annotation(ann) and first_param.name not in schema_props)
            )
        ):
            # A dict-style function expects a single dict; pass parameters directly
            return fn(parameters)
        # Multi-param or schema-aware: pass as **kwargs
        sent = _unwrap_nested_tool_input(parameters, schema_props)
        params = sent
        if schema_props and not _accepts_extra_keywords(sig):
            params = {k: v for k, v in sent.items() if k in schema_props}
        # DECISION plan-2026-09-29T103145-06a5ec0a/D-023: bind BEFORE calling.
        # When the arguments bind, the tool runs exactly once and a TypeError
        # from its body is a failed call. Do NOT go back to ``try: fn(**params)
        # except TypeError: <retry>``: that re-ran a tool whose body raised
        # TypeError after a side effect (TOOL-01). The fallback sees the keys the
        # model sent (``sent``), before the schema filter, like the old one did.
        # Positional-only parameters bind by position from their named keys
        # (`_bind_by_name`); the fallback's arguments are bound the same way,
        # so a bind error never reaches the tool body.
        try:
            bound = _bind_by_name(sig, params)
        except TypeError as bind_error:
            try:
                bound = _bind_by_name(
                    sig, _fallback_arguments(sig, sent, schema_props, bind_error)
                )
            except TypeError as unbound:
                # the tool never ran: the one cause that is `invalid_args` (D-006)
                raise ToolArgumentError(
                    str(unbound), tool_name=callable_name(original_fn)
                ) from unbound
        return fn(*bound.args, **bound.kwargs)

    def execute(self, tool_call: ToolCall, *, gated: bool = False) -> ToolResult:
        """Execute a tool call and return the result. Never raises.

        ``gated=True`` marks a call an approver granted (passed by the HITL
        executor). This registry runs it like any call; subclasses that
        re-run calls (``RetryingToolRegistry``) must never re-run a gated one.

        A tool with ``timeout_seconds`` runs in a worker thread: past the limit the
        call returns a failed result with ``timed_out=True`` whose error says
        the outcome is unknown. A Python thread cannot be killed, so the tool
        keeps running and its side effects may still happen; its late result
        is discarded with a WARNING.
        """
        # Single locked lookup instead of `in` + `[]`: the two-step form could see
        # the tool present and then KeyError when a concurrent caller replaced the
        # dict entry between the check and the read.
        tool = self._registry.get(tool_call.tool_name)
        if tool is None:
            # nothing ran, so nothing is logged (D-006, C-6); the code says why
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                error=ErrorMessages.TOOL_NOT_FOUND.format(name=tool_call.tool_name),
                error_code=ToolErrorCode.UNKNOWN_TOOL,
            )

        start_time = time.monotonic()
        handle: ToolHandle | None = None
        error: Exception | None = None

        try:
            fn = tool.execute_fn
            if fn is None:
                raise ToolExecutionError(
                    "Tool has no execute function", tool_name=tool.name
                )

            schema_props = self._validate_tool_params(tool, tool_call.parameters)
            # DECISION plan-2026-11-24T193347-3f4530d3/D-006
            # The event log hook: `tool/call` is written (and fsynced for a side-effecting
            # call) HERE, after the arguments are validated and BEFORE the body, and a write
            # that fails raises into the `except` below: a failed result, the body never ran.
            # `tool/result` is written by THIS caller after the body returned, raised or timed
            # out (a timeout is `unknown`, never `failed`); the worker thread never writes.
            # An unregistered name and a cache hit log nothing, a retry is its own call. Do NOT
            # import the log here (D-005), move the hook after the body, or let it raise out.
            handle = begin_tool(
                tool.name,
                tool_call.parameters,
                side_effecting=tool.annotations.read_only is not True,
                gated=gated,
            )
            if tool.timeout_seconds is None:
                result = self._invoke_tool_fn(fn, tool_call.parameters, schema_props)
            else:
                result = call_with_timeout(
                    functools.partial(
                        self._invoke_tool_fn, fn, tool_call.parameters, schema_props
                    ),
                    tool.name,
                    tool.timeout_seconds,
                )
            elapsed_ms = (time.monotonic() - start_time) * 1000

            outcome = ToolResult(
                tool_name=tool_call.tool_name,
                success=True,
                result=result,
                execution_time_ms=elapsed_ms,
            )

        except Exception as e:
            error = e
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                ErrorMessages.TOOL_EXECUTION_FAILED.format(
                    name=tool_call.tool_name, error=str(e)
                )
            )
            outcome = ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
                timed_out=isinstance(e, ToolTimedOut),
                error_code=_error_code(e),
            )
        self._log_outcome(handle, outcome, error)
        return outcome

    @staticmethod
    def _log_outcome(
        handle: ToolHandle | None, outcome: ToolResult, error: Exception | None
    ) -> None:
        """Write the ``tool/result`` of a call that has a ``tool/call``; never raises."""
        if handle is None:
            return
        try:
            end_tool(
                handle,
                status=_logged_status(outcome),
                error_type=None if error is None else type(error).__name__,
                error_code=None
                if outcome.error_code is None
                else outcome.error_code.value,
                truncated=outcome.truncated,
                elapsed_ms=round(outcome.execution_time_ms, 3),
            )
        except Exception as e:  # `truncated` reads a result's text: never let it escape
            logger.warning(f"tool result not logged ({type(e).__name__})")

    def to_prompt_description(self) -> str:
        """Generate a prompt-friendly description of all available tools."""
        tools = self.list_tools()
        if not tools:
            return "No tools available."

        lines = ["Available tools:"]
        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description}")
            schema = tool_parameters_schema(tool)
            params = schema.get("properties", {})
            if params:
                required_keys = set(schema.get("required", []))
                param_parts = []
                for pname, pschema in params.items():
                    ptype = " or ".join(schema_types(pschema, root=schema)) or "any"
                    pdesc = pschema.get("description", "")
                    marker = "[REQUIRED]" if pname in required_keys else "[optional]"
                    param_parts.append(f"{pname} {marker} ({ptype}): {pdesc}")
                lines.append(f"  Parameters: {', '.join(param_parts)}")

        return "\n".join(lines)

    def register_agent(
        self,
        agent: Any,
        name: str,
        description: str,
        *,
        replace: bool = False,
    ) -> ToolRegistry:
        """Register an agent as a tool, enabling supervisor/orchestrator patterns.

        The agent must have a ``run(task: str)`` method returning an object with
        an ``answer`` attribute (i.e. :class:`AgentResult`). A result whose
        ``success`` is ``False`` fails the tool call (``ToolExecutionError``,
        so the caller sees a failed ``ToolResult``); a result without a
        ``success`` attribute counts as successful.

        Args:
            agent: An agent instance with a ``.run()`` method.
            name: Tool name for the registry.
            description: Description exposed to the LLM.
            replace: As for ``register``.

        Returns:
            Self for chaining.
        """
        if not hasattr(agent, "run") or not callable(agent.run):
            raise ValueError(f"Agent must have a callable run() method: {agent}")

        def _agent_tool(task: str) -> str:
            result = agent.run(task)
            answer = str(getattr(result, "answer", result))
            if getattr(result, "success", True) is False:
                raise ToolExecutionError(
                    f"Agent '{name}' did not succeed: {answer}", tool_name=name
                )
            return answer

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
            replace=replace,
        )

    def get_json_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function-calling tool schemas.

        Produces the ``tools=[...]`` payload expected by provider-native
        function calling (OpenAI/Anthropic/litellm). Each entry has the shape::

            {
                "type": "function",
                "function": {
                    "name": str,
                    "description": str,
                    "parameters": {"type": "object", "properties": {...},
                                   "required": [...]},
                },
            }

        The ``parameters`` object is the tool's ``args_model`` JSON schema when
        it has one (``Optional``/``Union``/``Literal`` typed exactly, ``title``
        keywords dropped), else it is derived from ``parameter_schema``
        (``properties`` + ``required``). Tools with no schema get an
        empty-properties object.
        """
        schemas: list[dict[str, Any]] = []
        for tool in self.list_tools():
            schema = tool_parameters_schema(tool)
            parameters: dict[str, Any] = {
                "type": "object",
                "properties": schema.get("properties", {}),
            }
            required = schema.get("required")
            if required:
                parameters["required"] = required
            if "$defs" in schema:
                parameters["$defs"] = schema["$defs"]
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": parameters,
                    },
                }
            )
        return schemas

    def to_classification_schema(self) -> dict[str, Any]:
        """
        Generate a ClassificationSchema-compatible dict for tool selection.

        Can be passed to fsm_llm.ClassificationSchema().
        """
        intents = [
            {"name": tool.name, "description": tool.description}
            for tool in self.list_tools()
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
            "confidence_threshold": Defaults.TOOL_CONFIDENCE_THRESHOLD,
        }

    def register_python_tool(self, tool: Any, *, replace: bool = False) -> ToolRegistry:
        """Register a ``PythonToolDefinition`` as a tool.

        Convenience method that calls ``tool.to_tool_definition()`` and
        registers the result (``replace`` is that of ``register``).  Returns *self*
        for chaining.
        """
        return self.register(tool.to_tool_definition(), replace=replace)


def registry_from(tools: Iterable[Any]) -> ToolRegistry:
    """A new ``ToolRegistry`` holding *tools*, however each one is given.

    Interface contract (callers: ``create_agent(tools=[...])``,
    ``ConfiguredAgentBuilder.build``): each item is a ``ToolDefinition``, a
    ``@tool`` function (its ``_tool_definition``) or a plain function
    (``register_function``). Raises what ``register`` / ``register_function``
    raise (a duplicate name: ``NameCollisionError``, a bad signature, ``TypeError`` for a non-callable).
    """
    registry = ToolRegistry()
    for item in tools:
        if isinstance(item, ToolDefinition):
            registry.register(item)
        elif hasattr(item, "_tool_definition"):
            registry.register(item._tool_definition)
        else:
            registry.register_function(item)
    return registry
