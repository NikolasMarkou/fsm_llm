"""
Tool registry for agent tool management.
"""

from __future__ import annotations

import functools
import inspect
import json
import re
import threading
import time
import types
import typing
from collections.abc import Callable, Sequence
from typing import Any, get_type_hints

from fsm_llm.logging import logger
from fsm_llm.runner import _redact_context

from .constants import ContextKeys, ErrorMessages
from .definitions import ToolCall, ToolDefinition, ToolResult
from .exceptions import ToolExecutionError, ToolNotFoundError

# A string (``from __future__``) dict annotation: dict, dict[...], (typing.)Dict[...]
_DICT_ANNOTATION_RE = re.compile(r"\s*(dict|(typing\.)?Dict)(\[.*\])?\s*")

# Python type → JSON Schema type mapping for @tool auto-inference
_PYTHON_TO_JSON_SCHEMA: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}
_LIST_ORIGINS = (list, tuple, set, frozenset, Sequence)  # array-like generics


# DECISION plan-2026-09-24T091842-c1d5bfbc/D-011: a list reaches a positional
# fallback parameter as a list only when its schema type is ``array``; any other
# (string, untyped) parameter gets ``str(list)``, the exact pre-D-011 value. Do
# NOT pass the raw list to a string parameter (``query.lower()`` then fails).
def _as_param_value(value: Any, prop: dict[str, Any] | None) -> Any:
    if isinstance(value, list) and (prop or {}).get("type") != "array":
        return str(value)
    return value


# Wrapper key for `redact_secret_entries`: `_redact_context` matches mapping
# keys only, so a bare list or scalar is wrapped once. Not a secret-shaped name.
_REDACT_WRAPPER_KEY = "value"


def redact_secret_entries(value: Any) -> Any:
    """A copy of *value* with secret-looking entries' values redacted, for display.

    Contract:
        - Every mapping entry, at any nesting level (dicts inside dicts and
          inside lists/tuples), whose ``(key, value)`` matches
          ``fsm_llm.constants.is_forbidden_context_entry`` keeps its key and
          gets the value ``"<redacted>"``. Other values are unchanged.
        - A non-container value (a string, number, ``None``) is returned as is:
          it carries no key to match.
        - Never mutates *value*; mappings and lists in the result are new
          objects. Depth-bounded (fail closed) like the CLI log redaction.
        - Shared by every agent site that shows tool input or memory values in
          an observation, a trace, a log line or a tool reply.

    # DECISION plan-2026-09-29T103145-06a5ec0a/D-016
    # Redact a COPY for what is shown; the tool still receives the real input
    # and `ApprovalRequest.parameters` stays raw (the approver must see the
    # exact call). Do NOT inline a new key matcher or a second recursive walk
    # here: this reuses the core CLI redaction (`is_forbidden_context_entry`
    # at every level). Do NOT drop the key: the model and the reader should
    # still see which argument was passed. See decisions.md D-016.
    """
    return _redact_context({_REDACT_WRAPPER_KEY: value})[_REDACT_WRAPPER_KEY]


def normalize_tool_input(raw: Any) -> dict[str, Any]:
    """Normalize tool input to a dict.

    Handles string, dict, None, and other types by wrapping non-dict values
    in ``{"input": value}``. A string whose stripped text starts with ``{`` and
    decodes to a JSON object is returned as that dict. A list is wrapped as the
    list itself; other scalars are wrapped as their ``str()``.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        # DECISION plan-2026-09-19T175721-21cd7f8e/D-017: the core ``any`` union
        # (string/number/boolean/array/null, no ``object``, D-001) is what the
        # auto-minted ``tool_input`` field is held to on Ollama, so a JSON-object
        # tool input can only arrive as a JSON-encoded string. Decode it here, at
        # the single consumer seam. Do NOT add ``object`` back to the ``any``
        # union in core (reopens LV-01 for every auto-minted key) and do NOT wrap
        # the decoded text as ``{"input": raw}`` (kwargs tools then fail with
        # ``unexpected keyword argument 'input'``). Non-object JSON and invalid
        # JSON keep the ``{"input": raw}`` fallback. See decisions.md D-017.
        # (A JSON-array *string* stays a string here; a real list value, which
        # the ``array`` member of the union yields, is kept below: D-011 of
        # plan-2026-09-24T091842-c1d5bfbc.)
        if raw.lstrip().startswith("{"):
            try:
                parsed = json.loads(raw)
            except (ValueError, RecursionError):
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        return {"input": raw}
    # DECISION plan-2026-09-24T091842-c1d5bfbc/D-011: the core ``any`` union
    # includes ``array``, so a model can emit a list tool_input. Wrap the list
    # itself. Do NOT let it fall to the ``str(raw)`` catch-all (a list-typed
    # tool parameter then got the repr "['a', 'b']") and do NOT widen the
    # union in core to reach this (D-017 above still holds).
    if isinstance(raw, list):
        return {"input": raw}
    return {"input": str(raw)}


def _is_dict_annotation(ann: Any) -> bool:
    """True for ``dict``, a ``dict[...]`` generic, or a string annotation
    (``from __future__``) spelling one of those. Never raises."""
    return (
        ann is dict
        or typing.get_origin(ann) is dict
        or (isinstance(ann, str) and _DICT_ANNOTATION_RE.fullmatch(ann) is not None)
    )


def _unwrap_nested_tool_input(
    parameters: dict[str, Any], schema_props: dict[str, Any]
) -> dict[str, Any]:
    """Flatten ``{"tool_input": {...}, <siblings>}`` into one argument dict.

    Models sometimes echo the envelope ``{"tool_name": "x", "tool_input":
    {"a": 1}, "b": 2}``. Returns the nested dict merged with the siblings
    (``tool_name`` dropped). Precedence: the NESTED value wins on a key clash;
    a sibling fills only keys the nested dict lacks (TOOL-02: siblings used to
    be dropped). Returns ``parameters`` unchanged when there is no dict
    ``tool_input`` or the schema declares a real ``tool_input`` parameter.
    Never raises.
    """
    nested = parameters.get(ContextKeys.TOOL_INPUT)
    if not isinstance(nested, dict) or ContextKeys.TOOL_INPUT in schema_props:
        return parameters
    skip = {ContextKeys.TOOL_INPUT, ContextKeys.TOOL_NAME}
    merged = {k: v for k, v in parameters.items() if k not in skip}
    merged.update(nested)
    logger.debug(f"Unwrapped nested tool_input: {redact_secret_entries(merged)}")
    return merged


def _callable_name(fn: Any) -> str:
    """A display name for any callable: ``__name__``, else its type's name.

    ``functools.partial`` objects and callable instances have no ``__name__``
    (review item 3: ``register_function`` raised AttributeError for them).
    """
    return getattr(fn, "__name__", None) or type(fn).__name__


def _introspection_target(fn: Any) -> Any:
    """The function whose hints and coroutine flag describe *fn*.

    Unwraps ``functools.partial`` (nested too) to its ``func``; a callable
    instance maps to its class's ``__call__``. Functions, methods and builtins
    are returned as is. Never raises.
    """
    while isinstance(fn, functools.partial):
        fn = fn.func
    if inspect.isroutine(fn):
        return fn
    return type(fn).__call__ if callable(fn) else fn


def _bind_by_name(
    sig: inspect.Signature, params: dict[str, Any]
) -> inspect.BoundArguments:
    """``sig.bind`` for keyword input, placing positional-only parameters by position.

    A model can only send named values, so a positional-only parameter named
    in *params* is passed positionally (in order, up to the first one
    missing). Raises ``TypeError`` exactly like ``Signature.bind``.
    """
    positional: list[Any] = []
    rest = dict(params)
    for param in sig.parameters.values():
        if (
            param.kind is not inspect.Parameter.POSITIONAL_ONLY
            or param.name not in rest
        ):
            break
        positional.append(rest.pop(param.name))
    return sig.bind(*positional, **rest)


def _fallback_arguments(
    sig: inspect.Signature,
    params: dict[str, Any],
    schema_props: dict[str, Any],
    bind_error: TypeError,
) -> dict[str, Any]:
    """Recover keyword arguments after ``sig.bind(**params)`` failed.

    Only runs on a bind failure, so the tool has not been called yet; the
    caller calls it once with the result. A value is remapped only from a key
    that names no parameter of the function (a misnamed key): a named optional
    such as ``limit`` is never fed to a missing required such as ``query``
    (TOOL-02). Raises ``TypeError`` with a model-readable message when nothing
    can be recovered.
    """
    names = list(sig.parameters)
    required = [
        k for k, p in sig.parameters.items() if p.default is inspect.Parameter.empty
    ]
    stray = [v for k, v in params.items() if k not in sig.parameters]
    if len(names) == 1 and stray:
        # Single-param function: pass the first misnamed value
        mapped = {names[0]: _as_param_value(stray[0], schema_props.get(names[0]))}
        logger.debug(f"Tool kwarg mismatch, retrying: {redact_secret_entries(mapped)}")
        return mapped
    if params and schema_props:
        # Only positional-map a single required param from a single misnamed
        # value; for >=2 params the LLM's value order is untrusted and would
        # silently swap args.
        if len(required) == 1 and len(params) == 1 and len(stray) == 1:
            mapped = {
                required[0]: _as_param_value(stray[0], schema_props.get(required[0]))
            }
            logger.debug(
                "Tool kwarg mismatch, retrying with positional mapping: "
                f"{redact_secret_entries(mapped)}"
            )
            return mapped
        raise TypeError(
            f"Tool requires parameters: {required}. "
            f"Model sent keys {list(params.keys())} which do not match. "
            f"Provide values keyed by the expected parameter names."
        ) from None
    if not params and schema_props:
        required_params = list(schema_props.keys())
        example_params = {
            k: f"<{v.get('type', 'value')}>" for k, v in schema_props.items()
        }
        raise TypeError(
            f"Tool requires parameters: {required_params}. "
            f"Model sent empty parameters. "
            f"Expected format: {example_params}"
        ) from None
    raise bind_error


class ToolRegistry:
    """
    Registry for managing tools available to agents.

    Provides registration, lookup, prompt generation, and execution.
    """

    def __init__(self) -> None:
        # DECISION plan-2026-07-20T040150-876e7164/D-005 [STALE]: single NON-REENTRANT lock
        # guarding EVERY read, write and iteration of `_tools`. Registration racing
        # prompt/schema building was measured to raise
        # `RuntimeError: dictionary changed size during iteration` in 20/20 trials
        # (`ParallelReactAgent` dispatches tools on a ThreadPoolExecutor while the
        # main thread rebuilds the prompt), so the lock is NOT optional on the
        # iteration sites — the race came from iteration racing mutation, not from
        # concurrent mutation alone. Do NOT "fix" a deadlock here by switching to
        # `RLock`: every acquisition is leaf-level and releases before calling any
        # other method, mirroring `WorkingMemory._lock`'s shell/`_locked()` split
        # (D-007 of plan-2026-07-19T191147-4b664252) and
        # `CachingToolRegistry._cache_lock`. Iterating methods snapshot via
        # `list_tools()` and iterate the snapshot outside the lock, so the lock is
        # never held across a tool `execute_fn`, an LLM call or a user callback.
        self._tools_lock = threading.Lock()
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> ToolRegistry:
        """Register a tool definition. Returns self for chaining."""
        if tool.execute_fn is None:
            raise ValueError(f"Tool '{tool.name}' must have an execute_fn")
        if tool.name == ContextKeys.NO_TOOL:
            raise ValueError(
                f"Tool name '{tool.name}' is reserved (ContextKeys.NO_TOOL)"
            )
        with self._tools_lock:
            replaced = tool.name in self._tools
            self._tools[tool.name] = tool
        if replaced:
            logger.warning(
                f"Tool '{tool.name}' already registered; replacing it (last wins)"
            )
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
        tool_name = name or _callable_name(fn)
        tool_desc = description or fn.__doc__ or f"Tool: {tool_name}"

        if parameter_schema is None:
            # TOOL-03: infer like ``@tool`` so a one-param annotated function is
            # called by keyword, not handed the whole dict. A single unannotated
            # or dict-annotated param keeps ``{}`` (dict-style call). A function
            # taking ``**kwargs`` keeps ``{}`` so every key still reaches it.
            sig_params = list(inspect.signature(fn).parameters.values())
            dict_style = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig_params
            ) or (
                len(sig_params) == 1 and _is_dict_annotation(sig_params[0].annotation)
            )
            parameter_schema = {} if dict_style else _infer_schema_from_hints(fn)

        tool = ToolDefinition(
            name=tool_name,
            description=tool_desc.strip(),
            parameter_schema=parameter_schema,
            requires_approval=requires_approval,
            execute_fn=fn,
        )
        return self.register(tool)

    def get(self, name: str) -> ToolDefinition:
        """Get a tool by name."""
        with self._tools_lock:
            tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools.

        Returns a point-in-time **snapshot**: callers (including this class's own
        ``to_prompt_description``/``get_json_schemas``/``to_classification_schema``)
        iterate the returned list, never the live dict, so a concurrent
        ``register()`` cannot raise ``RuntimeError: dictionary changed size during
        iteration``. Never holds ``_tools_lock`` on return.
        """
        with self._tools_lock:
            return list(self._tools.values())

    @property
    def tool_names(self) -> list[str]:
        """Get all registered tool names."""
        with self._tools_lock:
            return list(self._tools.keys())

    def __len__(self) -> int:
        with self._tools_lock:
            return len(self._tools)

    def __contains__(self, name: str) -> bool:
        with self._tools_lock:
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
        # is not the legacy dict param. Do NOT pass it the dict positionally:
        # every zero-argument MCP tool and monitor stub tool failed that way.
        var_kw = first_param.kind is inspect.Parameter.VAR_KEYWORD
        if param_count == 1 and not schema_props and not var_kw:
            # Dict-style tool: single param with no schema -> pass dict
            return fn(parameters)
        # Detect the dict-style pattern: fn(params: dict) with schema
        ann = first_param.annotation
        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-024
        # A dict[...] generic or a string annotation (__future__) is the legacy
        # form only when the schema does not name the parameter; do NOT widen
        # the bare-`dict` rule, a schema-named dict param is called by keyword.
        if param_count == 1 and (
            ann is dict
            or (_is_dict_annotation(ann) and first_param.name not in schema_props)
        ):
            # A dict-style function expects a single dict; pass parameters directly
            return fn(parameters)
        # Multi-param or schema-aware: pass as **kwargs
        sent = _unwrap_nested_tool_input(parameters, schema_props)
        params = sent
        if schema_props:
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
            bound = _bind_by_name(
                sig, _fallback_arguments(sig, sent, schema_props, bind_error)
            )
        return fn(*bound.args, **bound.kwargs)

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        # Single locked lookup instead of `in` + `[]`: the two-step form could see
        # the tool present and then KeyError when a concurrent caller replaced the
        # dict entry between the check and the read.
        with self._tools_lock:
            tool = self._tools.get(tool_call.tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                error=ErrorMessages.TOOL_NOT_FOUND.format(name=tool_call.tool_name),
            )

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
        tools = self.list_tools()
        if not tools:
            return "No tools available."

        lines = ["Available tools:"]
        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description}")
            if tool.parameter_schema:
                params = tool.parameter_schema.get("properties", {})
                if params:
                    required_keys = set(tool.parameter_schema.get("required", []))
                    param_parts = []
                    for pname, pschema in params.items():
                        ptype = pschema.get("type", "any")
                        pdesc = pschema.get("description", "")
                        marker = (
                            "[REQUIRED]" if pname in required_keys else "[optional]"
                        )
                        param_parts.append(f"{pname} {marker} ({ptype}): {pdesc}")
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
        an ``answer`` attribute (i.e. :class:`AgentResult`). A result whose
        ``success`` is ``False`` fails the tool call (``ToolExecutionError``,
        so the caller sees a failed ``ToolResult``); a result without a
        ``success`` attribute counts as successful.

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

        The ``parameters`` object is derived from each tool's
        ``parameter_schema`` (``properties`` + ``required``). Tools with no
        schema get an empty-properties object. This is additive — the default
        agent dispatch path remains JSON-in-prompt; this method only enables
        opt-in native function calling (see ``NativeFunctionCallingReactAgent``).
        """
        schemas: list[dict[str, Any]] = []
        for tool in self.list_tools():
            schema = tool.parameter_schema or {}
            parameters: dict[str, Any] = {
                "type": "object",
                "properties": schema.get("properties", {}),
            }
            required = schema.get("required")
            if required:
                parameters["required"] = required
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

    Returns an empty dict for dict-style single-param functions (``params: dict``)
    and for zero-parameter functions.
    """
    # A partial or a callable instance carries no annotations of its own: read
    # them from the wrapped function / the class's __call__ (review item 3).
    try:
        hints = get_type_hints(_introspection_target(fn), include_extras=True)
    except Exception as exc:
        logger.debug(f"Could not resolve type hints for {_callable_name(fn)}: {exc}")
        return {}

    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]

    if not params:
        return {}

    # Dict-style tool: single dict parameter → skip inference
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

        # DECISION plan-2026-09-24T091842-c1d5bfbc/D-011 (P2-W4): a list-like
        # generic, bare or in Optional, is an ``array`` with ``items`` (OpenAI
        # rejects an array without them). Do NOT map it to "string": the
        # positional fallback then passes str(list). Other Optionals unchanged.
        inner, elem = hint, ()
        if typing.get_origin(hint) in (typing.Union, types.UnionType):
            opts = [a for a in typing.get_args(hint) if a is not type(None)]
            inner = opts[0] if len(opts) == 1 else hint
        if inner is list or typing.get_origin(inner) in _LIST_ORIGINS:
            hint, elem = list, typing.get_args(inner)[:1]
        # Default to "string" for missing or unknown types
        if hint is None:
            hint = str
        json_type = _PYTHON_TO_JSON_SCHEMA.get(hint, "string")
        prop: dict[str, Any] = {"type": json_type}
        if elem:
            prop["items"] = {"type": _PYTHON_TO_JSON_SCHEMA.get(elem[0], "string")}
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
        tool_name = name or _callable_name(fn)
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


def register_agent(
    registry: ToolRegistry,
    agent: Any,
    name: str,
    description: str,
) -> ToolRegistry:
    """Convenience function to register an agent as a tool.

    Equivalent to ``registry.register_agent(agent, name, description)``.
    """
    return registry.register_agent(agent, name, description)
