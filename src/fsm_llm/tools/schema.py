"""The JSON parameter schema a model is shown for a tool, inferred from type hints."""

from __future__ import annotations

import inspect
import typing
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model

from fsm_llm.logging import logger

from .hints import (
    _POSITIONAL_KINDS,
    _VAR_KINDS,
    _introspection_target,
    _is_dict_annotation,
    _is_mapping_hint,
    _without_none,
    callable_name,
)
from .models import ToolDefinition

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


def infer_schema_from_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """Infer a JSON-style parameter schema from a function's type hints.

    Supports standard Python types (str, int, float, bool, list, dict) and
    ``typing.Annotated[T, "description"]`` for per-parameter descriptions.

    Returns an empty dict for dict-style single-param functions (a positional
    ``params: dict``, ``dict[...]``, or unannotated; a keyword-only one keeps its
    named schema), for zero-parameter functions, and for a
    function whose only parameters are ``*args``/``**kwargs``. ``*args`` and
    ``**kwargs`` are never properties and never required: ``@tool`` and
    ``register_function`` share this one rule.
    """
    # A partial or a callable instance carries no annotations of its own: read
    # them from the wrapped function / the class's __call__ (review item 3).
    try:
        hints = get_type_hints(_introspection_target(fn), include_extras=True)
    except Exception as exc:
        logger.debug(f"Could not resolve type hints for {callable_name(fn)}: {exc}")
        return {}

    sig = inspect.signature(fn)
    all_params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]

    if not all_params:
        return {}

    # DECISION plan-2026-11-06T093029-0068b055/D-009: ONE rule for ``@tool`` and
    # ``register_function``. A lone POSITIONAL dict-style parameter (unannotated,
    # ``dict``, ``dict[...]``, or its string form) is the dict-style call: schema
    # ``{}``. A keyword-only one keeps its named schema and ``_invoke_tool_fn``
    # calls it by keyword (the dict-style call passes the dict positionally and
    # would raise TypeError); do NOT widen the rule to every kind.
    # ``*args``/``**kwargs`` are never properties (the model cannot name them)
    # and never required. Do NOT re-add a second pre-check in
    # ``register_function`` (the two drifted apart once), and do NOT list
    # ``**kw`` as a property: it bound as a required string and lost every
    # argument. Runtime side: ``_invoke_tool_fn`` passes extras to ``**kw``.
    if len(all_params) == 1 and all_params[0].kind in _POSITIONAL_KINDS:
        raw_hint = hints.get(all_params[0].name)
        if raw_hint is None or _is_dict_annotation(raw_hint):
            return {}
    params = [p for p in all_params if p.kind not in _VAR_KINDS]
    if not params:
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
        inner, elem = _without_none(hint), ()
        if inner is list or typing.get_origin(inner) in _LIST_ORIGINS:
            hint, elem = list, typing.get_args(inner)[:1]
        # A mapping type (``dict``/``dict[...]``, ``Mapping``, ``MutableMapping``,
        # a ``TypedDict``), bare or in Optional, is an ``object`` like bare
        # ``dict`` (the args model already says so); the "string" default
        # advertised it as a string. The lone positional dict-style rule above
        # (D-009) is untouched and stays dict-only.
        elif _is_mapping_hint(inner):
            hint = dict
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


# Keywords whose values are data, not schemas: never walked for ``title``.
_SCHEMA_DATA_KEYWORDS = frozenset({"default", "enum", "const", "examples"})


# Keywords whose values map names to schemas: the names are not keywords.
_SCHEMA_NAME_MAPS = frozenset({"properties", "$defs", "definitions"})


def _strip_schema_titles(node: Any) -> Any:
    """A copy of a JSON schema without pydantic's generated ``title`` keywords.

    Property names (even one called ``title``) and data values (``default``,
    ``enum``, ``const``, ``examples``) are kept as they are. Never raises.
    """
    if isinstance(node, list):
        return [_strip_schema_titles(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue
        if key in _SCHEMA_DATA_KEYWORDS:
            out[key] = value
        elif key in _SCHEMA_NAME_MAPS and isinstance(value, dict):
            out[key] = {name: _strip_schema_titles(s) for name, s in value.items()}
        else:
            out[key] = _strip_schema_titles(value)
    return out


def tool_parameters_schema(tool: ToolDefinition) -> dict[str, Any]:
    """The one parameter schema a model is shown for *tool*.

    Interface contract (callers: ``ToolRegistry.get_json_schemas`` for native
    tool calling, ``ToolRegistry.to_prompt_description`` and the think-state
    examples in ``prompts.py`` for prompt-mode agents):
        - The title-free JSON schema of ``tool.args_model`` when it has one
          (exact ``Optional``/``Union``/``Literal`` types, ``$defs`` kept),
          else ``tool.parameter_schema``, else ``{}``.
        - Returns the schema dict (``properties``, ``required``, ``$defs``
          when present); never raises.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-033: one schema source per
    # tool for everything a model is shown. Do NOT read
    # `tool.parameter_schema` directly in a prompt or a payload builder: a
    # prompt-mode agent was shown `Optional[int]` as `string` while the
    # native payload said integer-or-null. See decisions.md D-033.
    return _args_model_schema(tool.args_model) or tool.parameter_schema or {}


def schema_types(prop: Any, *, root: Mapping[str, Any] | None = None) -> list[str]:
    """The JSON types a property schema allows, in order, without repeats.

    Interface contract (callers: ``ToolRegistry.to_prompt_description`` and
    the think-state examples in ``prompts.py``, both passing the tool's
    ``tool_parameters_schema`` as *root*):
        - Reads ``type`` (a string or a list), ``anyOf``/``oneOf`` members and
          a local ``$ref`` (``#/...``, e.g. ``#/$defs/Color``), which is
          resolved against *root* and read as the schema it points to (a
          ``str`` Enum is ``string``, an ``IntEnum`` ``integer``, a model
          ``object``).
        - A ``$ref`` that cannot be resolved (no *root*, a non-local or
          dangling pointer, a reference cycle) adds no type: on its own the
          property reads as ``[]`` (any value), never as a guessed type.
        - ``[]`` when no type is stated (any value). Never raises.
    """
    return list(dict.fromkeys(_schema_types(prop, root, frozenset())))


def _schema_types(
    prop: Any, root: Mapping[str, Any] | None, seen: frozenset[str]
) -> list[str]:
    """``schema_types`` body; *seen* holds the ``$ref`` chain (cycle guard)."""
    if not isinstance(prop, dict):
        return []
    found: list[str] = []
    declared = prop.get("type")
    if isinstance(declared, str):
        found.append(declared)
    elif isinstance(declared, list):
        found.extend(t for t in declared if isinstance(t, str))
    # DECISION plan-2026-10-01T093600-944e2692/D-041: a `$ref` is read as the
    # schema it points to. Do NOT render it as "object": pydantic writes every
    # Enum parameter as `{"$ref": "#/$defs/Color"}`, and a prompt-mode model
    # told "object" sends an object the args_model rejects. Do NOT guess a
    # type for a ref that does not resolve: it reads as "any".
    ref = prop.get("$ref")
    if isinstance(ref, str) and ref not in seen:
        target = _resolve_local_ref(ref, root)
        if target is not None:
            found.extend(_schema_types(target, root, seen | {ref}))
    for key in ("anyOf", "oneOf"):
        members = prop.get(key)
        if isinstance(members, list):
            for member in members:
                found.extend(_schema_types(member, root, seen))
    return found


def _resolve_local_ref(ref: str, root: Mapping[str, Any] | None) -> Any:
    """The node a local JSON pointer (``#/a/b``) names in *root*, else ``None``."""
    if root is None or not ref.startswith("#/"):
        return None
    node: Any = root
    for raw in ref[2:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or token not in node:
            return None
        node = node[token]
    return node


def _args_model_schema(args_model: type[BaseModel] | None) -> dict[str, Any]:
    """The title-free JSON schema of *args_model*; ``{}`` when absent or failing."""
    if args_model is None:
        return {}
    try:
        return typing.cast(
            dict[str, Any], _strip_schema_titles(args_model.model_json_schema())
        )
    except Exception as exc:
        logger.debug(f"No JSON schema for {args_model.__name__}: {exc}")
        return {}


def _args_model_from_hints(
    fn: Callable[..., Any], tool_name: str
) -> type[BaseModel] | None:
    """A pydantic model of *fn*'s keyword arguments, built from its signature.

    Interface contract (callers: ``@tool`` and ``register_function``, only for
    a function whose inferred ``parameter_schema`` is non-empty):
        - One field per parameter (``self``/``cls`` skipped), typed by its
          resolved hint; ``Annotated[T, "desc"]`` becomes ``T`` with that
          description (other ``Annotated`` metadata is kept); a default makes
          the field optional.
        - Returns ``None`` when a parameter has no annotation, is ``*args``/
          ``**kwargs``, the hints cannot be resolved, or pydantic cannot build
          the model or its JSON schema: those tools keep the coarse
          ``parameter_schema`` and the registry binder.
        - Never raises.
    """
    try:
        hints = get_type_hints(_introspection_target(fn), include_extras=True)
        params = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.name not in ("self", "cls")
        ]
    except Exception as exc:
        logger.debug(f"No args model for {tool_name}: {exc}")
        return None
    fields: dict[str, Any] = {}
    for param in params:
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return None
        hint = hints.get(param.name)
        if hint is None:
            return None
        desc = None
        if typing.get_origin(hint) is typing.Annotated:
            base, *meta = typing.get_args(hint)
            desc = next((m for m in meta if isinstance(m, str)), None)
            kept = [m for m in meta if not isinstance(m, str)]
            hint = typing.Annotated[(base, *kept)] if kept else base
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (hint, Field(default, description=desc))
    try:
        # Field names that shadow BaseModel attributes only warn; the model works.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model: type[BaseModel] = create_model(
                f"{tool_name}_args",
                __config__=ConfigDict(arbitrary_types_allowed=True),
                **fields,
            )
            model.model_json_schema()
    except Exception as exc:
        logger.debug(f"No args model for {tool_name}: {exc}")
        return None
    return model
