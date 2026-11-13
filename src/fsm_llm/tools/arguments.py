"""Turning what a model sent into the arguments of a tool call, and showing them without secrets."""

from __future__ import annotations

import inspect
from typing import Any

from fsm_llm.context.filtering import redact_context
from fsm_llm.logging import logger

from .constants import ContextKeys
from .hints import _takes_mapping


# DECISION plan-2026-09-24T091842-c1d5bfbc/D-011: a list reaches a positional
# fallback parameter as a list only when its schema type is ``array``; any other
# (string, untyped) parameter gets ``str(list)``, the exact pre-D-011 value. Do
# NOT pass the raw list to a string parameter (``query.lower()`` then fails).
def _as_param_value(value: Any, prop: dict[str, Any] | None) -> Any:
    if isinstance(value, list) and (prop or {}).get("type") != "array":
        return str(value)
    return value


# Wrapper key for `redact_secret_entries`: `redact_context` matches mapping
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
    return redact_context({_REDACT_WRAPPER_KEY: value})[_REDACT_WRAPPER_KEY]


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
    # DECISION plan-2026-11-06T093029-0068b055/D-030 (extends D-009, step 12.35):
    # a stray value is never mapped into a KEYWORD-ONLY mapping-typed parameter.
    # A model that sends the flat object ``{"x": 1}`` for ``*, payload: dict``
    # would run the tool with ``payload=1``, a scalar in a dict parameter; the
    # call fails with the "requires parameters" message instead, so the model can
    # name it. Do NOT pass the whole flat object in as the dict: that is the
    # bare-dict rule, which is for a lone POSITIONAL parameter only. Do NOT widen
    # this to positional parameters here: tests of the HITL executor rely on the
    # old mapping for ``fn(params: dict, _name="x")`` (carried, CHANGELOG).
    mapping_params = {
        k
        for k, p in sig.parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY and _takes_mapping(p.annotation)
    }
    if len(names) == 1 and stray and names[0] not in mapping_params:
        # Single-param function: pass the first misnamed value
        mapped = {names[0]: _as_param_value(stray[0], schema_props.get(names[0]))}
        logger.debug(f"Tool kwarg mismatch, retrying: {redact_secret_entries(mapped)}")
        return mapped
    if params and schema_props:
        # Only positional-map a single required param from a single misnamed
        # value; for >=2 params the LLM's value order is untrusted and would
        # silently swap args.
        if (
            len(required) == 1
            and len(params) == 1
            and len(stray) == 1
            and required[0] not in mapping_params
        ):
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
