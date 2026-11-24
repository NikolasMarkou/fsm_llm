"""
Tool helpers that belong to the agents package: reading a tool call out of a run's
context, the approval gate handler that reads it, the ``gated=`` contract of a
registry and registering an agent as a tool.

The registry, the tool models and the decorator live in ``fsm_llm.tools``.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from fsm_llm.agents.constants import ContextKeys, ErrorMessages
from fsm_llm.core.definitions import AgentError
from fsm_llm.policy import ApprovalPolicy, HumanInTheLoop
from fsm_llm.tools.models import ToolCall

if TYPE_CHECKING:
    from fsm_llm.tools.registry import ToolRegistry


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


def tool_call_from_context(context: Mapping[str, Any]) -> ToolCall:
    """The tool call a context holds: ``tool_name``, ``tool_input`` (normalized by
    :func:`normalize_tool_input`) and ``reasoning``.

    Shared by the approval gate (:func:`make_hitl_checker`), the approval
    driver and ``AgentHandlers.approval_refusal``, so all three judge the same
    call. A missing name is ``""``. Never raises for a missing key.
    """
    return ToolCall(
        tool_name=str(context.get(ContextKeys.TOOL_NAME) or ""),
        parameters=normalize_tool_input(context.get(ContextKeys.TOOL_INPUT)),
        reasoning=context.get(ContextKeys.REASONING, ""),
    )


def make_hitl_checker(
    hitl: HumanInTheLoop,
    policy: ApprovalPolicy | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a reusable HITL approval checker handler.

    Returns a handler function that checks whether the current tool call
    requires human approval based on *policy* (default: the HITL policy,
    ``hitl.requires_approval``).  Agents pass their shared approval predicate
    so the gate, the ``await_approval`` state and the refusal agree.  Intended
    to be registered as a ``CONTEXT_UPDATE`` handler on the ``TOOL_NAME`` key.
    """
    decide: ApprovalPolicy = policy if policy is not None else hitl.requires_approval

    def check_approval(context: dict[str, Any]) -> dict[str, Any]:
        tool_name = context.get(ContextKeys.TOOL_NAME)
        if not tool_name or tool_name == ContextKeys.NO_TOOL:
            return {}

        if decide(tool_call_from_context(context), context):
            return {ContextKeys.APPROVAL_REQUIRED: True}

        return {ContextKeys.APPROVAL_REQUIRED: False}

    return check_approval


def flagged_tool_names(tools: Any) -> list[str]:
    """Names of the tools in *tools* registered with ``requires_approval=True``.

    Interface contract (callers: ``BaseAgent._hitl_active``, the construction
    warning, ``BaseAgent._refuse_flagged_tools`` and the two library factories that run
    tools with no approval path, ``agents.rewoo_executor`` and ``agents.native_fc``):

    Args:
        tools: a ``ToolRegistry``, or ``None``/any object without
            ``list_tools`` (an agent with no registry).

    Returns:
        The flagged names in registration order (a snapshot; ``[]`` when there
        is no registry or nothing is flagged). Never raises.
    """
    list_tools = getattr(tools, "list_tools", None)
    if list_tools is None:
        return []
    return [t.name for t in list_tools() if getattr(t, "requires_approval", False)]


def refuse_execute_without_gated(tools: Any) -> None:
    """Raise ``AgentError`` when *tools*' ``execute`` cannot take ``gated=``.

    Interface contract (callers: the two components that pass ``gated=``:
    ``AgentHandlers.__init__``, built at the start of every run of the
    agents whose executor it is (ReAct family, Reflexion, ReasoningReact,
    PlanExecute), before any model call; and
    ``reasoning_react._ReasonToolRegistry.__init__`` for the caller's
    registry it delegates to). Agents that call ``execute(call)`` without
    ``gated`` (REWOO, ParallelReact, native_fc) never check: an older
    override still works there.

    Args:
        tools: a ``ToolRegistry`` (or any object with ``execute``), or
            ``None``/an object without ``execute`` (no check).

    Raises:
        AgentError: ``execute`` has neither a ``gated`` parameter that can be
            passed by keyword nor ``**kwargs`` (``EXECUTE_WITHOUT_GATED``,
            naming the registry class). An ``execute`` whose signature
            cannot be read is not refused.
    """
    # DECISION plan-2026-10-01T093600-944e2692/D-033: refuse where `gated`
    # is passed, when that component is built (before any model call). An
    # override written against the old `execute(self, tool_call)` failed only
    # at the first tool call, deep in a handler, with a bare TypeError that
    # the run turned into a failed tool. Do NOT pass `gated` only when True to
    # dodge the break: it would hide it until the first approved call. Do NOT
    # move this into agent constructors or check agents that never pass
    # `gated` (REWOO, ParallelReact, native_fc): an agent built outside any
    # dispatch error handling, or a registry spy on `execute` with the old
    # signature, would break (STOP IF 2 of plan 944e2692).
    execute = getattr(tools, "execute", None)
    if execute is None:
        return
    try:
        params = inspect.signature(execute).parameters
    except (TypeError, ValueError):
        return
    gated = params.get("gated")
    keyword_kinds = (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if gated is not None and gated.kind in keyword_kinds:
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    raise AgentError(
        ErrorMessages.EXECUTE_WITHOUT_GATED.format(registry=type(tools).__name__)
    )


def register_agent(
    registry: ToolRegistry,
    agent: Any,
    name: str,
    description: str,
    *,
    replace: bool = False,
) -> ToolRegistry:
    """Convenience function to register an agent as a tool.

    Equivalent to ``registry.register_agent(agent, name, description, replace=replace)``.
    """
    return registry.register_agent(agent, name, description, replace=replace)
