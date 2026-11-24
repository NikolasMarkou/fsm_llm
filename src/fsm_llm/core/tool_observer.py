"""The tool observer seam (plan 3f4530d3, step 3): how a tool call reaches the event log.

``ToolRegistry.execute`` (layer ``tools``) is the one choke point where a tool body runs. The
event log (layer ``runtime``) lives above ``tools`` and a ``tools`` module may not import it
(contract ``l3-layers-import-no-llm``, since ``runtime.scope`` imports ``llm``). So the registry
calls the two helpers below and a consumer installs an object that implements ``ToolObserver``,
exactly as ``llm.observer`` does for the provider request. This module knows no consumer and
imports the standard library only.

Interface contract:
- ``install_tool_observer(observer) -> Token`` makes ``observer`` the active one for the current
  context (a ``ContextVar``, never process-global); ``reset_tool_observer(token)`` undoes it, in
  the context that installed it. ``install_tool_observer(None)`` installs "no observer". A thread
  sees the observer only when started through ``contextvars.copy_context().run`` (the tool
  timeout worker is; a plain ``ThreadPoolExecutor`` is not).
- ``begin_tool(name, arguments, *, side_effecting, gated)`` returns ``None`` without an observer
  (one ``ContextVar`` read, nothing built), else a ``ToolHandle``. The observer gets a READ-ONLY
  view of the arguments: it must not change what the tool receives. An exception from
  ``ToolObserver.begin_tool`` PROPAGATES (fail closed: the registry turns it into a failed
  ``ToolResult`` and the tool body never runs).
- ``end_tool(handle, ...)`` never raises: an observer error is a WARNING (type name only) and the
  call goes on. The caller that ran the tool writes it, once, after the body returned, raised or
  timed out; a worker thread never does.
- ``begin_approval`` and ``end_approval`` bracket one ask of ``HumanInTheLoop`` (D-007): BEST EFFORT,
  they never raise, so an unwritable log changes no verdict; nothing reads their handle back.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar, Token
from types import MappingProxyType
from typing import Any, NamedTuple, Protocol

from fsm_llm.logging import logger

# DECISION plan-2026-11-24T193347-3f4530d3/D-005
# The seam lives in `core`, not in `tools` or `runtime`: `tools` may not import `runtime` (the
# layer contract `l3-layers-import-no-llm` turns BROKEN, measured), and a callback argument
# threaded through every pattern would change every caller. Do NOT import `fsm_llm.runtime` (or
# `fsm_llm.llm`) here or in `tools/`, do NOT let `begin_tool` swallow an observer error (a call
# that cannot be logged must not run: the checkpoint is written before the body), do NOT let
# `end_tool` raise (it runs after the tool's own outcome is known and must not hide it), and do
# NOT hand the observer the live argument dict (an observer is evidence, never a way to change
# what a tool receives). See decisions.md D-005.


class ToolObserver(Protocol):
    """What ``ToolRegistry.execute`` calls around one tool body.

    ``begin_tool`` returns a token of its own (any object, or ``None``) that comes back to the
    ``end_tool`` of the same call.
    """

    def begin_tool(
        self,
        *,
        name: str,
        arguments: Mapping[str, Any],
        side_effecting: bool,
        gated: bool,
    ) -> object | None: ...

    def end_tool(
        self,
        token: object | None,
        *,
        status: str,
        error_type: str | None,
        error_code: str | None,
        truncated: bool,
        elapsed_ms: float,
    ) -> None: ...

    def begin_approval(
        self, *, tool_name: str, arguments: Mapping[str, Any], has_callback: bool
    ) -> object | None: ...

    def end_approval(
        self,
        token: object | None,
        *,
        approved: bool,
        reason: str,
        error_type: str | None,
    ) -> None: ...


class ToolHandle(NamedTuple):
    """The observer of one tool call and the token its ``begin_tool`` returned."""

    observer: ToolObserver
    token: object | None


_ACTIVE: ContextVar[ToolObserver | None] = ContextVar(
    "fsm_llm_tool_observer", default=None
)


def install_tool_observer(observer: ToolObserver | None) -> Token[ToolObserver | None]:
    """Make ``observer`` the active one in this context; ``None`` means no observer."""
    return _ACTIVE.set(observer)


def reset_tool_observer(token: Token[ToolObserver | None]) -> None:
    """Undo one ``install_tool_observer``, in the context that made it."""
    _ACTIVE.reset(token)


def begin_tool(
    name: str,
    arguments: Mapping[str, Any],
    *,
    side_effecting: bool,
    gated: bool,
) -> ToolHandle | None:
    """Tell the active observer a tool body is about to run; propagate its refusal."""
    observer = _ACTIVE.get()
    if observer is None:
        return None
    token = observer.begin_tool(
        name=name,
        arguments=MappingProxyType(arguments),
        side_effecting=side_effecting,
        gated=gated,
    )
    return ToolHandle(observer, token)


def end_tool(
    handle: ToolHandle | None,
    *,
    status: str,
    error_type: str | None,
    error_code: str | None,
    truncated: bool,
    elapsed_ms: float,
) -> None:
    """Hand the outcome of the tool call to the observer; never raises."""
    if handle is None:
        return
    try:
        handle.observer.end_tool(
            handle.token,
            status=status,
            error_type=error_type,
            error_code=error_code,
            truncated=truncated,
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:
        logger.warning(f"Tool observer end_tool failed: {type(e).__name__}")


# DECISION plan-2026-11-24T193347-3f4530d3/D-007
# Unlike `begin_tool`, these swallow every observer error: an approval event is evidence, never a
# gate. Do NOT let them raise, return what a caller could branch on, or pass live arguments.
def begin_approval(
    tool_name: str, arguments: Mapping[str, Any], *, has_callback: bool
) -> ToolHandle | None:
    """Tell the active observer an approval is about to be asked; never raises."""
    observer = _ACTIVE.get()
    if observer is None:
        return None
    try:
        token = observer.begin_approval(
            tool_name=tool_name,
            arguments=MappingProxyType(arguments),
            has_callback=has_callback,
        )
    except Exception as e:
        logger.warning(f"Tool observer begin_approval failed: {type(e).__name__}")
        return None
    return ToolHandle(observer, token)


def end_approval(
    handle: ToolHandle | None,
    *,
    approved: bool,
    reason: str,
    error_type: str | None = None,
) -> None:
    """Hand the verdict of the ask to the observer; never raises."""
    if handle is None:
        return
    try:
        handle.observer.end_approval(
            handle.token, approved=approved, reason=reason, error_type=error_type
        )
    except Exception as e:
        logger.warning(f"Tool observer end_approval failed: {type(e).__name__}")
