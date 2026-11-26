"""``ToolEventsMixin``: the tool and approval emitters of ``EventLog`` (layer ``runtime``).

A pure move out of ``runtime/log.py`` (plan 77e5fef9, step 2): bodies unchanged, no anchor in
the block. ``warn_on_failure`` stays there (a test patches ``fsm_llm.runtime.log.logger``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from fsm_llm.core.redaction import redact_context
from fsm_llm.logging import logger
from fsm_llm.runtime.blobs import BlobStore
from fsm_llm.runtime.constants import MAX_INLINE_BYTES, MAX_PAYLOAD_DEPTH
from fsm_llm.runtime.events import (
    EventPayloadError,
    EventType,
    canonical_json,
    check_depth,
    to_jsonable,
)
from fsm_llm.runtime.privacy import RedactionKind


class ToolEventsMixin:
    """``tool_begin``, ``tool_end``, ``approval_request`` and ``tool_call`` of ``EventLog``."""

    if TYPE_CHECKING:  # provided by ``EventLog``, declared for mypy only
        _blobs: BlobStore

        @property
        def stores_values(self) -> bool: ...
        def append(
            self,
            conversation_id: str,
            type: EventType | str,
            *,
            frame: int = 0,
            checkpoint: bool | None = None,
            **fields: Any,
        ) -> int: ...
        def _redacted(self, kind: RedactionKind | str | None, value: Any) -> Any: ...

    def tool_begin(
        self,
        conversation_id: str,
        *,
        name: str,
        call_id: str,
        arguments: Mapping[str, Any],
        side_effecting: bool = True,
        gated: bool | None = None,
        frame: int = 0,
    ) -> None:
        """Append ``tool/call`` BEFORE a tool body runs (the one emitter of that event).

        Fields: ``name``, ``call_id``, ``side_effecting``, ``gated`` when given, and the
        *arguments* after ``redact_context``: inline when small, else ``arguments_ref`` to a
        blob. A side-effecting call is fsynced first (``is_checkpoint``), so if
        ``LogWriteError`` is raised the body must not run. With a redaction hook the arguments
        pass through it (kind ``arguments``) after ``redact_context``, and a failing hook raises
        before the event. At level ``shape`` the event carries ``argument_keys`` (the sorted
        names) and no value. *side_effecting* defaults to True: a tool is side-effecting unless
        it says it only reads.
        """
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(call_id, str)
            or not call_id
        ):
            raise EventPayloadError("a tool call needs a non-empty name and call_id")
        self._append_call(
            EventType.TOOL_CALL,
            conversation_id,
            frame,
            arguments,
            gated,
            name=name,
            call_id=call_id,
            side_effecting=bool(side_effecting),
        )

    def _append_call(
        self,
        kind: EventType,
        conversation_id: str,
        frame: int,
        arguments: Mapping[str, Any],
        gated: bool | None = None,
        **fields: Any,
    ) -> int:
        """Append *kind* with *fields* and the *arguments* as ``arguments``, ``arguments_ref`` or (at
        shape) ``argument_keys``, after ``redact_context`` and the user hook (kind ``arguments``)."""
        if not isinstance(arguments, Mapping):
            raise EventPayloadError("tool call arguments are a mapping")
        safe = to_jsonable(redact_context(dict(arguments)))
        where: dict[str, Any]
        if not self.stores_values:
            where = {"argument_keys": sorted(str(name) for name in arguments)}
        else:
            safe = self._redacted(RedactionKind.ARGUMENTS, safe)
            # inline only what the event field can hold whole: the envelope is one level and
            # `to_jsonable` cuts at MAX_PAYLOAD_DEPTH - 1, so a hook output of depth 31 or 32
            # (which a blob keeps) would be cut inline
            if (
                check_depth(safe, MAX_PAYLOAD_DEPTH - 2)
                and len(canonical_json(safe, what="tool arguments")) <= MAX_INLINE_BYTES
            ):
                where = {"arguments": safe}
            else:  # the hook already ran: store the blob as it is
                where = {"arguments_ref": self._blobs.put_json(safe)}
        if gated is not None:
            where["gated"] = bool(gated)
        return self.append(conversation_id, kind, frame=frame, **fields, **where)

    def tool_end(
        self,
        conversation_id: str,
        *,
        name: str,
        call_id: str,
        status: str,
        frame: int = 0,
        error_type: str | None = None,
        error_code: str | None = None,
        truncated: bool | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        """Append ``tool/result`` for a ``tool_begin`` (the one emitter of that event).

        ``status`` is ``ok``, ``failed`` or ``unknown``; ``error_type`` is a class name, never
        message text; ``error_code`` a ``ToolErrorCode`` value; a field left ``None`` is
        omitted. No result value is ever written.
        """
        extra: dict[str, Any] = {
            key: value
            for key, value in (
                ("error_type", error_type),
                ("error_code", error_code),
                ("truncated", truncated),
                ("elapsed_ms", elapsed_ms),
            )
            if value is not None
        }
        self.append(
            conversation_id,
            EventType.TOOL_RESULT,
            frame=frame,
            name=name,
            call_id=call_id,
            status=status,
            **extra,
        )

    def approval_request(
        self,
        conversation_id: str,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        has_callback: bool,
        frame: int = 0,
    ) -> int:
        """Append ``approval/request`` BEFORE the ask and return its ``seq``: ``tool_name``,
        ``has_callback`` and the arguments as ``tool/call`` has them; never the context or a grant."""
        return self._append_call(
            EventType.APPROVAL_REQUEST,
            conversation_id,
            frame,
            arguments,
            tool_name=tool_name,
            has_callback=bool(has_callback),
        )

    @contextmanager
    def tool_call(
        self,
        conversation_id: str,
        *,
        name: str,
        call_id: str,
        arguments: Mapping[str, Any],
        side_effecting: bool = True,
        frame: int = 0,
    ) -> Iterator[None]:
        """Checkpoint a tool call: ``with log.tool_call(...): run_the_tool()``.

        ``tool_begin`` BEFORE the body (see it), then on a normal exit ``tool/result`` with
        ``status: "ok"``; when the body raises an ``Exception`` it appends ``status: "failed"``
        with the exception's type name (never its message) and re-raises (a failing result
        append is logged, not raised, so the tool's own error is not masked). When the process
        dies inside, or the body is interrupted by a ``BaseException`` that is not an
        ``Exception`` (an interrupt, ``SystemExit``), no result is written: the call stays
        unresolved and a reader reports it as ``unknown``, because the log cannot tell whether
        the tool ran. Nothing retries it. ``ToolRegistry.execute`` does not use this manager:
        it reaches the log through ``core.tool_observer``.
        """
        self.tool_begin(
            conversation_id,
            name=name,
            call_id=call_id,
            arguments=arguments,
            side_effecting=side_effecting,
            frame=frame,
        )
        try:
            yield
        except Exception as exc:
            try:
                self.tool_end(
                    conversation_id,
                    name=name,
                    call_id=call_id,
                    status="failed",
                    frame=frame,
                    error_type=type(exc).__name__,
                )
            except Exception as log_exc:
                logger.warning(
                    f"tool result of {name!r} not logged ({type(log_exc).__name__}); "
                    "the call stays unknown"
                )
            raise
        self.tool_end(
            conversation_id, name=name, call_id=call_id, status="ok", frame=frame
        )
