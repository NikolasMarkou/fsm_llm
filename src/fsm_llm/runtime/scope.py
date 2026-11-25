"""The turn scope: how the engine reaches the event log without carrying it in signatures.

A ``TurnScope`` names the log, the ROOT conversation id and the stack frame of what is running.
It lives in a ``ContextVar`` (never process-global, D-002 of plan 21e6f00d) and is bound around
a turn by ``bind`` (a new scope) or ``activate`` (an existing one). Binding it also installs the
scope's request observer (``llm.observer``), which is what logs the exact provider request of a
``LiteLLMInterface``. Everything here is a no-op when no scope is bound, so a conversation
without a log pays one ``ContextVar`` read per call.

Interface contract (callers: ``engine`` in steps 7 and 8, tests):
    - ``TurnScope(log, root_id, frame=0)``: frozen; ``.log``, ``.root_id``, ``.frame``,
      ``.observer`` (the ``RequestObserver`` that ``activate`` installs).
    - ``current_scope() -> TurnScope | None``.
    - ``bind(log, root_id, frame=0)`` and ``activate(scope)``: context managers that bind the
      scope and its observer, yield the scope and reset both on exit (error included). Reset
      in the context that bound; a thread sees the scope only when started through
      ``contextvars.copy_context().run`` (handler and tool-timeout threads are).
    - ``emit(type, **fields) -> int | None``: append one event under the scope's root id and
      frame and return its ``seq``; ``None`` and no I/O without a scope. A failing append
      RAISES (``EventLogError``): a log that drops events silently would lie, and the poisoned
      writer refuses the next checkpoint anyway. ``frame`` is not a field (the scope owns it).
    - ``scoped_iter(scope, iterator)``: a generator that binds *scope* around each ``next()``
      of *iterator* and around its ``close()``, and yields OUTSIDE the binding, so a stream
      consumed on another thread or context never resets a token it did not set.
    - ``llm_call`` and ``LlmCall`` (one engine model call) live in ``runtime.calls``; the
      request observer here logs the exact provider request of a ``LiteLLMInterface``.
    - Event fields: ``llm/request`` ``source`` (``"provider"`` or ``"engine"``), ``call_type``,
      ``purpose`` (a ``Purpose`` value, also on ``llm/response`` and ``provider/error``: the
      ``llm.purpose`` scope, else the one the call type implies), ``model``, ``stream``, ``prompt_ref`` (a blob: the allow-listed provider input, or
      ``{source, request_type, request}``); ``llm/response`` ``request_seq``, ``source``,
      ``call_type``, ``response_ref``, ``finish_reason``, or ``unrecorded: true`` when the reply
      cannot be written (the call still succeeds), plus ``stream`` and ``abandoned`` for a
      stream; a provider reply that carried token usage also holds the integers
      ``prompt_tokens``, ``completion_tokens`` and ``total_tokens`` (a provider reply also holds its ``model``; row 1.13:
      ``runtime.usage`` folds them, no blob is read; an engine-logged reply and a stream carry
      none, which the fold counts as ``usage_missing`` as the meter does);
      ``provider/error`` ``request_seq``, ``source``, ``call_type``, ``error_type`` (a
      class name), a ``failure_code`` (an ``LlmFailureCode`` value) and an integer
      ``status_code`` when the error has one, never message text,
      plus ``stream: true`` when the call was a stream.

At level ``shape`` (``EventLog(privacy="shape")``, ``runtime.privacy``) a request and a reply
build and store no blob: ``llm/request`` and ``llm/response`` carry no ``prompt_ref`` and no
``response_ref`` (the log would drop them anyway), only the codes, flags and integers above;
at ``values`` with a redaction hook the blobs are the hook's output (``prompt`` and
``response`` kinds).

Imports ``llm`` (the observer seam), ``core`` and ``runtime`` modules only, never ``engine``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Any, NamedTuple

from fsm_llm.aio import run_sync
from fsm_llm.core.redaction import redact_context
from fsm_llm.core.tool_observer import install_tool_observer, reset_tool_observer
from fsm_llm.llm.failures import classify_failure
from fsm_llm.llm.observer import install_observer, reset_observer
from fsm_llm.llm.parsing import _finish_reason
from fsm_llm.llm.purpose import resolve_purpose
from fsm_llm.llm.usage import usage_counts
from fsm_llm.logging import logger
from fsm_llm.runtime.constants import MAX_PAYLOAD_DEPTH, USAGE_TOKEN_FIELDS
from fsm_llm.runtime.events import EventType, LogWriteError, to_jsonable
from fsm_llm.runtime.log import EventLog, warn_on_failure
from fsm_llm.runtime.privacy import RedactionKind

SCOPE: ContextVar[TurnScope | None] = ContextVar("fsm_llm_turn_scope", default=None)

SOURCE_PROVIDER = "provider"
SOURCE_ENGINE = "engine"


class _Pending(NamedTuple):
    """What the observer remembers between ``begin_send`` and ``end_send`` or ``fail_send``."""

    request_seq: int
    call_type: str
    stream: bool
    model: str | None = None
    purpose: str = ""
    tools_sent: bool = False


@dataclass(frozen=True)
class TurnScope:
    """The log, root conversation id and frame of what is running (see the module docstring)."""

    log: EventLog
    root_id: str
    frame: int = 0
    observer: _ScopeObserver = field(init=False, repr=False, compare=False)
    tool_observer: _ToolObserver = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observer", _ScopeObserver(self))
        object.__setattr__(self, "tool_observer", _ToolObserver(self))


def current_scope() -> TurnScope | None:
    """Return the scope bound in this context, or ``None``."""
    return SCOPE.get()


@contextmanager
def activate(scope: TurnScope) -> Iterator[TurnScope]:
    """Bind *scope* and its request observer for the ``with`` body, then reset both."""
    scope_token = SCOPE.set(scope)
    observer_token = install_observer(scope.observer)
    tool_token = install_tool_observer(scope.tool_observer)
    try:
        yield scope
    finally:
        reset_tool_observer(tool_token)
        reset_observer(observer_token)
        SCOPE.reset(scope_token)


def bind(
    log: EventLog, root_id: str, frame: int = 0
) -> AbstractContextManager[TurnScope]:
    """Return the context manager of ``activate(TurnScope(log, root_id, frame))``."""
    return activate(TurnScope(log, root_id, frame))


def emit(type: EventType | str, **fields: Any) -> int | None:
    """Append one event under the current scope and return its ``seq`` (``None`` without one)."""
    scope = SCOPE.get()
    if scope is None:
        return None
    return scope.log.append(scope.root_id, type, frame=scope.frame, **fields)


def scoped_iter(scope: TurnScope, iterator: Iterable[Any]) -> Iterator[Any]:
    """Yield what *iterator* yields, with *scope* bound only while *iterator* runs."""
    inner = iter(iterator)
    try:
        while True:
            with activate(scope):
                try:
                    item = next(inner)
                except StopIteration:
                    return
            yield item
    finally:
        close = getattr(inner, "close", None)
        if close is not None:
            with activate(scope):
                close()


def write_quietly(scope: TurnScope, kind: EventType, **fields: Any) -> None:
    """Append an after-the-call event; a failure is a WARNING (``warn_on_failure``)."""
    with warn_on_failure(kind):
        scope.log.append(scope.root_id, kind, frame=scope.frame, **fields)


def _dump(value: Any) -> Any:
    """The JSON-ready form of a reply or request, through ``to_jsonable``.

    ``model_dump(mode="json")`` first (the exact wire form); a value JSON mode refuses (a plain
    object a handler put in a context) is dumped in Python mode instead. Either way the result
    goes through ``to_jsonable``, so a leaf JSON cannot hold (an object, NaN, a huge int)
    becomes a typed marker, the request degrades the way the state blob does and the extraction
    it guards behaves as it does with the log off. ``to_jsonable`` cuts only below
    ``MAX_PAYLOAD_DEPTH + 1``, so a value that is really too deep still reaches ``put_json``
    deeper than its limit and is refused, never recorded cut. A dump that fails either way
    raises (``log_request`` turns that into the deliberate refusal).
    """
    dump = getattr(value, "model_dump", None)
    if not callable(dump):
        return to_jsonable(value, max_depth=MAX_PAYLOAD_DEPTH + 1)
    try:
        dumped = dump(mode="json")
    except Exception:
        dumped = dump()
    return to_jsonable(dumped, max_depth=MAX_PAYLOAD_DEPTH + 1)


def request_body(request: Any) -> Any:
    """The JSON form of an engine request with secret-looking entries redacted.

    A request object carries raw context mappings (``FieldExtractionRequest.context``) beside
    its prompt text; the prompt builders keep secret-looking entries out of the text, but not
    out of those mappings, so the dump goes through ``redact_context`` (names and values judged
    by ``is_forbidden_context_entry``) before it becomes a blob.
    """
    body = _dump(request)
    return redact_context(body) if isinstance(body, dict) else body


def response_fields(
    log: EventLog, response: Any, *, wrap: str | None
) -> dict[str, Any]:
    """The ``response_ref`` and ``finish_reason`` fields of a reply, or ``unrecorded: true``.

    *wrap* names the reply type for an engine reply (stored as ``{source, response_type,
    response}``); ``None`` stores the provider reply as its own JSON.
    """
    fields: dict[str, Any] = {}
    if log.stores_values:
        try:
            body = _dump(response)
            if wrap is not None:
                body = {
                    "source": SOURCE_ENGINE,
                    "response_type": wrap,
                    "response": body,
                }
            fields["response_ref"] = log.put_json(body, kind=RedactionKind.RESPONSE)
        except Exception as exc:
            logger.warning(f"event log: reply not recorded ({type(exc).__name__})")
            return {"unrecorded": True}
    reason = (
        _finish_reason(response)
        if wrap is None
        else getattr(response, "finish_reason", None)
    )
    if isinstance(reason, str) and reason:
        fields["finish_reason"] = reason
    return fields


def _usage_fields(response: Any, model: str | None) -> dict[str, Any]:
    """The inline token integers of a provider reply (none without usage) and its model.

    Integers only (from ``usage_counts``, the meter's own reader): the fold of
    ``runtime.usage`` never reads a blob and a ``shape`` log can hold them.
    """
    counts = usage_counts(response)
    fields: dict[str, Any] = (
        {} if counts is None else dict(zip(USAGE_TOKEN_FIELDS, counts, strict=True))
    )
    if isinstance(model, str):
        fields["model"] = model
    return fields


def error_fields(error: BaseException, *, tools_sent: bool) -> dict[str, Any]:
    """``error_type`` (a class name), ``failure_code`` and an integer ``status_code``; never text."""
    fields: dict[str, Any] = {
        "error_type": type(error).__name__,
        "failure_code": classify_failure(error, tools_sent=tools_sent).value,
    }
    status = getattr(error, "status_code", None)
    if type(status) is int:
        fields["status_code"] = status
    return fields


def log_request(scope: TurnScope, build: Callable[[], Any], **fields: Any) -> int:
    """Build the request blob, write it and the ``llm/request`` checkpoint; fail closed.

    EVERY failure (a request that cannot be dumped, a blob or an append that cannot be
    written) is one ``LogWriteError`` and the model call is not made: a refusal chosen on
    purpose, never a raw serialization error for the call site to swallow.
    """
    try:
        if scope.log.stores_values:  # a shape log builds no request blob at all
            fields["prompt_ref"] = scope.log.put_json(
                build(), kind=RedactionKind.PROMPT
            )
        return scope.log.append(
            scope.root_id, EventType.LLM_REQUEST, frame=scope.frame, **fields
        )
    except Exception as exc:
        raise LogWriteError(
            "event log: the request could not be recorded, so the model call was not made "
            f"({type(exc).__name__})"
        ) from exc


def prebuilt(scope: TurnScope, build: Callable[[], Any]) -> Callable[[], Any]:
    """Evaluate *build* NOW (on the loop, it reads live objects) and return a callable giving its
    result; a failure is re-raised by that callable, so ``log_request`` makes it its one
    ``LogWriteError``. A ``shape`` log builds nothing."""
    if not scope.log.stores_values:
        return build
    try:
        body = build()
    except Exception as exc:

        def failed(error: Exception = exc) -> Any:
            raise error

        return failed
    return lambda: body


async def alog_request(
    scope: TurnScope,
    build: Callable[[], Any],
    *,
    tools_sent: bool = False,
    **fields: Any,
) -> int:
    """The async twin of ``log_request``: ONE lane job. *fields* hold ``source``, ``call_type``,
    ``purpose`` (and ``stream``): a cancelled await queues the request's ``provider/error``."""
    seen: list[int] = []
    body = prebuilt(scope, build)

    def job() -> int:
        seq = log_request(scope, body, **fields)
        seen.append(seq)
        return seq

    try:
        return await scope.log.lane_call(scope.root_id, job)
    except asyncio.CancelledError as exc:
        # DECISION plan-2026-11-24T193347-3f4530d3/D-015
        # The job still runs (an orphan). Its `provider/error` is queued on the SAME lane, so
        # FIFO runs it after the request job and `seen` is final: a request is never left
        # without its error, and nothing is awaited in the cancelled task. Do NOT shield or
        # await a begin hop here, write the error before the job (no `request_seq`), or narrow
        # the arm to `Exception` (D-008 of plan 51fde95b). See decisions.md D-015.
        scope.log.lane_after(
            scope.root_id,
            partial(_failed_request, scope, seen, exc, tools_sent, fields),
        )
        raise


def _failed_request(
    scope: TurnScope,
    seen: list[int],
    error: BaseException,
    tools_sent: bool,
    fields: Mapping[str, Any],
) -> None:
    """The ``provider/error`` of a request whose await was cancelled."""
    if seen:  # a job that failed wrote no request, so there is nothing to pair
        write_quietly(
            scope,
            EventType.PROVIDER_ERROR,
            request_seq=seen[0],
            source=fields["source"],
            call_type=fields["call_type"],
            purpose=fields["purpose"],
            **error_fields(error, tools_sent=tools_sent),
            **({"stream": True} if fields.get("stream") else {}),
        )


class _ScopeObserver:
    """The ``RequestObserver`` of one scope: logs the exact provider request of ``_send``."""

    def __init__(self, scope: TurnScope) -> None:
        self._scope = scope
        #: ``seq`` of the latest ``llm/request`` this observer wrote (a stream's reply links to it).
        self.last_request_seq: int | None = None

    def begin_send(self, call: dict[str, Any], *, call_type: str) -> object | None:
        return run_sync(self.abegin_send(call, call_type=call_type))

    async def abegin_send(
        self, call: dict[str, Any], *, call_type: str
    ) -> object | None:
        """The one body: the request is ONE lane job (``alog_request``), inline in sync mode."""
        purpose = resolve_purpose(call_type).value
        stream, tools_sent = bool(call.get("stream")), call.get("tools") is not None
        model = call.get("model")
        seq = await alog_request(
            self._scope,
            lambda: call,
            tools_sent=tools_sent,
            source=SOURCE_PROVIDER,
            call_type=call_type,
            purpose=purpose,
            model=model,
            stream=stream,
        )
        self.last_request_seq = seq
        model = model if isinstance(model, str) else None
        return _Pending(seq, call_type, stream, model, purpose, tools_sent)

    async def aend_send(self, token: object | None, response: Any) -> None:
        """The async twin of ``end_send``: the reply blob and line are ONE lane job."""
        if isinstance(token, _Pending) and not token.stream:
            await self._scope.log.lane_call(
                self._scope.root_id, self.end_send, token, response
            )

    def end_send(self, token: object | None, response: Any) -> None:
        # A stream is never read here: its one ``llm/response`` is ``LlmCall.stream_done``.
        if not isinstance(token, _Pending) or token.stream:
            return
        scope = self._scope
        write_quietly(
            scope,
            EventType.LLM_RESPONSE,
            request_seq=token.request_seq,
            source=SOURCE_PROVIDER,
            call_type=token.call_type,
            purpose=token.purpose,
            **response_fields(scope.log, response, wrap=None),
            **_usage_fields(response, token.model),
        )

    def fail_send(self, token: object | None, error: BaseException) -> None:
        # A stream fails while it is read, outside `_send`: `llm_call` writes its one
        # `provider/error` (as `end_send` leaves its one reply to `stream_done`).
        if not isinstance(token, _Pending) or token.stream:
            return
        write_quietly(
            self._scope,
            EventType.PROVIDER_ERROR,
            request_seq=token.request_seq,
            source=SOURCE_PROVIDER,
            call_type=token.call_type,
            purpose=token.purpose,
            **error_fields(error, tools_sent=token.tools_sent),
        )


class _OpenTool(NamedTuple):
    """What the tool observer remembers between ``begin_tool`` and ``end_tool``."""

    call_id: str
    name: str


class _ToolObserver:
    """The ``ToolObserver`` of one scope (``core.tool_observer``): ``tool/call`` and ``tool/result``.

    ``begin_tool`` writes ``tool/call`` before the body (fsynced when side-effecting) and
    RAISES when it cannot (the registry then runs no body); ``end_tool`` writes ``tool/result``
    and a failure is a WARNING, so the call stays ``unknown``. It never changes the arguments.
    """

    def __init__(self, scope: TurnScope) -> None:
        self._scope = scope

    def begin_tool(
        self,
        *,
        name: str,
        arguments: Mapping[str, Any],
        side_effecting: bool,
        gated: bool,
    ) -> object | None:
        call_id = uuid.uuid4().hex
        scope = self._scope
        scope.log.tool_begin(
            scope.root_id,
            name=name,
            call_id=call_id,
            arguments=arguments,
            side_effecting=side_effecting,
            gated=gated,
            frame=scope.frame,
        )
        return _OpenTool(call_id, name)

    def end_tool(
        self,
        token: object | None,
        *,
        status: str,
        error_type: str | None,
        error_code: str | None,
        truncated: bool,
        elapsed_ms: float,
    ) -> None:
        if not isinstance(token, _OpenTool):
            return
        scope = self._scope
        with warn_on_failure(EventType.TOOL_RESULT):
            scope.log.tool_end(
                scope.root_id,
                name=token.name,
                call_id=token.call_id,
                status=status,
                frame=scope.frame,
                error_type=error_type,
                error_code=error_code,
                truncated=truncated,
                elapsed_ms=elapsed_ms,
            )

    def begin_approval(
        self, *, tool_name: str, arguments: Mapping[str, Any], has_callback: bool
    ) -> object | None:
        scope = self._scope
        return scope.log.approval_request(
            scope.root_id,
            tool_name=tool_name,
            arguments=arguments,
            has_callback=has_callback,
            frame=scope.frame,
        )

    def end_approval(
        self,
        token: object | None,
        *,
        approved: bool,
        reason: str,
        error_type: str | None,
    ) -> None:
        scope = self._scope
        extra: dict[str, Any] = {} if error_type is None else {"error_type": error_type}
        scope.log.append(
            scope.root_id,
            EventType.APPROVAL_OUTCOME,
            frame=scope.frame,
            request_seq=token,
            approved=approved,
            reason=reason,
            **extra,
        )
