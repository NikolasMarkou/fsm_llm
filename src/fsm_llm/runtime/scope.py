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
    - ``llm_call(interface, call_type, request, *, stream=False)``: wraps one model call of the
      engine, ``with llm_call(llm, "completion", request) as call: reply = ...;
      call.response(reply)``. A stream (``stream=True``) reads its chunks through
      ``call.stream(chunks)`` (``call.astream(chunks)`` for an async generator, which also
      guards the call that opens the stream: it runs on the first step) and calls ``call.stream_done(joined_text)`` in its ``finally``
      (``abandoned`` is read from how the chunks ended, or given). An observed interface
      (``is_observed``) is logged by the observer, so the call only adds the joined reply of a
      stream (and a mid-stream error). Any other interface (a double, a mock, a custom class, a
      wrapper) is logged here: the ``llm/request`` checkpoint BEFORE the call, ``llm/response``
      after, ``provider/error`` when an ``Exception`` escapes or the call is cancelled
      (``asyncio.CancelledError``, raised again; D-008) (a stream that raised writes no
      ``llm/response``; a non-``Exception`` abort such as ``KeyboardInterrupt`` or ``SystemExit``
      mid-stream is no outage: it writes ``llm/response`` with ``abandoned`` and no
      ``provider/error``, and the turn ends ``interrupted``). The inner observer is switched
      off for the call (for a stream, per chunk), so a wrapper around a ``LiteLLMInterface`` is
      not logged twice. The request blob
      of an engine call is redacted (``redact_context``: secret-looking context entries of the
      request object become ``<redacted>``). A request that cannot be logged raises
      ``LogWriteError`` and the call is
      not made (fail closed, invariant 4); a failing ``llm/response`` or ``provider/error``
      write is a WARNING, never an error that hides the model's own outcome.
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
from collections.abc import AsyncGenerator, Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from fsm_llm.core.redaction import redact_context
from fsm_llm.core.tool_observer import install_tool_observer, reset_tool_observer
from fsm_llm.llm.failures import classify_failure
from fsm_llm.llm.interface import interface_model
from fsm_llm.llm.observer import install_observer, is_observed, reset_observer
from fsm_llm.llm.parsing import _finish_reason
from fsm_llm.llm.purpose import resolve_purpose
from fsm_llm.llm.usage import usage_counts
from fsm_llm.logging import logger
from fsm_llm.runtime.constants import MAX_PAYLOAD_DEPTH, USAGE_TOKEN_FIELDS
from fsm_llm.runtime.events import EventType, LogWriteError, to_jsonable
from fsm_llm.runtime.log import EventLog, warn_on_failure
from fsm_llm.runtime.privacy import RedactionKind

_SCOPE: ContextVar[TurnScope | None] = ContextVar("fsm_llm_turn_scope", default=None)

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
    return _SCOPE.get()


@contextmanager
def activate(scope: TurnScope) -> Iterator[TurnScope]:
    """Bind *scope* and its request observer for the ``with`` body, then reset both."""
    scope_token = _SCOPE.set(scope)
    observer_token = install_observer(scope.observer)
    tool_token = install_tool_observer(scope.tool_observer)
    try:
        yield scope
    finally:
        reset_tool_observer(tool_token)
        reset_observer(observer_token)
        _SCOPE.reset(scope_token)


def bind(
    log: EventLog, root_id: str, frame: int = 0
) -> AbstractContextManager[TurnScope]:
    """Return the context manager of ``activate(TurnScope(log, root_id, frame))``."""
    return activate(TurnScope(log, root_id, frame))


def emit(type: EventType | str, **fields: Any) -> int | None:
    """Append one event under the current scope and return its ``seq`` (``None`` without one)."""
    scope = _SCOPE.get()
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


def _write_quietly(scope: TurnScope, kind: EventType, **fields: Any) -> None:
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
    raises (``_log_request`` turns that into the deliberate refusal).
    """
    dump = getattr(value, "model_dump", None)
    if not callable(dump):
        return to_jsonable(value, max_depth=MAX_PAYLOAD_DEPTH + 1)
    try:
        dumped = dump(mode="json")
    except Exception:
        dumped = dump()
    return to_jsonable(dumped, max_depth=MAX_PAYLOAD_DEPTH + 1)


def _request_body(request: Any) -> Any:
    """The JSON form of an engine request with secret-looking entries redacted.

    A request object carries raw context mappings (``FieldExtractionRequest.context``) beside
    its prompt text; the prompt builders keep secret-looking entries out of the text, but not
    out of those mappings, so the dump goes through ``redact_context`` (names and values judged
    by ``is_forbidden_context_entry``) before it becomes a blob.
    """
    body = _dump(request)
    return redact_context(body) if isinstance(body, dict) else body


def _response_fields(
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


def _error_fields(error: BaseException, *, tools_sent: bool) -> dict[str, Any]:
    """``error_type`` (a class name), ``failure_code`` and an integer ``status_code``; never text."""
    fields: dict[str, Any] = {
        "error_type": type(error).__name__,
        "failure_code": classify_failure(error, tools_sent=tools_sent).value,
    }
    status = getattr(error, "status_code", None)
    if type(status) is int:
        fields["status_code"] = status
    return fields


def _log_request(scope: TurnScope, build: Callable[[], Any], **fields: Any) -> int:
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


class _ScopeObserver:
    """The ``RequestObserver`` of one scope: logs the exact provider request of ``_send``."""

    def __init__(self, scope: TurnScope) -> None:
        self._scope = scope
        #: ``seq`` of the latest ``llm/request`` this observer wrote (a stream's reply links to it).
        self.last_request_seq: int | None = None

    def begin_send(self, call: dict[str, Any], *, call_type: str) -> object | None:
        stream = bool(call.get("stream"))
        purpose = resolve_purpose(call_type).value
        seq = _log_request(
            self._scope,
            lambda: call,
            source=SOURCE_PROVIDER,
            call_type=call_type,
            purpose=purpose,
            model=call.get("model"),
            stream=stream,
        )
        self.last_request_seq = seq
        model = call.get("model")
        return _Pending(
            seq,
            call_type,
            stream,
            model if isinstance(model, str) else None,
            purpose,
            call.get("tools") is not None,
        )

    def end_send(self, token: object | None, response: Any) -> None:
        # A stream is never read here: its one ``llm/response`` is ``LlmCall.stream_done``.
        if not isinstance(token, _Pending) or token.stream:
            return
        scope = self._scope
        _write_quietly(
            scope,
            EventType.LLM_RESPONSE,
            request_seq=token.request_seq,
            source=SOURCE_PROVIDER,
            call_type=token.call_type,
            purpose=token.purpose,
            **_response_fields(scope.log, response, wrap=None),
            **_usage_fields(response, token.model),
        )

    def fail_send(self, token: object | None, error: BaseException) -> None:
        # A stream fails while it is read, outside `_send`: `llm_call` writes its one
        # `provider/error` (as `end_send` leaves its one reply to `stream_done`).
        if not isinstance(token, _Pending) or token.stream:
            return
        _write_quietly(
            self._scope,
            EventType.PROVIDER_ERROR,
            request_seq=token.request_seq,
            source=SOURCE_PROVIDER,
            call_type=token.call_type,
            purpose=token.purpose,
            **_error_fields(error, tools_sent=token.tools_sent),
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


class LlmCall:
    """The recorder ``llm_call`` yields (see the module docstring). An internal helper, not
    public API and not for construction."""

    __slots__ = (
        "_call_type",
        "_done",
        "_errored",
        "_exhausted",
        "_logs_reply",
        "_purpose",
        "_request_seq",
        "_scope",
        "_source",
        "_stream",
        "_tools_sent",
    )

    def __init__(
        self,
        scope: TurnScope | None,
        request_seq: int | None,
        call_type: str,
        source: str,
        *,
        logs_reply: bool,
        stream: bool = False,
        purpose: str = "",
        tools_sent: bool = False,
    ) -> None:
        self._purpose = purpose
        self._tools_sent = tools_sent
        self._scope = scope
        self._request_seq = request_seq
        self._call_type = call_type
        self._source = source
        self._logs_reply = logs_reply
        self._stream = stream
        self._done = False
        self._exhausted = False
        self._errored = False

    def _link(self) -> int | None:
        if self._request_seq is not None or self._scope is None:
            return self._request_seq
        return self._scope.observer.last_request_seq

    def response(self, reply: Any) -> None:
        """Record the reply of a non-streamed call (once; a no-op when the observer did)."""
        if self._scope is None or not self._logs_reply or self._done:
            return
        self._done = True
        _write_quietly(
            self._scope,
            EventType.LLM_RESPONSE,
            request_seq=self._request_seq,
            source=self._source,
            call_type=self._call_type,
            purpose=self._purpose,
            **_response_fields(self._scope.log, reply, wrap=type(reply).__name__),
        )

    def stream(self, chunks: Iterable[str]) -> Iterable[str]:
        """Yield the chunks of *chunks* and remember how the stream ended.

        An engine-logged call switches the inner request observer off only WHILE each chunk
        is being produced (never across a ``yield``), so a stream consumed on another thread
        or context never resets a token it did not set, and a wrapper around a
        ``LiteLLMInterface`` is still not logged twice. With no scope the stream is returned
        as it is.
        """
        if self._scope is None:
            return chunks
        return self._produce(chunks)

    def astream(self, chunks: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """The async form of ``stream``: the same bookkeeping over an async generator.

        An engine-logged call switches the inner request observer off only WHILE a chunk is
        awaited (a ``ContextVar`` set and reset inside one step, never across a ``yield``).
        With no scope *chunks* is returned as it is.
        """
        if self._scope is None:
            return chunks
        return self._aproduce(chunks)

    async def _aproduce(
        self, chunks: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        suppress = self._source == SOURCE_ENGINE
        try:
            while True:
                token = install_observer(None) if suppress else None
                try:
                    chunk = await chunks.__anext__()
                except StopAsyncIteration:
                    self._exhausted = True
                    return
                except Exception:
                    self._errored = True
                    raise
                finally:
                    if token is not None:
                        reset_observer(token)
                yield chunk
        finally:
            await chunks.aclose()

    def _produce(self, chunks: Iterable[str]) -> Iterator[str]:
        suppress = self._source == SOURCE_ENGINE
        inner = iter(chunks)
        try:
            while True:
                token = install_observer(None) if suppress else None
                try:
                    chunk = next(inner)
                except StopIteration:
                    self._exhausted = True
                    return
                except Exception:
                    self._errored = True
                    raise
                finally:
                    if token is not None:
                        reset_observer(token)
                yield chunk
        finally:
            # a reader that stops early closes the provider stream now, not when the last
            # reference to it goes (a wrapper may keep one), as `scoped_iter` does
            close = getattr(inner, "close", None)
            if close is not None:
                close()

    def stream_done(self, text: str, *, abandoned: bool | None = None) -> None:
        """Record the joined reply of a streamed call (once), and whether the reader gave up.

        *abandoned* ``None`` (the default) is read from ``stream``: abandoned unless the
        chunks ran out. A stream that raised writes no ``llm/response`` here: its
        ``provider/error`` is the outcome (``llm_call`` writes it when the error escapes).
        """
        if self._scope is None or self._done or self._errored:
            return
        self._done = True
        if abandoned is None:
            abandoned = not self._exhausted
        body = {"source": self._source, "response_type": "stream", "text": text}
        fields = _response_fields(self._scope.log, body, wrap=None)
        _write_quietly(
            self._scope,
            EventType.LLM_RESPONSE,
            request_seq=self._link(),
            source=self._source,
            call_type=self._call_type,
            purpose=self._purpose,
            stream=True,
            abandoned=bool(abandoned),
            **fields,
        )

    def _failed(self, error: BaseException) -> None:
        if self._scope is None or self._done:
            return
        self._done = True
        _write_quietly(
            self._scope,
            EventType.PROVIDER_ERROR,
            request_seq=self._link(),
            source=self._source,
            call_type=self._call_type,
            purpose=self._purpose,
            **_error_fields(error, tools_sent=self._tools_sent),
            **({"stream": True} if self._stream else {}),
        )


_NO_CALL = LlmCall(None, None, "", "", logs_reply=False)


@contextmanager
def llm_call(
    interface: Any, call_type: str, request: Any, *, stream: bool = False
) -> Iterator[LlmCall]:
    """Wrap one engine model call so the log holds its request and outcome (see the module docstring)."""
    scope = _SCOPE.get()
    if scope is None:
        yield _NO_CALL
        return
    purpose = resolve_purpose(call_type).value
    tools_sent = getattr(request, "tools", None) is not None
    if is_observed(interface):
        # an earlier call of this scope must not be the link of a stream that never sends
        scope.observer.last_request_seq = None
        observed = LlmCall(
            scope,
            None,
            call_type,
            SOURCE_PROVIDER,
            logs_reply=False,
            stream=stream,
            purpose=purpose,
            tools_sent=tools_sent,
        )
        try:
            yield observed
        except Exception as exc:
            # `_send` logs the error of a non-streamed call; a stream fails while it is read.
            if stream:
                observed._failed(exc)
            raise
        return
    seq = _log_request(
        scope,
        lambda: {
            "source": SOURCE_ENGINE,
            "request_type": type(request).__name__,
            "request": _request_body(request),
        },
        source=SOURCE_ENGINE,
        call_type=call_type,
        purpose=purpose,
        model=interface_model(interface),
        stream=bool(stream),
    )
    call = LlmCall(
        scope,
        seq,
        call_type,
        SOURCE_ENGINE,
        logs_reply=True,
        stream=bool(stream),
        purpose=purpose,
        tools_sent=tools_sent,
    )
    # An inner LiteLLMInterface (a wrapper) must not log the call a second time. A stream
    # is read after this block yields, so `LlmCall.stream` switches the observer off per chunk.
    token = None if stream else install_observer(None)
    try:
        yield call
    except (Exception, asyncio.CancelledError) as exc:
        # DECISION plan-2026-11-19T025425-51fde95b/D-008: the same rule as `_asend`. A cancelled
        # call was already requested (and may be billed), so the log gets its `provider/error`
        # and the cancel goes on unchanged. Do NOT narrow this to `Exception` (a dangling
        # `llm/request`). A stream whose pass already ran `stream_done` (its `finally`) stays
        # `llm/response` with `abandoned`: `_failed` writes once. See decisions.md D-046.
        call._failed(exc)
        raise
    finally:
        if token is not None:
            reset_observer(token)
