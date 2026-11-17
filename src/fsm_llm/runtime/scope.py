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
      call.response(reply)`` (a stream: ``call.stream_done(joined_text, abandoned=...)`` in its
      ``finally``). An observed interface (``is_observed``) is logged by the observer, so the
      call only adds the joined reply of a stream. Any other interface (a double, a mock, a
      custom class, a wrapper) is logged here: the ``llm/request`` checkpoint BEFORE the call,
      ``llm/response`` after, ``provider/error`` when an ``Exception`` escapes. The inner
      observer is switched off for the call, so a wrapper around a ``LiteLLMInterface`` is not
      logged twice. A request that cannot be logged raises ``LogWriteError`` and the call is
      not made (fail closed, invariant 4); a failing ``llm/response`` or ``provider/error``
      write is a WARNING, never an error that hides the model's own outcome.
    - Event fields: ``llm/request`` ``source`` (``"provider"`` or ``"engine"``), ``call_type``,
      ``model``, ``stream``, ``prompt_ref`` (a blob: the allow-listed provider input, or
      ``{source, request_type, request}``); ``llm/response`` ``request_seq``, ``source``,
      ``call_type``, ``response_ref``, ``finish_reason``, or ``unrecorded: true`` when the reply
      cannot be written (the call still succeeds), plus ``stream`` and ``abandoned`` for a
      stream; ``provider/error`` ``request_seq``, ``source``, ``call_type``, ``error_type`` (a
      class name) and an integer ``status_code`` when the error has one, never message text.

Imports ``llm`` (the observer seam), ``core`` and ``runtime`` modules only, never ``engine``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from fsm_llm.llm.interface import interface_model
from fsm_llm.llm.observer import install_observer, is_observed, reset_observer
from fsm_llm.llm.parsing import _finish_reason
from fsm_llm.logging import logger
from fsm_llm.runtime.events import EventType, LogWriteError, to_jsonable
from fsm_llm.runtime.log import EventLog

_SCOPE: ContextVar[TurnScope | None] = ContextVar("fsm_llm_turn_scope", default=None)

SOURCE_PROVIDER = "provider"
SOURCE_ENGINE = "engine"


class _Pending(NamedTuple):
    """What the observer remembers between ``begin_send`` and ``end_send`` or ``fail_send``."""

    request_seq: int
    call_type: str
    stream: bool


@dataclass(frozen=True)
class TurnScope:
    """The log, root conversation id and frame of what is running (see the module docstring)."""

    log: EventLog
    root_id: str
    frame: int = 0
    observer: _ScopeObserver = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observer", _ScopeObserver(self))


def current_scope() -> TurnScope | None:
    """Return the scope bound in this context, or ``None``."""
    return _SCOPE.get()


@contextmanager
def activate(scope: TurnScope) -> Iterator[TurnScope]:
    """Bind *scope* and its request observer for the ``with`` body, then reset both."""
    scope_token = _SCOPE.set(scope)
    observer_token = install_observer(scope.observer)
    try:
        yield scope
    finally:
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
    """Append an after-the-call event; a failure is a WARNING (type name only), not an error."""
    try:
        scope.log.append(scope.root_id, kind, frame=scope.frame, **fields)
    except Exception as exc:
        logger.warning(f"event log: {kind.value} not written ({type(exc).__name__})")


def _dump(value: Any) -> Any:
    """The JSON-ready form of a reply or request: ``model_dump(mode="json")`` or ``to_jsonable``."""
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else to_jsonable(value)


def _response_fields(
    log: EventLog, response: Any, *, wrap: str | None
) -> dict[str, Any]:
    """The ``response_ref`` and ``finish_reason`` fields of a reply, or ``unrecorded: true``.

    *wrap* names the reply type for an engine reply (stored as ``{source, response_type,
    response}``); ``None`` stores the provider reply as its own JSON.
    """
    try:
        body = _dump(response)
        if wrap is not None:
            body = {"source": SOURCE_ENGINE, "response_type": wrap, "response": body}
        fields: dict[str, Any] = {"response_ref": log.put_json(body)}
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


def _error_fields(error: BaseException) -> dict[str, Any]:
    """``error_type`` (a class name) and an integer ``status_code``; never message text."""
    fields: dict[str, Any] = {"error_type": type(error).__name__}
    status = getattr(error, "status_code", None)
    if type(status) is int:
        fields["status_code"] = status
    return fields


def _log_request(scope: TurnScope, request_blob: Any, **fields: Any) -> int:
    """Write the blob and the ``llm/request`` checkpoint; fail closed with ``LogWriteError``."""
    try:
        ref = scope.log.put_json(request_blob)
        return scope.log.append(
            scope.root_id,
            EventType.LLM_REQUEST,
            frame=scope.frame,
            prompt_ref=ref,
            **fields,
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
        seq = _log_request(
            self._scope,
            call,
            source=SOURCE_PROVIDER,
            call_type=call_type,
            model=call.get("model"),
            stream=stream,
        )
        self.last_request_seq = seq
        return _Pending(seq, call_type, stream)

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
            **_response_fields(scope.log, response, wrap=None),
        )

    def fail_send(self, token: object | None, error: BaseException) -> None:
        if not isinstance(token, _Pending):
            return
        _write_quietly(
            self._scope,
            EventType.PROVIDER_ERROR,
            request_seq=token.request_seq,
            source=SOURCE_PROVIDER,
            call_type=token.call_type,
            **_error_fields(error),
        )


class LlmCall:
    """The recorder ``llm_call`` yields (see the module docstring). Not for construction."""

    __slots__ = (
        "_call_type",
        "_done",
        "_logs_reply",
        "_request_seq",
        "_scope",
        "_source",
    )

    def __init__(
        self,
        scope: TurnScope | None,
        request_seq: int | None,
        call_type: str,
        source: str,
        *,
        logs_reply: bool,
    ) -> None:
        self._scope = scope
        self._request_seq = request_seq
        self._call_type = call_type
        self._source = source
        self._logs_reply = logs_reply
        self._done = False

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
            **_response_fields(self._scope.log, reply, wrap=type(reply).__name__),
        )

    def stream_done(self, text: str, *, abandoned: bool = False) -> None:
        """Record the joined reply of a streamed call (once), and whether the reader gave up."""
        if self._scope is None or self._done:
            return
        self._done = True
        body = {"source": self._source, "response_type": "stream", "text": text}
        try:
            fields: dict[str, Any] = {"response_ref": self._scope.log.put_json(body)}
        except Exception as exc:
            logger.warning(
                f"event log: stream reply not recorded ({type(exc).__name__})"
            )
            fields = {"unrecorded": True}
        _write_quietly(
            self._scope,
            EventType.LLM_RESPONSE,
            request_seq=self._link(),
            source=self._source,
            call_type=self._call_type,
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
            request_seq=self._request_seq,
            source=self._source,
            call_type=self._call_type,
            **_error_fields(error),
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
    if is_observed(interface):
        yield LlmCall(scope, None, call_type, SOURCE_PROVIDER, logs_reply=False)
        return
    seq = _log_request(
        scope,
        {
            "source": SOURCE_ENGINE,
            "request_type": type(request).__name__,
            "request": _dump(request),
        },
        source=SOURCE_ENGINE,
        call_type=call_type,
        model=interface_model(interface),
        stream=bool(stream),
    )
    call = LlmCall(scope, seq, call_type, SOURCE_ENGINE, logs_reply=True)
    # An inner LiteLLMInterface (a wrapper) must not log the call a second time.
    token = install_observer(None)
    try:
        yield call
    except Exception as exc:
        call._failed(exc)
        raise
    finally:
        reset_observer(token)
