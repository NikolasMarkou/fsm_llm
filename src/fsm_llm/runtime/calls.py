"""The engine-call recorder: ``llm_call`` and the ``LlmCall`` it yields (moved out of ``runtime.scope``).

Interface contract (callers: ``engine``, tests):
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

Imports ``llm`` (the observer seam), ``core`` and ``runtime`` modules only, never ``engine``.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager
from functools import partial
from typing import Any

from fsm_llm.aio import SYNC
from fsm_llm.llm.interface import interface_model
from fsm_llm.llm.observer import install_observer, is_observed, reset_observer
from fsm_llm.llm.purpose import resolve_purpose
from fsm_llm.runtime.events import EventType
from fsm_llm.runtime.scope import (
    SCOPE,
    SOURCE_ENGINE,
    SOURCE_PROVIDER,
    TurnScope,
    alog_request,
    error_fields,
    log_request,
    request_body,
    response_fields,
    write_quietly,
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
        "_replying",
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
        self._replying = False  # an ``aresponse`` job is queued or running

    def _link(self) -> int | None:
        if self._request_seq is not None or self._scope is None:
            return self._request_seq
        return self._scope.observer.last_request_seq

    def response(self, reply: Any) -> None:
        """Record the reply of a non-streamed call (once; a no-op when the observer did)."""
        if self._scope is None or not self._logs_reply or self._done:
            return
        self._done = True
        write_quietly(
            self._scope,
            EventType.LLM_RESPONSE,
            request_seq=self._request_seq,
            source=self._source,
            call_type=self._call_type,
            purpose=self._purpose,
            **response_fields(self._scope.log, reply, wrap=type(reply).__name__),
        )

    async def aresponse(self, reply: Any) -> None:
        """The async twin of ``response``: the reply blob and line are ONE lane job."""
        if self._scope is None or not self._logs_reply or self._done:
            return
        self._replying = (
            True  # stays set when the await is cancelled: the job still runs
        )
        await self._scope.log.lane_call(self._scope.root_id, self.response, reply)
        self._replying = False

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

    # DECISION plan-2026-11-26T213359-77e5fef9/D-007
    # [PARTLY SUPERSEDES clause 6 of D-017 of plan 3f4530d3: the joined reply is no longer
    # written on the loop.] In async mode the blob and the `llm/response` line are ONE job
    # queued behind the jobs already queued for the tree (`lane_after`, FIFO: after the
    # `llm/request`, before the `turn/end` the manager queues once the generator closes). Every
    # flag and the link are decided HERE, on the loop, before the job exists. Do NOT await a
    # lane call in this `finally` (a new cancel point in a generator whose cancel, `athrow`
    # and abandonment rules are pinned), read `_done`, `_exhausted` or the request link inside
    # the job (the loop moves on), split the blob from the line, skip the write on abandonment
    # or cancel (an `llm/request` without its end), run it inline outside `SYNC` mode, or add a
    # public name. A reader inside the turn may not see the line until the lane drains
    # (`release`, `close`, the next awaited write); a consumer cannot, `turn/end` is behind it.
    # See decisions.md D-007.
    def stream_done(self, text: str, *, abandoned: bool | None = None) -> None:
        """Record the joined reply of a streamed call (once), and whether the reader gave up.

        *abandoned* ``None`` (the default) is read from ``stream``: abandoned unless the
        chunks ran out. A stream that raised writes no ``llm/response`` here: its
        ``provider/error`` is the outcome (``llm_call`` writes it when the error escapes).
        The write runs inline in sync mode (and when no job ever started the lane of the tree)
        and as one queued lane job otherwise.
        """
        scope = self._scope
        if scope is None or self._done or self._errored:
            return
        self._done = True
        if abandoned is None:
            abandoned = not self._exhausted
        body = {"source": self._source, "response_type": "stream", "text": text}
        line = {
            "request_seq": self._link(),
            "source": self._source,
            "call_type": self._call_type,
            "purpose": self._purpose,
            "stream": True,
            "abandoned": bool(abandoned),
        }

        def write() -> None:
            fields = response_fields(scope.log, body, wrap=None)
            write_quietly(scope, EventType.LLM_RESPONSE, **line, **fields)

        if SYNC.get():
            write()
        else:
            scope.log.lane_after(
                scope.root_id,
                partial(contextvars.copy_context().run, write),
                ordered=True,
            )

    def _failed(self, error: BaseException) -> None:
        if self._scope is None or self._done:
            return
        self._done = True
        write_quietly(
            self._scope,
            EventType.PROVIDER_ERROR,
            request_seq=self._link(),
            source=self._source,
            call_type=self._call_type,
            purpose=self._purpose,
            **error_fields(error, tools_sent=self._tools_sent),
            **({"stream": True} if self._stream else {}),
        )


_NO_CALL = LlmCall(None, None, "", "", logs_reply=False)


@contextmanager
def llm_call(
    interface: Any, call_type: str, request: Any, *, stream: bool = False
) -> Iterator[LlmCall]:
    """Wrap one engine model call so the log holds its request and outcome (see the module docstring)."""
    scope = SCOPE.get()
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
    seq = log_request(
        scope,
        lambda: {
            "source": SOURCE_ENGINE,
            "request_type": type(request).__name__,
            "request": request_body(request),
        },
        source=SOURCE_ENGINE,
        call_type=call_type,
        purpose=purpose,
        model=interface_model(interface),
        stream=bool(stream),
    )
    with _recorded(scope, seq, call_type, purpose, tools_sent, stream) as call:
        yield call


@asynccontextmanager
async def allm_call(
    interface: Any, call_type: str, request: Any, *, stream: bool = False
) -> AsyncIterator[LlmCall]:
    """The async form of ``llm_call``: ``async with allm_call(...) as call``, then
    ``await call.aresponse(reply)``. The request of an engine-logged call is ONE lane job (the
    body is built on the loop first); an observed interface logs through the observer's async
    twins, so that case is ``llm_call``'s."""
    scope = SCOPE.get()
    if scope is None or is_observed(interface):
        with llm_call(interface, call_type, request, stream=stream) as call:
            yield call
        return
    purpose = resolve_purpose(call_type).value
    tools_sent = getattr(request, "tools", None) is not None
    seq = await alog_request(
        scope,
        lambda: {
            "source": SOURCE_ENGINE,
            "request_type": type(request).__name__,
            "request": request_body(request),
        },
        tools_sent=tools_sent,
        source=SOURCE_ENGINE,
        call_type=call_type,
        purpose=purpose,
        model=interface_model(interface),
        stream=bool(stream),
    )
    with _recorded(scope, seq, call_type, purpose, tools_sent, stream) as call:
        yield call


@contextmanager
def _recorded(
    scope: TurnScope,
    seq: int,
    call_type: str,
    purpose: str,
    tools_sent: bool,
    stream: bool,
) -> Iterator[LlmCall]:
    """Yield the ``LlmCall`` of an engine-logged request written as *seq*. A failure is written
    inline, except behind the reply job of a cancelled ``aresponse`` (on the tree's lane)."""
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
        if call._replying:
            scope.log.lane_after(scope.root_id, partial(call._failed, exc))
        else:
            call._failed(exc)
        raise
    finally:
        if token is not None:
            reset_observer(token)
