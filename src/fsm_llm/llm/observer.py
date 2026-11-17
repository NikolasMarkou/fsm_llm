"""The request observer seam of the ``llm`` layer (plan 21e6f00d, step 4).

``LiteLLMInterface._send`` is the one provider call. It calls the three helpers below
around ``completion(**call_params)`` so a consumer above this layer (the event log of
``fsm_llm.runtime``) can see the exact provider request. This module knows no consumer:
the layer rule is that ``llm`` never imports ``runtime``; the consumer installs an object
that implements ``RequestObserver``.

Interface contract:
- ``install_observer(observer) -> Token`` makes ``observer`` the active one for the current
  context (a ``ContextVar``, never process-global); ``reset_observer(token)`` undoes it
  and must run in the context that installed it. ``install_observer(None)`` installs
  "no observer", which is how a caller suppresses the seam for one call. A thread sees
  the observer only when started through ``contextvars.copy_context().run``.
- ``begin_send(call_params, *, call_type)`` returns ``None`` without an observer (nothing
  is built), else a ``SendHandle``. The observer gets a NEW dict of the keys in
  ``OBSERVED_CALL_KEYS`` (the values are shared with the request: treat them read-only).
  An exception from ``RequestObserver.begin_send`` PROPAGATES: the call is not made.
- ``end_send(handle, response)`` and ``fail_send(handle, error)`` never raise: an observer
  error is logged as a WARNING (type name only) and the call goes on. A streamed call gets
  ``end_send`` with the iterator, never read.
- ``is_observed(interface)`` is true only for an interface whose class sets
  ``observes_requests = True`` (``LiteLLMInterface``); a double or a mock is not observed,
  so the engine logs its calls itself.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, NamedTuple, Protocol

from fsm_llm.logging import logger

# DECISION plan-2026-11-17T100529-21e6f00d/D-002: the observer gets an ALLOW-LIST of the
# provider request, never the whole ``call_params`` and never a deny-list. The dict carries
# ``api_key``, ``api_base``, extra headers, ``timeout``, ``max_retries`` and any caller
# kwarg, and a key added to the builder tomorrow must not reach a log by default. Do NOT
# replace the allow-list with "copy everything but the secrets". The observer is a
# ContextVar and a callback, not a counter: the usage meter stays fed in ``_send`` alone
# (plan 944e2692 D-004) and nothing here counts. ``begin_send`` runs BEFORE the provider
# call and fails closed; ``end_send`` and ``fail_send`` must never break the call. Do NOT
# attach an observer to an interface instance (the classifier and agents build their own).
# See decisions.md D-002.
OBSERVED_CALL_KEYS: tuple[str, ...] = (
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "stream",
    "response_format",
    "tools",
    "tool_choice",
    "reasoning_effort",
    "seed",
)


class RequestObserver(Protocol):
    """What ``LiteLLMInterface._send`` calls around one provider request.

    ``begin_send`` returns a token of its own (any object, or ``None``) that comes back
    to ``end_send`` or ``fail_send`` of the same request.
    """

    def begin_send(self, call: dict[str, Any], *, call_type: str) -> object | None: ...

    def end_send(self, token: object | None, response: Any) -> None: ...

    def fail_send(self, token: object | None, error: BaseException) -> None: ...


class SendHandle(NamedTuple):
    """The observer of one request and the token its ``begin_send`` returned."""

    observer: RequestObserver
    token: object | None


_ACTIVE: ContextVar[RequestObserver | None] = ContextVar(
    "fsm_llm_request_observer", default=None
)


def install_observer(observer: RequestObserver | None) -> Token[RequestObserver | None]:
    """Make ``observer`` the active one in this context; ``None`` means no observer."""
    return _ACTIVE.set(observer)


def reset_observer(token: Token[RequestObserver | None]) -> None:
    """Undo one ``install_observer``, in the context that made it."""
    _ACTIVE.reset(token)


def begin_send(call_params: dict[str, Any], *, call_type: str) -> SendHandle | None:
    """Tell the active observer a request is about to be sent; propagate its refusal."""
    observer = _ACTIVE.get()
    if observer is None:
        return None
    call = {key: call_params[key] for key in OBSERVED_CALL_KEYS if key in call_params}
    return SendHandle(observer, observer.begin_send(call, call_type=call_type))


def end_send(handle: SendHandle | None, response: Any) -> None:
    """Hand the provider reply (a stream iterator when streaming) to the observer."""
    if handle is None:
        return
    try:
        handle.observer.end_send(handle.token, response)
    except Exception as e:
        logger.warning(f"Request observer end_send failed: {type(e).__name__}")


def fail_send(handle: SendHandle | None, error: BaseException) -> None:
    """Hand a provider error to the observer; the caller re-raises the error itself."""
    if handle is None:
        return
    try:
        handle.observer.fail_send(handle.token, error)
    except Exception as e:
        logger.warning(f"Request observer fail_send failed: {type(e).__name__}")


def is_observed(interface: object) -> bool:
    """True when ``interface`` sends through ``_send`` and so reaches the observer."""
    return getattr(interface, "observes_requests", False) is True
