"""The purpose of a model call: a ContextVar scope the event log reads (roadmap row 1.15).

``Purpose`` (``fsm_llm.core.codes``) says why a call was made. It is written on ``llm/request``,
``llm/response`` and ``provider/error`` by ``runtime.scope`` and read nowhere else: it never
reaches a provider request.

Interface contract:
    - ``purpose_scope(purpose)``: a context manager that makes *purpose* the current one for
      the ``with`` body and resets it on exit (an error included). Wrap a non-stream call only,
      never a ``yield``: a generator that holds the scope open across a yield would reset a
      token in the wrong context.
    - ``current_purpose() -> Purpose | None``: the scope in this context, or ``None``.
    - ``purpose_of(call_type) -> Purpose``: the label a call type implies
      (``PURPOSE_BY_CALL_TYPE``, fallback ``COMPLETE``). It is also how an old log, written
      before the field existed, folds into usage by purpose.
    - ``resolve_purpose(call_type) -> Purpose``: the scope when one is set, else
      ``purpose_of(call_type)``, so no event is ever unlabeled. Never raises.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType

from fsm_llm.core.codes import Purpose

__all__ = [
    "PURPOSE_BY_CALL_TYPE",
    "current_purpose",
    "purpose_of",
    "purpose_scope",
    "resolve_purpose",
]

# DECISION plan-2026-11-24T193347-3f4530d3/D-011
# The purpose is a ContextVar read where the events are written (`runtime/scope.py`), never a
# `CompletionRequest` field (it would move `kind_equality_baseline.json`, the D5 request
# goldens and the API snapshot), never a keyword of `RequestObserver` (every fake observer
# would break) and never the `call_type` (it is behaviour-bearing: Ollama temperature 0, the
# forced `response_format`, the usage kind). Do NOT read it in a request builder or pass it to
# a provider, and do NOT hold the scope across a `yield`. See decisions.md D-011.
_PURPOSE: ContextVar[Purpose | None] = ContextVar("fsm_llm_purpose", default=None)

#: The purpose each engine call type implies when no scope is set. Every call type of
#: ``constants.USAGE_KIND_BY_CALL_TYPE`` is here (a test holds that), plus ``response_stream``.
PURPOSE_BY_CALL_TYPE: Mapping[str, Purpose] = MappingProxyType(
    {
        "data_extraction": Purpose.EXTRACT,
        "field_extraction": Purpose.EXTRACT,
        "classification": Purpose.CLASSIFY,
        "response_generation": Purpose.REPLY,
        "response_stream": Purpose.REPLY,
        "greeting": Purpose.REPLY,
        "completion": Purpose.COMPLETE,
    }
)


@contextmanager
def purpose_scope(purpose: Purpose) -> Iterator[Purpose]:
    """Make *purpose* the current purpose for the ``with`` body (see the module docstring)."""
    token = _PURPOSE.set(purpose)
    try:
        yield purpose
    finally:
        _PURPOSE.reset(token)


def current_purpose() -> Purpose | None:
    """The purpose scope bound in this context, or ``None``."""
    return _PURPOSE.get()


def purpose_of(call_type: str) -> Purpose:
    """The purpose a call type implies; ``COMPLETE`` for one the table does not list."""
    return PURPOSE_BY_CALL_TYPE.get(call_type, Purpose.COMPLETE)


def resolve_purpose(call_type: str) -> Purpose:
    """The scope when one is set, else the purpose the call type implies."""
    return _PURPOSE.get() or purpose_of(call_type)
