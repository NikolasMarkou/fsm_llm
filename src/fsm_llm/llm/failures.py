"""Why a model call failed: the closed ``LlmFailureCode`` of an exception (roadmap row 1.15).

Interface contract (callers: ``runtime.scope`` when it writes ``provider/error``, tests):
    - ``classify_failure(exc, *, tools_sent) -> LlmFailureCode``: the code of one exception.
      *tools_sent* says whether the failed request carried tools (a request without tools cannot
      garble a tool call). Total and never raises: whatever it cannot place is ``UNKNOWN``.
      It reads class NAMES of ``type(exc).__mro__``, an integer ``status_code`` and the
      ``__cause__`` chain, and the one existing marker detector for a garbled tool call; it
      never reads message text otherwise and imports no provider library.
"""

from __future__ import annotations

import asyncio

from fsm_llm.core.codes import LlmFailureCode
from fsm_llm.llm.interface import is_malformed_tool_call_error

__all__ = ["classify_failure"]

# DECISION plan-2026-11-24T193347-3f4530d3/D-010
# The ORDER is the mapping. litellm's `Timeout` is a subclass of `APIConnectionError` and reports
# status 408, `APIConnectionError` reports 500, and `ContextWindowExceededError` and
# `ContentPolicyViolationError` are `BadRequestError`s (400), so a status-first or a
# nearest-base-first mapper puts them in the wrong bucket. Ollama's garbled tool call is a plain
# `APIConnectionError` whose text is the only difference, so the marker check runs BEFORE the
# class names (and needs `tools_sent`). Class names, not `isinstance`: litellm may be imported in
# exactly one module. Do NOT test the status first, move the marker check after the class names,
# import a provider class here, or match message text beyond that one detector. A renamed
# provider class falls through to `http_error` or `unknown`; `test_failure_codes` builds every
# row from the REAL litellm class so the pin refresh catches it. See decisions.md D-010.
_BY_CLASS_NAME: tuple[tuple[LlmFailureCode, frozenset[str]], ...] = (
    (LlmFailureCode.CONTEXT_OVERFLOW, frozenset({"ContextWindowExceededError"})),
    (LlmFailureCode.CONTENT_REFUSED, frozenset({"ContentPolicyViolationError"})),
    (LlmFailureCode.RATE_LIMITED, frozenset({"RateLimitError"})),
    (LlmFailureCode.AUTH, frozenset({"AuthenticationError", "PermissionDeniedError"})),
    (
        LlmFailureCode.TIMEOUT,
        frozenset({"Timeout", "APITimeoutError", "TimeoutException", "TimeoutError"}),
    ),
    (
        LlmFailureCode.TRANSPORT,
        frozenset({"APIConnectionError", "ConnectionError", "OSError"}),
    ),
)

_MAX_CAUSE_DEPTH = 8


def _classify_one(exc: BaseException, tools_sent: bool) -> LlmFailureCode | None:
    """The code of *exc* alone (no cause), or ``None`` when it matches nothing."""
    if is_malformed_tool_call_error(exc, tools_sent=tools_sent):
        return LlmFailureCode.MALFORMED_RESPONSE
    names = {cls.__name__ for cls in type(exc).__mro__}
    for code, class_names in _BY_CLASS_NAME:
        if names & class_names:
            return code
    status = getattr(exc, "status_code", None)
    if type(status) is int and status >= 400:
        return LlmFailureCode.HTTP_ERROR
    return None


def classify_failure(exc: BaseException, *, tools_sent: bool) -> LlmFailureCode:
    """The ``LlmFailureCode`` of *exc* (see the module docstring)."""
    if isinstance(exc, asyncio.CancelledError):
        return LlmFailureCode.CANCELLED
    current: BaseException | None = exc
    for _ in range(_MAX_CAUSE_DEPTH):
        if current is None:
            break
        code = _classify_one(current, tools_sent)
        if code is not None:
            return code
        current = current.__cause__
    return LlmFailureCode.UNKNOWN
