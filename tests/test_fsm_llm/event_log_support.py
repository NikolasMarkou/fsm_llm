"""Helpers the event log test files share (not collected: no ``test_`` prefix).

Interface contract: ``_reply(text)`` and ``_chunk(text)`` build a provider reply and a stream
chunk; ``_Wrapper(inner=None)`` is a custom interface around a real ``LiteLLMInterface`` (it
never reaches ``_send`` by itself); ``_Dead`` is a ``BaseException`` that stands for a vanished
process; ``child_env()`` is the one environment of every child Python process of these tests
(the scrubbed ``support._child_env`` plus ``PYTHONPATH`` when the run uses one);
``reopen_after_kill(open_)`` calls *open_*, retrying on ``LogLockedError`` only on Windows.
The ``log`` fixture lives in this folder's ``conftest.py``.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, TypeVar

from litellm.types.utils import Choices, Message, ModelResponse

from fsm_llm.llm import LiteLLMInterface, LLMInterface
from fsm_llm.runtime.events import LogLockedError
from tests.conftest import current_timing_factor
from tests.test_fsm_llm.support import _child_env

_T = TypeVar("_T")


class _Dead(BaseException):
    """The process is gone. A ``BaseException``, so no library ``except Exception`` eats it."""


def _reply(text: str = "Hello") -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=text), finish_reason="stop")]
    )


def _chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
    )


class _Wrapper(LLMInterface):
    """A custom interface around a real one: it never reaches ``_send`` by itself."""

    def __init__(self, inner: LiteLLMInterface | None = None) -> None:
        self.inner = inner or LiteLLMInterface(model="gpt-4o")

    def generate_response(self, request: Any) -> Any:
        return self.inner.generate_response(request)

    def generate_response_stream(self, request: Any) -> Iterator[str]:
        return self.inner.generate_response_stream(request)

    def extract_field(self, request: Any) -> Any:
        return self.inner.extract_field(request)

    def extract_bulk_data(self, request: Any) -> Any:
        return self.inner.extract_bulk_data(request)

    def complete(self, request: Any) -> Any:
        return self.inner.complete(request)


def child_env() -> dict[str, str]:
    """The environment of a child process: the scrubbed ``support._child_env`` (dead model
    port, litellm cost map flag, temp and home variables) and ``PYTHONPATH`` when set.
    ``PYTHONUTF8`` stays unset on purpose."""
    extra = (
        {"PYTHONPATH": os.environ["PYTHONPATH"]} if "PYTHONPATH" in os.environ else {}
    )
    return _child_env(**extra)


def reopen_after_kill(
    open_: Callable[[], _T], *, retry: bool = sys.platform == "win32"
) -> _T:
    """Return ``open_()`` after a ``Popen.kill()``, retrying on ``LogLockedError`` if *retry*.

    Windows frees the ``LockFileEx`` lock of a terminated process asynchronously, so the first
    open after a kill can see the log locked: with *retry* (the default on win32) the call is
    repeated until a deadline scaled by the timing factor. Elsewhere one plain call, so a lock
    that is really held still fails the test at once.
    """
    deadline = time.monotonic() + 60 * current_timing_factor()
    while True:
        try:
            return open_()
        except LogLockedError:
            if not retry or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
