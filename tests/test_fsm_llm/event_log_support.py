"""Helpers the event log test files share (not collected: no ``test_`` prefix).

Interface contract: ``_reply(text)`` and ``_chunk(text)`` build a provider reply and a stream
chunk; ``_Wrapper(inner=None)`` is a custom interface around a real ``LiteLLMInterface`` (it
never reaches ``_send`` by itself); ``_Dead`` is a ``BaseException`` that stands for a vanished
process; ``child_env()`` is the one environment of every child Python process of these tests
(the scrubbed ``support._child_env`` plus ``PYTHONPATH`` when the run uses one);
``reopen_after_kill(open_)`` calls *open_*, retrying on ``LogLockedError`` only on Windows;
``tree_files(root)`` lists the regular files under *root* in sorted order without the writer's
``*.lock`` files and ``tree_bytes(root)`` joins their bytes (the one way a test reads "every
byte the log wrote": Windows refuses a read of a byte-range-locked file, even in the same
process, so a test never reads a lock file).
``child_process(args, err, *, stdin=DEVNULL)`` is a context manager that starts ``python -m
<args>`` from the repository root with ``child_env()``, stderr in the file *err*, and kills and
reaps it on exit if it still runs; ``wait_ready(proc, ready, err)`` polls (a bounded number of
sleeps scaled by the timing factor, never a clock read) until the child has written *ready* and
fails the test with the tail of *err* when the child exits first.
The ``log`` fixture lives in this folder's ``conftest.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from fsm_llm.llm import LiteLLMInterface, LLMInterface
from fsm_llm.runtime.events import LogLockedError
from fsm_llm.runtime.stores.jsonl import LOCK_SUFFIX
from tests.conftest import current_timing_factor
from tests.test_fsm_llm.support import _child_env

_T = TypeVar("_T")

REPO = Path(__file__).resolve().parents[2]
CHILD_MODULE = "tests.test_fsm_llm.event_log_child"
READY_SECONDS = (
    300  # a deadline, scaled by the timing factor; the child imports fsm_llm
)
POLL_SECONDS = 0.05


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


def tree_files(root: str | os.PathLike[str]) -> list[Path]:
    """Return the regular files under *root*, sorted, except the writer's lock files.

    A ``LOCK_SUFFIX`` file carries a byte-range lock for as long as its writer is open, and on
    Windows a read of it fails with ``PermissionError`` even in the holding process.
    """
    return sorted(
        p
        for p in Path(root).rglob("*")
        if p.is_file() and not p.name.endswith(LOCK_SUFFIX)
    )


def tree_bytes(root: str | os.PathLike[str]) -> bytes:
    """Return the concatenated bytes of ``tree_files(root)`` (empty when there are none)."""
    return b"".join(p.read_bytes() for p in tree_files(root))


@contextmanager
def child_process(
    args: list[str], err: Path, *, stdin: int = subprocess.DEVNULL
) -> Iterator[subprocess.Popen[bytes]]:
    """Start ``python -m <args>`` and yield it; on exit kill and reap it if it still runs."""
    with err.open("wb") as errfile:
        proc = subprocess.Popen(
            [sys.executable, "-m", *args],
            cwd=REPO,
            env=child_env(),
            stdin=stdin,
            stdout=subprocess.DEVNULL,
            stderr=errfile,
        )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdin is not None:
            proc.stdin.close()


def wait_ready(proc: subprocess.Popen[bytes], ready: Path, err: Path) -> None:
    """Wait (a bounded number of polls) until *ready* exists; fail if the child exits first."""
    for _ in range(int(READY_SECONDS * current_timing_factor() / POLL_SECONDS)):
        if ready.exists():
            return
        if proc.poll() is not None:
            tail = err.read_text(encoding="utf-8", errors="replace")[-2000:]
            pytest.fail(
                f"the child exited {proc.returncode} before it was ready: {tail}"
            )
        time.sleep(POLL_SECONDS)
    pytest.fail("the child never reached its announced point")
