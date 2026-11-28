"""
The skeleton shared by ``fsm-llm-okf`` and ``fsm-llm-skills`` (the ``fsm-llm``
runner uses only the unencodable-output escape): exit-code mapping,
BrokenPipe parking, the unencodable-output escape, the ``--max-seconds`` argument
and the ``validate`` report. Each CLI keeps its parser, commands, loader and
program name. It also holds the ``--env-file`` flag of the CLIs that need a model
(``fsm-llm``, ``fsm-llm-eval``, ``fsm-llm-meta``, ``python -m fsm_llm.reasoning``,
``fsm-llm-monitor``); the okf, skills, validate and visualize CLIs call no model
and never get it. Interface contract: ``run_cli`` returns 0, 1 or 130 and lets a
``BrokenPipeError`` (on Windows also the ``EINVAL`` ``OSError`` of a closed pipe) and a
usage ``SystemExit`` through; ``exit_main`` never returns; ``report_issues`` prints and returns 0 or 1. Imports stdlib,
``constants``, ``env.load_env`` and ``text.capped_line`` / ``text.strip_unsafe_text``
only (a kernel module, held by an import contract); not imported by ``fsm_llm/__init__.py``.
"""

from __future__ import annotations

import argparse
import errno
import math
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from typing import Any, NoReturn

from .constants import CLI_EXIT_FAILURE, CLI_EXIT_INTERRUPTED, CLI_EXIT_OK
from .env import load_env
from .text import capped_line, strip_unsafe_text

__all__ = [
    "add_env_file",
    "add_max_seconds",
    "apply_env_file",
    "escape_unencodable_output",
    "exit_main",
    "load_env_file",
    "positive_seconds",
    "report_error",
    "report_issues",
    "run_cli",
]

_DEADLINE_CODES = ("load-deadline", "check-deadline")  # both obey --max-seconds


def escape_unencodable_output() -> None:
    """Make stdout and stderr write unencodable text as ``\\uXXXX``, not raise.
    Process-level, never undone; a stream without ``reconfigure`` is left alone."""
    for stream in (sys.stdout, sys.stderr):
        with suppress(AttributeError, ValueError):  # io.UnsupportedOperation too
            stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]


def report_error(prog: str, message: str) -> None:
    """One stderr line ``prog: message``, control-stripped and capped."""
    print(f"{prog}: {capped_line(message, 1000)}", file=sys.stderr)


def positive_seconds(text: str) -> float:
    """argparse ``type`` of ``--max-seconds``: a finite number above zero."""
    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not 0 < seconds < math.inf:
        raise argparse.ArgumentTypeError("must be a finite number above 0")
    return seconds


def add_max_seconds(
    parser: argparse.ArgumentParser, default: float, *, metavar: str | None = None
) -> None:
    """Add ``--max-seconds`` (type ``positive_seconds``) to *parser*."""
    parser.add_argument(
        "--max-seconds",
        type=positive_seconds,
        default=default,
        metavar=metavar,
        help="wall-clock budget of the load (default %(default)s)",
    )


def add_env_file(parser: argparse.ArgumentParser) -> None:
    """Add ``--env-file PATH`` (dest ``env_file``, default ``None``) to *parser*.

    Interface contract: only a CLI that can build a model interface adds it.
    Pair it with :func:`apply_env_file`, called after parsing and before the
    model is resolved.
    """
    parser.add_argument(
        "--env-file",
        default=None,
        metavar="PATH",
        help="read this .env file into the environment first (a variable the "
        "process already has wins, a blank one counts as unset (to keep one "
        "empty on purpose, do not name it in the file); no .env file is "
        "read without this flag)",
    )


# DECISION plan-2026-11-27T201735-9283c1fd/D-008
# The one place an unreadable or non-UTF-8 ``--env-file`` becomes a one-line failure for every model CLI.
# Do NOT translate inside the public ``load_env`` (a host must see the original exception), do NOT let a CLI
# call ``load_env`` directly for ``--env-file`` (it prints a traceback), do NOT echo any byte of the file.
def load_env_file(path: str | os.PathLike[str]) -> None:
    """``load_env(path)`` with a one-line failure for every CLI.

    Interface contract: raises ``FileNotFoundError`` for a path that is not a
    file (the ``load_env`` text) and, for a file that cannot be read or decoded,
    one that names the path and the class of failure (``permission denied`` or
    ``not UTF-8 text``) and never a byte of the file; other errors pass through.
    """
    try:
        load_env(path)
    except (PermissionError, UnicodeDecodeError) as exc:
        why = (
            "permission denied"
            if isinstance(exc, PermissionError)
            else "not UTF-8 text"
        )
        raise FileNotFoundError(f"Cannot read .env file at {path}: {why}") from None


def apply_env_file(args: argparse.Namespace) -> None:
    """Read ``args.env_file`` (when given) with ``load_env_file``.

    Interface contract: a no-op when the flag is absent or ``None``. The file
    fills only variables the process has not set (an empty or whitespace-only
    one counts as not set, as for ``resolve_model``), so the model precedence
    is the ``--model`` flag, the process environment, the file, then
    ``ModelNotConfiguredError``. Raises ``FileNotFoundError`` for a path that is
    not a file or cannot be read or decoded (:func:`load_env_file`); every
    caller turns it into one stderr line and exit 1.
    """
    path = getattr(args, "env_file", None)
    if path is not None:
        load_env_file(path)


def report_issues(
    issues: Iterable[Any],
    *,
    label: str,
    count: int,
    strict: bool,
    where: Callable[[Any], str],
) -> int:
    """Print one stdout line per issue (``severity code where message``, plus a
    ``--max-seconds`` hint on a deadline code) and ``count label, N errors, ...``
    on stderr; return 1 on an error, or a warning when *strict*, else 0. Each
    issue needs ``severity``, ``code`` and ``message``; *where* names its place."""
    found = list(issues)
    for issue in found:
        hint = " (raise with --max-seconds)" if issue.code in _DEADLINE_CODES else ""
        line = f"{issue.severity} {issue.code} {where(issue)} {issue.message}{hint}"
        print(strip_unsafe_text(line))
    n = Counter(issue.severity for issue in found)
    summary = f"{count} {label}, {n['error']} errors, {n['warning']} warnings, "
    print(strip_unsafe_text(f"{summary}{n['info']} info"), file=sys.stderr)
    return CLI_EXIT_FAILURE if n["error"] or (strict and n["warning"]) else CLI_EXIT_OK


def run_cli(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    *,
    prog: str,
    errors: tuple[type[BaseException], ...],
) -> int:
    """Run the command bound as ``func``. Ctrl-C is 130; *errors* (the domain error)
    or an ``OSError`` is one stderr line and 1. Output is made safe before parsing,
    so an unencodable argument echoed by a usage error is no traceback."""
    escape_unencodable_output()
    args = parser.parse_args(argv)
    caught: tuple[type[BaseException], ...] = (*errors, OSError)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        report_error(prog, "interrupted")
        return CLI_EXIT_INTERRUPTED
    except caught as exc:
        if _is_broken_pipe(exc):
            raise  # handled by exit_main, which owns stdout
        report_error(prog, f"error: {exc}")
        return CLI_EXIT_FAILURE


def _is_broken_pipe(exc: BaseException) -> bool:
    """Whether *exc* says the reader of stdout is gone.

    Interface contract: ``True`` for a ``BrokenPipeError`` on every platform, and on
    Windows (``sys.platform == "win32"``, read at call time) also for an ``OSError``
    whose ``errno`` is ``EINVAL`` and which names no file (``filename`` is ``None``):
    that is what a write or flush to a closed pipe raises there. Windows maps a bad
    path (a ``<`` or ``|`` in a name) to ``EINVAL`` too, but that error names the file
    and is an ordinary error. Any other error, or ``EINVAL`` elsewhere, is ``False``.
    Raises nothing.
    """
    if isinstance(exc, BrokenPipeError):
        return True
    return (
        sys.platform == "win32"
        and isinstance(exc, OSError)
        and exc.errno == errno.EINVAL
        and exc.filename is None
    )


def exit_main(main: Callable[[], int]) -> NoReturn:
    """Console-script body: exit with ``main()``'s code."""
    try:
        code = main()
        sys.stdout.flush()
    except OSError as exc:
        if not _is_broken_pipe(exc):
            raise
        # Piped into `head`: park stdout so the exit flush prints no second error.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = CLI_EXIT_FAILURE
    sys.exit(code)
