"""
The skeleton shared by ``fsm-llm-okf`` and ``fsm-llm-skills``: exit-code mapping,
BrokenPipe parking, the unencodable-output escape and the ``--max-seconds`` type.
Each CLI keeps its parser, commands, loader and program name. Interface contract:
``run_cli`` returns 0, 1 or 130 and lets a ``BrokenPipeError`` and a usage
``SystemExit`` through; ``exit_main`` never returns. Imports stdlib, ``constants``
and ``okf.capped_line`` only; not imported by ``fsm_llm/__init__.py``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import NoReturn

from .constants import CLI_EXIT_FAILURE, CLI_EXIT_INTERRUPTED
from .okf import capped_line

__all__ = [
    "escape_unencodable_output",
    "exit_main",
    "positive_seconds",
    "report_error",
    "run_cli",
]


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
    except BrokenPipeError:
        raise  # handled by exit_main, which owns stdout
    except caught as exc:
        report_error(prog, f"error: {exc}")
        return CLI_EXIT_FAILURE


def exit_main(main: Callable[[], int]) -> NoReturn:
    """Console-script body: exit with ``main()``'s code."""
    try:
        code = main()
        sys.stdout.flush()
    except BrokenPipeError:
        # Piped into `head`: park stdout so the exit flush prints no second error.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = CLI_EXIT_FAILURE
    sys.exit(code)
