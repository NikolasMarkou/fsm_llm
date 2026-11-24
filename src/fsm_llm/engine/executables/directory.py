"""One directory of executable documents loaded beside a set of built-ins (the ``--executables DIR`` flag).

Interface contract: ``load_directory_registry(directory, builtins, *, prog) -> ExecutableRegistry``.
*directory* must be an existing directory (``FileNotFoundError`` otherwise, one line for the CLI);
*builtins* are registered FIRST, so a file whose id collides with one is refused and never replaces
it. The file rules are the registry's (sorted ``*.json`` in any case of the suffix, a BOM accepted, a
symbolic link or a file over ``MAX_DOCUMENT_BYTES`` skipped). Each unreadable or invalid file and each
collision is ONE stderr line ``<prog>: warning: <label>: <why>``, capped and control-stripped, and the
load goes on. A file here loads as OPERATOR origin (the registry default), never as a model document:
nothing in it passes the V7 allow-list, so the directory is as trusted as the command line.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from fsm_llm.cli_support import report_error
from fsm_llm.text import capped_line

from .protocol import Executable
from .registry import ExecutableRegistry


def load_directory_registry(
    directory: Path, builtins: Iterable[Executable], *, prog: str
) -> ExecutableRegistry:
    """The registry of *builtins* plus the documents of *directory* (see the module docstring)."""
    if not directory.is_dir():
        raise FileNotFoundError(f"--executables {directory}: not an existing directory")
    registry = ExecutableRegistry([directory])
    for executable in builtins:
        registry.register(executable)
    report = registry.reload()
    for label, reasons in report.failed.items():
        for reason in reasons:
            report_error(prog, f"warning: {label}: {capped_line(reason, 300)}")
    for collision in report.collisions:
        report_error(prog, f"warning: {capped_line(collision, 300)}")
    return registry
