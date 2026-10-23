"""
Command-line tool for Agent Skills folders (``fsm-llm-skills``).

Usage::

    fsm-llm-skills validate PATH [PATH ...] [--strict] [--max-seconds S]
    fsm-llm-skills list PATH [PATH ...] [--max-seconds S]
    fsm-llm-skills show NAME PATH [PATH ...] [--file RELPATH] [--offset N]

Each PATH is a skills directory (child folders holding a ``SKILL.md``) or one
skill folder. Reports go to stdout, diagnostics to stderr; every string printed
passes ``okf.strip_unsafe_text``. Nothing in a skill is executed. Exit codes: 0
ok; 1 failure (an error issue under ``validate``, with ``--strict`` a warning;
a missing path; an unknown skill; a refused read); 2 usage; 130 Ctrl-C.

Imports core only (never a subpackage).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from typing import TextIO

from .agent_skills import SkillCatalog, SkillLimits, incomplete_note, load_skills
from .cli_support import exit_main, positive_seconds, report_error, run_cli
from .constants import (
    CLI_EXIT_FAILURE,
    CLI_EXIT_OK,
    SKILL_CATALOG_DESCRIPTION_CHARS,
    SKILL_LIST_MAX_ITEMS,
    SKILL_MAX_LOAD_SECONDS,
    SKILL_READ_PAGE_CHARS,
)
from .definitions import SkillError
from .okf import capped_line, strip_unsafe_text, text_page

__all__ = ["main", "main_cli"]

_PROG = "fsm-llm-skills"


def _say(text: str, stream: TextIO | None = None, *, multiline: bool = False) -> None:
    """Print *text* control-stripped; ``multiline`` keeps newlines and tabs."""
    print(strip_unsafe_text(text, multiline=multiline), file=stream or sys.stdout)


def _err(message: str) -> None:
    report_error(_PROG, message)


def _load(args: argparse.Namespace) -> SkillCatalog:
    # A stopped or empty load is flagged on stderr (validate reports the issues).
    catalog = load_skills(args.paths, limits=SkillLimits(max_seconds=args.max_seconds))
    note = incomplete_note(catalog) or ("" if catalog else "no skills found")
    if args.command != "validate" and note:
        _say(note, sys.stderr)
    return catalog


def _cmd_validate(args: argparse.Namespace) -> int:
    """check skills against the Agent Skills rules"""
    catalog = _load(args)
    for issue in catalog.issues:  # load_skills caps the list and says so
        hint = " (raise with --max-seconds)" if issue.code == "load-deadline" else ""
        _say(f"{issue.severity} {issue.code} {issue.path or '-'} {issue.message}{hint}")
    n = Counter(issue.severity for issue in catalog.issues)
    e, w, i = n["error"], n["warning"], n["info"]
    _say(f"{len(catalog)} skills, {e} errors, {w} warnings, {i} info", sys.stderr)
    return CLI_EXIT_FAILURE if e or (args.strict and w) else CLI_EXIT_OK


def _cmd_list(args: argparse.Namespace) -> int:
    """list the skills: name, description, file count"""
    for skill in _load(args):
        name = capped_line(skill.name)
        text = capped_line(skill.description, SKILL_CATALOG_DESCRIPTION_CHARS)
        _say(f"{name}\t{text}\t{len(skill.resources)} files", multiline=True)
    return CLI_EXIT_OK


def _cmd_show(args: argparse.Namespace) -> int:
    """print a skill body page, or a bundled file page with --file"""
    catalog = _load(args)
    skill = catalog.get(args.name)
    if skill is None:
        near = ", ".join(capped_line(n) for n in catalog.closest(args.name))
        hint = f"; did you mean: {near}" if near else ""
        _err(f"no skill named {capped_line(args.name, 200)!r}{hint}")
        return CLI_EXIT_FAILURE
    if args.file is None:
        page, following, _ = catalog.read_body(skill.name, args.offset)
        listed = ", ".join(skill.resources[:SKILL_LIST_MAX_ITEMS])
        what = f"files ({len(skill.resources)}): {listed}"
    elif files := catalog.files_under(skill.name, args.file):
        page, following, _ = text_page(
            "\n".join(files), args.offset, SKILL_READ_PAGE_CHARS
        )
        what = f"files under {capped_line(args.file, 300)!r} ({len(files)})"
    else:
        page, following, _ = catalog.read_resource(skill.name, args.file, args.offset)
        what = f"file: {capped_line(args.file, 300)}"
    name, text = capped_line(skill.name), capped_line(skill.description, 500)
    _say(f"name: {name}\ndescription: {text}\n{what}\n\n{page}", multiline=True)
    if following is not None:
        _say(f"next offset: {following}")
    return CLI_EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(_PROG, description="Browse Agent Skills folders.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(
        name: str, func: Callable[[argparse.Namespace], int], *lead: str
    ) -> argparse.ArgumentParser:
        command = sub.add_parser(name, help=func.__doc__, description=func.__doc__)
        command.set_defaults(func=func)
        for positional in lead:
            command.add_argument(positional, help="skill name")
        command.add_argument("paths", nargs="+", metavar="PATH")
        command.add_argument(
            "--max-seconds",
            type=positive_seconds,
            default=SKILL_MAX_LOAD_SECONDS,
            help="wall-clock budget of the load (default %(default)s)",
        )
        return command

    add("validate", _cmd_validate).add_argument("--strict", action="store_true")
    add("list", _cmd_list)
    show = add("show", _cmd_show, "name")
    show.add_argument("--file", metavar="RELPATH", help="a bundled file, not the body")
    show.add_argument("--offset", type=int, default=0, help="character offset")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``fsm-llm-skills`` command and return its exit code (0, 1 or 130).

    Interface contract: a usage error or ``--help`` raises ``SystemExit`` (argparse);
    a ``SkillError`` or ``OSError`` is one stderr line and exit 1, never a
    traceback; unencodable text is written as ``\\uXXXX`` escapes (the standard
    streams are reconfigured for the rest of the process, as in ``fsm-llm-okf``)."""
    return run_cli(_build_parser(), argv, prog=_PROG, errors=(SkillError,))


def main_cli() -> None:
    """Console-script entry point for ``fsm-llm-skills``."""
    exit_main(main)


if __name__ == "__main__":  # pragma: no cover - process entry point
    main_cli()
