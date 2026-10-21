"""
Command-line tool for Open Knowledge Format bundles (``fsm-llm-okf``).

Usage::

    fsm-llm-okf validate BUNDLE [--strict]
    fsm-llm-okf list BUNDLE [--type T] [--include-deprecated]
    fsm-llm-okf show BUNDLE ID [--offset N]
    fsm-llm-okf search BUNDLE QUERY [--limit N]
    fsm-llm-okf export MEMORY_JSON OUT_DIR [--overwrite]

The primary report goes to stdout, diagnostics and summaries to stderr. This
tool prints directly and does not depend on library logging (off by default).
Every string taken from a bundle passes ``okf.strip_unsafe_text`` first, so a
bundle cannot drive the terminal with escape sequences.

**Exit codes.** ``0`` success; ``1`` failure (a bundle error under ``validate``,
with ``--strict`` also a warning; an unreadable bundle or memory file; an
unknown concept id; a refused export); ``2`` usage error (argparse);
``130`` interrupted by Ctrl-C.

Imports core only (never a subpackage, never litellm).
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .constants import (
    CLI_EXIT_FAILURE,
    CLI_EXIT_INTERRUPTED,
    CLI_EXIT_OK,
    OKF_LIST_MAX_ITEMS,
    OKF_MAX_ISSUES,
    OKF_READ_PAGE_CHARS,
    OKF_SUMMARY_FIELD_CHARS,
)
from .definitions import OKFError
from .okf import (
    OKFBundle,
    OKFConcept,
    bundle_from_memory_entries,
    check_bundle,
    load_bundle,
    strip_unsafe_text,
    write_bundle,
)

__all__ = ["main", "main_cli"]

_PROG = "fsm-llm-okf"
_MAX_STORE_BYTES = 64 * 1024 * 1024
_MAX_CLOSE_MATCHES = 5
_MAX_ID_ARG_CHARS = 500
_HEADER_FIELD_CHARS = 500
_SEVERITIES = ("error", "warning", "info")


def _cell(text: object, cap: int = OKF_SUMMARY_FIELD_CHARS) -> str:
    """Single-line, control-stripped, capped text for one output field."""
    return strip_unsafe_text(text if isinstance(text, str) else str(text))[:cap]


def _err(message: str) -> None:
    print(f"{_PROG}: {_cell(message, 1000)}", file=sys.stderr)


def _flags(concept: OKFConcept) -> list[str]:
    flags = []
    if concept.is_stale():
        flags.append("stale")
    if concept.is_draft:
        flags.append("draft")
    if concept.is_deprecated:
        flags.append("deprecated")
    return flags


def _more(count: int, hint: str = "") -> None:
    print(f"... and {count} more{hint}")


# --------------------------------------------------------------
# Subcommands (each returns an exit code)
# --------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.path)
    issues = check_bundle(bundle)
    for issue in issues[:OKF_MAX_ISSUES]:
        where = issue.id or issue.path or "-"
        print(f"{issue.severity} {issue.code} {where} {issue.message}")
    if len(issues) > OKF_MAX_ISSUES:
        _more(len(issues) - OKF_MAX_ISSUES)
    count = {s: sum(1 for i in issues if i.severity == s) for s in _SEVERITIES}
    print(
        f"{len(bundle)} concepts, {count['error']} errors, "
        f"{count['warning']} warnings, {count['info']} info",
        file=sys.stderr,
    )
    if count["error"] or (args.strict and count["warning"]):
        return CLI_EXIT_FAILURE
    return CLI_EXIT_OK


def _cmd_list(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.path)
    concepts = bundle.list(type=args.type, include_deprecated=args.include_deprecated)
    for concept in concepts[:OKF_LIST_MAX_ITEMS]:
        print(
            "\t".join(
                (
                    _cell(concept.id),
                    _cell(concept.type),
                    _cell(concept.display_title),
                    " ".join(_flags(concept)),
                )
            ).rstrip()
        )
    if len(concepts) > OKF_LIST_MAX_ITEMS:
        _more(len(concepts) - OKF_LIST_MAX_ITEMS, " (narrow with --type)")
    return CLI_EXIT_OK


def _close_ids(bundle: OKFBundle, wanted: str) -> list[str]:
    ids = [c.id for c in bundle]
    return difflib.get_close_matches(wanted, ids, n=_MAX_CLOSE_MATCHES)


def _cmd_show(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.path)
    wanted = args.id[:_MAX_ID_ARG_CHARS]
    concept = bundle.get(wanted)
    if concept is None:
        _err(f"no concept with id {_cell(wanted)!r}")
        close = _close_ids(bundle, wanted)
        if close:
            print(
                "did you mean: " + ", ".join(_cell(i) for i in close), file=sys.stderr
            )
        return CLI_EXIT_FAILURE
    print(f"id: {_cell(concept.id, _HEADER_FIELD_CHARS)}")
    print(f"type: {_cell(concept.type, _HEADER_FIELD_CHARS)}")
    print(f"title: {_cell(concept.display_title, _HEADER_FIELD_CHARS)}")
    print(f"trust: {concept.trust.value}")
    print(f"flags: {' '.join(_flags(concept)) or '-'}")
    print()
    # Page over the stripped body so the pages add up to exactly what is shown.
    body = strip_unsafe_text(concept.body, multiline=True)
    start = min(max(args.offset, 0), len(body))
    end = start + OKF_READ_PAGE_CHARS
    print(body[start:end])
    if end < len(body):
        print(f"next offset: {end}")
    return CLI_EXIT_OK


def _cmd_search(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.path)
    limit = min(max(args.limit, 1), OKF_LIST_MAX_ITEMS)
    for concept, score in bundle.search(args.query, limit=limit):
        print(
            "\t".join(
                (
                    f"{score:.2f}",
                    _cell(concept.id),
                    _cell(concept.type),
                    _cell(concept.display_title),
                )
            )
        )
    return CLI_EXIT_OK


def _read_memory_entries(path: str) -> list[Any]:
    """Entries of a ``SemanticMemoryStore`` JSON file, or raise ``OKFError``."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(_MAX_STORE_BYTES + 1)
    except OSError as exc:
        raise OKFError(f"cannot read memory file: {type(exc).__name__}") from exc
    if len(raw) > _MAX_STORE_BYTES:
        raise OKFError(f"memory file is larger than {_MAX_STORE_BYTES} bytes")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise OKFError(f"memory file is not valid JSON: {type(exc).__name__}") from exc
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise OKFError(
            "memory file must be an object with an 'entries' list of objects"
        )
    return entries


def _cmd_export(args: argparse.Namespace) -> int:
    bundle = bundle_from_memory_entries(_read_memory_entries(args.memory_json))
    shown = bundle.issues[:OKF_LIST_MAX_ITEMS]
    for issue in shown:
        print(f"{issue.severity} {issue.code} {issue.message}", file=sys.stderr)
    if len(bundle.issues) > len(shown):
        print(f"... and {len(bundle.issues) - len(shown)} more", file=sys.stderr)
    if not len(bundle):
        _err("no exportable entries; nothing written")
        return CLI_EXIT_FAILURE
    written = write_bundle(bundle, args.out_dir, overwrite=args.overwrite)
    print(
        f"wrote {len(written)} files under {_cell(Path(args.out_dir).resolve(), 1000)}"
    )
    return CLI_EXIT_OK


# --------------------------------------------------------------
# Entry points
# --------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG, description="Validate, browse and export OKF knowledge bundles."
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(
        name: str, func: Callable[[argparse.Namespace], int], help_text: str
    ) -> argparse.ArgumentParser:
        command = sub.add_parser(name, help=help_text, description=help_text)
        command.set_defaults(func=func)
        return command

    validate = add("validate", _cmd_validate, "check a bundle (spec section 11)")
    validate.add_argument("path", help="bundle root directory")
    validate.add_argument(
        "--strict", action="store_true", help="exit 1 on warnings as well as errors"
    )

    list_ = add("list", _cmd_list, "list the concepts of a bundle")
    list_.add_argument("path", help="bundle root directory")
    list_.add_argument("--type", help="only concepts of this type")
    list_.add_argument(
        "--include-deprecated",
        action="store_true",
        help="also list deprecated concepts",
    )

    show = add("show", _cmd_show, "print one concept: header, then a page of its body")
    show.add_argument("path", help="bundle root directory")
    show.add_argument("id", help="concept id (the file path without .md)")
    show.add_argument(
        "--offset", type=int, default=0, help="body character offset (default 0)"
    )

    search = add("search", _cmd_search, "lexical search over titles, tags and bodies")
    search.add_argument("path", help="bundle root directory")
    search.add_argument("query", help="search words")
    search.add_argument(
        "--limit", type=int, default=5, help=f"hits to show, 1..{OKF_LIST_MAX_ITEMS}"
    )

    export = add(
        "export", _cmd_export, "write a SemanticMemoryStore JSON file as a bundle"
    )
    export.add_argument("memory_json", help="SemanticMemoryStore JSON file")
    export.add_argument(
        "out_dir", help="bundle directory to write (created if missing)"
    )
    export.add_argument(
        "--overwrite", action="store_true", help="replace files that already exist"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``fsm-llm-okf`` command and return its exit code.

    Interface contract:
        - ``argv``: arguments after the program name (``None`` = ``sys.argv``).
        - Returns 0, 1 or 130; a usage error (or ``--help``) raises
          ``SystemExit`` from argparse, as for every other console script.
        - Never raises for bundle, file or I/O problems: an ``OKFError`` or
          ``OSError`` becomes one stderr line and exit 1.
    """
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _err("interrupted")
        return CLI_EXIT_INTERRUPTED
    except BrokenPipeError:
        raise  # handled by main_cli, which owns stdout
    except (OKFError, OSError) as exc:
        _err(f"error: {exc}")
        return CLI_EXIT_FAILURE


def main_cli() -> None:
    """Console-script entry point for ``fsm-llm-okf``."""
    try:
        code = main()
        sys.stdout.flush()
    except BrokenPipeError:
        # Output piped into `head`: park stdout so the interpreter's exit flush
        # does not print a second error.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = CLI_EXIT_FAILURE
    sys.exit(code)


if __name__ == "__main__":  # pragma: no cover - process entry point
    main_cli()
