"""
Command-line tool for Open Knowledge Format bundles (``fsm-llm-okf``).

Usage::

    fsm-llm-okf validate BUNDLE [--strict] [--max-seconds S]
    fsm-llm-okf list BUNDLE [--type T] [--include-deprecated] [--max-seconds S]
    fsm-llm-okf show BUNDLE ID [--offset N] [--max-seconds S]
    fsm-llm-okf search BUNDLE QUERY [--limit N] [--max-seconds S]
    fsm-llm-okf export MEMORY_JSON OUT_DIR [--overwrite]

``--max-seconds`` is the load's wall-clock budget (default 120); a spent budget
is a ``load-deadline`` error issue, so ``validate`` exits 1.

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
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .cli_support import exit_main, positive_seconds, report_error, run_cli
from .constants import (
    CLI_EXIT_FAILURE,
    CLI_EXIT_OK,
    OKF_LIST_MAX_ITEMS,
    OKF_MAX_LOAD_SECONDS,
    OKF_READ_PAGE_CHARS,
)
from .definitions import OKFError
from .okf import (
    OKFBundle,
    OKFLimits,
    body_page,
    capped_line,
    check_bundle,
    close_ids,
    incomplete_note,
    load_bundle,
    write_bundle,
)
from .okf_bridge import bundle_from_memory_entries

__all__ = ["main", "main_cli"]

_PROG = "fsm-llm-okf"
_MAX_STORE_BYTES = 64 * 1024 * 1024
_MAX_ID_ARG_CHARS = 500
_HEADER_FIELD_CHARS = 500
_SEVERITIES = ("error", "warning", "info")


def _err(message: str) -> None:
    report_error(_PROG, message)


def _load(args: argparse.Namespace) -> OKFBundle:
    """Load ``args.path`` within the ``--max-seconds`` budget. A stopped load is
    flagged on stderr (``validate`` reports the issue itself)."""
    bundle = load_bundle(args.path, limits=OKFLimits(max_seconds=args.max_seconds))
    if args.command != "validate" and (note := incomplete_note(bundle)):
        print(note, file=sys.stderr)
    return bundle


# --------------------------------------------------------------
# Subcommands (each returns an exit code)
# --------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    bundle = _load(args)
    issues = check_bundle(bundle)
    for issue in issues:  # check_bundle caps the list and says so
        where = issue.id or issue.path or "-"
        hint = " (raise with --max-seconds)" if issue.code == "load-deadline" else ""
        print(f"{issue.severity} {issue.code} {where} {issue.message}{hint}")
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
    bundle = _load(args)
    concepts = bundle.list(type=args.type, include_deprecated=args.include_deprecated)
    for concept in concepts[:OKF_LIST_MAX_ITEMS]:
        print(
            "\t".join(
                (
                    capped_line(concept.id),
                    capped_line(concept.type),
                    capped_line(concept.display_title),
                    " ".join(concept.flags()),
                )
            ).rstrip()
        )
    if len(concepts) > OKF_LIST_MAX_ITEMS:
        print(f"... and {len(concepts) - OKF_LIST_MAX_ITEMS} more (narrow with --type)")
    return CLI_EXIT_OK


def _cmd_show(args: argparse.Namespace) -> int:
    bundle = _load(args)
    wanted = args.id[:_MAX_ID_ARG_CHARS]
    concept = bundle.get(wanted)
    if concept is None:
        _err(f"no concept with id {capped_line(wanted)!r}")
        if close := close_ids(bundle, wanted, include_deprecated=True):
            print(
                "did you mean: " + ", ".join(capped_line(i) for i in close),
                file=sys.stderr,
            )
        return CLI_EXIT_FAILURE
    print(f"id: {capped_line(concept.id, _HEADER_FIELD_CHARS)}")
    print(f"type: {capped_line(concept.type, _HEADER_FIELD_CHARS)}")
    print(f"title: {capped_line(concept.display_title, _HEADER_FIELD_CHARS)}")
    print(f"trust (claimed by bundle): {concept.trust.value}")
    print(f"flags: {' '.join(concept.flags()) or '-'}")
    print()
    page, following, _ = body_page(concept, args.offset, OKF_READ_PAGE_CHARS)
    print(page)
    if following is not None:
        print(f"next offset: {following}")
    return CLI_EXIT_OK


def _cmd_search(args: argparse.Namespace) -> int:
    bundle = _load(args)
    limit = min(max(args.limit, 1), OKF_LIST_MAX_ITEMS)
    for concept, score in bundle.search(args.query, limit=limit):
        print(
            "\t".join(
                (
                    f"{score:.2f}",
                    capped_line(concept.id),
                    capped_line(concept.type),
                    capped_line(concept.display_title),
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
        f"wrote {len(written)} files under "
        f"{capped_line(str(Path(args.out_dir).resolve()), 1000)}"
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

    loads = argparse.ArgumentParser(add_help=False)  # shared by the loading commands
    loads.add_argument("path", help="bundle root directory")
    loads.add_argument(
        "--max-seconds",
        type=positive_seconds,
        default=OKF_MAX_LOAD_SECONDS,
        metavar="SECONDS",
        help="wall-clock budget of the load (default %(default)s)",
    )

    def add(
        name: str,
        func: Callable[[argparse.Namespace], int],
        help_text: str,
        parents: Sequence[argparse.ArgumentParser] = (loads,),
    ) -> argparse.ArgumentParser:
        command = sub.add_parser(
            name, help=help_text, description=help_text, parents=parents
        )
        command.set_defaults(func=func)
        return command

    validate = add("validate", _cmd_validate, "check a bundle (spec section 11)")
    validate.add_argument(
        "--strict", action="store_true", help="exit 1 on warnings as well as errors"
    )

    list_ = add("list", _cmd_list, "list the concepts of a bundle")
    list_.add_argument("--type", help="only concepts of this type")
    list_.add_argument(
        "--include-deprecated",
        action="store_true",
        help="also list deprecated concepts",
    )

    show = add("show", _cmd_show, "print one concept: header, then a page of its body")
    show.add_argument("id", help="concept id (the file path without .md)")
    show.add_argument(
        "--offset", type=int, default=0, help="body character offset (default 0)"
    )

    search = add("search", _cmd_search, "lexical search over titles, tags and bodies")
    search.add_argument("query", help="search words")
    search.add_argument(
        "--limit", type=int, default=5, help=f"hits to show, 1..{OKF_LIST_MAX_ITEMS}"
    )

    export = add(
        "export",
        _cmd_export,
        "write a SemanticMemoryStore JSON file as a bundle",
        parents=(),
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
        - Text the terminal cannot encode (a CJK title on an ASCII or cp1252
          console) is written as ``\\uXXXX`` escapes, never a traceback; the
          standard streams are reconfigured for the rest of the process.
        - A bundle whose load was stopped (a spent time or size limit) prints
          one fixed note on stderr from ``list``, ``show`` and ``search``, and
          still exits 0; ``validate`` reports the error issue instead.
    """
    return run_cli(_build_parser(), argv, prog=_PROG, errors=(OKFError,))


def main_cli() -> None:
    """Console-script entry point for ``fsm-llm-okf``."""
    exit_main(main)


if __name__ == "__main__":  # pragma: no cover - process entry point
    main_cli()
