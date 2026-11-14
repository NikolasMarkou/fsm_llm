#!/usr/bin/env python
"""Check the shape of CHANGELOG.md and print one release section (Keep a Changelog).

Standard library only, Python 3.11+, deterministic, no network. It only reads files and never
writes one. Roadmap row 0.5, plan d454fd62, step 111. Two subcommands:

    changelog_check.py check   [--file PATH] [--docs-root PATH] [--next-version V] [--from VERSION]
    changelog_check.py extract VERSION [--file PATH] [--with-heading]

Exit codes: 0 clean, 1 findings (`check`) or a version that is missing or has no body
(`extract`), 2 usage error or an unreadable CHANGELOG file.

Sections. A section starts at a line `## ...` outside a code fence and runs to the next `## `
line (the last one stops before the trailing block of link reference definitions). The only
forms are `## [Unreleased]` and `## [X.Y.Z] - YYYY-MM-DD` (an optional ` [YANKED]` may follow).
CRLF and CR line ends are read as LF.

Scope. History is frozen, so `check` judges only the sections that are being prepared:
`[Unreleased]`, plus the newest dated release when `[Unreleased]` holds nothing yet (right after
a release the fresh `[Unreleased]` is empty and the release just cut is still being polished).
`--from VERSION` widens the scope to every section from the top down to VERSION inclusive
(`--from Unreleased` is the narrowest, `--from 0.12.0` includes that release too).

Findings, printed one per line as `path:line: CODE: message`, sorted by line then code:

    CL001  a `## ` heading that is not one of the two forms above (or has no section at all)
    CL002  `###` headings out of the order Added, Changed, Deprecated, Removed, Fixed, Security
    CL003  the same `###` heading twice in one section
    CL004  the `[Unreleased]:` compare reference is missing, or is not
           `<url>/compare/v<newest dated release>...HEAD` (`main` is accepted for `HEAD`)
    CL005  a bullet with `**Breaking:**`, `**breaking for ...**`, `**BREAKING**` or the prose
           "breaking change(s)" / "breaking for" instead of the exact marker `**BREAKING:**`
    CL006  a `**BREAKING:**` bullet with no link `docs/upgrade-guide/vX.Y.Z/<name>.md#<slug>`
    CL007  that link names a file that does not exist under the docs root
    CL008  that link has no `#slug`, or the file has no heading with that id
    CL009  sections not newest first (`[Unreleased]` first, then versions strictly descending)
    CL010  a `###` heading that is none of the six Keep a Changelog names
    CL011  the link names the directory of another version than the section's own

Rules of the BREAKING check. A bullet is a list item at any indent with its continuation lines
(a nested item is a bullet of its own). Code fences and inline code spans are removed first, so
`**BREAKING:**` inside backticks is not a marker. The link is an inline Markdown link whose
target holds `docs/upgrade-guide/` (any prefix: relative, `./`, or an absolute URL); it may sit
on a continuation line. `vX.Y.Z` must be the version of the section the bullet sits in; for
`[Unreleased]` it is `--next-version` (default: the minor bump of the newest dated release,
`0.12.0` gives `0.13.0`, and `0.1.0` when there is no release). The slug is the id the `toc`
extension of Python-Markdown gives a heading (the rule of the docs link test, replicated here
because this script imports nothing of the repository); `<docs-root>/upgrade-guide/vX.Y.Z/
<name>.md` must exist (docs root default: `docs/` beside the CHANGELOG file).

`extract VERSION` prints the body of one section: the lines after its `## [VERSION]` heading up
to the line before the next `## ` heading, with the leading and trailing blank lines removed
and one final newline; `--with-heading` prints the heading and a blank line first. VERSION is
`0.13.0`, `v0.13.0` or `Unreleased`. A missing version, and a section with no body, exit 1 (an
empty release note is a defect of the release, not an empty file).
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = ROOT / "CHANGELOG.md"
DEFAULT_FILE_LABEL = "CHANGELOG.md"

KAC_HEADINGS = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
BREAKING_MARKER = "**BREAKING:**"
GUIDE_PREFIX = "docs/upgrade-guide/"

_FENCE = re.compile(r"^\s*(```+|~~~+)")
_H2 = re.compile(r"^## ")
_H3 = re.compile(r"^###[ \t]+(.*?)[ \t]*$")
_UNRELEASED = re.compile(r"^## \[Unreleased\][ \t]*$")
_RELEASE = re.compile(
    r"^## \[(\d+)\.(\d+)\.(\d+)\] - (\d{4}-\d{2}-\d{2})(?: \[YANKED\])?[ \t]*$"
)
_REFERENCE = re.compile(r"^ {0,3}\[[^\]]+\]:\s+\S")
_UNRELEASED_REF = re.compile(r"^\[Unreleased\]:[ \t]*(\S*)")
_COMPARE = re.compile(r"/compare/v(\d+\.\d+\.\d+)\.\.\.(HEAD|main)$")
_ITEM = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])[ \t]+")
_ANY_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]")
_CODE_SPAN = re.compile(r"(`+)(.+?)\1", re.S)
_BOLD_BREAKING = re.compile(r"\*\*[ \t]*breaking\b[^*\n]*\*\*", re.I)
_PROSE_BREAKING = re.compile(r"\bbreaking[ \t\n]+(?:changes?|for)\b", re.I)
_LINK_TARGET = re.compile(
    r"\]\([ \t]*(<[^>\n]*>|[^)\s]*)(?:[ \t]+(?:\"[^\"]*\"|'[^']*'))?[ \t]*\)"
)
_GUIDE = re.compile(
    r"^v(\d+\.\d+\.\d+)/([A-Za-z0-9][A-Za-z0-9._-]*)\.md(?:#(.*))?$", re.S
)
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_MASK = "\x00"


@dataclass(frozen=True)
class Finding:
    line: int
    code: str
    message: str

    def format(self, label: str) -> str:
        return f"{label}:{self.line}: {self.code}: {self.message}"


@dataclass
class Section:
    """One `## ` section. `kind` is `unreleased`, `release` or `bad`."""

    line: int  # 1-based line of the heading
    kind: str
    heading: str
    version: tuple[int, int, int] | None
    end: int = 0  # 0-based index one past the last body line
    start: int = 0  # 0-based index of the first body line (the one after the heading)

    @property
    def version_text(self) -> str:
        return ".".join(map(str, self.version)) if self.version else ""


@dataclass
class Parsed:
    lines: list[str]  # the text, line ends normalised
    masked: list[str]  # the same lines with fenced code replaced by a placeholder
    sections: list[Section] = field(default_factory=list)
    ref_start: int = 0  # index of the trailing reference-definition block


def normalise(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def mask_fences(lines: list[str]) -> list[str]:
    """Replace every line of a code fence (markers included) by its indent and a placeholder."""
    out: list[str] = []
    marker = ""
    for raw in lines:
        fence = _FENCE.match(raw)
        indent = raw[: len(raw) - len(raw.lstrip())]
        if marker:
            out.append(indent + _MASK)
            if (
                fence
                and fence.group(1)[0] == marker[0]
                and len(fence.group(1)) >= len(marker)
            ):
                marker = ""
            continue
        if fence:
            marker = fence.group(1)
            out.append(indent + _MASK)
            continue
        out.append(raw)
    return out


def parse(text: str) -> Parsed:
    lines = normalise(text)
    masked = mask_fences(lines)
    ref_start = len(masked)
    while ref_start > 0 and (
        not masked[ref_start - 1].strip() or _REFERENCE.match(masked[ref_start - 1])
    ):
        ref_start -= 1
    # `ref_start` now sits at the first line of the trailing blank-or-reference block; the
    # block's leading blank lines do not matter, only that no section body runs into it.
    parsed = Parsed(lines=lines, masked=masked, ref_start=ref_start)
    for index, line in enumerate(masked):
        if not _H2.match(line):
            continue
        if _UNRELEASED.match(line):
            section = Section(index + 1, "unreleased", line, None)
        else:
            m = _RELEASE.match(line)
            version = None
            if m:
                try:
                    date.fromisoformat(m.group(4))
                    version = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    version = None
            section = Section(index + 1, "release" if version else "bad", line, version)
        section.start = index + 1
        parsed.sections.append(section)
    for position, section in enumerate(parsed.sections):
        if position + 1 < len(parsed.sections):
            section.end = parsed.sections[position + 1].line - 1
        else:
            section.end = max(section.start, ref_start)
    return parsed


def newest_release(parsed: Parsed) -> Section | None:
    return next((s for s in parsed.sections if s.kind == "release"), None)


def default_next_version(parsed: Parsed) -> str:
    newest = newest_release(parsed)
    if newest is None or newest.version is None:
        return "0.1.0"
    major, minor, _ = newest.version
    return f"{major}.{minor + 1}.0"


def _has_content(parsed: Parsed, section: Section) -> bool:
    return any(line.strip() for line in parsed.masked[section.start : section.end])


def scope_of(parsed: Parsed, from_version: str | None) -> list[Section]:
    """The sections the checks apply to (see the module docstring)."""
    sections = parsed.sections
    if not sections:
        return []
    if from_version is not None:
        wanted = _version_key(from_version)
        for index, section in enumerate(sections):
            if _matches(section, wanted):
                return sections[: index + 1]
        raise KeyError(from_version)
    scope = sections[:1]
    if (
        sections[0].kind == "unreleased"
        and len(sections) > 1
        and sections[1].kind == "release"
        and not _has_content(parsed, sections[0])
    ):
        scope = sections[:2]
    unreleased = next((s for s in sections if s.kind == "unreleased"), None)
    if unreleased is not None and unreleased not in scope:
        scope = [*scope, unreleased]  # a misplaced one still gets its reference checked
    return scope


def _version_key(value: str) -> str:
    value = value.strip()
    if value.lower() == "unreleased":
        return "unreleased"
    return value[1:] if value[:1] in "vV" else value


def _matches(section: Section, key: str) -> bool:
    if key == "unreleased":
        return section.kind == "unreleased"
    return section.kind == "release" and section.version_text == key


# --------------------------------------------------------------------------------------
# Heading ids (the `toc` extension of Python-Markdown; a copy of the rule of
# tests/test_fsm_llm/test_docs_links.py, held equal to it by a test)
# --------------------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*$")
_ATTR_ID = re.compile(r"\{[^}]*?#([\w.:-]+)[^}]*\}\s*$")
_HTML_ID = re.compile(r"""<[a-zA-Z][^>]*?\s(?:id|name)\s*=\s*["']([^"']+)["']""")
_BLOCK_ID = re.compile(r"\{:?\s*#([\w.:-]+)[^}]*\}")


def slugify(text: str) -> str:
    """`markdown.extensions.toc.slugify` with separator `-`: ASCII fold, strip, lowercase."""
    value = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def _unique(ident: str, taken: set[str]) -> str:
    while ident in taken or not ident:
        m = re.match(r"^(.*)_([0-9]+)$", ident)
        ident = f"{m.group(1)}_{int(m.group(2)) + 1}" if m else f"{ident}_1"
    taken.add(ident)
    return ident


def _heading_text(raw: str) -> str:
    text = re.sub(r"[ \t]+#+[ \t]*$", "", raw)
    text = re.sub(r"(`+)(.+?)\1", lambda m: "\x00" + m.group(2).strip() + "\x00", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("*", "")
    text = re.sub(r"(?<!\w)(_{1,2})(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", text)
    return text.replace("\x00", "")


def heading_ids(text: str) -> list[str]:
    """The id of every heading of a page outside code fences, in order."""
    explicit: list[str] = []
    rows: list[tuple[str | None, str]] = []
    for line in mask_fences(normalise(text)):
        m = _MD_HEADING.match(line)
        if not m:
            continue
        body = m.group(2)
        attr = _ATTR_ID.search(body)
        if attr:
            explicit.append(attr.group(1))
            body = body[: attr.start()].rstrip()
        rows.append((attr.group(1) if attr else None, body))
    taken = set(explicit)
    return [
        ident if ident else _unique(slugify(_heading_text(body)), taken)
        for ident, body in rows
    ]


def anchors_of(text: str) -> set[str]:
    anchors = set(heading_ids(text))
    anchors.update(m.group(1) for m in _HTML_ID.finditer(text))
    anchors.update(m.group(1) for m in _BLOCK_ID.finditer(text))
    return anchors


# --------------------------------------------------------------------------------------
# Bullets
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bullet:
    line: int  # 1-based line of the first line of the item
    text: str  # the item's own lines (nested items excluded), code spans removed


def _indent(line: str) -> int:
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())


def bullets(masked: list[str], start: int, end: int) -> list[Bullet]:
    """Every list item in `masked[start:end]` with its continuation lines."""
    found: list[Bullet] = []
    index = start
    while index < end:
        m = _ITEM.match(masked[index])
        if not m:
            index += 1
            continue
        marker_indent = _indent(masked[index])
        body = [masked[index]]
        j = index + 1
        while j < end:
            line = masked[j]
            if _ITEM.match(line) or _ANY_HEADING.match(line):
                break
            if not line.strip():
                k = j + 1
                while k < end and not masked[k].strip():
                    k += 1
                if (
                    k < end
                    and _indent(masked[k]) >= marker_indent + 2
                    and not _ITEM.match(masked[k])
                    and not _ANY_HEADING.match(masked[k])
                ):
                    body.extend(masked[j:k])
                    j = k
                    continue
                break
            body.append(line)
            j += 1
        text = _CODE_SPAN.sub(_MASK, "\n".join(body))
        found.append(Bullet(index + 1, text))
        index = j
    return found


# --------------------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------------------


def _check_headings(parsed: Parsed, section: Section) -> Iterable[Finding]:
    seen: dict[str, int] = {}
    highest = -1
    for number in range(section.start, section.end):
        m = _H3.match(parsed.masked[number])
        if not m:
            continue
        name = m.group(1)
        if name not in KAC_HEADINGS:
            yield Finding(
                number + 1,
                "CL010",
                f"unknown section `### {name}`; use one of {', '.join(KAC_HEADINGS)}",
            )
            continue
        rank = KAC_HEADINGS.index(name)
        if name in seen:
            yield Finding(
                number + 1,
                "CL003",
                f"`### {name}` appears twice in this section (first at line {seen[name]})",
            )
            continue
        seen[name] = number + 1
        if rank < highest:
            yield Finding(
                number + 1,
                "CL002",
                f"`### {name}` is out of order; the order is {', '.join(KAC_HEADINGS)}",
            )
        highest = max(highest, rank)


def _guide_findings(
    bullet: Bullet,
    version_dir: str,
    docs_root: Path,
    cache: dict[Path, set[str] | None],
) -> Iterable[Finding]:
    targets = []
    for m in _LINK_TARGET.finditer(bullet.text):
        target = m.group(1).strip("<>")
        at = target.find(GUIDE_PREFIX)
        if at >= 0:
            targets.append(target[at + len(GUIDE_PREFIX) :])
    if not targets:
        yield Finding(
            bullet.line,
            "CL006",
            f"a {BREAKING_MARKER} bullet must link its upgrade guide, "
            f"`docs/upgrade-guide/v{version_dir}/<name>.md#<slug>`",
        )
        return
    for rest in targets:
        shown = GUIDE_PREFIX + rest
        m = _GUIDE.match(rest)
        if not m:
            yield Finding(
                bullet.line,
                "CL006",
                f"the guide link `{shown}` is not of the form "
                f"`docs/upgrade-guide/v{version_dir}/<name>.md#<slug>`",
            )
            continue
        version, name, fragment = m.group(1), m.group(2), m.group(3)
        if version != version_dir:
            yield Finding(
                bullet.line,
                "CL011",
                f"the guide link `{shown}` is in the directory of v{version}, "
                f"this section is v{version_dir}",
            )
            continue
        page = docs_root / "upgrade-guide" / f"v{version}" / f"{name}.md"
        if page not in cache:
            try:
                cache[page] = anchors_of(page.read_text(encoding="utf-8"))
            except (OSError, UnicodeError):
                cache[page] = None
        anchors = cache[page]
        if anchors is None:
            yield Finding(
                bullet.line,
                "CL007",
                f"the guide `{shown.split('#')[0]}` does not exist",
            )
            continue
        slug = unquote(fragment or "")
        if not slug:
            yield Finding(
                bullet.line, "CL008", f"the guide link `{shown}` has no `#slug`"
            )
        elif slug not in anchors:
            yield Finding(
                bullet.line,
                "CL008",
                f"the guide `{shown.split('#')[0]}` has no heading with the id `{slug}`",
            )


def _check_bullets(
    parsed: Parsed,
    section: Section,
    version_dir: str,
    docs_root: Path,
    cache: dict[Path, set[str] | None],
) -> Iterable[Finding]:
    for bullet in bullets(parsed.masked, section.start, section.end):
        text = bullet.text
        has_marker = False
        for m in _BOLD_BREAKING.finditer(text):
            if m.group(0) == BREAKING_MARKER:
                has_marker = True
            else:
                yield Finding(
                    bullet.line,
                    "CL005",
                    f"`{m.group(0).replace(_MASK, '`..`')}` is not the marker; "
                    f"write exactly {BREAKING_MARKER}",
                )
        if not _BOLD_BREAKING.search(text) and _PROSE_BREAKING.search(text):
            yield Finding(
                bullet.line,
                "CL005",
                f"the bullet says it is breaking in prose; start the breaking part "
                f"with {BREAKING_MARKER} and link its upgrade guide",
            )
        if has_marker:
            yield from _guide_findings(bullet, version_dir, docs_root, cache)


def _check_order(parsed: Parsed, in_scope: list[Section]) -> Iterable[Finding]:
    sections = parsed.sections
    if sections and sections[0].kind == "release":
        if not any(s.kind == "unreleased" for s in sections):
            yield Finding(
                sections[0].line,
                "CL009",
                "the first section must be `## [Unreleased]`",
            )
    for index in range(1, len(sections)):
        above, here = sections[index - 1], sections[index]
        if here.kind == "unreleased":
            yield Finding(
                here.line, "CL009", "`## [Unreleased]` must be the first section"
            )
        elif (
            above in in_scope
            and above.version is not None
            and here.version is not None
            and here.version >= above.version
        ):
            yield Finding(
                here.line,
                "CL009",
                f"`[{here.version_text}]` is not older than `[{above.version_text}]` "
                "above it; releases go newest first",
            )


def _check_reference(parsed: Parsed) -> Iterable[Finding]:
    unreleased = next((s for s in parsed.sections if s.kind == "unreleased"), None)
    if unreleased is None:
        return
    newest = newest_release(parsed)
    for number, line in enumerate(parsed.masked, 1):
        m = _UNRELEASED_REF.match(line)
        if not m:
            continue
        compare = _COMPARE.search(m.group(1))
        if not compare:
            yield Finding(
                number,
                "CL004",
                "the `[Unreleased]:` reference must end in `/compare/v<tag>...HEAD`",
            )
        elif newest is not None and compare.group(1) != newest.version_text:
            yield Finding(
                number,
                "CL004",
                f"the `[Unreleased]:` reference compares from v{compare.group(1)}, "
                f"the newest release is v{newest.version_text}",
            )
        return
    yield Finding(
        unreleased.line,
        "CL004",
        "no `[Unreleased]: <url>/compare/v<newest release>...HEAD` reference line",
    )


def check_text(
    text: str,
    *,
    docs_root: Path,
    next_version: str | None = None,
    from_version: str | None = None,
) -> list[Finding]:
    """The findings of a CHANGELOG text, sorted; see the module docstring for the rules.

    Raises `KeyError(from_version)` when `from_version` names no section.
    """
    parsed = parse(text)
    if not parsed.sections:
        return [
            Finding(1, "CL001", "no `## [Unreleased]` or `## [X.Y.Z] - date` section")
        ]
    in_scope = scope_of(parsed, from_version)
    if next_version is None:
        next_version = default_next_version(parsed)
    findings: list[Finding] = list(_check_order(parsed, in_scope))
    cache: dict[Path, set[str] | None] = {}
    for section in in_scope:
        if section.kind == "bad":
            findings.append(
                Finding(
                    section.line,
                    "CL001",
                    f"`{section.heading}` is not `## [Unreleased]` or "
                    "`## [X.Y.Z] - YYYY-MM-DD`",
                )
            )
            continue
        version_dir = (
            next_version if section.kind == "unreleased" else section.version_text
        )
        findings.extend(_check_headings(parsed, section))
        findings.extend(_check_bullets(parsed, section, version_dir, docs_root, cache))
    if any(s.kind == "unreleased" for s in in_scope):
        findings.extend(_check_reference(parsed))
    return sorted(set(findings), key=lambda f: (f.line, f.code, f.message))


def extract_text(text: str, version: str, *, with_heading: bool = False) -> str | None:
    """The body of one section (see the module docstring), or None when it is missing."""
    parsed = parse(text)
    key = _version_key(version)
    for section in parsed.sections:
        if not _matches(section, key):
            continue
        body = parsed.lines[section.start : section.end]
        while body and not body[-1].strip():
            body.pop()
        while body and not body[0].strip():
            body.pop(0)
        if not body:
            return ""
        out = ([parsed.lines[section.line - 1], ""] if with_heading else []) + body
        return "\n".join(out) + "\n"
    return None


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(2)


def _next_version(value: str) -> str:
    if not _VERSION.match(value):
        raise argparse.ArgumentTypeError(f"expected X.Y.Z, got {value!r}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="changelog_check.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    check = sub.add_parser("check", help="check the shape of the CHANGELOG")
    extract = sub.add_parser("extract", help="print the body of one release section")
    for p in (check, extract):
        p.add_argument("--file", type=Path, default=None, help="default: CHANGELOG.md")
    check.add_argument("--docs-root", type=Path, default=None, help="default: docs/")
    check.add_argument("--next-version", type=_next_version, default=None)
    check.add_argument("--from", dest="from_version", default=None, metavar="VERSION")
    extract.add_argument("version", help="0.13.0, v0.13.0 or Unreleased")
    extract.add_argument("--with-heading", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    stdout: Callable[[str], Any] | None = None,
) -> int:
    """Run the CLI; return the exit code. `stdout` receives text with its own newlines."""
    if stdout is None:
        stdout = sys.stdout.write
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    path: Path = args.file or DEFAULT_FILE
    label = str(args.file) if args.file else DEFAULT_FILE_LABEL
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"changelog_check: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    if args.command == "extract":
        body = extract_text(text, args.version, with_heading=args.with_heading)
        if body is None:
            print(
                f"changelog_check: no section [{args.version}] in {label}",
                file=sys.stderr,
            )
            return 1
        if not body:
            print(
                f"changelog_check: section [{args.version}] has no body",
                file=sys.stderr,
            )
            return 1
        stdout(body)
        return 0
    docs_root = args.docs_root or path.resolve().parent / "docs"
    try:
        findings = check_text(
            text,
            docs_root=docs_root,
            next_version=args.next_version,
            from_version=args.from_version,
        )
    except KeyError:
        print(
            f"changelog_check: no section [{args.from_version}] for --from",
            file=sys.stderr,
        )
        return 2
    for finding in findings:
        stdout(finding.format(label) + "\n")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
