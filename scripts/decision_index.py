#!/usr/bin/env python
"""Find every `# DECISION` anchor of the repository and say which ones resolve.

Standard library only. It never imports `fsm_llm` and never touches `sys.path`: run it
with a plain `python scripts/decision_index.py --report` (roadmap row 0.5, plan d454fd62,
steps 98 to 100; the generator `--write` / `--check` of the committed index arrives at
step 100).

An anchor is a comment of the form `# DECISION [plan-id/]D-NNN [STALE]` next to the code
a decision guards. `plans/ANCHORS.md` (the append-only manifest, one line per anchored
decision: `<plan-id>/D-NNN | YYYY-MM-DD | rationale`) is what lets the anchor resolve on a
fresh clone, after the plan directory is gone. Each anchor gets one state:

* `resolved`   a manifest line exists for its `<plan-id>/D-NNN`;
* `retired`    it carries `[STALE]` and there is no manifest line: the plan record is gone
               for good and the comment in the code is the record (resolved by tag, D-008);
* `unresolved` a live anchor with no manifest line;
* `badprefix`  its prefix is not a plan id (the commit-tag form `plan-YYYY-MM-DD-xxxxxxxx`,
               a typo) or is missing (a bare `D-NNN`).

Grammar: the one of the planner, `validate-plan.mjs` `buildAnchorRegexes` (the strict form
and `LOOSE_ANCHOR_PREFIX_PATTERN`) and its manifest line regex. A change there changes
this file. The plan id is `plan-YYYY-MM-DDTHHMMSS-xxxxxxxx` or the legacy
`plan_YYYY-MM-DD_xxxxxxxx`; the number has three or more digits; the line forms are
`#` (hash family), `//`, `--`, `/* */` and `<!-- -->` (markdown and html, code spans and
fences masked), chosen by file suffix as the planner does. Differences from the planner,
on purpose: the four root files (`pyproject.toml`, `constraints.txt`, `Makefile`,
`tox.ini`) are scanned as hash-comment files whatever their suffix (the planner skips
`.txt`, `.ini` and extension-less files and so never saw the two anchors of
`constraints.txt`), and there is no string-aware lexer for block comments (a `/*` inside a
string literal opens a phantom comment: it over-reports, never loses an anchor; no scanned
file has one).

Interface (the contract the generator of step 100 builds on):

* `scan(root) -> list[Anchor]`: every anchor under `root` (a project directory), in
  deterministic order (path, then order of the text). Never raises: an unreadable file is
  skipped, bytes that are not UTF-8 are decoded with replacement, a missing root is `[]`.
  Roots: `SCAN_DIRS` and `SCAN_FILES`. Skipped: dot entries, symlinks, `SKIP_DIRS`,
  `*.egg-info`, anything deeper than `MAX_DEPTH` (the project root is depth 0).
* `Anchor` (frozen): `token` (as written, `plan-id/D-NNN` or a bare `D-NNN`), `prefix`
  (the text before the slash, or None), `plan_id` (the prefix when it is a plan id, else
  None), `number` (int), `stale`, `path` (POSIX, relative to the root), `scope` (the
  enclosing `Class.method` or function of a `.py` file; the nearest preceding heading,
  table or target for markdown, toml or ini, and Makefile; else `-`), `text` (the comment
  run that holds the anchor, marker and head removed). `key` is `(plan_id, number)`.
  There is no line number on purpose (D-008): a line is stale at the next commit.
* `load_manifest(path) -> dict[(plan_id, number), ManifestEntry]` in file order; the first
  line of a repeated key wins; a missing or unreadable file is `{}`. A manifest line is
  `^<plan-id>/D-NNN[ \\t]*|` (the planner's resolver); the rest is split on the first `|`
  into the date and the rationale, so a `|` or `<` inside the rationale survives.
* `state_of(anchor, manifest) -> str` and `classify(anchors, manifest) -> dict[str,
  list[Anchor]]` with every state of `STATES` present, in that order.
* `main(argv) -> int`: `--root PATH`, `--manifest PATH`, `--report` (counts, exit 0).
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Directories scanned (a dot directory is skipped only when met while descending).
SCAN_DIRS = ("src", "tests", "scripts", ".github")
#: Root files scanned as hash-comment files whatever their suffix.
SCAN_FILES = ("pyproject.toml", "constraints.txt", "Makefile", "tox.ini")
#: The planner's `ANCHOR_SKIP_DIRS`, plus `*.egg-info` (generated).
SKIP_DIRS = frozenset(
    {
        "node_modules",
        "dist",
        "build",
        "plans",
        "target",
        "__pycache__",
        "vendor",
        "out",
    }
)
#: The planner's `ANCHOR_WALK_MAX_DEPTH`; the project root is depth 0, `src` depth 1.
MAX_DEPTH = 12

RESOLVED = "resolved"
RETIRED = "retired"
UNRESOLVED = "unresolved"
BADPREFIX = "badprefix"
STATES = (RESOLVED, RETIRED, UNRESOLVED, BADPREFIX)

# The planner's `ANY_PLAN_ID_PATTERN` (shared.mjs), spelled with [0-9] so a non-ASCII digit
# is not one (JavaScript's `\d` is ASCII only).
_PLAN_ID_PATTERN = (
    r"plan-[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{6}-[0-9a-f]{8}"
    r"|plan_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9a-f]{8}"
)
_PLAN_ID = re.compile(_PLAN_ID_PATTERN)
# The planner's `buildAnchorRegexes` body with the LOOSE prefix: any run of non-space,
# non-slash characters where a plan id belongs; whether it is one is decided afterwards.
_BODY = (
    r"(?:(?P<prefix>[^\s/]+)/)?D-(?P<number>[0-9]{3,})(?![0-9])"
    r"(?P<stale>\s+\[STALE\])?(?::|\s|$)"
)
_LINE_MARKERS = {
    "#": re.compile(r"(?:^|\s)(?P<marker>#)\s+DECISION\s+" + _BODY),
    "//": re.compile(r"(?:^|\s)(?P<marker>//)\s+DECISION\s+" + _BODY),
    "--": re.compile(r"(?:^|\s)(?P<marker>--)\s+DECISION\s+" + _BODY),
}
_INNER = re.compile(r"DECISION\s+" + _BODY)
_BLOCK_COMMENT = re.compile(r"/\*(.*?)\*/", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)")
_MANIFEST_LINE = re.compile(
    r"^(?P<plan>" + _PLAN_ID_PATTERN + r")/D-(?P<number>[0-9]{3,})(?![0-9])[ \t]*\|"
)
_SEPARATORS = " \t:-\u2013\u2014"

_HASH_EXTS = (
    ".py .rb .sh .bash .zsh .yml .yaml .toml .r .pl .pm .tf".split()
)  # planner: hash style
_SLASH_EXTS = (
    ".js .jsx .ts .tsx .mjs .cjs .go .rs .c .h .cpp .hpp .cc .java .swift .kt "
    ".scala .cs .php".split()
)
_BLOCK_EXTS = (
    ".js .jsx .mjs .cjs .ts .tsx .go .rs .c .h .cpp .hpp .cc .java .kt .swift "
    ".scala .cs .php".split()
)
_HTML_EXTS = ".md .markdown .mdx .html .htm".split()
_MARKDOWN_EXTS = frozenset(_HTML_EXTS)
_TABLE_EXTS = frozenset({".toml", ".ini", ".cfg"})

#: file suffix -> the comment families to read, in a fixed order
_STYLES: dict[str, tuple[str, ...]] = {}
for _marker, _suffixes in (
    ("#", _HASH_EXTS),
    ("//", _SLASH_EXTS),
    ("/*", _BLOCK_EXTS),
    ("--", [".sql"]),
    ("<!--", _HTML_EXTS),
):
    for _suffix in _suffixes:
        _STYLES[_suffix] = (*_STYLES.get(_suffix, ()), _marker)


@dataclass(frozen=True)
class Anchor:
    """One `# DECISION` anchor. No line number: see the module docstring."""

    token: str
    prefix: str | None
    plan_id: str | None
    number: int
    stale: bool
    path: str
    scope: str
    text: str

    @property
    def key(self) -> tuple[str | None, int]:
        return (self.plan_id, self.number)


@dataclass(frozen=True)
class ManifestEntry:
    """One manifest line: `<plan-id>/D-NNN | date | rationale` (rationale verbatim)."""

    plan_id: str
    number: int
    date: str
    rationale: str


# --------------------------------------------------------------------------- reading


def _read(path: Path) -> str | None:
    """The text of a file (UTF-8, bad bytes replaced), or None when it cannot be read."""
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None


def load_manifest(path: Path) -> dict[tuple[str, int], ManifestEntry]:
    """Parse the manifest `plans/ANCHORS.md`; `{}` when the file cannot be read."""
    text = _read(Path(path))
    if text is None:
        return {}
    entries: dict[tuple[str, int], ManifestEntry] = {}
    # split on "\n" only: a rationale may hold U+2028, \x0b, \x0c or \x85, which
    # `str.splitlines` would cut in two
    for raw in text.split("\n"):
        line = raw[:-1] if raw.endswith("\r") else raw
        match = _MANIFEST_LINE.match(line)
        if match is None:
            continue
        date, _, rationale = line[match.end() :].partition("|")
        key = (match.group("plan"), int(match.group("number")))
        entries.setdefault(
            key, ManifestEntry(key[0], key[1], date.strip(), rationale.strip())
        )
    return entries


# --------------------------------------------------------------------------- walking


def _style_of(rel: str) -> tuple[str, ...]:
    if rel in SCAN_FILES:
        return ("#",)
    return _STYLES.get(Path(rel).suffix, ())


def _descend(directory: Path, rel: str, depth: int, found: dict[str, Path]) -> None:
    if depth > MAX_DEPTH:
        return
    try:
        with os.scandir(directory) as it:
            entries = list(it)
    except OSError:
        return
    for entry in entries:
        name = entry.name
        if name.startswith(".") or entry.is_symlink():
            continue
        child = f"{rel}/{name}"
        if entry.is_dir(follow_symlinks=False):
            if name not in SKIP_DIRS and not name.endswith(".egg-info"):
                _descend(Path(entry.path), child, depth + 1, found)
        elif entry.is_file(follow_symlinks=False) and _style_of(child):
            found[child] = Path(entry.path)


def _files(root: Path) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for name in SCAN_FILES:
        path = root / name
        if path.is_file() and not path.is_symlink():
            found[name] = path
    for name in SCAN_DIRS:
        top = root / name
        if top.is_dir() and not top.is_symlink():
            _descend(top, name, 1, found)
    return sorted(found.items())


# --------------------------------------------------------------------------- scopes


def _python_scopes(text: str) -> list[tuple[int, int, str]]:
    """`(first line, last line, dotted name)` of every def and class, or `[]` if no parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    scopes: list[tuple[int, int, str]] = []
    # a work list, not recursion: a definition sits in a statement body, never in an
    # expression, so only statements are descended
    work: list[tuple[ast.AST, tuple[str, ...]]] = [(tree, ())]
    while work:
        node, names = work.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                inner = (*names, child.name)
                first = min([child.lineno, *(d.lineno for d in child.decorator_list)])
                scopes.append(
                    (first, child.end_lineno or child.lineno, ".".join(inner))
                )
                work.append((child, inner))
            elif isinstance(child, ast.stmt | ast.ExceptHandler | ast.match_case):
                work.append((child, names))
    return scopes


def _fenced(lines: list[str]) -> list[bool]:
    """Which lines sit in a closed fenced code block (the planner's pass 1)."""
    inside = [False] * len(lines)
    i = 0
    while i < len(lines):
        opener = _FENCE.match(lines[i])
        if opener:
            mark = opener.group(1)
            for j in range(i + 1, len(lines)):
                closer = _FENCE.match(lines[j])
                if (
                    closer
                    and closer.group(1)[0] == mark[0]
                    and len(closer.group(1)) >= len(mark)
                ):
                    inside[i : j + 1] = [True] * (j - i + 1)
                    i = j
                    break
        i += 1
    return inside


def _mask_markdown_code(text: str) -> str:
    """The text with fenced blocks and inline code spans blanked, same offsets."""
    lines = text.split("\n")
    fenced = _fenced(lines)
    return "\n".join(
        " " * len(line)
        if inside
        else _INLINE_CODE.sub(lambda m: " " * len(m.group(0)), line)
        for line, inside in zip(lines, fenced, strict=True)
    )


_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_TABLE = re.compile(r"^\s*\[+\s*([^\[\]]+?)\s*\]+\s*$")
_TARGET = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_./-]*)\s*:(?![=:])")


def _nearest(
    lines: list[str], index: int, pattern: re.Pattern[str], skip: list[bool]
) -> str:
    """The last match of `pattern` (group 1) strictly before line `index`, else `-`."""
    for j in range(index - 1, -1, -1):
        if skip[j]:
            continue
        match = pattern.match(lines[j])
        if match:
            return match.group(1)
    return "-"


class _Scopes:
    """The enclosing definition of a line of one file."""

    def __init__(self, rel: str, text: str, lines: list[str]) -> None:
        self._lines = lines
        suffix = Path(rel).suffix
        self._python = _python_scopes(text) if suffix == ".py" else []
        self._pattern: re.Pattern[str] | None = None
        self._skip = [False] * len(lines)
        if suffix in _MARKDOWN_EXTS:
            self._pattern, self._skip = _HEADING, _fenced(lines)
        elif suffix in _TABLE_EXTS:
            self._pattern = _TABLE
        elif rel == "Makefile" or rel.endswith("/Makefile"):
            self._pattern = _TARGET

    def of(self, line_number: int) -> str:
        """The scope of the 1-based `line_number`."""
        if self._python:
            inside = [s for s in self._python if s[0] <= line_number <= s[1]]
            return max(inside, key=lambda s: (s[0], -s[1]))[2] if inside else "-"
        if self._pattern is None:
            return "-"
        return _nearest(self._lines, line_number - 1, self._pattern, self._skip)


# --------------------------------------------------------------------------- anchors


def _clean_head(text: str) -> str:
    return text.lstrip(_SEPARATORS).rstrip()


def _line_text(lines: list[str], index: int, marker: str, tail: str) -> str:
    """The comment run of a line anchor: the rest of its line, then the full-line comments
    that follow, up to a blank line, code or the next anchor."""
    out = [_clean_head(tail)] if _clean_head(tail) else []
    pattern = _LINE_MARKERS[marker]
    for line in lines[index + 1 :]:
        stripped = line.lstrip()
        if not stripped.startswith(marker) or pattern.search(line):
            break
        body = stripped[len(marker) :]
        out.append((body[1:] if body.startswith(" ") else body).rstrip())
    return "\n".join(out).strip("\n").rstrip()


def _span_text(rest: str, block: bool) -> str:
    """The text of a block or html comment after the anchor head."""
    first, *others = rest.split("\n")
    out = [_clean_head(first)] if _clean_head(first) else []
    for line in others:
        out.append(re.sub(r"^\s*\*\s?", "", line).rstrip() if block else line.strip())
    return "\n".join(out).strip("\n").rstrip()


def _make(match: re.Match[str], rel: str, scope: str, text: str) -> Anchor | None:
    prefix = match.group("prefix")
    digits = match.group("number")
    try:
        number = int(digits)
    except ValueError:  # more digits than `int` reads: not a decision number
        return None
    token = (f"{prefix}/" if prefix else "") + f"D-{digits}"
    plan_id = prefix if prefix and _PLAN_ID.fullmatch(prefix) else None
    return Anchor(
        token=token,
        prefix=prefix,
        plan_id=plan_id,
        number=number,
        stale=match.group("stale") is not None,
        path=rel,
        scope=scope,
        text=text,
    )


def _scan_file(rel: str, text: str) -> list[Anchor]:
    lines = text.split("\n")
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line) + 1)
    scopes = _Scopes(rel, text, lines)
    hits: list[tuple[int, Anchor]] = []

    def line_of(offset: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    styles = _style_of(rel)
    for marker in (m for m in styles if m in _LINE_MARKERS):
        for i, line in enumerate(lines):
            match = _LINE_MARKERS[marker].search(line)
            if match is None:
                continue
            tail = line[match.end() :]
            anchor = _make(
                match, rel, scopes.of(i + 1), _line_text(lines, i, marker, tail)
            )
            if anchor is not None:
                hits.append((starts[i] + match.start("marker"), anchor))
    for style, comment, masked in (
        ("/*", _BLOCK_COMMENT, text),
        (
            "<!--",
            _HTML_COMMENT,
            _mask_markdown_code(text) if "<!--" in styles else text,
        ),
    ):
        if style not in styles:
            continue
        for span in comment.finditer(masked):
            body = text[span.start(1) : span.end(1)]
            found = list(_INNER.finditer(body))
            for k, match in enumerate(found):
                stop = found[k + 1].start() if k + 1 < len(found) else len(body)
                offset = span.start(1) + match.start()
                rest = body[match.end() : stop]
                anchor = _make(
                    match,
                    rel,
                    scopes.of(line_of(offset)),
                    _span_text(rest, style == "/*"),
                )
                if anchor is not None:
                    hits.append((offset, anchor))
    return [anchor for _, anchor in sorted(hits, key=lambda hit: hit[0])]


def scan(root: Path) -> list[Anchor]:
    """Every anchor under `root`, in deterministic order. See the module docstring."""
    anchors: list[Anchor] = []
    for rel, path in _files(Path(root)):
        text = _read(path)
        if text is not None:
            anchors.extend(_scan_file(rel, text))
    return anchors


# --------------------------------------------------------------------------- states


def state_of(anchor: Anchor, manifest: dict[tuple[str, int], ManifestEntry]) -> str:
    """The state of one anchor. A bad or missing prefix beats everything else."""
    if anchor.plan_id is None:
        return BADPREFIX
    if (anchor.plan_id, anchor.number) in manifest:
        return RESOLVED
    return RETIRED if anchor.stale else UNRESOLVED


def classify(
    anchors: Sequence[Anchor], manifest: dict[tuple[str, int], ManifestEntry]
) -> dict[str, list[Anchor]]:
    """The anchors by state, every state present, each list in scan order."""
    out: dict[str, list[Anchor]] = {state: [] for state in STATES}
    for anchor in anchors:
        out[state_of(anchor, manifest)].append(anchor)
    return out


# --------------------------------------------------------------------------- command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decision_index.py",
        description="Scan the `# DECISION` anchors and sort them against plans/ANCHORS.md.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="project directory")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="default: <root>/plans/ANCHORS.md"
    )
    parser.add_argument(
        "--report", action="store_true", help="print the counts, exit 0"
    )
    args = parser.parse_args(argv)
    if not args.report:
        parser.error("nothing to do: pass --report")
    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 1
    manifest = load_manifest(args.manifest or args.root / "plans" / "ANCHORS.md")
    anchors = scan(args.root)
    states = classify(anchors, manifest)
    print(f"anchors: {len(anchors)}")
    for state in STATES:
        print(f"{state}: {len(states[state])}")
    print(f"distinct decisions: {len({a.key for a in anchors if a.plan_id})}")
    print(f"manifest entries: {len(manifest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
