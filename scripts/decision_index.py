#!/usr/bin/env python
"""Find every `# DECISION` anchor of the repository and say which ones resolve.

Standard library only. It never imports `fsm_llm` and never touches `sys.path`: run it
with a plain `python scripts/decision_index.py --report` (roadmap row 0.5, plan d454fd62,
steps 98 to 100). Step 100 adds the generator of the committed index `docs/decisions/
index.md` (`--write`), the gate (`--check`) and the extended `--report`.

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

Scope, the one deliberate gap: the planner walks the WHOLE project (`walkSourceFiles`:
`docs/`, `examples/`, the root `.md` files, dot-named files such as
`.pre-commit-config.yaml`), this scanner reads only `SCAN_DIRS` and `SCAN_FILES`, skips
dot entries and knows only the file suffixes of `_STYLES`. The reason is the committed
index and the gate: `--check` must give the same answer on every checkout and in CI with
no planner, the pages of `docs/` quote anchors as examples, and `examples/` are frozen
baselines, so widening the roots would move the generated index for text that is not a
decision. The price is that a real anchor written outside the roots would be checked by
the planner and never by CI. `outside_scope(root, tracked)` closes that: it reads every
TRACKED file `scan` does not read with this same grammar (a file with no known suffix is
read in every comment family), and `tests/test_fsm_llm_regression/test_decision_scan_scope.py`
fails with the list, so such an anchor is never silently ignored. Citations of a decision
in prose or in a docstring (no comment marker) are outside both grammars and stay prose.

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
* `outside_scope(root, tracked) -> list[Anchor]`: the anchors of the files in `tracked` (POSIX
  paths relative to `root`, for instance `git ls-files`) that `scan` does not read, same
  order and shape as `scan`; the guard of the scope gap above. Never raises: a missing,
  unreadable, symlinked or binary (a NUL byte) file is skipped.
* `load_manifest(path) -> dict[(plan_id, number), ManifestEntry]` in file order; the first
  line of a repeated key wins; a missing or unreadable file is `{}`. A manifest line is
  `^<plan-id>/D-NNN[ \\t]*|` (the planner's resolver); the rest is split on the first `|`
  into the date and the rationale, so a `|` or `<` inside the rationale survives.
* `state_of(anchor, manifest) -> str` and `classify(anchors, manifest) -> dict[str,
  list[Anchor]]` with every state of `STATES` present, in that order.
* `render(anchors, manifest) -> str`: the decision index (markdown) from the anchors and
  the manifest ONLY. Pure: it reads no file, no environment variable, no clock and no
  absolute path, so the text is the same on every machine and every checkout. Plans in
  date order, decisions by number, locations as `path` plus the enclosing `Class.method`
  (never a line number), the rule summary cut at `SUMMARY_WIDTH` characters from the
  anchor's comment, a "Retired" section, a section for anchors that do not resolve (empty
  on a healthy tree) and the manifest entries with no anchor in the tree. ASCII only (any
  other character becomes a visible `\\uXXXX` escape), no trailing space, one trailing
  newline.
* `problems(anchors, manifest) -> list[str]`: why the gate is red (an `unresolved` or a
  `badprefix` anchor), each with its remedy; `[]` when green.
* `check_index(path, anchors, manifest) -> list[str]`: `problems` plus the freshness of
  the committed file against `render` (CRLF read as LF).
* `comment_share(root) -> CommentShare`: comment tokens over source rows of the `.py`
  files under `src`, by `tokenize`: the definition of the baseline 12.87% of the plan
  (10,169 of 78,983 at commit 4b24387d). Reported, never gated.
* `local_resolved(root, anchors)`: the unresolved anchors that a local
  `plans/<plan-id>/decisions.md` names, only while the environment variable `CI` is
  unset (the `resolved-local` count of `--report`). It is a REPORT-ONLY tier: it changes
  neither `render` nor `check_index`, so `--check` says the same in CI and on a developer
  machine, and the committed index never depends on a plan directory that only one
  machine has.
* `main(argv) -> int`: `--root PATH`, `--manifest PATH` and one mode: `--report` (counts,
  exit 0), `--write PATH` (write the index) or `--check PATH` (exit 1 when the file
  differs from `render` or any anchor is `unresolved` or `badprefix`, with the cause and
  the fix command on stderr). `make decision-index` and `make decision-check` call them.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import io
import os
import re
import sys
import tokenize
from collections.abc import Iterable, Sequence
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


def _scan_file(
    rel: str, text: str, styles: tuple[str, ...] | None = None
) -> list[Anchor]:
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

    styles = _style_of(rel) if styles is None else styles
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


#: Every comment family, read in a file `outside_scope` meets with no known suffix.
_ALL_STYLES = ("#", "//", "/*", "--", "<!--")


def outside_scope(root: Path, tracked: Iterable[str]) -> list[Anchor]:
    """The anchors of the `tracked` files that `scan` does not read. See the docstring."""
    base = Path(root)
    scanned = {rel for rel, _ in _files(base)}
    anchors: list[Anchor] = []
    for rel in sorted(set(tracked) - scanned):
        path = base / rel
        if not path.is_file() or path.is_symlink():
            continue
        text = _read(path)
        if text is None or "\x00" in text:
            continue
        anchors.extend(_scan_file(rel, text, _style_of(rel) or _ALL_STYLES))
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


# --------------------------------------------------------------------------- text


#: The rule summary of a decision: the anchor's comment, whitespace collapsed, cut here.
SUMMARY_WIDTH = 120

#: Non-ASCII characters that read better as ASCII than as an escape.
_TRANSLITERATE = {
    " ": " ",
    "–": "-",
    "—": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "←": "<-",
    "→": "->",
    "≤": "<=",
    "≥": ">=",
    "×": "x",
}
_CODE_SPAN = re.compile(r"(`[^`]*`)")
_PLAN_DATE = re.compile(
    r"^plan[-_](?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})(?:T(?P<time>[0-9]{6}))?[-_]"
    r"(?P<hex>[0-9a-f]{8})$"
)


def _ascii(text: str) -> str:
    """`text` with every character outside printable ASCII spelled in ASCII: a known
    punctuation mark becomes its ASCII twin, anything else a visible `\\uXXXX` escape.
    Table-driven, never `unicodedata`: the result must not depend on the Python version."""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch in _TRANSLITERATE:
            out.append(_TRANSLITERATE[ch])
        elif 32 <= code < 127:
            out.append(ch)
        elif code < 128:
            out.append(f"\\x{code:02x}")
        elif code <= 0xFFFF:
            out.append(f"\\u{code:04x}")
        else:
            out.append(f"\\U{code:08x}")
    return "".join(out)


def _squash(text: str) -> str:
    """One line: every run of whitespace is one space, no trailing or leading space."""
    return " ".join(text.split())


def _prose(text: str) -> str:
    """A manifest rationale as one markdown line: a `<` outside a code span is escaped
    (a raw one starts an HTML tag in mkdocs), the rest is ASCII."""
    parts = _CODE_SPAN.split(_squash(text))
    return _ascii(
        "".join(p if i % 2 else p.replace("<", "&lt;") for i, p in enumerate(parts))
    )


def _code(text: str) -> str:
    """`text` as one inline code span: markdown characters stay inert."""
    return "`" + _ascii(_squash(text).replace("`", "'")) + "`"


def _summary(text: str) -> str:
    """The rule summary: the anchor's comment, one line, cut at `SUMMARY_WIDTH`."""
    flat = _squash(text)
    if len(flat) > SUMMARY_WIDTH:
        flat = flat[: SUMMARY_WIDTH - 3].rstrip() + "..."
    return flat


# --------------------------------------------------------------------------- the index


def _plan_order(plan_id: str) -> tuple[str, str, str, str]:
    """Plans by date, then time, then hex id (a legacy id has no time); the id last
    keeps the order total for an id the pattern does not read."""
    match = _PLAN_DATE.match(plan_id)
    if match is None:
        return ("", "", "", plan_id)
    return (match["date"], match["time"] or "", match["hex"], plan_id)


def _where(scope: str) -> str:
    return "module level" if scope == "-" else _code(scope)


def _locations(anchors: Sequence[Anchor]) -> list[str]:
    """One place per (path, scope) of a decision, sorted; no line number."""
    by_place: dict[tuple[str, str], list[Anchor]] = {}
    for anchor in anchors:
        by_place.setdefault((anchor.path, anchor.scope), []).append(anchor)
    out = []
    for (path, scope), group in sorted(by_place.items()):
        bullet = f"{_code(path)}, {_where(scope)}"
        if len(group) > 1:
            bullet += f" ({len(group)} anchors)"
        if all(a.stale for a in group):
            bullet += " [STALE]"
        out.append(bullet)
    return out


def _rule(anchors: Sequence[Anchor]) -> str | None:
    """The summary of the first anchor, in (path, scope) order, that has comment text."""
    for anchor in sorted(anchors, key=lambda a: (a.path, a.scope)):
        if _squash(anchor.text):
            return _summary(anchor.text)
    return None


def _decision_block(
    heading: str, facts: list[str], anchors: Sequence[Anchor]
) -> list[str]:
    block = ["", f"#### {heading}"]
    block += facts
    rule = _rule(anchors)
    if rule is not None:
        block.append(f"- Rule: {_code(rule)}")
    places = _locations(anchors)
    if len(places) == 1:
        block.append(f"- Where: {places[0]}")
    else:
        block.append("- Where:")
        block += [f"  - {place}" for place in places]
    return block


def _by_plan(anchors: Sequence[Anchor]) -> list[tuple[str, dict[int, list[Anchor]]]]:
    plans: dict[str, dict[int, list[Anchor]]] = {}
    for anchor in anchors:
        if anchor.plan_id is not None:
            plans.setdefault(anchor.plan_id, {}).setdefault(anchor.number, []).append(
                anchor
            )
    return [(p, plans[p]) for p in sorted(plans, key=_plan_order)]


def _key(plan_id: str, number: int) -> str:
    return f"{plan_id}/D-{number:03d}"


def render(
    anchors: Sequence[Anchor], manifest: dict[tuple[str, int], ManifestEntry]
) -> str:
    """The decision index as markdown. Pure and deterministic; see the module docstring."""
    states = classify(anchors, manifest)
    seen = {a.key for a in anchors if a.plan_id is not None}
    unanchored = sorted(
        (k for k in manifest if k not in seen), key=lambda k: (_plan_order(k[0]), k[1])
    )
    out = [
        "# Decision index",
        "",
        "Generated by `scripts/decision_index.py`: do not edit by hand. Regenerate it with",
        "`make decision-index` after any change to a `# DECISION` anchor or its comment, or to",
        "`plans/ANCHORS.md`; `make decision-check` (and a test in the default suite) fails",
        "when this file is out of date or an anchor does not resolve.",
        "",
        "A decision anchor is a comment in the code that states a rule and what not to do",
        "(`decision-anchoring` in the planner skill). Each one names a decision as",
        "`<plan-id>/D-NNN`; the one-line record of a decision (its date and rationale) is a",
        "line of the append-only manifest `plans/ANCHORS.md`. Places are given as a path and",
        "the enclosing definition, never as a line number, which is stale at the next commit.",
        "",
        "## Summary",
        "",
        f"- Anchors: {len(anchors)} in {len({a.path for a in anchors})} files, "
        f"{len(seen)} distinct decisions.",
        f"- Resolved (the manifest has the decision): {len(states[RESOLVED])}.",
        f"- Retired (marked `[STALE]`, no manifest line): {len(states[RETIRED])}.",
        f"- Unresolved: {len(states[UNRESOLVED])}. Bad prefix: {len(states[BADPREFIX])}.",
        f"- Manifest entries: {len(manifest)}, of which {len(unanchored)} have no anchor "
        "in the tree.",
        "",
        "## Decisions",
    ]
    for plan_id, numbers in _by_plan(states[RESOLVED]):
        out += ["", f"### {plan_id}"]
        for number in sorted(numbers):
            entry = manifest[(plan_id, number)]
            facts = [f"- Rationale ({_ascii(entry.date)}): {_prose(entry.rationale)}"]
            out += _decision_block(_key(plan_id, number), facts, numbers[number])
    out += [
        "",
        "## Retired",
        "",
        "These anchors carry `[STALE]` and the manifest has no line for their decision: the",
        "plan directory that held the record is gone. They resolve by tag: the comment in",
        "the code is the only record left, and the rule it states may still bind.",
    ]
    if not states[RETIRED]:
        out += ["", "None."]
    for plan_id, numbers in _by_plan(states[RETIRED]):
        out += ["", f"### {plan_id}"]
        for number in sorted(numbers):
            out += _decision_block(_key(plan_id, number), [], numbers[number])
    broken = [*states[UNRESOLVED], *states[BADPREFIX]]
    if broken:
        out += [
            "",
            "## Anchors that do not resolve",
            "",
            "Each of these fails `make decision-check`: add the decision's line to",
            "`plans/ANCHORS.md` (unresolved) or write the full plan id (bad prefix).",
            "",
        ]
        for anchor in sorted(broken, key=lambda a: (a.token, a.path, a.scope)):
            state = BADPREFIX if anchor.plan_id is None else UNRESOLVED
            out.append(
                f"- {_code(anchor.token)} ({state}): {_code(anchor.path)}, "
                f"{_where(anchor.scope)}"
            )
    out += [
        "",
        "## Manifest entries with no code anchor",
        "",
        "The manifest has a line for each of these decisions but no `# DECISION` anchor in",
        "the tree names it (an anchor is removed when the code it guarded is reverted).",
        "",
    ]
    if not unanchored:
        out.append("None.")
    for plan_id, number in unanchored:
        entry = manifest[(plan_id, number)]
        out.append(
            f"- {_code(_key(plan_id, number))}, {_ascii(entry.date)}: "
            f"{_prose(entry.rationale)}"
        )
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- the gate


def problems(
    anchors: Sequence[Anchor], manifest: dict[tuple[str, int], ManifestEntry]
) -> list[str]:
    """Why the anchor gate is red, one message per cause with its remedy; `[]` if green.
    Reads the manifest only, never the environment: `CI` changes nothing here."""
    states = classify(anchors, manifest)
    out = []
    if states[UNRESOLVED]:
        keys = sorted({_key(a.plan_id or "", a.number) for a in states[UNRESOLVED]})
        out.append(
            f"{len(states[UNRESOLVED])} live anchor(s) have no line in plans/ANCHORS.md: "
            + ", ".join(keys[:10])
            + (f" and {len(keys) - 10} more" if len(keys) > 10 else "")
            + ". Fix: the plan's archivist appends one `<plan-id>/D-NNN | date | rationale` "
            "line per decision (append only, from the plan's decisions.md); if the "
            "anchor is a leftover, mark it [STALE]."
        )
    if states[BADPREFIX]:
        tokens = sorted({a.token for a in states[BADPREFIX]})
        out.append(
            f"{len(states[BADPREFIX])} anchor(s) name no plan id: "
            + ", ".join(tokens[:10])
            + (f" and {len(tokens) - 10} more" if len(tokens) > 10 else "")
            + ". Fix: write the full plan directory name before /D-NNN."
        )
    return out


def check_index(
    path: Path,
    anchors: Sequence[Anchor],
    manifest: dict[tuple[str, int], ManifestEntry],
) -> list[str]:
    """`problems` plus the freshness of the committed index at `path`; `[]` if green."""
    out = problems(anchors, manifest)
    wanted = render(anchors, manifest)
    fix = f"run `make decision-index` (or `python scripts/decision_index.py --write {path}`)"
    text = _read(Path(path))
    if text is None:
        out.append(f"{path} is missing or unreadable: {fix}")
    elif text.replace("\r\n", "\n") != wanted:
        diff = list(
            difflib.unified_diff(
                text.replace("\r\n", "\n").split("\n"),
                wanted.split("\n"),
                "committed",
                "generated",
                n=0,
                lineterm="",
            )
        )
        hint = "\n".join(f"    {d}"[:160] for d in diff[2:14])
        out.append(
            f"{path} is out of date ({max(len(diff) - 2, 0)} diff row(s)); {fix}, then "
            f"commit it. First differences (- committed, + generated):\n{hint}"
        )
    return out


# --------------------------------------------------------------------------- the report


@dataclass(frozen=True)
class CommentShare:
    """Comment tokens against source rows (`str.splitlines`) of the `.py` files of `src`."""

    comments: int
    rows: int
    files: int

    @property
    def percent(self) -> float | None:
        return 100.0 * self.comments / self.rows if self.rows else None


def comment_share(root: Path) -> CommentShare:
    """The comment share of `src`, by `tokenize`: one COMMENT token is one comment (a
    full-line comment or a trailing one), the denominator is every physical row. A file
    that does not tokenize is left out of both."""
    comments = rows = files = 0
    for rel, path in _files(Path(root)):
        if not rel.startswith("src/") or not rel.endswith(".py"):
            continue
        text = _read(path)
        if text is None:
            continue
        try:
            found = sum(
                tok.type == tokenize.COMMENT
                for tok in tokenize.generate_tokens(io.StringIO(text).readline)
            )
        except (tokenize.TokenError, SyntaxError, ValueError):
            continue
        comments += found
        rows += len(text.splitlines())
        files += 1
    return CommentShare(comments, rows, files)


_LOCAL_HEADING = re.compile(r"^##[ \t]+D-(?P<number>[0-9]{3,})(?![0-9])", re.MULTILINE)


def ci_is_set() -> bool:
    """True when the environment variable `CI` is set to anything but an empty value."""
    return bool(os.environ.get("CI"))


def local_resolved(root: Path, anchors: Sequence[Anchor]) -> list[Anchor]:
    """Live `unresolved` anchors that a local `plans/<plan-id>/decisions.md` names with a
    `## D-NNN` heading: the `resolved-local` tier. Report-only; empty when `CI` is set."""
    if ci_is_set():
        return []
    known: dict[str, set[int]] = {}
    out = []
    for anchor in anchors:
        if anchor.plan_id is None:
            continue
        if anchor.plan_id not in known:
            text = _read(Path(root) / "plans" / anchor.plan_id / "decisions.md")
            known[anchor.plan_id] = {
                int(m["number"]) for m in _LOCAL_HEADING.finditer(text or "")
            }
        if anchor.number in known[anchor.plan_id]:
            out.append(anchor)
    return out


def _percent(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "n/a"


def report_lines(
    root: Path,
    anchors: Sequence[Anchor],
    manifest: dict[tuple[str, int], ManifestEntry],
) -> list[str]:
    """The `--report` text. The word for the share is `comment share`, never gated."""
    states = classify(anchors, manifest)
    seen = {a.key for a in anchors if a.plan_id is not None}
    lines = [f"anchors: {len(anchors)}"]
    lines += [f"{state}: {len(states[state])}" for state in STATES]
    lines += [
        f"distinct decisions: {len(seen)}",
        f"manifest entries: {len(manifest)}",
        f"manifest entries without an anchor: {len([k for k in manifest if k not in seen])}",
        f"resolved + retired: {len(states[RESOLVED]) + len(states[RETIRED])} of "
        f"{len(anchors)} ({_percent(len(states[RESOLVED]) + len(states[RETIRED]), len(anchors))})",
    ]
    if ci_is_set():
        lines.append("resolved-local: n/a (CI is set)")
    else:
        local = local_resolved(root, states[UNRESOLVED])
        lines.append(
            f"resolved-local: {len(local)} of {len(states[UNRESOLVED])} unresolved"
        )
    share = comment_share(root)
    lines.append(
        f"comment share: {_percent(share.comments, share.rows)} "
        f"({share.comments} comment tokens of {share.rows} source rows, "
        f"{share.files} files of src; reported, not gated)"
    )
    return lines


# --------------------------------------------------------------------------- command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decision_index.py",
        description="Scan the `# DECISION` anchors and sort them against plans/ANCHORS.md; "
        "write or check the generated decision index.",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="project directory")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="default: <root>/plans/ANCHORS.md"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report", action="store_true", help="print the counts, exit 0")
    mode.add_argument(
        "--write", type=Path, metavar="PATH", help="write the decision index to PATH"
    )
    mode.add_argument(
        "--check",
        type=Path,
        metavar="PATH",
        help="exit 1 when PATH differs from the generated index or an anchor does "
        "not resolve",
    )
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 1
    manifest = load_manifest(args.manifest or args.root / "plans" / "ANCHORS.md")
    anchors = scan(args.root)
    if args.report:
        print("\n".join(report_lines(args.root, anchors, manifest)))
        return 0
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        # bytes, not text: no newline translation, and a non-ASCII character is an error
        args.write.write_bytes(render(anchors, manifest).encode("ascii"))
        return 0
    found = check_index(args.check, anchors, manifest)
    if found:
        for message in found:
            print(f"error: {message}", file=sys.stderr)
        return 1
    print(
        f"ok: {args.check} is current ({len(anchors)} anchors, all resolved or retired)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
