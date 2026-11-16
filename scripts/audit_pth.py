#!/usr/bin/env python3
"""Audit site-packages for suspicious .pth and .start files.

Python .pth files in site-packages execute on every interpreter startup, and
Python 3.15 also runs the entry points of `.start` files (PEP 829). This script
detects known malicious files and any .pth file containing executable code
(imports, exec, subprocess, etc.) rather than simple path entries, and any
.start file holding an entry point (none is vetted). A .pth file may carry code
only when it is on the `KNOWN_SAFE` name list, or when its name and every line
are vetted by content (`VETTED_CODE_LINES`).

The content is read and split as `site` reads it, on every supported
interpreter: UTF-8 with an optional BOM, then the locale encoding (`site`'s
fallback for a file that is not UTF-8), and BOTH line splits, the newline split
of Python 3.11 and `str.splitlines()` of 3.12.15 and later, 3.13, 3.14 and 3.15
(a form feed, \\x0b, \\x1c to \\x1e, \\x85, \\u2028 and \\u2029 end a line there). A
line is judged under both, so code that hides behind a break only one of them
honours is flagged.

Residuals, stated so nobody has to rediscover them:

- The vetted `pywin32.pth` line `import pywin32_bootstrap` imports whatever module
  of that name `site` finds first. A top-level `pywin32_bootstrap.py` in a scanned
  site directory shadows pywin32's own (`win32/lib/pywin32_bootstrap.py`) and the
  audit does not look for it. Whoever can write a file into site-packages can
  already run code there, so this is accepted, not guarded.
- Only `site.getsitepackages()` is scanned. The user site directory
  (`site.getusersitepackages()`, which `site` also processes when
  `site.ENABLE_USER_SITE` is true) is not, so a `.pth` or `.start` file there is
  never audited; `make audit` and the CI jobs audit the interpreter's own
  site-packages.

Run: python scripts/audit_pth.py
Or:  make audit
"""

import locale
import re
import site
import sys
from pathlib import Path

KNOWN_MALICIOUS = {"litellm_init.pth", "litellm_startup.pth"}

# Standard tooling .pth files that legitimately contain import statements.
KNOWN_SAFE = {
    "_virtualenv.pth",
    "distutils-precedence.pth",
    "a1_coverage.pth",
    "coverage.pth",
    "setuptools.pth",
}

# A file that is code-bearing by design is accepted by CONTENT, never by name alone:
# the name must be a key of this table and every line of the file must be blank, a
# `#` comment, a path line, or (after `strip()`) exactly one of the vetted lines.
# `pywin32.pth` is written by pywin32, which `mcp` pulls in on win32 only; the
# vetted line is the one the pywin32 wheels ship. A second statement, a `;`
# suffix or another import in that file is flagged, and the same line in any
# other file name is flagged too. If a release changes the line the audit fails
# closed: widen the table only after reading the new file.
VETTED_CODE_LINES = {"pywin32.pth": {"import pywin32_bootstrap"}}

# Legitimate .pth files contain only directory paths (one per line).
# Suspicious ones contain Python statements.
SUSPICIOUS_PATTERN = re.compile(
    r"^\s*(import|exec|eval|__import__|os\.|subprocess|urllib|requests|http)",
    re.MULTILINE,
)

# `pip install -e .` of the src layout (`src/fsm_llm/`, `packages.find`) writes a
# plain `.pth` holding only `<repo>/src`: a path line, not code, so no allowlist
# entry is needed for the editable install and none exists. A
# `__editable___*_finder` hook (the flat layout of before, a leftover of an old
# install) is code and is flagged like any other.


def startup_lines(content: str) -> list[str]:
    """Every line `site` may see in `content`: the lines of BOTH splitting styles.

    Contract: `content` is decoded text (a BOM already removed); returns the lines
    of the newline split followed by the lines of `str.splitlines()`. Python 3.11
    reads a `.pth` as a text file, where only a newline, a `\\r\\n` or a lone `\\r` ends
    a line: that is the newline split (a `\\r` before the newline is stripped) with
    a lone `\\r` also seen by `str.splitlines()`, which is what 3.12.15 and later
    use and which splits at more (see the module docstring). A reader of either
    style sees a subset of these lines, so a line is safe only when every one of
    them is. Raises nothing.
    """
    return content.split("\n") + content.splitlines()


def has_code(content: str) -> bool:
    """True when any line `site` may see (`startup_lines`) looks like a statement."""
    return any(SUSPICIOUS_PATTERN.search(line) for line in startup_lines(content))


def only_vetted_code(name: str, content: str) -> bool:
    """True when `name` is in `VETTED_CODE_LINES` and so is every line of `content`.

    Contract: `content` is the decoded text (a BOM already removed). Every line of
    `startup_lines(content)` is judged, so a line break only one of the two
    splitting styles honours cannot hide code: a vetted line followed by a form
    feed and `; import os` is one line to the newline split and a comment followed
    by a form feed and `import os` is two to `str.splitlines()`, and both are
    refused. A line is accepted when it is blank, a `#` comment, exactly a vetted
    line after `strip()`, or does not match `SUSPICIOUS_PATTERN` (a path entry:
    `site` executes only a line that starts with "import"). Any other line, or a
    name outside the table, returns False.
    """
    vetted = VETTED_CODE_LINES.get(name)
    if vetted is None:
        return False
    for line in startup_lines(content):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped in vetted:
            continue
        if SUSPICIOUS_PATTERN.search(line):
            return False
    return True


def read_startup_file(path: Path) -> str:
    """The text of `path` as `site` decodes it.

    Contract: UTF-8 with an optional BOM; when the bytes are not UTF-8, the
    locale encoding (`site`'s fallback, which Windows tools need for an ANSI
    byte in a path); when that fails too, UTF-8 with replacement characters, so
    the ASCII code lines are still scanned and the audit never crashes on a
    file it cannot decode. Raises `OSError` when the file cannot be read.
    """
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode(locale.getencoding())
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8-sig", errors="replace")


def has_entry_point(content: str) -> bool:
    """True when a `.start` file holds a line that is neither blank nor a comment.

    Contract: judged over `startup_lines(content)`, like a `.pth`. No entry point
    is vetted: Python 3.15 imports and calls every one at startup.
    """
    for line in startup_lines(content):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return True
    return False


def audit() -> int:
    """Scan all site-packages directories for suspicious .pth and .start files.

    Returns 0 if clean, 1 if issues found.
    """
    issues: list[str] = []

    for sp in site.getsitepackages():
        sp_path = Path(sp)
        if not sp_path.exists():
            continue
        for pth in sp_path.glob("*.pth"):
            if pth.name in KNOWN_MALICIOUS:
                issues.append(f"CRITICAL: Known malicious file: {pth}")
                continue
            if pth.name in KNOWN_SAFE:
                continue
            try:
                content = read_startup_file(pth)
            except OSError:
                continue
            if has_code(content) and not only_vetted_code(pth.name, content):
                issues.append(f"WARNING: Executable code in {pth}")
        for start in sp_path.glob("*.start"):
            try:
                content = read_startup_file(start)
            except OSError:
                continue
            if has_entry_point(content):
                issues.append(f"WARNING: Startup entry point in {start}")

    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return 1

    print("OK: No suspicious .pth files found.")
    return 0


if __name__ == "__main__":
    sys.exit(audit())
