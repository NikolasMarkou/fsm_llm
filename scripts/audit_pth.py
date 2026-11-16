#!/usr/bin/env python3
"""Audit site-packages for suspicious .pth files.

Python .pth files in site-packages execute on every interpreter startup.
This script detects known malicious files and any .pth files containing
executable code (imports, exec, subprocess, etc.) rather than simple
path entries. A file may carry code only when it is on the `KNOWN_SAFE` name
list, or when its name and every line are vetted by content (`VETTED_CODE_LINES`).

Run: python scripts/audit_pth.py
Or:  make audit
"""

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


def only_vetted_code(name: str, content: str) -> bool:
    """True when `name` is in `VETTED_CODE_LINES` and so is every line of `content`.

    Contract: `content` is the decoded text (a BOM already removed). It is split
    on the newline character only, as `site` does after universal-newline reading,
    and never with `str.splitlines()`: that also splits on a form feed, which
    would let a vetted line followed by a form feed and `; import os` pass as two
    harmless lines while `site` executes the whole line. A line is accepted when
    it is blank, a `#` comment, exactly a vetted line after `strip()`, or does not
    match `SUSPICIOUS_PATTERN` (a path entry: `site` executes only a line that
    starts with "import"). Any other line, or a name outside the table, returns
    False.
    """
    vetted = VETTED_CODE_LINES.get(name)
    if vetted is None:
        return False
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped in vetted:
            continue
        if SUSPICIOUS_PATTERN.search(line):
            return False
    return True


def audit() -> int:
    """Scan all site-packages directories for suspicious .pth files.

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
                # `site` reads UTF-8 with an optional BOM (then the locale
                # encoding on 3.13+); Windows tools write both. `utf-8-sig`
                # keeps a BOM from hiding an `import` line from the pattern,
                # and `replace` keeps an ANSI byte in a path from crashing the
                # audit while the code lines (ASCII) are still scanned.
                content = pth.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            if SUSPICIOUS_PATTERN.search(content) and not only_vetted_code(
                pth.name, content
            ):
                issues.append(f"WARNING: Executable code in {pth}")

    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return 1

    print("OK: No suspicious .pth files found.")
    return 0


if __name__ == "__main__":
    sys.exit(audit())
