#!/usr/bin/env python3
"""Audit site-packages for suspicious .pth files.

Python .pth files in site-packages execute on every interpreter startup.
This script detects known malicious files and any .pth files containing
executable code (imports, exec, subprocess, etc.) rather than simple
path entries.

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

# Legitimate .pth files contain only directory paths (one per line).
# Suspicious ones contain Python statements.
SUSPICIOUS_PATTERN = re.compile(
    r"^\s*(import|exec|eval|__import__|os\.|subprocess|urllib|requests|http)",
    re.MULTILINE,
)


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
                content = pth.read_text()
            except OSError:
                continue
            if SUSPICIOUS_PATTERN.search(content):
                issues.append(f"WARNING: Executable code in {pth}")

    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return 1

    print("OK: No suspicious .pth files found.")
    return 0


if __name__ == "__main__":
    sys.exit(audit())
