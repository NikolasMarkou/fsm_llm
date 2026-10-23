"""
The one YAML frontmatter seam: split, bounded load, no-follow read, surrogate walk.

Shared by every file format that keeps YAML between two ``---`` lines (OKF
documents, Agent Skills ``SKILL.md``). The text is untrusted: the loader refuses
YAML aliases and anchors, keeps timestamps as strings, bounds size and depth
before any parsing work, and never lets a PyYAML constructor error escape as
anything but a note. Stdlib and ``yaml`` only; imports nothing else of the
package, so a format module may import it without a cycle.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import yaml

__all__ = [
    "SURROGATE_RE",
    "Note",
    "load_yaml",
    "lone_surrogate",
    "read_bytes",
    "split_frontmatter",
]

# A problem callback: ``note(severity, code, message)``.
Note = Callable[[Literal["error", "warning", "info"], str, str], None]
SURROGATE_RE = re.compile("[\ud800-\udfff]")
_TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"
_CLOSING_RE = re.compile(r"^---[ \t]*$", re.MULTILINE)


class _FrontmatterLoader(yaml.SafeLoader):
    """``SafeLoader`` without timestamps, aliases and anchors (refused in
    ``compose_node``, before any expansion); duplicate keys are recorded, the
    last wins. SECURITY: the resolver table is a COPY (below)."""

    # DECISION plan-2026-10-21T140931-095ea5f8/D-014
    # The timestamp resolver is left out of a COPY of the table. Do NOT call
    # `.remove()` or `del` on `yaml.SafeLoader.yaml_implicit_resolvers`: that
    # table is shared, so it would change `yaml.safe_load` for the whole
    # process. See decisions.md D-014.
    yaml_implicit_resolvers: ClassVar[dict[Any, Any]] = {
        first: [(tag, rx) for tag, rx in rules if tag != _TIMESTAMP_TAG]
        for first, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self.duplicate_keys: list[str] = []

    def compose_node(self, parent: Any, index: Any) -> Any:
        event = self.peek_event()
        if getattr(event, "anchor", None):  # AliasEvent and anchored nodes
            raise yaml.YAMLError("YAML aliases and anchors are not allowed")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> Any:
        seen: set[str] = set()
        for key_node, _ in node.value:
            if isinstance(key_node, yaml.ScalarNode):
                if key_node.value in seen:
                    self.duplicate_keys.append(key_node.value)
                seen.add(key_node.value)
        return super().construct_mapping(node, deep)


def split_frontmatter(text: str) -> tuple[str | None, str, str | None]:
    """``(yaml text | None, body, problem code | None)``; BOM removed, line ends
    normalised to ``\\n``. Without a complete ``---`` block all text is body.

    Interface contract: never raises; the problem code is
    ``frontmatter-missing`` or ``frontmatter-unterminated`` when the yaml text
    is ``None``, else ``None``."""
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    first, _, rest = text.partition("\n")
    if first.rstrip() != "---":
        return None, text, "frontmatter-missing"
    closing = _CLOSING_RE.search(rest)
    if closing is None:
        return None, text, "frontmatter-unterminated"
    return rest[: closing.start()], rest[closing.end() :].removeprefix("\n"), None


def _flow_depth_exceeded(raw: str, max_depth: int) -> bool:
    """Bound nested ``[``/``{`` first (PyYAML's scanner is quadratic on them)."""
    depth = 0
    for ch in raw:
        if ch in "[{":
            depth += 1
            if depth > max_depth:
                return True
        elif ch in "]}" and depth:
            depth -= 1
    return False


def load_yaml(
    raw: str, note: Note, *, max_bytes: int, max_depth: int
) -> dict[Any, Any]:
    """Load the frontmatter mapping in *raw* under a size and a depth bound.

    Interface contract: never raises. Returns the mapping, or ``{}`` after one
    ``note`` call for ``frontmatter-too-large`` (over *max_bytes* UTF-8 bytes,
    checked before any parsing), ``frontmatter-too-deep`` (nesting over
    *max_depth*, or a structure the surrogate walk refuses),
    ``frontmatter-invalid`` (any parser or constructor error),
    ``frontmatter-empty`` (warning) or ``frontmatter-not-mapping``; a repeated
    key adds a ``key-duplicate`` warning and the last value wins."""
    # SECURITY: size bound BEFORE any YAML work (pure-Python parse: up to ~27 ms/KB).
    if len(raw.encode("utf-8", "replace")) > max_bytes:
        note("error", "frontmatter-too-large", "frontmatter block is too large")
        return {}
    duplicates: list[str] = []
    try:
        loader = _FrontmatterLoader(raw)
        if _flow_depth_exceeded(raw, max_depth):
            raise RecursionError
        data = cast("object", loader.get_single_data())
        duplicates = loader.duplicate_keys
        lone_surrogate(data, max_depth=max_depth, max_nodes=max_bytes)  # depth walk
    except RecursionError:  # first: a RuntimeError, not the catch-all's
        note("error", "frontmatter-too-deep", "frontmatter is nested too deeply")
        return {}
    except Exception as exc:  # PyYAML constructors raise anything (``!!timestamp x``)
        problem = getattr(exc, "problem", None) or exc
        note("error", "frontmatter-invalid", f"{type(exc).__name__}: {problem}"[:160])
        return {}
    for key in duplicates:
        note("warning", "key-duplicate", f"duplicate key {key!r}; the last value wins")
    if data is None:
        note("warning", "frontmatter-empty", "frontmatter block is empty")
        return {}
    if not isinstance(data, dict):
        note("error", "frontmatter-not-mapping", "frontmatter is not a mapping")
        return {}
    return data


def read_bytes(path: Path, limit: int) -> bytes:
    """Read at most ``limit + 1`` bytes; never follows a final symlink.

    Interface contract: raises ``OSError`` for a symlink, a missing or
    unreadable file; a FIFO opens without blocking and reads as empty."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        return handle.read(limit + 1)


def lone_surrogate(value: Any, *, max_depth: int, max_nodes: int) -> bool:
    """True when a string, key or item in *value* is a lone surrogate. Iterative
    and walked to the end: ``RecursionError`` past *max_depth* (a cycle ends
    there) or *max_nodes* nodes, surrogate or not."""
    stack, nodes, found = [(value, 0)], 0, False
    while stack:
        item, depth = stack.pop()
        if isinstance(item, str):
            found = found or bool(SURROGATE_RE.search(item))
        elif isinstance(item, (Mapping, list, tuple, set, frozenset)):
            kids = [*item, *item.values()] if isinstance(item, Mapping) else item
            nodes += len(kids)
            if depth >= max_depth or nodes > max_nodes:
                raise RecursionError("too deep, too large or cyclic")
            stack.extend((kid, depth + 1) for kid in kids)
    return found
