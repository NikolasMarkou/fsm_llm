"""
An in-house RFC 6902 subset: ``add``, ``remove``, ``replace`` and ``test``.

``apply_patch`` edits a trusted definition from model-authored ops, so the ops
are untrusted input: every bound is applied before the walk it protects, a
refusal is one ``PatchError`` with a code, and the caller's dict is never
touched. It does not validate the FSM; the caller re-runs the gates.

# DECISION plan-2026-11-10T194139-2cefeb61/D-051
# Scope is judged on the unescaped pointer TOKENS, never on the path string
# (`/states/x/..`, `~1` in a state id and a state id that prefixes another are
# all decided by token equality), and only inside `/states/<scope state>/` of
# the stripped definition, never `abort`. Nothing touches an op value before
# `_measure` has bounded its size and depth (no `json.dumps`, `repr`, `==` or
# `copy.deepcopy` on unchecked text). Do NOT scope by `startswith`, do NOT
# unescape `~0` before `~1`, do NOT add `move` or `copy`, do NOT validate the
# FSM here (the gates are the one judge).
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Iterable
from typing import Any

from .constants import (
    ABORT_STATE_ID,
    MAX_INDEX_DIGITS,
    MAX_PATCH_DEPTH,
    MAX_POINTER_TOKEN_CHARS,
    MAX_POINTER_TOKENS,
    PATCH_OPS,
)
from .exceptions import PatchError
from .policy import SynthesisPolicy

_BAD_ESCAPE = re.compile(r"~(?![01])")
_SCALAR_COST = {bool: 5, type(None): 5, int: 20, float: 24}


def _measure(ops: list[Any], limit: int) -> None:
    """Refuse text over ``limit`` characters, nesting over ``MAX_PATCH_DEPTH``
    below an op object, and any non-JSON value. Iterative; a container's length
    is charged before its children are pushed."""
    size = 0
    stack: list[tuple[Any, int]] = [(ops, 1)]
    while stack:
        node, depth = stack.pop()
        kind = type(node)
        if kind is list or kind is dict:
            if depth > MAX_PATCH_DEPTH + 2:
                raise PatchError("too-deep")
            size += 2 + len(node)
            if size > limit:
                raise PatchError("too-large")
            if kind is dict:
                if not all(type(k) is str for k in node):
                    raise PatchError("bad-value")
                size += sum(len(k) + 3 for k in node)
                node = list(node.values())
            stack.extend((child, depth + 1) for child in node)
        elif kind is str:
            size += len(node) + 2
        elif kind in _SCALAR_COST:
            if (kind is int and node.bit_length() > 64) or (
                kind is float and not math.isfinite(node)
            ):
                raise PatchError("bad-value")
            size += _SCALAR_COST[kind]
        else:
            raise PatchError("bad-value")
        if size > limit:
            raise PatchError("too-large")


def _tokens(path: object, op: int) -> list[str]:
    """RFC 6901: ``~1`` is unescaped to ``/`` before ``~0`` to ``~``."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise PatchError("bad-pointer", op)
    raw = path[1:].split("/")
    if len(raw) > MAX_POINTER_TOKENS or any(
        not t or len(t) > MAX_POINTER_TOKEN_CHARS or _BAD_ESCAPE.search(t) for t in raw
    ):
        raise PatchError("bad-pointer", op)
    return [t.replace("~1", "/").replace("~0", "~") for t in raw]


def _index(token: str, op: int) -> int:
    ok = token.isascii() and token.isdigit() and len(token) <= MAX_INDEX_DIGITS
    if not ok or (token[0] == "0" and token != "0"):
        raise PatchError("bad-index", op)
    return int(token)


def _same(a: Any, b: Any) -> bool:
    """Strict JSON equality, iterative: identical types (a bool is not an int,
    ``1`` is not ``1.0``), equal keys and lengths."""
    stack = [(a, b)]
    while stack:
        x, y = stack.pop()
        if type(x) is not type(y):
            return False
        if isinstance(x, dict):
            if x.keys() != y.keys():
                return False
            stack.extend((x[k], y[k]) for k in x)
        elif isinstance(x, list):
            if len(x) != len(y):
                return False
            stack.extend(zip(x, y, strict=True))
        elif x != y:
            return False
    return True


def _apply(
    doc: dict[str, Any], op: int, name: str, tokens: list[str], value: Any
) -> None:
    parent: Any = doc
    for token in tokens[:-1]:
        if type(parent) is dict and token in parent:
            parent = parent[token]
        elif type(parent) is list and (n := _index(token, op)) < len(parent):
            parent = parent[n]
        else:
            raise PatchError("no-target", op)
    last: Any = tokens[-1]
    if type(parent) is list:
        last = len(parent) if (name == "add" and last == "-") else _index(last, op)
        exists = last < len(parent)
    elif type(parent) is dict:
        exists = last in parent
    else:
        raise PatchError("no-target", op)
    if name == "add":
        if type(parent) is dict:
            parent[last] = value
        elif last > len(parent):
            raise PatchError("no-target", op)
        else:
            parent.insert(last, value)
    elif not exists:
        raise PatchError("no-target", op)
    elif name == "remove":
        del parent[last]
    elif name == "replace":
        parent[last] = value
    elif not _same(parent[last], value):
        raise PatchError("test-failed", op)


def apply_patch(
    document: dict[str, Any],
    ops: list[dict[str, Any]],
    *,
    scope_states: Iterable[str],
    policy: SynthesisPolicy,
) -> dict[str, Any]:
    """Apply ``ops`` to a deep copy of ``document`` and return it.

    Interface contract (callers: ``repair_procedure``, tests): ``ops`` is a list
    of at most ``policy.max_patch_ops`` objects, each exactly ``{op, path}``
    (``remove``) or ``{op, path, value}`` (``add``, ``replace``, ``test``);
    ``move``, ``copy`` and extra keys are refused. A path is an RFC 6901
    pointer of at most 12 tokens that must resolve under
    ``/states/<s>/...`` for an ``s`` in ``scope_states`` (never ``abort``).
    ``add`` needs an existing parent, an index up to the length or ``-``;
    the others need an existing target; ``test`` is strict JSON equality.
    The whole patch is at most ``policy.max_patch_chars`` characters and its
    values nest at most 16 deep. Returns a new dict; ``document`` is never
    touched. Raises only ``PatchError`` (``code`` in ``constants.PATCH_CODES``),
    for every defect of the patch, and applies nothing on any refusal.
    """
    if not isinstance(document, dict):
        raise PatchError("bad-document")
    if isinstance(scope_states, str):
        raise PatchError("bad-scope")
    scope = frozenset(scope_states) - {ABORT_STATE_ID}
    if type(ops) is not list:
        raise PatchError("op-shape")
    if len(ops) > policy.max_patch_ops:
        raise PatchError("too-many-ops")
    _measure(ops, policy.max_patch_chars)
    plan = []
    for i, op in enumerate(ops):
        if type(op) is not dict:
            raise PatchError("op-shape", i)
        name = op.get("op")
        if not (isinstance(name, str) and name in PATCH_OPS):
            raise PatchError("op-not-allowed", i)
        if set(op) != {"op", "path"} | ({"value"} if name != "remove" else set()):
            raise PatchError("op-shape", i)
        tokens = _tokens(op["path"], i)
        if len(tokens) < 3 or tokens[0] != "states" or tokens[1] not in scope:
            raise PatchError("out-of-scope", i)
        plan.append((i, name, tokens, op.get("value")))
    try:
        out = copy.deepcopy(document)
    except RecursionError:
        raise PatchError("bad-document") from None
    for i, name, tokens, value in plan:
        _apply(out, i, name, tokens, copy.deepcopy(value))
    return out
