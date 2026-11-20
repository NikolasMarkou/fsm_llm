"""
A generic JSON patch applier: the RFC 6902 subset ``add``, ``remove``, ``replace``
and ``test``, with no knowledge of what the document is.

``apply_json_patch`` edits a trusted document from untrusted ops (a model wrote
them), so every bound is applied before the walk it protects, a refusal is one
``JsonPatchError`` with a code, and the caller's dict is never touched. The one
way a caller narrows where ops may land is the ``check_path`` hook; the
synthesis repair path uses it for its state scope. It does not validate the
document: the caller re-runs its own gates.

# DECISION plan-2026-11-20T055240-ace7a8c8/D-043
# This is the engine half of plan 2cefeb61 D-051, moved here unchanged in
# behaviour: nothing touches an op value before `_measure` has bounded its size
# and depth (no `json.dumps`, `repr`, `==` or `copy.deepcopy` on unchecked text),
# `~1` is unescaped before `~0`, and the walk is iterative. Do NOT add `move` or
# `copy`, do NOT validate the document here (the caller's gates are the one
# judge), do NOT import `fsm_llm.synthesis` (core is below it; the scope rule
# stays in `synthesis/patch.py`) and do NOT give the hook a way to rewrite ops.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable
from typing import Any

from .errors import FSMError

#: The four ops of the subset; ``move`` and ``copy`` are refused (``op-not-allowed``).
PATCH_OPS = ("add", "remove", "replace", "test")
MAX_PATCH_DEPTH = 16  # nesting of a value below its op object
MAX_POINTER_TOKENS = 12
MAX_POINTER_TOKEN_CHARS = 100
MAX_INDEX_DIGITS = 9


class JsonPatchError(FSMError):
    """A patch was refused whole (nothing was applied).

    ``code`` names the refusal (``too-many-ops``, ``too-large``, ``too-deep``,
    ``bad-value``, ``bad-document``, ``op-shape``, ``op-not-allowed``,
    ``bad-pointer``, ``bad-index``, ``out-of-scope``, ``no-target``,
    ``test-failed``) and ``op`` is the index of the op at fault (``None`` for a
    refusal of the whole patch). The message holds the code and the index only,
    never any text of the patch.
    """

    def __init__(self, code: str, op: int | None = None):
        super().__init__(
            f"patch refused: {code}" + ("" if op is None else f" (op {op})")
        )
        self.code = code
        self.op = op


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
                raise JsonPatchError("too-deep")
            size += 2 + len(node)
            if size > limit:
                raise JsonPatchError("too-large")
            if kind is dict:
                if not all(type(k) is str for k in node):
                    raise JsonPatchError("bad-value")
                size += sum(len(k) + 3 for k in node)
                node = list(node.values())
            stack.extend((child, depth + 1) for child in node)
        elif kind is str:
            size += len(node) + 2
        elif kind in _SCALAR_COST:
            if (kind is int and node.bit_length() > 64) or (
                kind is float and not math.isfinite(node)
            ):
                raise JsonPatchError("bad-value")
            size += _SCALAR_COST[kind]
        else:
            raise JsonPatchError("bad-value")
        if size > limit:
            raise JsonPatchError("too-large")


def _tokens(path: object, op: int) -> list[str]:
    """RFC 6901: ``~1`` is unescaped to ``/`` before ``~0`` to ``~``."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise JsonPatchError("bad-pointer", op)
    raw = path[1:].split("/")
    if len(raw) > MAX_POINTER_TOKENS or any(
        not t or len(t) > MAX_POINTER_TOKEN_CHARS or _BAD_ESCAPE.search(t) for t in raw
    ):
        raise JsonPatchError("bad-pointer", op)
    return [t.replace("~1", "/").replace("~0", "~") for t in raw]


def _index(token: str, op: int) -> int:
    ok = token.isascii() and token.isdigit() and len(token) <= MAX_INDEX_DIGITS
    if not ok or (token[0] == "0" and token != "0"):
        raise JsonPatchError("bad-index", op)
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
            raise JsonPatchError("no-target", op)
    last: Any = tokens[-1]
    if type(parent) is list:
        last = len(parent) if (name == "add" and last == "-") else _index(last, op)
        exists = last < len(parent)
    elif type(parent) is dict:
        exists = last in parent
    else:
        raise JsonPatchError("no-target", op)
    if name == "add":
        if type(parent) is dict:
            parent[last] = value
        elif last > len(parent):
            raise JsonPatchError("no-target", op)
        else:
            parent.insert(last, value)
    elif not exists:
        raise JsonPatchError("no-target", op)
    elif name == "remove":
        del parent[last]
    elif name == "replace":
        parent[last] = value
    elif not _same(parent[last], value):
        raise JsonPatchError("test-failed", op)


def apply_json_patch(
    document: dict[str, Any],
    ops: list[dict[str, Any]],
    *,
    max_ops: int,
    max_chars: int,
    check_path: Callable[[list[str]], bool] | None = None,
) -> dict[str, Any]:
    """Apply ``ops`` to a deep copy of ``document`` and return it.

    Interface contract (callers: ``synthesis.apply_patch``, tests): ``ops`` is a
    list of at most ``max_ops`` objects, each exactly ``{op, path}`` (``remove``)
    or ``{op, path, value}`` (``add``, ``replace``, ``test``); ``move``, ``copy``
    and extra keys are refused. A path is an RFC 6901 pointer of at most 12
    tokens. ``check_path``, when given, is called once per op, in order, with
    the unescaped tokens of a well-formed path (``["states", "a", "x"]``) and
    returns whether the path may be touched; a false return refuses the whole
    patch with ``out-of-scope`` at that op. It must not raise and must not keep
    the list. ``add`` needs an existing parent, an index up to the length or
    ``-``; the others need an existing target; ``test`` is strict JSON equality.
    The whole patch is at most ``max_chars`` characters (an estimate from one
    iterative walk, taken before any op is read) and its values nest at most
    ``MAX_PATCH_DEPTH`` deep. Returns a new dict; ``document`` is never touched.
    Raises only ``JsonPatchError`` for every defect of the patch, and applies
    nothing on any refusal; the checks run in this order: document type,
    ``ops`` type, ``max_ops``, the size and depth walk, then each op in turn.
    """
    if not isinstance(document, dict):
        raise JsonPatchError("bad-document")
    if type(ops) is not list:
        raise JsonPatchError("op-shape")
    if len(ops) > max_ops:
        raise JsonPatchError("too-many-ops")
    _measure(ops, max_chars)
    plan = []
    for i, op in enumerate(ops):
        if type(op) is not dict:
            raise JsonPatchError("op-shape", i)
        name = op.get("op")
        if not (isinstance(name, str) and name in PATCH_OPS):
            raise JsonPatchError("op-not-allowed", i)
        if set(op) != {"op", "path"} | ({"value"} if name != "remove" else set()):
            raise JsonPatchError("op-shape", i)
        tokens = _tokens(op["path"], i)
        if check_path is not None and not check_path(tokens):
            raise JsonPatchError("out-of-scope", i)
        plan.append((i, name, tokens, op.get("value")))
    try:
        out = copy.deepcopy(document)
    except RecursionError:
        raise JsonPatchError("bad-document") from None
    for i, name, tokens, value in plan:
        _apply(out, i, name, tokens, copy.deepcopy(value))
    return out
