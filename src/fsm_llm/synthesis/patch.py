"""
The scope wrapper of the generic JSON patch applier of core.

``apply_patch`` edits a trusted definition from model-authored ops, so the ops
are untrusted input: a refusal is one ``PatchError`` with a code, and the
caller's dict is never touched. The engine (the bounds, the pointer rules, the
four ops) is ``fsm_llm.core.definitions.apply_json_patch``; this module adds
what only repair needs: which states an op may touch. It does not validate the
FSM; the caller re-runs the gates.

# DECISION plan-2026-11-10T194139-2cefeb61/D-051
# Scope is judged on the unescaped pointer TOKENS, never on the path string
# (`/states/x/..`, `~1` in a state id and a state id that prefixes another are
# all decided by token equality), and only inside `/states/<scope state>/` of
# the stripped definition, never `abort`. Do NOT scope by `startswith`, do NOT
# widen the scope in the hook, do NOT validate the FSM here (the gates are the
# one judge). The engine half of this rule (bounds before the walk, `~1` before
# `~0`, no `move` or `copy`) moved to `core/definitions/patch.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fsm_llm.core.definitions import JsonPatchError, apply_json_patch

from .constants import ABORT_STATE_ID
from .exceptions import PatchError
from .policy import SynthesisPolicy


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
    ``scope_states`` is an iterable of names (a bare string or ``bytes``, a
    non-iterable, an unhashable item or a name that is not a ``str`` is
    ``bad-scope``).
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
    try:
        scope = frozenset(scope_states) - {ABORT_STATE_ID}
    except TypeError:  # not an iterable of hashable names
        raise PatchError("bad-scope") from None
    if not all(isinstance(name, str) for name in scope):  # bytes iterate as ints
        raise PatchError("bad-scope")

    def in_scope(tokens: list[str]) -> bool:
        return len(tokens) >= 3 and tokens[0] == "states" and tokens[1] in scope

    try:
        return apply_json_patch(
            document,
            ops,
            max_ops=policy.max_patch_ops,
            max_chars=policy.max_patch_chars,
            check_path=in_scope,
        )
    except JsonPatchError as error:
        raise PatchError(error.code, error.op) from None
