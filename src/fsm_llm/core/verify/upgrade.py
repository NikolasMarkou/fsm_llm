"""``upgrade_fsm_file``: stamp the current definition version on an older FSM file in place.

``--upgrade`` is the one place a user's file is rewritten: it validates first and
rewrites only a 4.0 or 4.1 file (any version from 4.0 up to below the
default; see ``upgrade_fsm_file``).

Layer ``core`` (plan d454fd62, step 75): split out of the old ``fsm_llm.validator``.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.core.definitions.schema import (
    SCHEMA_VERSION_DEFAULT,
    SCHEMA_VERSION_OLDEST,
    parse_version,
)
from fsm_llm.logging import logger

# Deepest container nesting a file may have and still be REWRITTEN. The dump is
# `indent=2`, whose cost grows with the square of the depth, and a parser whose
# recursion limit follows the stack (3.14 and later) accepts a 100,000-deep value.
# The deepest definition the loader accepts (a condition at MAX_JSONLOGIC_DEPTH
# with list arguments) is about 108 containers deep; 256 leaves more than 2x room.
_MAX_NESTING_DEPTH = 256


@dataclass(frozen=True)
class UpgradeResult:
    """What `upgrade_fsm_file` did.

    rewritten: the file now declares ``SCHEMA_VERSION_DEFAULT``.
    refusal: why an asked upgrade was NOT done (a sentence for the log), or
        None when it was done or there was nothing to do (a current, newer or
        omitted version). ``refusal is not None`` is the failure `main` exits 1 on.
    """

    rewritten: bool
    refusal: str | None = None


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """`json` object hook: a repeated key would be dropped by a rewrite."""
    seen: set[str] = set()
    repeated: set[str] = set()
    for key, _ in pairs:
        (repeated if key in seen else seen).add(key)
    if repeated:
        raise ValueError(f"duplicate key(s) {sorted(repeated)}")
    return dict(pairs)


def _nests_deeper_than(value: Any, limit: int) -> bool:
    """True when a container of `value` sits deeper than `limit` (the root is 1).

    Iterative (an explicit stack): a recursive walk would hit the very limit it
    guards against. `value` is what `json.load` returned: dicts, lists, scalars.
    """
    stack = [(value, 1)]
    while stack:
        node, level = stack.pop()
        children: Iterable[Any]
        if isinstance(node, dict):
            children = node.values()
        elif isinstance(node, list):
            children = node
        else:
            continue
        if level > limit:
            return True
        stack.extend((child, level + 1) for child in children)
    return False


def upgrade_fsm_file(path: str) -> UpgradeResult:
    """Stamp the current definition version on an older FSM file, in place.

    Contract:
        path: an FSM definition JSON file (a symlink is followed; the link is
            kept and its target rewritten).
        Returns: an `UpgradeResult`. ``rewritten`` is True after a rewrite.
            ``refusal`` names the reason when the file was NOT upgraded for a
            cause other than "already current or newer": an I/O failure,
            invalid JSON, a non-object top level, duplicate keys, a file that
            does not load as an FSM definition, a present non-string or empty
            ``version``, one outside ``[4.0, SCHEMA_VERSION_DEFAULT)``, or a
            file about to be rewritten that nests containers deeper than
            ``_MAX_NESTING_DEPTH`` (256). An omitted ``version`` already means
            the default: nothing to do.
        Never raises, whatever the file holds (a deeply nested one included):
        the reason is logged and the file is left byte-identical.
        Rewriting normalises the file: indent 2, LF line endings, numbers
        re-serialised, key order and non-ASCII text kept; the new version is
        the default, so the file's ``fsm_id`` and library digest change.
    """
    # DECISION plan-2026-11-12T023759-9013b0e3/D-012
    # DECISION plan-2026-11-12T023759-9013b0e3/D-016 (supersedes the rewrite
    # rule of D-012: validate first, [4.0, default) only) and
    # plan-2026-11-12T023759-9013b0e3/D-017 (an omitted `version` is nothing to
    # do; the read and the load refuse on ANY exception)
    # Do NOT stamp the version before the file is known to be an FSM: a
    # package.json-like object, or an FSM whose `version` is its own revision
    # ("1.3.0", "3.0"), must stay byte-identical. Load it through
    # `FSMDefinition` first and rewrite only a declared N.M with
    # 4.0 <= v < default (the range the registry knows; below 4.0 is
    # "unknown", D-002). Refuse duplicate keys (a rewrite would drop one).
    # Do NOT downgrade (a "5.0" file is never touched), do NOT rewrite or
    # reformat a file when nothing changes (no write, mtime kept), do NOT write
    # to stdout (it holds the JSON report only; the upgrade is a logger.info),
    # and do NOT open the target for writing directly: the write is a
    # same-directory temp file + os.replace (the `session.py` pattern) so a
    # failure leaves the original intact. The cost is that a rewritten file
    # is normalised (indent 2; JSON has no comments to lose). Do NOT narrow the
    # two `except Exception` below: a 100,000-deep `[` file raises RecursionError
    # (not a ValueError) and the contract is "never raises".
    target = os.path.realpath(path)
    try:
        with open(target, encoding="utf-8") as f:
            data = json.load(f, object_pairs_hook=_no_duplicate_keys)
        if not isinstance(data, dict):
            raise ValueError("FSM definition must be a JSON object")
    except Exception as e:  # OSError, ValueError, RecursionError, ...
        return _refuse(path, str(e) or type(e).__name__)
    try:
        FSMDefinition(**data)
    except Exception as e:  # pydantic ValidationError, TypeError, RecursionError
        detail = " ".join(str(e).split())[:200] or type(e).__name__
        return _refuse(path, f"it does not load as an FSM definition ({detail})")

    if "version" not in data:
        return UpgradeResult(rewritten=False)  # already the default: nothing to do
    version = data["version"]
    if not isinstance(version, str):
        return _refuse(path, "its 'version' is not a string")
    parsed = parse_version(version)
    oldest = parse_version(SCHEMA_VERSION_OLDEST)
    current = parse_version(SCHEMA_VERSION_DEFAULT)
    if parsed is None or oldest is None or current is None or parsed < oldest:
        return _refuse(
            path,
            f"version {version!r} is not an N.M version from {SCHEMA_VERSION_OLDEST} "
            "up, so it may be the file's own revision",
        )
    if parsed >= current:
        return UpgradeResult(rewritten=False)  # nothing to do, not a failure
    if _nests_deeper_than(data, _MAX_NESTING_DEPTH):
        return _refuse(path, f"it is nested more than {_MAX_NESTING_DEPTH} levels deep")
    data["version"] = SCHEMA_VERSION_DEFAULT
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"

    try:
        mode = stat.S_IMODE(os.stat(target).st_mode)
        fd, tmp_name = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".tmp")
    except OSError as e:
        return _refuse(path, str(e))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, target)
    except (OSError, ValueError) as e:  # ValueError: a lone surrogate in a string
        return _refuse(path, str(e))
    finally:
        # No-op after a successful os.replace, which consumed tmp_name.
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    logger.info(f"Upgraded '{path}' to definition version {SCHEMA_VERSION_DEFAULT}")
    return UpgradeResult(rewritten=True)


def _refuse(path: str, reason: str) -> UpgradeResult:
    logger.warning(f"Not upgrading '{path}': {reason}")
    return UpgradeResult(rewritten=False, refusal=reason)
